# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os.path as osp
from functools import partial
from subprocess import DEVNULL, CalledProcessError, run  # nosec

import mmcv
import onnx
import numpy as np
import torch
import torch.nn as nn
from mmcv.onnx import register_extra_symbolics
from onnxoptimizer import optimize
from torch.onnx.symbolic_helper import _onnx_stable_opsets as available_opsets


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))

    del module

    return module_output


def _update_input_img(img_list, img_meta_list, update_ori_shape=False):
    N, C, H, W = img_list[0].shape
    img_meta = img_meta_list[0][0]
    img_shape = (H, W, C)
    pad_shape = img_shape

    if update_ori_shape:
        ori_shape = img_shape
    else:
        ori_shape = img_meta['ori_shape']

    new_img_meta_list = [[{
        'img_shape':
        img_shape,
        'ori_shape':
        ori_shape,
        'pad_shape':
        pad_shape,
        'filename':
        img_meta['filename'],
        'scale_factor':
        (img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]) * 2,
        'flip':
        False,
    } for _ in range(N)]]

    return img_list, new_img_meta_list


def export_to_onnx(model,
                   fake_inputs,
                   export_name,
                   output_logits=False,
                   opset=11,
                   verbose=False):
    register_extra_symbolics(opset)

    imgs = fake_inputs.pop('imgs')
    img_metas = fake_inputs.pop('img_metas')

    img_list = [img[None, :] for img in imgs]
    img_meta_list = [[img_meta] for img_meta in img_metas]
    img_list, img_meta_list = _update_input_img(img_list, img_meta_list)

    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        output_logits=output_logits,
        rescale=True
    )

    if model.test_cfg.mode == 'slide':
        dynamic_axes = {
            'input': {0: 'batch'},
            'output': {1: 'batch'}
        }
    else:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {1: 'batch', 2: 'height', 3: 'width'}
        }

    output_names = ['output']

    output_names.extend(['feature_vector'])

    with torch.no_grad():
        torch.onnx.export(
            model,
            (img_list,),
            f=export_name,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            verbose=verbose,
            opset_version=opset,
            keep_initializers_as_inputs=False,
        )

    model.forward = origin_forward


def check_onnx_model(export_name):
    try:
        onnx.checker.check_model(export_name)
        print('ONNX check passed.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
        print('ONNX check failed.')
        print(ex)


def _get_mo_cmd():
    for mo_cmd in ('mo', 'mo.py'):
        try:
            run([mo_cmd, '-h'], stdout=DEVNULL, stderr=DEVNULL, shell=False, check=True)
            return mo_cmd
        except CalledProcessError:
            pass

    raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly')


def export_to_openvino(cfg, onnx_model_path, output_dir_path, input_shape=None,
                       input_format='rgb', precision='FP32'):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    onnx_model = onnx.load(onnx_model_path)

    output_names = set(out.name for out in onnx_model.graph.output)
    # Clear names of the nodes that produce network's output blobs.
    for node in onnx_model.graph.node:
        if output_names.intersection(node.output):
            node.ClearField('name')
    onnx.save(onnx_model, onnx_model_path)
    output_names = ','.join(output_names)

    mo_cmd = _get_mo_cmd()

    normalize = [v for v in cfg.data.test.pipeline[1].transforms if v['type'] == 'Normalize'][0]

    mean_values = normalize['mean']
    scale_values = normalize['std']
    command_line = [mo_cmd,
                    f'--input_model={onnx_model_path}',
                    f'--mean_values={mean_values}',
                    f'--scale_values={scale_values}',
                    f'--output_dir={output_dir_path}',
                    f'--output={output_names}',
                    f'--data_type={precision}']

    assert input_format.lower() in ['bgr', 'rgb']

    if input_shape is not None:
        command_line.append(f'--input_shape={input_shape}')
    if normalize['to_rgb'] and input_format.lower() == 'bgr' or \
            not normalize['to_rgb'] and input_format.lower() == 'rgb':
        command_line.append('--reverse_input_channels')

    run(command_line, shell=False, check=True)


def optimize_onnx_graph(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = optimize(onnx_model, ['extract_constant_to_initializer',
                                       'eliminate_unused_initializer'])

    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(onnx_model, onnx_model_path)


def _get_fake_inputs(input_shape, num_classes):
    N, C, H, W = input_shape
    rng = np.random.RandomState(42)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }

    return mm_inputs


def export_model(model, config, output_dir, target='openvino', onnx_opset=11,
                 input_format='rgb', precision='FP32', output_logits=False):
    assert onnx_opset in available_opsets

    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module

    img_scale = config.data.test.pipeline[1]['img_scale']
    input_shape = (1, 3, img_scale[1], img_scale[0])

    # create dummy input
    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes
    fake_inputs = _get_fake_inputs(input_shape, num_classes)

    model = _convert_batchnorm(model)
    model.cpu().eval()

    mmcv.mkdir_or_exist(osp.abspath(output_dir))
    onnx_model_path = osp.join(output_dir, config.get('model_name', 'model') + '.onnx')

    export_to_onnx(model,
                   fake_inputs,
                   output_logits=output_logits,
                   export_name=onnx_model_path,
                   opset=onnx_opset,
                   verbose=False)
    print(f'ONNX model has been saved to "{onnx_model_path}"')

    optimize_onnx_graph(onnx_model_path)

    if target == 'openvino':
        export_to_openvino(config,
                           onnx_model_path,
                           output_dir,
                           input_shape,
                           input_format,
                           precision)
    else:
        check_onnx_model(onnx_model_path)
