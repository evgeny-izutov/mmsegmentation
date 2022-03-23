# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2019 Vladislav Sovrasov
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse

import torch
import numpy as np
from mmcv import Config

from mmseg.models import build_segmentor
from mmseg.models.backbones.efficientnet import Conv2dDynamicSamePadding, Conv2dStaticSamePadding

try:
    from ptflops import get_model_complexity_info
except ImportError:
    raise ImportError('Please install ptflops: `pip install ptflops`')
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    raise ImportError('Please install fvcore: `pip install -U fvcore`')


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def parse_args():
    parser = argparse.ArgumentParser(description='Count FLops and MParams of a segmentor')
    parser.add_argument('config',
                        help='train config file path')
    parser.add_argument('--shape', type=int, nargs='+', default=[2048, 1024],
                        help='input image size')
    parser.add_argument('--per_layer_stat', action='store_true',
                        help='Show per layer stat of the model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()

    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        print_per_layer_stat=args.per_layer_stat,
        custom_modules_hooks={
            Conv2dDynamicSamePadding: conv_flops_counter_hook,
            Conv2dStaticSamePadding: conv_flops_counter_hook,
        })

    input = torch.ones(()).new_empty((1, *input_shape),
                                     dtype=next(model.parameters()).dtype,
                                     device=next(model.parameters()).device)
    flops_alt = FlopCountAnalysis(model, input)
    flops_alt = flops_to_string(flops_alt.total())

    split_line = '=' * 30
    print('{0}\n'
          'Input shape: {1}\n'
          'Macs (ptflops): {2}\n'
          'Macs (fvcore): {3}\n'
          'Params: {4}\n'
          '{0}'.format(split_line, input_shape, flops, flops_alt, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
