# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import re

import torch
import numpy as np
from terminaltables import AsciiTable
from mmcv.runner.checkpoint import _load_checkpoint
from mmcv.runner.dist_utils import get_dist_info


def _is_cls_layer(name):
    return 'fc_angular' in name or 'fc_cls_out' in name


def _get_dataset_id(name):
    return int(name.split('cls_head.')[-1].split('.')[0])


def load_state_dict(module, in_state, class_maps=None, strict=False, logger=None, force_matching=False,
                    show_converted=False, ignore_keys=None):
    rank, _ = get_dist_info()

    unexpected_keys = []
    converted_pairs = []
    shape_mismatch_pairs = []
    shape_casted_pairs = []

    own_state = module.state_dict()
    for name, in_param in in_state.items():
        if ignore_keys is not None:
            ignored = any(re.match(ignore_key, name) for ignore_key in ignore_keys)
            if ignored:
                continue

        if name not in own_state:
            unexpected_keys.append(name)
            continue

        out_param = own_state[name]
        if isinstance(out_param, torch.nn.Parameter):
            out_param = out_param.data
        if isinstance(in_param, torch.nn.Parameter):
            in_param = in_param.data

        src_shape = in_param.size()
        trg_shape = out_param.size()
        if src_shape != trg_shape:
            if np.prod(src_shape) == np.prod(trg_shape):
                out_param.copy_(in_param.view(trg_shape))
                shape_casted_pairs.append([name, list(out_param.size()), list(in_param.size())])
                continue

            is_valid = False
            if force_matching:
                is_valid = len(src_shape) == len(trg_shape)
                for i in range(len(src_shape)):
                    is_valid &= src_shape[i] >= trg_shape[i]

            if is_valid:
                if not (name.endswith('.weight') or name.endswith('.bias')):
                    continue

                if class_maps is not None and _is_cls_layer(name):
                    dataset_id = 0
                    if len(class_maps) > 1:
                        dataset_id = _get_dataset_id(name)
                    class_map = class_maps[dataset_id]

                    if 'fc_angular' in name:
                        for src_id, trg_id in class_map.items():
                            out_param[:, src_id] = in_param[:, trg_id]
                    else:
                        for src_id, trg_id in class_map.items():
                            out_param[src_id] = in_param[trg_id]
                else:
                    ind = [slice(0, d) for d in list(trg_shape)]
                    out_param.copy_(in_param[ind])

                shape_casted_pairs.append([name, list(out_param.size()), list(in_param.size())])
            else:
                shape_mismatch_pairs.append([name, list(out_param.size()), list(in_param.size())])
        else:
            out_param.copy_(in_param)
            if show_converted:
                converted_pairs.append([name, list(out_param.size())])

    missing_keys = list(set(own_state.keys()) - set(in_state.keys()))
    if ignore_keys is not None:
        filtered_missing_keys = []
        for missing_key in missing_keys:
            ignored = any(re.match(ignore_key, missing_key) for ignore_key in ignore_keys)
            if not ignored:
                filtered_missing_keys.append(missing_key)

        missing_keys = filtered_missing_keys

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))

    if shape_mismatch_pairs:
        casted_info = 'these keys have mismatched shape:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(casted_info + table.table)

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)

    ok_message = []
    if converted_pairs:
        converted_info = 'These keys have been matched correctly:\n'
        header = ['key', 'shape']
        table_data = [header] + converted_pairs
        table = AsciiTable(table_data)
        ok_message.append(converted_info + table.table)

    if len(ok_message) > 0 and rank == 0:
        ok_message = '\n'.join(ok_message)
        if logger is not None:
            logger.info(ok_message)

    warning_msg = []
    if shape_casted_pairs:
        casted_info = 'these keys have been shape casted:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_casted_pairs
        table = AsciiTable(table_data)
        warning_msg.append(casted_info + table.table)

    if len(warning_msg) > 0 and rank == 0:
        warning_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        warning_msg = '\n'.join(warning_msg)
        if logger is not None:
            logger.warning(warning_msg)


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None,
                    force_matching=False,
                    show_converted=False,
                    revise_keys=[(r'^module\.', '')],
                    ignore_keys=None):
    # load checkpoint
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    # extract model
    model = model.module if hasattr(model, 'module') else model

    # # load model classes
    # assert hasattr(model, 'CLASSES')
    # assert isinstance(model.CLASSES, dict)
    # model_all_classes = model.CLASSES
    #
    # # build class mapping between model.classes and checkpoint.classes
    # if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
    #     checkpoint_all_classes = checkpoint['meta']['CLASSES']
    #
    #     assert set(model_all_classes.keys()).issubset(checkpoint_all_classes.keys()),\
    #         f'The model set of datasets is not a subset of checkpoint datasets: ' \
    #         f'{model_all_classes.keys()} vs {checkpoint_all_classes.keys()}'
    #
    #     class_maps = dict()
    #     for dataset_id in model_all_classes.keys():
    #         model_dataset_classes = model_all_classes[dataset_id]
    #         checkpoint_dataset_classes = checkpoint_all_classes[dataset_id]
    #         assert set(model_dataset_classes.values()).issubset(checkpoint_dataset_classes.values()), \
    #             f'The model set of classes is not a subset of checkpoint classes'
    #
    #         checkpoint_inv_class_map = {v: k for k, v in checkpoint_dataset_classes.items()}
    #         class_maps[dataset_id] = {k: checkpoint_inv_class_map[v] for k, v in model_dataset_classes.items()}
    # else:
    #     class_maps = model_all_classes
    class_maps = None

    if ignore_keys is not None and not isinstance(ignore_keys, (tuple, list)):
        ignore_keys = [ignore_keys]

    # load weights
    load_state_dict(model, state_dict, class_maps,
                    strict, logger, force_matching,
                    show_converted, ignore_keys)

    return checkpoint
