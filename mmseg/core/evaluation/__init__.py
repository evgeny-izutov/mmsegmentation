# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .class_names import get_classes, get_palette
from .eval_hooks import (
    DistEvalHook,
    DistEvalPlusBeforeRunHook,
    EvalHook,
    EvalPlusBeforeRunHook
)
from .metrics import eval_metrics, mean_dice, mean_fscore, mean_iou
from .checkpoints import CustomCheckpointHook

__all__ = [
    'DistEvalHook',
    'DistEvalPlusBeforeRunHook',
    'EvalHook',
    'EvalPlusBeforeRunHook',
    'eval_metrics',
    'get_classes',
    'get_palette',
    'mean_dice',
    'mean_fscore',
    'mean_iou',
    'CustomCheckpointHook',
]
