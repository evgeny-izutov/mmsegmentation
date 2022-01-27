# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import CrossEntropyLoss, binary_cross_entropy, cross_entropy, mask_cross_entropy
from .lovasz_loss import LovaszLoss
from .generalized_dice_loss import GeneralizedDiceLoss
from .am_softmax import AMSoftmaxLoss
from .mutual_loss import MutualLoss
from .margin_calibration import MarginCalibrationLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss, LossEqualizer

__all__ = [
    'accuracy',
    'Accuracy',
    'cross_entropy',
    'binary_cross_entropy',
    'mask_cross_entropy',
    'CrossEntropyLoss',
    'reduce_loss',
    'weight_reduce_loss',
    'weighted_loss',
    'LovaszLoss',
    'GeneralizedDiceLoss',
    'AMSoftmaxLoss',
    'MutualLoss',
    'MarginCalibrationLoss',
    'LossEqualizer',
]
