# Copyright (C) 2019-2020 LikeLy-Journey
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018-2021 kornia
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Modified from:
   - https://github.com/LikeLy-Journey/SegmenTron/blob/master/segmentron/solver/loss.py (Apache-2.0 License)
   - https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/tversky.html"""


import torch
import torch.nn.functional as F

from .. import builder
from ..builder import LOSSES
from .utils import get_class_weight
from .base import BaseWeightedLoss


def compute_loss(pred, target, valid_mask, smooth, gamma, alpha, beta, focal_gamma=None,
                 class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
    assert pred.shape[0] == target.shape[0]

    num_classes = pred.shape[1]
    if num_classes == 1:
        class_ids = [0] if ignore_index != 0 else []
    elif num_classes == 2:
        class_ids = [1] if ignore_index != 1 else []
    else:
        class_ids = [i for i in range(num_classes) if i != ignore_index]
    assert len(class_ids) >= 1

    class_losses = []
    for i in class_ids:
        target_loss_value = binary_target_loss(
            pred[:, i],
            target[..., i],
            valid_mask=valid_mask,
            smooth=smooth,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
        )

        if focal_gamma is not None and focal_gamma != 1.0:
            target_loss_value = target_loss_value.pow(focal_gamma)

        if class_weight is not None:
            target_loss_value = class_weight[i] * target_loss_value

        class_losses.append(target_loss_value)

    if avg_factor is None:
        if reduction == 'mean':
            loss = sum(class_losses) / float(len(class_losses))
        elif reduction == 'sum':
            loss = sum(class_losses)
        elif reduction == 'none':
            loss = class_losses
        else:
            raise ValueError(f'unknown reduction type: {reduction}')
    else:
        if reduction == 'mean':
            loss = sum(class_losses) / avg_factor
        elif reduction == 'none':
            loss = class_losses
        else:
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss


def binary_target_loss(pred, target, valid_mask, smooth, gamma, alpha, beta):
    assert pred.shape[0] == target.shape[0]

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    valid_pred = torch.mul(pred, valid_mask)
    valid_target = torch.mul(target, valid_mask)

    tps = torch.sum(valid_pred * valid_target, dim=1)
    fps = torch.sum((valid_pred * (1.0 - valid_target)).pow(gamma), dim=1)
    fns = torch.sum(((1.0 - valid_pred) * valid_target).pow(gamma), dim=1)

    numerator = tps + smooth
    denominator = tps + alpha * fps + beta * fns + smooth

    return torch.mean(1.0 - numerator / denominator)


@LOSSES.register_module()
class GeneralizedDiceLoss(BaseWeightedLoss):
    """GeneralizedDiceLoss.

    Implements several common losses:

    * Dice loss. This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
      Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

      Parameter values: gamma = 1.0, alpha = 0.5, beta = 0.5

    * Tversky loss. This loss is proposed in `Tversky loss function for image segmentation
      using 3D fully convolutional deep networks <https://arxiv.org/abs/1706.05721>`_.
      Modified from https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/tversky.html

      Parameter values: gamma = 1.0, alpha = 0.3, beta = 0.7

    * Dice++ loss. his loss is proposed in 'Calibrating the Dice loss to handle neural network
       overconfidence for biomedical image segmentation <https://arxiv.org/abs/2111.00528>'.

       Parameter values: gamma = 2.0, alpha = 0.5, beta = 0.5


    """

    def __init__(self,
                 smooth=1.0,
                 gamma=1.0,
                 alpha=0.5,
                 beta=0.5,
                 focal_gamma=1.0,
                 class_weight=None,
                 scale_cfg=None,
                 **kwargs):
        super(GeneralizedDiceLoss, self).__init__(**kwargs)

        self.smooth = float(smooth)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.focal_gamma = float(focal_gamma)
        self.class_weight = get_class_weight(class_weight)

        self._scale_scheduler = builder.build_scheduler(scale_cfg, default_value=1.0)

        self._last_scale = 0.0

    @property
    def last_scale(self):
        return self._last_scale

    @property
    def name(self):
        return 'gdice'

    def _forward(self,
                 pred,
                 target,
                 avg_factor=None,
                 reduction_override=None,
                 **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        self._last_scale = self._scale_scheduler(self.iter, self.epoch_size)

        pred = F.softmax(self._last_scale * pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes
        )

        valid_mask = (target != self.ignore_index).long()

        loss = compute_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            smooth=self.smooth,
            gamma=self.gamma,
            alpha=self.alpha,
            beta=self.beta,
            focal_gamma=self.focal_gamma,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index
        )
        meta = dict()

        return loss, meta
