# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import torch.nn.functional as F

from mmseg.core import build_classification_loss, focal_loss
from ..builder import LOSSES
from .pixel_base import BasePixelLoss


@LOSSES.register_module()
class AMSoftmaxLoss(BasePixelLoss):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', margin=0.5, gamma=0.0, t=1.0, target_loss='ce', **kwargs):
        super(AMSoftmaxLoss, self).__init__(**kwargs)

        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0.0
        self.gamma = gamma
        assert margin >= 0.0
        self.m = margin
        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m)
        assert t >= 1
        self.t = t

        self.target_loss = build_classification_loss(target_loss)

    @property
    def name(self):
        return 'am-softmax'

    @staticmethod
    def _one_hot_mask(target, num_classes):
        return F.one_hot(target.detach(), num_classes).permute(0, 3, 1, 2).bool()

    def _calculate(self, cos_theta, target, scale):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        num_classes = cos_theta.size(1)
        one_hot_mask = self._one_hot_mask(target, num_classes)
        output = torch.where(one_hot_mask, phi_theta, cos_theta)

        if self.t > 1.0:
            h_theta = self.t - 1 + self.t * cos_theta
            support_vectors_mask = (~one_hot_mask) * \
                torch.lt(torch.masked_select(phi_theta, one_hot_mask).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vectors_mask, h_theta, output)

        out_losses = self.target_loss(scale * output, target)

        if self.gamma > 0.0:
            out_losses = focal_loss(out_losses, self.gamma)

        return out_losses, output
