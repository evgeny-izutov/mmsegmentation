# Copyright (C) 2020 Litao Yu
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Modified from: https://github.com/yutao1008/margin_calibration"""


import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .pixel_base import BasePixelLoss


@LOSSES.register_module()
class MarginCalibrationLoss(BasePixelLoss):
    """Computes the Margin Calibration loss: https://arxiv.org/abs/2112.11554
    """

    def __init__(self, **kwargs):
        super(MarginCalibrationLoss, self).__init__(**kwargs)

    @property
    def name(self):
        return 'margin-calibration'
    
    def set_margins(self, margins):
        self.register_buffer('margins', torch.tensor(margins))

    def _calculate(self, scores, target, scale):
        assert hasattr(self, 'margins')

        b, num_classes, h, w = scores.size()
        logits = scores.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        fixed_target = torch.where(valid_mask, target, 0)

        max2_score, inds = logits.topk(k=2, dim=1)
        sub_max_inds = inds[:, 0].expand(num_classes, -1).t() == torch.arange(num_classes, device=scores.device)
        sub_max_score = torch.gather(max2_score, 1, sub_max_inds.long())
        scores_all = scale * (logits - sub_max_score)

        with torch.no_grad():
            margins_all = -self.margins[1].expand(scores_all.shape)
            p_margins = torch.gather(self.margins[0], 0, fixed_target)
            margins_all.scatter_(1, fixed_target.unsqueeze(1), p_margins.unsqueeze(1))
        scores_all -= margins_all

        all_losses = F.binary_cross_entropy_with_logits(
            scores_all,
            (margins_all > 0).float(),
            reduction='none'
        )
        out_losses = all_losses.view(b, h, w, num_classes).sum(dim=3)

        return out_losses, scores
