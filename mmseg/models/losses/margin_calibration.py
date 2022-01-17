import numpy as np
import torch
import torch.nn.functional as F

from mmseg.core import focal_loss
from ..builder import LOSSES
from .pixel_base import BasePixelLoss


@LOSSES.register_module()
class MarginCalibrationLoss(BasePixelLoss):
    """Computes the Margin Calibration loss: https://arxiv.org/abs/2112.11554"""

    def __init__(self, **kwargs):
        super(MarginCalibrationLoss, self).__init__(**kwargs)

        self.margins = None
        self.neg_score = -1e7

    @property
    def name(self):
        return 'margin-calibration'
    
    def set_margins(self, margins):
        self.register_buffer('margins', torch.tensor(margins))
    
    @staticmethod
    def _one_hot_mask(target, num_classes):
        return F.one_hot(target.detach(), num_classes).permute(0, 3, 1, 2).bool()

    def _calculate(self, scores, target, scale):
        assert self.margins is not None

        num_classes = scores.size(1)
        logits = scores.premute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        valid_logits = logits[valid_mask]
        valid_target = target[valid_mask]

        max2_score, inds = valid_logits.topk(k=2, dim=1)
        sub_max_inds = inds[:, 0].expand(num_classes, -1).t() == torch.arange(num_classes, device=scores.device)
        sub_max_score = torch.gather(max2_score, 1, sub_max_inds.long())
        scores_all = valid_logits - sub_max_score

        with torch.no_grad():
            margins_all = -self.margins[1].expand(scores_all.shape)
            p_margins = torch.gather(self.margins[0], 0, valid_target)
            margins_all.scatter_(1, valid_target.unsqueeze(1), p_margins.unsqueeze(1))
        scores_all -= margins_all

        loss_mean = F.binary_cross_entropy_with_logits(
            scores_all,
            (margins_all > 0).float(),
            pos_weight=torch.tensor(num_classes)
        )

        



        return out_losses, cos_theta
