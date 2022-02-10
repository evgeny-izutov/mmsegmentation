import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss
from .utils import weight_reduce_loss


def mutual_loss(logits_a, logits_b):
    with torch.no_grad():
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        trg_probs = 0.5 * (probs_a + probs_b)

    log_probs_a = torch.log_softmax(logits_a, dim=1)
    log_probs_b = torch.log_softmax(logits_b, dim=1)

    losses_a = torch.sum(trg_probs * log_probs_a, dim=1).neg()
    losses_b = torch.sum(trg_probs * log_probs_b, dim=1).neg()
    losses = 0.5 * (losses_a + losses_b)

    return losses


@LOSSES.register_module()
class MutualLoss(BaseWeightedLoss):
    """MutualLoss.
    """

    def __init__(self,
                 head_a_name,
                 head_b_name,
                 **kwargs):
        super(MutualLoss, self).__init__(**kwargs)

        self.head_a_name = head_a_name
        self.head_b_name = head_b_name

    @property
    def trg_a_name(self):
        return f'{self.head_a_name}_scaled_logits'

    @property
    def trg_b_name(self):
        return f'{self.head_b_name}_scaled_logits'

    @property
    def name(self):
        return 'mutual'

    def _forward(self,
                 logits_a,
                 logits_b,
                 labels,
                 avg_factor=None,
                 reduction_override=None):
        assert logits_a.size() == logits_b.size()

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        num_classes = logits_a.size(1)
        valid_labels = torch.clamp(labels, 0, num_classes - 1)
        valid_mask = labels != self.ignore_index

        losses = mutual_loss(logits_a, logits_b)
        losses = torch.where(valid_mask, losses, torch.zeros_like(losses))

        weight = None
        if self.sampler is not None:
            weight = self.sampler(losses, None, valid_labels, valid_mask)

        loss = weight_reduce_loss(
            losses,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor
        )

        meta = dict()

        return loss, meta
