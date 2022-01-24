# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F

from mmseg.core import focal_loss
from ..builder import LOSSES
from .utils import get_class_weight
from .pixel_base import BasePixelLoss


def cross_entropy(pred,
                  label,
                  class_weight=None,
                  ignore_index=255):
    """The wrapper function for :func:`F.cross_entropy`"""

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index
    )

    return loss


def _expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels


def binary_cross_entropy(pred,
                         label,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """

    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label = _expand_onehot_labels(label, pred.shape, ignore_index)

    loss = F.binary_cross_entropy_with_logits(
        pred,
        label.float(),
        pos_weight=class_weight,
        reduction='none'
    )

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """

    assert ignore_index is None, 'BCE loss does not support ignore_index'

    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    loss = F.binary_cross_entropy_with_logits(
        pred_slice,
        target,
        weight=class_weight,
        reduction='mean'
    )[None]

    return loss


@LOSSES.register_module()
class CrossEntropyLoss(BasePixelLoss):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 gamma=0.0,
                 **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)

        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.class_weight = get_class_weight(class_weight)
        assert gamma >= 0.0
        self.gamma = gamma

        assert (use_sigmoid is False) or (use_mask is False)
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    @property
    def name(self):
        return 'ce'

    def _calculate(self, cls_score, label, scale):
        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        out_losses = self.cls_criterion(
            scale * cls_score,
            label,
            class_weight=class_weight,
            ignore_index=self.ignore_index
        )

        if self.gamma > 0.0:
            out_losses = focal_loss(out_losses, self.gamma)

        return out_losses, cls_score
