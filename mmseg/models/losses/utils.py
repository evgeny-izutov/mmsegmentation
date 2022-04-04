# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

import mmcv
import numpy as np
import torch
from nncf.torch.dynamic_graph.context import no_nncf_trace


def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = mmcv.load(class_weight)

    return class_weight


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean", "valid_mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'valid_mean':
        valid_mask = loss > 0.0
        num_valid = valid_mask.sum().float().clamp_min(1.0)
        return loss.sum() / num_valid
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError(f'avg_factor can not be used with reduction="{reduction}"')

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


class LossEqualizer:
    def __init__(self, weights=None, momentum=0.1):
        self.momentum = momentum

        self.trg_ratios = None
        if weights is not None:
            assert isinstance(weights, dict)
            assert len(weights) > 0

            sum_weight = 0.0
            for loss_weight in weights.values():
                assert loss_weight > 0
                sum_weight += float(loss_weight)
            assert sum_weight > 0.0

            self.trg_ratios = {
                loss_name: float(loss_weight) / sum_weight
                for loss_name, loss_weight in weights.items()
            }

        self._smoothed_values = dict()

    def reweight(self, losses):
        assert isinstance(losses, dict)

        if len(losses) == 0:
            return losses

        for loss_name, loss_value in losses.items():
            if loss_name not in self._smoothed_values:
                self._smoothed_values[loss_name] = loss_value.item()
            else:
                smoothed_loss = self._smoothed_values[loss_name]
                self._smoothed_values[loss_name] = (1.0 - self.momentum) * smoothed_loss + \
                                                   self.momentum * loss_value.item()

        if len(self._smoothed_values) == 1:
            return losses

        with no_nncf_trace():
            total_sum = sum(self._smoothed_values.values())
            trg_value_default = total_sum / float(len(self._smoothed_values))

            weighted_losses = dict()
            for loss_name, loss_value in losses.items():
                if self.trg_ratios is not None:
                    assert loss_name in self.trg_ratios.keys()
                    trg_value = self.trg_ratios[loss_name] * total_sum
                else:
                    trg_value = trg_value_default

                loss_weight = trg_value / self._smoothed_values[loss_name]
                weighted_losses[loss_name] = loss_weight * loss_value

        return weighted_losses
