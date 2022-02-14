# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0
        assert isinstance(dilation, int) and dilation > 0

        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size

        super(FCNHead, self).__init__(**kwargs)

        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation

        self.convs = None
        if num_convs > 0:
            convs = []
            for i in range(num_convs):
                convs.append(self._build_conv_module(
                    self.in_channels if i == 0 else self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg if i < num_convs - 1 else None
                ))
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = self._build_conv_module(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )

    def _build_conv_module(self, in_channels, out_channels, **kwargs):
        return ConvModule(
            in_channels,
            out_channels,
            **kwargs
        )
    
    def process_logits(self, logits, features):
        return logits

    def forward(self, inputs):
        """Forward function."""

        x = self._transform_inputs(inputs)

        y = self.convs(x) if self.convs is not None else x
        if self.concat_input:
            y = self.conv_cat(torch.cat([x, y], dim=1))

        logits = self.cls_seg(y)
        logits = self.process_logits(logits, y)

        return logits
