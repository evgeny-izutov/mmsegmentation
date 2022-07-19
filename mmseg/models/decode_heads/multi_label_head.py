# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
import torch.nn.functional as F

from ..builder import HEADS
from mmcls.models.heads.multi_label_linear_head import MultiLabelLinearClsHead
from mmseg.models.backbones.resnet import Bottleneck


@HEADS.register_module()
class MultiLabelHead(MultiLabelLinearClsHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self, pre_stages=None, **kwargs):
        super(MultiLabelHead, self).__init__(**kwargs)
        if pre_stages and isinstance(pre_stages, list):
            # Classification Head
            self.incre_modules, self.downsamp_modules, \
                self.final_layer = self._make_head(pre_stages)

    def init_weights(self):
        normal_init(self.incre_modules, mean=0, std=0.01, bias=0)
        normal_init(self.downsamp_modules, mean=0, std=0.01, bias=0)
        normal_init(self.final_layer, mean=0, std=0.01, bias=0)
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        BN_MOMENTUM = 0.1
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels):
        BN_MOMENTUM = 0.1
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        # head_channels = [16, 32, 64, 128]


        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward_train(self, x, gt_label):
        
        gt_label = [torch.unique(gt) for gt in gt_label]
        onehot_labels = []

        for label in gt_label:
            onehot_label = torch.zeros(self.num_classes)
            for lbl in label:
                if lbl == 255:
                    continue
                onehot_label[lbl] = 1
            onehot_labels.append(onehot_label)
        onehot_labels = torch.stack(onehot_labels)
        if isinstance(x, list):
            y = self.incre_modules[0](x[0])
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i+1](x[i+1]) + \
                            self.downsamp_modules[i](y)
            y = self.final_layer(y)

            if torch._C._get_tracing_state():
                y = y.flatten(start_dim=2).mean(dim=2)
            else:
                y = F.avg_pool2d(y, kernel_size=y.size()
                                    [2:]).view(y.size(0), -1)
            cls_score = self.fc(y)

        with torch.no_grad():
            pred = torch.sigmoid(cls_score) > 0.5
            accuracy = (pred == onehot_labels.cuda()).sum() / pred.numel()
        losses = self.loss(cls_score, onehot_labels)
        losses['multilabel_loss'] = losses.pop('loss')
        losses['multilabel_acc'] = accuracy*100

        return losses


    def simple_test(self, img):
        """Test without augmentation."""

        if isinstance(img, list):
            y = self.incre_modules[0](img[0])
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i+1](img[i+1]) + \
                            self.downsamp_modules[i](y)

            y = self.final_layer(y)

            if torch._C._get_tracing_state():
                y = y.flatten(start_dim=2).mean(dim=2)
            else:
                y = F.avg_pool2d(y, kernel_size=y.size()
                                    [2:]).view(y.size(0), -1)
            cls_score = self.fc(y)
        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.sigmoid(cls_score) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred