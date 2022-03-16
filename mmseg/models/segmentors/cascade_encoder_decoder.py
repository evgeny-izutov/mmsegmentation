# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""

        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def set_step_params(self, init_iter, epoch_size):
        for decode_head in self.decode_head:
            decode_head.set_step_params(init_iter, epoch_size)

        if self.auxiliary_head is not None:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.set_step_params(init_iter, epoch_size)
            else:
                self.auxiliary_head.set_step_params(init_iter, epoch_size)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        features = self.extract_feat(img)

        out = self.decode_head[0].forward_test(features, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(features, out, img_metas, self.test_cfg)

        out_scale = self.test_cfg.get('output_scale', None)
        if out_scale is not None and not self.training:
            out = out_scale * out

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        repr_vector = None
        if self.test_cfg.get('return_repr_vector', False):
            if len(features) == 1:
                repr_vector = F.adaptive_avg_pool2d(features[0], (1, 1))
            else:
                pooled_features = [F.adaptive_avg_pool2d(fea_map, (1, 1))
                                   for fea_map in features]
                repr_vector = torch.cat(pooled_features, dim=1)

        return out, repr_vector

    def _decode_head_forward_train(self, x, img_metas, pixel_weights=None, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""

        losses, meta = dict(), dict()

        trg_map = self._get_argument_by_name(self.decode_head[0].loss_target_name, kwargs)
        loss_decode, prev_logits = self.decode_head[0].forward_train(
            x, img_metas, trg_map, self.train_cfg, pixel_weights
        )

        prev_scale = self.decode_head[0].last_scale
        prev_scaled_logits = prev_scale * prev_logits

        name_prefix = 'decode_0'
        losses.update(add_prefix(loss_decode, name_prefix))
        meta[f'{name_prefix}_scaled_logits'] = prev_scaled_logits

        for i in range(1, self.num_stages):
            trg_map = self._get_argument_by_name(self.decode_head[i].loss_target_name, kwargs)
            loss_decode, prev_logits = self.decode_head[i].forward_train(
                x, prev_scaled_logits, img_metas, trg_map, self.train_cfg, pixel_weights
            )

            prev_scale = self.decode_head[i].last_scale
            prev_scaled_logits = prev_scale * prev_logits

            name_prefix = f'decode_{i}'
            losses.update(add_prefix(loss_decode, name_prefix))
            meta[f'{name_prefix}_scaled_logits'] = prev_scaled_logits

        return losses, meta
