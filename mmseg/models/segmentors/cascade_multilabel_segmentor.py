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
from mmcv import ConfigDict
from .. import builder
from ..builder import SEGMENTORS
from .cascade_encoder_decoder import CascadeEncoderDecoder


@SEGMENTORS.register_module()
class CascadeMultilabelSegmentor(CascadeEncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self, multilabel_head=None, **kwargs):
        super(CascadeMultilabelSegmentor, self).__init__(**kwargs)
        multilabel_head.num_classes = self.decode_head[-1].num_classes
        self.multilabel_head = builder.build_head(multilabel_head)
        self.multilabel_head.init_weights()

    def forward_train(self, img, img_metas, gt_semantic_seg, aux_img=None, pixel_weights=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            aux_img (Tensor): Auxiliary images.
            pixel_weights (Tensor): Pixels weights.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        if not hasattr(self.train_cfg, 'mix_loss'):
            self.train_cfg.mix_loss = ConfigDict(dict(enable=False))
        enable_mix_loss = self.train_cfg.get('mix_loss') and self.train_cfg.mix_loss.get('enable', False)
        self.train_cfg.mix_loss.enable = aux_img is not None and enable_mix_loss
        if self.train_cfg.mix_loss.enable:
            img = torch.cat([img, aux_img], dim=0)
            gt_semantic_seg = torch.cat([gt_semantic_seg, gt_semantic_seg], dim=0)

        features = self.extract_feat(img)

        loss_mlc = self.multilabel_head.forward_train(features, gt_semantic_seg)
        losses.update(loss_mlc)

        loss_decode, meta_decode = self._decode_head_forward_train(
            features, img_metas, pixel_weights, gt_semantic_seg=gt_semantic_seg, **kwargs
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux, meta_aux = self._auxiliary_head_forward_train(
                features, img_metas, gt_semantic_seg=gt_semantic_seg, **kwargs
            )
            losses.update(loss_aux)

        if self.mutual_losses is not None and self.with_auxiliary_head:
            meta = dict()
            meta.update(meta_decode)
            meta.update(meta_aux)

            out_mutual_losses = dict()
            for mutual_loss_idx, mutual_loss in enumerate(self.mutual_losses):
                logits_a = self._get_argument_by_name(mutual_loss.trg_a_name, meta)
                logits_b = self._get_argument_by_name(mutual_loss.trg_b_name, meta)

                logits_a = resize(input=logits_a, size=gt_semantic_seg.shape[2:],
                                  mode='bilinear', align_corners=self.align_corners)
                logits_b = resize(input=logits_b, size=gt_semantic_seg.shape[2:],
                                  mode='bilinear', align_corners=self.align_corners)

                mutual_labels = gt_semantic_seg.squeeze(1)
                mutual_loss_value, mutual_loss_meta = mutual_loss(logits_a, logits_b, mutual_labels)

                mutual_loss_name = mutual_loss.name + f'-{mutual_loss_idx}'
                out_mutual_losses[mutual_loss_name] = mutual_loss_value
                losses[mutual_loss_name] = mutual_loss_value
                losses.update(add_prefix(mutual_loss_meta, mutual_loss_name))

            losses['loss_mutual'] = sum(out_mutual_losses.values())

        if self.loss_equalizer is not None:
            unweighted_losses = {loss_name: loss for loss_name, loss in losses.items() if 'loss' in loss_name}
            weighted_losses = self.loss_equalizer.reweight(unweighted_losses)

            for loss_name, loss_value in weighted_losses.items():
                losses[loss_name] = loss_value

        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        features = self.extract_feat(img)

        mlc_score = self.multilabel_head.simple_test(features)[0]
        mlc_filter = mlc_score > 0.5

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

        with torch.no_grad():
            for i in range(len(mlc_filter)):
                if not mlc_filter[i]:
                    out[:, i] = -1e10

        repr_vector = None
        if self.test_cfg.get('return_repr_vector', False):
            if len(features) == 1:
                repr_vector = F.adaptive_avg_pool2d(features[0], (1, 1))
            else:
                pooled_features = [F.adaptive_avg_pool2d(fea_map, (1, 1))
                                   for fea_map in features]
                repr_vector = torch.cat(pooled_features, dim=1)

        return out, repr_vector
