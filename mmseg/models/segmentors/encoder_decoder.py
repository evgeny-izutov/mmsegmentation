# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict

from mmseg.core import add_prefix, FeatureVectorHook
from mmseg.ops import resize
from mmseg.models.losses import LossEqualizer
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_train_components(self.train_cfg)

        self.init_weights(pretrained=pretrained)
        self.feature_maps = None
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""

        self.auxiliary_head = None
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def set_step_params(self, init_iter, epoch_size):
        self.decode_head.set_step_params(init_iter, epoch_size)

        if self.auxiliary_head is not None:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.set_step_params(init_iter, epoch_size)
            else:
                self.auxiliary_head.set_step_params(init_iter, epoch_size)

    def _init_train_components(self, train_cfg):
        if train_cfg is None:
            self.mutual_losses = None
            self.loss_equalizer = None
            return

        mutual_loss_configs = train_cfg.get('mutual_loss')
        if mutual_loss_configs:
            if isinstance(mutual_loss_configs, dict):
                mutual_loss_configs = [mutual_loss_configs]

            self.mutual_losses = nn.ModuleList()
            for mutual_loss_config in mutual_loss_configs:
                self.mutual_losses.append(builder.build_loss(mutual_loss_config))
        else:
            self.mutual_losses = None

        loss_reweighting_config = train_cfg.get('loss_reweighting')
        if loss_reweighting_config:
            self.loss_equalizer = LossEqualizer(**loss_reweighting_config)
        else:
            self.loss_equalizer = None

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""

        x = self.backbone(img)

        if torch.onnx.is_in_onnx_export():
            self.feature_maps = x

        if self.with_neck:
            x = self.neck(x)

        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        features = self.extract_feat(img)

        out = self._decode_head_forward_test(features, img_metas)

        out_scale = self.test_cfg.get('output_scale', None)
        if out_scale is not None and not self.training:
            out = out_scale * out

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        return out

    def _decode_head_forward_train(self, x, img_metas, pixel_weights=None, **kwargs):
        """Run forward function and calculate loss for decode head in training."""

        trg_map = self._get_argument_by_name(self.decode_head.loss_target_name, kwargs)
        loss_decode, logits_decode = self.decode_head.forward_train(
            x,
            img_metas,
            trg_map,
            self.train_cfg,
            pixel_weights
        )

        scale = self.decode_head.last_scale
        scaled_logits_decode = scale * logits_decode

        name_prefix = 'decode'

        losses, meta = dict(), dict()
        losses.update(add_prefix(loss_decode, name_prefix))
        meta[f'{name_prefix}_scaled_logits'] = scaled_logits_decode

        return losses, meta

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, **kwargs):
        """Run forward function and calculate loss for auxiliary head in training."""

        losses, meta = dict(), dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                trg_map = self._get_argument_by_name(aux_head.loss_target_name, kwargs)
                loss_aux, logits_aux = aux_head.forward_train(
                    x,
                    img_metas,
                    trg_map,
                    self.train_cfg
                )

                scale = aux_head.last_scale
                scaled_logits_aux = scale * logits_aux

                name_prefix = f'aux_{idx}'
                losses.update(add_prefix(loss_aux, name_prefix))
                meta[f'{name_prefix}_scaled_logits'] = scaled_logits_aux
        else:
            trg_map = self._get_argument_by_name(self.auxiliary_head.loss_target_name, kwargs)
            loss_aux, logits_aux = self.auxiliary_head.forward_train(
                x,
                img_metas,
                trg_map,
                self.train_cfg
            )

            scale = self.auxiliary_head.last_scale
            scaled_logits_aux = scale * logits_aux

            name_prefix = 'aux'
            losses.update(add_prefix(loss_aux, name_prefix))
            meta[f'{name_prefix}_scaled_logits'] = scaled_logits_aux

        return losses, meta

    @staticmethod
    def _get_argument_by_name(trg_name, arguments):
        assert trg_name in arguments.keys()

        return arguments[trg_name]

    def _reweight_losses(self, losses):
        pass

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)
        return seg_logit

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

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        # TODO[EUGENE]: Not used for MPA seg, and need to find a way to aggregate per-tile feature vec and map
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat

        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)

        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, output_logits=False):
        """Simple test with single image."""

        seg_logit = self.inference(img, img_meta, rescale)
        if output_logits:
            seg_pred = seg_logit
        else:
            seg_pred = seg_logit.argmax(dim=1)

        if torch.onnx.is_in_onnx_export():
            feature_vector = FeatureVectorHook.func(self.feature_maps)
            if not output_logits:
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
            return seg_pred, feature_vector

        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)

        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """

        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit, _ = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)

        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)  # unravel batch dim

        return seg_pred
