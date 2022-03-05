# Copyright (C) 2020-2021 YuvalNirkin
# SPDX-License-Identifier: CC0-1.0
#
# Copyright (c) 2020-2021 Luke Melas-Kyriazi
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Source repo: https://github.com/YuvalNirkin/hyperseg

   model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
   Author: lukemelas (github username)
   Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
   With adjustments and added comments by workingcoder (github username).
"""

import re
import math
import collections
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import constant_init, normal_init

from mmseg.utils import get_root_logger
from ..builder import BACKBONES

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate',
    'depth_divisor', 'min_depth', 'out_feat_scale', 'base_embedding_channels',
    'add_input',
])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'
])

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """

    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """

    if input_image_size is None:
        return None

    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))

    return [image_height, image_width]


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """

    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw) # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class Identity(nn.Module):
    """Identity mapping.
       Send input to output directly.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)

        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size

        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """

    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters

    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth

    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.

    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new repeat: New repeat number after calculating.
    """

    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats

    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        inputs (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """

    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor

    return output


class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string)
        )

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """

        assert isinstance(string_list, list)

        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))

        return blocks_args


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """

    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-c0': (1.0, 1.0, 224, 0.2),
        'efficientnet-c1': (1.0, 1.1, 240, 0.2),
        'efficientnet-c2': (1.1, 1.2, 260, 0.3),
        'efficientnet-c3': (1.2, 1.4, 300, 0.3),
        'efficientnet-c4': (1.4, 1.8, 380, 0.4),
        'efficientnet-c5': (1.6, 2.2, 456, 0.4),
        'efficientnet-c6': (1.8, 2.6, 528, 0.5),
        'efficientnet-c7': (2.0, 3.1, 600, 0.5),
        'efficientnet-c8': (2.2, 3.6, 672, 0.5),
        'efficientnet-s0': (1.0, 1.0, 224, 0.2),
        'efficientnet-s1': (1.0, 1.1, 240, 0.2),
        'efficientnet-s2': (1.1, 1.2, 260, 0.3),
        'efficientnet-s3': (1.2, 1.4, 300, 0.3),
        'efficientnet-s4': (1.4, 1.8, 380, 0.4),
        'efficientnet-s5': (1.6, 2.2, 456, 0.4),
        'efficientnet-s6': (1.8, 2.6, 528, 0.5),
        'efficientnet-s7': (2.0, 3.1, 600, 0.5),
        'efficientnet-s8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }

    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, out_feat_scale=0.25, blocks_args=None, base_embedding_channels=1280):

    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    if blocks_args is None:
        blocks_args = [
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25',
            'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        out_feat_scale=out_feat_scale,
        base_embedding_channels=base_embedding_channels,
        add_input=False
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.

    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.

    Returns:
        blocks_args, global_params
    """

    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2

        blocks_args = None
        base_embedding_channels = 1280
        if model_name[-2] == 'c':
            blocks_args = [
                'r1_k3_s11_e1_i32_o16_se0.25',
                'r2_k3_s22_e6_i16_o24_se0.25',
                'r2_k5_s22_e6_i24_o40_se0.25',
                'r3_k3_s22_e6_i40_o80_se0.25',
                'r3_k5_s11_e6_i80_o112_se0.25',
                'r4_k5_s22_e6_i112_o192_se0.25',
                'r4_k5_s22_e6_i192_o320_se0.25',
                'r1_k3_s11_e6_i320_o480_se0.25',
            ]
            base_embedding_channels = 1920
        elif model_name[-2] == 's':
            blocks_args = [
                'r1_k3_s22_e1_i32_o16_se0.25',
                'r2_k3_s22_e6_i16_o24_se0.25',
                'r2_k5_s22_e6_i24_o40_se0.25',
                'r3_k3_s22_e6_i40_o80_se0.25',
                'r3_k5_s11_e6_i80_o112_se0.25',
                'r4_k5_s22_e6_i112_o192_se0.25',
                'r1_k3_s11_e6_i192_o320_se0.25',
            ]

        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p,
                                                  image_size=s, blocks_args=blocks_args,
                                                  base_embedding_channels=base_embedding_channels)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)

    return blocks_args, global_params


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()

        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1,1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection

        return x


@BACKBONES.register_module()
class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False):
        super().__init__()

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        model_name = self.extra['model_name']
        self._check_model_name_is_valid(model_name)

        override_params = dict(out_feat_scale=list(self.extra['out_feat_scale']),
                               add_input=self.extra['add_input'])
        blocks_args, global_params = get_model_params(model_name, override_params)

        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        self._res_feat_mask = []
        feat_nc = []
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # Calculate whether the current block returns the final feature map for its corresponding resolution
            if block_args.stride[0] > 1:
                self._res_feat_mask[-1] = True
            self._res_feat_mask += ([False] * block_args.num_repeat)
            feat_nc += ([block_args.output_filters] * block_args.num_repeat)

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))

        # Add the last resolution feature map
        self._res_feat_mask[-1] = True
        self.feat_channels = list(np.array(feat_nc)[self._res_feat_mask])

        # Add feature fcs
        if self._global_params.out_feat_scale is not None:
            for i, in_nc in enumerate(self.feat_channels):
                if isinstance(self._global_params.out_feat_scale, (list, tuple)):
                    scale = self._global_params.out_feat_scale[i]
                else:
                    scale = self._global_params.out_feat_scale

                if scale != 1.:
                    out_nc = int(round(in_nc * scale))
                    fc = nn.Sequential(
                        Conv2d(in_nc, out_nc, kernel_size=1, bias=False),
                        nn.BatchNorm2d(num_features=out_nc, momentum=bn_mom, eps=bn_eps))
                    self.add_module(f'_feat_fc_{i}', fc)
                else:
                    out_nc = in_nc
                    setattr(self, f'_feat_fc_{i}', None)

                self.feat_channels[i] = out_nc

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self.feat_channels.append(out_channels)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._swish = Swish()

    @staticmethod
    def _check_model_name_is_valid(model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """

        valid_models = ['efficientnet-b' + str(i) for i in range(9)]

        # Support the construction of 'efficientnet-l2' without pretrained weights
        valid_models += ['efficientnet-l2']

        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        input_x = x

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        y_list = []
        for idx, block in enumerate(self._blocks):

            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

            # Add final feature of each resolution to the list of features
            if self._res_feat_mask[idx]:
                if self._global_params.out_feat_scale is None:
                    y_list.append(x)
                else:
                    fc = getattr(self, f'_feat_fc_{len(y_list)}')
                    if fc is None:
                        y_list.append(x)
                    else:
                        y_list.append(fc(x))

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        x = self._dropout(x)
        y_list.append(x)

        if self._global_params.add_input:
            y_list = [input_x] + y_list

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""

        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
