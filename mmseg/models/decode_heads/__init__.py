# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead, AuxOCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .uper_head import UPerHead
from .ddr_head import DDRHead
from .bise_head import BiSeHead
from .shelf_head import ShelfHead
from .hyperseg_head import HyperSegHead
from .memory_head import MemoryHead
from .sptaial_gather_fcn_head import SpatialGatherFCNHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'PSAHead',
    'NLHead',
    'GCHead',
    'CCHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'ANNHead',
    'DAHead',
    'OCRHead',
    'AuxOCRHead',
    'EncHead',
    'DepthwiseSeparableFCNHead',
    'FPNHead',
    'EMAHead',
    'DNLHead',
    'PointHead',
    'APCHead',
    'DMHead',
    'LRASPPHead',
    'DDRHead',
    'BiSeHead',
    'ShelfHead',
    'HyperSegHead',
    'MemoryHead',
    'SpatialGatherFCNHead',
]
