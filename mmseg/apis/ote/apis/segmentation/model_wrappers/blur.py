# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import cv2
import numpy as np
from typing import Any, Dict

from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue

from ote_sdk.entities.model import ModelConfiguration
from ote_sdk.utils.argument_checks import (
    OptionalParamTypeCheck,
    RequiredParamTypeCheck,
    check_input_param_type,
)
from ote_sdk.usecases.adapters.model_adapter import ModelAdapter
from ote_sdk.utils.segmentation_utils import create_hard_prediction_from_soft_prediction


class BlurSegmentation(SegmentationModel):
    __model__ = 'blur_segmentation'

    def __init__(self, model_adapter, configuration=None, preload=False):
        check_input_param_type(
            RequiredParamTypeCheck(model_adapter, "model_adapter", ModelAdapter),
            OptionalParamTypeCheck(configuration, "configuration", ModelConfiguration),
            RequiredParamTypeCheck(preload, "preload", bool),
        )
        super().__init__(model_adapter, configuration, preload)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'soft_threshold': NumericalValue(default_value=0.5, min=0.0, max=1.0),
            'blur_strength': NumericalValue(value_type=int, default_value=1, min=0, max=25)
        })

        return parameters

    def postprocess(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        check_input_param_type(
            RequiredParamTypeCheck(outputs, "outputs", Dict[str, np.ndarray]),
            RequiredParamTypeCheck(metadata, "metadata", Dict[str, Any]),
        )
        predictions = outputs[self.output_blob_name].squeeze()
        soft_prediction = np.transpose(predictions, axes=(1, 2, 0))

        hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=self.soft_threshold,
            blur_strength=self.blur_strength
        )
        hard_prediction = cv2.resize(hard_prediction, metadata['original_shape'][1::-1], 0, 0, interpolation=cv2.INTER_NEAREST)
        soft_prediction = cv2.resize(soft_prediction, metadata['original_shape'][1::-1], 0, 0, interpolation=cv2.INTER_NEAREST)
        metadata['soft_predictions'] = soft_prediction

        return hard_prediction
