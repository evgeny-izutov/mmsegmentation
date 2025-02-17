# Copyright (C) 2022 Intel Corporation
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

from __future__ import annotations
from typing import Union

import torch


class FeatureVectorHook:
    """While registered with the designated PyTorch module, this class caches feature vector during forward pass.

    Example::
        with FeatureVectorHook(model.module.backbone) as hook:
            with torch.no_grad():
                result = model(return_loss=False, **data)
            print(hook.records)
    Args:
        module (torch.nn.Module): The PyTorch module to be registered in forward pass
    """
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module
        self._handle = None
        self._records = []

    @property
    def records(self):
        return self._records

    @staticmethod
    def func(feature_map: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        """Generate the feature vector by average pooling feature maps.

        If the input is a list of feature maps from FPN, per-layer feature vector is first generated by averaging 
        feature maps in each FPN layer, then concatenate all the per-layer feature vector as the final result.

        Args:
            feature_map (Union[torch.Tensor, list[torch.Tensor]]): feature maps from backbone or list of feature maps 
                                                                    from FPN.

        Returns:
            torch.Tensor: feature vector(representation vector)
        """
        if isinstance(feature_map, list):
            # aggregate feature maps from Feature Pyramid Network
            feature_vector = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)) for f in feature_map]
            feature_vector = torch.cat(feature_vector, 1)
        else:
            feature_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
        return feature_vector

    def _recording_forward(
        self, _: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        feature_vector = self.func(output)
        feature_vector = feature_vector.detach().cpu().numpy()
        if len(feature_vector) > 1:
            for tensor in feature_vector:
                self._records.append(tensor)
        else:
            self._records.append(feature_vector)

    def __enter__(self) -> FeatureVectorHook:
        self._handle = self._module.register_forward_hook(self._recording_forward)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.remove()
