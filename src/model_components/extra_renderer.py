# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import torch
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal
from typing import Optional

import nerfacc
import torch
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples


class STDRender(nn.Module):
    """Calculate std along the ray."""

    @classmethod
    def forward(
        cls,
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate std along the ray."""
        std = torch.std(weights, dim=-2)
        return std


class DepthRenderer(nn.Module):
    """Calculate depth along ray.

    Depth Method:
        - median: Depth is set to the distance where the accumulated weight reaches 0.5.
        - expected: Expected depth along ray. Same procedure as rendering rgb, but with depth.

    Args:
        method: Depth calculation method.
    """

    def __init__(self, method: Literal["median", "expected"] = "median") -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        weights: TensorType[..., "num_samples", 1],
        ray_samples: RaySamples,
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType[..., 1]:
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of depth values.
        """
        if self.method == "median":
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError(
                    "Median depth calculation is not implemented for packed samples."
                )
            cumulative_weights = torch.cumsum(
                weights[..., 0], dim=-1
            )  # [..., num_samples]
            split = (
                torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5
            )  # [..., 1]
            median_index = torch.searchsorted(
                cumulative_weights, split, side="left"
            )  # [..., 1]
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
            median_depth = torch.gather(
                steps[..., 0], dim=-1, index=median_index
            )  # [..., 1]
            ray_depth = median_depth
        if self.method == "expected":
            eps = 1e-10
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                # Necessary for packed samples from volumetric ray sampler
                depth = nerfacc.accumulate_along_rays(
                    weights, ray_indices, steps, num_rays
                )
                accumulation = nerfacc.accumulate_along_rays(
                    weights, ray_indices, None, num_rays
                )
                depth = depth / (accumulation + eps)
            else:
                depth = torch.sum(weights * steps, dim=-2) / (
                    torch.sum(weights, -2) + eps
                )

            ray_depth = torch.clip(depth, steps.min(), steps.max())

        if True:  # FIXME:
            # try:
            factor_rd_2_d = ray_samples.metadata["factor_depth_coords"][:, 0]
            assert ray_depth.shape == factor_rd_2_d.shape
            depth = ray_depth * factor_rd_2_d
            # FIXME -> strange dim
            # import ipdb

            # ipdb.set_trace()
            # except:
            #     import ipdb

            #     ipdb.set_trace()
        return depth

        raise NotImplementedError(f"Method {self.method} not implemented")
