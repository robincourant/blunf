"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter, CrossEntropyLoss
from typing_extensions import Literal
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    DephtLossType,
    depth_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    # DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from torchtyping import TensorType

from utils.colormap import get_colormap
from src.callbacks.metrics import RgbMetrics, SemanticMetrics
from src.fields.nerfacto_field import TCNNNerfactoField
from src.model_components.extra_renderer import STDRender, DepthRenderer

num_row_grid, num_col_grid, num_samples = None, None, None


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["background", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 1024
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 64,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 256,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed noramls."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    semantic_loss_mult: float = 0.001
    """Semantic loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    use_semantics: bool = False
    """Whether to predict sematic or not."""
    num_semantic_classes: int = 101
    """Number of semantic classes."""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    depth_loss_mult: float = 0.001
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = True
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: str = "sigma"
    """Depth loss type."""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        if self.config.use_semantics:
            self.colors = get_colormap(self.config.num_semantic_classes + 1)[1:]
        scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_semantics=self.config.use_semantics,
            num_semantic_classes=self.config.num_semantic_classes,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (
                len(self.config.proposal_net_args_list) == 1
            ), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(
                step,
                [0, self.config.proposal_warmup],
                [0, self.config.proposal_update_every],
            ),
            1,
            self.config.proposal_update_every,
        )
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane, far_plane=self.config.far_plane
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        if self.config.use_semantics:
            self.renderer_semantic = SemanticRenderer()
        self.renderer_std = STDRender()

        # losses
        self.depth_sigma = (
            torch.tensor([self.config.starting_depth_sigma])
            if self.config.should_decay_sigma
            else torch.tensor([self.config.depth_sigma])
        )
        self.step_count = 0
        self.semantic_factor = 0.01
        self.rgb_loss = MSELoss()
        if self.config.use_semantics:
            self.semantic_loss = CrossEntropyLoss()

        # Metrics
        self.rgb_metrics = RgbMetrics()
        if self.config.use_semantics:
            self.sem_metrics = SemanticMetrics(self.config.num_semantic_classes)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            # WIP decay
            # def decay_loss_weight_semantic(step):
            #     self.step_count += 1
            #     self.semantic_factor = 0.01 * (1 - ((1 - 1e-4) ** self.step_count))
            # callbacks.append(
            #     TrainingCallback(
            #         where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            #         update_every_num_iters=1,
            #         func=decay_loss_weight_semantic,
            #     )
            # )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_outputs = self.field(
            ray_samples, compute_normals=self.config.predict_normals
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        std = self.renderer_std(weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        if torch.isnan(accumulation).any():
            print("Nan Problem happen here.")

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "std": std}

        if self.config.predict_normals:
            outputs["normals"] = self.renderer_normals(
                normals=field_outputs[FieldHeadNames.NORMALS], weights=weights
            )
            outputs["pred_normals"] = self.renderer_normals(
                field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights
            )

        if self.config.use_semantics:
            semantic = self.renderer_semantic(
                semantics=field_outputs[FieldHeadNames.SEMANTICS],
                weights=weights,
            )
            outputs["semantic"] = semantic

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS],
                ray_bundle.directions,
            )
            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )
        # AD HOC implementation
        # self.step_count += 1
        # save_density_depth = {
        #     "density": field_outputs[FieldHeadNames.DENSITY].detach(),
        #     "weights": weights.detach(),
        #     "depth": depth.detach(),
        #     "steps": (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2,
        # }

        # outputs["xp_density"] = save_density_depth["density"]
        # outputs["xp_weights"] = save_density_depth["weights"]
        # outputs["xp_depth"] = save_density_depth["depth"]
        # outputs["xp_steps"] = save_density_depth["steps"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        target_dict = dict(
            rgb=batch["image"].to(self.device),
            semantic=batch["semantic"].to(self.device),
        )
        if "depth" in batch:
            target_dict["depth"] = batch["depth"].to(self.device)

        if "semantic" in outputs:
            outputs["semantic_pred"] = (
                torch.nn.functional.softmax(outputs["semantic"], dim=-1)
                .argmax(dim=-1)
                .to(target_dict["semantic"].dtype)
            )

        metrics_dict = self.rgb_metrics.get_batch_metrics(
            outputs, target_dict, self.config.num_proposal_iterations
        )
        if "semantic" in outputs:
            metrics_dict.update(
                self.sem_metrics.get_batch_metrics(outputs, target_dict)
            )

        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.training and self.config.depth_loss_mult:
            loss_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            termination_depth = batch["depth"].to(self.device)

            for i in range(len(outputs["weights_list"])):
                if self.config.depth_loss_type == "sigma":
                    depth_loss_type = DephtLossType.DS_NERF
                elif self.config.depth_loss_type == "urf":
                    depth_loss_type = DephtLossType.URF

                depth_loss_value = depth_loss(
                    weights=outputs["weights_list"][i],
                    ray_samples=outputs["ray_samples_list"][i],
                    termination_depth=termination_depth,
                    predicted_depth=outputs["depth"],
                    sigma=sigma,
                    directions_norm=outputs["directions_norm"],
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=depth_loss_type,
                )
                loss_dict["depth_loss"] += (
                    self.config.depth_loss_mult
                    * depth_loss_value
                    / len(outputs["weights_list"])
                )

        if self.training:
            loss_dict[
                "interlevel_loss"
            ] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = (
                self.config.distortion_loss_mult * metrics_dict["distortion"]
            )
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict[
                    "orientation_loss"
                ] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict[
                    "pred_normal_loss"
                ] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        if self.config.use_semantics:
            semantic_target = batch["semantic"].to(self.device).long()
            loss_dict["semantic_loss"] = self.semantic_factor * self.semantic_loss(
                outputs["semantic"], semantic_target
            )
            if torch.isnan(loss_dict["semantic_loss"]).any():
                print("Nan Problem happen here.")

        return loss_dict

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(
            self.config.sigma_decay_rate * self.depth_sigma,
            torch.tensor([self.config.depth_sigma]),
        )
        return self.depth_sigma

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # Gather all ground-truth
        target_dict = dict(
            rgb=batch["image"].to(self.device),
            semantic=batch["semantic"].to(self.device),
        )
        if "depth" in batch:
            target_dict["depth"] = batch["depth"].to(self.device)

        if "semantic" in outputs:
            # Get semantic class prediction
            outputs["semantic_pred"] = (
                torch.nn.functional.softmax(outputs["semantic"], dim=-1)
                .argmax(dim=-1)
                .to(target_dict["semantic"].dtype)
            )

        # Compute RGB and semantic metrics
        metrics_dict = self.rgb_metrics.get_batch_metrics(
            outputs, target_dict, self.config.num_proposal_iterations
        )

        # Draw RGB, depth and semantic
        colored_acc = colormaps.apply_colormap(outputs["accumulation"])

        near_plane = float(torch.min(target_dict["depth"]))
        far_plane = float(torch.max(target_dict["depth"]))

        colored_depth_target = colormaps.apply_depth_colormap(
            target_dict["depth"], near_plane=near_plane, far_plane=far_plane
        )
        colored_depth_pred = colormaps.apply_depth_colormap(
            outputs["depth"], near_plane=near_plane, far_plane=far_plane
        )
        colored_depth_diff = colormaps.apply_depth_colormap(
            torch.abs(outputs["depth"] - target_dict["depth"]),
        )

        images_dict = dict(
            rgb=torch.cat([target_dict["rgb"], outputs["rgb"]], dim=1),
            accumulation=torch.cat([colored_acc], dim=1),
            depth=torch.cat(
                [colored_depth_target, colored_depth_pred, colored_depth_diff], dim=1
            ),
            std=torch.cat([outputs["std"]], dim=1),
        )

        colored_depth_pred = colormaps.apply_depth_colormap(outputs["depth"])
        if "depth" in target_dict:
            colored_depth_target = colormaps.apply_depth_colormap(target_dict["depth"])
            images_dict["depth"] = torch.cat(
                [colored_depth_target, colored_depth_pred], dim=1
            )
        else:
            images_dict["depth"] = colored_depth_pred

        if "semantic" in outputs:
            colored_target = self.colors[
                target_dict["semantic"].to(int) % (len(self.colors) - 1)
            ]
            colored_pred = self.colors[
                outputs["semantic_pred"].to(int) % (len(self.colors) - 1)
            ]
            combined_semantic = torch.cat([colored_target, colored_pred], dim=1)
            images_dict["semantic"] = combined_semantic
        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
                near_plane=near_plane,
                far_plane=far_plane,
            )
            colored_depth_diff_i = colormaps.apply_depth_colormap(
                torch.abs(outputs[key] - target_dict["depth"]),
            )
            images_dict[key] = torch.cat(
                [colored_depth_target, prop_depth_i, colored_depth_diff_i], dim=1
            )

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_global_blueprint(
        self,
        global_coordinates: TensorType["num_row_grid", "num_col_grid", 2],
        min_height: float = -0.30,
        max_height: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Infer the global blueprint of a grid of coordinates."""
        self.field.half()
        num_points_per_chunk = self.config.eval_num_rays_per_chunk

        # Sample blueprint over the height, and remove the ceiling layer
        num_height_layers = self.config.num_nerf_samples_per_ray
        height_range = torch.linspace(min_height, max_height, num_height_layers + 1)[
            :-1
        ]

        global_coordinates = global_coordinates.view(-1, 2)
        blueprint_num_points = global_coordinates.shape[0]
        blueprint_outputs_lists = defaultdict(list)
        for i in range(0, blueprint_num_points, num_points_per_chunk):
            start_idx = i
            end_idx = i + num_points_per_chunk
            point_bundle = global_coordinates[start_idx:end_idx]

            density_out, semantics_out, coords_chunks = [], [], []
            for height in height_range.flip(dims=(0,)):
                num_samples = point_bundle.shape[0]
                # hpoint_bundle = torch.hstack(
                #     [
                #         point_bundle,
                #         height
                #         * torch.ones((num_samples, 1), device=point_bundle.device),
                #     ]
                # )
                hpoint_bundle = torch.hstack(
                    [
                        point_bundle[:, 0].unsqueeze(-1),
                        height
                        * torch.ones((num_samples, 1), device=point_bundle.device),
                        point_bundle[:, 1].unsqueeze(-1),
                    ]
                )
                density, semantics = self.field.infer_density(hpoint_bundle)

                coords_chunks.append(hpoint_bundle.cpu())
                density_out.append(density.cpu())
                semantics_out.append(semantics.cpu())
            out = {
                "density": torch.stack(density_out).permute(1, 0, 2),
                "hcoordinates": torch.stack(coords_chunks).permute(1, 0, 2),
                "semantics": torch.stack(semantics_out).permute(1, 0, 2),
            }

            for output_name, x in out.items():
                blueprint_outputs_lists[output_name].append(x)
            blueprint_outputs_lists["coordinates"].append(point_bundle)

        outputs = {}
        for output_name, outputs_list in blueprint_outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[f"blueprint_{output_name}"] = torch.cat(outputs_list)

        semantics = outputs["blueprint_semantics"]
        density = outputs["blueprint_density"]
        deltas = (height_range[1] - height_range[0]) * torch.ones(
            density.shape, device=density.device
        )
        weights = get_weights(deltas, density)
        nerf_blueprint = self.renderer_semantic(semantics=semantics, weights=weights)

        return {"blueprint_blueprint": nerf_blueprint}


def get_weights(
    deltas: TensorType[..., "num_samples", 1],
    densities: TensorType[..., "num_samples", 1],
) -> TensorType[..., "num_samples", 1]:
    """Return weights based on predicted densities

    Args:
        densities: Predicted densities for samples along ray

    Returns:
        Weights for each sample
    """

    delta_density = deltas * densities
    alphas = 1 - torch.exp(-delta_density)

    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [
            torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device),
            transmittance,
        ],
        dim=-2,
    )
    transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

    weights = alphas * transmittance  # [..., "num_samples"]
    weights = torch.nan_to_num(weights)

    return weights
