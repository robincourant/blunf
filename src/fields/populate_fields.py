from typing import List, Tuple

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.fields.density_fields import HashMLPDensityField
import numpy as np
import torch.nn as nn

from src.fields.nerfacto_field import TCNNNerfactoField
from src.fields.siren_field import Siren

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def _populate_blunfnerfacto(**kwargs) -> nn.Module:
    """Populate BluNF nerfacto module."""
    if kwargs["encoding_type"] == "identity":
        blunf = tcnn.Network(
            n_input_dims=2,
            n_output_dims=kwargs["num_semantic_classes"],
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": kwargs["hidden_dim"],
                "n_hidden_layers": kwargs["num_layers"] - 1,
            },
        )
        return blunf

    if kwargs["encoding_type"] == "frequency":
        encoding_config = {
            "otype": "Frequency",
            "n_frequencies": kwargs["n_frequencies"],
        }
    else:
        growth_factor = np.exp(
            (np.log(kwargs["max_res"]) - np.log(kwargs["base_res"]))
            / (kwargs["num_levels"] - 1)
        )
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": kwargs["num_levels"],
            "n_features_per_level": kwargs["features_per_level"],
            "log2_hashmap_size": kwargs["log2_hashmap_size"],
            "base_resolution": kwargs["base_res"],
            "per_level_scale": growth_factor,
        }

    blunf = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=kwargs["num_semantic_classes"],
        encoding_config=encoding_config,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": kwargs["hidden_dim"],
            "n_hidden_layers": kwargs["num_layers"] - 1,
        },
    )
    return blunf


def _populate_blunfsiren(**kwargs) -> nn.Module:
    """Populate BluNF SIREN module."""
    blunf = Siren(
        n_input_dims=2,
        n_output_dims=kwargs["num_semantic_classes"],
        hidden_features=kwargs["hidden_dim"],
        hidden_layers=kwargs["num_layers"] - 1,
    )
    return blunf


def _populate_nerfacto(**kwargs) -> nn.Module:
    nerf = TCNNNerfactoField(
        kwargs["aabb"],
        num_levels=kwargs["num_levels"],
        max_res=kwargs["max_res"],
        log2_hashmap_size=kwargs["log2_hashmap_size"],
        spatial_distortion=kwargs["scene_contraction"],
        num_images=kwargs["num_train_data"],
        use_pred_normals=kwargs["predict_normals"],
        use_average_appearance_embedding=kwargs["use_average_appearance_embedding"],
        use_semantics=True,
        num_semantic_classes=kwargs["num_semantic_classes"],
    )
    return nerf


def _populate_proposal_networks(**kwargs) -> nn.ModuleList:
    density_fns = []
    num_prop_nets = kwargs["num_proposal_iterations"]
    proposal_networks = nn.ModuleList()

    if kwargs["use_same_proposal_network"]:
        assert (
            len(kwargs["proposal_net_args_list"]) == 1
        ), "Only one proposal network is allowed."
        prop_net_args = kwargs["proposal_net_args_list"][0]
        network = HashMLPDensityField(
            kwargs["aabb"],
            spatial_distortion=kwargs["scene_contraction"],
            **prop_net_args,
        )
        proposal_networks.append(network)
        density_fns.extend([network.density_fn for _ in range(num_prop_nets)])

    else:
        for i in range(num_prop_nets):
            prop_net_args = kwargs["proposal_net_args_list"][
                min(i, len(kwargs["proposal_net_args_list"]) - 1)
            ]
            network = HashMLPDensityField(
                kwargs["aabb"],
                spatial_distortion=kwargs["scene_contraction"],
                **prop_net_args,
            )
            proposal_networks.append(network)
        density_fns.extend([network.density_fn for network in proposal_networks])
    return proposal_networks, density_fns


def populate_blueprint(config: ModelConfig) -> nn.Module:
    """Populate BluNF nerfacto or SIREN module for blueprint (BluNF only)."""
    if config.field_name == "blueprint-nerfacto":
        kwargs = dict(
            base_res=config.base_res,
            max_res=config.max_res,
            num_levels=config.num_levels,
            num_semantic_classes=config.num_semantic_classes,
            features_per_level=config.features_per_level,
            log2_hashmap_size=config.log2_hashmap_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            encoding_type=config.encoding_type,
            n_frequencies=config.n_frequencies,
        )
        return _populate_blunfnerfacto(**kwargs)

    if config.field_name == "blueprint-siren":
        kwargs = dict(
            num_semantic_classes=config.num_semantic_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        return _populate_blunfsiren(**kwargs)

    else:
        raise NotImplementedError(f"Module: {config.blunf_name} is not implemented!")


def populate_blunf(config: ModelConfig) -> nn.Module:
    """Populate BluNF nerfacto or SIREN module for NeRF+BluNF."""
    if config.blunf_name == "blunf-nerfacto":
        kwargs = dict(
            max_res=config.blunf_max_res,
            base_res=config.blunf_base_res,
            num_levels=config.blunf_num_levels,
            num_semantic_classes=config.blunf_num_semantic_classes,
            features_per_level=config.blunf_features_per_level,
            log2_hashmap_size=config.blunf_log2_hashmap_size,
            hidden_dim=config.blunf_hidden_dim,
            num_layers=config.blunf_num_layers,
            encoding_type=config.encoding_type,
            n_frequencies=config.n_frequencies,
        )
        return _populate_blunfnerfacto(**kwargs)

    if config.blunf_name == "blunf-siren":
        kwargs = dict(
            num_semantic_classes=config.blunf_num_semantic_classes,
            hidden_dim=config.blunf_hidden_dim,
            num_layers=config.blunf_num_layers,
        )
        return _populate_blunfsiren(**kwargs)

    else:
        raise NotImplementedError(f"Module: {config.blunf_name} is not implemented!")


def populate_nerf(
    scene_box: SceneBox,
    scene_contraction: SceneContraction,
    num_train_data: int,
    config: ModelConfig,
) -> nn.Module:
    kwargs = dict(
        aabb=scene_box.aabb,
        num_levels=config.nerf_num_levels,
        max_res=config.nerf_max_res,
        log2_hashmap_size=config.nerf_log2_hashmap_size,
        scene_contraction=scene_contraction,
        num_train_data=num_train_data,
        predict_normals=config.nerf_predict_normals,
        use_average_appearance_embedding=config.nerf_use_average_appearance_embedding,
        num_semantic_classes=config.nerf_num_semantic_classes,
    )
    return _populate_nerfacto(**kwargs)


def populate_proposals(
    scene_box: SceneBox, scene_contraction: SceneContraction, config: ModelConfig
) -> Tuple[nn.ModuleList, List[nn.Module]]:
    kwargs = dict(
        aabb=scene_box.aabb,
        scene_contraction=scene_contraction,
        num_proposal_iterations=config.num_proposal_iterations,
        use_same_proposal_network=config.use_same_proposal_network,
        proposal_net_args_list=config.proposal_net_args_list,
    )
    return _populate_proposal_networks(**kwargs)
