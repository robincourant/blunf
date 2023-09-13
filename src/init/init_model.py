from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from omegaconf import DictConfig

from src.models.nerfacto import NerfactoModelConfig
from src.models.blueprint_nerfacto import (
    BlueprintNerfactoModelConfig,
    BlueprintSirenModelConfig,
)
from src.models.stereo import StereoModelConfig


def initialize_model(cfg: DictConfig) -> ModelConfig:
    cfg = dict(cfg)
    model_name = cfg.pop("name")
    # RGB only
    if model_name == "vanilla":
        model = VanillaModelConfig(**cfg)
    elif model_name == "nerfacto":
        model = NerfactoModelConfig(**cfg)
    # RGB+Semantic
    elif model_name == "rgb-sem_nerfacto":
        model = NerfactoModelConfig(**cfg)
    # BluNF only
    elif model_name == "blueprint_nerfacto":
        model = BlueprintNerfactoModelConfig(**cfg)
    elif model_name == "blueprint_siren":
        model = BlueprintSirenModelConfig(**cfg)
    # Stereo
    elif model_name == "stereo":
        model = StereoModelConfig(**cfg)

    else:
        raise NotImplementedError(f"Model: {model_name} is not implemented!")

    return model
