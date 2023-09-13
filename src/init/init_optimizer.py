from typing import Dict

from nerfstudio.engine.optimizers import AdamOptimizerConfig, OptimizerConfig
from omegaconf import DictConfig


def _initialize_optimizer(cfg: Dict) -> OptimizerConfig:
    name = cfg.pop("name")
    if name == "adam":
        optimizer = AdamOptimizerConfig(**cfg)
    else:
        raise NotImplementedError(f"Optimizer: {name} is not implemented!")

    return optimizer


def initialize_optim(cfg: DictConfig) -> Dict:
    optimizer = {}

    if hasattr(cfg, "field"):
        field_optimizer = _initialize_optimizer(dict(cfg.field))
        optimizer["fields"] = dict(optimizer=field_optimizer, scheduler=None)

    if hasattr(cfg, "proposal"):
        prop_optimizer = _initialize_optimizer(dict(cfg.proposal))
        optimizer["proposal_networks"] = dict(optimizer=prop_optimizer, scheduler=None)

    if hasattr(cfg, "blunf"):
        blunf_optimizer = _initialize_optimizer(dict(cfg.blunf))
        optimizer["blunf_fields"] = dict(optimizer=blunf_optimizer, scheduler=None)

    return optimizer
