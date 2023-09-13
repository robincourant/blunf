import os

import hydra
from nerfstudio.configs import base_config  # noqa # Needed to avoid circular imports
from omegaconf import DictConfig

from src.init.init_trainer import initialize_trainer


@hydra.main(version_base="1.2", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    if "CUDA" in os.environ:
        cfg.compnode.device = "cuda:" + os.environ["CUDA"]
        cfg.compnode.local_rank = int(os.environ["CUDA"])

    cfg.vis = None  # Disable wandb
    trainer = initialize_trainer(cfg)
    trainer.setup()
    trainer.infer_eval()


if __name__ == "__main__":
    main()
