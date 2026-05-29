import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.seed)

    print()
    print(f"  experiment_name : {cfg.experiment_name}")
    print(f"  encoder         : {cfg.encoder.name} (stage {cfg.encoder.stage})")
    print(f"  n_clusters      : {cfg.clustering.n_clusters}")
    print(f"  seed            : {cfg.seed}")
    print()
    print("  Setup OK — prêt pour les expériences")
    print()


if __name__ == "__main__":
    main()
