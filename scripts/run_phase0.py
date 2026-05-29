"""
Phase 0 : exploration des features TextureSAM sur images MEB.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(cfg):
    # TODO 1 : charger les images MEB
    # - lister cfg.data.raw_dir pour les fichiers .tif
    # - normaliser si cfg.encoder.normalize
    # - retourner une liste de tenseurs (C, H, W)

    # TODO 2 : extraire les features TextureSAM
    # - instancier TextureSAM avec cfg.encoder.checkpoint
    # - extraire les features du stage cfg.encoder.stage
    # - agréger par masque/patch → tableau (N, D)

    # TODO 3 : t-SNE + UMAP
    # - réduire à cfg.evaluation.pca_components avec PCA
    # - projeter en 2-D avec t-SNE (cfg.evaluation.tsne)
    # - projeter en 2-D avec UMAP (cfg.evaluation.umap)
    # - sauvegarder les figures dans cfg.paths.outputs

    # TODO 4 : métriques statistiques
    # - silhouette score    si cfg.evaluation.silhouette
    # - Davies-Bouldin      si cfg.evaluation.davies_bouldin
    # - variance intra/inter si cfg.evaluation.variance
    # - Fisher criterion    si cfg.evaluation.fisher
    # - bootstrap (cfg.evaluation.bootstrap)

    # TODO 5 : sauvegarder les résultats
    # - écrire un CSV récapitulatif dans cfg.paths.outputs
    # - sauvegarder les embeddings numpy pour réutilisation

    pass


if __name__ == "__main__":
    run(cfg=None)
