"""
build_vlad_database.py — Ajoute des descripteurs VLAD dans database_meb.h5.

Ajoute 4 nouvelles clés dans features/ :
  block_0_vlad_k16  (N, 1536d)   block_0 × K=16
  block_0_vlad_k32  (N, 3072d)   block_0 × K=32
  block_1_vlad_k16  (N, 3072d)   block_1 × K=16
  block_1_vlad_k32  (N, 6144d)   block_1 × K=32

Phase 1 — Apprentissage codebooks (MiniBatchKMeans)
Phase 2 — Calcul VLAD par patch (intra-norm + L2 global)

Usage :
    python build_vlad_database.py [--db DB_PATH] [--img-dir IMG_DIR]
                                  [--annot ANNOT] [--checkpoint CKPT]
                                  [--seed SEED] [--n-sample N_SAMPLE]
"""

import argparse
import logging
import os
import pickle
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# ── SAM2 path ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SAM2 = _HERE / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
IMG_SIZE   = 1024
_MEAN      = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD       = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

BLOCKS      = ["block_0", "block_1"]
BLOCK_DIMS  = {"block_0": 96, "block_1": 192}
K_VALUES    = [16, 32]

# Images MEB : toutes 1280×768
ORIG_W = 1280
ORIG_H = 768

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "db":         _HERE / "data" / "feature_database" / "database_meb.h5",
    "img_dir":    _HERE / "PatchTagger_Output" / "full_images",
    "checkpoint": _HERE / "checkpoints" / "sam2.1_hiera_small_1.pt",
    "codebooks":  _HERE / "data" / "feature_database" / "vlad_codebooks.pkl",
}


# ─────────────────────────────────────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────────────────────────────────────

def build_image_encoder() -> ImageEncoder:
    trunk = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000
        ),
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model="nearest",
        fuse_type="sum",
        fpn_top_down_levels=[2, 3],
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def load_model(ckpt_path: Path, device: str) -> ImageEncoder:
    encoder = build_image_encoder()

    tmp_path = None
    if ckpt_path.is_file():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    elif ckpt_path.is_dir():
        archive_dir = ckpt_path / "archive" if (ckpt_path / "archive").is_dir() else ckpt_path
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        tmp_path = tmp.name
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for fp in sorted(archive_dir.rglob("*")):
                if fp.is_file():
                    info = zipfile.ZipInfo(str(fp.relative_to(archive_dir.parent)))
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    with open(fp, "rb") as fh:
                        zf.writestr(info, fh.read())
        sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
    else:
        log.warning("Checkpoint introuvable : %s — poids aléatoires", ckpt_path)
        return encoder.to(device).eval()

    if tmp_path:
        os.unlink(tmp_path)

    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log.warning("Checkpoint partiel (%d manquantes, %d inattendues)", len(missing), len(unexpected))
    else:
        log.info("Checkpoint chargé : %s", ckpt_path.name)

    return encoder.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────────────

def register_hooks(encoder: ImageEncoder) -> tuple[dict, list]:
    captured = {}
    handles  = []

    for block_name, block_idx in [("block_0", 0), ("block_1", 1)]:
        def _hook(m, inp, out, name=block_name):
            captured[name] = out.detach()
        handles.append(encoder.trunk.blocks[block_idx].register_forward_hook(_hook))

    return captured, handles


def remove_hooks(handles: list):
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Prétraitement
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image, device: str) -> torch.Tensor:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x   = torch.from_numpy(np.array(img)).float() / 255.0
    x   = x.permute(2, 0, 1)
    x   = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction vecteurs spatiaux
# ─────────────────────────────────────────────────────────────────────────────

def extract_spatial_vectors(
    feat_map: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """
    feat_map : (H_feat, W_feat, C)
    position : [x_min, y_min, x_max, y_max] en coordonnées originales
    Retourne : (N_spatial, C) — au moins 1 vecteur
    """
    H_feat, W_feat, C = feat_map.shape
    scale_x = W_feat / ORIG_W
    scale_y = H_feat / ORIG_H

    fx1 = max(0, int(position[0] * scale_x))
    fy1 = max(0, int(position[1] * scale_y))
    fx2 = min(W_feat, max(fx1 + 1, int(position[2] * scale_x)))
    fy2 = min(H_feat, max(fy1 + 1, int(position[3] * scale_y)))

    if fx2 - fx1 < 1:
        fx1 = min(fx1, W_feat - 1)
        fx2 = fx1 + 1
    if fy2 - fy1 < 1:
        fy1 = min(fy1, H_feat - 1)
        fy2 = fy1 + 1

    region = feat_map[fy1:fy2, fx1:fx2, :]   # (h, w, C)
    return region.reshape(-1, C)               # (N, C)


# ─────────────────────────────────────────────────────────────────────────────
# VLAD
# ─────────────────────────────────────────────────────────────────────────────

def compute_vlad(
    vectors: np.ndarray,
    centroids: np.ndarray,
    K: int,
    D: int,
) -> np.ndarray:
    """
    vectors   : (N, D) vecteurs spatiaux du patch
    centroids : (K, D) centroïdes du codebook
    Retourne  : (K*D,) vecteur VLAD intra-normalisé + L2 global
    """
    dists       = cdist(vectors, centroids, metric="euclidean")
    assignments = dists.argmin(axis=1)   # (N,)

    V = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 0:
            residus  = vectors[mask] - centroids[k]
            V[k]     = residus.sum(axis=0)
            norme    = np.linalg.norm(V[k])
            if norme > 1e-8:
                V[k] /= norme

    vlad_vec = V.flatten()
    norme    = np.linalg.norm(vlad_vec)
    if norme > 1e-8:
        vlad_vec /= norme

    return vlad_vec


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Apprentissage codebooks
# ─────────────────────────────────────────────────────────────────────────────

def learn_codebooks(
    encoder: ImageEncoder,
    captured: dict,
    images_uniq: np.ndarray,
    img_dir: Path,
    device: str,
    n_sample: int,
    seed: int,
) -> dict:
    rng            = np.random.default_rng(seed)
    vectors_sample = {b: [] for b in BLOCKS}

    for img_name in tqdm(images_uniq, desc="Phase 1 — Sampling"):
        img_path = img_dir / img_name.decode()
        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = preprocess(img, device)
        except Exception as e:
            log.warning("Erreur image %s : %s", img_name.decode(), e)
            continue

        captured.clear()
        with torch.no_grad():
            encoder(tensor)

        for block in BLOCKS:
            feat_map = captured[block][0]              # (H, W, C)
            H, W, C  = feat_map.shape
            flat     = feat_map.reshape(-1, C).cpu().numpy().astype(np.float32)

            idx = rng.choice(len(flat), size=min(n_sample, len(flat)), replace=False)
            vectors_sample[block].append(flat[idx])

    codebooks = {}
    for block in BLOCKS:
        all_vecs = np.vstack(vectors_sample[block])
        log.info("%s : %d vecteurs pour K-means", block, len(all_vecs))

        for K in K_VALUES:
            log.info("  K-means K=%d ...", K)
            km = MiniBatchKMeans(
                n_clusters   = K,
                random_state = seed,
                batch_size   = 4096,
                n_init       = 5,
            )
            km.fit(all_vecs)
            codebooks[(block, K)] = km
            log.info("  Inertie = %.2f", km.inertia_)

    return codebooks


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Calcul VLAD
# ─────────────────────────────────────────────────────────────────────────────

def compute_vlad_database(
    encoder: ImageEncoder,
    captured: dict,
    db_path: Path,
    img_dir: Path,
    image_names: np.ndarray,
    positions: np.ndarray,
    images_uniq: np.ndarray,
    codebooks: dict,
    device: str,
):
    N_PATCHES = len(image_names)

    # Pré-allouer les datasets dans HDF5
    with h5py.File(db_path, "a") as h5:
        for block in BLOCKS:
            D = BLOCK_DIMS[block]
            for K in K_VALUES:
                key = f"{block}_vlad_k{K}"
                dim = K * D
                if key not in h5["features"]:
                    h5["features"].create_dataset(
                        key,
                        shape  = (N_PATCHES, dim),
                        dtype  = np.float32,
                        chunks = (min(256, N_PATCHES), dim),
                    )
                    log.info("Dataset créé : features/%s  (%d, %d)", key, N_PATCHES, dim)
                else:
                    log.info("Dataset existant : features/%s — sera écrasé", key)

    # Grouper patches par image
    patches_by_image = defaultdict(list)
    for i in range(N_PATCHES):
        patches_by_image[image_names[i]].append(i)

    for img_name in tqdm(images_uniq, desc="Phase 2 — VLAD"):
        img_path = img_dir / img_name.decode()
        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = preprocess(img, device)
        except Exception as e:
            log.warning("Erreur image %s : %s", img_name.decode(), e)
            continue

        captured.clear()
        with torch.no_grad():
            encoder(tensor)

        feat_maps = {
            block: captured[block][0].cpu().numpy().astype(np.float32)
            for block in BLOCKS
        }

        patch_indices = patches_by_image[img_name]

        with h5py.File(db_path, "a") as h5:
            for patch_idx in patch_indices:
                pos = positions[patch_idx]   # [x_min, y_min, x_max, y_max]

                for block in BLOCKS:
                    vecs = extract_spatial_vectors(feat_maps[block], pos)

                    for K in K_VALUES:
                        centroids = codebooks[(block, K)].cluster_centers_
                        D         = BLOCK_DIMS[block]
                        vlad_vec  = compute_vlad(vecs, centroids, K, D)
                        key       = f"{block}_vlad_k{K}"
                        h5["features"][key][patch_idx] = vlad_vec


# ─────────────────────────────────────────────────────────────────────────────
# Vérification finale
# ─────────────────────────────────────────────────────────────────────────────

def verify(db_path: Path):
    log.info("")
    log.info("=== Vérification finale ===")
    with h5py.File(db_path, "r") as h5:
        for block in BLOCKS:
            for K in K_VALUES:
                key    = f"{block}_vlad_k{K}"
                shape  = h5["features"][key].shape
                sample = h5["features"][key][:100]
                norms  = np.linalg.norm(sample, axis=1)
                log.info(
                    "%-22s : shape=%-15s  norm mean=%.4f  std=%.6f",
                    key, str(shape), norms.mean(), norms.std(),
                )

    size_mb = os.path.getsize(db_path) / 1e6
    log.info("")
    log.info("Taille HDF5 : %.1f MB", size_mb)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db",         default=str(_DEFAULTS["db"]),
                   help="Fichier HDF5 existant")
    p.add_argument("--img-dir",    default=str(_DEFAULTS["img_dir"]),
                   help="Dossier des images .tif")
    p.add_argument("--checkpoint", default=str(_DEFAULTS["checkpoint"]),
                   help="Checkpoint SAM2 (.pt ou répertoire archive)")
    p.add_argument("--codebooks",  default=str(_DEFAULTS["codebooks"]),
                   help="Fichier .pkl pour sauvegarder/charger les codebooks")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n-sample",   type=int, default=5000,
                   help="Vecteurs à échantillonner par image pour K-means")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Charger les codebooks existants (--codebooks) sans refaire K-means")
    return p.parse_args()


def main():
    args     = parse_args()
    db_path  = Path(args.db)
    img_dir  = Path(args.img_dir)
    ckpt     = Path(args.checkpoint)
    cb_path  = Path(args.codebooks)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device : %s", device)

    # ── Charger metadata ───────────────────────────────────────────────────────
    log.info("Chargement metadata depuis %s", db_path.name)
    with h5py.File(db_path, "r") as h5:
        image_names  = h5["metadata/image_names"][:]
        positions    = h5["metadata/positions"][:]

    N_PATCHES    = len(image_names)
    images_uniq  = np.unique(image_names)
    log.info("%d patches, %d images uniques", N_PATCHES, len(images_uniq))

    # ── Modèle + hooks ─────────────────────────────────────────────────────────
    encoder          = load_model(ckpt, device)
    captured, handles = register_hooks(encoder)

    try:
        # ── Phase 1 ───────────────────────────────────────────────────────────
        if args.skip_phase1:
            log.info("Chargement codebooks depuis %s", cb_path)
            with open(cb_path, "rb") as f:
                cb_raw = pickle.load(f)
            # Reconstruire des objets MiniBatchKMeans légers avec juste les centres
            codebooks = {}
            for (block, K), centers in cb_raw.items():
                km = MiniBatchKMeans(n_clusters=K)
                km.cluster_centers_ = centers
                codebooks[(block, K)] = km
        else:
            codebooks = learn_codebooks(
                encoder, captured, images_uniq, img_dir,
                device, args.n_sample, args.seed,
            )
            # Sauvegarder
            cb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cb_path, "wb") as f:
                pickle.dump(
                    {(b, k): codebooks[(b, k)].cluster_centers_
                     for b in BLOCKS for k in K_VALUES},
                    f,
                )
            log.info("Codebooks sauvegardés → %s", cb_path)

        # ── Phase 2 ───────────────────────────────────────────────────────────
        compute_vlad_database(
            encoder, captured, db_path, img_dir,
            image_names, positions, images_uniq,
            codebooks, device,
        )

        # ── Vérification ──────────────────────────────────────────────────────
        verify(db_path)

    finally:
        remove_hooks(handles)


if __name__ == "__main__":
    main()
