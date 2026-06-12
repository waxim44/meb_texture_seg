"""
test_noise_robustness.py — Mesure la stabilité des features block_0 au bruit gaussien.

Compare les features clean (depuis HDF5) aux features noisy (recalculées par
forward pass) via similarité cosine, pour N_PER_CAT patches par catégorie.

Usage :
    python test_noise_robustness.py [--db DB_PATH] [--img-dir IMG_DIR]
                                    [--checkpoint CHECKPOINT] [--config CONFIG]
                                    [--n-per-cat N] [--output OUTPUT_DIR]
                                    [--seed SEED]
"""

import argparse
import json
import sys
import zipfile
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── SAM2 path ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SAM2 = _HERE / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Constantes ────────────────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

IMG_SIZE     = 1024
CATS_EXCLUDE = {2, 8, 10, 11, 12, 13}
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
N_PER_CAT    = 50
SEED         = 42

_DEFAULTS = {
    "db":         _HERE / "data" / "feature_database" / "database_meb.h5",
    "img_dir":    _HERE / "PatchTagger_Output" / "full_images",
    "checkpoint": _HERE / "checkpoints" / "sam2.1_hiera_small_1.pt",
    "config":     _HERE / "PatchTagger_Output" / "config" / "config.json",
    "output":     _HERE / "outputs" / "noise_robustness",
}


# ─────────────────────────────────────────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────────────────────────────────────────

def _build_encoder() -> ImageEncoder:
    trunk = Hiera(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000
        ),
        d_model=256, backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model="nearest", fuse_type="sum",
        fpn_top_down_levels=[2, 3],
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def _load_model(ckpt_path: Path, device: str) -> ImageEncoder:
    encoder = _build_encoder()
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint introuvable : {ckpt_path} — poids aléatoires")
        return encoder.to(device).eval()

    tmp_path = None
    if ckpt_path.is_file():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
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
        os.unlink(tmp_path)

    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    encoder.load_state_dict(sd, strict=False)
    print(f"[INFO] Checkpoint chargé : {ckpt_path.name}")
    return encoder.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Prétraitement
# ─────────────────────────────────────────────────────────────────────────────

def _load_resize(img_path: Path) -> tuple[np.ndarray, int, int]:
    """Charge et redimensionne l'image. Retourne (np [0,1] HxWx3, orig_H, orig_W)."""
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0, orig_h, orig_w


def _to_tensor(img_np: np.ndarray, device: str) -> torch.Tensor:
    """HxWx3 float32 [0,1] → (1, 3, H, W) normalisé ImageNet."""
    x = torch.from_numpy(img_np).permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction région
# ─────────────────────────────────────────────────────────────────────────────

def _extract_region(feat_map: torch.Tensor, pos: np.ndarray,
                    orig_H: int, orig_W: int) -> np.ndarray:
    """feat_map : (H_f, W_f, C).  pos : [x_min, y_min, x_max, y_max].
    Retourne le vecteur moyen L2-normalisé (C,)."""
    H_f, W_f, _ = feat_map.shape
    sx, sy = W_f / orig_W, H_f / orig_H

    fx1 = max(0, int(pos[0] * sx))
    fy1 = max(0, int(pos[1] * sy))
    fx2 = min(W_f, max(fx1 + 1, int(pos[2] * sx)))
    fy2 = min(H_f, max(fy1 + 1, int(pos[3] * sy)))

    region = feat_map[fy1:fy2, fx1:fx2, :]
    vec = region.mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run(db_path: Path, img_dir: Path, ckpt_path: Path,
        cfg_path: Path, out_dir: Path, n_per_cat: int, seed: int):

    out_dir.mkdir(parents=True, exist_ok=True)
    _noise_rng    = np.random.default_rng(seed)
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")

    # ── Config catégories ─────────────────────────────────────────────────────
    with open(cfg_path) as f:
        _noise_cfg = json.load(f)
    CATEGORIES = {
        int(k): v["name"] for k, v in _noise_cfg["available_categories"].items()
    }

    # ── Chargement metadata + features clean ──────────────────────────────────
    with h5py.File(db_path, "r") as h5:
        _noise_IMAGE_NAMES  = h5["metadata/image_names"][:]
        _noise_POSITIONS    = h5["metadata/positions"][:]
        _noise_CATEGORY_IDS = h5["metadata/category_ids"][:]
        _noise_X_clean_all  = h5["features"]["block_0"][:]  # (N, 96)

    _noise_CATS_VALID = sorted([
        int(c) for c in np.unique(_noise_CATEGORY_IDS)
        if int(c) not in CATS_EXCLUDE
        and int((_noise_CATEGORY_IDS == c).sum()) >= 30
    ])
    print(f"[INFO] Catégories valides ({len(_noise_CATS_VALID)}) :")
    for c in _noise_CATS_VALID:
        n = int((_noise_CATEGORY_IDS == c).sum())
        print(f"  {c:2d}  {CATEGORIES[c]:<28}  N={n}")

    # ── Échantillonnage N_PER_CAT patches par catégorie ───────────────────────
    _noise_sample_idx = []
    for c in _noise_CATS_VALID:
        idx_c = np.where(_noise_CATEGORY_IDS == c)[0]
        n_sel = min(n_per_cat, len(idx_c))
        _noise_sample_idx.extend(
            _noise_rng.choice(idx_c, size=n_sel, replace=False).tolist()
        )
    _noise_sample_idx = np.array(_noise_sample_idx)
    print(f"\n[INFO] {len(_noise_sample_idx)} patches échantillonnés "
          f"({len(_noise_CATS_VALID)} catégories)")

    # ── Modèle + hook block_0 ─────────────────────────────────────────────────
    _noise_encoder = _load_model(ckpt_path, device)
    _noise_cap     = {}
    _noise_hook    = _noise_encoder.trunk.blocks[0].register_forward_hook(
        lambda m, i, o: _noise_cap.update({"block_0": o.detach()})
    )

    # ── Grouper par image (1 forward pass par image par σ) ────────────────────
    _noise_patches_by_img = defaultdict(list)
    for idx in _noise_sample_idx:
        _noise_patches_by_img[_noise_IMAGE_NAMES[idx]].append(int(idx))

    # ── Calcul ────────────────────────────────────────────────────────────────
    _noise_results         = {s: [] for s in NOISE_LEVELS}
    _noise_results_per_cat = {s: {c: [] for c in _noise_CATS_VALID}
                               for s in NOISE_LEVELS}

    for _noise_img_name in tqdm(_noise_patches_by_img, desc="Images"):
        _noise_img_path = img_dir / _noise_img_name.decode()
        if not _noise_img_path.exists():
            print(f"[WARN] Image introuvable : {_noise_img_path.name}")
            continue

        _noise_img_np, _noise_orig_H, _noise_orig_W = _load_resize(_noise_img_path)

        for _noise_sigma in NOISE_LEVELS:
            if _noise_sigma == 0.0:
                _noise_img_proc = _noise_img_np
            else:
                _noise_noise    = _noise_rng.normal(0, _noise_sigma,
                                                    _noise_img_np.shape).astype(np.float32)
                _noise_img_proc = np.clip(_noise_img_np + _noise_noise, 0.0, 1.0)

            _noise_tensor = _to_tensor(_noise_img_proc, device)
            _noise_cap.clear()
            with torch.no_grad():
                _noise_encoder(_noise_tensor)

            _noise_feat_map = _noise_cap["block_0"][0]  # (256, 256, 96)

            for _noise_idx in _noise_patches_by_img[_noise_img_name]:
                _noise_pos = _noise_POSITIONS[_noise_idx]
                _noise_cat = int(_noise_CATEGORY_IDS[_noise_idx])

                _noise_f_noisy = _extract_region(
                    _noise_feat_map, _noise_pos,
                    _noise_orig_H, _noise_orig_W,
                )
                _noise_f_clean = _noise_X_clean_all[_noise_idx].copy()
                _noise_n = np.linalg.norm(_noise_f_clean)
                if _noise_n > 1e-8:
                    _noise_f_clean = _noise_f_clean / _noise_n

                _noise_sim = float(_noise_f_clean @ _noise_f_noisy)
                _noise_results[_noise_sigma].append(_noise_sim)
                _noise_results_per_cat[_noise_sigma][_noise_cat].append(_noise_sim)

    _noise_hook.remove()

    # ── Console ───────────────────────────────────────────────────────────────
    print(f'\n{"σ bruit":>8} │ {"Sim moyenne":>12} │ {"Std":>8}')
    print("─" * 35)
    for _noise_s in NOISE_LEVELS:
        _noise_sims = _noise_results[_noise_s]
        print(f"{_noise_s:>8.2f} │ {np.mean(_noise_sims):>12.4f} │ {np.std(_noise_sims):>8.4f}")

    # ── Figure 1×2 ────────────────────────────────────────────────────────────
    _noise_fig, (_noise_ax1, _noise_ax2) = plt.subplots(1, 2, figsize=(15, 5))

    _noise_means = [np.mean(_noise_results[s]) for s in NOISE_LEVELS]
    _noise_stds  = [np.std(_noise_results[s])  for s in NOISE_LEVELS]

    _noise_ax1.plot(NOISE_LEVELS, _noise_means, "o-", lw=2.5, ms=8, color="#1B4F72")
    _noise_ax1.fill_between(
        NOISE_LEVELS,
        np.array(_noise_means) - np.array(_noise_stds),
        np.array(_noise_means) + np.array(_noise_stds),
        alpha=0.2, color="#1B4F72",
    )
    _noise_ax1.axhline(1.0, color="green", ls=":", alpha=0.6, label="stabilité parfaite")
    _noise_ax1.set_xlabel("σ du bruit gaussien", fontsize=11)
    _noise_ax1.set_ylabel("Cosine(clean, noisy)", fontsize=11)
    _noise_ax1.set_title("Stabilité globale block_0 au bruit", fontsize=12)
    _noise_ax1.set_ylim([0, 1.05])
    _noise_ax1.legend(fontsize=9)
    _noise_ax1.grid(True, alpha=0.3)

    for _noise_c in _noise_CATS_VALID:
        _noise_cat_sims = [
            np.mean(_noise_results_per_cat[s][_noise_c]) for s in NOISE_LEVELS
        ]
        _noise_hex = _noise_cfg["available_categories"][str(_noise_c)]["color"]
        _noise_ax2.plot(NOISE_LEVELS, _noise_cat_sims, "o-", lw=1.5, ms=5,
                        color=_noise_hex, label=CATEGORIES[_noise_c])

    _noise_ax2.set_xlabel("σ du bruit gaussien", fontsize=11)
    _noise_ax2.set_ylabel("Cosine(clean, noisy)", fontsize=11)
    _noise_ax2.set_title("Stabilité par catégorie", fontsize=12)
    _noise_ax2.set_ylim([0, 1.05])
    _noise_ax2.legend(fontsize=8)
    _noise_ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Robustesse au bruit — features block_0\n"
        f"{len(_noise_sample_idx)} patches · {len(_noise_CATS_VALID)} catégories",
        fontsize=12,
    )
    plt.tight_layout()
    _noise_fig_path = out_dir / "noise_robustness.png"
    plt.savefig(_noise_fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n→ Figure sauvegardée : {_noise_fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db",         default=str(_DEFAULTS["db"]),
                   help="Chemin vers database_meb.h5")
    p.add_argument("--img-dir",    default=str(_DEFAULTS["img_dir"]),
                   help="Dossier des images .tif")
    p.add_argument("--checkpoint", default=str(_DEFAULTS["checkpoint"]),
                   help="Checkpoint SAM2 (.pt ou répertoire archive)")
    p.add_argument("--config",     default=str(_DEFAULTS["config"]),
                   help="config.json (catégories)")
    p.add_argument("--output",     default=str(_DEFAULTS["output"]),
                   help="Dossier de sortie pour la figure")
    p.add_argument("--n-per-cat",  type=int, default=N_PER_CAT,
                   help=f"Patches par catégorie (défaut : {N_PER_CAT})")
    p.add_argument("--seed",       type=int, default=SEED,
                   help=f"Seed aléatoire (défaut : {SEED})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        db_path   = Path(args.db),
        img_dir   = Path(args.img_dir),
        ckpt_path = Path(args.checkpoint),
        cfg_path  = Path(args.config),
        out_dir   = Path(args.output),
        n_per_cat = args.n_per_cat,
        seed      = args.seed,
    )
