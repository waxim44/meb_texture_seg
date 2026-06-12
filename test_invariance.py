"""
test_invariance.py — Invariance des features block_0 à 3 transformations
photométriques : luminosité, contraste, gamma.

Mesure : cosine(features_clean, features_transformed).

Usage :
    python test_invariance.py [--db DB_PATH] [--img-dir IMG_DIR]
                               [--checkpoint CHECKPOINT] [--config CONFIG]
                               [--n-per-cat N] [--output OUTPUT_DIR]
                               [--seed SEED]
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
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
N_PER_CAT    = 50
SEED         = 42

BRIGHTNESS = [-0.3, -0.15, 0.0,  0.15, 0.3]
CONTRAST   = [0.5,   0.75, 1.0,  1.25, 1.5]
GAMMA      = [0.5,   0.75, 1.0,  1.5,  2.0]

_DEFAULTS = {
    "db":         _HERE / "data" / "feature_database" / "database_meb.h5",
    "img_dir":    _HERE / "PatchTagger_Output" / "full_images",
    "checkpoint": _HERE / "checkpoints" / "sam2.1_hiera_small_1.pt",
    "config":     _HERE / "PatchTagger_Output" / "config" / "config.json",
    "output":     _HERE / "outputs" / "invariance",
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

    if ckpt_path.is_file():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        archive_dir = ckpt_path / "archive" if (ckpt_path / "archive").is_dir() else ckpt_path
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_STORED) as zf:
            for fp in sorted(archive_dir.rglob("*")):
                if fp.is_file():
                    info = zipfile.ZipInfo(str(fp.relative_to(archive_dir.parent)))
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    with open(fp, "rb") as fh:
                        zf.writestr(info, fh.read())
        sd = torch.load(tmp.name, map_location="cpu", weights_only=False)
        os.unlink(tmp.name)

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
    """Retourne (np float32 [0,1] HxWx3, orig_H, orig_W)."""
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
# Transformations photométriques
# ─────────────────────────────────────────────────────────────────────────────

def _apply_brightness(img: np.ndarray, beta: float) -> np.ndarray:
    return np.clip(img + beta, 0.0, 1.0)

def _apply_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((img - 0.5) * alpha + 0.5, 0.0, 1.0)

def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(img ** gamma, 0.0, 1.0)

# (fonction, niveaux, label axe, valeur neutre)
TRANSFORMS = {
    "brightness": (_apply_brightness, BRIGHTNESS, "β (ajout)",    0.0),
    "contrast":   (_apply_contrast,   CONTRAST,   "α (facteur)",  1.0),
    "gamma":      (_apply_gamma,      GAMMA,      "γ (exposant)", 1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Extraction région
# ─────────────────────────────────────────────────────────────────────────────

def _extract_region(feat_map: torch.Tensor, pos: np.ndarray,
                    orig_H: int, orig_W: int) -> np.ndarray:
    """feat_map : (H_f, W_f, C).  pos : [x_min, y_min, x_max, y_max].
    Retourne vecteur moyen L2-normalisé (C,)."""
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
    _inv_rng  = np.random.default_rng(seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device : {device}")

    # ── Config catégories ─────────────────────────────────────────────────────
    with open(cfg_path) as f:
        _inv_cfg = json.load(f)
    CATEGORIES = {
        int(k): v["name"] for k, v in _inv_cfg["available_categories"].items()
    }

    # ── Metadata ──────────────────────────────────────────────────────────────
    with h5py.File(db_path, "r") as h5:
        _inv_IMAGE_NAMES  = h5["metadata/image_names"][:]
        _inv_POSITIONS    = h5["metadata/positions"][:]
        _inv_CATEGORY_IDS = h5["metadata/category_ids"][:]

    _inv_CATS_VALID = sorted([
        int(c) for c in np.unique(_inv_CATEGORY_IDS)
        if int(c) not in CATS_EXCLUDE
        and int((_inv_CATEGORY_IDS == c).sum()) >= 30
    ])
    print(f"[INFO] Catégories valides ({len(_inv_CATS_VALID)}) :")
    for c in _inv_CATS_VALID:
        n = int((_inv_CATEGORY_IDS == c).sum())
        print(f"  {c:2d}  {CATEGORIES[c]:<28}  N={n}")

    # ── Échantillonnage ────────────────────────────────────────────────────────
    _inv_sample_idx = []
    for c in _inv_CATS_VALID:
        idx_c = np.where(_inv_CATEGORY_IDS == c)[0]
        n_sel = min(n_per_cat, len(idx_c))
        _inv_sample_idx.extend(
            _inv_rng.choice(idx_c, size=n_sel, replace=False).tolist()
        )
    _inv_sample_idx = np.array(_inv_sample_idx)
    print(f"\n[INFO] {len(_inv_sample_idx)} patches échantillonnés")

    # ── Features clean pré-chargées et normalisées ────────────────────────────
    _inv_sorted_idx  = np.sort(_inv_sample_idx)
    with h5py.File(db_path, "r") as h5:
        _inv_X_clean = h5["features"]["block_0"][_inv_sorted_idx]
    for i in range(len(_inv_X_clean)):
        n = np.linalg.norm(_inv_X_clean[i])
        if n > 1e-8:
            _inv_X_clean[i] /= n
    _inv_clean_map = {int(idx): _inv_X_clean[i]
                      for i, idx in enumerate(_inv_sorted_idx)}

    # ── Modèle + hook block_0 ─────────────────────────────────────────────────
    _inv_encoder = _load_model(ckpt_path, device)
    _inv_cap     = {}
    _inv_hook    = _inv_encoder.trunk.blocks[0].register_forward_hook(
        lambda m, i, o: _inv_cap.update({"block_0": o.detach()})
    )

    # ── Grouper par image ─────────────────────────────────────────────────────
    _inv_patches_by_img = defaultdict(list)
    for idx in _inv_sample_idx:
        _inv_patches_by_img[_inv_IMAGE_NAMES[idx]].append(int(idx))

    # ── Structures de résultats ───────────────────────────────────────────────
    _inv_results = {
        t: {lvl: [] for lvl in TRANSFORMS[t][1]}
        for t in TRANSFORMS
    }
    _inv_results_cat = {
        t: {lvl: {c: [] for c in _inv_CATS_VALID} for lvl in TRANSFORMS[t][1]}
        for t in TRANSFORMS
    }

    # ── Calcul ────────────────────────────────────────────────────────────────
    for _inv_img_name in tqdm(_inv_patches_by_img, desc="Images"):
        _inv_img_path = img_dir / _inv_img_name.decode()
        if not _inv_img_path.exists():
            print(f"[WARN] Image introuvable : {_inv_img_path.name}")
            continue

        _inv_img_np, _inv_orig_H, _inv_orig_W = _load_resize(_inv_img_path)

        for _inv_tname, (_inv_tfunc, _inv_levels, _, _) in TRANSFORMS.items():
            for _inv_lvl in _inv_levels:
                _inv_img_t  = _inv_tfunc(_inv_img_np, _inv_lvl)
                _inv_tensor = _to_tensor(_inv_img_t, device)

                _inv_cap.clear()
                with torch.no_grad():
                    _inv_encoder(_inv_tensor)

                _inv_feat_map = _inv_cap["block_0"][0]  # (256, 256, 96)

                for _inv_idx in _inv_patches_by_img[_inv_img_name]:
                    _inv_pos = _inv_POSITIONS[_inv_idx]
                    _inv_cat = int(_inv_CATEGORY_IDS[_inv_idx])

                    _inv_f_t = _extract_region(
                        _inv_feat_map, _inv_pos,
                        _inv_orig_H, _inv_orig_W,
                    )
                    _inv_sim = float(_inv_clean_map[_inv_idx] @ _inv_f_t)
                    _inv_results[_inv_tname][_inv_lvl].append(_inv_sim)
                    _inv_results_cat[_inv_tname][_inv_lvl][_inv_cat].append(_inv_sim)

    _inv_hook.remove()

    # ── Console ───────────────────────────────────────────────────────────────
    for _inv_tname, (_, _inv_levels, _inv_xlabel, _) in TRANSFORMS.items():
        print(f"\n=== {_inv_tname.upper()} ===")
        print(f"{_inv_xlabel:>15} │ {'Sim moyenne':>12} │ {'Std':>8}")
        print("─" * 42)
        for _inv_lvl in _inv_levels:
            _inv_sims = _inv_results[_inv_tname][_inv_lvl]
            print(f"{_inv_lvl:>15} │ {np.mean(_inv_sims):>12.4f} │ "
                  f"{np.std(_inv_sims):>8.4f}")

    # ── Figure 2×3 ────────────────────────────────────────────────────────────
    _inv_fig, _inv_axes = plt.subplots(2, 3, figsize=(19, 10))

    for _inv_col, _inv_tname in enumerate(["brightness", "contrast", "gamma"]):
        _, _inv_levels, _inv_xlabel, _inv_neutral = TRANSFORMS[_inv_tname]

        # Ligne 0 : courbe globale
        _inv_ax = _inv_axes[0, _inv_col]
        _inv_means = [np.mean(_inv_results[_inv_tname][lvl]) for lvl in _inv_levels]
        _inv_stds  = [np.std(_inv_results[_inv_tname][lvl])  for lvl in _inv_levels]

        _inv_ax.plot(_inv_levels, _inv_means, "o-", lw=2.5, ms=8, color="#1B4F72")
        _inv_ax.fill_between(
            _inv_levels,
            np.array(_inv_means) - np.array(_inv_stds),
            np.array(_inv_means) + np.array(_inv_stds),
            alpha=0.2, color="#1B4F72",
        )
        _inv_ax.axhline(1.0, color="green", ls=":", alpha=0.6, label="invariant parfait")
        _inv_ax.axvline(_inv_neutral, color="gray", ls="--", alpha=0.5, label="neutre")
        for lvl, m in zip(_inv_levels, _inv_means):
            _inv_ax.text(lvl, m + 0.018, f"{m:.3f}",
                         ha="center", va="bottom", fontsize=7.5, color="#1B4F72")
        _inv_ax.set_xlabel(_inv_xlabel, fontsize=10)
        _inv_ax.set_ylabel("Cosine(clean, transformed)", fontsize=10)
        _inv_ax.set_title(f"{_inv_tname.capitalize()} — global", fontsize=11)
        _inv_ax.set_ylim([0, 1.12])
        _inv_ax.legend(fontsize=8)
        _inv_ax.grid(True, alpha=0.3)

        # Ligne 1 : par catégorie
        _inv_ax2 = _inv_axes[1, _inv_col]
        for _inv_c in _inv_CATS_VALID:
            _inv_cat_sims = [
                np.mean(_inv_results_cat[_inv_tname][lvl][_inv_c])
                for lvl in _inv_levels
            ]
            _inv_hex = _inv_cfg["available_categories"][str(_inv_c)]["color"]
            _inv_ax2.plot(_inv_levels, _inv_cat_sims, "o-", lw=1.5, ms=4,
                          color=_inv_hex, label=CATEGORIES[_inv_c])
        _inv_ax2.axhline(1.0, color="green", ls=":", alpha=0.5)
        _inv_ax2.axvline(_inv_neutral, color="gray", ls="--", alpha=0.5)
        _inv_ax2.set_xlabel(_inv_xlabel, fontsize=10)
        _inv_ax2.set_ylabel("Cosine(clean, transformed)", fontsize=10)
        _inv_ax2.set_title(f"{_inv_tname.capitalize()} — par catégorie", fontsize=11)
        _inv_ax2.set_ylim([0, 1.12])
        if _inv_col == 0:
            _inv_ax2.legend(fontsize=7, loc="lower left")
        _inv_ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Invariance photométrique — features block_0\n"
        f"{len(_inv_sample_idx)} patches · cosine(clean, transformed) · "
        "proche de 1 = invariant",
        fontsize=12,
    )
    plt.tight_layout()
    _inv_fig_path = out_dir / "invariance_photometric.png"
    plt.savefig(_inv_fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n→ Figure sauvegardée : {_inv_fig_path}")

    # ── Interprétation auto ───────────────────────────────────────────────────
    print("\n=== INVARIANCE (point le plus extrême) ===")
    for _inv_tname, (_, _inv_levels, _, _) in TRANSFORMS.items():
        _inv_sim_ext = min(
            np.mean(_inv_results[_inv_tname][_inv_levels[0]]),
            np.mean(_inv_results[_inv_tname][_inv_levels[-1]]),
        )
        if _inv_sim_ext > 0.95:
            _inv_verdict = "très invariant ✅"
        elif _inv_sim_ext > 0.85:
            _inv_verdict = "assez invariant ⚠️"
        else:
            _inv_verdict = "sensible ❌"
        print(f"  {_inv_tname:<12} : sim={_inv_sim_ext:.3f} → {_inv_verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db",         default=str(_DEFAULTS["db"]))
    p.add_argument("--img-dir",    default=str(_DEFAULTS["img_dir"]))
    p.add_argument("--checkpoint", default=str(_DEFAULTS["checkpoint"]))
    p.add_argument("--config",     default=str(_DEFAULTS["config"]))
    p.add_argument("--output",     default=str(_DEFAULTS["output"]))
    p.add_argument("--n-per-cat",  type=int, default=N_PER_CAT)
    p.add_argument("--seed",       type=int, default=SEED)
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
