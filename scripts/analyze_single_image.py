"""
Analyse PCA RGB + t-SNE des embeddings TextureSAM sur une seule image STMD.

Usage:
    python scripts/analyze_single_image.py
"""

import os
import sys
import time
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "TextureSAM" / "sam2"
if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR / 'sam2'))

CKPT_PT  = ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
CKPT_DIR = ROOT / "checkpoints" / "sam2.1_hiera_small_1"
IMG_DIR  = ROOT / "data" / "raw" / "stmd" / "images"
LBL_DIR  = ROOT / "data" / "raw" / "stmd" / "labels"
OUT_DIR  = ROOT / "outputs" / "single_image_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
IMAGE_SIZE = 1024  # taille d'entrée SAM2
MAX_TSNE   = 2000  # points max par stage pour t-SNE

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

np.random.seed(SEED)
torch.manual_seed(SEED)

# Stage label → (conv index, spatial size)
STAGES = {
    "Stage 1": {"conv_idx": 3, "size": 256},
    "Stage 2": {"conv_idx": 2, "size": 128},
    "Stage 3": {"conv_idx": 1, "size": 64},
    "Stage 4": {"conv_idx": 0, "size": 32},
}
STAGE_NAMES   = list(STAGES.keys())
SELECTED_STAGE = "Stage 3"

# ── Chargement checkpoint ──────────────────────────────────────────────────────

def _load_ckpt():
    if CKPT_PT.is_file():
        try:
            sd = torch.load(CKPT_PT, map_location="cpu", weights_only=True)
            return sd.get("model", sd)
        except Exception:
            pass

    archive_dir = CKPT_DIR / "archive" if (CKPT_DIR / "archive").is_dir() else CKPT_DIR
    if archive_dir.is_dir():
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp_path = tmp.name
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for fp in archive_dir.rglob("*"):
                    if fp.is_file():
                        info = zipfile.ZipInfo(str(fp.relative_to(archive_dir.parent)))
                        info.date_time = (1980, 1, 1, 0, 0, 0)
                        with open(fp, "rb") as fh:
                            zf.writestr(info, fh.read())
            sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.unlink(tmp_path)
            return sd.get("model", sd)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return None


# ── Construction du modèle ─────────────────────────────────────────────────────

def build_encoder():
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine

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


# ── Prétraitement image ────────────────────────────────────────────────────────

def preprocess(img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
    x = x.permute(2, 0, 1)                               # (3, H, W)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0)                                 # (1, 3, H, W)


# ── Normalisation L2 ───────────────────────────────────────────────────────────

def l2_normalize(feat):
    """feat: (N, D) → (N, D) L2-normalisé."""
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.maximum(norms, 1e-8)


# ── PCA RGB ────────────────────────────────────────────────────────────────────

def pca_rgb(feat_hw_d):
    """
    feat_hw_d : (H, W, 256)
    Retourne  : (H, W, 3) float [0,1]
    """
    H, W, D = feat_hw_d.shape
    X = feat_hw_d.reshape(-1, D)
    X = l2_normalize(X)
    pca = PCA(n_components=3, random_state=SEED)
    rgb = pca.fit_transform(X)          # (H*W, 3)
    # normaliser chaque composante dans [0, 1]
    for c in range(3):
        mn, mx = rgb[:, c].min(), rgb[:, c].max()
        rgb[:, c] = (rgb[:, c] - mn) / max(mx - mn, 1e-8)
    return rgb.reshape(H, W, 3)


# ── Sous-échantillonnage stratifié ─────────────────────────────────────────────

def stratified_subsample(X, labels, max_pts, seed=SEED):
    """
    X      : (N, D)
    labels : (N,)
    Retourne X_sub (n, D), labels_sub (n,) avec n ≤ max_pts.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)

    indices = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(labels == cls)[0]
        quota = int(round(max_pts * cnt / n_total))
        quota = max(1, min(quota, len(cls_idx)))
        chosen = rng.choice(cls_idx, size=quota, replace=False)
        indices.append(chosen)

    idx = np.concatenate(indices)
    return X[idx], labels[idx], len(idx)


# ── Couleurs par classe ────────────────────────────────────────────────────────

def class_colors(classes):
    palette = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
        "#fabed4", "#469990",
    ]
    return {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}


# ── ÉTAPE 1 ────────────────────────────────────────────────────────────────────

def step1_load_image():
    print("\n═" * 30)
    print("ÉTAPE 1 — Chargement de l'image")
    print("═" * 30)

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in exts])
    assert images, f"Aucune image dans {IMG_DIR}"

    img_path = images[0]
    img_name = img_path.name

    # Label correspondant (.png)
    lbl_name = img_path.stem + ".png"
    lbl_path = LBL_DIR / lbl_name

    img_pil = Image.open(img_path)
    lbl_arr = np.array(Image.open(lbl_path))

    classes, counts = np.unique(lbl_arr, return_counts=True)

    print(f"  Image choisie : {img_name}")
    print(f"  Shape         : {np.array(img_pil).shape}  mode={img_pil.mode}")
    print(f"  Label         : {lbl_name}")
    print(f"  Classes GT    :")
    for cls, cnt in zip(classes, counts):
        print(f"    classe {cls:>3} : {cnt:>7} pixels ({100*cnt/lbl_arr.size:.1f}%)")

    return img_path, lbl_path, img_pil, lbl_arr


# ── ÉTAPE 2 ────────────────────────────────────────────────────────────────────

def step2_extract_features(encoder, device, img_path, lbl_arr):
    print("\n═" * 30)
    print("ÉTAPE 2 — Extraction des features")
    print("═" * 30)

    features = {}
    hooks = []

    for stage_name, info in STAGES.items():
        def _hook(module, inp, out, _name=stage_name):
            features[_name] = out.detach().cpu()
        h = encoder.neck.convs[info["conv_idx"]].register_forward_hook(_hook)
        hooks.append(h)

    x = preprocess(img_path).to(device)
    with torch.no_grad():
        encoder(x)

    for h in hooks:
        h.remove()

    result = {}
    gt_maps = {}
    for stage_name, info in STAGES.items():
        feat = features[stage_name]          # (1, 256, H, W)
        feat_np = feat[0].permute(1, 2, 0).numpy()  # (H, W, 256)
        result[stage_name] = feat_np

        # GT resizé nearest neighbor
        sz = info["size"]
        lbl_pil = Image.fromarray(lbl_arr)
        lbl_r   = np.array(lbl_pil.resize((sz, sz), Image.NEAREST))
        gt_maps[stage_name] = lbl_r

        print(f"  {stage_name}: feature map {feat_np.shape}  GT resizé {lbl_r.shape}")

    return result, gt_maps


# ── ÉTAPE 3 ────────────────────────────────────────────────────────────────────

def step3_pca_rgb(features, img_pil):
    print("\n═" * 30)
    print("ÉTAPE 3 — PCA RGB (4 stages)")
    print("═" * 30)

    img_rgb = img_pil.convert("RGB")

    col_titles = [
        "Original",
        "Stage 1\n(256×256)",
        "Stage 2\n(128×128)",
        "Stage 3 ★\n(64×64)",
        "Stage 4\n(32×32)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(img_rgb, cmap="gray")
    axes[0].set_title(col_titles[0], fontsize=11, fontweight="bold")
    axes[0].axis("off")

    pca_images = {}
    for i, stage_name in enumerate(STAGE_NAMES):
        feat = features[stage_name]
        rgb  = pca_rgb(feat)
        pca_images[stage_name] = rgb

        ax = axes[i + 1]
        ax.imshow(rgb)
        title = col_titles[i + 1]
        weight = "bold" if stage_name == SELECTED_STAGE else "normal"
        ax.set_title(title, fontsize=11, fontweight=weight)
        ax.axis("off")
        print(f"  {stage_name}: PCA RGB calculée → shape {rgb.shape}")

    plt.suptitle("PCA RGB des feature maps — 4 stages TextureSAM", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = OUT_DIR / "pca_rgb.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sauvegardé : {out_path}")
    return pca_images


# ── ÉTAPE 4 ────────────────────────────────────────────────────────────────────

def step4_tsne(features, gt_maps):
    print("\n═" * 30)
    print("ÉTAPE 4 — t-SNE par stage")
    print("═" * 30)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    tsne_counts = {}

    for i, stage_name in enumerate(STAGE_NAMES):
        feat = features[stage_name]   # (H, W, 256)
        gt   = gt_maps[stage_name]    # (H, W)
        H, W, D = feat.shape

        X = feat.reshape(-1, D)
        X = l2_normalize(X)
        y = gt.flatten()

        # Sous-échantillonnage stratifié
        X_sub, y_sub, n_sub = stratified_subsample(X, y, MAX_TSNE)
        tsne_counts[stage_name] = n_sub

        classes = np.unique(y_sub)
        n_cls   = len(classes)
        colors  = class_colors(classes)

        print(f"  {stage_name}: {n_sub} points, {n_cls} classes → PCA 50d + t-SNE 2d ...")

        # PCA 256 → 50
        n_components_pca = min(50, X_sub.shape[0] - 1, D)
        pca_pre = PCA(n_components=n_components_pca, random_state=SEED)
        X_pca   = pca_pre.fit_transform(X_sub)

        # t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            max_iter=1000,
            random_state=SEED,
        )
        X_2d = tsne.fit_transform(X_pca)

        ax = axes_flat[i]
        for cls in classes:
            mask = y_sub == cls
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors[cls],
                s=5,
                alpha=0.7,
                label=f"cls {cls}",
            )
        ax.set_title(f"{stage_name} — {n_cls} classes", fontsize=11)
        ax.legend(markerscale=2, fontsize=8, loc="best")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"    → t-SNE terminée.")

    plt.suptitle("t-SNE des embeddings par stage — colorié par classe GT", fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "tsne_stages.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sauvegardé : {out_path}")
    return tsne_counts


# ── ÉTAPE 5 ────────────────────────────────────────────────────────────────────

def step5_stage3_vs_gt(img_pil, pca_images, lbl_arr):
    print("\n═" * 30)
    print("ÉTAPE 5 — Stage 3 vs GT (superposition contours)")
    print("═" * 30)

    img_rgb  = np.array(img_pil.convert("RGB"))
    pca_s3   = pca_images[SELECTED_STAGE]            # (64, 64, 3)
    classes  = np.unique(lbl_arr)
    colors   = class_colors(classes)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Col 1 : image originale
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image originale", fontsize=11)
    axes[0].axis("off")

    # Col 2 : PCA RGB Stage 3
    axes[1].imshow(pca_s3)
    axes[1].set_title("PCA RGB Stage 3 ★", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Col 3 : GT mask
    cmap_gt = plt.cm.get_cmap("tab10", len(classes))
    gt_colored = np.zeros((*lbl_arr.shape, 3))
    for k, cls in enumerate(classes):
        c = np.array(mcolors.to_rgb(cmap_gt(k)))
        gt_colored[lbl_arr == cls] = c
    axes[2].imshow(gt_colored)
    axes[2].set_title("GT mask", fontsize=11)
    axes[2].axis("off")

    # Col 4 : PCA RGB Stage 3 + contours GT
    # Redimensionner PCA à la taille originale pour superposer
    pca_s3_large = np.array(
        Image.fromarray((pca_s3 * 255).astype(np.uint8)).resize(
            (lbl_arr.shape[1], lbl_arr.shape[0]), Image.BILINEAR
        )
    ) / 255.0

    axes[3].imshow(pca_s3_large)

    # Contours par classe
    try:
        from skimage.segmentation import find_boundaries
        for k, cls in enumerate(classes):
            mask = (lbl_arr == cls).astype(np.uint8)
            boundary = find_boundaries(mask, mode="outer")
            c = mcolors.to_rgb(cmap_gt(k))
            # Dessiner les contours en surimpression
            overlay = np.zeros((*lbl_arr.shape, 4))
            overlay[boundary, :3] = c
            overlay[boundary, 3]  = 1.0
            axes[3].imshow(overlay)
    except ImportError:
        # Sans scikit-image : contours matplotlib
        for k, cls in enumerate(classes):
            mask = (lbl_arr == cls).astype(float)
            c = mcolors.to_rgb(cmap_gt(k))
            axes[3].contour(mask, levels=[0.5], colors=[c], linewidths=1.0)

    axes[3].set_title("PCA RGB Stage 3 + contours GT", fontsize=11)
    axes[3].axis("off")

    plt.suptitle("Stage 3 (64×64) — Correspondance entre PCA RGB et zones GT", fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "stage3_vs_gt.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {out_path}")


# ── ÉTAPE 6 ────────────────────────────────────────────────────────────────────

def step6_summary(img_name, lbl_arr, features, tsne_counts, elapsed):
    print("\n" + "═" * 50)
    print("RÉSUMÉ")
    print("═" * 50)
    print(f"  Image analysée    : {img_name}")
    classes = np.unique(lbl_arr)
    print(f"  Classes GT        : {list(classes)}")
    print()
    for stage_name in STAGE_NAMES:
        feat = features[stage_name]
        n    = tsne_counts.get(stage_name, "?")
        star = " ★" if stage_name == SELECTED_STAGE else ""
        print(f"  {stage_name}{star}:")
        print(f"    feature map shape   : {feat.shape}")
        print(f"    points t-SNE        : {n}")
    print()
    print(f"  Temps total       : {elapsed:.1f} s")
    print("═" * 50)
    print(f"\n  Figures générées dans : {OUT_DIR}/")
    print(f"    • pca_rgb.png")
    print(f"    • tsne_stages.png")
    print(f"    • stage3_vs_gt.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("\nChargement du modèle...")
    encoder = build_encoder()
    sd = _load_ckpt()
    if sd is not None:
        prefix = "image_encoder."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if not missing and not unexpected:
            print("  [encoder] Checkpoint chargé.")
        else:
            print(f"  [encoder] Partiel ({len(missing)} manquantes, {len(unexpected)} inattendues).")
    else:
        print("  [encoder] WARNING — poids aléatoires.")
    encoder = encoder.to(device).eval()

    # ── Étapes ────────────────────────────────────────────────────────────────
    img_path, lbl_path, img_pil, lbl_arr = step1_load_image()
    features, gt_maps = step2_extract_features(encoder, device, img_path, lbl_arr)
    pca_images        = step3_pca_rgb(features, img_pil)
    tsne_counts       = step4_tsne(features, gt_maps)
    step5_stage3_vs_gt(img_pil, pca_images, lbl_arr)
    step6_summary(img_path.name, lbl_arr, features, tsne_counts, time.time() - t0)


if __name__ == "__main__":
    main()
