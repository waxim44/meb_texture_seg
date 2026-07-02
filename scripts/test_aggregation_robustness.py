"""
test_aggregation_robustness.py
══════════════════════════════════════════════════════════════════════════════
Teste si une agrégation robuste (médiane, trimmed-mean) bat la moyenne
à poids égal sur la séparabilité LP LOIO.

Hypothèse : un patch contient de la vraie texture + zones de bruit.
La moyenne intègre le bruit → agrégation robuste pourrait aider,
surtout sur les blocs early (beaucoup de vecteurs → beaucoup de bruit potentiel).

Pipeline :
  forward pass images entières → extraction à la volée → LP LOIO

Étape 0 (contrôle extraction) : SKIPPÉE — validée par validate_coordinate_extraction.py.
══════════════════════════════════════════════════════════════════════════════
"""

import sys, os, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ── SAM2 path ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SAM2 = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

H5_PATH   = _ROOT / "data" / "feature_database" / "database_meb.h5"
IMG_DIR   = _ROOT / "PatchTagger_Output" / "full_images"
CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
OUTDIR    = _ROOT / "output_ouassim" / "aggregation_robustness"
OUTDIR.mkdir(parents=True, exist_ok=True)

ORIG_H, ORIG_W = 768, 1280
PATCH_SZ = 128
IMG_SIZE = 1024

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

BLOCKS = ["block_0", "block_1", "block_4", "block_7", "block_13", "block_15"]
BLOCK_STRIDE = {
    "block_0": 4, "block_1": 8,
    "block_4": 16, "block_7": 16, "block_13": 16,
    "block_15": 32,
}
BLOCK_IDX = {"block_0": 0, "block_1": 1, "block_4": 4,
             "block_7": 7, "block_13": 13, "block_15": 15}
BLOCK_DIM = {"block_0": 96, "block_1": 192,
             "block_4": 384, "block_7": 384, "block_13": 384,
             "block_15": 768}

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TEXTURE_NAMES = {
    1: "Hom.", 3: "Faisceaux", 4: "Filaments",
    5: "Strat.rect.", 6: "Strat.sin.", 7: "Granuleux", 9: "Trou",
}

AGGS = ["mean", "median", "trim10", "trim20"]
AGG_LABELS = {"mean": "Moyenne", "median": "Médiane",
              "trim10": "Trim-10%", "trim20": "Trim-20%"}

LP_PCA_DIM = 50
LP_C = 1.0
SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# Nombre de vecteurs locaux par patch / bloc
# ═════════════════════════════════════════════════════════════════════════════

def n_vectors(block_key: str) -> int:
    s = BLOCK_STRIDE[block_key]
    w_p = int(PATCH_SZ * (IMG_SIZE / ORIG_W) / s)
    h_p = int(PATCH_SZ * (IMG_SIZE / ORIG_H) / s)
    return max(1, w_p * h_p)

N_VEC = {bk: n_vectors(bk) for bk in BLOCKS}


# ═════════════════════════════════════════════════════════════════════════════
# Modèle + hooks
# ═════════════════════════════════════════════════════════════════════════════

def build_encoder() -> ImageEncoder:
    trunk = Hiera(
        embed_dim=96, num_heads=1,
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


def load_encoder(device: str) -> ImageEncoder:
    enc = build_encoder()
    if CKPT_PATH.is_file():
        sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        prefix = "image_encoder."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexp = enc.load_state_dict(sd, strict=False)
        if not missing and not unexp:
            print(f"  [encoder] Checkpoint chargé : {CKPT_PATH.name}")
        else:
            print(f"  [encoder] Checkpoint partiel ({len(missing)} missing)")
    else:
        print(f"  [encoder] AVERTISSEMENT : checkpoint absent, poids aléatoires")
    return enc.to(device).eval()


def register_hooks(encoder: ImageEncoder, target_indices: list) -> tuple[dict, list]:
    """
    Hook uniquement les blocs nécessaires (target_indices).
    Sortie bloc Hiera : (B, H, W, C) — stockée telle quelle.
    """
    captured = {}
    handles  = []
    for i, block in enumerate(encoder.trunk.blocks):
        if i in target_indices:
            def _bh(m, inp, out, idx=i):
                captured[f"block_{idx}"] = out.detach()
            handles.append(block.register_forward_hook(_bh))
    return captured, handles


# ═════════════════════════════════════════════════════════════════════════════
# Prétraitement
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(img_path: Path, device: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# Extraction des vecteurs locaux d'un patch (conv. IDENTIQUE à build_feature_database.py)
# ═════════════════════════════════════════════════════════════════════════════

def extract_local_vectors(feat_bhwc: torch.Tensor, x_min, y_min, x_max, y_max) -> np.ndarray:
    """
    feat_bhwc : (1, H_feat, W_feat, C)
    Retourne (n_pos, C) — vecteurs locaux du patch, non normalisés.
    """
    feat = feat_bhwc[0]            # (H_feat, W_feat, C)
    H_feat, W_feat, C = feat.shape

    scale_x = W_feat / ORIG_W
    scale_y = H_feat / ORIG_H

    fx1 = max(0, int(x_min * scale_x))
    fy1 = max(0, int(y_min * scale_y))
    fx2 = min(W_feat, max(fx1 + 1, int(x_max * scale_x)))
    fy2 = min(H_feat, max(fy1 + 1, int(y_max * scale_y)))

    region = feat[fy1:fy2, fx1:fx2, :]    # (h_p, w_p, C)
    n_loc = (fy2 - fy1) * (fx2 - fx1)
    if n_loc == 0:
        return feat[fy1:fy1+1, fx1:fx1+1, :].reshape(1, C).cpu().numpy().astype(np.float32)
    return region.reshape(-1, C).cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Les 4 agrégations
# ═════════════════════════════════════════════════════════════════════════════

def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def aggregate(V: np.ndarray) -> dict:
    """
    V : (n_pos, D)
    Retourne dict {agg_name: vecteur L2-normalisé (D,)}
    """
    out = {}

    # Moyenne
    out["mean"] = l2norm(V.mean(axis=0))

    # Médiane (par dimension)
    out["median"] = l2norm(np.median(V, axis=0))

    # Trimmed-mean : on retire les vecteurs les plus loin du centre (distance cosinus)
    centre = V.mean(axis=0)
    c_norm = np.linalg.norm(centre)
    if c_norm > 1e-8 and len(V) >= 3:
        c_unit = centre / c_norm
        V_norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_unit  = np.where(V_norms > 1e-8, V / V_norms, 0.0)
        cos_sim = V_unit @ c_unit             # (n_pos,) — en [−1, 1]
        cos_dist = 1.0 - cos_sim              # plus c'est grand, plus c'est loin

        # trim10 : garder les 90% les plus proches
        thresh10 = np.percentile(cos_dist, 90)
        keep10 = V[cos_dist <= thresh10]
        out["trim10"] = l2norm(keep10.mean(axis=0)) if len(keep10) > 0 else out["mean"]

        # trim20 : garder les 80% les plus proches
        thresh20 = np.percentile(cos_dist, 80)
        keep20 = V[cos_dist <= thresh20]
        out["trim20"] = l2norm(keep20.mean(axis=0)) if len(keep20) > 0 else out["mean"]
    else:
        # Pas assez de vecteurs pour trimmer → retomber sur mean
        out["trim10"] = out["mean"].copy()
        out["trim20"] = out["mean"].copy()

    return out


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement H5 + forward pass + agrégation à la volée
# ═════════════════════════════════════════════════════════════════════════════

def step1_extract(device: str):
    """
    Retourne :
      feats[block_key][agg_name] = np.ndarray (N_tex, D)
      meta = dict avec 'cats', 'img_names', 'patch_ids' (indices dans H5)
    """
    print()
    print("=" * 72)
    print("ÉTAPE 1 — Forward pass + agrégation à la volée")
    print("=" * 72)

    # ── Charger les métadonnées H5 ────────────────────────────────────────────
    with h5py.File(H5_PATH, "r") as h5:
        h5_cats  = h5["metadata/category_ids"][:]
        h5_imgs  = np.array([n.decode() for n in h5["metadata/image_names"][:]])
        h5_pos   = h5["metadata/positions"][:]    # (N, 4) x_min,y_min,x_max,y_max

    # Filtrer sur les textures cibles
    tex_mask  = np.isin(h5_cats, TEXTURES)
    tex_ids   = np.where(tex_mask)[0]             # indices dans H5
    tex_cats  = h5_cats[tex_ids]
    tex_imgs  = h5_imgs[tex_ids]
    tex_pos   = h5_pos[tex_ids]
    N_tex     = len(tex_ids)

    print(f"  Patches de texture : {N_tex}  (images : {len(set(tex_imgs))})")
    print(f"  Blocs : {BLOCKS}")
    print(f"  Agrégations : {AGGS}")
    print()

    # ── Pré-allouer les features ──────────────────────────────────────────────
    feats = {
        bk: {ag: np.zeros((N_tex, BLOCK_DIM[bk]), dtype=np.float32) for ag in AGGS}
        for bk in BLOCKS
    }

    # ── Encoder + hooks ───────────────────────────────────────────────────────
    encoder = load_encoder(device)
    target_indices = [BLOCK_IDX[bk] for bk in BLOCKS]
    captured, handles = register_hooks(encoder, target_indices)

    # ── Forward image par image ───────────────────────────────────────────────
    imgs_unique = sorted(set(tex_imgs))
    patch_counter = 0
    t0 = time.time()

    for img_name in tqdm(imgs_unique, desc="Forward pass", unit="img"):
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            print(f"  [WARN] {img_name} introuvable — skip")
            continue

        tensor = preprocess(img_path, device)
        captured.clear()
        with torch.no_grad():
            encoder(tensor)

        # Patches de cette image (dans le sous-ensemble texture)
        img_mask = tex_imgs == img_name
        img_patch_idx = np.where(img_mask)[0]   # indices dans tex_ids

        for local_idx in img_patch_idx:
            x_min, y_min, x_max, y_max = tex_pos[local_idx]

            for bk in BLOCKS:
                feat_tensor = captured.get(f"block_{BLOCK_IDX[bk]}")
                if feat_tensor is None:
                    continue
                V = extract_local_vectors(feat_tensor, x_min, y_min, x_max, y_max)
                agg_dict = aggregate(V)
                for ag in AGGS:
                    feats[bk][ag][local_idx] = agg_dict[ag]

            patch_counter += 1

    for h in handles:
        h.remove()

    elapsed = time.time() - t0
    print(f"\n  {patch_counter} patches traités en {elapsed:.1f}s")
    print(f"  Stockage : {sum(BLOCK_DIM[bk]*4*4*N_tex/1e6 for bk in BLOCKS):.1f} MB")

    meta = {"cats": tex_cats, "img_names": tex_imgs, "h5_ids": tex_ids}
    return feats, meta


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — LP LOIO
# ═════════════════════════════════════════════════════════════════════════════

def loio_recall_one(X: np.ndarray, y_binary: np.ndarray,
                    img_names: np.ndarray, cat_labels: np.ndarray,
                    texture_id: int) -> tuple[float, float, int]:
    """
    Leave-One-Image-Out pour une texture (one-vs-rest).
    Retourne (mean_recall, std_recall, n_images_avec_texture).
    """
    unique_imgs = sorted(set(img_names))
    recalls = []

    for img in unique_imgs:
        test_mask  = img_names == img
        train_mask = ~test_mask

        # Uniquement si l'image de test contient au moins un patch de cette texture
        if not np.any(y_binary[test_mask]):
            continue

        X_tr, y_tr = X[train_mask], y_binary[train_mask]
        X_te, y_te = X[test_mask],  y_binary[test_mask]

        # PCA : fit sur train uniquement
        n_comp = min(LP_PCA_DIM, X_tr.shape[0] - 1, X_tr.shape[1])
        if n_comp < 1:
            continue
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        # LogReg one-vs-rest
        if len(np.unique(y_tr)) < 2:
            continue
        lr = LogisticRegression(
            class_weight="balanced", C=LP_C,
            max_iter=1000, random_state=SEED, solver="lbfgs"
        )
        lr.fit(X_tr_pca, y_tr)
        y_pred = lr.predict(X_te_pca)

        # Recall de la classe positive
        tp = int(np.sum((y_pred == 1) & (y_te == 1)))
        fn = int(np.sum((y_pred == 0) & (y_te == 1)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    if not recalls:
        return 0.0, 0.0, 0
    return float(np.mean(recalls)), float(np.std(recalls)), len(recalls)


def step2_loio(feats: dict, meta: dict):
    """
    Retourne results[block_key][agg_name][texture_id] = (mean_rec, std_rec, n_img)
    """
    print()
    print("=" * 72)
    print("ÉTAPE 2 — LP LOIO (Leave-One-Image-Out)")
    print("=" * 72)

    cats      = meta["cats"]
    img_names = meta["img_names"]

    results = {bk: {ag: {} for ag in AGGS} for bk in BLOCKS}

    n_total = len(BLOCKS) * len(AGGS) * len(TEXTURES)
    bar = tqdm(total=n_total, desc="LP LOIO", unit="run")

    for bk in BLOCKS:
        for ag in AGGS:
            X = feats[bk][ag]   # (N_tex, D)
            for tex_id in TEXTURES:
                y_bin = (cats == tex_id).astype(int)
                mean_r, std_r, n_img = loio_recall_one(
                    X, y_bin, img_names, cats, tex_id
                )
                results[bk][ag][tex_id] = (mean_r, std_r, n_img)
                bar.update(1)
    bar.close()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SORTIES
# ═════════════════════════════════════════════════════════════════════════════

def print_recall_table(results: dict):
    print()
    print("=" * 72)
    print("SORTIE 2 — Recall par texture × (bloc, agrégation)")
    print("=" * 72)

    for bk in BLOCKS:
        stride = BLOCK_STRIDE[bk]
        nv     = N_VEC[bk]
        dim    = BLOCK_DIM[bk]
        print()
        print(f"  ── {bk}  [stride={stride}, {nv} vec/patch, dim={dim}] ──────────────────")
        header = f"  {'Texture':<18s}"
        for ag in AGGS:
            header += f"  {AGG_LABELS[ag]:>12s}"
        print(header)
        print("  " + "-" * (18 + 14 * len(AGGS)))

        for tex in TEXTURES:
            tname = TEXTURE_NAMES[tex]
            row   = f"  {tname:<18s}"
            best_val = max(results[bk][ag][tex][0] for ag in AGGS)
            for ag in AGGS:
                mean_r, std_r, n_img = results[bk][ag][tex]
                val_str = f"{mean_r*100:.1f}±{std_r*100:.1f}"
                marker  = "*" if abs(mean_r - best_val) < 1e-9 else " "
                row += f"  {marker}{val_str:>11s}"
            row += f"   (n={results[bk][AGGS[0]][tex][2]})"
            print(row)


def compute_gains(results: dict) -> dict:
    """
    gains[bk][ag] = gain moyen (recall robuste − recall mean), en points.
    """
    gains = {bk: {} for bk in BLOCKS}
    for bk in BLOCKS:
        baseline = np.array([results[bk]["mean"][t][0] for t in TEXTURES])
        for ag in ["median", "trim10", "trim20"]:
            robust = np.array([results[bk][ag][t][0] for t in TEXTURES])
            gains[bk][ag] = float((robust - baseline).mean() * 100)
    return gains


def print_gain_table(gains: dict):
    print()
    print("=" * 72)
    print("SORTIE 3 — Gain moyen vs Moyenne (en points de recall)")
    print("=" * 72)
    print(f"  {'Bloc':<12s}  {'n_vec':>6s}  {'Médiane':>10s}  {'Trim-10%':>10s}  {'Trim-20%':>10s}")
    print("  " + "-" * 55)
    for bk in BLOCKS:
        nv = N_VEC[bk]
        g_med  = gains[bk]["median"]
        g_t10  = gains[bk]["trim10"]
        g_t20  = gains[bk]["trim20"]
        def fmt(g):
            return f"{g:+.2f}"
        print(f"  {bk:<12s}  {nv:>6d}  {fmt(g_med):>10s}  {fmt(g_t10):>10s}  {fmt(g_t20):>10s}")


def plot_gain_vs_nvec(gains: dict, results: dict):
    """
    Graphe : gain (médiane − mean) vs nombre de vecteurs par bloc.
    Hypothèse : le gain DÉCROÎT quand n_vec diminue.
    """
    nvecs = [N_VEC[bk] for bk in BLOCKS]
    colors = {"median": "blue", "trim10": "orange", "trim20": "red"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panneau 1 : gain vs n_vectors ─────────────────────────────────────────
    ax = axes[0]
    for ag in ["median", "trim10", "trim20"]:
        g_vals = [gains[bk][ag] for bk in BLOCKS]
        ax.plot(nvecs, g_vals, "o-", color=colors[ag],
                label=AGG_LABELS[ag], lw=2, ms=8)
        for bk, nv, g in zip(BLOCKS, nvecs, g_vals):
            ax.annotate(bk.replace("block_", "b"), (nv, g),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Nombre de vecteurs locaux par patch", fontsize=11)
    ax.set_ylabel("Gain moyen recall (points)", fontsize=11)
    ax.set_title("Gain agrégation robuste vs Moyenne\nen fonction du nb de vecteurs",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.invert_xaxis()   # plus de vecteurs à gauche (early blocks)

    # ── Panneau 2 : recall absolu des 4 agrégations, moyenné sur textures ─────
    ax2 = axes[1]
    agg_colors = {"mean": "gray", "median": "blue", "trim10": "orange", "trim20": "red"}
    for ag in AGGS:
        mean_recalls = [
            np.mean([results[bk][ag][t][0] for t in TEXTURES]) for bk in BLOCKS
        ]
        ax2.plot(nvecs, [r*100 for r in mean_recalls], "o-",
                 color=agg_colors[ag], label=AGG_LABELS[ag], lw=2, ms=7)

    ax2.set_xlabel("Nombre de vecteurs locaux par patch", fontsize=11)
    ax2.set_ylabel("Recall moyen (%)", fontsize=11)
    ax2.set_title("Recall moyen (toutes textures)\nen fonction du nb de vecteurs",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.invert_xaxis()

    plt.tight_layout()
    fname = OUTDIR / "gain_vs_nvec.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {fname.name}")


def plot_heatmap(results: dict):
    """
    Heatmap recall par (texture × bloc) pour chaque agrégation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ai, ag in enumerate(AGGS):
        ax = axes[ai]
        mat = np.array([
            [results[bk][ag][t][0] * 100 for bk in BLOCKS]
            for t in TEXTURES
        ])
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(BLOCKS)))
        ax.set_xticklabels([bk.replace("block_","b")+f"\n({N_VEC[bk]}v)"
                            for bk in BLOCKS], fontsize=8)
        ax.set_yticks(range(len(TEXTURES)))
        ax.set_yticklabels([TEXTURE_NAMES[t] for t in TEXTURES], fontsize=9)
        ax.set_title(f"Agrégation : {AGG_LABELS[ag]}", fontsize=11)
        for i, t in enumerate(TEXTURES):
            for j, bk in enumerate(BLOCKS):
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if 20 < mat[i,j] < 80 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (%)")

    plt.suptitle("Recall LP LOIO par (texture × bloc) — 4 agrégations", fontsize=13)
    plt.tight_layout()
    fname = OUTDIR / "heatmap_recall.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_gain_heatmap(results: dict):
    """
    Heatmap des gains (robuste − mean) par texture × bloc.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ai, ag in enumerate(["median", "trim10", "trim20"]):
        ax = axes[ai]
        mat = np.array([
            [(results[bk][ag][t][0] - results[bk]["mean"][t][0]) * 100
             for bk in BLOCKS]
            for t in TEXTURES
        ])
        vmax = max(3.0, float(np.abs(mat).max()))
        im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(BLOCKS)))
        ax.set_xticklabels([bk.replace("block_","b")+f"\n({N_VEC[bk]}v)"
                            for bk in BLOCKS], fontsize=8)
        ax.set_yticks(range(len(TEXTURES)))
        ax.set_yticklabels([TEXTURE_NAMES[t] for t in TEXTURES], fontsize=9)
        ax.set_title(f"Gain {AGG_LABELS[ag]} − Moyenne\n(rouge=pire, bleu=meilleur)", fontsize=10)
        for i in range(len(TEXTURES)):
            for j in range(len(BLOCKS)):
                ax.text(j, i, f"{mat[i,j]:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ Recall (points)")

    plt.suptitle("Gain agrégation robuste vs Moyenne — par (texture × bloc)",
                 fontsize=13)
    plt.tight_layout()
    fname = OUTDIR / "gain_heatmap.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname.name}")


def print_verdict(gains: dict, results: dict):
    print()
    print("=" * 72)
    print("SORTIE 4 — TEST DE L'HYPOTHÈSE + VERDICT")
    print("=" * 72)

    # Est-ce que le gain décroît avec n_vec ?
    print()
    print("  Hypothèse : gain(robuste − mean) DÉCROÎT quand n_vec diminue.")
    print("  (block_0=1050 vec → block_15=15 vec)")
    print()

    for ag in ["median", "trim10", "trim20"]:
        g_vals = [gains[bk][ag] for bk in BLOCKS]
        nvecs  = [N_VEC[bk] for bk in BLOCKS]

        # Corrélation de Spearman entre n_vec et gain
        from scipy.stats import spearmanr
        rho, pval = spearmanr(nvecs, g_vals)
        direction = "oui (corrélé positivement)" if rho > 0.3 and pval < 0.15 \
                    else "non (pas de tendance claire)"

        print(f"  {AGG_LABELS[ag]:10s} : gains sur blocs = "
              f"{[round(g,2) for g in g_vals]}")
        print(f"               Spearman(n_vec, gain) = {rho:.2f}  p={pval:.3f}")
        print(f"               Gain décroît avec n_vec ? → {direction}")
        print()

    # Verdict global
    any_gain = any(
        gains[bk][ag] > 0.5
        for bk in BLOCKS for ag in ["median", "trim10", "trim20"]
    )
    any_loss = any(
        gains[bk][ag] < -0.5
        for bk in BLOCKS for ag in ["median", "trim10", "trim20"]
    )
    best_mean = np.mean([results[bk]["mean"][t][0] for bk in BLOCKS for t in TEXTURES])
    best_med  = np.mean([results[bk]["median"][t][0] for bk in BLOCKS for t in TEXTURES])
    best_t10  = np.mean([results[bk]["trim10"][t][0] for bk in BLOCKS for t in TEXTURES])
    best_t20  = np.mean([results[bk]["trim20"][t][0] for bk in BLOCKS for t in TEXTURES])

    print(f"  Recall moyen global (toutes textures/blocs) :")
    print(f"    Moyenne   : {best_mean*100:.2f}%")
    print(f"    Médiane   : {best_med*100:.2f}%  (Δ={  (best_med-best_mean)*100:+.2f} pts)")
    print(f"    Trim-10%  : {best_t10*100:.2f}%  (Δ={(best_t10-best_mean)*100:+.2f} pts)")
    print(f"    Trim-20%  : {best_t20*100:.2f}%  (Δ={(best_t20-best_mean)*100:+.2f} pts)")
    print()

    if not any_gain and not any_loss:
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │ PAS DE GAIN SIGNIFICATIF                                    │")
        print("  │ Aucune agrégation robuste ne bat la moyenne de façon nette. │")
        print("  │ → Le bruit est diffus : la moyenne le nettoie déjà.         │")
        print("  │ → Cohérent avec l'hypothèse 'problème = encodeur,           │")
        print("  │   pas agrégation'. L'agrégation n'est pas le levier.        │")
        print("  └─────────────────────────────────────────────────────────────┘")
    elif any_gain and not any_loss:
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │ GAIN DÉTECTÉ                                                │")
        print("  │ Au moins une agrégation robuste bat la moyenne.             │")
        print("  │ → Le bruit était directionnel, contaminait la moyenne.      │")
        print("  │ → Gain gratuit à explorer.                                  │")
        print("  └─────────────────────────────────────────────────────────────┘")
    else:
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │ RÉSULTATS MIXTES                                            │")
        print("  │ Certains blocs/textures : gain; d'autres : perte.           │")
        print("  │ → Effet non systématique, voir heatmap détaillée.           │")
        print("  └─────────────────────────────────────────────────────────────┘")

    print()
    print(f"  Figures sauvées dans : {OUTDIR}/")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  TEST AGRÉGATION ROBUSTE : MOYENNE vs MÉDIANE vs TRIMMED        ║")
    print("║  Protocol : LP LOIO identique, seule l'agrégation change        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  H5       : {H5_PATH}")
    print(f"  Images   : {IMG_DIR}")
    print(f"  N_VEC/patch par bloc : {N_VEC}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    # Étape 0 skippée (validée par validate_coordinate_extraction.py)
    print()
    print("  [Étape 0 : skippée — extraction validée par validate_coordinate_extraction.py]")

    # Étape 1 : extraction
    feats, meta = step1_extract(device)

    # Étape 2 : LP LOIO
    results = step2_loio(feats, meta)

    # Sorties texte
    print_recall_table(results)
    gains = compute_gains(results)
    print_gain_table(gains)
    print_verdict(gains, results)

    # Sorties graphiques
    print()
    print("  Génération des figures...")
    plot_gain_vs_nvec(gains, results)
    plot_heatmap(results)
    plot_gain_heatmap(results)
    print("  Figures générées.")


if __name__ == "__main__":
    main()
