"""
Test de discrimination des feature maps TextureSAM sur KAUST256 et STMD.

Lance:
    python scripts/test_features.py

Sorties dans outputs/feature_test/ :
    pca_{dataset}_stage{1-4}.png
    heatmap_{dataset}_stage{1-4}.png
    scores.json
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT / "TextureSAM" / "sam2"))
sys.path.insert(0, str(_ROOT))

# sam2/__init__.py appelle initialize_config_module("sam2") au chargement,
# ce qui initialise GlobalHydra. On importe sam2 en premier, puis on clear
# GlobalHydra pour pouvoir utiliser notre propre config via compose.
from src.encoder.feature_extractor import TextureSAMExtractor  # déclenche sam2

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
initialize_config_dir(config_dir=str(_ROOT / "configs"), version_base=None)
cfg = compose(config_name="config")


# ── Constantes ─────────────────────────────────────────────────────────────────
STAGES = ["stage_4", "stage_3", "stage_2", "stage_1"]   # ordre résolution croissante
N_POINTS_PER_CLASS = 10   # pixels par classe pour heatmap
N_SUBSAMPLE = 2000        # points max pour métriques robustes
SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def paired_images(img_dir: Path, lbl_dir: Path):
    """Retourne liste de (img_path, lbl_path) avec correspondance par stem."""
    exts = {".jpg", ".jpeg", ".png", ".tif"}
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in exts}
    lbls = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() in exts}
    common = sorted(set(imgs) & set(lbls))
    return [(imgs[s], lbls[s]) for s in common]


def load_gt(lbl_path: Path) -> np.ndarray:
    """Charge un masque GT, le normalise en indices 0..K-1."""
    arr = np.array(Image.open(lbl_path))
    vals = np.unique(arr)
    mapping = {v: i for i, v in enumerate(vals)}
    return np.vectorize(mapping.get)(arr).astype(np.int32)


# ── Visualisation PCA RGB ──────────────────────────────────────────────────────

def pca_rgb(feat: np.ndarray, n_components=3) -> np.ndarray:
    """
    feat : (H, W, 256)
    Retourne : (H, W, 3) float32 dans [0, 1]
    """
    H, W, C = feat.shape
    flat = feat.reshape(-1, C)
    pca = PCA(n_components=n_components, random_state=SEED)
    proj = pca.fit_transform(flat)                 # (H*W, 3)
    # Normaliser chaque composante en [0, 1]
    for i in range(n_components):
        mn, mx = proj[:, i].min(), proj[:, i].max()
        if mx > mn:
            proj[:, i] = (proj[:, i] - mn) / (mx - mn)
        else:
            proj[:, i] = 0.0
    return proj.reshape(H, W, n_components).astype(np.float32)


def save_pca_figure(feat: np.ndarray, img_path: Path, stage: str, out_path: Path):
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    pca_img = pca_rgb(feat)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"PCA RGB — {stage}  |  {img_path.name}", fontsize=12, fontweight="bold")
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image originale", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(pca_img)
    axes[1].set_title(f"Features ({stage}) → PCA 3 composantes", fontsize=10)
    axes[1].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Heatmap distances cosines ──────────────────────────────────────────────────

def sample_pixels_by_class(feat: np.ndarray, gt: np.ndarray,
                            n_per_class: int = N_POINTS_PER_CLASS):
    """
    Sélectionne n_per_class pixels aléatoires par classe.
    feat : (H, W, 256), gt : (H, W) avec valeurs 0..K-1.
    Retourne (vectors, labels) — vectors : (n_total, 256).
    """
    rng = np.random.RandomState(SEED)
    # Adapter feat à la résolution du GT si différent
    H_f, W_f = feat.shape[:2]
    H_g, W_g = gt.shape
    if (H_f, W_f) != (H_g, W_g):
        from PIL import Image as PIL_Image
        gt_pil = PIL_Image.fromarray(gt.astype(np.uint8))
        gt_pil = gt_pil.resize((W_f, H_f), PIL_Image.NEAREST)
        gt = np.array(gt_pil).astype(np.int32)

    classes = np.unique(gt)
    vectors, labels = [], []
    for cls in classes:
        ys, xs = np.where(gt == cls)
        if len(ys) < 2:
            continue
        idx = rng.choice(len(ys), size=min(n_per_class, len(ys)), replace=False)
        vecs = feat[ys[idx], xs[idx], :]   # (n, 256)
        vectors.append(vecs)
        labels.extend([cls] * len(vecs))

    if not vectors:
        return None, None
    return np.vstack(vectors), np.array(labels)


def compute_scores(vectors: np.ndarray, labels: np.ndarray):
    """
    Calcule score = d_inter_mean / d_intra_mean (distances cosines).
    score > 1 → discriminant, > 2 → très bon.
    """
    dist = cosine_distances(vectors)   # (N, N), valeurs dans [0, 2]
    n = len(labels)

    intra_vals, inter_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_vals.append(dist[i, j])
            else:
                inter_vals.append(dist[i, j])

    if not intra_vals or not inter_vals:
        return None, None, None

    d_intra = float(np.mean(intra_vals))
    d_inter = float(np.mean(inter_vals))
    score = d_inter / d_intra if d_intra > 1e-8 else float("inf")
    return score, d_intra, d_inter


def collect_all_pixels(feat: np.ndarray, gt: np.ndarray):
    """
    Retourne (vecs, labels) : TOUS les pixels de l'image.
    feat : (H, W, 256) — gt : (H, W) indices 0..K-1.
    Redimensionne gt à la résolution de feat si nécessaire.
    """
    H_f, W_f = feat.shape[:2]
    H_g, W_g = gt.shape
    if (H_f, W_f) != (H_g, W_g):
        gt_pil = Image.fromarray(gt.astype(np.uint8))
        gt_pil = gt_pil.resize((W_f, H_f), Image.NEAREST)
        gt = np.array(gt_pil).astype(np.int32)
    vecs   = feat.reshape(-1, feat.shape[2])   # (H*W, 256)
    labels = gt.reshape(-1)                    # (H*W,)
    return vecs, labels


def subsample_stratified(vecs: np.ndarray, labels: np.ndarray,
                         n_total: int = N_SUBSAMPLE, seed: int = SEED):
    """
    Sous-échantillonnage stratifié : n_total points équilibrés entre classes.
    Même seed pour tous les stages → comparaison équitable.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    n_cls   = len(classes)
    n_each  = n_total // n_cls

    idx_list = []
    for cls in classes:
        idx_cls = np.where(labels == cls)[0]
        n_pick  = min(n_each, len(idx_cls))
        idx_list.append(rng.choice(idx_cls, size=n_pick, replace=False))

    idx_all = np.concatenate(idx_list)
    rng.shuffle(idx_all)
    return vecs[idx_all], labels[idx_all]


def compute_robust_metrics(vecs: np.ndarray, labels: np.ndarray):
    """
    Calcule Silhouette (cosine), Davies-Bouldin et Calinski-Harabasz.
    Retourne dict ou None si moins de 2 classes après sous-échantillonnage.
    """
    if len(np.unique(labels)) < 2:
        return None

    vecs_s, labels_s = subsample_stratified(vecs, labels)

    sil = float(silhouette_score(vecs_s, labels_s, metric="cosine"))
    db  = float(davies_bouldin_score(vecs_s, labels_s))
    ch  = float(calinski_harabasz_score(vecs_s, labels_s))

    return {
        "silhouette":        round(sil, 4),
        "davies_bouldin":    round(db,  4),
        "calinski_harabasz": round(ch,  2),
        "n_points":          len(labels_s),
        "n_classes":         int(len(np.unique(labels_s))),
    }


def save_heatmap(vectors: np.ndarray, labels: np.ndarray,
                 stage: str, dataset: str, out_path: Path):
    dist = cosine_distances(vectors)

    # Construire les séparateurs de classes
    unique_cls = []
    seen = set()
    for lbl in labels:
        if lbl not in seen:
            unique_cls.append(lbl)
            seen.add(lbl)
    boundaries = []
    count = 0
    for cls in unique_cls:
        n = (labels == cls).sum()
        count += n
        boundaries.append(count - 0.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist, cmap="viridis", vmin=0, vmax=dist.max())
    plt.colorbar(im, ax=ax, label="Distance cosine")

    for b in boundaries[:-1]:
        ax.axhline(b, color="red", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(b, color="red", linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_title(f"Distance cosine — {dataset} {stage}\n"
                 f"(blocs diagonaux sombres = bonne discrimination)",
                 fontsize=10)
    ax.set_xlabel("Pixels (par classe)")
    ax.set_ylabel("Pixels (par classe)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Tableau récapitulatif ──────────────────────────────────────────────────────

def print_score_table(dataset_name: str, stage_scores: dict):
    W1, W2 = 9, 22
    print()
    print("╔" + "═" * (W1 + W2 + 3) + "╗")
    title = f"Scores discrimination {dataset_name}"
    print("║" + title.center(W1 + W2 + 3) + "║")
    print("╠" + "═" * W1 + "╦" + "═" * (W2 + 2) + "╣")
    for stage in STAGES:
        sid = stage.split("_")[1]
        entry = stage_scores.get(stage)
        if entry is None:
            val = "N/A"
            badge = ""
        else:
            sc = entry["score"]
            if sc == float("inf"):
                val = "∞ (intra~0)"
                badge = " ✅✅"
            elif sc > 2:
                val = f"{sc:.3f}"
                badge = " ✅✅"
            elif sc > 1:
                val = f"{sc:.3f}"
                badge = " ✅"
            else:
                val = f"{sc:.3f}"
                badge = " ❌"
            val = f"score = {val}{badge}"
        print(f"║ {'Stage '+sid:^{W1-2}} ║ {val:<{W2}} ║")
    print("╚" + "═" * W1 + "╩" + "═" * (W2 + 2) + "╝")


# ── Tableau métriques robustes ────────────────────────────────────────────────

def print_metrics_table(dataset_name: str, metrics: dict, timings: dict):
    """
    metrics : {stage_name: {silhouette, davies_bouldin, calinski_harabasz, ...}}
    timings : {stage_name: float (secondes)}
    """
    W = [10, 14, 16, 18, 10]
    total_w = sum(W) + len(W) - 1

    def row_sep(left="╠", mid="╬", right="╣"):
        print(left + mid.join("═" * w for w in W) + right)

    print()
    title = f"Métriques complètes — {dataset_name} (checkpoint)"
    print("╔" + "═" * total_w + "╗")
    print("║" + title.center(total_w) + "║")
    row_sep("╠", "╦", "╣")
    headers = [" Stage ", " Silhouette ", " Davies-Bouldin ", " Calinski-Harabasz ", " Temps "]
    print("║" + "║".join(h.center(w) for h, w in zip(headers, W)) + "║")
    row_sep()

    best_sil = max(
        (m["silhouette"] for m in metrics.values() if m is not None), default=None
    )
    best_db = min(
        (m["davies_bouldin"] for m in metrics.values() if m is not None), default=None
    )
    best_ch = max(
        (m["calinski_harabasz"] for m in metrics.values() if m is not None), default=None
    )

    for stage in STAGES:
        sid = stage.split("_")[1]
        m   = metrics.get(stage)
        t   = timings.get(stage, 0.0)

        if m is None:
            cells = [f" Stage {sid} ", " N/A ", " N/A ", " N/A ", f" {t:.1f}s "]
        else:
            sil_b = " ★" if m["silhouette"] == best_sil else ""
            db_b  = " ★" if m["davies_bouldin"] == best_db else ""
            ch_b  = " ★" if m["calinski_harabasz"] == best_ch else ""
            cells = [
                f" Stage {sid} ",
                f" {m['silhouette']:+.4f}{sil_b} ",
                f" {m['davies_bouldin']:.4f}{db_b} ",
                f" {m['calinski_harabasz']:.1f}{ch_b} ",
                f" {t:.1f}s ",
            ]
        print("║" + "║".join(c.center(w) for c, w in zip(cells, W)) + "║")

    print("╚" + "╩".join("═" * w for w in W) + "╝")

    # Meilleurs stages
    if best_sil is not None:
        best_sil_stage = [s for s, m in metrics.items()
                          if m and m["silhouette"] == best_sil][0]
        best_db_stage  = [s for s, m in metrics.items()
                          if m and m["davies_bouldin"] == best_db][0]
        sid_sil = best_sil_stage.split("_")[1]
        sid_db  = best_db_stage.split("_")[1]
        print(f"  → meilleur Silhouette    : Stage {sid_sil}  "
              f"({best_sil:+.4f})  ✅  (↑ max)")
        print(f"  → meilleur Davies-Bouldin: Stage {sid_db}  "
              f"({best_db:.4f})   ✅  (↓ min)")


# ── Pipeline par dataset ───────────────────────────────────────────────────────

def run_dataset(extractor: TextureSAMExtractor, dataset_name: str,
                img_dir: Path, lbl_dir: Path, out_dir: Path) -> tuple:
    """
    Retourne (legacy_scores, robust_metrics, timings).

    legacy_scores : {stage: {score, d_intra, d_inter}}  (ancien score d_inter/d_intra)
    robust_metrics: {stage: {silhouette, davies_bouldin, calinski_harabasz, ...}}
    timings       : {stage: float}  secondes de calcul métrique
    """
    pairs = paired_images(img_dir, lbl_dir)
    if not pairs:
        print(f"  [WARN] Aucune paire image/GT dans {img_dir}")
        return {}, {}, {}

    print(f"\n  Dataset : {dataset_name}  ({len(pairs)} paires)")

    viz_img_path, viz_lbl_path = pairs[0]

    # Accumulation complète : tous les pixels de toutes les images
    accum_all  = {s: {"vecs": [], "labels": []} for s in STAGES}
    # Accumulation légère (10/classe/image) pour heatmap et legacy score
    accum_lite = {s: {"vecs": [], "labels": []} for s in STAGES}

    for img_path, lbl_path in pairs:
        feats = extractor.extract(str(img_path))
        gt    = load_gt(lbl_path)

        for stage in STAGES:
            feat = feats.get(stage)
            if feat is None:
                continue
            # Tous les pixels → métriques robustes
            v_all, l_all = collect_all_pixels(feat, gt)
            accum_all[stage]["vecs"].append(v_all)
            accum_all[stage]["labels"].append(l_all)
            # 10 par classe → heatmap / legacy
            v_lite, l_lite = sample_pixels_by_class(feat, gt)
            if v_lite is not None:
                accum_lite[stage]["vecs"].append(v_lite)
                accum_lite[stage]["labels"].append(l_lite)

    # Visualisations sur la première image
    viz_feats = extractor.extract(str(viz_img_path))
    viz_gt    = load_gt(viz_lbl_path)

    legacy_scores  = {}
    robust_metrics = {}
    timings        = {}

    for stage in STAGES:
        sid  = stage.split("_")[1]
        feat = viz_feats.get(stage)

        # ── PCA RGB ──
        if feat is not None:
            pca_path = out_dir / f"pca_{dataset_name.lower()}_stage{sid}.png"
            save_pca_figure(feat, viz_img_path, stage, pca_path)
            print(f"  → PCA       : {pca_path.relative_to(_ROOT)}")

        # ── Heatmap + legacy score (léger) ──
        lite_vecs  = accum_lite[stage]["vecs"]
        lite_lbls  = accum_lite[stage]["labels"]
        if lite_vecs:
            full_lite_v = np.vstack(lite_vecs)
            full_lite_l = np.concatenate(lite_lbls)

            if feat is not None:
                hm_v, hm_l = sample_pixels_by_class(viz_feats[stage], viz_gt)
                if hm_v is not None:
                    hm_path = out_dir / f"heatmap_{dataset_name.lower()}_stage{sid}.png"
                    save_heatmap(hm_v, hm_l, stage, dataset_name, hm_path)
                    print(f"  → Heatmap   : {hm_path.relative_to(_ROOT)}")

            sc, d_intra, d_inter = compute_scores(full_lite_v, full_lite_l)
            if sc is not None:
                legacy_scores[stage] = {
                    "score":   round(sc, 4),
                    "d_intra": round(d_intra, 4),
                    "d_inter": round(d_inter, 4),
                }

        # ── Métriques robustes (tous pixels, sous-échantillonnage 2000) ──
        all_vecs  = accum_all[stage]["vecs"]
        all_lbls  = accum_all[stage]["labels"]
        if not all_vecs:
            continue

        full_v = np.vstack(all_vecs)    # (N_total, 256)
        full_l = np.concatenate(all_lbls)

        n_raw = len(full_l)
        t0 = time.time()
        m  = compute_robust_metrics(full_v, full_l)
        elapsed = time.time() - t0

        timings[stage] = round(elapsed, 2)
        robust_metrics[stage] = m

        if m:
            print(
                f"  → Métriques {stage} : "
                f"sil={m['silhouette']:+.4f}  "
                f"db={m['davies_bouldin']:.4f}  "
                f"ch={m['calinski_harabasz']:.1f}  "
                f"({m['n_points']} pts/{n_raw} raw, {elapsed:.1f}s)"
            )

    return legacy_scores, robust_metrics, timings


# ── Main ───────────────────────────────────────────────────────────────────────

RANDOM_SCORES = {
    "KAUST": {
        "stage_4": 1.0125, "stage_3": 1.0126,
        "stage_2": 1.0088, "stage_1": 1.0077,
    },
    "STMD": {
        "stage_4": 1.0436, "stage_3": 1.0341,
        "stage_2": 1.0228, "stage_1": 1.0026,
    },
}


def print_comparison_table(dataset_name: str, ckpt_scores: dict):
    random_ds = RANDOM_SCORES.get(dataset_name, {})
    W = 14
    print()
    title = f"Random vs Checkpoint — {dataset_name}"
    total = W * 3 + 6
    print("╔" + "═" * total + "╗")
    print("║" + title.center(total) + "║")
    print("╠" + "═"*W + "╦" + "═"*W + "╦" + "═"*W + "╦" + "═"*W + "╣")
    print("║" + " Stage ".center(W) + "║" + " Random ".center(W) +
          "║" + " Checkpoint ".center(W) + "║" + " Δ ".center(W) + "║")
    print("╠" + "═"*W + "╬" + "═"*W + "╬" + "═"*W + "╬" + "═"*W + "╣")
    for stage in STAGES:
        sid = stage.split("_")[1]
        r_sc = random_ds.get(stage)
        entry = ckpt_scores.get(stage)
        c_sc = entry["score"] if entry else None

        r_str = f"{r_sc:.4f}" if r_sc is not None else "N/A"
        c_str = f"{c_sc:.4f}" if c_sc is not None else "N/A"

        if r_sc is not None and c_sc is not None:
            delta = (c_sc - r_sc) / r_sc * 100
            badge = " ✅✅" if delta > 10 else " ✅" if delta > 0 else " ❌"
            d_str = f"{delta:+.1f}%{badge}"
        else:
            d_str = "N/A"

        print("║" + f" Stage {sid}".center(W) + "║" + r_str.center(W) +
              "║" + c_str.center(W) + "║" + d_str.center(W) + "║")
    print("╚" + "═"*W + "╩" + "═"*W + "╩" + "═"*W + "╩" + "═"*W + "╝")


def main(cfg) -> None:
    root = _ROOT

    set_seeds(cfg.seed)

    out_dir = root / "outputs" / "feature_test_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("═" * 60)
    print("  Test features TextureSAM — Checkpoint entraîné")
    print("═" * 60)
    print(f"  Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Checkpoint : {cfg.encoder.checkpoint}")
    print(f"  Stage cfg  : {cfg.encoder.stage} (0=tous)")

    # Construire l'extracteur
    extractor = TextureSAMExtractor(cfg, root_dir=root)

    datasets = {
        "KAUST": (
            root / "data" / "raw" / "kaust" / "images",
            root / "data" / "raw" / "kaust" / "labels",
        ),
        "STMD": (
            root / "data" / "raw" / "stmd" / "images",
            root / "data" / "raw" / "stmd" / "labels",
        ),
    }

    all_legacy  = {}
    all_robust  = {}
    all_timings = {}

    for ds_name, (img_dir, lbl_dir) in datasets.items():
        if not img_dir.is_dir():
            print(f"  [WARN] {img_dir} introuvable — dataset ignoré")
            continue
        legacy, robust, timings = run_dataset(
            extractor, ds_name, img_dir, lbl_dir, out_dir
        )
        all_legacy[ds_name]  = legacy
        all_robust[ds_name]  = robust
        all_timings[ds_name] = timings

    # ── Tableaux métriques robustes ──────────────────────────────────────────
    print()
    print("═" * 62)
    print("  Métriques statistiques robustes (N=2000, seed=42)")
    print("═" * 62)
    for ds_name in all_robust:
        print_metrics_table(ds_name, all_robust[ds_name], all_timings[ds_name])

    # ── Tableaux legacy score ─────────────────────────────────────────────────
    print()
    print("═" * 62)
    print("  Ancien score d_inter/d_intra (10 pts/classe)")
    print("═" * 62)
    for ds_name, scores in all_legacy.items():
        print_score_table(ds_name, scores)

    # ── Comparaison Random vs Checkpoint (legacy) ─────────────────────────────
    print()
    print("═" * 62)
    print("  Comparaison Random vs Checkpoint")
    print("═" * 62)
    for ds_name, scores in all_legacy.items():
        print_comparison_table(ds_name, scores)

    # ── Amélioration moyenne ──────────────────────────────────────────────────
    print()
    print("  Amélioration relative par stage (moyenne datasets) :")
    for stage in STAGES:
        sid = stage.split("_")[1]
        improvements = []
        for ds_name, scores in all_legacy.items():
            r_sc  = RANDOM_SCORES.get(ds_name, {}).get(stage)
            entry = scores.get(stage)
            if r_sc and entry:
                improvements.append((entry["score"] - r_sc) / r_sc * 100)
        if improvements:
            avg   = sum(improvements) / len(improvements)
            badge = " ✅✅" if avg > 10 else " ✅" if avg > 0 else " ❌"
            print(f"  Stage {sid} : {avg:+.1f}%{badge}")

    print()
    print("  Interprétation Silhouette :")
    print("  → sil > 0.5  : clusters bien séparés ✅✅")
    print("  → sil > 0.2  : structure présente ✅")
    print("  → sil < 0    : chevauchement de classes ❌")
    print("  Interprétation Davies-Bouldin :")
    print("  → db < 1.0   : bonne séparation ✅✅")
    print("  → db < 2.0   : séparation acceptable ✅")
    print("  → db ↓ = mieux")

    # ── Sauvegardes ───────────────────────────────────────────────────────────
    scores_path = out_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(all_legacy, f, indent=2)

    metrics_path = root / "outputs" / "metrics_full.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            ds: {
                stage: all_robust[ds].get(stage)
                for stage in STAGES
            }
            for ds in all_robust
        }, f, indent=2)

    print(f"\n  Legacy scores  : {scores_path.relative_to(root)}")
    print(f"  Métriques full : {metrics_path.relative_to(root)}")
    print(f"  Visuels        : {out_dir.relative_to(root)}")
    print()

    extractor.remove_hooks()


if __name__ == "__main__":
    main(cfg)
