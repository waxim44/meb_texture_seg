"""
Tests de validation des checkpoints TextureSAM.

TEST 1 — Comparaison 3 checkpoints sur Stage 1 PCA
TEST 2 — Nearest Texture Neighbor (NNT) score
TEST 3 — Stabilité sous perturbations
TEST 4 — PCA RGB Stage 1 vs Stage 2

Usage:
    python scripts/test_checkpoints.py
"""

import os
import sys
import json
import random
import zipfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TextureSAM" / "sam2"))
sys.path.insert(0, str(ROOT))

from src.encoder.feature_extractor import (
    _build_image_encoder, TextureSAMExtractor,
)

SEED  = 42
RNG   = np.random.RandomState(SEED)

STAGES_TEST = ["stage_3", "stage_2", "stage_1"]   # stages pour NNT + stabilité

CKPT_NAMES = {
    "orig":  "sam2.1_hiera_small",
    "0.3":   "sam2.1_hiera_small_0.3",
    "1.0":   "sam2.1_hiera_small_1",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def paired_images(img_dir: Path, lbl_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif"}
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in exts}
    lbls = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() in exts}
    common = sorted(set(imgs) & set(lbls))
    return [(imgs[s], lbls[s]) for s in common]


def load_gt(lbl_path: Path) -> np.ndarray:
    arr = np.array(Image.open(lbl_path))
    vals = np.unique(arr)
    mapping = {v: i for i, v in enumerate(vals)}
    return np.vectorize(mapping.get)(arr).astype(np.int32)


def pca_rgb(feat: np.ndarray) -> np.ndarray:
    H, W, C = feat.shape
    flat = feat.reshape(-1, C)
    pca  = PCA(n_components=3, random_state=SEED)
    proj = pca.fit_transform(flat)
    for i in range(3):
        mn, mx = proj[:, i].min(), proj[:, i].max()
        proj[:, i] = (proj[:, i] - mn) / (mx - mn + 1e-8)
    return proj.reshape(H, W, 3).astype(np.float32)


def resize_gt(gt: np.ndarray, H: int, W: int) -> np.ndarray:
    if gt.shape == (H, W):
        return gt
    gt_pil = Image.fromarray(gt.astype(np.uint8))
    return np.array(gt_pil.resize((W, H), Image.NEAREST)).astype(np.int32)


# ── Chargement checkpoint (fichier .pt ou répertoire archive) ──────────────────

def _zip_dir_to_pt(ckpt_dir: Path) -> str:
    archive = ckpt_dir / "archive" if (ckpt_dir / "archive").is_dir() else ckpt_dir
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for fp in sorted(archive.rglob("*")):
            if fp.is_file():
                info = zipfile.ZipInfo(str(fp.relative_to(archive.parent)))
                info.date_time = (1980, 1, 1, 0, 0, 0)
                with open(fp, "rb") as fh:
                    zf.writestr(info, fh.read())
    return tmp_path


def load_encoder_from_ckpt(ckpt_label: str, ckpt_name: str, device: str):
    """
    Charge un ImageEncoder depuis checkpoints/{ckpt_name} (.pt ou répertoire).
    Retourne (encoder, True) si succès, (None, False) si absent.
    """
    ckpt_pt  = ROOT / "checkpoints" / f"{ckpt_name}.pt"
    ckpt_dir = ROOT / "checkpoints" / ckpt_name

    tmp_path = None
    sd = None

    if ckpt_pt.is_file():
        try:
            sd = torch.load(ckpt_pt, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [WARN] {ckpt_label}: échec lecture .pt — {e}")

    elif ckpt_dir.is_dir():
        try:
            tmp_path = _zip_dir_to_pt(ckpt_dir)
            sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [WARN] {ckpt_label}: échec lecture répertoire — {e}")
            return None, False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        print(f"  [WARN] {ckpt_label}: checkpoint absent "
              f"(ni {ckpt_pt.name} ni {ckpt_dir.name}/) — skippé")
        return None, False

    if sd is None:
        return None, False

    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    encoder = _build_image_encoder()
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [WARN] {ckpt_label}: {len(missing)} clés manquantes")
    encoder = encoder.to(device).eval()
    print(f"  ✅ {ckpt_label} ({ckpt_name}) chargé")
    return encoder, True


def extract_stage(encoder, img_path: Path, stage: str, device: str) -> np.ndarray:
    """Extrait la feature map d'un stage sur une image. Retourne (H,W,256)."""
    CONV_IDX = {"stage_4": 0, "stage_3": 1, "stage_2": 2, "stage_1": 3}
    result = {}

    def hook(m, inp, out, _s=stage):
        result[_s] = out.detach().cpu()

    h = encoder.neck.convs[CONV_IDX[stage]].register_forward_hook(hook)

    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((1024, 1024), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
    x = (x - MEAN) / STD

    with torch.no_grad():
        encoder(x)
    h.remove()

    feat = result[stage][0].permute(1, 2, 0).numpy()   # (H,W,256)
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Comparaison 3 checkpoints, Stage 1 PCA
# ══════════════════════════════════════════════════════════════════════════════

def test1_checkpoint_comparison(encoders: dict, stmd_pairs: list, out_dir: Path):
    print("\n  TEST 1 — Comparaison PCA Stage 1 (5 images STMD)")
    out_dir.mkdir(parents=True, exist_ok=True)
    first_enc = next(iter(encoders.values()))
    device = next(first_enc.parameters()).device

    n_imgs = min(5, len(stmd_pairs))
    ckpt_keys = list(encoders.keys())   # ex: ["orig", "0.3", "1.0"]
    n_cols = 1 + len(ckpt_keys)        # original + N checkpoints

    for img_path, _ in stmd_pairs[:n_imgs]:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        fig.suptitle(f"Stage 1 PCA — {img_path.name}", fontsize=11, fontweight="bold")

        axes[0].imshow(img_rgb)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")

        for j, (label, enc) in enumerate(encoders.items()):
            feat = extract_stage(enc, img_path, "stage_1", str(device))
            pca  = pca_rgb(feat)
            axes[j + 1].imshow(pca)
            axes[j + 1].set_title(f"ckpt {label}", fontsize=9)
            axes[j + 1].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{img_path.stem}_stage1_compare.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  → {out_path.relative_to(ROOT)}")

    return n_imgs


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Nearest Texture Neighbor
# ══════════════════════════════════════════════════════════════════════════════

def _ntt_one_image(feat: np.ndarray, gt: np.ndarray,
                   n_per_half: int = 30) -> tuple:
    """
    Calcule d_same (même classe, moitiés gauche/droite) et
    d_diff (classes différentes) pour une image.
    Retourne (d_same, d_diff) ou (None, None) si pas assez de données.
    """
    H, W = feat.shape[:2]
    gt_r = resize_gt(gt, H, W)

    classes = np.unique(gt_r)
    if len(classes) < 2:
        return None, None

    left_mask  = np.zeros((H, W), dtype=bool)
    left_mask[:, :W // 2] = True
    right_mask = ~left_mask

    same_pairs_left  = []
    same_pairs_right = []
    diff_a, diff_b   = [], []

    for cls in classes:
        mask_cls = gt_r == cls
        left_idx  = np.argwhere(mask_cls & left_mask)
        right_idx = np.argwhere(mask_cls & right_mask)
        if len(left_idx) < 2 or len(right_idx) < 2:
            continue
        nl = min(n_per_half, len(left_idx))
        nr = min(n_per_half, len(right_idx))
        il = RNG.choice(len(left_idx),  nl, replace=False)
        ir = RNG.choice(len(right_idx), nr, replace=False)
        same_pairs_left.append(feat[left_idx[il, 0],  left_idx[il, 1],  :])
        same_pairs_right.append(feat[right_idx[ir, 0], right_idx[ir, 1], :])

    if len(same_pairs_left) < 1:
        return None, None

    vl = np.vstack(same_pairs_left)
    vr = np.vstack(same_pairs_right)
    d_same = float(np.mean(cosine_distances(vl, vr)))

    # d_diff : entre classes différentes
    all_class_vecs = []
    all_class_lbls = []
    for cls in classes:
        ys, xs = np.where(gt_r == cls)
        if len(ys) < 2:
            continue
        idx = RNG.choice(len(ys), min(40, len(ys)), replace=False)
        all_class_vecs.append(feat[ys[idx], xs[idx], :])
        all_class_lbls.extend([cls] * min(40, len(ys)))

    if len(all_class_vecs) < 2:
        return d_same, None

    vecs   = np.vstack(all_class_vecs)
    labels = np.array(all_class_lbls)
    dist   = cosine_distances(vecs)
    n      = len(labels)
    inter  = [dist[i, j] for i in range(n) for j in range(i+1, n)
              if labels[i] != labels[j]]
    d_diff = float(np.mean(inter)) if inter else None

    return d_same, d_diff


def test2_ntt(encoder, stmd_pairs: list, device: str) -> dict:
    print("\n  TEST 2 — Nearest Texture Neighbor (NNT)")
    results = {}
    n_imgs  = min(10, len(stmd_pairs))

    for stage in STAGES_TEST:
        d_same_list, d_diff_list = [], []
        for img_path, lbl_path in stmd_pairs[:n_imgs]:
            feat = extract_stage(encoder, img_path, stage, device)
            gt   = load_gt(lbl_path)
            ds, dd = _ntt_one_image(feat, gt)
            if ds is not None:
                d_same_list.append(ds)
            if dd is not None:
                d_diff_list.append(dd)

        if d_same_list and d_diff_list:
            d_same = float(np.mean(d_same_list))
            d_diff = float(np.mean(d_diff_list))
            ntt    = d_diff / d_same if d_same > 1e-8 else float("inf")
        else:
            d_same, d_diff, ntt = None, None, None

        results[stage] = {
            "d_same_texture": round(d_same, 4) if d_same else None,
            "d_diff_texture": round(d_diff, 4) if d_diff else None,
            "ntt_score":      round(ntt, 4)    if ntt and ntt != float("inf") else ntt,
        }
        sid = stage.split("_")[1]
        if ntt is not None:
            print(f"  {stage}: d_same={d_same:.4f}  d_diff={d_diff:.4f}  "
                  f"NNT={ntt:.4f}")

    return results


def print_ntt_table(ntt_results: dict):
    W = [10, 16, 12]
    total = sum(W) + len(W) - 1
    print()
    print("╔" + "═"*total + "╗")
    print("║" + "Nearest Texture Neighbor Test".center(total) + "║")
    print("╠" + "╦".join("═"*w for w in W) + "╣")
    print("║" + "║".join(h.center(w) for h, w in zip(
        [" Stage ", " NNT Score     ", " Verdict   "], W)) + "║")
    print("╠" + "╬".join("═"*w for w in W) + "╣")
    for stage in STAGES_TEST:
        sid = stage.split("_")[1]
        m   = ntt_results.get(stage, {})
        sc  = m.get("ntt_score") if m else None
        if sc is None:
            verdict = "N/A"
            sc_str  = "N/A"
        elif sc == float("inf"):
            sc_str  = "∞"
            verdict = "✅✅ texture"
        elif sc > 1.5:
            sc_str  = f"{sc:.4f}"
            verdict = "✅✅ texture"
        elif sc > 1.0:
            sc_str  = f"{sc:.4f}"
            verdict = "✅ texture"
        else:
            sc_str  = f"{sc:.4f}"
            verdict = "❌ position"
        print("║" + "║".join(c.center(w) for c, w in zip(
            [f" Stage {sid} ", f" {sc_str} ", f" {verdict} "], W)) + "║")
    print("╚" + "╩".join("═"*w for w in W) + "╝")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Stabilité sous perturbations
# ══════════════════════════════════════════════════════════════════════════════

def _apply_perturbation(img_rgb: np.ndarray, kind: str) -> np.ndarray:
    arr = img_rgb.astype(np.float32)
    if kind == "noise":
        arr = arr + RNG.randn(*arr.shape).astype(np.float32) * 0.05 * 255
        arr = np.clip(arr, 0, 255)
    elif kind == "rotation":
        from PIL import Image as PILImage
        pil = PILImage.fromarray(arr.astype(np.uint8))
        pil = pil.rotate(5, resample=PILImage.BILINEAR, expand=False)
        arr = np.array(pil).astype(np.float32)
    elif kind == "contrast":
        factor = 1.1   # +10 %
        mean   = arr.mean()
        arr    = mean + factor * (arr - mean)
        arr    = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _feats_from_array(encoder, img_arr: np.ndarray, stage: str,
                      device: str) -> np.ndarray:
    """Extrait les features depuis un np.ndarray (H,W,3) uint8."""
    CONV_IDX = {"stage_4": 0, "stage_3": 1, "stage_2": 2, "stage_1": 3}
    result = {}

    def hook(m, inp, out, _s=stage):
        result[_s] = out.detach().cpu()

    h = encoder.neck.convs[CONV_IDX[stage]].register_forward_hook(hook)

    img = Image.fromarray(img_arr).convert("RGB").resize((1024, 1024), Image.BILINEAR)
    x   = torch.from_numpy(np.array(img)).float() / 255.0
    x   = x.permute(2, 0, 1).unsqueeze(0).to(device)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
    x = (x - MEAN) / STD

    with torch.no_grad():
        encoder(x)
    h.remove()
    return result[stage][0].permute(1, 2, 0).numpy()


def _stability_one(encoder, img_path: Path, lbl_path: Path,
                   stage: str, device: str, n_patches: int = 20) -> dict:
    """
    Calcule stability_score = 1 - mean_cosine_distance(orig, perturbed)
    pour les 3 perturbations, sur des patches d'une classe de référence.
    """
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    gt      = load_gt(lbl_path)

    feat_orig = _feats_from_array(encoder, img_rgb, stage, device)
    H, W      = feat_orig.shape[:2]
    gt_r      = resize_gt(gt, H, W)

    # Choisir la classe la plus représentée (hors fond = classe 0)
    classes   = np.unique(gt_r)
    classes   = classes[classes > 0] if len(classes) > 1 else classes
    counts    = [(cls, int((gt_r == cls).sum())) for cls in classes]
    ref_cls   = max(counts, key=lambda x: x[1])[0]

    ys, xs = np.where(gt_r == ref_cls)
    if len(ys) < n_patches:
        return {}
    idx     = RNG.choice(len(ys), n_patches, replace=False)
    ref_vecs = feat_orig[ys[idx], xs[idx], :]   # (n, 256)

    scores = {}
    for kind in ["noise", "rotation", "contrast"]:
        perturbed    = _apply_perturbation(img_rgb, kind)
        feat_pert    = _feats_from_array(encoder, perturbed, stage, device)
        pert_vecs    = feat_pert[ys[idx], xs[idx], :]
        dist         = np.diag(cosine_distances(ref_vecs, pert_vecs))
        stability    = float(1.0 - np.mean(dist))
        scores[kind] = round(stability, 4)
    return scores


def test3_stability(encoder, stmd_pairs: list, device: str) -> dict:
    print("\n  TEST 3 — Stabilité sous perturbations")
    n_imgs = min(3, len(stmd_pairs))
    results = {}

    for stage in STAGES_TEST:
        stage_scores = {"noise": [], "rotation": [], "contrast": []}
        for img_path, lbl_path in stmd_pairs[:n_imgs]:
            sc = _stability_one(encoder, img_path, lbl_path, stage, device)
            for k in stage_scores:
                if k in sc:
                    stage_scores[k].append(sc[k])

        results[stage] = {
            k: round(float(np.mean(v)), 4) if v else None
            for k, v in stage_scores.items()
        }
        sid = stage.split("_")[1]
        r   = results[stage]
        print(f"  {stage}: bruit={r['noise']}  "
              f"rotation={r['rotation']}  contraste={r['contrast']}")

    return results


def print_stability_table(stab_results: dict):
    W = [10, 11, 11, 12]
    total = sum(W) + len(W) - 1
    print()
    print("╔" + "═"*total + "╗")
    print("║" + "Stabilité sous perturbations".center(total) + "║")
    print("╠" + "╦".join("═"*w for w in W) + "╣")
    print("║" + "║".join(h.center(w) for h, w in zip(
        [" Stage ", " Bruit   ", " Rotation", " Contraste "], W)) + "║")
    print("╠" + "╬".join("═"*w for w in W) + "╣")
    for stage in STAGES_TEST:
        sid = stage.split("_")[1]
        r   = stab_results.get(stage, {})
        def fmt(v):
            if v is None: return "N/A"
            badge = " ✅" if v >= 0.8 else " ❌"
            return f"{v:.4f}{badge}"
        cells = [f" Stage {sid} ", fmt(r.get("noise")),
                 fmt(r.get("rotation")), fmt(r.get("contrast"))]
        print("║" + "║".join(c.center(w) for c, w in zip(cells, W)) + "║")
    print("╚" + "╩".join("═"*w for w in W) + "╝")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — PCA RGB Stage 1 vs Stage 2
# ══════════════════════════════════════════════════════════════════════════════

def test4_pca_comparison(encoder, stmd_pairs: list, kaust_pairs: list,
                         out_dir: Path, device: str):
    print("\n  TEST 4 — PCA RGB Stage 1 vs Stage 2")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [("STMD", stmd_pairs[:5]), ("KAUST", kaust_pairs[:5])]

    for ds_name, pairs in datasets:
        for img_path, _ in pairs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            f1 = extract_stage(encoder, img_path, "stage_1", device)
            f2 = extract_stage(encoder, img_path, "stage_2", device)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig.suptitle(f"{ds_name} — {img_path.name}", fontsize=11, fontweight="bold")

            axes[0].imshow(img_rgb)
            axes[0].set_title("Original", fontsize=9)
            axes[0].axis("off")

            axes[1].imshow(pca_rgb(f1))
            axes[1].set_title("Stage 1 — 256×256 (4px/vecteur)", fontsize=9)
            axes[1].axis("off")

            axes[2].imshow(pca_rgb(f2))
            axes[2].set_title("Stage 2 — 128×128 (8px/vecteur)", fontsize=9)
            axes[2].axis("off")

            plt.tight_layout()
            out_path = out_dir / f"{ds_name.lower()}_{img_path.stem}_comparison.png"
            plt.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close()
            print(f"  → {out_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMANDATION FINALE
# ══════════════════════════════════════════════════════════════════════════════

def recommend(ntt_results: dict, stab_results: dict):
    """Retourne le stage recommandé avec justification."""
    scores = {}
    for stage in STAGES_TEST:
        sid  = stage.split("_")[1]
        ntt  = ntt_results.get(stage, {}).get("ntt_score") or 0
        stab = stab_results.get(stage, {})
        stab_mean = np.mean([v for v in stab.values() if v is not None] or [0])
        scores[stage] = {"ntt": ntt, "stab": stab_mean, "sid": sid}

    # Normaliser et combiner (NNT pondéré 60%, stabilité 40%)
    ntt_vals  = [scores[s]["ntt"]  for s in STAGES_TEST]
    stab_vals = [scores[s]["stab"] for s in STAGES_TEST]
    ntt_max   = max(ntt_vals)  or 1
    stab_max  = max(stab_vals) or 1

    combined = {}
    for stage in STAGES_TEST:
        n = scores[stage]["ntt"]  / ntt_max
        s = scores[stage]["stab"] / stab_max
        combined[stage] = 0.6 * n + 0.4 * s

    best = max(combined, key=combined.get)
    sid  = best.split("_")[1]

    print()
    print("═" * 52)
    print("  RECOMMANDATION FINALE")
    print("═" * 52)
    print()
    print(f"  Stage recommandé pour la CAH MEB : Stage {sid}")
    print()
    print("  Critères de décision :")
    for stage in STAGES_TEST:
        s  = scores[stage]
        c  = combined[stage]
        marker = " ←── RECOMMANDÉ" if stage == best else ""
        print(f"  Stage {s['sid']}:  NNT={s['ntt']:.4f}  "
              f"stab={s['stab']:.4f}  score={c:.3f}{marker}")
    print()
    print("  Justification :")
    if sid == "1":
        print("  Stage 1 (256×256) — résolution maximale, détail fin.")
        print("  Idéal pour textures MEB à haute fréquence spatiale.")
    elif sid == "2":
        print("  Stage 2 (128×128) — bon compromis résolution/sémantique.")
        print("  Captures les patterns texturaux à échelle intermédiaire.")
    elif sid == "3":
        print("  Stage 3 (64×64) — représentation plus sémantique.")
        print("  Discriminant sur des classes texturales distinctes.")
    print()
    return best, sid


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_base = ROOT / "outputs" / "validation"
    out_base.mkdir(parents=True, exist_ok=True)

    print()
    print("═" * 60)
    print("  Test Checkpoints — TextureSAM")
    print("═" * 60)
    print(f"  Device : {device}")

    # Paires images/labels
    stmd_pairs  = paired_images(
        ROOT / "data" / "raw" / "stmd"  / "images",
        ROOT / "data" / "raw" / "stmd"  / "labels",
    )
    kaust_pairs = paired_images(
        ROOT / "data" / "raw" / "kaust" / "images",
        ROOT / "data" / "raw" / "kaust" / "labels",
    )
    print(f"  STMD  : {len(stmd_pairs)} paires")
    print(f"  KAUST : {len(kaust_pairs)} paires")

    # ── Chargement checkpoints ─────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Chargement des checkpoints")
    print("─" * 60)

    encoders = {}
    for label, name in CKPT_NAMES.items():
        enc, ok = load_encoder_from_ckpt(label, name, device)
        if ok:
            encoders[label] = enc

    if not encoders:
        print("  ❌ Aucun checkpoint disponible — abandon")
        return

    # Encoder principal = 1.0 (ou le meilleur disponible)
    main_enc = encoders.get("1.0") or next(iter(encoders.values()))

    # ── TEST 1 ─────────────────────────────────────────────────────────────────
    test1_checkpoint_comparison(
        encoders, stmd_pairs,
        out_base / "checkpoint_comparison",
    )

    # ── TEST 2 ─────────────────────────────────────────────────────────────────
    ntt_results = test2_ntt(main_enc, stmd_pairs, device)
    print_ntt_table(ntt_results)

    # ── TEST 3 ─────────────────────────────────────────────────────────────────
    stab_results = test3_stability(main_enc, stmd_pairs, device)
    print_stability_table(stab_results)

    # ── TEST 4 ─────────────────────────────────────────────────────────────────
    test4_pca_comparison(
        main_enc, stmd_pairs, kaust_pairs,
        out_base / "stage_comparison",
        device,
    )

    # ── RECOMMANDATION ─────────────────────────────────────────────────────────
    best_stage, best_sid = recommend(ntt_results, stab_results)

    # ── Sauvegarde JSON ────────────────────────────────────────────────────────
    result_json = {
        "ntt":      {s: ntt_results.get(s) for s in STAGES_TEST},
        "stability": {s: stab_results.get(s) for s in STAGES_TEST},
        "recommended_stage": best_stage,
        "checkpoints_loaded": list(encoders.keys()),
    }
    out_json = out_base / "scores_validation.json"
    with open(out_json, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  scores_validation.json → {out_json.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
