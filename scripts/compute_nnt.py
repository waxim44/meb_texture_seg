"""
compute_nnt.py — Calcul du score NNT (Nearest Texture Neighbor)
par stage de TextureSAM sur le dataset STMD.

Usage :
    python scripts/compute_nnt.py
"""

import sys
import json
import zipfile
import tempfile
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Chemins ────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).resolve()
_ROOT   = _HERE.parents[1]
_SAM2   = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Constantes ─────────────────────────────────────────────────────────────────
CKPT_PATH   = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
CKPT_DIR    = _ROOT / "checkpoints" / "sam2.1_hiera_small_1"
IMG_DIR     = _ROOT / "data" / "raw" / "stmd" / "images"
LBL_DIR     = _ROOT / "data" / "raw" / "stmd" / "labels"
OUT_DIR     = _ROOT / "outputs" / "nnt_results"
IMG_SIZE    = 1024
MIN_PATCHES = 10
MAX_SAMPLE  = 20
SEED        = 42

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Résolution feature par stage
STAGE_RES = {
    "stage_1": 256,
    "stage_2": 128,
    "stage_3": 64,
    "stage_4": 32,
}
# Mapping conv FPN → stage
CONV_TO_STAGE = {0: "stage_4", 1: "stage_3", 2: "stage_2", 3: "stage_1"}


# ── Construction du modèle ─────────────────────────────────────────────────────

def _build_image_encoder() -> ImageEncoder:
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


def _zip_dir_to_pt(ckpt_dir: Path) -> str:
    """Re-zippe un répertoire checkpoint PyTorch en fichier .pt temporaire."""
    archive_dir = ckpt_dir / "archive" if (ckpt_dir / "archive").is_dir() else ckpt_dir
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_STORED) as zf:
        for fp in sorted(archive_dir.rglob("*")):
            if fp.is_file():
                info = zipfile.ZipInfo(str(fp.relative_to(archive_dir.parent)))
                info.date_time = (1980, 1, 1, 0, 0, 0)
                with open(fp, "rb") as fh:
                    zf.writestr(info, fh.read())
    return tmp.name


def load_model(device: str) -> ImageEncoder:
    encoder = _build_image_encoder()

    # Charger le state dict
    tmp_path = None
    if CKPT_PATH.is_file():
        sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    elif CKPT_DIR.is_dir():
        tmp_path = _zip_dir_to_pt(CKPT_DIR)
        sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"Checkpoint introuvable : {CKPT_PATH} / {CKPT_DIR}")

    if tmp_path:
        os.unlink(tmp_path)

    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [WARN] Checkpoint partiel ({len(missing)} manquantes, "
              f"{len(unexpected)} inattendues)")
    else:
        print("  [OK] Checkpoint chargé")

    return encoder.to(device).eval()


# ── Hooks ──────────────────────────────────────────────────────────────────────

def register_hooks(encoder: ImageEncoder) -> dict:
    features = {s: None for s in CONV_TO_STAGE.values()}
    handles = []
    for conv_idx, stage_name in CONV_TO_STAGE.items():
        def _hook(module, inp, out, _name=stage_name):
            features[_name] = out.detach().cpu()
        h = encoder.neck.convs[conv_idx].register_forward_hook(_hook)
        handles.append(h)
    return features, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Prétraitement ──────────────────────────────────────────────────────────────

def preprocess_image(img_path: Path, device: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0   # (H, W, 3)
    x = x.permute(2, 0, 1)                                # (3, H, W)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device)                      # (1, 3, H, W)


def resize_labels(lbl_arr: np.ndarray, size: int) -> np.ndarray:
    """Resize label map en nearest-neighbor → préserve les IDs discrets."""
    img = Image.fromarray(lbl_arr.astype(np.uint8))
    img = img.resize((size, size), Image.NEAREST)
    return np.array(img)


# ── Calcul NNT pour une image ──────────────────────────────────────────────────

def _cosine_dist_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """
    Distance cosine moyenne entre deux ensembles de vecteurs normalisés L2.
    A : (N, D), B : (M, D)  — déjà normalisés.
    Retourne la moyenne sur toutes les paires (i, j).
    """
    # dot product → (N, M)
    sims = A @ B.T
    # cosine distance = 1 - cosine similarity
    dists = 1.0 - sims
    return float(dists.mean())


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return vecs / norms


def _sample(vecs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(vecs) <= n:
        return vecs
    idx = rng.choice(len(vecs), size=n, replace=False)
    return vecs[idx]


def compute_nnt_single_image(
    features_stage: np.ndarray,   # (H, W, 256)
    labels_resized: np.ndarray,   # (H, W) entiers
    min_patches: int = MIN_PATCHES,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Retourne (nnt, d_intra_mean, d_inter_mean, n_classes_used).
    Retourne (None, None, None, 0) si pas assez de classes valides.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    H, W, _ = features_stage.shape
    W_half = W // 2

    left_feat  = features_stage[:, :W_half, :]   # (H, W/2, 256)
    right_feat = features_stage[:, W_half:, :]
    left_mask  = labels_resized[:, :W_half]
    right_mask = labels_resized[:, W_half:]

    classes = np.unique(labels_resized)
    if len(classes) < 2:
        return None, None, None, 0

    # ── ÉTAPE B : d_intra par classe ──────────────────────────────────────────
    d_intra_list = []
    valid_classes = []

    for c in classes:
        pl = left_feat[left_mask == c].reshape(-1, 256)   # (N_l, 256)
        pr = right_feat[right_mask == c].reshape(-1, 256)  # (N_r, 256)

        if len(pl) < min_patches or len(pr) < min_patches:
            continue

        pl = _l2_normalize(_sample(pl, MAX_SAMPLE, rng))
        pr = _l2_normalize(_sample(pr, MAX_SAMPLE, rng))

        d = _cosine_dist_matrix(pl, pr)
        d_intra_list.append(d)
        valid_classes.append(c)

    if len(valid_classes) < 2:
        return None, None, None, len(valid_classes)

    d_intra_mean = float(np.mean(d_intra_list))

    # ── ÉTAPE C : d_inter entre paires de classes ─────────────────────────────
    d_inter_list = []
    all_feats_flat = features_stage.reshape(-1, 256)   # (H*W, 256)
    all_labels_flat = labels_resized.reshape(-1)

    for i in range(len(valid_classes)):
        for j in range(i + 1, len(valid_classes)):
            c1, c2 = valid_classes[i], valid_classes[j]
            p1 = all_feats_flat[all_labels_flat == c1]
            p2 = all_feats_flat[all_labels_flat == c2]

            p1 = _l2_normalize(_sample(p1, MAX_SAMPLE, rng))
            p2 = _l2_normalize(_sample(p2, MAX_SAMPLE, rng))

            d = _cosine_dist_matrix(p1, p2)
            d_inter_list.append(d)

    d_inter_mean = float(np.mean(d_inter_list))

    # ── ÉTAPE D : NNT ─────────────────────────────────────────────────────────
    if d_intra_mean < 1e-8:
        return None, None, None, len(valid_classes)

    nnt = d_inter_mean / d_intra_mean
    return nnt, d_intra_mean, d_inter_mean, len(valid_classes)


# ── Boucle principale ──────────────────────────────────────────────────────────

def main():
    print()
    print("════════════════════════════════════════════════════════════")
    print("  Compute NNT — TextureSAM sur STMD")
    print("════════════════════════════════════════════════════════════")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    # Paires (image, label)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    pairs = []
    for img_path in sorted(IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        lbl_path = LBL_DIR / (img_path.stem + ".png")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    print(f"  Images STMD avec GT : {len(pairs)}")
    if not pairs:
        print("  ERREUR : aucune paire trouvée.")
        return

    # Charger le modèle
    print()
    print("  Chargement du modèle …")
    encoder = load_model(device)

    features, handles = register_hooks(encoder)

    # Résultats par stage
    results_per_image: dict[str, list] = defaultdict(list)  # stage → list of per-image dicts
    rng = np.random.default_rng(SEED)

    print()
    print("  Extraction et calcul NNT …")
    n_classes_per_img = []

    with torch.no_grad():
        for img_path, lbl_path in tqdm(pairs, unit="img"):
            # Charger et forward
            x = preprocess_image(img_path, device)
            _ = encoder(x)

            # Charger le GT
            lbl_arr = np.array(Image.open(lbl_path))
            classes_in_img = np.unique(lbl_arr)
            n_classes_per_img.append(len(classes_in_img))

            if len(classes_in_img) < 2:
                continue

            # NNT par stage
            for stage_name, feat_tensor in features.items():
                if feat_tensor is None:
                    continue
                feat_np = feat_tensor[0].permute(1, 2, 0).numpy()   # (H, W, 256)
                H = feat_np.shape[0]
                lbl_resized = resize_labels(lbl_arr, H)

                nnt, d_intra, d_inter, n_cls = compute_nnt_single_image(
                    feat_np, lbl_resized, rng=rng
                )
                if nnt is None:
                    continue

                results_per_image[stage_name].append({
                    "image": img_path.name,
                    "nnt": nnt,
                    "d_intra": d_intra,
                    "d_inter": d_inter,
                    "n_classes": n_cls,
                })

    remove_hooks(handles)

    # ── Agrégation ────────────────────────────────────────────────────────────
    summary = {}
    stage_order = ["stage_1", "stage_2", "stage_3", "stage_4"]

    for stage in stage_order:
        entries = results_per_image.get(stage, [])
        if not entries:
            summary[stage] = None
            continue
        nnts    = [e["nnt"]     for e in entries]
        d_intras = [e["d_intra"] for e in entries]
        d_inters = [e["d_inter"] for e in entries]
        summary[stage] = {
            "nnt_mean":     float(np.mean(nnts)),
            "nnt_std":      float(np.std(nnts)),
            "d_intra_mean": float(np.mean(d_intras)),
            "d_inter_mean": float(np.mean(d_inters)),
            "n_images":     len(entries),
        }

    mean_classes = float(np.mean(n_classes_per_img)) if n_classes_per_img else 0

    # ── Affichage ──────────────────────────────────────────────────────────────
    print()
    print("════════════════════════════════════════════════════════════")
    print("  Résultats NNT détaillés")
    print("════════════════════════════════════════════════════════════")
    print(f"  Images traitées    : {len(pairs)}")
    print(f"  Classes moy/image  : {mean_classes:.2f}")
    print()

    # Trouver le meilleur stage (NNT max)
    valid_stages = [(s, summary[s]["nnt_mean"]) for s in stage_order if summary.get(s)]
    best_stage   = max(valid_stages, key=lambda x: x[1])[0] if valid_stages else None

    print("  d_intra / d_inter par stage :")
    for stage in stage_order:
        s = summary.get(stage)
        if s is None:
            continue
        print(f"    {stage} : d_intra={s['d_intra_mean']:.4f}  "
              f"d_inter={s['d_inter_mean']:.4f}")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              Nearest Texture Neighbor Test               ║")
    print("║                   Dataset : STMD                        ║")
    print("║           Checkpoint : sam2.1_hiera_small_1.pt          ║")
    print("╠═════════╦══════════════╦═══════════╦════════════════════╣")
    print("║ Stage   ║  NNT Score   ║  Std      ║  Verdict           ║")
    print("╠═════════╬══════════════╬═══════════╬════════════════════╣")

    for stage in stage_order:
        s = summary.get(stage)
        if s is None:
            row = f"║ {stage:<7} ║    N/A       ║  N/A      ║  N/A               ║"
            print(row)
            continue

        nnt  = s["nnt_mean"]
        std  = s["nnt_std"]
        ok   = "✅" if nnt > 1.0 else "❌"
        star = " ★" if stage == best_stage else "  "
        verdict = f"{ok} texture{star}" if nnt > 1.0 else f"{ok} position{star}"

        print(f"║ {stage:<7} ║   {nnt:7.4f}    ║  ±{std:.3f}   ║  {verdict:<17} ║")

    print("╚═════════╩══════════════╩═══════════╩════════════════════╝")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nnt_scores_path     = OUT_DIR / "nnt_scores.json"
    nnt_per_image_path  = OUT_DIR / "nnt_per_image.json"

    with open(nnt_scores_path, "w") as f:
        json.dump(summary, f, indent=2)

    per_image_data = {
        stage: results_per_image.get(stage, [])
        for stage in stage_order
    }
    with open(nnt_per_image_path, "w") as f:
        json.dump(per_image_data, f, indent=2)

    print()
    print(f"  → {nnt_scores_path.relative_to(_ROOT)}")
    print(f"  → {nnt_per_image_path.relative_to(_ROOT)}")
    print()


if __name__ == "__main__":
    main()
