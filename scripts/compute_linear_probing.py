"""
compute_linear_probing.py — Linear Probing Accuracy par stage de TextureSAM sur STMD.

Usage :
    python scripts/compute_linear_probing.py
"""

import sys
import json
import time
import zipfile
import tempfile
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ── Chemins ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
_SAM2 = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Constantes ─────────────────────────────────────────────────────────────────
CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
CKPT_DIR  = _ROOT / "checkpoints" / "sam2.1_hiera_small_1"
IMG_DIR   = _ROOT / "data" / "raw" / "stmd" / "images"
LBL_DIR   = _ROOT / "data" / "raw" / "stmd" / "labels"
OUT_DIR   = _ROOT / "outputs" / "linear_probing"
IMG_SIZE  = 1024
SEED      = 42

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CONV_TO_STAGE = {0: "stage_4", 1: "stage_3", 2: "stage_2", 3: "stage_1"}
STAGE_ORDER   = ["stage_1", "stage_2", "stage_3", "stage_4"]
STAGE_RES     = {"stage_1": 256, "stage_2": 128, "stage_3": 64, "stage_4": 32}

# Quantization des labels STMD :
# les valeurs pixel varient légèrement d'une image à l'autre
# (ex : 55/57/58 = même classe) → on les regroupe par seuils
# Bins : [30, 90, 145, 215] → classes {0, 1, 2, 3, 4}
#   0       → 0 (fond)
#   55–79   → 1
#   121–123 → 2
#   163–186 → 3
#   247–249 → 4
_LABEL_BINS = np.array([30, 90, 145, 215])


def quantize_labels(arr: np.ndarray) -> np.ndarray:
    """Mappe les valeurs pixel STMD vers des IDs de classe 0..4 stables."""
    return np.digitize(arr, _LABEL_BINS).astype(np.int32)


# ── Modèle ─────────────────────────────────────────────────────────────────────

def _build_image_encoder() -> ImageEncoder:
    trunk = Hiera(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
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
    tmp_path = None
    if CKPT_PATH.is_file():
        sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    elif CKPT_DIR.is_dir():
        tmp_path = _zip_dir_to_pt(CKPT_DIR)
        sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"Checkpoint introuvable : {CKPT_PATH}")
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

def register_hooks(encoder: ImageEncoder):
    features = {s: None for s in CONV_TO_STAGE.values()}
    handles = []
    for conv_idx, stage_name in CONV_TO_STAGE.items():
        def _hook(module, inp, out, _name=stage_name):
            features[_name] = out.detach().cpu()
        handles.append(encoder.neck.convs[conv_idx].register_forward_hook(_hook))
    return features, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Extraction ─────────────────────────────────────────────────────────────────

def _preprocess(img_path: Path, device: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device)


def extract_features_and_labels(
    img_path: Path,
    lbl_path: Path,
    stage_name: str,
    encoder: ImageEncoder,
    features: dict,
    device: str,
):
    """
    Retourne (X, y) où X est (N, 256) normalisé L2 et y est (N,) entiers.
    Lève ValueError si l'image n'a qu'une seule classe.
    """
    x = _preprocess(img_path, device)
    with torch.no_grad():
        _ = encoder(x)

    feat_tensor = features[stage_name]
    if feat_tensor is None:
        raise RuntimeError(f"Hook n'a rien capturé pour {stage_name}")

    # (1, 256, H, W) → (H, W, 256)
    if feat_tensor.dim() == 4:
        feat_np = feat_tensor[0].permute(1, 2, 0).numpy()
    else:
        feat_np = feat_tensor.numpy()

    H, W, _ = feat_np.shape

    lbl_arr = np.array(Image.open(lbl_path))
    lbl_resized = np.array(
        Image.fromarray(lbl_arr.astype(np.uint8)).resize((W, H), Image.NEAREST)
    )
    lbl_resized = quantize_labels(lbl_resized)

    assert feat_np.shape[:2] == lbl_resized.shape, (
        f"Shape mismatch: feat={feat_np.shape[:2]} lbl={lbl_resized.shape}"
    )

    classes = np.unique(lbl_resized)
    if len(classes) < 2:
        raise ValueError(f"Image {img_path.name} n'a qu'une seule classe : {classes}")

    X = feat_np.reshape(-1, 256).astype(np.float32)
    y = lbl_resized.reshape(-1).astype(np.int32)
    X = normalize(X, norm="l2")
    return X, y


# ── Collecte par split ─────────────────────────────────────────────────────────

def collect_split(
    pairs: list,
    stage_name: str,
    encoder: ImageEncoder,
    features: dict,
    device: str,
    desc: str,
):
    Xs, ys = [], []
    skipped = 0
    for img_path, lbl_path in tqdm(pairs, desc=f"    {desc}", unit="img", leave=False):
        try:
            X, y = extract_features_and_labels(
                img_path, lbl_path, stage_name, encoder, features, device
            )
            Xs.append(X)
            ys.append(y)
        except ValueError as e:
            print(f"\n  [WARN] {e} — ignorée")
            skipped += 1
    if not Xs:
        raise RuntimeError(f"Aucune image valide dans {desc}")
    return np.concatenate(Xs), np.concatenate(ys), len(Xs), skipped


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    print("════════════════════════════════════════════════════════════")
    print("  Linear Probing Accuracy — TextureSAM sur STMD")
    print("════════════════════════════════════════════════════════════")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    pairs = []
    for img_path in sorted(IMG_DIR.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        lbl_path = LBL_DIR / (img_path.stem + ".png")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    print(f"  Images STMD avec GT : {len(pairs)}")

    # Split 80/20 par image
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.8)
    train_pairs = [pairs[i] for i in idx[:n_train]]
    test_pairs  = [pairs[i] for i in idx[n_train:]]
    print(f"  Split : {len(train_pairs)} train / {len(test_pairs)} test")

    print()
    print("  Chargement du modèle …")
    encoder = load_model(device)
    features, handles = register_hooks(encoder)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    table_rows = []

    for stage_name in STAGE_ORDER:
        print()
        print(f"  ── {stage_name} ({STAGE_RES[stage_name]}×{STAGE_RES[stage_name]}) ──")
        t0 = time.time()

        # Extraction
        X_train, y_train, n_tr_imgs, sk_tr = collect_split(
            train_pairs, stage_name, encoder, features, device, "train"
        )
        X_test, y_test, n_te_imgs, sk_te = collect_split(
            test_pairs, stage_name, encoder, features, device, "test "
        )

        classes_train = np.unique(y_train)
        n_classes = len(classes_train)
        baseline = 1.0 / n_classes

        print(f"    Train : {n_tr_imgs} imgs, {len(X_train):,} vecteurs")
        print(f"    Test  : {n_te_imgs} imgs, {len(X_test):,} vecteurs")
        print(f"    Classes dans train : {classes_train.tolist()} ({n_classes} classes)")

        # Classifieur
        print(f"    Entraînement LogReg …")
        clf = LogisticRegression(
            max_iter=1000,
            random_state=SEED,
            C=1.0,
            solver="lbfgs",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Évaluation
        accuracy = clf.score(X_test, y_test)
        y_pred   = clf.predict(X_test)
        report   = classification_report(y_test, y_pred, zero_division=0)

        elapsed = time.time() - t0
        print(f"    Accuracy : {accuracy * 100:.2f}%  |  baseline : {baseline * 100:.2f}%  "
              f"|  temps : {elapsed:.1f}s")

        verdict = "✅" if accuracy > 2 * baseline else "❌"
        table_rows.append((stage_name, accuracy, baseline, verdict, elapsed))

        # Sauvegarder report texte
        report_path = OUT_DIR / f"classification_report_{stage_name}.txt"
        header = (
            f"Linear Probing — {stage_name} "
            f"({STAGE_RES[stage_name]}×{STAGE_RES[stage_name]})\n"
            f"Checkpoint : sam2.1_hiera_small_1.pt\n"
            f"Dataset    : STMD\n"
            f"Accuracy   : {accuracy * 100:.2f}%  |  Baseline : {baseline * 100:.2f}%\n"
            f"{'─' * 60}\n"
        )
        with open(report_path, "w") as f:
            f.write(header + report)

        all_scores[stage_name] = {
            "accuracy":        round(accuracy, 6),
            "baseline":        round(baseline, 6),
            "n_train_vectors": int(len(X_train)),
            "n_test_vectors":  int(len(X_test)),
            "n_train_images":  int(n_tr_imgs),
            "n_test_images":   int(n_te_imgs),
            "n_classes":       int(n_classes),
            "elapsed_s":       round(elapsed, 2),
        }

    remove_hooks(handles)

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    best_stage = max(table_rows, key=lambda r: r[1])[0]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                  Linear Probing Accuracy                     ║")
    print("║                    Dataset : STMD                            ║")
    print("║           Checkpoint : sam2.1_hiera_small_1.pt              ║")
    print("╠═════════╦════════════════╦═══════════════════╦══════════════╣")
    print("║ Stage   ║  Accuracy (%)  ║  Baseline (%)     ║  Verdict     ║")
    print("╠═════════╬════════════════╬═══════════════════╬══════════════╣")
    for stage_name, acc, base, verd, elapsed in table_rows:
        star = " ★" if stage_name == best_stage else "  "
        print(f"║ {stage_name:<7} ║    {acc * 100:6.2f}%     ║     {base * 100:6.2f}%         ║  "
              f"{verd}{star}        ║")
    print("╚═════════╩════════════════╩═══════════════════╩══════════════╝")

    # ── Classification reports ─────────────────────────────────────────────────
    print()
    print("  Classification reports :")
    for stage_name in STAGE_ORDER:
        report_path = OUT_DIR / f"classification_report_{stage_name}.txt"
        if report_path.exists():
            print()
            print(f"  ── {stage_name} ──")
            print(report_path.read_text())

    # ── Sauvegarder JSON ──────────────────────────────────────────────────────
    scores_path = OUT_DIR / "linear_probing_scores.json"
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"  → {scores_path.relative_to(_ROOT)}")
    for stage_name in STAGE_ORDER:
        p = OUT_DIR / f"classification_report_{stage_name}.txt"
        print(f"  → {p.relative_to(_ROOT)}")
    print()


if __name__ == "__main__":
    main()
