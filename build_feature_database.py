"""
build_feature_database.py — Construit une base HDF5 de toutes les
représentations internes de TextureSAM pour chaque patch annoté MEB.

Usage :
    python build_feature_database.py [--config CONFIG_JSON] [--output OUTPUT_H5]
                                     [--img-dir IMG_DIR] [--annot ANNOT]
                                     [--checkpoint CHECKPOINT]

ANNOT peut être :
  - un fichier .xlsx (format PatchTagger : colonnes Image_name, x_min, x_max,
    y_min, y_max, category — convention x=row, y=col → corrigée en interne)
  - un dossier contenant des .json par image (liste de dicts avec x_min, y_min,
    x_max, y_max, category — convention image standard)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
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
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Normalisation ImageNet ────────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

IMG_SIZE = 1024

# ── Architecture Hiera Small ──────────────────────────────────────────────────
BLOCK_INFO = {
    0:  {"stage": 1, "dim": 96},
    1:  {"stage": 2, "dim": 192},
    2:  {"stage": 2, "dim": 192},
    3:  {"stage": 3, "dim": 384},
    4:  {"stage": 3, "dim": 384},
    5:  {"stage": 3, "dim": 384},
    6:  {"stage": 3, "dim": 384},
    7:  {"stage": 3, "dim": 384},
    8:  {"stage": 3, "dim": 384},
    9:  {"stage": 3, "dim": 384},
    10: {"stage": 3, "dim": 384},
    11: {"stage": 3, "dim": 384},
    12: {"stage": 3, "dim": 384},
    13: {"stage": 3, "dim": 384},
    14: {"stage": 4, "dim": 768},
    15: {"stage": 4, "dim": 768},
}
FPN_INFO = {
    "stage_1_fpn": 256,  # neck.convs[3]
    "stage_2_fpn": 256,  # neck.convs[2]
    "stage_3_fpn": 256,  # neck.convs[1]
    "stage_4_fpn": 256,  # neck.convs[0]
}

ALL_KEYS = {f"block_{i}": BLOCK_INFO[i]["dim"] for i in range(16)}
ALL_KEYS.update(FPN_INFO)

# ── Defaults ──────────────────────────────────────────────────────────────────
_ROOT_DEFAULTS = {
    "config":     _HERE / "PatchTagger_Output" / "config" / "config.json",
    "img_dir":    _HERE / "PatchTagger_Output" / "full_images",
    "annot":      _HERE / "PatchTagger_Output" / "categories.xlsx",
    "output":     _HERE / "data" / "feature_database" / "database_meb.h5",
    "checkpoint": _HERE / "checkpoints" / "sam2.1_hiera_small_1.pt",
}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement config
# ─────────────────────────────────────────────────────────────────────────────

def load_categories(config_json: Path) -> dict:
    with open(config_json) as f:
        cfg = json.load(f)
    return {
        int(k): {"name": v["name"], "color": v["color"]}
        for k, v in cfg["available_categories"].items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chargement annotations
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_xlsx(xlsx_path: Path, categories: dict) -> dict:
    """
    Lit categories.xlsx (format PatchTagger).
    Convention Excel : x = row (vertical 0-768), y = col (horizontal 0-1280).
    On remet en convention image standard : x = horizontal, y = vertical.
    Retourne {img_name: [patch_dict, ...]}.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas requis pour lire un fichier .xlsx — pip install pandas openpyxl")

    df = pd.read_excel(xlsx_path)
    required = {"Image_name", "x_min", "x_max", "y_min", "y_max", "category"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Colonnes manquantes dans {xlsx_path.name}: {missing}")

    index = defaultdict(list)
    skipped = 0
    for _, row in df.iterrows():
        cat_id = int(row["category"])
        if cat_id not in categories:
            skipped += 1
            continue
        img_name = str(row["Image_name"])
        # Swap coordonnées : Excel stocke x=row, y=col
        index[img_name].append({
            "x_min":         int(row["y_min"]),
            "x_max":         int(row["y_max"]),
            "y_min":         int(row["x_min"]),
            "y_max":         int(row["x_max"]),
            "category":      cat_id,
            "category_name": categories[cat_id]["name"],
        })

    if skipped:
        log.warning("%d lignes ignorées (category_id absent du config)", skipped)
    return dict(index)


def _load_from_json_dir(json_dir: Path, categories: dict) -> dict:
    """
    Lit un dossier de fichiers JSON par image.
    Chaque fichier = liste de dicts {x_min, y_min, x_max, y_max, category}.
    Convention déjà en espace image (x=horizontal, y=vertical).
    Retourne {img_name: [patch_dict, ...]}.
    """
    index = {}
    for jf in sorted(json_dir.glob("*.json")):
        with open(jf) as f:
            patches = json.load(f)
        valid = []
        for p in patches:
            cat_id = int(p.get("category", -1))
            if cat_id not in categories:
                continue
            valid.append({
                "x_min":         int(p["x_min"]),
                "x_max":         int(p["x_max"]),
                "y_min":         int(p["y_min"]),
                "y_max":         int(p["y_max"]),
                "category":      cat_id,
                "category_name": categories[cat_id]["name"],
            })
        if valid:
            img_name = jf.stem + ".tif"  # convention : même nom que l'image
            index[img_name] = valid
    return index


def load_annotations(annot_path: Path, categories: dict) -> dict:
    if annot_path.is_file() and annot_path.suffix.lower() in (".xlsx", ".xls"):
        return _load_from_xlsx(annot_path, categories)
    elif annot_path.is_dir():
        return _load_from_json_dir(annot_path, categories)
    else:
        raise ValueError(
            f"--annot doit être un fichier .xlsx ou un dossier JSON : {annot_path}"
        )


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


def _zip_dir(ckpt_dir: Path) -> str:
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


def load_model(ckpt_path: Path, device: str) -> ImageEncoder:
    encoder = build_image_encoder()

    tmp_path = None
    if ckpt_path.is_file():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    elif ckpt_path.is_dir():
        tmp_path = _zip_dir(ckpt_path)
        sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
    else:
        log.warning("Checkpoint introuvable : %s — poids aléatoires utilisés", ckpt_path)
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
    """
    Pose les hooks sur les 16 blocs trunk et les 4 convs FPN.
    Retourne (captured, handles).
    """
    captured = {}
    handles  = []

    # Blocs trunk — sortie shape : (B, H, W, C)
    for i, block in enumerate(encoder.trunk.blocks):
        def _block_hook(m, inp, out, idx=i):
            captured[f"block_{idx}"] = out.detach()
        handles.append(block.register_forward_hook(_block_hook))

    # Convs FPN — neck.convs[0]=stage_4, [1]=stage_3, [2]=stage_2, [3]=stage_1
    # sortie shape : (B, 256, H, W) → permutée en (B, H, W, 256)
    for conv_idx, stage_num in enumerate([4, 3, 2, 1]):
        key = f"stage_{stage_num}_fpn"
        def _fpn_hook(m, inp, out, k=key):
            captured[k] = out.detach().permute(0, 2, 3, 1)
        handles.append(encoder.neck.convs[conv_idx].register_forward_hook(_fpn_hook))

    return captured, handles


def remove_hooks(handles: list):
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Prétraitement image
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img_path: Path, device: str) -> tuple[torch.Tensor, int, int]:
    """
    Charge, convertit en RGB (images MEB en niveaux de gris), redimensionne
    à 1024×1024. Retourne (tensor, orig_H, orig_W).
    """
    img = Image.open(img_path)
    orig_w, orig_h = img.size  # PIL : (width, height)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device), orig_h, orig_w


# ─────────────────────────────────────────────────────────────────────────────
# Extraction features par patch
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch_features(
    captured: dict,
    patch: dict,
    orig_H: int,
    orig_W: int,
) -> dict:
    """
    Pour chaque feature map dans captured, average-pool la région du patch
    et L2-normalise le vecteur résultant.

    patch contient x_min/x_max (horizontal) et y_min/y_max (vertical)
    en coordonnées image originale.
    """
    features = {}
    for key, feat_map in captured.items():
        # feat_map : (B, H_feat, W_feat, C) après permutation des hooks
        feat = feat_map[0]  # (H_feat, W_feat, C)
        H_feat, W_feat, C = feat.shape

        scale_x = W_feat / orig_W
        scale_y = H_feat / orig_H

        fx1 = max(0, int(patch["x_min"] * scale_x))
        fy1 = max(0, int(patch["y_min"] * scale_y))
        fx2 = min(W_feat, max(fx1 + 1, int(patch["x_max"] * scale_x)))
        fy2 = min(H_feat, max(fy1 + 1, int(patch["y_max"] * scale_y)))

        # Région dégénérée : prendre le pixel le plus proche
        if fx2 - fx1 < 1:
            fx1 = min(fx1, W_feat - 1)
            fx2 = fx1 + 1
        if fy2 - fy1 < 1:
            fy1 = min(fy1, H_feat - 1)
            fy2 = fy1 + 1

        region = feat[fy1:fy2, fx1:fx2, :]  # (h, w, C)
        vec = region.mean(dim=(0, 1)).cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        features[key] = vec
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Création du fichier HDF5
# ─────────────────────────────────────────────────────────────────────────────

def create_h5(output_h5: Path, all_patches: list):
    N = len(all_patches)
    with h5py.File(output_h5, "w") as h5:
        meta = h5.create_group("metadata")
        meta.create_dataset(
            "image_names",
            data=np.array(
                [p["image_name"].encode("utf-8") for p in all_patches], dtype="S200"
            ),
        )
        meta.create_dataset(
            "positions",
            data=np.array(
                [[p["x_min"], p["y_min"], p["x_max"], p["y_max"]] for p in all_patches],
                dtype=np.float32,
            ),
        )
        meta.create_dataset(
            "category_ids",
            data=np.array([p["category_id"] for p in all_patches], dtype=np.int16),
        )
        meta.create_dataset(
            "category_names",
            data=np.array(
                [p["category_name"].encode("utf-8") for p in all_patches], dtype="S100"
            ),
        )

        feats_grp = h5.create_group("features")
        for key, dim in ALL_KEYS.items():
            feats_grp.create_dataset(
                key,
                shape=(N, dim),
                dtype=np.float32,
                chunks=(min(256, N), dim),
            )
    log.info("HDF5 créé : %s  (%d patches, %d feature layers)", output_h5.name, N, len(ALL_KEYS))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def build_database(
    config_json: Path,
    img_dir: Path,
    annot_path: Path,
    output_h5: Path,
    ckpt_path: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device : %s", device)

    # ── Catégories ─────────────────────────────────────────────────────────────
    categories = load_categories(config_json)
    log.info("%d catégories chargées depuis %s", len(categories), config_json.name)

    # ── Annotations ────────────────────────────────────────────────────────────
    image_index = load_annotations(annot_path, categories)
    log.info("%d images avec annotations", len(image_index))

    # Construire la liste ordonnée de tous les patches
    all_patches = []
    images_present = []
    for img_name in sorted(image_index.keys()):
        img_path = img_dir / img_name
        if not img_path.exists():
            log.warning("Image introuvable : %s — ignorée", img_name)
            continue
        images_present.append(img_name)
        for p in image_index[img_name]:
            all_patches.append({
                "image_name":    img_name,
                "x_min":         p["x_min"],
                "y_min":         p["y_min"],
                "x_max":         p["x_max"],
                "y_max":         p["y_max"],
                "category_id":   p["category"],
                "category_name": p["category_name"],
            })

    N_PATCHES = len(all_patches)
    log.info("%d patches annotés sur %d images", N_PATCHES, len(images_present))

    if N_PATCHES == 0:
        log.error("Aucun patch trouvé — vérifier IMG_DIR et ANNOT.")
        return

    # ── HDF5 — création et pré-allocation ──────────────────────────────────────
    create_h5(output_h5, all_patches)

    # ── Modèle + hooks ─────────────────────────────────────────────────────────
    encoder = load_model(ckpt_path, device)
    captured, handles = register_hooks(encoder)

    # ── Remplissage ────────────────────────────────────────────────────────────
    patch_idx = 0
    n_errors  = 0

    # Regrouper patches par image pour ne faire qu'un forward pass par image
    patches_by_image = defaultdict(list)
    for p in all_patches:
        patches_by_image[p["image_name"]].append(p)

    with h5py.File(output_h5, "a") as h5:
        for img_name in tqdm(images_present, desc="Images", unit="img"):
            img_path = img_dir / img_name

            try:
                tensor, orig_H, orig_W = preprocess(img_path, device)
            except Exception as e:
                n_img_patches = len(patches_by_image[img_name])
                log.error("Erreur prétraitement %s : %s — %d patches ignorés", img_name, e, n_img_patches)
                patch_idx += n_img_patches
                n_errors  += n_img_patches
                continue

            captured.clear()
            try:
                with torch.no_grad():
                    encoder(tensor)
            except Exception as e:
                n_img_patches = len(patches_by_image[img_name])
                log.error("Erreur forward %s : %s — %d patches ignorés", img_name, e, n_img_patches)
                patch_idx += n_img_patches
                n_errors  += n_img_patches
                continue

            img_patches = patches_by_image[img_name]
            for p in img_patches:
                try:
                    feats = extract_patch_features(captured, p, orig_H, orig_W)
                    for key, vec in feats.items():
                        h5["features"][key][patch_idx] = vec
                except Exception as e:
                    log.error("Erreur patch %s @(%d,%d) : %s", img_name, p["x_min"], p["y_min"], e)
                    n_errors += 1
                patch_idx += 1

            log.debug("%s → %d patches extraits", img_name, len(img_patches))

    remove_hooks(handles)

    if n_errors:
        log.warning("%d erreurs au total (features laissées à zéro)", n_errors)

    # ── Vérification finale ────────────────────────────────────────────────────
    log.info("")
    log.info("=== Database MEB ===")
    with h5py.File(output_h5, "r") as h5:
        n_total = h5["metadata/category_ids"].shape[0]
        log.info("Patches totaux : %d", n_total)
        log.info("Features disponibles :")
        for key in sorted(h5["features"].keys()):
            shape = h5["features"][key].shape
            log.info("  %-15s : %s", key, shape)

        cats = h5["metadata/category_ids"][:]
        log.info("")
        log.info("Distribution par catégorie :")
        for cat_id in sorted(np.unique(cats)):
            n = int((cats == cat_id).sum())
            name = categories.get(int(cat_id), {}).get("name", "?")
            log.info("  Cat %2d  %-28s : %d patches", cat_id, name, n)

    # ── Sauvegarder categories.json ────────────────────────────────────────────
    cat_json = output_h5.with_name("categories.json")
    with open(cat_json, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in categories.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("")
    log.info("→ %s", output_h5)
    log.info("→ %s", cat_json)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",     default=str(_ROOT_DEFAULTS["config"]),
                   help="Chemin vers config.json")
    p.add_argument("--img-dir",    default=str(_ROOT_DEFAULTS["img_dir"]),
                   help="Dossier des images .tif")
    p.add_argument("--annot",      default=str(_ROOT_DEFAULTS["annot"]),
                   help="Fichier .xlsx ou dossier JSON d'annotations")
    p.add_argument("--output",     default=str(_ROOT_DEFAULTS["output"]),
                   help="Fichier HDF5 de sortie")
    p.add_argument("--checkpoint", default=str(_ROOT_DEFAULTS["checkpoint"]),
                   help="Chemin du checkpoint (.pt ou répertoire archive)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_database(
        config_json = Path(args.config),
        img_dir     = Path(args.img_dir),
        annot_path  = Path(args.annot),
        output_h5   = Path(args.output),
        ckpt_path   = Path(args.checkpoint),
    )
