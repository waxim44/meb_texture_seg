#!/usr/bin/env python3
"""
Extraction des grilles de vecteurs locaux (avec positions) pour les blocs
manquants du cache vote_analysis_cache/ : block_9, block_10, stage_3_fpn.
Réutilise exactement la méthode validée de notebooks/vote_analysis_patch_types.ipynb
(cellule 3) : hooks sur Hiera trunk blocks / FPN neck convs, région de patch
projetée sur la feature map, vecteurs L2-normalisés, rows/cols/feat_h/feat_w.
"""
import sys, pickle, tempfile, zipfile, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image as PILImage
import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("extract_missing")

ROOT      = Path("/home/aidouni/meb_texture_seg")
H5_PATH   = ROOT / "data/feature_database/database_meb_ouassim.h5"
IMG_DIR   = ROOT / "Image_Ouassim"
CACHE_DIR = ROOT / "vote_analysis_cache"
CKPT_PATH = ROOT / "checkpoints/sam2.1_hiera_small_1.pt"
CKPT_DIR  = ROOT / "checkpoints/sam2.1_hiera_small_1"

assert H5_PATH.name == "database_meb_ouassim.h5", f"H5 inattendu : {H5_PATH}"
print(f"H5 confirmé : {H5_PATH}")

MISSING_BLOCS = ["block_9", "block_10", "stage_3_fpn"]
TEXTURES = [1, 3, 4, 5, 6, 7, 9]
IMG_SIZE = 1024
SEED = 42

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

sys.path.insert(0, str(ROOT / "TextureSAM" / "sam2"))
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine


def _build_encoder():
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
                  global_att_blocks=(7, 10, 13),
                  window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True,
                                                            scale=None, temperature=10000),
                   d_model=256, backbone_channel_list=[768, 384, 192, 96],
                   kernel_size=1, stride=1, padding=0,
                   fpn_interp_model="nearest", fuse_type="sum", fpn_top_down_levels=[2, 3])
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def _load_model(device):
    enc = _build_encoder()
    tmp = None
    if CKPT_PATH.is_file():
        sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    elif CKPT_DIR.is_dir():
        arch = CKPT_DIR / "archive" if (CKPT_DIR / "archive").is_dir() else CKPT_DIR
        tf = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tf.close(); tmp = tf.name
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for fp in sorted(arch.rglob("*")):
                if fp.is_file():
                    info = zipfile.ZipInfo(str(fp.relative_to(arch.parent)))
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    with open(fp, "rb") as fh:
                        zf.writestr(info, fh.read())
        sd = torch.load(tmp, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError("Checkpoint introuvable")
    if tmp:
        import os; os.unlink(tmp)
    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    m, u = enc.load_state_dict(sd, strict=False)
    log.info("Checkpoint : %d manquants, %d inattendus", len(m), len(u))
    return enc.to(device).eval()


def _register_hooks(enc, blocs):
    captured, handles = {}, []
    for i, block in enumerate(enc.trunk.blocks):
        key = f"block_{i}"
        if key in blocs:
            def _bh(m, inp, out, k=key):
                captured[k] = out.detach()
            handles.append(block.register_forward_hook(_bh))
    fpn_map = {0: "stage_4_fpn", 1: "stage_3_fpn", 2: "stage_2_fpn", 3: "stage_1_fpn"}
    for ci, key in fpn_map.items():
        if key in blocs:
            def _fh(m, inp, out, k=key):
                captured[k] = out.detach().permute(0, 2, 3, 1)
            handles.append(enc.neck.convs[ci].register_forward_hook(_fh))
    return captured, handles


def _preprocess(img_path, device):
    img = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - torch.tensor(_MEAN)) / torch.tensor(_STD)
    return x.unsqueeze(0).to(device), orig_h, orig_w


def _patch_region(feat_hw, orig_h, orig_w, x_min, y_min, x_max, y_max):
    H_f, W_f = feat_hw
    sx = W_f / orig_w
    sy = H_f / orig_h
    fx1 = max(0, int(x_min * sx))
    fy1 = max(0, int(y_min * sy))
    fx2 = min(W_f, max(fx1 + 1, int(x_max * sx)))
    fy2 = min(H_f, max(fy1 + 1, int(y_max * sy)))
    return fy1, fy2, fx1, fx2


def _extract_local_vecs(feat_map, fy1, fy2, fx1, fx2):
    region = feat_map[fy1:fy2, fx1:fx2, :]
    h, w, C = region.shape
    vecs = region.reshape(-1, C).cpu().numpy().astype(np.float32)
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows = rr.ravel()
    cols = cc.ravel()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return vecs, rows, cols, h, w


to_extract = [b for b in MISSING_BLOCS if not (CACHE_DIR / f"vecs_{b}.pkl").exists()]
print(f"Blocs à extraire : {to_extract}")
if not to_extract:
    print("Rien à faire, cache déjà complet.")
    sys.exit(0)

with h5py.File(H5_PATH, "r") as f:
    all_cats = f["metadata/category_ids"][:]
    all_imgs = np.array([x.decode() for x in f["metadata/image_names"][:]])
    all_pos  = f["metadata/positions"][:]

mask = np.isin(all_cats, TEXTURES)
cats        = all_cats[mask]
imgs        = all_imgs[mask]
pos         = all_pos[mask].astype(int)
pids_global = np.where(mask)[0]
stems       = np.array([n.replace(".tif", "") for n in imgs])
N = int(mask.sum())
print(f"{N} patches, {len(TEXTURES)} textures, {len(np.unique(stems))} images")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
enc = _load_model(device)
captured, handles = _register_hooks(enc, to_extract)

patches_by_bloc = {b: [] for b in to_extract}
by_img = defaultdict(list)
for i in range(N):
    by_img[stems[i]].append(i)

unique_stems = sorted(by_img.keys())
for i_img, stem in enumerate(unique_stems):
    img_path = IMG_DIR / (stem + ".tif")
    if not img_path.exists():
        log.warning("[%d/%d] Image introuvable : %s.tif", i_img + 1, len(unique_stems), stem)
        continue
    tensor, orig_h, orig_w = _preprocess(img_path, device)

    captured.clear()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            enc(tensor)

    if not all(b in captured for b in to_extract):
        log.warning("Blocs manquants pour %s", stem)
        continue

    for i in by_img[stem]:
        x_min, y_min, x_max, y_max = pos[i]
        pid = int(pids_global[i])
        for b in to_extract:
            feat = captured[b][0]
            H_f, W_f, _ = feat.shape
            fy1, fy2, fx1, fx2 = _patch_region((H_f, W_f), orig_h, orig_w, x_min, y_min, x_max, y_max)
            vecs, rows, cols, fh, fw = _extract_local_vecs(feat, fy1, fy2, fx1, fx2)
            patches_by_bloc[b].append({
                "patch_id": pid, "texture": int(cats[i]), "image": stem,
                "vecs": vecs, "rows": rows, "cols": cols, "feat_h": fh, "feat_w": fw,
            })

    if (i_img + 1) % 10 == 0 or i_img == len(unique_stems) - 1:
        print(f"  [{i_img + 1}/{len(unique_stems)}] {stem}")

for h in handles:
    h.remove()

for b in to_extract:
    cf = CACHE_DIR / f"vecs_{b}.pkl"
    with open(cf, "wb") as fh:
        pickle.dump(patches_by_bloc[b], fh)
    sz_mb = cf.stat().st_size / 1e6
    nv_moy = sum(len(p["vecs"]) for p in patches_by_bloc[b]) // max(len(patches_by_bloc[b]), 1)
    print(f"  Cache sauvé : {cf.name}  ({len(patches_by_bloc[b])} patches, ~{nv_moy} vec/patch, {sz_mb:.0f} MB)")
