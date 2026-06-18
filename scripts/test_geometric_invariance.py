#!/usr/bin/env python3
"""
test_geometric_invariance.py
Tester si les features block_0 (TextureSAM SAM2 Hiera Small) sont invariantes
aux transformations géométriques : rotations 90°/180°/270° et flips H/V.

PRINCIPE : on transforme l'IMAGE ENTIÈRE avant le forward pass, puis on
recalcule la position du patch dans l'image transformée pour extraire
ses features — jamais le patch isolé.
"""

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

# ── Paramètres ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
SAM2_DIR     = ROOT / 'TextureSAM' / 'sam2'
DB_PATH      = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
CFG_PATH     = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
IMG_DIR      = ROOT / 'Image_Ouassim'
CHECKPOINT   = 'checkpoints/sam2.1_hiera_small_1.pt'
OUTPUT_DIR   = ROOT / 'outputs' / 'geometric_invariance'
SEED         = 42
N_PER_CAT    = 50
CATS_EXCLUDE = [2, 8, 10, 11, 12, 13]
MIN_N        = 30
TRANSFORMS   = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Résultats → {OUTPUT_DIR}')

# ── Config ─────────────────────────────────────────────────────────────────────
with open(CFG_PATH) as _f:
    _geo_cfg = json.load(_f)
CATEGORIES = {int(k): v['name']  for k, v in _geo_cfg['available_categories'].items()}
CAT_COLORS = {int(k): v['color'] for k, v in _geo_cfg['available_categories'].items()}

_geo_TLABELS = {
    'rot90'  : 'Rot 90°↻',
    'rot180' : 'Rot 180°',
    'rot270' : 'Rot 270°↻',
    'flip_h' : 'Flip ↔',
    'flip_v' : 'Flip ↕',
}

# ── HDF5 ───────────────────────────────────────────────────────────────────────
with h5py.File(DB_PATH, 'r') as _h5:
    _geo_IMAGE_NAMES  = _h5['metadata/image_names'][:]
    _geo_POSITIONS    = _h5['metadata/positions'][:]    # (N, 4) x1,y1,x2,y2
    _geo_CATEGORY_IDS = _h5['metadata/category_ids'][:].astype(int)
    _geo_X_all        = _h5['features']['block_0'][:]   # (N, 96)

_geo_EXCL_SET = set(CATS_EXCLUDE)
_geo_CATS_VALID = sorted(
    int(c) for c in np.unique(_geo_CATEGORY_IDS)
    if int(c) not in _geo_EXCL_SET
    and (_geo_CATEGORY_IDS == int(c)).sum() >= MIN_N
)
print(f'Catégories valides ({len(_geo_CATS_VALID)}) : '
      f'{[CATEGORIES[c] for c in _geo_CATS_VALID]}')

# ── Modèle SAM2 ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(SAM2_DIR / 'sam2'))
os.chdir(ROOT)

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir=str(SAM2_DIR / 'sam2' / 'configs'),
    version_base='1.2',
)
from sam2.build_sam import build_sam2

_geo_DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_geo_MODEL_CFG = 'sam2.1/sam2.1_hiera_s.yaml'
_geo_IMG_SIZE  = 1024
_geo_MEAN      = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_geo_STD       = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
print(f'Device : {_geo_DEVICE}')


def _geo_load_ckpt_sd(ckpt_path):
    _p = Path(ckpt_path)
    if _p.is_file():
        _sd = torch.load(_p, map_location='cpu', weights_only=False)
        return _sd.get('model', _sd)
    _arch = _p / 'archive' if (_p / 'archive').is_dir() else _p
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as _tmp:
        _tmp_path = _tmp.name
    with zipfile.ZipFile(_tmp_path, 'w', compression=zipfile.ZIP_STORED) as _zf:
        for _fp in _arch.rglob('*'):
            if _fp.is_file():
                _info = zipfile.ZipInfo(str(_fp.relative_to(_arch.parent)))
                _info.date_time = (1980, 1, 1, 0, 0, 0)
                with open(_fp, 'rb') as _fh:
                    _zf.writestr(_info, _fh.read())
    _sd = torch.load(_tmp_path, map_location='cpu', weights_only=False)
    os.unlink(_tmp_path)
    return _sd.get('model', _sd)


def _geo_load_model():
    _base = str(ROOT / 'checkpoints' / 'sam2.1_hiera_small')
    _m = build_sam2(_geo_MODEL_CFG, ckpt_path=None,
                    device=_geo_DEVICE, apply_postprocessing=False)
    _m.load_state_dict(_geo_load_ckpt_sd(_base), strict=False)
    _ck = str(ROOT / CHECKPOINT)
    if Path(_ck).resolve() != Path(_base).resolve():
        _miss, _unex = _m.load_state_dict(_geo_load_ckpt_sd(_ck), strict=False)
        print(f'Fine-tuned : missing={len(_miss)}  unexpected={len(_unex)}')
    _m.eval()
    return _m


print('Chargement du modèle...')
_geo_model = _geo_load_model()
print(f'Modèle chargé  ({sum(p.numel() for p in _geo_model.parameters())/1e6:.1f}M params)')

# ── Hook block_0 ───────────────────────────────────────────────────────────────
_geo_cap = {}


def _geo_hook_fn(_mod, _inp, _out):
    _geo_cap['feat'] = _out.detach()   # (B, H_feat, W_feat, 96)


_geo_hook = _geo_model.image_encoder.trunk.blocks[0].register_forward_hook(_geo_hook_fn)

# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def _geo_preprocess(img_pil):
    """PIL image → (1, 3, 1024, 1024) tensor normalisé."""
    _img = img_pil.convert('RGB').resize((_geo_IMG_SIZE, _geo_IMG_SIZE), Image.BILINEAR)
    _x = torch.from_numpy(np.array(_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((_x - _geo_MEAN) / _geo_STD).unsqueeze(0).to(_geo_DEVICE)


def _geo_get_feat_map(img_pil):
    """Forward pass → (H_feat, W_feat, 96) CPU tensor."""
    _geo_cap.clear()
    with torch.no_grad():
        _geo_model.image_encoder(_geo_preprocess(img_pil))
    return _geo_cap['feat'][0].cpu()   # (H_feat, W_feat, 96)


def _geo_transform_pil(img_pil, transform):
    """Appliquer une transformation géométrique via numpy (agnostique au mode PIL)."""
    arr = np.array(img_pil)
    if   transform == 'rot90':   arr = np.rot90(arr, k=3)   # CW 90°
    elif transform == 'rot180':  arr = np.rot90(arr, k=2)
    elif transform == 'rot270':  arr = np.rot90(arr, k=1)   # CW 270° = CCW 90°
    elif transform == 'flip_h':  arr = arr[:, ::-1]
    elif transform == 'flip_v':  arr = arr[::-1, :]
    return Image.fromarray(np.ascontiguousarray(arr))


def _geo_transform_bbox(x1, y1, x2, y2, W, H, transform):
    """
    Transformer un rectangle (x1,y1,x2,y2) selon la transformation.
    Conventions : x=colonne, y=ligne, origine en haut-gauche.
    Transformations des coins, puis min/max pour le nouveau rectangle.
    Retourne : (x1_new, y1_new, x2_new, y2_new, new_W, new_H).
    """
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    if transform == 'rot90':      # CW 90° : (x,y)→(H-y, x), image W×H → H×W
        nc = [(H - y, x) for x, y in corners];  new_W, new_H = H, W
    elif transform == 'rot180':   # (x,y)→(W-x, H-y), même taille
        nc = [(W - x, H - y) for x, y in corners];  new_W, new_H = W, H
    elif transform == 'rot270':   # CW 270° : (x,y)→(y, W-x), image W×H → H×W
        nc = [(y, W - x) for x, y in corners];  new_W, new_H = H, W
    elif transform == 'flip_h':   # (x,y)→(W-x, y)
        nc = [(W - x, y) for x, y in corners];  new_W, new_H = W, H
    elif transform == 'flip_v':   # (x,y)→(x, H-y)
        nc = [(x, H - y) for x, y in corners];  new_W, new_H = W, H
    else:
        raise ValueError(f'Unknown transform: {transform}')

    xs = [c[0] for c in nc];  ys = [c[1] for c in nc]
    return min(xs), min(ys), max(xs), max(ys), new_W, new_H


def _geo_extract_feat(feat_map, x1, y1, x2, y2, img_W, img_H):
    """
    Extraire et moyenner les features block_0 d'une région.
    Positions en coordonnées de l'image (avant preprocessing 1024×1024).
    Retourne (96,) numpy array, ou None si la région est vide.
    """
    H_feat, W_feat = feat_map.shape[:2]
    fx1 = max(0,      int(x1 * W_feat / img_W))
    fy1 = max(0,      int(y1 * H_feat / img_H))
    fx2 = min(W_feat, max(fx1 + 1, int(x2 * W_feat / img_W)))
    fy2 = min(H_feat, max(fy1 + 1, int(y2 * H_feat / img_H)))
    region = feat_map[fy1:fy2, fx1:fx2, :]
    if region.numel() == 0:
        return None
    return region.float().mean(dim=(0, 1)).numpy()   # (96,)


def _geo_l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Échantillonner N_PER_CAT patches par catégorie
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 1 — Échantillonnage ===')

_geo_selected = {}   # cat → [global_idx, ...]
for _c in _geo_CATS_VALID:
    _idx_c = np.where(_geo_CATEGORY_IDS == _c)[0]
    _n = min(N_PER_CAT, len(_idx_c))
    _geo_selected[_c] = np.random.default_rng(SEED + _c).choice(
        _idx_c, size=_n, replace=False
    ).tolist()
    print(f'  {CATEGORIES[_c]:<25}  n={_n}')

# Grouper par image pour n'ouvrir chaque image qu'une seule fois
_geo_by_img = defaultdict(list)   # img_name → [(global_idx, cat)]
for _c, _idxs in _geo_selected.items():
    for _idx in _idxs:
        _geo_by_img[_geo_IMAGE_NAMES[_idx]].append((_idx, _c))

_geo_n_patches_total = sum(len(v) for v in _geo_by_img.values())
print(f'Total : {_geo_n_patches_total} patches dans {len(_geo_by_img)} images')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Features de référence depuis HDF5
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 2 — Features de référence (HDF5) ===')

_geo_ref = {}   # global_idx → (96,) L2-normalisé
for _c, _idxs in _geo_selected.items():
    for _idx in _idxs:
        _geo_ref[_idx] = _geo_l2(_geo_X_all[_idx].astype(np.float32))
print(f'{len(_geo_ref)} vecteurs de référence chargés.')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Forward pass par image × transformation, extraction + similarité
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 3 — Forward passes ===')
print(f'(~{len(_geo_by_img) * len(TRANSFORMS)} forward passes en tout)')

_geo_sims      = {_t: {_c: [] for _c in _geo_CATS_VALID} for _t in TRANSFORMS}
_geo_sims_flat = {}   # (transform, global_idx) → float  [pour fig 3]

for _i_img, (_nm, _patches) in enumerate(_geo_by_img.items()):
    try:
        _img_orig = Image.open(IMG_DIR / _nm.decode()).convert('RGB')
    except Exception as _e:
        print(f'  Manquante : {_nm.decode()} ({_e})')
        continue

    _orig_W, _orig_H = _img_orig.size   # PIL: (width, height)

    if _i_img % 10 == 0:
        print(f'  [{_i_img+1:3d}/{len(_geo_by_img)}] {_nm.decode()[:55]}')

    for _tr in TRANSFORMS:
        # a) Transformer l'image entière
        _img_t = _geo_transform_pil(_img_orig, _tr)
        _new_W, _new_H = _img_t.size   # (width, height) de l'image transformée

        # b) Forward pass → feature map de l'image transformée
        _fm = _geo_get_feat_map(_img_t)   # (H_feat, W_feat, 96)

        # c) Extraire features pour chaque patch de cette image
        for _gidx, _cat in _patches:
            _x1, _y1, _x2, _y2 = _geo_POSITIONS[_gidx]

            # Nouvelle position du patch dans l'image transformée
            _x1t, _y1t, _x2t, _y2t, _, _ = _geo_transform_bbox(
                _x1, _y1, _x2, _y2, _orig_W, _orig_H, _tr
            )

            # Extraire + moyenner les features dans le feature map transformé
            _ft = _geo_extract_feat(_fm, _x1t, _y1t, _x2t, _y2t, _new_W, _new_H)
            if _ft is None:
                continue

            # Similarité cosine avec la feature de référence (HDF5)
            _sim = float(np.dot(_geo_ref[_gidx], _geo_l2(_ft)))
            _geo_sims[_tr][_cat].append(_sim)
            _geo_sims_flat[(_tr, _gidx)] = _sim

print('Forward passes terminés.')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Agréger
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 4 — Agrégation ===')

_geo_mean_t = {}
_geo_std_t  = {}
for _tr in TRANSFORMS:
    _all = [_s for _cs in _geo_sims[_tr].values() for _s in _cs]
    _geo_mean_t[_tr] = float(np.mean(_all)) if _all else 0.0
    _geo_std_t[_tr]  = float(np.std(_all))  if _all else 0.0

_geo_sim_mat = np.full((len(_geo_CATS_VALID), len(TRANSFORMS)), np.nan)
for _ci, _c in enumerate(_geo_CATS_VALID):
    for _ti, _tr in enumerate(TRANSFORMS):
        _s = _geo_sims[_tr][_c]
        if _s:
            _geo_sim_mat[_ci, _ti] = np.mean(_s)

print('Similarité par transformation (toutes catégories) :')
for _tr in TRANSFORMS:
    print(f'  {_geo_TLABELS[_tr]:<14}  {_geo_mean_t[_tr]:.4f} ± {_geo_std_t[_tr]:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Figures
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 5 — Figures ===')

_geo_tl = [_geo_TLABELS[_tr] for _tr in TRANSFORMS]

# ── Figure 1 : Barplot par transformation ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
_means = [_geo_mean_t[_tr] for _tr in TRANSFORMS]
_stds  = [_geo_std_t[_tr]  for _tr in TRANSFORMS]
ax1.bar(range(len(TRANSFORMS)), _means, yerr=_stds, capsize=5,
        color='#1B4F72', alpha=0.72, edgecolor='black', linewidth=0.6)
ax1.axhline(1.0,  color='green',  ls='--', lw=1.5, label='Invariance parfaite (1.0)')
ax1.axhline(0.9,  color='orange', ls=':',  lw=1.3, label='Seuil ⚠ (0.9)')
ax1.axhline(0.75, color='red',    ls=':',  lw=1.3, label='Seuil ❌ (0.75)')
ax1.set_xticks(range(len(TRANSFORMS)))
ax1.set_xticklabels(_geo_tl, fontsize=11)
ax1.set_ylabel('Similarité cosine moyenne', fontsize=11)
ax1.set_ylim([max(0, min(_means) - 0.15), 1.06])
ax1.set_title(
    'Invariance géométrique — block_0 TextureSAM\n'
    'cosine(features HDF5, features après transformation de l\'image entière)',
    fontsize=11,
)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3, axis='y')
for _i, (_m, _s) in enumerate(zip(_means, _stds)):
    ax1.text(_i, _m + _s + 0.005, f'{_m:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'geo_barplot_by_transform.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: geo_barplot_by_transform.png')

# ── Figure 2 : Heatmap catégorie × transformation ─────────────────────────────
_geo_cat_lbl = [CATEGORIES[_c] for _c in _geo_CATS_VALID]
fig2, ax2 = plt.subplots(figsize=(10, max(4, len(_geo_CATS_VALID) * 1.15)))
_vmin = max(0.3, np.nanmin(_geo_sim_mat) - 0.05)
im = ax2.imshow(_geo_sim_mat, cmap='RdYlGn', vmin=_vmin, vmax=1.0, aspect='auto')
ax2.set_xticks(range(len(TRANSFORMS)))
ax2.set_yticks(range(len(_geo_CATS_VALID)))
ax2.set_xticklabels(_geo_tl, fontsize=10)
ax2.set_yticklabels(_geo_cat_lbl, fontsize=9)
ax2.set_xlabel('Transformation', fontsize=11)
ax2.set_ylabel('Texture', fontsize=11)
ax2.set_title('Similarité cosine par texture × transformation\nvert=invariant · rouge=sensible',
              fontsize=11)
plt.colorbar(im, ax=ax2, fraction=0.046, label='Similarité cosine')
for _ci in range(len(_geo_CATS_VALID)):
    for _ti in range(len(TRANSFORMS)):
        _v = _geo_sim_mat[_ci, _ti]
        if not np.isnan(_v):
            ax2.text(_ti, _ci, f'{_v:.3f}', ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'geo_heatmap_cat_transform.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: geo_heatmap_cat_transform.png')

# ── Figure 3 : Diagnostic — vignettes + similarité ────────────────────────────
print('  Génération figure diagnostique...')

_n_rows = len(_geo_CATS_VALID)
_n_cols = 1 + len(TRANSFORMS)
fig3, axes3 = plt.subplots(_n_rows, _n_cols, figsize=(_n_cols * 2.4, _n_rows * 2.4))

for _ci, _cat in enumerate(_geo_CATS_VALID):
    _ex_idx = _geo_selected[_cat][0]
    _ex_nm  = _geo_IMAGE_NAMES[_ex_idx]
    _ex_pos = _geo_POSITIONS[_ex_idx].astype(int)

    try:
        _ex_img = Image.open(IMG_DIR / _ex_nm.decode()).convert('L')
    except Exception:
        for _col in range(_n_cols):
            axes3[_ci, _col].axis('off')
        continue

    _ex_W, _ex_H = _ex_img.size   # PIL (width, height)
    _ex_x1, _ex_y1, _ex_x2, _ex_y2 = _ex_pos

    # Col 0 : patch original
    axes3[_ci, 0].imshow(np.array(_ex_img.crop((_ex_x1, _ex_y1, _ex_x2, _ex_y2))),
                         cmap='gray', interpolation='nearest')
    axes3[_ci, 0].set_title('Original', fontsize=7)
    axes3[_ci, 0].set_ylabel(CATEGORIES[_cat], fontsize=7, rotation=90, va='center')
    axes3[_ci, 0].yaxis.set_label_coords(-0.12, 0.5)
    axes3[_ci, 0].set_xticks([]);  axes3[_ci, 0].set_yticks([])

    # Cols 1+ : patch transformé + similarité
    for _ti, _tr in enumerate(TRANSFORMS):
        _ax = axes3[_ci, _ti + 1]
        _img_t = _geo_transform_pil(_ex_img, _tr)
        _xt1, _yt1, _xt2, _yt2, _, _ = _geo_transform_bbox(
            _ex_x1, _ex_y1, _ex_x2, _ex_y2, _ex_W, _ex_H, _tr
        )
        _patch_t = _img_t.crop((int(_xt1), int(_yt1), int(_xt2), int(_yt2)))
        _ax.imshow(np.array(_patch_t), cmap='gray', interpolation='nearest')

        _sim = _geo_sims_flat.get((_tr, _ex_idx), float('nan'))
        _col = ('green' if _sim >= 0.9 else 'orange' if _sim >= 0.75 else 'red')
        _ax.set_title(f'{_geo_TLABELS[_tr]}\n{_sim:.3f}', fontsize=6, color=_col)
        _ax.set_xticks([]);  _ax.set_yticks([])

fig3.suptitle(
    'Diagnostic — patches originaux vs transformés\n'
    'titre = similarité cosine block_0  (vert ≥ 0.9 · orange ≥ 0.75 · rouge < 0.75)',
    fontsize=9,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'geo_diagnostic_patches.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: geo_diagnostic_patches.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Interprétation auto + .md résultats
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 6 — Interprétation ===')

_geo_THRESH_INV  = 0.9
_geo_THRESH_PART = 0.75
_geo_ORIENTED    = {3, 4, 5, 6}   # Faisceaux, Filaments, Stratifié rectiligne, sinueux

print('\nPar transformation :')
for _tr in TRANSFORMS:
    _m = _geo_mean_t[_tr]
    _v = ('✅ invariant' if _m >= _geo_THRESH_INV
          else '⚠️  partiel'  if _m >= _geo_THRESH_PART
          else '❌ sensible')
    print(f'  {_geo_TLABELS[_tr]:<14}  {_m:.3f}  {_v}')

print('\nPar texture (min sur toutes les transformations) :')
_geo_mean_per_cat = {}
for _ci, _c in enumerate(_geo_CATS_VALID):
    _row = _geo_sim_mat[_ci, ~np.isnan(_geo_sim_mat[_ci, :])]
    _mn  = float(_row.min())  if len(_row) > 0 else 0.0
    _mc  = float(_row.mean()) if len(_row) > 0 else 0.0
    _geo_mean_per_cat[_c] = _mc
    _v   = ('✅' if _mn >= _geo_THRESH_INV else '⚠️' if _mn >= _geo_THRESH_PART else '❌')
    _ori = ' (orientée)' if _c in _geo_ORIENTED else ''
    print(f'  {CATEGORIES[_c]:<25}  min={_mn:.3f}  moy={_mc:.3f}  {_v}{_ori}')

_most_inv  = max(_geo_CATS_VALID, key=lambda _c: _geo_mean_per_cat[_c])
_least_inv = min(_geo_CATS_VALID, key=lambda _c: _geo_mean_per_cat[_c])
_worst_t   = min(TRANSFORMS, key=lambda _tr: _geo_mean_t[_tr])
print(f'\nTexture la + invariante : {CATEGORIES[_most_inv]}  ({_geo_mean_per_cat[_most_inv]:.3f})')
print(f'Texture la - invariante : {CATEGORIES[_least_inv]}  ({_geo_mean_per_cat[_least_inv]:.3f})')
print(f'Transformation la + perturbatrice : '
      f'{_geo_TLABELS[_worst_t]}  ({_geo_mean_t[_worst_t]:.3f})')

# Générer results .md dans OUTPUT_DIR
_geo_md = f"""\
# Invariance géométrique — block_0 TextureSAM — Résultats

## Configuration
- N_PER_CAT = {N_PER_CAT} · SEED = {SEED}
- Catégories : {[CATEGORIES[c] for c in _geo_CATS_VALID]}
- Transformations : {list(_geo_TLABELS.values())}

## Similarité par transformation (toutes catégories)

| Transformation | Similarité moy. | Std | Verdict |
|---|---|---|---|
"""
for _tr in TRANSFORMS:
    _m = _geo_mean_t[_tr];  _s = _geo_std_t[_tr]
    _v = ('✅ invariant' if _m >= _geo_THRESH_INV
          else '⚠️ partiel' if _m >= _geo_THRESH_PART else '❌ sensible')
    _geo_md += f'| {_geo_TLABELS[_tr]} | {_m:.4f} | {_s:.4f} | {_v} |\n'

_geo_md += '\n## Similarité par texture (moyenne sur toutes les transformations)\n\n'
_geo_md += '| Texture | Sim moy. | Sim min | Verdict | Note |\n|---|---|---|---|---|\n'
for _ci, _c in enumerate(_geo_CATS_VALID):
    _row = _geo_sim_mat[_ci, ~np.isnan(_geo_sim_mat[_ci, :])]
    _mn  = float(_row.min())  if len(_row) > 0 else 0.0
    _mc  = float(_row.mean()) if len(_row) > 0 else 0.0
    _v   = '✅' if _mn >= _geo_THRESH_INV else ('⚠️' if _mn >= _geo_THRESH_PART else '❌')
    _n   = 'orientée' if _c in _geo_ORIENTED else 'isotrope'
    _geo_md += f'| {CATEGORIES[_c]} | {_mc:.4f} | {_mn:.4f} | {_v} | {_n} |\n'

_geo_md += f"""
## Synthèse

- **Texture la plus invariante** : {CATEGORIES[_most_inv]} ({_geo_mean_per_cat[_most_inv]:.4f})
- **Texture la moins invariante** : {CATEGORIES[_least_inv]} ({_geo_mean_per_cat[_least_inv]:.4f})
- **Transformation la plus perturbatrice** : {_geo_TLABELS[_worst_t]} ({_geo_mean_t[_worst_t]:.4f})

## Interprétation

Une similarité cosine < 0.9 entre features originales et features après
transformation indique que block_0 est SENSIBLE à cette transformation.
Les textures orientées (Faisceaux, Filaments, Stratifié) pourraient
légitimement être moins invariantes à la rotation : cela reflète une
propriété réelle de la texture (l'orientation fait partie de sa définition),
pas un défaut du réseau.

## Fichiers
- `geo_barplot_by_transform.png` : Similarité par transformation
- `geo_heatmap_cat_transform.png` : Heatmap texture × transformation
- `geo_diagnostic_patches.png` : Vignettes avec similarité par patch
"""

with open(OUTPUT_DIR / 'test_geometric_invariance.md', 'w') as _f:
    _f.write(_geo_md)
print('Saved: test_geometric_invariance.md')

# ── Nettoyage ──────────────────────────────────────────────────────────────────
_geo_hook.remove()
print('\nHook block_0 supprimé.')

print(f'\n=== Fichiers dans {OUTPUT_DIR} ===')
for _p in sorted(OUTPUT_DIR.iterdir()):
    print(f'  {_p.name}')
