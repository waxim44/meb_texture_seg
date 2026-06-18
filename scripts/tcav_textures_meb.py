#!/usr/bin/env python3
"""
tcav_textures_meb.py
Appliquer TCAV (Concept Activation Vectors) aux textures MEB.
Pour chaque texture : extraire son CAV (direction de concept),
mesurer la sensibilité spatiale, tester le steering causal.
"""

import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
SAM2_DIR   = ROOT / 'TextureSAM' / 'sam2'
DB_PATH    = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
CFG_PATH   = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
IMG_DIR    = ROOT / 'Image_Ouassim'
CHECKPOINT = 'checkpoints/sam2.1_hiera_small_1.pt'
OUTPUT_DIR = ROOT / 'outputs' / 'tcav_textures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Résultats → {OUTPUT_DIR}')

_tcav_SEED       = 42
_tcav_KEY        = 'block_0'
_tcav_CATS_EXCL  = {2, 8, 10, 11, 12, 13}
_tcav_TARGET_IMG = '310120-pat18-WholeMount-24.tif'
_tcav_ALPHAS     = [-3, -2, -1, 0, 1, 2, 3]

# ── Config ────────────────────────────────────────────────────────────────────
with open(CFG_PATH) as _f:
    _tcav_cfg = json.load(_f)
CATEGORIES = {int(k): v['name']  for k, v in _tcav_cfg['available_categories'].items()}
CAT_COLORS = {int(k): v['color'] for k, v in _tcav_cfg['available_categories'].items()}

# ── HDF5 ──────────────────────────────────────────────────────────────────────
with h5py.File(DB_PATH, 'r') as _h5:
    _tcav_IMAGE_NAMES  = _h5['metadata/image_names'][:]
    _tcav_POSITIONS    = _h5['metadata/positions'][:]
    _tcav_CATEGORY_IDS = _h5['metadata/category_ids'][:].astype(int)
    _tcav_X_all        = _h5['features'][_tcav_KEY][:]

_tcav_CATS_VALID = sorted(
    int(c) for c in np.unique(_tcav_CATEGORY_IDS)
    if int(c) not in _tcav_CATS_EXCL
    and (_tcav_CATEGORY_IDS == int(c)).sum() >= 30
)
_tcav_mask_valid = np.isin(_tcav_CATEGORY_IDS, _tcav_CATS_VALID)
_tcav_X    = _tcav_X_all[_tcav_mask_valid]          # (N, 96)
_tcav_y    = _tcav_CATEGORY_IDS[_tcav_mask_valid]
_tcav_imgs = _tcav_IMAGE_NAMES[_tcav_mask_valid]

print(f'Patches : {len(_tcav_X)}  ·  Catégories : {len(_tcav_CATS_VALID)}')
for _c in _tcav_CATS_VALID:
    print(f'  {_c:2d}  {CATEGORIES[_c]:<25}  n={(_tcav_y == _c).sum()}')

# ─────────────────────────────────────────────────────────────────────────────
# Chargement du modèle SAM2 (pour les cartes de sensibilité)
# ─────────────────────────────────────────────────────────────────────────────

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

_tcav_device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_tcav_MODEL_CFG = 'sam2.1/sam2.1_hiera_s.yaml'
_tcav_IMG_SIZE  = 1024
_tcav_MEAN      = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_tcav_STD       = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
print(f'Device : {_tcav_device}')


def _tcav_load_ckpt_sd(ckpt_path):
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


def _tcav_load_model(checkpoint):
    _base = str(ROOT / 'checkpoints' / 'sam2.1_hiera_small')
    _sam2 = build_sam2(_tcav_MODEL_CFG, ckpt_path=None,
                       device=_tcav_device, apply_postprocessing=False)
    _sam2.load_state_dict(_tcav_load_ckpt_sd(_base), strict=False)
    _ck = str(ROOT / checkpoint) if not Path(checkpoint).is_absolute() else checkpoint
    if Path(_ck).resolve() != Path(_base).resolve():
        _miss, _unex = _sam2.load_state_dict(_tcav_load_ckpt_sd(_ck), strict=False)
        print(f'Fine-tuned weights  missing={len(_miss)} unexpected={len(_unex)}')
    _sam2.eval()
    return _sam2


def _tcav_preprocess(img_pil):
    """PIL Image → (1, 3, 1024, 1024) tensor normalisé."""
    _img = img_pil.convert('RGB').resize((_tcav_IMG_SIZE, _tcav_IMG_SIZE), Image.BILINEAR)
    _x = torch.from_numpy(np.array(_img)).float() / 255.0
    _x = _x.permute(2, 0, 1)
    _x = (_x - _tcav_MEAN) / _tcav_STD
    return _x.unsqueeze(0).to(_tcav_device)


_tcav_model = _tcav_load_model(CHECKPOINT)
print(f'Modèle chargé  ({sum(p.numel() for p in _tcav_model.parameters())/1e6:.1f}M params)')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Extraire les CAV (un par texture)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 1 — Extraction des CAV ===')

_tcav_scaler   = StandardScaler()
_tcav_X_scaled = _tcav_scaler.fit_transform(_tcav_X)   # (N, 96), sans PCA

_tcav_cavs    = {}   # cat → vecteur CAV normalisé (96,)
_tcav_cav_acc = {}   # cat → accuracy one-vs-rest

for _c in _tcav_CATS_VALID:
    _y_bin = (_tcav_y == _c).astype(int)
    _clf = LogisticRegression(
        class_weight='balanced', max_iter=1000,
        C=1.0, random_state=_tcav_SEED,
    )
    _clf.fit(_tcav_X_scaled, _y_bin)
    _cav = _clf.coef_[0]                      # (96,)
    _cav = _cav / (np.linalg.norm(_cav) + 1e-10)
    _tcav_cavs[_c]    = _cav
    _tcav_cav_acc[_c] = _clf.score(_tcav_X_scaled, _y_bin)

print('CAV extraits :')
for _c in _tcav_CATS_VALID:
    print(f'  {CATEGORIES[_c]:<25} acc one-vs-rest = {_tcav_cav_acc[_c]*100:.1f}%')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Similarité entre CAV
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 2 — Similarité entre CAV ===')

_tcav_n      = len(_tcav_CATS_VALID)
_tcav_cav_sim = np.zeros((_tcav_n, _tcav_n))
for _i, _ci in enumerate(_tcav_CATS_VALID):
    for _j, _cj in enumerate(_tcav_CATS_VALID):
        _tcav_cav_sim[_i, _j] = _tcav_cavs[_ci] @ _tcav_cavs[_cj]

_tcav_labels = [CATEGORIES[_c] for _c in _tcav_CATS_VALID]

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(_tcav_cav_sim, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(_tcav_n)); ax.set_yticks(range(_tcav_n))
ax.set_xticklabels(_tcav_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(_tcav_labels, fontsize=9)
for _i in range(_tcav_n):
    for _j in range(_tcav_n):
        ax.text(
            _j, _i, f'{_tcav_cav_sim[_i, _j]:.2f}',
            ha='center', va='center', fontsize=8,
            color='white' if abs(_tcav_cav_sim[_i, _j]) > 0.6 else 'black',
        )
plt.colorbar(im, ax=ax, fraction=0.046, label='Similarité cosine entre CAV')
ax.set_title(
    'Similarité entre CAV de textures\n'
    'CAV proches = concepts texturaux similaires',
    fontsize=12,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cav_similarity.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cav_similarity.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Cartes de sensibilité spatiale
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n=== Étape 3 — Sensibilité spatiale sur {_tcav_TARGET_IMG} ===')

_tcav_cap  = {}
_tcav_hook = _tcav_model.image_encoder.trunk.blocks[0].register_forward_hook(
    lambda m, i, o: _tcav_cap.update({_tcav_KEY: o.detach()})
)

try:
    _tcav_pil   = Image.open(IMG_DIR / _tcav_TARGET_IMG).convert('RGB')
    _tcav_orig_H, _tcav_orig_W = _tcav_pil.height, _tcav_pil.width
    _tcav_tensor = _tcav_preprocess(_tcav_pil)

    _tcav_cap.clear()
    with torch.no_grad():
        _tcav_model.image_encoder(_tcav_tensor)

    _tcav_feat_map = _tcav_cap[_tcav_KEY][0].cpu().numpy()   # (H, W, 96)
    _tcav_H, _tcav_W, _tcav_C = _tcav_feat_map.shape
    print(f'feat_map shape : {_tcav_feat_map.shape}')

    _tcav_flat        = _tcav_feat_map.reshape(-1, _tcav_C)
    _tcav_flat_scaled = _tcav_scaler.transform(_tcav_flat)

    _tcav_sensitivity_maps = {}
    for _c in _tcav_CATS_VALID:
        _proj = _tcav_flat_scaled @ _tcav_cavs[_c]          # (H*W,)
        _tcav_sensitivity_maps[_c] = _proj.reshape(_tcav_H, _tcav_W)

    _tcav_img_gray = np.array(_tcav_pil.convert('L'))

    _tcav_ncols = 4
    _tcav_nrows = int(np.ceil((_tcav_n + 1) / _tcav_ncols))
    fig, axes = plt.subplots(_tcav_nrows, _tcav_ncols,
                             figsize=(_tcav_ncols * 4.5, _tcav_nrows * 4))
    axes = axes.flatten()

    axes[0].imshow(_tcav_img_gray, cmap='gray')
    axes[0].set_title('Image originale', fontsize=10)
    axes[0].axis('off')

    for _idx, _c in enumerate(_tcav_CATS_VALID):
        _smap = _tcav_sensitivity_maps[_c]
        _smap_n = (_smap - _smap.min()) / (_smap.max() - _smap.min() + 1e-8)
        _smap_full = cv2.resize(_smap_n, (_tcav_orig_W, _tcav_orig_H),
                                interpolation=cv2.INTER_LINEAR)
        _ax = axes[_idx + 1]
        _ax.imshow(_tcav_img_gray, cmap='gray', alpha=0.5)
        _ax.imshow(_smap_full, cmap='hot', alpha=0.6)
        _ax.set_title(f'Sensibilité : {CATEGORIES[_c]}', fontsize=9)
        _ax.axis('off')

    for _k in range(_tcav_n + 1, len(axes)):
        axes[_k].axis('off')

    plt.suptitle(
        f'Cartes de sensibilité TCAV — {_tcav_TARGET_IMG}\n'
        'chaud = concept texture détecté',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tcav_sensitivity_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: tcav_sensitivity_maps.png')

finally:
    _tcav_hook.remove()
    print('Hook retiré.')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Test de steering causal
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 4 — Steering causal ===')

_tcav_clf_multi = LogisticRegression(
    class_weight='balanced', max_iter=1000,
    random_state=_tcav_SEED,
)
_tcav_clf_multi.fit(_tcav_X_scaled, _tcav_y)

_tcav_steering_results = {}
_tcav_rng = np.random.default_rng(_tcav_SEED)

for _c_target in _tcav_CATS_VALID:
    _cav_t     = _tcav_cavs[_c_target]
    _mask_other = _tcav_y != _c_target
    _X_other   = _tcav_X_scaled[_mask_other]
    _sample    = _tcav_rng.choice(len(_X_other),
                                   size=min(100, len(_X_other)), replace=False)
    _X_sample  = _X_other[_sample]

    _props = []
    for _alpha in _tcav_ALPHAS:
        _X_steered  = _X_sample + _alpha * _cav_t
        _preds      = _tcav_clf_multi.predict(_X_steered)
        _props.append(float((_preds == _c_target).mean() * 100))
    _tcav_steering_results[_c_target] = _props

fig, ax = plt.subplots(figsize=(12, 6))
for _c in _tcav_CATS_VALID:
    ax.plot(_tcav_ALPHAS, _tcav_steering_results[_c], 'o-',
            lw=1.8, ms=6, color=CAT_COLORS[_c], label=CATEGORIES[_c])
ax.axvline(0, color='gray', ls='--', alpha=0.5, label='pas de steering')
ax.set_xlabel('α (intensité du déplacement le long du CAV)', fontsize=11)
ax.set_ylabel('% patches classés comme la texture cible', fontsize=11)
ax.set_title(
    'Test de steering causal — TCAV\n'
    'pousser des patches vers une texture le long de son CAV',
    fontsize=12,
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tcav_steering.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: tcav_steering.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Dimensions dominantes par CAV
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 5 — Dimensions dominantes ===')

_tcav_mat_dims = np.array([np.abs(_tcav_cavs[_c]) for _c in _tcav_CATS_VALID])

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(_tcav_mat_dims, cmap='viridis', aspect='auto')
ax.set_yticks(range(_tcav_n))
ax.set_yticklabels(_tcav_labels, fontsize=9)
ax.set_xlabel('Dimension de block_0 (0–95)', fontsize=11)
ax.set_title(
    'Poids absolus des dimensions par CAV\n'
    'clair = dimension importante pour ce concept',
    fontsize=12,
)
plt.colorbar(im, ax=ax, fraction=0.02, label='|poids CAV|')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cav_dimensions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cav_dimensions.png')

_tcav_top_dims = {}
for _c in _tcav_CATS_VALID:
    _top = np.argsort(np.abs(_tcav_cavs[_c]))[-5:][::-1]
    _tcav_top_dims[_c] = _top.tolist()
    print(f'  {CATEGORIES[_c]:<25} top dims : {_top.tolist()}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Sauvegardes + README.md
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 6 — Sauvegarde ===')

with open(OUTPUT_DIR / 'cavs.pkl', 'wb') as _f:
    pickle.dump({
        'cavs'     : _tcav_cavs,
        'cav_acc'  : _tcav_cav_acc,
        'scaler'   : _tcav_scaler,
        'top_dims' : _tcav_top_dims,
        'cats_valid': _tcav_CATS_VALID,
    }, _f)
print('Saved: cavs.pkl')

_tcav_readme = """\
# Analyse TCAV — Textures MEB (block_0)

## Objectif

Comprendre comment block_0 de TextureSAM encode chaque texture
en extrayant un **Concept Activation Vector (CAV)** par texture :
une direction dans l'espace des features (96d) qui représente
le concept de cette texture.

Référence : Kim et al. (2018), "Interpretability Beyond Feature
Attribution: Quantitative Testing with Concept Activation
Vectors (TCAV)", ICML.

## Méthode

### Étape 1 — Extraire les CAV

Pour chaque texture c :
- On prend tous les patches de la base HDF5
- On étiquette : 1 si texture = c, 0 sinon (one-vs-rest)
- On entraîne une régression logistique (sans PCA, sur 96d bruts)
- Le vecteur de poids appris = direction du concept "texture c"
- On le normalise → CAV(c)

Le CAV est la direction dans l'espace 96d qui sépare le mieux
cette texture des autres.

### Étape 2 — Similarité entre CAV

Pour chaque paire de textures :
- Similarité cosine entre leurs CAV (déjà normalisés → produit scalaire)
- CAV proches → concepts texturaux encodés de façon similaire
- Révèle quelles textures partagent un encodage commun

### Étape 3 — Cartes de sensibilité spatiale

Pour l'image cible :
- On extrait la feature map block_0 (H × W × 96) via forward hook
- On applique le même StandardScaler (fitté sur la base)
- Pour chaque position spatiale, on projette son vecteur sur CAV(c)
- Score élevé = cette zone ressemble au concept "texture c"
- On upsampe vers la taille originale (INTER_LINEAR)

### Étape 4 — Test de steering causal

Pour vérifier que le CAV contrôle vraiment la texture :
- On prend des patches d'autres textures (échantillon de 100)
- On les déplace : x_scaled + alpha × CAV(c)  (alpha ∈ [-3, 3])
- On reclasse ces patches déplacés avec un classifieur multi-classe
- Si la proportion classée "c" augmente avec alpha positif
  → le CAV contrôle causalement la perception de la texture

### Étape 5 — Dimensions dominantes

Pour chaque CAV :
- |poids| par dimension (0–95 de block_0)
- Top-5 dimensions avec le poids absolu le plus élevé
- Identifie les dimensions clés de chaque concept textural

## Fichiers générés

| Fichier | Description |
|---|---|
| `cav_similarity.png` | Matrice de similarité cosine entre CAV |
| `tcav_sensitivity_maps.png` | Cartes de sensibilité par texture sur l'image cible |
| `tcav_steering.png` | Test de steering causal (α de -3 à +3) |
| `cav_dimensions.png` | Poids absolus |CAV| par dimension (heatmap) |
| `cavs.pkl` | CAV sauvegardés + scaler + top dimensions |
| `README.md` | Ce fichier |

## Interprétation

- **Similarité CAV élevée** entre deux textures → block_0 les encode
  de façon proche (ex : Faisceaux/Filaments attendus proches)
- **Steering efficace** (proportion monte avec alpha) → le concept
  est une direction linéaire manipulable dans l'espace feature
- **Dimensions dominantes partagées** entre textures → encodage
  distribué et polysémantique (normal pour un réseau profond)
- **CAV négatifs** (similarité négative) → concepts "opposés"
  dans l'espace feature (ex : texture homogène vs texture complexe)

"""

_tcav_readme += '\n## Résultats\n\n'
_tcav_readme += '### Qualité des CAV (accuracy one-vs-rest)\n\n'
for _c in _tcav_CATS_VALID:
    _tcav_readme += f'- {CATEGORIES[_c]} : {_tcav_cav_acc[_c]*100:.1f}%\n'

_tcav_readme += '\n### Top-5 dimensions par texture\n\n'
for _c in _tcav_CATS_VALID:
    _dims_str = ', '.join(str(_d) for _d in _tcav_top_dims[_c])
    _tcav_readme += f'- {CATEGORIES[_c]} : dims [{_dims_str}]\n'

_tcav_readme += f'\n### Image cible\n\n`{_tcav_TARGET_IMG}`\n'

with open(OUTPUT_DIR / 'README.md', 'w') as _f:
    _f.write(_tcav_readme)
print('Saved: README.md')

print(f'\n=== Fichiers générés dans {OUTPUT_DIR} ===')
for _fname in ['README.md', 'cav_similarity.png', 'tcav_sensitivity_maps.png',
               'tcav_steering.png', 'cav_dimensions.png', 'cavs.pkl']:
    _p = OUTPUT_DIR / _fname
    print(f'  {"✓" if _p.exists() else "✗"}  {_fname}')
