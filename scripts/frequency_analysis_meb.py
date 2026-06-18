#!/usr/bin/env python3
"""
frequency_analysis_meb.py
Analyser le lien entre les features block_0 et le contenu fréquentiel
des textures (FFT). Tester si block_0 encode l'information de fréquence
spatiale.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from tqdm.auto import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DB_PATH    = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
CFG_PATH   = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUTPUT_DIR = ROOT / 'outputs' / 'frequency_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Résultats → {OUTPUT_DIR}')

# ── Hyperparamètres ───────────────────────────────────────────────────────────
_freq_SEED        = 42
_freq_KEY         = 'block_0'
_freq_CATS_EXCL   = {2, 8, 10, 11, 12, 13}
_freq_MIN_PATCHES = 30
_freq_N_BINS      = 20
_freq_PCA_DIM     = 20
_freq_CCA_COMP    = 10

# ── Config + DB ───────────────────────────────────────────────────────────────
with open(CFG_PATH) as _f:
    _freq_cfg = json.load(_f)
CATEGORIES = {int(k): v['name']  for k, v in _freq_cfg['available_categories'].items()}
CAT_COLORS = {int(k): v['color'] for k, v in _freq_cfg['available_categories'].items()}

with h5py.File(DB_PATH, 'r') as _h5:
    _freq_IMAGE_NAMES  = _h5['metadata/image_names'][:]
    _freq_POSITIONS    = _h5['metadata/positions'][:]
    _freq_CATEGORY_IDS = _h5['metadata/category_ids'][:].astype(int)
    _freq_X_block0_all = _h5['features'][_freq_KEY][:]

_freq_CATS_VALID = sorted(
    int(c) for c in np.unique(_freq_CATEGORY_IDS)
    if int(c) not in _freq_CATS_EXCL
    and (_freq_CATEGORY_IDS == int(c)).sum() >= _freq_MIN_PATCHES
)
_freq_mask_valid = np.isin(_freq_CATEGORY_IDS, _freq_CATS_VALID)

print(f'Catégories valides : {[CATEGORIES[c] for c in _freq_CATS_VALID]}')
print(f'Patches valides    : {_freq_mask_valid.sum()}')

# ─────────────────────────────────────────────────────────────────────────────
# Fonction — Profil radial FFT
# ─────────────────────────────────────────────────────────────────────────────

def _freq_radial_profile(patch_gray, n_bins=_freq_N_BINS):
    """
    Profil d'énergie radial de la FFT 2D.
    Retourne (n_bins,) : énergie par bande de fréquence, log-normalisée.
    """
    _f      = np.fft.fft2(patch_gray)
    _fshift = np.fft.fftshift(_f)
    _power  = np.abs(_fshift) ** 2

    _H, _W  = patch_gray.shape
    _cy, _cx = _H // 2, _W // 2
    _y, _x  = np.ogrid[:_H, :_W]
    _r      = np.sqrt((_x - _cx) ** 2 + (_y - _cy) ** 2)
    _r_max  = _r.max()

    _profile = np.zeros(n_bins)
    for _b in range(n_bins):
        _r_in  = _b       * _r_max / n_bins
        _r_out = (_b + 1) * _r_max / n_bins
        _ring  = (_r >= _r_in) & (_r < _r_out)
        if _ring.sum() > 0:
            _profile[_b] = _power[_ring].mean()

    _profile = np.log1p(_profile)
    _profile = _profile / (_profile.sum() + 1e-8)
    return _profile

# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Extraire les profils fréquentiels
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 1 — Extraction des profils FFT ===')

_freq_patches_by_img = defaultdict(list)
for _i in np.where(_freq_mask_valid)[0]:
    _freq_patches_by_img[_freq_IMAGE_NAMES[_i]].append(int(_i))

_freq_profiles = {}   # idx → profil radial (n_bins,)

for _img_name in tqdm(_freq_patches_by_img, desc='FFT'):
    try:
        _img = Image.open(IMG_DIR / _img_name.decode()).convert('L')
    except Exception as _e:
        print(f'  Image manquante : {_img_name} ({_e})')
        continue
    _img_np = np.array(_img).astype(np.float32)

    for _idx in _freq_patches_by_img[_img_name]:
        _pos = _freq_POSITIONS[_idx].astype(int)
        _x1, _y1, _x2, _y2 = _pos
        _patch = _img_np[_y1:_y2, _x1:_x2]
        if _patch.shape[0] < 8 or _patch.shape[1] < 8:
            continue
        _freq_profiles[_idx] = _freq_radial_profile(_patch)

_freq_valid_idx = sorted(_freq_profiles.keys())
_freq_y_valid   = _freq_CATEGORY_IDS[_freq_valid_idx]
_freq_F         = np.array([_freq_profiles[_i] for _i in _freq_valid_idx])   # (N, n_bins)
_freq_X_feat    = _freq_X_block0_all[_freq_valid_idx]                         # (N, 96)

print(f'Profils extraits : {len(_freq_valid_idx)} patches')
print(f'F_freq shape : {_freq_F.shape}')
print(f'X_feat shape : {_freq_X_feat.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Signature fréquentielle par texture
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 2 — Signatures fréquentielles ===')

_freq_bins_x = np.linspace(0, 1, _freq_N_BINS)

fig, ax = plt.subplots(figsize=(12, 7))
for _c in _freq_CATS_VALID:
    _mask_c = _freq_y_valid == _c
    _mean   = _freq_F[_mask_c].mean(axis=0)
    _std    = _freq_F[_mask_c].std(axis=0)
    ax.plot(_freq_bins_x, _mean, '-', lw=2,
            color=CAT_COLORS[_c], label=CATEGORIES[_c])
    ax.fill_between(_freq_bins_x, _mean - _std, _mean + _std,
                    alpha=0.1, color=CAT_COLORS[_c])

ax.set_xlabel(
    'Fréquence spatiale normalisée\n(0 = basse/grossier · 1 = haute/fin)',
    fontsize=11,
)
ax.set_ylabel('Énergie normalisée (log)', fontsize=11)
ax.set_title(
    'Signature fréquentielle par texture\nprofil radial FFT moyen ± std',
    fontsize=12,
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'frequency_signatures.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: frequency_signatures.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — CCA block_0 ↔ profil fréquentiel
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 3 — CCA block_0 ↔ fréquences ===')

_freq_pca   = PCA(n_components=_freq_PCA_DIM, random_state=_freq_SEED)
_freq_X_pca = _freq_pca.fit_transform(_freq_X_feat)
print(f'PCA variance expliquée : {_freq_pca.explained_variance_ratio_.sum()*100:.1f}%')

_freq_n_comp = min(_freq_CCA_COMP, _freq_N_BINS, _freq_PCA_DIM)
_freq_cca    = CCA(n_components=_freq_n_comp)
_freq_X_c, _freq_F_c = _freq_cca.fit_transform(_freq_X_pca, _freq_F)

_freq_canonical_corrs = []
for _k in range(_freq_n_comp):
    _r, _ = pearsonr(_freq_X_c[:, _k], _freq_F_c[:, _k])
    _freq_canonical_corrs.append(abs(float(_r)))

print('Corrélations canoniques :')
for _k, _r in enumerate(_freq_canonical_corrs):
    print(f'  Composante {_k+1:2d} : {_r:.3f}')
print(f'Max : {max(_freq_canonical_corrs):.3f}  ·  Moyenne : {np.mean(_freq_canonical_corrs):.3f}')

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, _freq_n_comp + 1), _freq_canonical_corrs, color='#1B4F72')
ax.set_xlabel('Composante canonique', fontsize=11)
ax.set_ylabel('Corrélation canonique', fontsize=11)
ax.set_title(
    "CCA — Corrélation block_0 ↔ contenu fréquentiel\n"
    "élevé = block_0 encode l'information de fréquence",
    fontsize=12,
)
ax.set_ylim([0, 1])
ax.axhline(0.5, color='red', ls=':', alpha=0.6, label='corrélation modérée')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cca_freq_features.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cca_freq_features.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Prédire les fréquences depuis block_0 (Ridge + CV)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 4 — Prédictibilité fréquences depuis block_0 ===')

_freq_ridge  = Ridge(alpha=1.0)
_freq_F_pred = cross_val_predict(_freq_ridge, _freq_X_pca, _freq_F, cv=5)

_freq_r2_per_bin = [
    float(r2_score(_freq_F[:, _b], _freq_F_pred[:, _b]))
    for _b in range(_freq_N_BINS)
]
print(f'R² moyen : {np.mean(_freq_r2_per_bin):.3f}')
print(f'R² max   : {max(_freq_r2_per_bin):.3f}  (bande {np.argmax(_freq_r2_per_bin)})')

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(_freq_bins_x, _freq_r2_per_bin, width=0.04, color='#27AE60')
ax.set_xlabel('Fréquence spatiale normalisée', fontsize=11)
ax.set_ylabel('R² (prédiction depuis block_0)', fontsize=11)
ax.set_title(
    'Prédictibilité du contenu fréquentiel depuis block_0\n'
    'R² élevé = block_0 encode cette bande de fréquence',
    fontsize=12,
)
ax.axhline(0, color='black', lw=0.5)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'frequency_predictability.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: frequency_predictability.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — README.md + sauvegarde
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 5 — README + sauvegarde ===')

_freq_readme = """\
# Analyse fréquentielle — Textures MEB (block_0)

## Objectif

Tester si les features de block_0 encodent l'information de
**fréquence spatiale**, qui est la définition classique de la
texture (distribution spatiale répétée de motifs).

## Méthode

### Étape 1 — Profil fréquentiel par patch

Pour chaque patch :
- On calcule la FFT 2D (transformée de Fourier discrète)
- On obtient le spectre de puissance : énergie par fréquence
- On centre le spectre (fftshift) : basses fréquences au centre
- On moyenne l'énergie par anneaux concentriques :
  - Anneau central = basses fréquences (motifs grossiers)
  - Anneaux externes = hautes fréquences (motifs fins)
- On log-normalise le profil pour compresser la dynamique
- Résultat : vecteur de {n_bins} valeurs (énergie par bande)

### Étape 2 — Signature fréquentielle par texture

Pour chaque texture :
- On moyenne les profils fréquentiels de ses patches (± std)
- Chaque texture a une signature caractéristique :
  - Granuleux → forte énergie haute fréquence
  - Trou → énergie concentrée basses fréquences
  - Filaments → pics à des fréquences intermédiaires

### Étape 3 — Corrélation features ↔ fréquences (CCA)

L'analyse de corrélation canonique (CCA) cherche les directions
dans l'espace block_0 (réduit à {pca_dim}d par PCA) et dans
l'espace fréquentiel qui sont maximalement corrélées.

- Corrélation canonique élevée (> 0.5) → block_0 encode les fréquences
- On calcule {n_comp} composantes canoniques

### Étape 4 — Prédictibilité (Ridge regression)

On entraîne une régression Ridge (5-fold CV) :
  block_0 (PCA-{pca_dim}d) → profil fréquentiel ({n_bins} bandes)

- R² élevé pour une bande → block_0 encode bien cette fréquence
- R² faible → block_0 ne contient pas cette information fréquentielle

## Interprétation

- **Signatures distinctes** entre textures → la fréquence spatiale
  discrimine les textures (cohérent avec la définition classique)
- **CCA élevée** → block_0 encode l'information fréquentielle
- **R² élevé** → on peut reconstruire le contenu fréquentiel
  depuis block_0, prouvant qu'il agit comme un analyseur
  fréquentiel appris (lien avec les filtres de Gabor classiques)
- **Lien Gabor** : les filtres de Gabor (utilisés dans compare_descriptors)
  sont des filtrages fréquentiels orientés ; si block_0 corrèle avec FFT,
  les deux approches capturent la même information fondamentale

## Fichiers générés

| Fichier | Description |
|---|---|
| `frequency_signatures.png` | Signature FFT par texture (profil moyen ± std) |
| `cca_freq_features.png` | Corrélations canoniques block_0 ↔ FFT |
| `frequency_predictability.png` | R² par bande de fréquence |
| `frequency_results.pkl` | Profils, résultats CCA et R² sauvegardés |
| `README.md` | Ce fichier |

""".format(
    n_bins=_freq_N_BINS,
    pca_dim=_freq_PCA_DIM,
    n_comp=_freq_n_comp,
)

_freq_readme += '## Résultats\n\n'
_freq_readme += f'- Patches analysés : {len(_freq_valid_idx)}\n'
_freq_readme += f'- Catégories : {len(_freq_CATS_VALID)}\n'
_freq_readme += f'- Corrélation canonique max   : {max(_freq_canonical_corrs):.3f}\n'
_freq_readme += f'- Corrélation canonique moy.  : {np.mean(_freq_canonical_corrs):.3f}\n'
_freq_readme += f'- R² moyen (prédiction FFT)   : {np.mean(_freq_r2_per_bin):.3f}\n'
_freq_readme += f'- R² max   (bande {np.argmax(_freq_r2_per_bin)})      : {max(_freq_r2_per_bin):.3f}\n'
_freq_readme += '\n### Signatures fréquentielles moyennes par texture\n\n'
for _c in _freq_CATS_VALID:
    _mean_c = _freq_F[_freq_y_valid == _c].mean(axis=0)
    _peak   = int(np.argmax(_mean_c))
    _freq_readme += (
        f'- {CATEGORIES[_c]:<25} : pic à la bande {_peak} '
        f'({_freq_bins_x[_peak]:.2f} fréq. norm.)\n'
    )

with open(OUTPUT_DIR / 'README.md', 'w') as _f:
    _f.write(_freq_readme)
print('Saved: README.md')

with open(OUTPUT_DIR / 'frequency_results.pkl', 'wb') as _f:
    pickle.dump({
        'freq_profiles'    : _freq_F,
        'valid_idx'        : _freq_valid_idx,
        'y_valid'          : _freq_y_valid,
        'canonical_corrs'  : _freq_canonical_corrs,
        'r2_per_bin'       : _freq_r2_per_bin,
        'cats_valid'       : _freq_CATS_VALID,
    }, _f)
print('Saved: frequency_results.pkl')

print(f'\n=== Fichiers générés dans {OUTPUT_DIR} ===')
for _fname in ['README.md', 'frequency_signatures.png',
               'cca_freq_features.png', 'frequency_predictability.png',
               'frequency_results.pkl']:
    _p = OUTPUT_DIR / _fname
    print(f'  {"✓" if _p.exists() else "✗"}  {_fname}')
