#!/usr/bin/env python3
"""
compare_descriptors_meb.py
Comparer block_0 (TextureSAM) avec LBP, GLCM, Gabor sur les mêmes
patches MEB. Sortie : tableau comparatif (Linear Probing + Fisher)
+ figures + CSV dans OUTPUT_DIR.
"""

# ── Dépendances optionnelles ───────────────────────────────────────────────────
import subprocess, sys
for _pkg, _import in [('scikit-image', 'skimage'), ('tqdm', 'tqdm')]:
    try:
        __import__(_import)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg])

# ── Imports ───────────────────────────────────────────────────────────────────
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import mode as _scipy_mode
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DB_PATH    = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
CFG_PATH   = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUTPUT_DIR = ROOT / 'outputs' / 'comparison_descriptors'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Résultats → {OUTPUT_DIR}')

# ── Hyperparamètres ───────────────────────────────────────────────────────────
_cmp_SEED        = 42
_cmp_PCA_DIM     = 50
_cmp_N_FOLDS     = 5
_cmp_CATS_EXCL   = {2, 8, 10, 11, 12, 13}
_cmp_MIN_PATCHES = 30

# ── Chargement config + DB ────────────────────────────────────────────────────
with open(CFG_PATH) as _f:
    _cmp_cfg = json.load(_f)
CATEGORIES = {int(k): v['name'] for k, v in _cmp_cfg['available_categories'].items()}

with h5py.File(DB_PATH, 'r') as _h5:
    _cmp_IMAGE_NAMES  = _h5['metadata/image_names'][:]
    _cmp_POSITIONS    = _h5['metadata/positions'][:]
    _cmp_CATEGORY_IDS = _h5['metadata/category_ids'][:].astype(int)

_cmp_CATS_VALID = sorted(
    int(c) for c in np.unique(_cmp_CATEGORY_IDS)
    if int(c) not in _cmp_CATS_EXCL
    and (_cmp_CATEGORY_IDS == int(c)).sum() >= _cmp_MIN_PATCHES
)
_cmp_mask_valid = np.isin(_cmp_CATEGORY_IDS, _cmp_CATS_VALID)

print(f'Catégories valides : {[CATEGORIES[c] for c in _cmp_CATS_VALID]}')
print(f'Patches valides    : {_cmp_mask_valid.sum()}')

# ─────────────────────────────────────────────────────────────────────────────
# Descripteurs classiques
# ─────────────────────────────────────────────────────────────────────────────

def _cmp_extract_lbp(patch_gray):
    """LBP uniforme P=8, R=1 → histogramme normalisé (~10d)."""
    _P, _R = 8, 1
    _lbp = local_binary_pattern(patch_gray, _P, _R, method='uniform')
    _hist, _ = np.histogram(_lbp, bins=_P + 2, range=(0, _P + 2), density=True)
    return _hist


def _cmp_extract_glcm(patch_gray):
    """GLCM → 5 props × 2 distances × 4 angles = 40d."""
    _patch_q = (patch_gray / 256 * 32).astype(np.uint8)
    _distances = [1, 2]
    _angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    _glcm = graycomatrix(
        _patch_q, _distances, _angles,
        levels=32, symmetric=True, normed=True,
    )
    _props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    _feats = []
    for _p in _props:
        _feats.extend(graycoprops(_glcm, _p).flatten())
    return np.array(_feats)


# Banque Gabor pré-calculée : 4 orientations × 4 fréquences = 16 filtres
_cmp_gabor_kernels = [
    np.real(gabor_kernel(freq, theta=theta))
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for freq in [0.1, 0.2, 0.3, 0.4]
]


def _cmp_extract_gabor(patch_gray):
    """Réponses moyenne + std à 16 filtres Gabor = 32d."""
    _patch_f = patch_gray.astype(float)
    _feats = []
    for _kernel in _cmp_gabor_kernels:
        _filtered = ndi.convolve(_patch_f, _kernel, mode='wrap')
        _feats.append(_filtered.mean())
        _feats.append(_filtered.std())
    return np.array(_feats)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction pour tous les patches (groupé par image)
# ─────────────────────────────────────────────────────────────────────────────

_cmp_patches_by_img = defaultdict(list)
for _i in np.where(_cmp_mask_valid)[0]:
    _cmp_patches_by_img[_cmp_IMAGE_NAMES[_i]].append(int(_i))

_cmp_feats_lbp   = {}
_cmp_feats_glcm  = {}
_cmp_feats_gabor = {}

for _img_name in tqdm(_cmp_patches_by_img, desc='Extraction descripteurs'):
    try:
        _img = Image.open(IMG_DIR / _img_name.decode()).convert('L')
    except Exception as _e:
        print(f'  Image manquante : {_img_name} ({_e})')
        continue
    _img_np = np.array(_img)

    for _idx in _cmp_patches_by_img[_img_name]:
        _pos = _cmp_POSITIONS[_idx].astype(int)
        _x1, _y1, _x2, _y2 = _pos
        _patch = _img_np[_y1:_y2, _x1:_x2]

        if _patch.shape[0] < 8 or _patch.shape[1] < 8:
            continue

        _cmp_feats_lbp[_idx]   = _cmp_extract_lbp(_patch)
        _cmp_feats_glcm[_idx]  = _cmp_extract_glcm(_patch)
        _cmp_feats_gabor[_idx] = _cmp_extract_gabor(_patch)

# ─────────────────────────────────────────────────────────────────────────────
# Assembler les matrices
# ─────────────────────────────────────────────────────────────────────────────

_cmp_valid_idx  = sorted(_cmp_feats_lbp.keys())
_cmp_y_valid    = _cmp_CATEGORY_IDS[_cmp_valid_idx]
_cmp_imgs_valid = _cmp_IMAGE_NAMES[_cmp_valid_idx]

_cmp_X_lbp   = np.array([_cmp_feats_lbp[_i]   for _i in _cmp_valid_idx])
_cmp_X_glcm  = np.array([_cmp_feats_glcm[_i]  for _i in _cmp_valid_idx])
_cmp_X_gabor = np.array([_cmp_feats_gabor[_i] for _i in _cmp_valid_idx])

with h5py.File(DB_PATH, 'r') as _h5:
    _cmp_X_block0_all = _h5['features']['block_0'][:]
_cmp_X_block0 = _cmp_X_block0_all[_cmp_valid_idx]

REPRESENTATIONS = {
    'block_0 (TextureSAM)': _cmp_X_block0,
    'LBP':                  _cmp_X_lbp,
    'GLCM':                 _cmp_X_glcm,
    'Gabor':                _cmp_X_gabor,
}

print('\nDimensions :')
for _name, _X in REPRESENTATIONS.items():
    print(f'  {_name:<22} : {_X.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# Protocole d'évaluation — K-Fold stratifié par image
# ─────────────────────────────────────────────────────────────────────────────

_cmp_images_uniq = np.unique(_cmp_imgs_valid)
_cmp_cat_dom = np.array([
    int(_scipy_mode(_cmp_y_valid[_cmp_imgs_valid == _img]).mode)
    for _img in _cmp_images_uniq
])
_cmp_skf   = StratifiedKFold(n_splits=_cmp_N_FOLDS, shuffle=True,
                              random_state=_cmp_SEED)
_cmp_FOLDS = list(_cmp_skf.split(_cmp_images_uniq, _cmp_cat_dom))


def _cmp_pca_or_keep(X, X_tr_idx, X_te_idx):
    """PCA-50d si dim > PCA_DIM, sinon copie brute."""
    if X.shape[1] > _cmp_PCA_DIM:
        _pca = PCA(n_components=_cmp_PCA_DIM, random_state=_cmp_SEED)
        return _pca.fit_transform(X[X_tr_idx]), _pca.transform(X[X_te_idx])
    return X[X_tr_idx].copy(), X[X_te_idx].copy()


def _cmp_eval_linear_probing(X):
    _accs = []
    for _tr_i, _te_i in _cmp_FOLDS:
        _tr_imgs = _cmp_images_uniq[_tr_i]
        _te_imgs = _cmp_images_uniq[_te_i]
        _m_tr = np.isin(_cmp_imgs_valid, _tr_imgs)
        _m_te = np.isin(_cmp_imgs_valid, _te_imgs)

        _X_tr, _X_te = _cmp_pca_or_keep(X, _m_tr, _m_te)

        _scaler = StandardScaler()
        _X_tr = _scaler.fit_transform(_X_tr)
        _X_te = _scaler.transform(_X_te)

        _clf = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=_cmp_SEED, n_jobs=-1,
        )
        _clf.fit(_X_tr, _cmp_y_valid[_m_tr])
        _accs.append(balanced_accuracy_score(
            _cmp_y_valid[_m_te], _clf.predict(_X_te)
        ))
    return float(np.mean(_accs)), float(np.std(_accs))


def _cmp_eval_fisher(X):
    _X50 = (
        PCA(n_components=_cmp_PCA_DIM, random_state=_cmp_SEED).fit_transform(X)
        if X.shape[1] > _cmp_PCA_DIM else X.copy()
    )
    _mu  = _X50.mean(axis=0)
    _D   = _X50.shape[1]
    _S_B = np.zeros((_D, _D))
    _S_W = np.zeros((_D, _D))
    for _c in _cmp_CATS_VALID:
        _mask = _cmp_y_valid == _c
        _N_c  = _mask.sum()
        _mu_c = _X50[_mask].mean(axis=0)
        _diff = (_mu_c - _mu).reshape(-1, 1)
        _S_B += _diff @ _diff.T
        _dc   = _X50[_mask] - _mu_c
        _S_W += (1.0 / _N_c) * (_dc.T @ _dc)
    return float(np.trace(_S_B) / (np.trace(_S_W) + 1e-10))


def _cmp_eval_per_cat(X):
    """Recall moyen par catégorie sur les K folds."""
    _per_cat = {_c: [] for _c in _cmp_CATS_VALID}
    for _tr_i, _te_i in _cmp_FOLDS:
        _tr_imgs = _cmp_images_uniq[_tr_i]
        _te_imgs = _cmp_images_uniq[_te_i]
        _m_tr = np.isin(_cmp_imgs_valid, _tr_imgs)
        _m_te = np.isin(_cmp_imgs_valid, _te_imgs)

        _X_tr, _X_te = _cmp_pca_or_keep(X, _m_tr, _m_te)
        _scaler = StandardScaler()
        _X_tr = _scaler.fit_transform(_X_tr)
        _X_te = _scaler.transform(_X_te)

        _clf = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=_cmp_SEED, n_jobs=-1,
        )
        _clf.fit(_X_tr, _cmp_y_valid[_m_tr])
        _rep = classification_report(
            _cmp_y_valid[_m_te], _clf.predict(_X_te),
            labels=_cmp_CATS_VALID, output_dict=True, zero_division=0,
        )
        for _c in _cmp_CATS_VALID:
            _per_cat[_c].append(_rep.get(str(_c), {}).get('recall', 0.0))
    return {_c: float(np.mean(_per_cat[_c])) * 100 for _c in _cmp_CATS_VALID}


# ─────────────────────────────────────────────────────────────────────────────
# Calcul comparatif
# ─────────────────────────────────────────────────────────────────────────────

_cmp_results = {}
for _name, _X in REPRESENTATIONS.items():
    print(f'\nÉvaluation : {_name} (dim={_X.shape[1]})...')
    _lp_mean, _lp_std = _cmp_eval_linear_probing(_X)
    _fisher            = _cmp_eval_fisher(_X)
    _per_cat           = _cmp_eval_per_cat(_X)
    _cmp_results[_name] = {
        'dim'    : _X.shape[1],
        'lp_mean': _lp_mean * 100,
        'lp_std' : _lp_std * 100,
        'fisher' : _fisher,
        'per_cat': _per_cat,
    }
    print(f'  LP = {_lp_mean*100:.1f}% ± {_lp_std*100:.1f}  ·  Fisher = {_fisher:.2f}')

# ─────────────────────────────────────────────────────────────────────────────
# Tableau final (console + fichiers)
# ─────────────────────────────────────────────────────────────────────────────

_cmp_baseline = 100.0 / len(_cmp_CATS_VALID)
_cmp_best     = max(_cmp_results, key=lambda k: _cmp_results[k]['lp_mean'])

_cmp_lines = [
    '=' * 70,
    'TABLEAU COMPARATIF — block_0 vs descripteurs classiques',
    '=' * 70,
    f'\nBaseline aléatoire : {_cmp_baseline:.1f}%',
    f'Patches : {len(_cmp_valid_idx)} · Catégories : {len(_cmp_CATS_VALID)}\n',
    f'{"Méthode":<22} │ {"Dim":>5} │ {"Linear Probing":>16} │ {"Fisher J":>9}',
    '─' * 62,
]
for _name in sorted(_cmp_results, key=lambda k: -_cmp_results[k]['lp_mean']):
    _r = _cmp_results[_name]
    _cmp_lines.append(
        f'{_name:<22} │ {_r["dim"]:>5} │ '
        f'{_r["lp_mean"]:>6.1f}% ± {_r["lp_std"]:>4.1f} │ '
        f'{_r["fisher"]:>9.2f}'
    )
_cmp_lines.append(
    f'\n→ Meilleure méthode : {_cmp_best} ({_cmp_results[_cmp_best]["lp_mean"]:.1f}%)'
)

_cmp_lines += [
    '\n' + '=' * 70,
    'RECALL PAR CATÉGORIE (%)',
    '=' * 70,
]
_cmp_repr_names = list(REPRESENTATIONS.keys())
_cmp_header = f'{"Catégorie":<22} │ ' + ' │ '.join(
    f'{_n.split()[0]:>10}' for _n in _cmp_repr_names
)
_cmp_lines.append(_cmp_header)
_cmp_lines.append('─' * len(_cmp_header))
for _c in _cmp_CATS_VALID:
    _row = f'{CATEGORIES[_c]:<22} │ '
    _row += ' │ '.join(
        f'{_cmp_results[_n]["per_cat"][_c]:>9.1f}%' for _n in _cmp_repr_names
    )
    _cmp_lines.append(_row)

_cmp_texte = '\n'.join(_cmp_lines)
print('\n' + _cmp_texte)

with open(OUTPUT_DIR / 'results_table.txt', 'w') as _f:
    _f.write(_cmp_texte + '\n')

with open(OUTPUT_DIR / 'results.csv', 'w', newline='') as _f:
    _writer = csv.writer(_f)
    _writer.writerow(['methode', 'dim', 'lp_mean', 'lp_std', 'fisher'])
    for _name, _r in _cmp_results.items():
        _writer.writerow([
            _name, _r['dim'],
            f'{_r["lp_mean"]:.2f}', f'{_r["lp_std"]:.2f}',
            f'{_r["fisher"]:.4f}',
        ])

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_cmp_names  = list(_cmp_results.keys())
_cmp_colors = ['#1B4F72', '#E67E22', '#27AE60', '#8E44AD']

# ── Figure 1 : Linear Probing + Fisher ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

_cmp_lp_vals = [_cmp_results[_n]['lp_mean'] for _n in _cmp_names]
_cmp_lp_errs = [_cmp_results[_n]['lp_std']  for _n in _cmp_names]
_cmp_bars = ax1.bar(
    _cmp_names, _cmp_lp_vals, yerr=_cmp_lp_errs,
    capsize=5, color=_cmp_colors,
)
ax1.axhline(_cmp_baseline, color='red', ls=':', lw=1.5,
            label=f'Baseline {_cmp_baseline:.1f}%')
ax1.set_ylabel('Balanced Accuracy (%)', fontsize=11)
ax1.set_title('Linear Probing', fontsize=12)
ax1.set_xticklabels(_cmp_names, rotation=20, ha='right', fontsize=9)
ax1.legend(fontsize=9)
for _bar, _v in zip(_cmp_bars, _cmp_lp_vals):
    ax1.text(
        _bar.get_x() + _bar.get_width() / 2, _v + 1,
        f'{_v:.1f}%', ha='center', fontsize=9, fontweight='bold',
    )

_cmp_fisher_vals = [_cmp_results[_n]['fisher'] for _n in _cmp_names]
ax2.bar(_cmp_names, _cmp_fisher_vals, color=_cmp_colors)
ax2.set_ylabel('Fisher J (log)', fontsize=11)
ax2.set_title('Fisher Criterion', fontsize=12)
ax2.set_xticklabels(_cmp_names, rotation=20, ha='right', fontsize=9)
ax2.set_yscale('log')

plt.suptitle(
    'TextureSAM (block_0) vs Descripteurs classiques\n'
    'Mêmes patches · même protocole 5-fold stratifié',
    fontsize=13,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'comparison_main.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: comparison_main.png')

# ── Figure 2 : Heatmap recall par catégorie ────────────────────────────────────
_cmp_mat = np.array([
    [_cmp_results[_n]['per_cat'][_c] for _n in _cmp_names]
    for _c in _cmp_CATS_VALID
])

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(_cmp_mat, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(_cmp_names)))
ax.set_xticklabels(_cmp_names, rotation=20, ha='right', fontsize=9)
ax.set_yticks(range(len(_cmp_CATS_VALID)))
ax.set_yticklabels([CATEGORIES[_c] for _c in _cmp_CATS_VALID], fontsize=9)
for _i in range(len(_cmp_CATS_VALID)):
    for _j in range(len(_cmp_names)):
        ax.text(
            _j, _i, f'{_cmp_mat[_i, _j]:.0f}',
            ha='center', va='center', fontsize=8,
            color='black' if _cmp_mat[_i, _j] > 40 else 'white',
        )
plt.colorbar(im, ax=ax, fraction=0.04, label='Recall (%)')
ax.set_title('Recall par catégorie et par méthode', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'comparison_per_category.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: comparison_per_category.png')

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde features + résultats
# ─────────────────────────────────────────────────────────────────────────────

with open(OUTPUT_DIR / 'classical_features.pkl', 'wb') as _f:
    pickle.dump({
        'lbp':       _cmp_X_lbp,
        'glcm':      _cmp_X_glcm,
        'gabor':     _cmp_X_gabor,
        'valid_idx': _cmp_valid_idx,
        'y_valid':   _cmp_y_valid,
    }, _f)

with open(OUTPUT_DIR / 'results.pkl', 'wb') as _f:
    pickle.dump(_cmp_results, _f)

print(f'\n=== Fichiers générés dans {OUTPUT_DIR} ===')
for _fname in [
    'comparison_main.png', 'comparison_per_category.png',
    'results_table.txt', 'results.csv',
    'results.pkl', 'classical_features.pkl',
]:
    _p = OUTPUT_DIR / _fname
    print(f'  {"✓" if _p.exists() else "✗"}  {_fname}')
