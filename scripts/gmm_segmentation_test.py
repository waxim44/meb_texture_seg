#!/usr/bin/env python3
"""
gmm_segmentation_test.py
Tester l'effet du desserrage des GMM (PCA réduit + reg_covar) sur la
généralisation à des images MEB jamais vues.
Grille : PCA_DIM ∈ [10, 20] × REG_COVAR ∈ [1e-3, 1e-2, 1e-1]
Seuil : percentile=5 fixe, par texture, vraisemblance absolue.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

# ── Paramètres (faciles à modifier) ───────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DB_PATH      = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
CFG_PATH     = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
IMG_DIR      = ROOT / 'Image_Ouassim'
CHECKPOINT   = ROOT / 'checkpoints' / 'sam2.1_hiera_small_1.pt'   # features pré-extraites
OUTPUT_DIR   = ROOT / 'outputs' / 'gmm_segmentation_test'
SEED         = 42
N_TEST_IMGS  = 3
PERCENTILE   = 5
PCA_DIMS     = [10, 20]
REG_COVARS   = [1e-3, 1e-2, 1e-1]
CATS_EXCLUDE = [2, 8, 10, 11, 12, 13]
MIN_N        = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Résultats → {OUTPUT_DIR}')

_gmmseg_UNKNOWN_COLOR = '#303030'
_gmmseg_UNKNOWN_LABEL = -1

# ── Config ─────────────────────────────────────────────────────────────────────
with open(CFG_PATH) as _f:
    _gmmseg_cfg = json.load(_f)
CATEGORIES = {int(k): v['name']  for k, v in _gmmseg_cfg['available_categories'].items()}
CAT_COLORS = {int(k): v['color'] for k, v in _gmmseg_cfg['available_categories'].items()}


def _gmmseg_hex_rgba(hex_color, alpha=0.5):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


def _gmmseg_draw_patches(ax, img_gray, positions, labels, title, alpha=0.45):
    ax.imshow(img_gray, cmap='gray', aspect='auto')
    for _k in range(len(positions)):
        _x1, _y1, _x2, _y2 = positions[_k].astype(int)
        _lbl = int(labels[_k])
        _col = (_gmmseg_hex_rgba(_gmmseg_UNKNOWN_COLOR, alpha)
                if _lbl == _gmmseg_UNKNOWN_LABEL
                else _gmmseg_hex_rgba(CAT_COLORS.get(_lbl, '#808080'), alpha))
        ax.add_patch(plt.Rectangle(
            (_x1, _y1), _x2 - _x1, _y2 - _y1,
            linewidth=0.7,
            edgecolor=_col[:3] + (1.0,),
            facecolor=_col,
        ))
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis('off')


# ── HDF5 ───────────────────────────────────────────────────────────────────────
with h5py.File(DB_PATH, 'r') as _h5:
    _gmmseg_IMAGE_NAMES  = _h5['metadata/image_names'][:]
    _gmmseg_POSITIONS    = _h5['metadata/positions'][:]
    _gmmseg_CATEGORY_IDS = _h5['metadata/category_ids'][:].astype(int)
    _gmmseg_X_all        = _h5['features']['block_0'][:]

_gmmseg_EXCL_SET = set(CATS_EXCLUDE)
_gmmseg_CATS_VALID = sorted(
    int(c) for c in np.unique(_gmmseg_CATEGORY_IDS)
    if int(c) not in _gmmseg_EXCL_SET
    and (_gmmseg_CATEGORY_IDS == int(c)).sum() >= MIN_N
)
print(f'Catégories valides ({len(_gmmseg_CATS_VALID)}) : '
      f'{[CATEGORIES[c] for c in _gmmseg_CATS_VALID]}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Sélectionner les N_TEST_IMGS images les plus diverses
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 1 — Sélection des images test ===')

_gmmseg_img_info = defaultdict(lambda: {'n_patches': 0, 'cats': set()})
for _i in range(len(_gmmseg_IMAGE_NAMES)):
    if _gmmseg_CATEGORY_IDS[_i] not in _gmmseg_EXCL_SET:
        _nm = _gmmseg_IMAGE_NAMES[_i]
        _gmmseg_img_info[_nm]['n_patches'] += 1
        _gmmseg_img_info[_nm]['cats'].add(int(_gmmseg_CATEGORY_IDS[_i]))

_gmmseg_sorted_imgs = sorted(
    _gmmseg_img_info.items(),
    key=lambda x: (len(x[1]['cats']), x[1]['n_patches']),
    reverse=True,
)
_gmmseg_test_imgs = [_nm for _nm, _ in _gmmseg_sorted_imgs[:N_TEST_IMGS]]

print('Images de test sélectionnées :')
for _nm in _gmmseg_test_imgs:
    _info = _gmmseg_img_info[_nm]
    _cat_names = [CATEGORIES.get(_c, str(_c)) for _c in sorted(_info['cats'])]
    print(f'  {_nm.decode():<60}  cats={len(_info["cats"])}  patches={_info["n_patches"]}')
    print(f'    → {", ".join(_cat_names)}')

# ── Séparer patches train / test ───────────────────────────────────────────────
_gmmseg_test_set = set(_gmmseg_test_imgs)

_gmmseg_train_idx = [
    _i for _i in range(len(_gmmseg_IMAGE_NAMES))
    if _gmmseg_IMAGE_NAMES[_i] not in _gmmseg_test_set
    and _gmmseg_CATEGORY_IDS[_i] not in _gmmseg_EXCL_SET
]
_gmmseg_test_idx = [
    _i for _i in range(len(_gmmseg_IMAGE_NAMES))
    if _gmmseg_IMAGE_NAMES[_i] in _gmmseg_test_set
    and _gmmseg_CATEGORY_IDS[_i] not in _gmmseg_EXCL_SET
]

_gmmseg_X_train_raw  = _gmmseg_X_all[_gmmseg_train_idx]
_gmmseg_y_train_raw  = _gmmseg_CATEGORY_IDS[_gmmseg_train_idx]
_gmmseg_X_test_raw   = _gmmseg_X_all[_gmmseg_test_idx]
_gmmseg_y_test_raw   = _gmmseg_CATEGORY_IDS[_gmmseg_test_idx]
_gmmseg_pos_test     = _gmmseg_POSITIONS[_gmmseg_test_idx]
_gmmseg_names_test   = _gmmseg_IMAGE_NAMES[_gmmseg_test_idx]

print(f'\nTrain : {len(_gmmseg_train_idx)} patches  |  Test : {len(_gmmseg_test_idx)} patches')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Grille PCA_DIM × REG_COVAR
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 2 — Grille de configurations ===')

_gmmseg_grid = {}   # (pca_dim, reg) → dict résultats

for _pca_dim in PCA_DIMS:
    # PCA fit sur train uniquement — réutilisé pour tous les reg de ce PCA_DIM
    _pca = PCA(n_components=_pca_dim, random_state=SEED)
    _Xtr_pca  = _pca.fit_transform(_gmmseg_X_train_raw)
    _Xtr_norm = normalize(_Xtr_pca, norm='l2')
    _Xte_pca  = _pca.transform(_gmmseg_X_test_raw)
    _Xte_norm = normalize(_Xte_pca, norm='l2')
    _var_exp  = _pca.explained_variance_ratio_.sum() * 100

    for _reg in REG_COVARS:
        _key = (_pca_dim, _reg)
        print(f'\n  Config PCA={_pca_dim}d (var={_var_exp:.0f}%)  reg_covar={_reg:.0e}')

        # a) Entraîner un GMM par texture
        _gmms = {}
        for _c in _gmmseg_CATS_VALID:
            _mask_c = _gmmseg_y_train_raw == _c
            if _mask_c.sum() < 2:
                continue
            _gmm = GaussianMixture(
                n_components=1, covariance_type='full',
                reg_covar=_reg, random_state=SEED,
            )
            _gmm.fit(_Xtr_norm[_mask_c])
            _gmms[_c] = _gmm

        # b) Calibrer θ_c = PERCENTILE-ième percentile des log-vraisemblances train
        _thresholds = {}
        for _c, _gmm in _gmms.items():
            _mask_c = _gmmseg_y_train_raw == _c
            _scores_c = _gmm.score_samples(_Xtr_norm[_mask_c])
            _thresholds[_c] = float(np.percentile(_scores_c, PERCENTILE))

        # c) Prédire les patches test
        _n = len(_gmmseg_X_test_raw)
        _log_probs = np.full((_n, len(_gmmseg_CATS_VALID)), -np.inf)
        for _j, _c in enumerate(_gmmseg_CATS_VALID):
            if _c in _gmms:
                _log_probs[:, _j] = _gmms[_c].score_samples(_Xte_norm)

        _cstar_idx = np.argmax(_log_probs, axis=1)
        _cstar     = np.array([_gmmseg_CATS_VALID[_j] for _j in _cstar_idx])
        _cstar_lp  = _log_probs[np.arange(_n), _cstar_idx]
        _theta_v   = np.array([_thresholds.get(_c, -np.inf) for _c in _cstar])
        _preds     = np.where(_cstar_lp >= _theta_v, _cstar, _gmmseg_UNKNOWN_LABEL)

        # d) Métriques
        _known    = _preds != _gmmseg_UNKNOWN_LABEL
        _pct_unk  = (1 - _known.mean()) * 100
        _acc      = (balanced_accuracy_score(_gmmseg_y_test_raw[_known], _preds[_known]) * 100
                     if _known.sum() > 1 else 0.0)

        _gmmseg_grid[_key] = {
            'pct_unknown': _pct_unk,
            'accuracy'   : _acc,
            'preds'      : _preds,
            'thresholds' : _thresholds,
            'pca'        : _pca,
            'gmms'       : _gmms,
            'n_known'    : int(_known.sum()),
            'n_total'    : _n,
            'var_exp'    : _var_exp,
        }
        print(f'    % inconnu={_pct_unk:.1f}%  accuracy={_acc:.1f}%  '
              f'n_reconnus={_known.sum()}/{_n}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Tableau récapitulatif + sélection meilleure config
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 3 — Tableau récapitulatif ===')

_gmmseg_rows = [
    {
        'PCA_DIM'  : _k[0],
        'REG_COVAR': _k[1],
        'pct_unk'  : _v['pct_unknown'],
        'accuracy' : _v['accuracy'],
        'n_known'  : _v['n_known'],
    }
    for _k, _v in _gmmseg_grid.items()
]

# Sélection : priorité aux configs avec accuracy >= 80%
_gmmseg_acc80 = [_r for _r in _gmmseg_rows if _r['accuracy'] >= 80]
if _gmmseg_acc80:
    _gmmseg_best_row = min(_gmmseg_acc80, key=lambda _r: _r['pct_unk'])
else:
    # Compromis : maximiser acc × (100 - pct_unk) = couverture × précision
    _gmmseg_best_row = max(_gmmseg_rows,
                           key=lambda _r: _r['accuracy'] * (100 - _r['pct_unk']))
_gmmseg_best_key = (_gmmseg_best_row['PCA_DIM'], _gmmseg_best_row['REG_COVAR'])

_hdr = (f'{"PCA_DIM":>8}  {"REG_COVAR":>10}  {"% inconnu":>10}  '
        f'{"Accuracy":>10}  {"N reconnus":>12}')
print(_hdr)
print('─' * (len(_hdr) + 4))
for _r in sorted(_gmmseg_rows, key=lambda _r: (_r['PCA_DIM'], _r['REG_COVAR'])):
    _mark = '  ← BEST' if (_r['PCA_DIM'], _r['REG_COVAR']) == _gmmseg_best_key else ''
    print(f'  {_r["PCA_DIM"]:>6}  {_r["REG_COVAR"]:>10.0e}  {_r["pct_unk"]:>9.1f}%  '
          f'{_r["accuracy"]:>9.1f}%  {_r["n_known"]:>12}{_mark}')
print(f'\nMeilleure config : PCA={_gmmseg_best_key[0]}d  '
      f'reg_covar={_gmmseg_best_key[1]:.0e}  '
      f'→ {_gmmseg_best_row["pct_unk"]:.1f}% inconnu  '
      f'{_gmmseg_best_row["accuracy"]:.1f}% accuracy')

# Sauver txt
with open(OUTPUT_DIR / 'grid_results.txt', 'w') as _f:
    _f.write(f'# Grille PCA_DIM × REG_COVAR — percentile={PERCENTILE}\n\n')
    _f.write(_hdr + '\n')
    _f.write('─' * (len(_hdr) + 4) + '\n')
    for _r in sorted(_gmmseg_rows, key=lambda _r: (_r['PCA_DIM'], _r['REG_COVAR'])):
        _mark = '  ← BEST' if (_r['PCA_DIM'], _r['REG_COVAR']) == _gmmseg_best_key else ''
        _f.write(f'  {_r["PCA_DIM"]:>6}  {_r["REG_COVAR"]:>10.0e}  {_r["pct_unk"]:>9.1f}%  '
                 f'{_r["accuracy"]:>9.1f}%  {_r["n_known"]:>12}{_mark}\n')
    _f.write(f'\nMeilleure : PCA={_gmmseg_best_key[0]}d  '
             f'reg={_gmmseg_best_key[1]:.0e}  '
             f'inconnu={_gmmseg_best_row["pct_unk"]:.1f}%  '
             f'accuracy={_gmmseg_best_row["accuracy"]:.1f}%\n')
print('Saved: grid_results.txt')

# Sauver csv
with open(OUTPUT_DIR / 'grid_results.csv', 'w', newline='') as _f:
    _writer = csv.writer(_f)
    _writer.writerow(['PCA_DIM', 'REG_COVAR', 'pct_unknown', 'accuracy', 'n_known', 'best'])
    for _r in sorted(_gmmseg_rows, key=lambda _r: (_r['PCA_DIM'], _r['REG_COVAR'])):
        _writer.writerow([
            _r['PCA_DIM'],
            f'{_r["REG_COVAR"]:.0e}',
            f'{_r["pct_unk"]:.2f}',
            f'{_r["accuracy"]:.2f}',
            _r['n_known'],
            int((_r['PCA_DIM'], _r['REG_COVAR']) == _gmmseg_best_key),
        ])
print('Saved: grid_results.csv')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Heatmaps (% inconnu et accuracy)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 4 — Heatmaps de la grille ===')

_gmmseg_mat_unk = np.zeros((len(PCA_DIMS), len(REG_COVARS)))
_gmmseg_mat_acc = np.zeros((len(PCA_DIMS), len(REG_COVARS)))
for _pi, _pd in enumerate(PCA_DIMS):
    for _ri, _rc in enumerate(REG_COVARS):
        _gmmseg_mat_unk[_pi, _ri] = _gmmseg_grid[(_pd, _rc)]['pct_unknown']
        _gmmseg_mat_acc[_pi, _ri] = _gmmseg_grid[(_pd, _rc)]['accuracy']

_reg_labels = [f'{_r:.0e}' for _r in REG_COVARS]
_pca_labels = [f'PCA-{_d}d' for _d in PCA_DIMS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3, len(PCA_DIMS) * 1.8)))

for _ax, _mat, _title, _cmap in [
    (ax1, _gmmseg_mat_unk, '% Patches inconnus', 'Reds'),
    (ax2, _gmmseg_mat_acc, 'Accuracy (patches reconnus) %', 'Greens'),
]:
    _im = _ax.imshow(_mat, cmap=_cmap, aspect='auto',
                     vmin=_mat.min(), vmax=_mat.max())
    _ax.set_xticks(range(len(REG_COVARS)))
    _ax.set_yticks(range(len(PCA_DIMS)))
    _ax.set_xticklabels(_reg_labels, fontsize=10)
    _ax.set_yticklabels(_pca_labels, fontsize=10)
    _ax.set_xlabel('reg_covar', fontsize=11)
    _ax.set_ylabel('PCA dimension', fontsize=11)
    _ax.set_title(_title, fontsize=11)
    plt.colorbar(_im, ax=_ax, fraction=0.046)
    _vmax = _mat.max()
    for _pi in range(len(PCA_DIMS)):
        for _ri in range(len(REG_COVARS)):
            _v = _mat[_pi, _ri]
            _is_best = (PCA_DIMS[_pi], REG_COVARS[_ri]) == _gmmseg_best_key
            _txt = f'{_v:.1f}%' + (' ★' if _is_best else '')
            _c_txt = 'white' if _v > _vmax * 0.65 else 'black'
            _ax.text(_ri, _pi, _txt, ha='center', va='center',
                     fontsize=9, color=_c_txt,
                     fontweight='bold' if _is_best else 'normal')

fig.suptitle(
    f'Effet du desserrage des GMM — seuil percentile={PERCENTILE}%\n'
    'reg_covar ↑ = gaussienne plus large ; PCA dim ↓ = espace moins pointu\n'
    '★ = meilleure config',
    fontsize=10,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gmm_grid_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: gmm_grid_heatmap.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Visualiser la meilleure config (3 images test)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 5 — Visualisation (meilleure config) ===')

_gmmseg_best_preds = _gmmseg_grid[_gmmseg_best_key]['preds']

for _nm in _gmmseg_test_imgs:
    _mask_img = _gmmseg_names_test == _nm
    if not _mask_img.any():
        continue

    try:
        _img_gray = np.array(Image.open(IMG_DIR / _nm.decode()).convert('L'))
    except Exception as _e:
        print(f'  Image manquante : {_nm.decode()} ({_e})')
        continue

    _pos_img   = _gmmseg_pos_test[_mask_img]
    _ytrue_img = _gmmseg_y_test_raw[_mask_img]
    _pred_img  = _gmmseg_best_preds[_mask_img]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(_img_gray, cmap='gray')
    axes[0].set_title('Image originale', fontsize=9)
    axes[0].axis('off')
    _gmmseg_draw_patches(axes[1], _img_gray, _pos_img, _ytrue_img, 'Vrais labels')
    _gmmseg_draw_patches(
        axes[2], _img_gray, _pos_img, _pred_img,
        f'GMM  PCA-{_gmmseg_best_key[0]}d  reg={_gmmseg_best_key[1]:.0e}\n'
        f'p={PERCENTILE}%  '
        f'({int(_pred_img[_pred_img != _gmmseg_UNKNOWN_LABEL].shape[0])}/'
        f'{len(_pred_img)} reconnus)',
    )

    _handles = [
        mpatches.Patch(color=CAT_COLORS[_c], label=CATEGORIES[_c])
        for _c in _gmmseg_CATS_VALID if _c in CAT_COLORS
    ]
    _handles.append(mpatches.Patch(color=_gmmseg_UNKNOWN_COLOR, label='Inconnu'))
    fig.legend(handles=_handles, loc='lower center', ncol=min(len(_handles), 5),
               fontsize=7, framealpha=0.85, bbox_to_anchor=(0.5, -0.04))

    _stem = Path(_nm.decode()).stem
    fig.suptitle(_stem, fontsize=9, y=1.02)
    plt.tight_layout()
    _out = OUTPUT_DIR / f'gmm_best_{_stem}.png'
    plt.savefig(_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {_out.name}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Matrice de confusion (meilleure config)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 6 — Matrice de confusion (meilleure config) ===')

_known_mask = _gmmseg_best_preds != _gmmseg_UNKNOWN_LABEL
if _known_mask.sum() > 1:
    _cm = confusion_matrix(
        _gmmseg_y_test_raw[_known_mask],
        _gmmseg_best_preds[_known_mask],
        labels=_gmmseg_CATS_VALID,
    )
    _cm_float  = _cm.astype(float)
    _row_sums  = _cm_float.sum(axis=1, keepdims=True)
    _cm_norm   = np.where(_row_sums > 0, _cm_float / _row_sums, 0.0)

    _labels_str = [CATEGORIES[_c] for _c in _gmmseg_CATS_VALID]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(_cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(_gmmseg_CATS_VALID)))
    ax.set_yticks(range(len(_gmmseg_CATS_VALID)))
    ax.set_xticklabels(_labels_str, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(_labels_str, fontsize=9)
    ax.set_xlabel('Prédit', fontsize=11)
    ax.set_ylabel('Vrai', fontsize=11)
    ax.set_title(
        f'Matrice de confusion — meilleure config\n'
        f'PCA={_gmmseg_best_key[0]}d  reg_covar={_gmmseg_best_key[1]:.0e}  '
        f'p={PERCENTILE}%  (normalisée par ligne)',
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, label='Proportion')
    for _i in range(len(_gmmseg_CATS_VALID)):
        for _j in range(len(_gmmseg_CATS_VALID)):
            _v = _cm_norm[_i, _j]
            ax.text(_j, _i, f'{_v:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if _v > 0.6 else 'black')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gmm_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: gmm_confusion_matrix.png')
else:
    print('  Aucun patch reconnu — matrice de confusion ignorée')

# ── Résumé final ───────────────────────────────────────────────────────────────
print('\n' + '═' * 60)
print('RÉSUMÉ FINAL')
print('═' * 60)
print(f'\nImages test ({N_TEST_IMGS} plus diverses) :')
for _nm in _gmmseg_test_imgs:
    _info = _gmmseg_img_info[_nm]
    print(f'  {_nm.decode()}  ({len(_info["cats"])} cat., {_info["n_patches"]} patches)')

print(f'\nGrille (PERCENTILE={PERCENTILE}%, seuil par texture) :')
print(f'{"PCA_DIM":>8}  {"REG_COVAR":>10}  {"% inconnu":>10}  {"Accuracy":>10}')
print('─' * 44)
for _r in sorted(_gmmseg_rows, key=lambda _r: (_r['PCA_DIM'], _r['REG_COVAR'])):
    _mark = '  ← BEST' if (_r['PCA_DIM'], _r['REG_COVAR']) == _gmmseg_best_key else ''
    print(f'  {_r["PCA_DIM"]:>6}  {_r["REG_COVAR"]:>10.0e}  '
          f'{_r["pct_unk"]:>9.1f}%  {_r["accuracy"]:>9.1f}%{_mark}')
print(f'\nMeilleure config : PCA={_gmmseg_best_key[0]}d  '
      f'reg_covar={_gmmseg_best_key[1]:.0e}')
print(f'  % inconnu = {_gmmseg_best_row["pct_unk"]:.1f}%')
print(f'  Accuracy  = {_gmmseg_best_row["accuracy"]:.1f}%')

print(f'\n=== Fichiers dans {OUTPUT_DIR} ===')
for _p in sorted(OUTPUT_DIR.iterdir()):
    print(f'  {_p.name}')
