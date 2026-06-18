#!/usr/bin/env python3
"""
clahe_test_meb.py
Teste si CLAHE sur images Ouassim brutes récupère la performance perdue.
Hypothèse : chute PatchTagger→Ouassim due à l'absence de rehaussement contraste.

Protocole :
  - Référence haute  : PatchTagger (database_meb.h5)
  - Référence basse  : Ouassim brut (database_meb_ouassim.h5)
  - Test CLAHE       : clipLimit ∈ [2.0, 3.0, 4.0], appliqué sur niveaux de gris
                       AVANT convert RGB + resize (même pipeline que preprocess())
  Blocks testés : block_0, block_4, stage_3_fpn
  Métriques     : balanced accuracy 5-fold par image + recall par catégorie
"""

import json, sys, warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy.stats import mode as _sp_mode
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
_clt_ROOT        = Path(__file__).resolve().parents[1]
_clt_DB_OUA      = _clt_ROOT / 'data' / 'feature_database' / 'database_meb_ouassim.h5'
_clt_DB_PT       = _clt_ROOT / 'data' / 'feature_database' / 'database_meb.h5'
_clt_CFG_PATH    = _clt_ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
_clt_IMG_DIR     = _clt_ROOT / 'Image_Ouassim'
_clt_CKPT        = _clt_ROOT / 'checkpoints' / 'sam2.1_hiera_small_1.pt'
_clt_OUTPUT_DIR  = _clt_ROOT / 'output_ouassim' / 'clahe_test'
_clt_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_clt_BLOCKS      = ['block_0', 'block_4', 'stage_3_fpn']
_clt_CLIP_LIMITS = [2.0, 3.0, 4.0]
_clt_TILE        = (8, 8)
_clt_SEED        = 42
_clt_PCA_DIM     = 50
_clt_N_FOLDS     = 5
_clt_CATS_EXCL   = {2, 8, 10, 11, 12, 13}
_clt_MIN_N       = 30

np.random.seed(_clt_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Config + métadonnées
# ─────────────────────────────────────────────────────────────────────────────
with open(_clt_CFG_PATH) as _f:
    _clt_cfg = json.load(_f)
_clt_CATEGORIES = {int(k): v['name'] for k, v in _clt_cfg['available_categories'].items()}

with h5py.File(_clt_DB_OUA, 'r') as _h5:
    _clt_ALL_NAMES    = _h5['metadata/image_names'][:]
    _clt_ALL_CATS     = _h5['metadata/category_ids'][:].astype(int)
    _clt_ALL_POSITIONS = _h5['metadata/positions'][:]

# Filtre catégories valides
_clt_CATS_VALID = sorted(
    int(c) for c in np.unique(_clt_ALL_CATS)
    if int(c) not in _clt_CATS_EXCL
    and (_clt_ALL_CATS == int(c)).sum() >= _clt_MIN_N
)
_clt_ALL_NAMES = np.array([
    n.decode('utf-8') if isinstance(n, (bytes, np.bytes_)) else str(n)
    for n in _clt_ALL_NAMES
])
_clt_mask  = np.isin(_clt_ALL_CATS, _clt_CATS_VALID)
_clt_y     = _clt_ALL_CATS[_clt_mask]
_clt_imgs  = _clt_ALL_NAMES[_clt_mask]
_clt_pos   = _clt_ALL_POSITIONS[_clt_mask]   # (x_min, y_min, x_max, y_max)
_clt_N_CATS = len(_clt_CATS_VALID)
_clt_CAT_LABELS = [_clt_CATEGORIES[c] for c in _clt_CATS_VALID]

print(f'Patches valides : {_clt_mask.sum()}, {_clt_N_CATS} catégories')
print(f'Catégories : {_clt_CAT_LABELS}')

# ─────────────────────────────────────────────────────────────────────────────
# Folds (stratifiés par image — même protocole que structure_block4)
# ─────────────────────────────────────────────────────────────────────────────
_clt_imgs_uniq = np.unique(_clt_imgs)
_clt_cat_dom   = np.array([
    int(_sp_mode(_clt_y[_clt_imgs == _img]).mode)
    for _img in _clt_imgs_uniq
])
_clt_skf   = StratifiedKFold(n_splits=_clt_N_FOLDS, shuffle=True, random_state=_clt_SEED)
_clt_FOLDS = list(_clt_skf.split(_clt_imgs_uniq, _clt_cat_dom))


# ─────────────────────────────────────────────────────────────────────────────
# LP helper
# ─────────────────────────────────────────────────────────────────────────────
def _clt_run_lp(X_raw, y, imgs):
    y_true_all, y_pred_all, accs = [], [], []
    for tr_i, te_i in _clt_FOLDS:
        tr_imgs = _clt_imgs_uniq[tr_i]
        te_imgs = _clt_imgs_uniq[te_i]
        m_tr = np.isin(imgs, tr_imgs)
        m_te = np.isin(imgs, te_imgs)
        if m_te.sum() == 0:
            continue
        n = min(_clt_PCA_DIM, X_raw.shape[1])
        pca = PCA(n_components=n, random_state=_clt_SEED)
        Xtr = pca.fit_transform(X_raw[m_tr])
        Xte = pca.transform(X_raw[m_te])
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                                 random_state=_clt_SEED)
        clf.fit(Xtr, y[m_tr])
        pred = clf.predict(Xte)
        y_true_all.extend(y[m_te].tolist())
        y_pred_all.extend(pred.tolist())
        accs.append(balanced_accuracy_score(y[m_te], pred))
    yt, yp = np.array(y_true_all), np.array(y_pred_all)
    rep = classification_report(yt, yp, labels=_clt_CATS_VALID,
                                output_dict=True, zero_division=0)
    recall = {c: rep.get(str(c), {}).get('recall', 0.0) * 100
              for c in _clt_CATS_VALID}
    return float(np.mean(accs)) * 100, recall


# ─────────────────────────────────────────────────────────────────────────────
# Résultats référence (depuis HDF5 existants)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Références (HDF5 existants) ──')
_clt_ref_bacc   = {}   # {label: {block: bacc}}
_clt_ref_recall = {}   # {label: {block: {cat: recall}}}

for _label, _db_path in [('PatchTagger', _clt_DB_PT), ('Ouassim_brut', _clt_DB_OUA)]:
    _clt_ref_bacc[_label]   = {}
    _clt_ref_recall[_label] = {}
    with h5py.File(_db_path, 'r') as _h5:
        for _blk in _clt_BLOCKS:
            print(f'  LP {_label}/{_blk} ...', end=' ', flush=True)
            X = _h5['features'][_blk][:].astype(np.float32)[_clt_mask]
            bacc, rec = _clt_run_lp(X, _clt_y, _clt_imgs)
            _clt_ref_bacc[_label][_blk]   = bacc
            _clt_ref_recall[_label][_blk] = rec
            print(f'{bacc:.1f}%')

# ─────────────────────────────────────────────────────────────────────────────
# Chargement modèle (import direct depuis build_feature_database.py)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(_clt_ROOT))
from build_feature_database import (
    load_model             as _bfdb_load_model,
    register_hooks         as _bfdb_register_hooks,
    remove_hooks           as _bfdb_remove_hooks,
    extract_patch_features as _bfdb_extract_patch,
)

_clt_IMG_SIZE = 1024
_clt_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_clt_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_clt_device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('\nChargement modèle...')
_clt_encoder = _bfdb_load_model(_clt_CKPT, str(_clt_device))
_clt_captured, _clt_handles = _bfdb_register_hooks(_clt_encoder)


def _clt_preprocess_clahe(img_path: Path, clip_limit: float,
                           tile_grid: tuple) -> tuple:
    """
    Charge image → CLAHE sur grayscale → convert RGB → resize → normalize.
    Retourne (tensor, orig_H, orig_W).
    """
    img_pil = Image.open(img_path)
    orig_w, orig_h = img_pil.size
    # Convertir en grayscale numpy pour CLAHE
    gray = np.array(img_pil.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(gray)
    # RGB = stack du canal grayscale amélioré
    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    resized = cv2.resize(rgb, (_clt_IMG_SIZE, _clt_IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _clt_MEAN) / _clt_STD
    return x.unsqueeze(0).to(_clt_device), orig_h, orig_w


# ─────────────────────────────────────────────────────────────────────────────
# Extraction features CLAHE pour chaque clipLimit
# ─────────────────────────────────────────────────────────────────────────────
_clt_clahe_bacc   = {}   # {clip: {block: bacc}}
_clt_clahe_recall = {}   # {clip: {block: {cat: recall}}}

try:
    for _clip in _clt_CLIP_LIMITS:
        print(f'\n── CLAHE clipLimit={_clip} ──')
        _clt_clahe_bacc[_clip]   = {}
        _clt_clahe_recall[_clip] = {}

        # Extraire features pour toutes les images valides
        _clt_feat_buf = {blk: [] for blk in _clt_BLOCKS}
        _img_list = list(np.unique(_clt_imgs))

        for _img_name in tqdm(_img_list, desc=f'  CLAHE {_clip}'):
            _img_path = _clt_IMG_DIR / _img_name
            if not _img_path.exists():
                continue

            _tensor, _orig_H, _orig_W = _clt_preprocess_clahe(
                _img_path, _clip, _clt_TILE)

            with torch.no_grad():
                _clt_encoder(_tensor)

            # Patches de cette image
            _idx = np.where(_clt_imgs == _img_name)[0]
            for _pi in _idx:
                _x_min, _y_min, _x_max, _y_max = _clt_pos[_pi]
                _p = {'x_min': int(_x_min), 'y_min': int(_y_min),
                      'x_max': int(_x_max), 'y_max': int(_y_max)}
                _feats = _bfdb_extract_patch(_clt_captured, _p, _orig_H, _orig_W)
                for _blk in _clt_BLOCKS:
                    _clt_feat_buf[_blk].append(_feats.get(_blk))

        # LP sur chaque block
        for _blk in _clt_BLOCKS:
            X = np.stack(_clt_feat_buf[_blk], axis=0).astype(np.float32)
            print(f'  LP {_blk} ...', end=' ', flush=True)
            bacc, rec = _clt_run_lp(X, _clt_y, _clt_imgs)
            _clt_clahe_bacc[_clip][_blk]   = bacc
            _clt_clahe_recall[_clip][_blk] = rec
            print(f'{bacc:.1f}%')

finally:
    _bfdb_remove_hooks(_clt_handles)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Barplot LP : PatchTagger / Ouassim brut / CLAHE×3
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 1 — clahe_comparison.png')

_clt_LABELS = (
    ['PatchTagger', 'Ouassim_brut']
    + [f'CLAHE {c}' for c in _clt_CLIP_LIMITS]
)
_clt_N_CONDS = len(_clt_LABELS)
_clt_N_BLKS  = len(_clt_BLOCKS)
_clt_x_pos   = np.arange(_clt_N_BLKS)
_clt_width   = 0.14
_clt_offsets = np.linspace(-(_clt_N_CONDS-1)/2, (_clt_N_CONDS-1)/2, _clt_N_CONDS) * _clt_width

_clt_COLORS_BAR = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
_clt_HATCHES    = ['', '///', '', '..', 'xx']

fig1, ax1 = plt.subplots(figsize=(11, 5))

for _li, _lab in enumerate(_clt_LABELS):
    if _lab == 'PatchTagger':
        _d = _clt_ref_bacc['PatchTagger']
    elif _lab == 'Ouassim_brut':
        _d = _clt_ref_bacc['Ouassim_brut']
    else:
        _clip_val = float(_lab.split()[1])
        _d = _clt_clahe_bacc[_clip_val]
    _vals = [_d.get(_b, 0.0) for _b in _clt_BLOCKS]
    ax1.bar(_clt_x_pos + _clt_offsets[_li], _vals,
            width=_clt_width, label=_lab,
            color=_clt_COLORS_BAR[_li], hatch=_clt_HATCHES[_li],
            edgecolor='white', linewidth=0.5, alpha=0.88)

# Ligne baseline
ax1.axhline(100.0 / _clt_N_CATS, color='grey', linestyle=':', linewidth=1,
            label=f'Baseline ({100/_clt_N_CATS:.1f}%)')

ax1.set_xticks(_clt_x_pos)
ax1.set_xticklabels(_clt_BLOCKS, fontsize=11)
ax1.set_ylabel('Balanced accuracy (%)', fontsize=11)
ax1.set_title('Impact de CLAHE sur les features Ouassim\n(5-fold LP par image)',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig(_clt_OUTPUT_DIR / 'clahe_comparison.png', dpi=150)
plt.close(fig1)
print('  Saved.')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Recall par catégorie (block_4) : brut vs meilleur CLAHE
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2 — recall_clahe_stratifie.png')

# Trouver clipLimit qui maximise la balanced acc sur block_4
_clt_best_clip = max(_clt_CLIP_LIMITS,
                     key=lambda c: _clt_clahe_bacc[c].get('block_4', 0))
print(f'  Meilleur CLAHE (block_4) : clipLimit={_clt_best_clip}  '
      f'({_clt_clahe_bacc[_clt_best_clip]["block_4"]:.1f}%)')

_clt_x_cats = np.arange(_clt_N_CATS)
_clt_w = 0.22
_clt_sources = [
    ('PatchTagger',                    '#2ecc71', _clt_ref_recall['PatchTagger']['block_4']),
    ('Ouassim brut',                   '#e74c3c', _clt_ref_recall['Ouassim_brut']['block_4']),
    (f'CLAHE {_clt_best_clip}',        '#3498db', _clt_clahe_recall[_clt_best_clip]['block_4']),
    (f'CLAHE {_clt_CLIP_LIMITS[-1]}',  '#9b59b6',
     _clt_clahe_recall[_clt_CLIP_LIMITS[-1]]['block_4']),
]

fig2, ax2 = plt.subplots(figsize=(13, 5))
_off4 = np.linspace(-1.5, 1.5, 4) * _clt_w
for _si, (_lab, _col, _rec_d) in enumerate(_clt_sources):
    _vals = [_rec_d.get(c, 0.0) for c in _clt_CATS_VALID]
    ax2.bar(_clt_x_cats + _off4[_si], _vals, width=_clt_w,
            label=_lab, color=_col, alpha=0.85, edgecolor='white', linewidth=0.5)

ax2.axhline(100.0 / _clt_N_CATS, color='grey', linestyle=':', linewidth=1)
ax2.set_xticks(_clt_x_cats)
ax2.set_xticklabels(_clt_CAT_LABELS, rotation=25, ha='right', fontsize=9)
ax2.set_ylabel('Recall (%)', fontsize=11)
ax2.set_title('Recall par catégorie — block_4\n(PatchTagger / Ouassim brut / CLAHE)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()
fig2.savefig(_clt_OUTPUT_DIR / 'recall_clahe_stratifie.png', dpi=150)
plt.close(fig2)
print('  Saved.')


# ─────────────────────────────────────────────────────────────────────────────
# Texte — clahe_results.txt
# ─────────────────────────────────────────────────────────────────────────────
print('Génération clahe_results.txt...')

_clt_SEP  = '=' * 72
_clt_SEP2 = '─' * 72
_lines    = [
    _clt_SEP,
    'RÉSULTATS TEST CLAHE  —  Ouassim brut → CLAHE',
    _clt_SEP,
    f'Images source  : {_clt_IMG_DIR}',
    f'CLAHE tile     : {_clt_TILE}',
    f'Protocole      : PCA-{_clt_PCA_DIM}d → LR balanced, 5-fold par image, SEED={_clt_SEED}',
    '',
    _clt_SEP2,
    'BALANCED ACCURACY PAR BLOCK',
    _clt_SEP2,
    f'{"Condition":<22}' + ''.join(f'{b:>16}' for b in _clt_BLOCKS),
    _clt_SEP2,
]

for _lab in _clt_LABELS:
    if _lab == 'PatchTagger':
        _d = _clt_ref_bacc['PatchTagger']
    elif _lab == 'Ouassim_brut':
        _d = _clt_ref_bacc['Ouassim_brut']
    else:
        _clip_val = float(_lab.split()[1])
        _d = _clt_clahe_bacc[_clip_val]
    _row = f'{_lab:<22}' + ''.join(f'{_d.get(b, 0.0):>15.1f}%' for b in _clt_BLOCKS)
    _lines.append(_row)

_lines += ['', _clt_SEP2, 'RÉCUPÉRATION CLAHE SUR block_4', _clt_SEP2]

_bacc_brut   = _clt_ref_bacc['Ouassim_brut']['block_4']
_bacc_pt     = _clt_ref_bacc['PatchTagger']['block_4']
_gap_total   = _bacc_pt - _bacc_brut
for _clip in _clt_CLIP_LIMITS:
    _bacc_c = _clt_clahe_bacc[_clip]['block_4']
    _gain   = _bacc_c - _bacc_brut
    _pct    = _gain / _gap_total * 100 if _gap_total > 0 else 0.0
    _lines.append(
        f'CLAHE {_clip:.1f} : {_bacc_c:.1f}%  '
        f'({_gain:+.1f} pts vs brut = {_pct:+.0f}% du gap récupéré)'
    )

_lines += ['', _clt_SEP2,
           'RECALL STRATIFIÉ RECTILIGNE (block_4)',
           _clt_SEP2]

# Trouver l'id de Stratifié rectiligne
_clt_strat_id = None
for _cid, _cname in _clt_CATEGORIES.items():
    if 'rectiligne' in _cname.lower() and _cid in _clt_CATS_VALID:
        _clt_strat_id = _cid
        break

if _clt_strat_id is not None:
    _lines.append(f'Catégorie : {_clt_CATEGORIES[_clt_strat_id]}')
    _r_pt    = _clt_ref_recall['PatchTagger']['block_4'].get(_clt_strat_id, 0)
    _r_brut  = _clt_ref_recall['Ouassim_brut']['block_4'].get(_clt_strat_id, 0)
    _lines.append(f'  PatchTagger  : {_r_pt:.1f}%')
    _lines.append(f'  Ouassim brut : {_r_brut:.1f}%')
    for _clip in _clt_CLIP_LIMITS:
        _r_c = _clt_clahe_recall[_clip]['block_4'].get(_clt_strat_id, 0)
        _lines.append(f'  CLAHE {_clip:.1f}    : {_r_c:.1f}%')
else:
    _lines.append('Catégorie Stratifié rectiligne non trouvée dans les catégories valides.')

# ── Verdict final ──
_best_gains = {c: _clt_clahe_bacc[c]['block_4'] - _bacc_brut
               for c in _clt_CLIP_LIMITS}
_best_c     = max(_best_gains, key=_best_gains.get)
_best_gain  = _best_gains[_best_c]
_best_pct   = _best_gain / _gap_total * 100 if _gap_total > 0 else 0.0

_lines += ['', _clt_SEP2, 'VERDICT', _clt_SEP2]

# Récupération Stratifié rectiligne (catégorie la plus sensible)
_strat_gain = None
if _clt_strat_id is not None:
    _r_brut_s  = _clt_ref_recall['Ouassim_brut']['block_4'].get(_clt_strat_id, 0)
    _r_clahe_s = _clt_clahe_recall[_best_c]['block_4'].get(_clt_strat_id, 0)
    _strat_gain = _r_clahe_s - _r_brut_s

if _best_gain >= 5.0 and _best_pct >= 20:
    _verdict = (
        f'CLAHE AIDE GLOBALEMENT (meilleur clipLimit={_best_c}) : '
        f'{_best_gain:+.1f} pts ({_best_pct:+.0f}% du gap récupéré).\n'
        f'Le manque de contraste explique PARTIELLEMENT la chute PatchTagger→Ouassim.\n'
        f'Recommandation : appliquer CLAHE {_best_c} en pré-traitement standard.'
    )
elif _best_gain > 0:
    _verdict = (
        f'CLAHE aide marginalement en global (meilleur clipLimit={_best_c}) : '
        f'{_best_gain:+.1f} pts ({_best_pct:+.0f}% du gap récupéré).\n'
        f'La chute vient d\'autres facteurs (couleur RGB, artefacts de traitement,\n'
        f'domaine-shift plus profond). CLAHE seul ne suffit pas.'
    )
else:
    _verdict = (
        f'CLAHE N\'AIDE PAS GLOBALEMENT (meilleur clipLimit={_best_c}) : '
        f'{_best_gain:+.1f} pts vs Ouassim brut.\n'
        f'La chute globale n\'est pas due à l\'absence de rehaussement de contraste.\n'
        f'Explorer : fine-tuning du modèle ou augmentation de données MEB.'
    )

if _strat_gain is not None and _strat_gain >= 15:
    _verdict += (
        f'\n\nNOTE IMPORTANTE — Stratifié rectiligne (catégorie la plus affectée) :\n'
        f'  recall brut={_r_brut_s:.1f}%  →  CLAHE {_best_c}={_r_clahe_s:.1f}%  '
        f'({_strat_gain:+.1f} pts).\n'
        f'Cette texture profite significativement du rehaussement de contraste,\n'
        f'mais d\'autres catégories se dégradent, annulant le gain global.\n'
        f'Piste : pré-traitement sélectif ou fine-tuning ciblé sur cette classe.'
    )

_lines.append(_verdict)
_lines += ['', _clt_SEP]

_clt_txt = '\n'.join(_lines)
print(_clt_txt)

with open(_clt_OUTPUT_DIR / 'clahe_results.txt', 'w') as _fout:
    _fout.write(_clt_txt + '\n')

print(f'\nFichiers générés dans {_clt_OUTPUT_DIR}:')
for _fn in ['clahe_comparison.png', 'recall_clahe_stratifie.png', 'clahe_results.txt']:
    _p = _clt_OUTPUT_DIR / _fn
    print(f'  {"✓" if _p.exists() else "✗"}  {_fn}')
