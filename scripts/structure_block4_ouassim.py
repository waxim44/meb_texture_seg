#!/usr/bin/env python3
"""
structure_block4_ouassim.py
Analyse de structure sur block_4 (meilleur LP Ouassim) :
  1. Matrice de confusion (5-fold LP cumulées)
  2. Recall par catégorie — comparaison PatchTagger vs Ouassim
  3. Manifold UMAP (PCA-50d → UMAP-2d)
  4. Matrice de connectivité inter-catégories
  5. structure_summary.txt
"""

import csv, json, sys, warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from scipy.stats import mode as _sp_mode
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
_str_ROOT       = Path(__file__).resolve().parents[1]
_str_DB_OUA     = _str_ROOT / 'data' / 'feature_database' / 'database_meb_ouassim.h5'
_str_DB_PT      = _str_ROOT / 'data' / 'feature_database' / 'database_meb.h5'
_str_CFG_PATH   = _str_ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
_str_OUTPUT_DIR = _str_ROOT / 'output_ouassim'
_str_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_str_KEY       = 'block_4'
_str_SEED      = 42
_str_PCA_DIM   = 50
_str_N_FOLDS   = 5
_str_CATS_EXCL = {2, 8, 10, 11, 12, 13}
_str_MIN_N     = 30

np.random.seed(_str_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement config + métadonnées
# ─────────────────────────────────────────────────────────────────────────────
with open(_str_CFG_PATH) as _f:
    _str_cfg = json.load(_f)
_str_CATEGORIES = {int(k): v['name'] for k, v in _str_cfg['available_categories'].items()}
_str_CAT_COLORS = {int(k): v['color'] for k, v in _str_cfg['available_categories'].items()}

with h5py.File(_str_DB_OUA, 'r') as _h5:
    _str_NAMES = _h5['metadata/image_names'][:]
    _str_CATS  = _h5['metadata/category_ids'][:].astype(int)

_str_CATS_VALID = sorted(
    int(c) for c in np.unique(_str_CATS)
    if int(c) not in _str_CATS_EXCL
    and (_str_CATS == int(c)).sum() >= _str_MIN_N
)
_str_mask = np.isin(_str_CATS, _str_CATS_VALID)
_str_y    = _str_CATS[_str_mask]
_str_imgs = _str_NAMES[_str_mask]

_str_N_CATS  = len(_str_CATS_VALID)
_str_BASELINE = 100.0 / _str_N_CATS
_str_CAT_LABELS = [_str_CATEGORIES[c] for c in _str_CATS_VALID]

# Couleurs stables par catégorie
def _str_col(c):
    raw = _str_CAT_COLORS.get(c, '#888888')
    try:
        return mcolors.to_hex(raw)
    except Exception:
        return '#888888'

_str_COLORS = [_str_col(c) for c in _str_CATS_VALID]

print(f'Base Ouassim : {_str_mask.sum()} patches, {_str_N_CATS} catégories')
print(f'Catégories   : {_str_CAT_LABELS}')

# ─────────────────────────────────────────────────────────────────────────────
# Folds (stratifiés par image)
# ─────────────────────────────────────────────────────────────────────────────
_str_imgs_uniq = np.unique(_str_imgs)
_str_cat_dom   = np.array([
    int(_sp_mode(_str_y[_str_imgs == _img]).mode)
    for _img in _str_imgs_uniq
])
_str_skf   = StratifiedKFold(n_splits=_str_N_FOLDS, shuffle=True, random_state=_str_SEED)
_str_FOLDS = list(_str_skf.split(_str_imgs_uniq, _str_cat_dom))


# ─────────────────────────────────────────────────────────────────────────────
# Helper : 5-fold LP → (y_true_all, y_pred_all, recall_per_cat, bal_acc)
# ─────────────────────────────────────────────────────────────────────────────
def _str_run_lp(X_raw, y, imgs):
    _y_true_all, _y_pred_all = [], []
    _accs = []
    for _tr_i, _te_i in _str_FOLDS:
        _tr_imgs = _str_imgs_uniq[_tr_i]
        _te_imgs = _str_imgs_uniq[_te_i]
        _m_tr = np.isin(imgs, _tr_imgs)
        _m_te = np.isin(imgs, _te_imgs)
        if _m_te.sum() == 0:
            continue

        _n = min(_str_PCA_DIM, X_raw.shape[1])
        _pca = PCA(n_components=_n, random_state=_str_SEED)
        _Xtr = _pca.fit_transform(X_raw[_m_tr])
        _Xte = _pca.transform(X_raw[_m_te])

        _sc = StandardScaler()
        _Xtr = _sc.fit_transform(_Xtr)
        _Xte = _sc.transform(_Xte)

        _clf = LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=_str_SEED,
        )
        _clf.fit(_Xtr, y[_m_tr])
        _pred = _clf.predict(_Xte)
        _y_true_all.extend(y[_m_te].tolist())
        _y_pred_all.extend(_pred.tolist())
        _accs.append(balanced_accuracy_score(y[_m_te], _pred))

    _yt = np.array(_y_true_all)
    _yp = np.array(_y_pred_all)
    _rep = classification_report(
        _yt, _yp, labels=_str_CATS_VALID,
        output_dict=True, zero_division=0,
    )
    _recall = {c: _rep.get(str(c), {}).get('recall', 0.0) * 100
               for c in _str_CATS_VALID}
    return _yt, _yp, _recall, float(np.mean(_accs)) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Ouassim — block_4
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLP Ouassim ({_str_KEY})...')
with h5py.File(_str_DB_OUA, 'r') as _h5:
    _str_X_oua = _h5['features'][_str_KEY][:].astype(np.float32)[_str_mask]

_str_yt_oua, _str_yp_oua, _str_rec_oua, _str_bacc_oua = _str_run_lp(
    _str_X_oua, _str_y, _str_imgs)
print(f'  Balanced accuracy Ouassim : {_str_bacc_oua:.1f}%')

# ─────────────────────────────────────────────────────────────────────────────
# PatchTagger — block_4 (même folds, même protocole)
# ─────────────────────────────────────────────────────────────────────────────
print(f'LP PatchTagger ({_str_KEY})...')
with h5py.File(_str_DB_PT, 'r') as _h5:
    _str_X_pt = _h5['features'][_str_KEY][:].astype(np.float32)[_str_mask]

_str_yt_pt, _str_yp_pt, _str_rec_pt, _str_bacc_pt = _str_run_lp(
    _str_X_pt, _str_y, _str_imgs)
print(f'  Balanced accuracy PatchTagger : {_str_bacc_pt:.1f}%')

# ─────────────────────────────────────────────────────────────────────────────
# PCA-50d commun pour UMAP + confusion
# ─────────────────────────────────────────────────────────────────────────────
_str_n_pca = min(_str_PCA_DIM, _str_X_oua.shape[1])
_str_pca   = PCA(n_components=_str_n_pca, random_state=_str_SEED)
_str_X50   = _str_pca.fit_transform(_str_X_oua)
_str_norms = np.linalg.norm(_str_X50, axis=1, keepdims=True)
_str_X50n  = _str_X50 / np.where(_str_norms < 1e-8, 1.0, _str_norms)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Matrice de confusion Ouassim (normalisée par ligne = recall)
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 1 — confusion_block4.png')
_str_cm = confusion_matrix(_str_yt_oua, _str_yp_oua,
                            labels=_str_CATS_VALID)
_str_cm_norm = _str_cm.astype(float) / (_str_cm.sum(axis=1, keepdims=True) + 1e-8)

fig1, ax1 = plt.subplots(figsize=(8, 7))
_im1 = ax1.imshow(_str_cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(_im1, ax=ax1, fraction=0.04, label='Recall (fraction)')

for _i in range(_str_N_CATS):
    for _j in range(_str_N_CATS):
        _v = _str_cm_norm[_i, _j]
        _raw = _str_cm[_i, _j]
        _col = 'white' if _v > 0.55 else 'black'
        ax1.text(_j, _i, f'{_v:.2f}\n({_raw})',
                 ha='center', va='center', fontsize=7.5, color=_col)

ax1.set_xticks(range(_str_N_CATS))
ax1.set_yticks(range(_str_N_CATS))
ax1.set_xticklabels(_str_CAT_LABELS, rotation=35, ha='right', fontsize=9)
ax1.set_yticklabels(_str_CAT_LABELS, fontsize=9)
ax1.set_xlabel('Prédit', fontsize=10)
ax1.set_ylabel('Réel', fontsize=10)
ax1.set_title(
    f'Matrice de confusion — {_str_KEY} Ouassim\n'
    f'5-fold par image  |  Balanced Accuracy = {_str_bacc_oua:.1f}%',
    fontsize=11,
)
plt.tight_layout()
fig1.savefig(_str_OUTPUT_DIR / 'confusion_block4.png', dpi=150, bbox_inches='tight')
plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Recall par catégorie : PatchTagger vs Ouassim
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2 — recall_par_categorie_comparaison.png')
_str_rec_oua_vals = [_str_rec_oua[c] for c in _str_CATS_VALID]
_str_rec_pt_vals  = [_str_rec_pt[c]  for c in _str_CATS_VALID]
_str_deltas       = [o - p for o, p in zip(_str_rec_oua_vals, _str_rec_pt_vals)]

_str_xr = np.arange(_str_N_CATS)
_str_wr = 0.35

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 9),
                                    gridspec_kw={'height_ratios': [3, 1]})

# Barplot comparatif
_bars_pt  = ax2a.bar(_str_xr - _str_wr/2, _str_rec_pt_vals,  _str_wr,
                      label=f'PatchTagger ({_str_bacc_pt:.1f}%)',
                      color='#1B4F72', alpha=0.85)
_bars_oua = ax2a.bar(_str_xr + _str_wr/2, _str_rec_oua_vals, _str_wr,
                      label=f'Ouassim ({_str_bacc_oua:.1f}%)',
                      color='#E63946', alpha=0.85)
ax2a.axhline(50, color='gray', ls='--', lw=1, alpha=0.6, label='Seuil 50%')
ax2a.axhline(_str_BASELINE, color='gray', ls=':', lw=1, alpha=0.5,
             label=f'Baseline {_str_BASELINE:.1f}%')
ax2a.set_xticks(_str_xr)
ax2a.set_xticklabels(_str_CAT_LABELS, fontsize=10)
ax2a.set_ylabel('Recall (%)', fontsize=11)
ax2a.set_ylim(0, 110)
ax2a.set_title(
    f'Recall par catégorie — {_str_KEY}  (PatchTagger vs Ouassim)\n'
    'Vert = tient sur Ouassim  |  Rouge = s\'effondre',
    fontsize=11,
)
ax2a.legend(fontsize=9, loc='upper right')
ax2a.grid(axis='y', alpha=0.25)
for _b, _v in zip(_bars_oua, _str_rec_oua_vals):
    ax2a.text(_b.get_x() + _b.get_width()/2, _v + 1.5,
              f'{_v:.0f}', ha='center', fontsize=8, color='#E63946', fontweight='bold')

# Delta subplot
_str_dcols = ['#2D6A4F' if d >= -10 else '#C1121F' for d in _str_deltas]
ax2b.bar(_str_xr, _str_deltas, color=_str_dcols, alpha=0.85)
ax2b.axhline(0, color='black', lw=0.8)
ax2b.axhline(-10, color='orange', ls=':', lw=1, alpha=0.7)
ax2b.set_xticks(_str_xr)
ax2b.set_xticklabels(_str_CAT_LABELS, fontsize=10)
ax2b.set_ylabel('ΔRecall (Ouassim − PT)', fontsize=9)
ax2b.grid(axis='y', alpha=0.25)
for _bi, (_v, _col) in enumerate(zip(_str_deltas, _str_dcols)):
    ax2b.text(_bi, _v + (1 if _v >= 0 else -3),
              f'{_v:+.0f}', ha='center', fontsize=8, color=_col, fontweight='bold')

plt.tight_layout()
fig2.savefig(_str_OUTPUT_DIR / 'recall_par_categorie_comparaison.png',
             dpi=150, bbox_inches='tight')
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — UMAP manifold
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 3 — manifold_block4.png (UMAP en cours...)')
_str_reducer = umap.UMAP(
    n_components=2, n_neighbors=20, min_dist=0.15,
    metric='cosine', random_state=_str_SEED,
)
_str_X_umap = _str_reducer.fit_transform(_str_X50n)

fig3, ax3 = plt.subplots(figsize=(9, 7))
for _ci, _c in enumerate(_str_CATS_VALID):
    _mask_c = _str_y == _c
    _pts    = _str_X_umap[_mask_c]
    ax3.scatter(_pts[:, 0], _pts[:, 1],
                c=_str_COLORS[_ci], s=22, alpha=0.65, zorder=3,
                label=f'{_str_CATEGORIES[_c]} (n={_mask_c.sum()})',
                edgecolors='none')

# Centroïdes annotés
for _ci, _c in enumerate(_str_CATS_VALID):
    _mask_c = _str_y == _c
    _cx, _cy = _str_X_umap[_mask_c].mean(axis=0)
    ax3.text(_cx, _cy, _str_CATEGORIES[_c].split()[0],
             fontsize=8, fontweight='bold',
             color=_str_COLORS[_ci],
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, lw=0))

ax3.set_xlabel('UMAP-1', fontsize=10)
ax3.set_ylabel('UMAP-2', fontsize=10)
ax3.set_title(
    f'Manifold UMAP — {_str_KEY} Ouassim (PCA-{_str_n_pca}d → UMAP-2d, cosine)\n'
    'Îlots séparés = textures distinctes  |  Mélange = confusion',
    fontsize=11,
)
ax3.legend(fontsize=8, loc='best', framealpha=0.85, markerscale=1.5)
ax3.grid(alpha=0.2)
plt.tight_layout()
fig3.savefig(_str_OUTPUT_DIR / 'manifold_block4.png', dpi=150, bbox_inches='tight')
plt.close(fig3)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Matrice de connectivité inter-catégories
# (confusion normalisée symétriquement : combien de fois A et B se confondent)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 4 — connectivity_block4.png')

# Matrice de confusion brute → connectivité = (C[i,j] + C[j,i]) / total
_str_conn = (_str_cm + _str_cm.T).astype(float)
np.fill_diagonal(_str_conn, 0)
_str_conn_norm = _str_conn / (_str_conn.max() + 1e-10)

fig4, ax4 = plt.subplots(figsize=(8, 7))
_im4 = ax4.imshow(_str_conn_norm, cmap='Reds', vmin=0, vmax=1)
plt.colorbar(_im4, ax=ax4, fraction=0.04, label='Confusion symétrique (normalisée)')

for _i in range(_str_N_CATS):
    for _j in range(_str_N_CATS):
        if _i == _j:
            continue
        _v = _str_conn[_i, _j]
        _n = _str_conn_norm[_i, _j]
        _col = 'white' if _n > 0.55 else 'black'
        ax4.text(_j, _i, f'{int(_v)}', ha='center', va='center',
                 fontsize=8, color=_col)

ax4.set_xticks(range(_str_N_CATS))
ax4.set_yticks(range(_str_N_CATS))
ax4.set_xticklabels(_str_CAT_LABELS, rotation=35, ha='right', fontsize=9)
ax4.set_yticklabels(_str_CAT_LABELS, fontsize=9)
ax4.set_title(
    f'Connectivité inter-catégories — {_str_KEY} Ouassim\n'
    'C[i,j] = (erreurs i→j) + (erreurs j→i)  (rouge intense = fort mélange)',
    fontsize=10,
)
# Identifier les 3 paires les plus confuses
_str_pairs_conf = []
for _i in range(_str_N_CATS):
    for _j in range(_i + 1, _str_N_CATS):
        _str_pairs_conf.append((_str_conn[_i, _j], _i, _j))
_str_pairs_conf.sort(reverse=True)

for _rank, (_val, _i, _j) in enumerate(_str_pairs_conf[:3]):
    ax4.add_patch(plt.Rectangle((_j - 0.5, _i - 0.5), 1, 1,
                                  fill=False, edgecolor='blue', lw=2.5))
    ax4.add_patch(plt.Rectangle((_i - 0.5, _j - 0.5), 1, 1,
                                  fill=False, edgecolor='blue', lw=2.5))

ax4.text(0.02, 0.01,
         'Cadres bleus = 3 paires les plus confuses',
         transform=ax4.transAxes, fontsize=8, color='blue')

plt.tight_layout()
fig4.savefig(_str_OUTPUT_DIR / 'connectivity_block4.png', dpi=150, bbox_inches='tight')
plt.close(fig4)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict texte
# ─────────────────────────────────────────────────────────────────────────────
print('\nGénération structure_summary.txt...')

_str_top_oua = sorted(_str_CATS_VALID, key=lambda c: -_str_rec_oua[c])
_str_bot_oua = sorted(_str_CATS_VALID, key=lambda c: _str_rec_oua[c])
_str_stable  = [c for c in _str_CATS_VALID if abs(_str_rec_oua[c] - _str_rec_pt[c]) < 10]
_str_degraded = sorted(_str_CATS_VALID, key=lambda c: _str_rec_oua[c] - _str_rec_pt[c])

_str_top_pairs = _str_pairs_conf[:5]

_str_lines = [
    '=' * 80,
    f'ANALYSE DE STRUCTURE — {_str_KEY}  (Ouassim vs PatchTagger)',
    '=' * 80,
    f'Base Ouassim     : {_str_mask.sum()} patches valides, {_str_N_CATS} catégories',
    f'Baseline         : {_str_BASELINE:.1f}%',
    f'Balanced Acc Ouassim     : {_str_bacc_oua:.1f}%',
    f'Balanced Acc PatchTagger : {_str_bacc_pt:.1f}%',
    f'Chute globale            : {_str_bacc_oua - _str_bacc_pt:+.1f} pts',
    '',
    '─' * 80,
    'RECALL PAR CATÉGORIE',
    '─' * 80,
    f'{"Catégorie":<22} {"PatchTagger":>12} {"Ouassim":>10} {"Δ":>8}  Statut',
    '─' * 60,
]
for _c in _str_top_oua:
    _rpt  = _str_rec_pt[_c]
    _roua = _str_rec_oua[_c]
    _d    = _roua - _rpt
    _status = ('TIENT ✓' if _roua >= 50
               else ('>baseline' if _roua > _str_BASELINE * 1.5
               else 'CHUTE ✗'))
    _str_lines.append(
        f'{_str_CATEGORIES[_c]:<22} {_rpt:>10.1f}% {_roua:>9.1f}% {_d:>+7.1f}  {_status}'
    )

_str_lines += [
    '',
    '─' * 80,
    'TEXTURES QUI TIENNENT (recall ≥ 50% sur Ouassim)',
    '─' * 80,
]
_str_holds = [c for c in _str_CATS_VALID if _str_rec_oua[c] >= 50]
if _str_holds:
    for _c in sorted(_str_holds, key=lambda _k: -_str_rec_oua[_k]):
        _str_lines.append(
            f'  {_str_CATEGORIES[_c]:<22} recall={_str_rec_oua[_c]:.1f}%  '
            f'(Δ={_str_rec_oua[_c]-_str_rec_pt[_c]:+.1f} vs PatchTagger)'
        )
else:
    _str_lines.append('  Aucune catégorie ne dépasse 50% de recall sur Ouassim.')

_str_lines += [
    '',
    '─' * 80,
    'TEXTURES QUI S\'EFFONDRENT (recall < baseline × 1.5)',
    '─' * 80,
]
_str_crashes = [c for c in _str_CATS_VALID if _str_rec_oua[c] < _str_BASELINE * 1.5]
for _c in sorted(_str_crashes, key=lambda _k: _str_rec_oua[_k]):
    _str_lines.append(
        f'  {_str_CATEGORIES[_c]:<22} recall={_str_rec_oua[_c]:.1f}%  '
        f'(vs {_str_rec_pt[_c]:.1f}% sur PatchTagger)'
    )

_str_lines += [
    '',
    '─' * 80,
    'PAIRES LES PLUS CONFUSES (Ouassim)',
    '─' * 80,
]
for _rank, (_val, _i, _j) in enumerate(_str_top_pairs, 1):
    _ci, _cj = _str_CATS_VALID[_i], _str_CATS_VALID[_j]
    _str_lines.append(
        f'  {_rank}. {_str_CATEGORIES[_ci]:<22} ↔ {_str_CATEGORIES[_cj]:<22} : '
        f'{int(_val)} erreurs cumulées'
    )

_str_lines += [
    '',
    '─' * 80,
    'INTERPRÉTATION',
    '─' * 80,
    f'La chute de {_str_bacc_oua - _str_bacc_pt:+.1f} pts de balanced accuracy (PatchTagger → Ouassim)',
    'suggère que block_4 encode partiellement des propriétés d\'image (contraste,',
    'luminosité, traitement RGB) en plus de la texture pure.',
    '',
    'Les textures qui TIENNENT sur Ouassim ont des signatures texturales',
    'suffisamment distinctes pour survivre au changement de rendu.',
    '',
    'Les textures qui CHUTENT sont ambiguës en grayscale : leurs différences',
    'étaient visibles grâce au traitement PatchTagger (ex. accentuation de contraste).',
    '',
    '→ Recommandation : normalisation de contraste CLAHE sur images Ouassim,',
    '  ou fine-tuning du modèle sur ce domaine.',
    '=' * 80,
]

_str_summary = '\n'.join(_str_lines)
print('\n' + _str_summary)
with open(_str_OUTPUT_DIR / 'structure_summary.txt', 'w') as _f:
    _f.write(_str_summary + '\n')

# ─────────────────────────────────────────────────────────────────────────────
# Bilan fichiers
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nFichiers générés dans {_str_OUTPUT_DIR} :')
for _fname in [
    'confusion_block4.png',
    'recall_par_categorie_comparaison.png',
    'manifold_block4.png',
    'connectivity_block4.png',
    'structure_summary.txt',
]:
    _p = _str_OUTPUT_DIR / _fname
    print(f'  {"✓" if _p.exists() else "✗"}  {_fname}')
