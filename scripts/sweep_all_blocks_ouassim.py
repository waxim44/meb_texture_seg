#!/usr/bin/env python3
"""
sweep_all_blocks_ouassim.py
Balayage complet des 16 blocks trunk + 4 FPN sur la base Ouassim.
Identifie le vrai meilleur encodeur textural sur les images MEB brutes.

Métriques :
  - Linear Probing  : balanced accuracy, 5-fold stratifié par image
  - Fisher J        : séparabilité inter/intra-classe (balancé, PCA-50d)
  - τ cross/intra  : cosine sim cross-image vs intra-image → généralisation
"""

import csv, json, sys, warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mode as _sp_mode
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
_swp_ROOT       = Path(__file__).resolve().parents[1]
_swp_DB_PATH    = _swp_ROOT / 'data' / 'feature_database' / 'database_meb_ouassim.h5'
_swp_CFG_PATH   = _swp_ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
_swp_OUTPUT_DIR = _swp_ROOT / 'output_ouassim'
_swp_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_swp_SEED        = 42
_swp_PCA_DIM     = 50
_swp_N_FOLDS     = 5
_swp_CATS_EXCL   = {2, 8, 10, 11, 12, 13}
_swp_MIN_N       = 30

# Ordre architectural complet
_swp_ALL_KEYS = (
    [f'block_{i}' for i in range(16)]
    + ['stage_1_fpn', 'stage_2_fpn', 'stage_3_fpn', 'stage_4_fpn']
)
_swp_BLOCK_KEYS = [f'block_{i}' for i in range(16)]
_swp_FPN_KEYS   = ['stage_1_fpn', 'stage_2_fpn', 'stage_3_fpn', 'stage_4_fpn']

np.random.seed(_swp_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des métadonnées
# ─────────────────────────────────────────────────────────────────────────────
print('Chargement des métadonnées...')
with open(_swp_CFG_PATH) as _f:
    _swp_cfg = json.load(_f)
_swp_CATEGORIES = {int(k): v['name'] for k, v in _swp_cfg['available_categories'].items()}

with h5py.File(_swp_DB_PATH, 'r') as _h5:
    _swp_ALL_NAMES = _h5['metadata/image_names'][:]
    _swp_ALL_CATS  = _h5['metadata/category_ids'][:].astype(int)

_swp_CATS_VALID = sorted(
    int(c) for c in np.unique(_swp_ALL_CATS)
    if int(c) not in _swp_CATS_EXCL
    and (_swp_ALL_CATS == int(c)).sum() >= _swp_MIN_N
)
_swp_mask       = np.isin(_swp_ALL_CATS, _swp_CATS_VALID)
_swp_y          = _swp_ALL_CATS[_swp_mask]
_swp_imgs       = _swp_ALL_NAMES[_swp_mask]   # bytes

_swp_N_CATS  = len(_swp_CATS_VALID)
_swp_BASELINE = 100.0 / _swp_N_CATS

print(f'  Base    : {_swp_DB_PATH.name}')
print(f'  Patches valides : {_swp_mask.sum()} ({_swp_N_CATS} catégories)')
print(f'  Catégories : {[_swp_CATEGORIES[c] for c in _swp_CATS_VALID]}')
print(f'  Baseline : {_swp_BASELINE:.1f}%')

# ─────────────────────────────────────────────────────────────────────────────
# Construction des folds (stratifié par image → jamais de fuite de patches)
# ─────────────────────────────────────────────────────────────────────────────
_swp_imgs_uniq = np.unique(_swp_imgs)
_swp_cat_dom   = np.array([
    int(_sp_mode(_swp_y[_swp_imgs == _img]).mode)
    for _img in _swp_imgs_uniq
])
_swp_skf   = StratifiedKFold(n_splits=_swp_N_FOLDS, shuffle=True, random_state=_swp_SEED)
_swp_FOLDS = list(_swp_skf.split(_swp_imgs_uniq, _swp_cat_dom))

# ─────────────────────────────────────────────────────────────────────────────
# Fonctions métriques
# ─────────────────────────────────────────────────────────────────────────────

def _swp_pca_l2(X_raw, n_comp=_swp_PCA_DIM, seed=_swp_SEED):
    """PCA-min(n_comp, dim)d → L2-normalise."""
    n = min(n_comp, X_raw.shape[1])
    Xp = PCA(n_components=n, random_state=seed).fit_transform(X_raw)
    norms = np.linalg.norm(Xp, axis=1, keepdims=True)
    return Xp / np.where(norms < 1e-8, 1.0, norms)


def _swp_compute_lp(X_raw):
    """Linear Probing : balanced accuracy, 5-fold, PCA-50d, L2-norm, StandardScaler."""
    _accs = []
    for _tr_i, _te_i in _swp_FOLDS:
        _tr_imgs = _swp_imgs_uniq[_tr_i]
        _te_imgs = _swp_imgs_uniq[_te_i]
        _m_tr = np.isin(_swp_imgs, _tr_imgs)
        _m_te = np.isin(_swp_imgs, _te_imgs)
        if _m_te.sum() == 0:
            continue

        _n = min(_swp_PCA_DIM, X_raw.shape[1])
        _pca = PCA(n_components=_n, random_state=_swp_SEED)
        _Xtr = _pca.fit_transform(X_raw[_m_tr])
        _Xte = _pca.transform(X_raw[_m_te])

        _sc = StandardScaler()
        _Xtr = _sc.fit_transform(_Xtr)
        _Xte = _sc.transform(_Xte)

        _clf = LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=_swp_SEED,
        )
        _clf.fit(_Xtr, _swp_y[_m_tr])
        _accs.append(balanced_accuracy_score(_swp_y[_m_te], _clf.predict(_Xte)))
    return float(np.mean(_accs)) * 100, float(np.std(_accs)) * 100


def _swp_compute_fisher(X_raw):
    """Fisher J balancé (S_W divisé par N_c) sur PCA-50d."""
    _n = min(_swp_PCA_DIM, X_raw.shape[1])
    _X50 = PCA(n_components=_n, random_state=_swp_SEED).fit_transform(X_raw)
    _mu  = _X50.mean(axis=0)
    _D   = _X50.shape[1]
    _S_B = np.zeros((_D, _D))
    _S_W = np.zeros((_D, _D))
    for _c in _swp_CATS_VALID:
        _mask = _swp_y == _c
        _N_c  = _mask.sum()
        _mu_c = _X50[_mask].mean(axis=0)
        _diff = (_mu_c - _mu).reshape(-1, 1)
        _S_B += _diff @ _diff.T
        _dc   = _X50[_mask] - _mu_c
        _S_W += (1.0 / _N_c) * (_dc.T @ _dc)
    return float(np.trace(_S_B) / (np.trace(_S_W) + 1e-10))


def _swp_compute_tau(X_raw):
    """τ cross/intra : cosine sim cross-image vs intra-image, macro sur catégories.
    Retourne (τ_cross, τ_intra, ratio) tous macro-moyennés."""
    _Xn = _swp_pca_l2(X_raw)
    _cross_vals, _intra_vals = [], []

    for _c in _swp_CATS_VALID:
        _mask_c = _swp_y == _c
        _Xc     = _Xn[_mask_c]
        _imgs_c = _swp_imgs[_mask_c]
        _N_c    = _Xc.shape[0]

        _sim   = _Xc @ _Xc.T
        _upper = np.triu(np.ones((_N_c, _N_c), dtype=bool), k=1)
        _m_cr  = (_imgs_c[:, None] != _imgs_c[None, :]) & _upper
        _m_in  = (_imgs_c[:, None] == _imgs_c[None, :]) & _upper

        if _m_cr.any():
            _cross_vals.append(float(_sim[_m_cr].mean()))
        if _m_in.any():
            _intra_vals.append(float(_sim[_m_in].mean()))

    _tau_cross = float(np.mean(_cross_vals)) if _cross_vals else np.nan
    _tau_intra = float(np.mean(_intra_vals)) if _intra_vals else np.nan
    _ratio     = _tau_cross / (_tau_intra + 1e-8) if _intra_vals else np.nan
    return _tau_cross, _tau_intra, _ratio


# ─────────────────────────────────────────────────────────────────────────────
# Sweep principal
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nSweep : {len(_swp_ALL_KEYS)} représentations...')
_swp_results = {}

with h5py.File(_swp_DB_PATH, 'r') as _h5:
    for _key in tqdm(_swp_ALL_KEYS, desc='Blocks', unit='bloc'):
        if _key not in _h5['features']:
            tqdm.write(f'  {_key} ABSENT — ignoré')
            continue

        _X_raw = _h5['features'][_key][:].astype(np.float32)[_swp_mask]
        _dim   = _X_raw.shape[1]

        _lp, _lp_std          = _swp_compute_lp(_X_raw)
        _fisher               = _swp_compute_fisher(_X_raw)
        _tau_cr, _tau_in, _rt = _swp_compute_tau(_X_raw)

        _swp_results[_key] = {
            'dim':       _dim,
            'lp':        _lp,
            'lp_std':    _lp_std,
            'fisher':    _fisher,
            'tau_cross': _tau_cr,
            'tau_intra': _tau_in,
            'tau_ratio': _rt,
        }
        tqdm.write(
            f'  {_key:<14} dim={_dim:3d}  '
            f'LP={_lp:5.1f}%  Fisher={_fisher:6.2f}  '
            f'τ_cross={_tau_cr:.3f}  τ_intra={_tau_in:.3f}  ratio={_rt:.3f}'
        )

_swp_KEYS_OK  = [k for k in _swp_ALL_KEYS if k in _swp_results]
_swp_BLOCK_OK = [k for k in _swp_BLOCK_KEYS if k in _swp_results]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers visuels
# ─────────────────────────────────────────────────────────────────────────────
_swp_LABEL_MAP = {
    **{f'block_{i}': f'B{i}' for i in range(16)},
    'stage_1_fpn': 'FPN1', 'stage_2_fpn': 'FPN2',
    'stage_3_fpn': 'FPN3', 'stage_4_fpn': 'FPN4',
}

def _swp_labels(keys):
    return [_swp_LABEL_MAP.get(k, k) for k in keys]

def _swp_best_color(vals, idx, col_best='#E63946', col_norm='#457B9D'):
    return [col_best if v == max(vals) else col_norm for v in vals]


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Linear Probing par block
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 1 — lp_par_block.png')
_swp_lp_vals = [_swp_results[k]['lp']     for k in _swp_KEYS_OK]
_swp_lp_errs = [_swp_results[k]['lp_std'] for k in _swp_KEYS_OK]
_swp_best_lp  = max(_swp_lp_vals)

fig1, ax1 = plt.subplots(figsize=(16, 5))
_swp_x = np.arange(len(_swp_KEYS_OK))
_swp_cols = _swp_best_color(_swp_lp_vals, None)
_bars = ax1.bar(_swp_x, _swp_lp_vals, yerr=_swp_lp_errs,
                color=_swp_cols, capsize=4, alpha=0.9, zorder=3)
ax1.axhline(_swp_BASELINE, color='crimson', ls='--', lw=1.5,
            label=f'Baseline {_swp_BASELINE:.1f}%', zorder=4)
ax1.set_xticks(_swp_x)
ax1.set_xticklabels(_swp_labels(_swp_KEYS_OK), fontsize=9)
ax1.set_ylabel('Balanced Accuracy (%)', fontsize=11)
ax1.set_title(
    'Linear Probing — tous les blocks Ouassim\n'
    '(5-fold stratifié par image, PCA-50d, class_weight=balanced)',
    fontsize=11,
)
ax1.legend(fontsize=9)
ax1.set_ylim(0, min(100, _swp_best_lp + 20))
ax1.grid(axis='y', alpha=0.3, zorder=0)
# séparateur blocs / FPN
if _swp_FPN_KEYS[0] in _swp_results:
    _swp_sep = len(_swp_BLOCK_OK) - 0.5
    ax1.axvline(_swp_sep, color='#888', ls=':', lw=1.2)
    ax1.text(_swp_sep + 0.1, _swp_best_lp * 0.97, 'FPN →', fontsize=8, color='#888')
for _bar, _v, _e in zip(_bars, _swp_lp_vals, _swp_lp_errs):
    ax1.text(_bar.get_x() + _bar.get_width() / 2, _v + _e + 0.5,
             f'{_v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
plt.tight_layout()
fig1.savefig(_swp_OUTPUT_DIR / 'lp_par_block.png', dpi=150, bbox_inches='tight')
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Fisher J par block
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2 — fisher_par_block.png')
_swp_fish_vals = [_swp_results[k]['fisher'] for k in _swp_KEYS_OK]
_swp_best_fish = max(_swp_fish_vals)
_swp_log_scale = (_swp_best_fish / (min(v for v in _swp_fish_vals if v > 0) + 1e-10)) > 20

fig2, ax2 = plt.subplots(figsize=(16, 5))
_swp_fcols = _swp_best_color(_swp_fish_vals, None, '#2D6A4F', '#52B788')
ax2.bar(_swp_x, _swp_fish_vals, color=_swp_fcols, alpha=0.9, zorder=3)
if _swp_log_scale:
    ax2.set_yscale('log')
ax2.set_xticks(_swp_x)
ax2.set_xticklabels(_swp_labels(_swp_KEYS_OK), fontsize=9)
ax2.set_ylabel('Fisher J' + (' (log)' if _swp_log_scale else ''), fontsize=11)
ax2.set_title(
    'Fisher J balancé — tous les blocks Ouassim\n'
    '(PCA-50d, S_W balancé par N_c)',
    fontsize=11,
)
ax2.grid(axis='y', alpha=0.3, zorder=0)
if _swp_FPN_KEYS[0] in _swp_results:
    ax2.axvline(_swp_sep, color='#888', ls=':', lw=1.2)
for _bi, (_v, _k) in enumerate(zip(_swp_fish_vals, _swp_KEYS_OK)):
    ax2.text(_bi, _v * (1.05 if not _swp_log_scale else 1.3),
             f'{_v:.2f}', ha='center', va='bottom', fontsize=6)
plt.tight_layout()
fig2.savefig(_swp_OUTPUT_DIR / 'fisher_par_block.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — τ cross vs τ intra côte à côte + ratio
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 3 — tau_par_block.png')
_swp_tc_vals = [_swp_results[k]['tau_cross'] for k in _swp_KEYS_OK]
_swp_ti_vals = [_swp_results[k]['tau_intra'] for k in _swp_KEYS_OK]
_swp_rt_vals = [_swp_results[k]['tau_ratio'] for k in _swp_KEYS_OK]

fig3, ax3a = plt.subplots(figsize=(17, 5))
ax3b = ax3a.twinx()

_swp_w = 0.35
_swp_xc = _swp_x - _swp_w / 2
_swp_xi = _swp_x + _swp_w / 2

ax3a.bar(_swp_xc, _swp_tc_vals, _swp_w, label='τ cross-image', color='#1B4F72', alpha=0.85, zorder=3)
ax3a.bar(_swp_xi, _swp_ti_vals, _swp_w, label='τ intra-image', color='#A8DADC', alpha=0.85, zorder=3)
ax3b.plot(_swp_x, _swp_rt_vals, 'D--', color='#E63946', ms=5, lw=1.5,
          label='ratio cross/intra', zorder=5)
ax3b.axhline(1.0, color='#E63946', ls=':', lw=0.8, alpha=0.5)

ax3a.set_xticks(_swp_x)
ax3a.set_xticklabels(_swp_labels(_swp_KEYS_OK), fontsize=9)
ax3a.set_ylabel('Similarité cosine', fontsize=11)
ax3b.set_ylabel('Ratio cross/intra', fontsize=11, color='#E63946')
ax3b.tick_params(colors='#E63946', axis='y')
ax3a.set_title(
    'τ cross-image vs τ intra-image — tous les blocks Ouassim\n'
    'τ_cross élevé → généralise aux nouvelles images  |  ratio > 1 → encode la texture, pas le rendu',
    fontsize=10,
)
if _swp_FPN_KEYS[0] in _swp_results:
    ax3a.axvline(_swp_sep, color='#888', ls=':', lw=1.2)

_h3a, _l3a = ax3a.get_legend_handles_labels()
_h3b, _l3b = ax3b.get_legend_handles_labels()
ax3a.legend(_h3a + _h3b, _l3a + _l3b, fontsize=8, loc='upper right')
ax3a.grid(axis='y', alpha=0.3, zorder=0)
plt.tight_layout()
fig3.savefig(_swp_OUTPUT_DIR / 'tau_par_block.png', dpi=150, bbox_inches='tight')
plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Tendances en profondeur (blocks 0→15 uniquement)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 4 — profondeur_trends.png')
_swp_d_keys  = _swp_BLOCK_OK
_swp_d_idx   = list(range(len(_swp_d_keys)))
_swp_d_lp    = [_swp_results[k]['lp']        for k in _swp_d_keys]
_swp_d_fish  = [_swp_results[k]['fisher']    for k in _swp_d_keys]
_swp_d_tc    = [_swp_results[k]['tau_cross'] for k in _swp_d_keys]

# Normaliser Fisher 0-1 pour superposition
_swp_fmin, _swp_fmax = min(_swp_d_fish), max(_swp_d_fish)
_swp_d_fish_n = [(v - _swp_fmin) / (_swp_fmax - _swp_fmin + 1e-10) for v in _swp_d_fish]

fig4, ax4a = plt.subplots(figsize=(14, 5))
ax4b = ax4a.twinx()

ax4a.plot(_swp_d_idx, _swp_d_lp,    'o-', color='#1B4F72', lw=2, ms=6,
          label='Linear Probing (%)')
ax4a.fill_between(_swp_d_idx, _swp_d_lp, alpha=0.12, color='#1B4F72')
ax4a.plot(_swp_d_idx, [v * 100 for v in _swp_d_tc], 's--', color='#E63946', lw=1.8, ms=5,
          label='τ cross-image (×100)')
ax4a.axhline(_swp_BASELINE, color='gray', ls=':', lw=1, alpha=0.6)

ax4b.plot(_swp_d_idx, _swp_d_fish_n, '^:', color='#2D6A4F', lw=1.5, ms=5,
          label='Fisher J (normalisé 0-1)', alpha=0.8)

ax4a.set_xticks(_swp_d_idx)
ax4a.set_xticklabels([f'B{i}' for i in range(len(_swp_d_keys))], fontsize=9)
ax4a.set_xlabel('Profondeur du block (trunk Hiera Small)', fontsize=10)
ax4a.set_ylabel('LP (%) / τ×100', fontsize=10)
ax4b.set_ylabel('Fisher J normalisé', fontsize=10, color='#2D6A4F')
ax4b.tick_params(colors='#2D6A4F', axis='y')

_h4a, _l4a = ax4a.get_legend_handles_labels()
_h4b, _l4b = ax4b.get_legend_handles_labels()
ax4a.legend(_h4a + _h4b, _l4a + _l4b, fontsize=8, loc='best')
ax4a.set_title(
    'Tendances en profondeur — Linear Probing, Fisher J, τ cross-image\n'
    'Base Ouassim (images MEB grayscale brutes)',
    fontsize=11,
)
ax4a.grid(alpha=0.25)

# Annoter le best LP et le best τ_cross
_swp_best_lp_i  = int(np.argmax(_swp_d_lp))
_swp_best_tc_i  = int(np.argmax(_swp_d_tc))
ax4a.annotate(
    f'Best LP\n{_swp_d_lp[_swp_best_lp_i]:.1f}%',
    (_swp_best_lp_i, _swp_d_lp[_swp_best_lp_i]),
    xytext=(4, 8), textcoords='offset points',
    fontsize=8, color='#1B4F72', fontweight='bold',
)
ax4a.annotate(
    f'Best τ\n{_swp_d_tc[_swp_best_tc_i]*100:.1f}',
    (_swp_best_tc_i, _swp_d_tc[_swp_best_tc_i] * 100),
    xytext=(4, 8), textcoords='offset points',
    fontsize=8, color='#E63946', fontweight='bold',
)

plt.tight_layout()
fig4.savefig(_swp_OUTPUT_DIR / 'profondeur_trends.png', dpi=150, bbox_inches='tight')
plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Heatmap synthèse
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 5 — synthese_heatmap.png')
_swp_METRICS = ['lp', 'fisher', 'tau_cross']
_swp_METRIC_LABELS = ['Linear Probing (%)', 'Fisher J', 'τ cross-image']

_swp_mat_raw = np.array([
    [_swp_results[k][m] for m in _swp_METRICS]
    for k in _swp_KEYS_OK
], dtype=float)

# Normaliser chaque colonne 0-1
_swp_mat_norm = _swp_mat_raw.copy()
for _j in range(_swp_mat_norm.shape[1]):
    _col = _swp_mat_norm[:, _j]
    _mn, _mx = _col.min(), _col.max()
    _swp_mat_norm[:, _j] = (_col - _mn) / (_mx - _mn + 1e-10)

_swp_n_rows = len(_swp_KEYS_OK)
fig5, ax5 = plt.subplots(figsize=(7, max(6, _swp_n_rows * 0.38)))
_im = ax5.imshow(_swp_mat_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax5.set_xticks(range(len(_swp_METRICS)))
ax5.set_xticklabels(_swp_METRIC_LABELS, fontsize=10)
ax5.set_yticks(range(_swp_n_rows))
ax5.set_yticklabels(_swp_labels(_swp_KEYS_OK), fontsize=9)
for _i in range(_swp_n_rows):
    for _j in range(len(_swp_METRICS)):
        _raw = _swp_mat_raw[_i, _j]
        _txt = f'{_raw:.1f}%' if _swp_METRICS[_j] == 'lp' else f'{_raw:.3f}'
        _col = 'black' if _swp_mat_norm[_i, _j] > 0.35 else 'white'
        ax5.text(_j, _i, _txt, ha='center', va='center', fontsize=7.5, color=_col)
plt.colorbar(_im, ax=ax5, fraction=0.035, label='Score normalisé 0-1')
ax5.set_title(
    'Synthèse — tous les blocks Ouassim\n'
    '(valeurs normalisées 0-1 par métrique)',
    fontsize=11,
)
# ligne de séparation blocks/FPN
if _swp_FPN_KEYS[0] in _swp_results:
    _swp_sep_row = len(_swp_BLOCK_OK) - 0.5
    ax5.axhline(_swp_sep_row, color='white', lw=2)
    ax5.text(len(_swp_METRICS) - 0.5, _swp_sep_row + 0.3, '← FPN', fontsize=8,
             color='white', ha='right')
plt.tight_layout()
fig5.savefig(_swp_OUTPUT_DIR / 'synthese_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig5)

# ─────────────────────────────────────────────────────────────────────────────
# Tableau texte + CSV
# ─────────────────────────────────────────────────────────────────────────────
print('\nTableau des résultats...')
_swp_sorted = sorted(_swp_KEYS_OK, key=lambda k: -_swp_results[k]['lp'])

_swp_lines = [
    '=' * 100,
    'SWEEP COMPLET — blocks Ouassim (images MEB grayscale brutes)',
    f'Base : {_swp_DB_PATH.name}  |  Patches valides : {_swp_mask.sum()}',
    f'Catégories : {[_swp_CATEGORIES[c] for c in _swp_CATS_VALID]}',
    f'Protocole : PCA-{_swp_PCA_DIM}d, {_swp_N_FOLDS}-fold par image, SEED={_swp_SEED}',
    f'Baseline : {_swp_BASELINE:.1f}%  (1/{_swp_N_CATS} aléatoire)',
    '=' * 100,
    f'\n{"Rang":<5} {"Block":<14} {"Dim":>5} │ '
    f'{"LP (%)":>14} │ {"Fisher J":>10} │ '
    f'{"τ_cross":>9} │ {"τ_intra":>9} │ {"ratio":>7}',
    '─' * 100,
]

for _rank, _k in enumerate(_swp_sorted, 1):
    _r = _swp_results[_k]
    _swp_lines.append(
        f'{_rank:<5} {_k:<14} {_r["dim"]:>5} │ '
        f'{_r["lp"]:>6.1f} ± {_r["lp_std"]:>4.1f}   │ '
        f'{_r["fisher"]:>10.3f} │ '
        f'{_r["tau_cross"]:>9.3f} │ '
        f'{_r["tau_intra"]:>9.3f} │ '
        f'{_r["tau_ratio"]:>7.3f}'
    )

# Verdict
_swp_best_lp_key  = max(_swp_KEYS_OK, key=lambda k: _swp_results[k]['lp'])
_swp_best_tc_key  = max(_swp_KEYS_OK, key=lambda k: _swp_results[k]['tau_cross'])
_swp_best_fi_key  = max(_swp_KEYS_OK, key=lambda k: _swp_results[k]['fisher'])

# Top 3 LP
_swp_top3 = _swp_sorted[:3]

# Sweet spot : block intermédiaire avec LP > baseline*2 ET τ_cross élevé
_swp_sweet = sorted(
    [k for k in _swp_BLOCK_OK if _swp_results[k]['lp'] > _swp_BASELINE * 2],
    key=lambda k: _swp_results[k]['tau_cross'],
    reverse=True,
)
_swp_sweet_key = _swp_sweet[0] if _swp_sweet else _swp_best_tc_key

_swp_lines += [
    '─' * 100,
    '',
    'VERDICT',
    '─' * 60,
    f'Top 3 par Linear Probing :',
]
for _i, _k in enumerate(_swp_top3, 1):
    _r = _swp_results[_k]
    _swp_lines.append(
        f'  {_i}. {_k:<14} LP={_r["lp"]:.1f}%  τ_cross={_r["tau_cross"]:.3f}  Fisher={_r["fisher"]:.3f}'
    )

_swp_lines += [
    '',
    f'Meilleur LP        : {_swp_best_lp_key}  → {_swp_results[_swp_best_lp_key]["lp"]:.1f}%',
    f'Meilleur τ_cross   : {_swp_best_tc_key}  → {_swp_results[_swp_best_tc_key]["tau_cross"]:.3f}  '
    f'(LA PLUS IMPORTANTE : généralise aux nouvelles images)',
    f'Meilleur Fisher J  : {_swp_best_fi_key}  → {_swp_results[_swp_best_fi_key]["fisher"]:.3f}',
    '',
    f'Sweet spot (LP>2×baseline ET τ_cross max) : {_swp_sweet_key}',
    f'  LP={_swp_results[_swp_sweet_key]["lp"]:.1f}%  '
    f'τ_cross={_swp_results[_swp_sweet_key]["tau_cross"]:.3f}  '
    f'Fisher={_swp_results[_swp_sweet_key]["fisher"]:.3f}',
    '',
    'COMPARAISON AVEC PATCHAGGER :',
    '  PatchTagger → block_0 semblait le meilleur (LP=97%, τ=0.984)',
    f'  Ouassim     → block_0 s\'effondre (LP={_swp_results.get("block_0", {}).get("lp", "?"):.1f}%, '
    f'τ={_swp_results.get("block_0", {}).get("tau_cross", "?"):.3f})',
    f'  La séparabilité de block_0 sur PatchTagger était liée au traitement image (RGB, contraste),',
    f'  pas à la texture biologique réelle.',
    f'  Sur les vraies images MEB, recommander : {_swp_sweet_key}',
    '=' * 100,
]

_swp_tableau = '\n'.join(_swp_lines)
print('\n' + _swp_tableau)

with open(_swp_OUTPUT_DIR / 'results_table.txt', 'w') as _f:
    _f.write(_swp_tableau + '\n')

with open(_swp_OUTPUT_DIR / 'results.csv', 'w', newline='') as _f:
    _w = csv.writer(_f)
    _w.writerow(['block', 'dim', 'lp_mean', 'lp_std', 'fisher', 'tau_cross', 'tau_intra', 'tau_ratio'])
    for _k in _swp_sorted:
        _r = _swp_results[_k]
        _w.writerow([
            _k, _r['dim'],
            f'{_r["lp"]:.4f}', f'{_r["lp_std"]:.4f}',
            f'{_r["fisher"]:.6f}',
            f'{_r["tau_cross"]:.6f}', f'{_r["tau_intra"]:.6f}', f'{_r["tau_ratio"]:.6f}',
        ])

# ─────────────────────────────────────────────────────────────────────────────
# Résumé fichiers générés
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nFichiers générés dans {_swp_OUTPUT_DIR} :')
for _fname in [
    'lp_par_block.png', 'fisher_par_block.png', 'tau_par_block.png',
    'profondeur_trends.png', 'synthese_heatmap.png',
    'results_table.txt', 'results.csv',
]:
    _p = _swp_OUTPUT_DIR / _fname
    print(f'  {"✓" if _p.exists() else "✗"}  {_fname}')
