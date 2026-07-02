#!/usr/bin/env python3
"""
compare_texturesam_gradient.py
Compare les 3 checkpoints SAM-2.1 (base → TextureSAM η0.3 → η1.0)
sur la séparabilité des textures MEB Ouassim (20 représentations).

Hypothèse : le fine-tuning texture (domaine non-MEB) lisse les
micro-contours → possible dégradation sur textures granulaires MEB.
"""

import csv, json, subprocess, sys, warnings
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

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
_grad_ROOT       = Path(__file__).resolve().parents[1]
_grad_IMG_DIR    = _grad_ROOT / 'Image_Ouassim'
_grad_CFG_PATH   = _grad_ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
_grad_OUTPUT_DIR = _grad_ROOT / 'output_ouassim' / 'compare_gradient'
_grad_H5_DIR     = _grad_ROOT / 'data' / 'feature_database'
_grad_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_grad_SEED      = 42
_grad_PCA_DIM   = 50
_grad_N_FOLDS   = 5
_grad_CATS_EXCL = {2, 8, 10, 11, 12, 13}
_grad_MIN_N     = 30

# Checkpoints (répertoire ou .pt)
_grad_CKPT = {
    'base':   _grad_ROOT / 'checkpoints' / 'sam2.1_hiera_small',
    'eta0.3': _grad_ROOT / 'checkpoints' / 'sam2.1_hiera_small_0.3',
    'eta1.0': _grad_ROOT / 'checkpoints' / 'sam2.1_hiera_small_1.pt',
}

# Bases HDF5 à construire (eta1.0 = déjà construite)
_grad_H5 = {
    'base':   _grad_H5_DIR / 'database_gradient_base.h5',
    'eta0.3': _grad_H5_DIR / 'database_gradient_eta03.h5',
    'eta1.0': _grad_H5_DIR / 'database_meb_ouassim.h5',   # déjà existante
}

_grad_CK_LABELS = ['base', 'eta0.3', 'eta1.0']
_grad_CK_COLORS = {'base': '#2c3e50', 'eta0.3': '#e67e22', 'eta1.0': '#e74c3c'}
_grad_CK_MARKS  = {'base': 'o', 'eta0.3': 's', 'eta1.0': '^'}

_grad_ALL_KEYS = (
    [f'block_{i}' for i in range(16)]
    + ['stage_1_fpn', 'stage_2_fpn', 'stage_3_fpn', 'stage_4_fpn']
)
_grad_N_KEYS = len(_grad_ALL_KEYS)

np.random.seed(_grad_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Config + métadonnées (depuis eta1.0 qui existe déjà)
# ─────────────────────────────────────────────────────────────────────────────
with open(_grad_CFG_PATH) as _f:
    _grad_cfg = json.load(_f)
_grad_CATEGORIES = {int(k): v['name'] for k, v in _grad_cfg['available_categories'].items()}
_grad_CAT_COLORS = {int(k): v.get('color', '#888888')
                    for k, v in _grad_cfg['available_categories'].items()}

with h5py.File(_grad_H5['eta1.0'], 'r') as _h5ref:
    _grad_ALL_NAMES = _h5ref['metadata/image_names'][:]
    _grad_ALL_CATS  = _h5ref['metadata/category_ids'][:].astype(int)

_grad_CATS_VALID = sorted(
    int(c) for c in np.unique(_grad_ALL_CATS)
    if int(c) not in _grad_CATS_EXCL
    and (_grad_ALL_CATS == int(c)).sum() >= _grad_MIN_N
)
_grad_mask   = np.isin(_grad_ALL_CATS, _grad_CATS_VALID)
_grad_y      = _grad_ALL_CATS[_grad_mask]
_grad_imgs   = _grad_ALL_NAMES[_grad_mask]
_grad_N_CATS = len(_grad_CATS_VALID)
_grad_BASELINE = 100.0 / _grad_N_CATS

_grad_CAT_NAMES = [_grad_CATEGORIES[c] for c in _grad_CATS_VALID]
print(f'Patches valides : {_grad_mask.sum()}  |  {_grad_N_CATS} catégories')
print(f'Catégories : {_grad_CAT_NAMES}')

# Identifier textures à grain (Granuleux, Filaments)
_grad_GRAIN_IDS = [c for c in _grad_CATS_VALID
                   if any(kw in _grad_CATEGORIES[c].lower()
                          for kw in ('granul', 'filament'))]
print(f'Textures à grain : {[_grad_CATEGORIES[c] for c in _grad_GRAIN_IDS]}')

# ─────────────────────────────────────────────────────────────────────────────
# Folds (mêmes pour les 3 checkpoints)
# ─────────────────────────────────────────────────────────────────────────────
_grad_imgs_uniq = np.unique(_grad_imgs)
_grad_cat_dom   = np.array([
    int(_sp_mode(_grad_y[_grad_imgs == _img]).mode)
    for _img in _grad_imgs_uniq
])
_grad_skf   = StratifiedKFold(n_splits=_grad_N_FOLDS, shuffle=True, random_state=_grad_SEED)
_grad_FOLDS = list(_grad_skf.split(_grad_imgs_uniq, _grad_cat_dom))


# ─────────────────────────────────────────────────────────────────────────────
# Métriques (mêmes protocoles que sweep_all_blocks_ouassim.py)
# ─────────────────────────────────────────────────────────────────────────────
def _grad_pca_l2(X, n=_grad_PCA_DIM):
    n = min(n, X.shape[1])
    Xp = PCA(n_components=n, random_state=_grad_SEED).fit_transform(X)
    nm = np.linalg.norm(Xp, axis=1, keepdims=True)
    return Xp / np.where(nm < 1e-8, 1.0, nm)


def _grad_compute_lp(X):
    accs = []
    for tr_i, te_i in _grad_FOLDS:
        tr_imgs = _grad_imgs_uniq[tr_i]
        te_imgs = _grad_imgs_uniq[te_i]
        m_tr = np.isin(_grad_imgs, tr_imgs)
        m_te = np.isin(_grad_imgs, te_imgs)
        if m_te.sum() == 0:
            continue
        n = min(_grad_PCA_DIM, X.shape[1])
        pca = PCA(n_components=n, random_state=_grad_SEED)
        Xtr = pca.fit_transform(X[m_tr])
        Xte = pca.transform(X[m_te])
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                                 random_state=_grad_SEED)
        clf.fit(Xtr, _grad_y[m_tr])
        accs.append(balanced_accuracy_score(_grad_y[m_te], clf.predict(Xte)))
    return float(np.mean(accs)) * 100 if accs else 0.0


def _grad_compute_fisher(X):
    """Fisher J balancé (S_W pondéré 1/N_c) sur PCA-50d."""
    n  = min(_grad_PCA_DIM, X.shape[1])
    X50 = PCA(n_components=n, random_state=_grad_SEED).fit_transform(X)
    mu  = X50.mean(axis=0)
    D   = X50.shape[1]
    SB  = np.zeros((D, D))
    SW  = np.zeros((D, D))
    for c in _grad_CATS_VALID:
        mc  = _grad_y == c
        Nc  = mc.sum()
        muc = X50[mc].mean(axis=0)
        d   = (muc - mu).reshape(-1, 1)
        SB += d @ d.T
        dc  = X50[mc] - muc
        SW += (1.0 / Nc) * (dc.T @ dc)
    return float(np.trace(SB) / (np.trace(SW) + 1e-10))


def _grad_compute_tau(X):
    """τ cross / intra — cosine sim, macro sur catégories."""
    Xn = _grad_pca_l2(X)
    cross, intra = [], []
    for c in _grad_CATS_VALID:
        mc  = _grad_y == c
        Xc  = Xn[mc]
        ic  = _grad_imgs[mc]
        N   = Xc.shape[0]
        sim = Xc @ Xc.T
        up  = np.triu(np.ones((N, N), bool), k=1)
        mcr = (ic[:, None] != ic[None, :]) & up
        min_ = (ic[:, None] == ic[None, :]) & up
        if mcr.any():
            cross.append(float(sim[mcr].mean()))
        if min_.any():
            intra.append(float(sim[min_].mean()))
    tc = float(np.mean(cross)) if cross else np.nan
    ti = float(np.mean(intra)) if intra else np.nan
    return tc, ti, tc / (ti + 1e-8) if intra else np.nan


def _grad_fisher_ovr(X, cat_id):
    """Fisher one-vs-rest pour une texture, sur PCA-50d."""
    n  = min(_grad_PCA_DIM, X.shape[1])
    X50 = PCA(n_components=n, random_state=_grad_SEED).fit_transform(X)
    mc  = _grad_y == cat_id
    mo  = ~mc
    Nc, No = mc.sum(), mo.sum()
    if Nc < 2 or No < 2:
        return np.nan
    muc = X50[mc].mean(axis=0)
    muo = X50[mo].mean(axis=0)
    mu  = X50.mean(axis=0)
    D   = X50.shape[1]
    dc  = (muc - mu).reshape(-1, 1)
    do  = (muo - mu).reshape(-1, 1)
    SB  = dc @ dc.T + do @ do.T
    dc_ = X50[mc] - muc;  do_ = X50[mo] - muo
    SW  = (1.0 / Nc) * (dc_.T @ dc_) + (1.0 / No) * (do_.T @ do_)
    return float(np.trace(SB) / (np.trace(SW) + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Construire les bases HDF5 manquantes
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Vérification / construction des bases HDF5 ──')
_grad_ck_available = {}

for _ck in _grad_CK_LABELS:
    _h5p  = _grad_H5[_ck]
    _ckpt = _grad_CKPT[_ck]

    if not _ckpt.exists():
        print(f'  {_ck}: checkpoint introuvable ({_ckpt}) — SKIP')
        continue

    if _h5p.exists():
        print(f'  {_ck}: HDF5 déjà présent ({_h5p.name}) — réutilisé')
        _grad_ck_available[_ck] = _h5p
        continue

    print(f'  {_ck}: construction de {_h5p.name}...')
    _cmd = [
        sys.executable, str(_grad_ROOT / 'build_feature_database.py'),
        '--img-dir',    str(_grad_IMG_DIR),
        '--checkpoint', str(_ckpt),
        '--output',     str(_h5p),
    ]
    _ret = subprocess.run(_cmd, capture_output=True, text=True, cwd=str(_grad_ROOT))
    if _ret.returncode != 0:
        print(f'  ERREUR construction {_ck}:\n{_ret.stderr[-500:]}')
    else:
        print(f'  {_ck}: HDF5 construit ✓')
        _grad_ck_available[_ck] = _h5p

print(f'\nCheckpoints disponibles : {list(_grad_ck_available.keys())}')
if not _grad_ck_available:
    raise RuntimeError('Aucun checkpoint disponible.')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Sweep : LP / Fisher / τ pour chaque (checkpoint, block)
# ─────────────────────────────────────────────────────────────────────────────
print('\n── Sweep métriques (3 checkpoints × 20 représentations) ──')

# {ck: {key: {lp, fisher, tau_cross, tau_intra, grain_fisher: {cat_id: v}}}}
_grad_results = {ck: {} for ck in _grad_ck_available}

for _ck in tqdm(_grad_ck_available, desc='Checkpoints'):
    with h5py.File(_grad_ck_available[_ck], 'r') as _h5:
        for _key in tqdm(_grad_ALL_KEYS, desc=f'  {_ck}', leave=False):
            if _key not in _h5.get('features', {}):
                tqdm.write(f'    {_ck}/{_key} absent')
                continue

            X = _h5['features'][_key][:].astype(np.float32)[_grad_mask]

            _lp                  = _grad_compute_lp(X)
            _fisher              = _grad_compute_fisher(X)
            _tc, _ti, _rt        = _grad_compute_tau(X)
            _grain = {c: _grad_fisher_ovr(X, c) for c in _grad_GRAIN_IDS}

            _grad_results[_ck][_key] = {
                'lp':          _lp,
                'fisher':      _fisher,
                'tau_cross':   _tc,
                'tau_intra':   _ti,
                'tau_ratio':   _rt,
                'grain_fisher': _grain,
            }

# ─────────────────────────────────────────────────────────────────────────────
# CSV complet
# ─────────────────────────────────────────────────────────────────────────────
_grad_CSV = _grad_OUTPUT_DIR / 'gradient_sweep_results.csv'
with open(_grad_CSV, 'w', newline='') as _cf:
    _w = csv.writer(_cf)
    _grain_cols = [f'fisher_ovr_{_grad_CATEGORIES[c][:12].replace(" ","_")}'
                   for c in _grad_GRAIN_IDS]
    _w.writerow(['checkpoint', 'block', 'lp', 'fisher', 'tau_cross',
                 'tau_intra', 'tau_ratio'] + _grain_cols)
    for _ck in _grad_CK_LABELS:
        for _key in _grad_ALL_KEYS:
            if _key not in _grad_results.get(_ck, {}):
                continue
            _r = _grad_results[_ck][_key]
            _gv = [_r['grain_fisher'].get(c, np.nan) for c in _grad_GRAIN_IDS]
            _w.writerow([_ck, _key,
                         f'{_r["lp"]:.3f}', f'{_r["fisher"]:.4f}',
                         f'{_r["tau_cross"]:.4f}', f'{_r["tau_intra"]:.4f}',
                         f'{_r["tau_ratio"]:.4f}']
                        + [f'{v:.4f}' if not np.isnan(v) else '' for v in _gv])

print(f'CSV : {_grad_CSV.name}')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de préparation des séries pour plots
# ─────────────────────────────────────────────────────────────────────────────
def _grad_series(metric, ck):
    """Retourne array (N_KEYS,) de la métrique pour le checkpoint ck."""
    return np.array([
        _grad_results[ck].get(k, {}).get(metric, np.nan)
        for k in _grad_ALL_KEYS
    ])


_grad_x      = np.arange(_grad_N_KEYS)
_grad_xlab   = [k.replace('block_', 'B').replace('stage_', 'S').replace('_fpn', '')
                for k in _grad_ALL_KEYS]
_grad_ck_avail = [ck for ck in _grad_CK_LABELS if ck in _grad_ck_available]

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — LP tous blocks, 3 checkpoints
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 1 — lp_tous_blocks_3ckpt.png')
fig1, ax1 = plt.subplots(figsize=(14, 5))
for _ck in _grad_ck_avail:
    _y = _grad_series('lp', _ck)
    ax1.plot(_grad_x, _y, marker=_grad_CK_MARKS[_ck],
             color=_grad_CK_COLORS[_ck], linewidth=2,
             markersize=5, label=_ck, alpha=0.9)
ax1.axhline(_grad_BASELINE, color='grey', linestyle=':', linewidth=1,
            label=f'Baseline ({_grad_BASELINE:.1f}%)')
ax1.axvline(15.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
ax1.text(15.7, ax1.get_ylim()[0] + 1, 'FPN →', fontsize=8, color='grey')
ax1.set_xticks(_grad_x);  ax1.set_xticklabels(_grad_xlab, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Balanced accuracy (%)', fontsize=11)
ax1.set_title('Linear Probing — 20 représentations × 3 checkpoints\n(5-fold par image, images Ouassim)',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left');  ax1.grid(axis='y', alpha=0.3)
ax1.set_xlim(-0.5, _grad_N_KEYS - 0.5)
fig1.tight_layout()
fig1.savefig(_grad_OUTPUT_DIR / 'lp_tous_blocks_3ckpt.png', dpi=150)
plt.close(fig1);  print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Fisher tous blocks, 3 checkpoints
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2 — fisher_tous_blocks_3ckpt.png')
fig2, ax2 = plt.subplots(figsize=(14, 5))
for _ck in _grad_ck_avail:
    _y = _grad_series('fisher', _ck)
    ax2.plot(_grad_x, _y, marker=_grad_CK_MARKS[_ck],
             color=_grad_CK_COLORS[_ck], linewidth=2,
             markersize=5, label=_ck, alpha=0.9)
ax2.axvline(15.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
ax2.set_xticks(_grad_x);  ax2.set_xticklabels(_grad_xlab, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Fisher J balancé', fontsize=11)
ax2.set_title('Fisher J — 20 représentations × 3 checkpoints\n(inter/intra-classe balancé, PCA-50d)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left');  ax2.grid(axis='y', alpha=0.3)
ax2.set_xlim(-0.5, _grad_N_KEYS - 0.5)
fig2.tight_layout()
fig2.savefig(_grad_OUTPUT_DIR / 'fisher_tous_blocks_3ckpt.png', dpi=150)
plt.close(fig2);  print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — τ_cross tous blocks, 3 checkpoints
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 3 — tau_tous_blocks_3ckpt.png')
fig3, ax3 = plt.subplots(figsize=(14, 5))
for _ck in _grad_ck_avail:
    _y = _grad_series('tau_cross', _ck)
    ax3.plot(_grad_x, _y, marker=_grad_CK_MARKS[_ck],
             color=_grad_CK_COLORS[_ck], linewidth=2,
             markersize=5, label=_ck, alpha=0.9)
ax3.axvline(15.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
ax3.set_xticks(_grad_x);  ax3.set_xticklabels(_grad_xlab, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('τ cross-image (cosine, macro)', fontsize=11)
ax3.set_title('τ cross-image — généralisation inter-images × 3 checkpoints',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=9, loc='upper left');  ax3.grid(axis='y', alpha=0.3)
ax3.set_xlim(-0.5, _grad_N_KEYS - 0.5)
fig3.tight_layout()
fig3.savefig(_grad_OUTPUT_DIR / 'tau_tous_blocks_3ckpt.png', dpi=150)
plt.close(fig3);  print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Gradient par block clé (barplot LP)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 4 — gradient_par_block.png')

# Trouver blocks clés automatiquement : meilleur LP pour eta1.0 + block_4, stage_2_fpn
_grad_best_key = max(_grad_results.get('eta1.0', {}),
                     key=lambda k: _grad_results['eta1.0'].get(k, {}).get('lp', 0),
                     default='block_4')
_grad_KEY_BLOCKS = list(dict.fromkeys(['block_4', 'block_10', _grad_best_key,
                                        'stage_2_fpn', 'stage_3_fpn']))[:5]

_grad_n_kb = len(_grad_KEY_BLOCKS)
_grad_bar_w = 0.25
_grad_kb_x  = np.arange(_grad_n_kb)
_grad_offsets = np.linspace(
    -(_grad_bar_w * (len(_grad_ck_avail) - 1)) / 2,
     (_grad_bar_w * (len(_grad_ck_avail) - 1)) / 2,
    len(_grad_ck_avail)
)

fig4, ax4 = plt.subplots(figsize=(10, 5))
for _ki, _ck in enumerate(_grad_ck_avail):
    _vals = [_grad_results[_ck].get(k, {}).get('lp', 0.0) for k in _grad_KEY_BLOCKS]
    ax4.bar(_grad_kb_x + _grad_offsets[_ki], _vals, width=_grad_bar_w,
            color=_grad_CK_COLORS[_ck], label=_ck, alpha=0.88, edgecolor='white')
ax4.axhline(_grad_BASELINE, color='grey', linestyle=':', linewidth=1)
ax4.set_xticks(_grad_kb_x)
ax4.set_xticklabels(_grad_KEY_BLOCKS, fontsize=10)
ax4.set_ylabel('Balanced accuracy (%)', fontsize=11)
ax4.set_title('Effet du fine-tuning sur blocks clés — LP\n(base → η0.3 → η1.0)',
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=9);  ax4.grid(axis='y', alpha=0.3)
fig4.tight_layout()
fig4.savefig(_grad_OUTPUT_DIR / 'gradient_par_block.png', dpi=150)
plt.close(fig4);  print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Grain focus : Fisher one-vs-rest pour Granuleux + Filaments
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 5 — grain_focus_tous_blocks.png')

if _grad_GRAIN_IDS:
    fig5, axes5 = plt.subplots(1, len(_grad_GRAIN_IDS),
                                figsize=(7 * len(_grad_GRAIN_IDS), 5), squeeze=False)
    for _gi, _gcat in enumerate(_grad_GRAIN_IDS):
        ax5 = axes5[0, _gi]
        for _ck in _grad_ck_avail:
            _y = np.array([
                _grad_results[_ck].get(k, {}).get('grain_fisher', {}).get(_gcat, np.nan)
                for k in _grad_ALL_KEYS
            ])
            ax5.plot(_grad_x, _y, marker=_grad_CK_MARKS[_ck],
                     color=_grad_CK_COLORS[_ck], linewidth=2,
                     markersize=5, label=_ck, alpha=0.9)
        ax5.axvline(15.5, color='black', linestyle='--', linewidth=1, alpha=0.4)
        ax5.set_xticks(_grad_x)
        ax5.set_xticklabels(_grad_xlab, rotation=45, ha='right', fontsize=7)
        ax5.set_ylabel('Fisher one-vs-rest', fontsize=10)
        ax5.set_title(f'{_grad_CATEGORIES[_gcat]}\n(one-vs-rest Fisher, PCA-50d)',
                      fontsize=10, fontweight='bold')
        ax5.legend(fontsize=8);  ax5.grid(axis='y', alpha=0.3)
        ax5.set_xlim(-0.5, _grad_N_KEYS - 0.5)
    fig5.suptitle('Hypothèse contours — Granuleux / Filaments : base > η1.0 ?',
                  fontsize=12, fontweight='bold')
    fig5.tight_layout()
    fig5.savefig(_grad_OUTPUT_DIR / 'grain_focus_tous_blocks.png', dpi=150)
    plt.close(fig5)
    print('  Saved.')
else:
    print('  Aucune texture à grain trouvée — skipped.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Heatmap checkpoint × texture (Fisher one-vs-rest au meilleur block)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 6 — heatmap_ckpt_texture.png')

# Pour chaque texture, trouver le best block (max Fisher global sur eta1.0 ou premier ck dispo)
_grad_ref_ck = _grad_ck_avail[0]
_grad_best_block_per_cat = {}
for _c in _grad_CATS_VALID:
    _bests = {}
    for _ck in _grad_ck_avail:
        for _key in _grad_ALL_KEYS:
            _f = _grad_results[_ck].get(_key, {}).get('grain_fisher', {}).get(_c, np.nan)
            if not np.isnan(_f):
                _bests[(_ck, _key)] = _f
    if _bests:
        _best_pair = max(_bests, key=_bests.get)
        _grad_best_block_per_cat[_c] = _best_pair[1]
    else:
        _grad_best_block_per_cat[_c] = 'block_4'

# Construire la matrice heatmap : checkpoints × catégories
_grad_hm = np.full((len(_grad_ck_avail), _grad_N_CATS), np.nan)
for _ki, _ck in enumerate(_grad_ck_avail):
    for _ci, _c in enumerate(_grad_CATS_VALID):
        _best_blk = _grad_best_block_per_cat[_c]
        # Utiliser Fisher global (pas grain-only) pour avoir toutes les catégories
        _grad_hm[_ki, _ci] = _grad_results[_ck].get(_best_blk, {}).get('fisher', np.nan)

fig6, ax6 = plt.subplots(figsize=(max(8, _grad_N_CATS * 1.2 + 2), 4))
_im6 = ax6.imshow(_grad_hm, aspect='auto', cmap='YlOrRd')
plt.colorbar(_im6, ax=ax6, label='Fisher J (au meilleur block par texture)')

for _ki in range(len(_grad_ck_avail)):
    for _ci in range(_grad_N_CATS):
        _v = _grad_hm[_ki, _ci]
        if not np.isnan(_v):
            _col = 'white' if _v > np.nanmax(_grad_hm) * 0.7 else 'black'
            ax6.text(_ci, _ki, f'{_v:.1f}', ha='center', va='center',
                     fontsize=8, color=_col)
            ax6.text(_ci, _ki + 0.35, f'({_grad_best_block_per_cat.get(_grad_CATS_VALID[_ci], "?").replace("block_", "B")})',
                     ha='center', va='center', fontsize=6, color='grey')

ax6.set_xticks(range(_grad_N_CATS))
ax6.set_xticklabels(_grad_CAT_NAMES, rotation=30, ha='right', fontsize=9)
ax6.set_yticks(range(len(_grad_ck_avail)))
ax6.set_yticklabels(_grad_ck_avail, fontsize=10)
ax6.set_title('Heatmap checkpoint × texture — Fisher J\n(valeur au meilleur block par texture)',
              fontsize=11, fontweight='bold')
fig6.tight_layout()
fig6.savefig(_grad_OUTPUT_DIR / 'heatmap_ckpt_texture.png', dpi=150)
plt.close(fig6);  print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Résumé quantitatif
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('CLASSEMENT — MEILLEUR CHECKPOINT PAR MÉTRIQUE (tous blocks, max)')
print('=' * 72)

for _metric in ['lp', 'fisher', 'tau_cross']:
    _ck_best_val = {}
    for _ck in _grad_ck_avail:
        _vals = [_grad_results[_ck].get(k, {}).get(_metric, np.nan)
                 for k in _grad_ALL_KEYS]
        _valid = [v for v in _vals if not np.isnan(v)]
        _ck_best_val[_ck] = (max(_valid), _grad_ALL_KEYS[np.nanargmax(_vals)])
    _winner = max(_ck_best_val, key=lambda ck: _ck_best_val[ck][0])
    print(f'\n{_metric.upper()}:')
    for _ck in _grad_ck_avail:
        _v, _k = _ck_best_val[_ck]
        _mark  = ' ← MEILLEUR' if _ck == _winner else ''
        print(f'  {_ck:<8}: {_v:.2f}  (@{_k}){_mark}')

print('\n── Textures à grain (Granuleux, Filaments) ──')
for _gcat in _grad_GRAIN_IDS:
    print(f'\n  {_grad_CATEGORIES[_gcat]}:')
    for _ck in _grad_ck_avail:
        _vals = [_grad_results[_ck].get(k, {}).get('grain_fisher', {}).get(_gcat, np.nan)
                 for k in _grad_ALL_KEYS]
        _valid = [(v, _grad_ALL_KEYS[i]) for i, v in enumerate(_vals) if not np.isnan(v)]
        if _valid:
            _best_v, _best_k = max(_valid)
            print(f'    {_ck:<8}: max Fisher OvR = {_best_v:.3f}  (@{_best_k})')

# Verdict hypothèse contours
print('\n── VERDICT HYPOTHÈSE CONTOURS ──')
for _gcat in _grad_GRAIN_IDS:
    _rank = {}
    for _ck in _grad_ck_avail:
        _vals = [_grad_results[_ck].get(k, {}).get('grain_fisher', {}).get(_gcat, np.nan)
                 for k in _grad_ALL_KEYS]
        _valid = [v for v in _vals if not np.isnan(v)]
        _rank[_ck] = max(_valid) if _valid else 0.0
    _sorted_ck = sorted(_grad_ck_avail, key=lambda c: _rank[c], reverse=True)
    _expected  = ['base', 'eta0.3', 'eta1.0']
    _holds     = (_sorted_ck[:len(_grad_ck_avail)] == _expected[:len(_grad_ck_avail)])
    print(f'  {_grad_CATEGORIES[_gcat]:20} : ordre = {_sorted_ck}')
    print(f'    → Hypothèse {"CONFIRMÉE ✓" if _holds else "INFIRMÉE ✗"}')
    print(f'       (attendu base > 0.3 > 1.0, observé {_sorted_ck[0]} > ...)')

print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────────
# Génération du fichier Markdown
# ─────────────────────────────────────────────────────────────────────────────
print('\nGénération compare_texturesam_gradient.md...')

# Calcul rapide des meilleurs config par checkpoint (LP)
_grad_ck_best_lp = {}
for _ck in _grad_ck_avail:
    _vals = {k: _grad_results[_ck].get(k, {}).get('lp', 0) for k in _grad_ALL_KEYS}
    _bk   = max(_vals, key=_vals.get)
    _grad_ck_best_lp[_ck] = (_vals[_bk], _bk)

# Vérifier si eta1.0 domine
_lp_winner = max(_grad_ck_avail,
                 key=lambda ck: _grad_ck_best_lp[ck][0])

_md = [
    '# Comparaison SAM-2.1 base → TextureSAM η0.3 → η1.0',
    '## sur la séparabilité des textures MEB Ouassim',
    '',
    '## Objectif',
    '',
    'Mesurer l\'effet du fine-tuning texture (base → η0.3 → η1.0) sur la séparabilité',
    'des textures MEB sur TOUS les 20 blocs (block_0..15 + 4 FPN).',
    'Métriques : Linear Probing (balanced accuracy), Fisher J balancé, τ cross-image.',
    '',
    '## Les 3 checkpoints',
    '',
    '| Nom | Fichier | Fine-tuning |',
    '|-----|---------|-------------|',
    '| base | sam2.1_hiera_small | SAM-2.1 original, SANS fine-tuning texture |',
    '| η0.3 | sam2.1_hiera_small_0.3 | TextureSAM 19 epochs, augmentation modérée (clipLimit ≤ 0.3) |',
    '| η1.0 | sam2.1_hiera_small_1.pt | TextureSAM 25 epochs, augmentation forte (clipLimit ≤ 1.0) |',
    '',
    'Même architecture Hiera Small pour les 3 (embed_dim=96, stages(1,2,11,2),',
    'global_att_blocks(7,10,13)). Seuls les poids diffèrent.',
    '',
    '## Hypothèse',
    '',
    'Les textures granulaires MEB (Granuleux, Filaments) correspondent à des **micro-contours**.',
    'SAM base préserve ces contours (pas de fine-tuning).',
    'Le fine-tuning texture (domaine non-MEB) pourrait les lisser, dégradant leur séparabilité.',
    'Prédiction : **base > η0.3 > η1.0** sur ces textures.',
    '',
    '## Résultats',
    '',
    '### Meilleur checkpoint par métrique (max sur 20 blocs)',
    '',
    '| Métrique | base | η0.3 | η1.0 |',
    '|----------|------|------|------|',
]

for _metric, _mlabel in [('lp', 'LP (%)'), ('fisher', 'Fisher J'), ('tau_cross', 'τ cross')]:
    _row_vals = []
    for _ck in ['base', 'eta0.3', 'eta1.0']:
        if _ck not in _grad_ck_avail:
            _row_vals.append('N/A')
            continue
        _vals = [_grad_results[_ck].get(k, {}).get(_metric, np.nan) for k in _grad_ALL_KEYS]
        _best = max(v for v in _vals if not np.isnan(v)) if any(not np.isnan(v) for v in _vals) else 0.0
        _best_k = _grad_ALL_KEYS[int(np.nanargmax(_vals))]
        _row_vals.append(f'{_best:.2f} (@{_best_k})')
    _md.append(f'| {_mlabel} | {" | ".join(_row_vals)} |')

_md += [
    '',
    '### Textures à grain — Fisher one-vs-rest',
    '',
    '| Texture | base (max) | η0.3 (max) | η1.0 (max) | Hypothèse |',
    '|---------|-----------|-----------|-----------|-----------|',
]

for _gcat in _grad_GRAIN_IDS:
    _row_f = []
    for _ck in ['base', 'eta0.3', 'eta1.0']:
        if _ck not in _grad_ck_avail:
            _row_f.append('N/A')
            continue
        _vals = [_grad_results[_ck].get(k, {}).get('grain_fisher', {}).get(_gcat, np.nan)
                 for k in _grad_ALL_KEYS]
        _valid = [v for v in _vals if not np.isnan(v)]
        _row_f.append(f'{max(_valid):.3f}' if _valid else 'N/A')
    _ranks = {_ck: float(_row_f[i].split(' ')[0]) if _row_f[i] != 'N/A' else 0.0
              for i, _ck in enumerate(['base', 'eta0.3', 'eta1.0'])}
    _sorted_r = sorted([ck for ck in ['base', 'eta0.3', 'eta1.0'] if ck in _grad_ck_avail],
                       key=lambda c: _ranks[c], reverse=True)
    _holds  = (_sorted_r[:3] == ['base', 'eta0.3', 'eta1.0'][:len(_sorted_r)])
    _hyp    = '✓ confirmée' if _holds else '✗ infirmée'
    _md.append(f'| {_grad_CATEGORIES[_gcat]} | {" | ".join(_row_f)} | {_hyp} |')

_md += [
    '',
    '## Conclusion',
    '',
    f'**Meilleur checkpoint global (LP)** : `{_lp_winner}` '
    f'({_grad_ck_best_lp[_lp_winner][0]:.1f}% @ {_grad_ck_best_lp[_lp_winner][1]})',
    '',
]

if _lp_winner == 'base':
    _md.append(
        'Le **SAM-2.1 base** (sans fine-tuning) surpasse les checkpoints fine-tunés sur MEB Ouassim.'
    )
    _md.append(
        'Cela suggère que le fine-tuning texture (domaine non-MEB) nuit à la séparabilité des textures MEB.'
    )
    _md.append(
        'Piste : fine-tuning directement sur images MEB, ou augmentation de contraste CLAHE avant fine-tuning.'
    )
elif _lp_winner == 'eta0.3':
    _md.append(
        'Le checkpoint **η0.3** (fine-tuning modéré) est le meilleur compromis sur MEB Ouassim.'
    )
    _md.append(
        'Le fine-tuning fort (η1.0) lisse trop les micro-contours, confirmant partiellement l\'hypothèse.'
    )
else:
    _md.append(
        'Le checkpoint **η1.0** (fine-tuning fort) reste le meilleur sur MEB Ouassim.'
    )
    _md.append(
        'L\'hypothèse de lissage des micro-contours n\'est pas confirmée ici.'
    )

_md += [
    '',
    '- **Blocks les plus affectés** par le fine-tuning : typiquement les blocks précoces',
    '  (block_0..4) où les contours locaux sont encodés.',
    '- **FPN** : les couches FPN intègrent les caractéristiques multi-échelles ;',
    '  leur réponse au fine-tuning peut différer des blocks trunk.',
    '',
    '## Fichiers générés',
    '',
    '- `lp_tous_blocks_3ckpt.png` — LP par block, 3 courbes',
    '- `fisher_tous_blocks_3ckpt.png` — Fisher par block',
    '- `tau_tous_blocks_3ckpt.png` — τ cross par block',
    '- `gradient_par_block.png` — barplot LP blocks clés',
    '- `grain_focus_tous_blocks.png` — Fisher OvR Granuleux/Filaments',
    '- `heatmap_ckpt_texture.png` — heatmap checkpoint × texture',
    '- `gradient_sweep_results.csv` — tableau complet',
]

with open(_grad_OUTPUT_DIR / 'compare_texturesam_gradient.md', 'w') as _fmd:
    _fmd.write('\n'.join(_md) + '\n')
print('  Saved.')

print(f'\nFichiers générés dans {_grad_OUTPUT_DIR}:')
for _fn in ['lp_tous_blocks_3ckpt.png', 'fisher_tous_blocks_3ckpt.png',
            'tau_tous_blocks_3ckpt.png', 'gradient_par_block.png',
            'grain_focus_tous_blocks.png', 'heatmap_ckpt_texture.png',
            'compare_texturesam_gradient.md', 'gradient_sweep_results.csv']:
    _p = _grad_OUTPUT_DIR / _fn
    print(f'  {"✓" if _p.exists() else "✗"}  {_fn}')
