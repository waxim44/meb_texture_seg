#!/usr/bin/env python3
"""
source_robustness_meb.py
Vérifie si les conclusions (block_0 = meilleur encodeur) sont robustes
au changement de source d'images (PatchTagger vs Ouassim).

Étape 1 : construire database_meb_ouassim.h5 (si absent)
Étape 2 : LP + Fisher J + τ cross/intra sur 4 blocs × 2 sources
Étape 3 : tableau comparatif + verdict

Protocole : PCA-min(50,dim)d, L2-norm, 5-fold stratifié par image,
            class_weight balanced, SEED=42, catégories [1,3,4,5,6,7,9].
"""

import subprocess, sys, json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mode as _sp_mode
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Chemins & constantes
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DB_PT       = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
DB_OUA      = ROOT / 'data' / 'feature_database' / 'database_meb_ouassim.h5'
IMG_OUA     = ROOT / 'Image_Ouassim'
CHECKPOINT  = ROOT / 'checkpoints' / 'sam2.1_hiera_small_1.pt'
ANNOT       = ROOT / 'PatchTagger_Output' / 'categories.xlsx'
CFG_PATH    = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
OUTPUT_DIR  = ROOT / 'outputs' / 'source_robustness'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
PCA_DIM     = 50
N_FOLDS     = 5
CATS_EXCL   = {2, 8, 10, 11, 12, 13}
MIN_PATCHES = 30

# Blocs à évaluer
BLOCKS_EVAL = ['block_0', 'block_3', 'stage_3_fpn', 'block_13']

with open(CFG_PATH) as _f:
    _cfg = json.load(_f)
CATEGORIES = {int(k): v['name'] for k, v in _cfg['available_categories'].items()}

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Construire la base Ouassim si nécessaire
# ─────────────────────────────────────────────────────────────────────────────
if DB_OUA.exists():
    print(f'Base Ouassim déjà présente : {DB_OUA.name}')
else:
    print('Construction database_meb_ouassim.h5...')
    _cmd = [
        sys.executable,
        str(ROOT / 'build_feature_database.py'),
        '--img-dir',    str(IMG_OUA),
        '--output',     str(DB_OUA),
        '--annot',      str(ANNOT),
        '--checkpoint', str(CHECKPOINT),
    ]
    print(f'  cmd : {" ".join(str(c) for c in _cmd)}')
    _ret = subprocess.run(_cmd, cwd=str(ROOT))
    if _ret.returncode != 0:
        print('ERREUR : build_feature_database.py a échoué.')
        sys.exit(1)
    print(f'  → {DB_OUA}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Charger les métadonnées communes
# ─────────────────────────────────────────────────────────────────────────────
print('\nChargement des métadonnées...')
with h5py.File(DB_PT, 'r') as _h5:
    _ALL_NAMES = _h5['metadata/image_names'][:]
    _ALL_CATS  = _h5['metadata/category_ids'][:].astype(int)

_CATS_VALID = sorted(
    int(c) for c in np.unique(_ALL_CATS)
    if int(c) not in CATS_EXCL
    and (_ALL_CATS == int(c)).sum() >= MIN_PATCHES
)
_mask_valid = np.isin(_ALL_CATS, _CATS_VALID)
_y          = _ALL_CATS[_mask_valid]
_imgs       = _ALL_NAMES[_mask_valid]

print(f'  Catégories valides : {[CATEGORIES[c] for c in _CATS_VALID]}')
print(f'  Patches valides    : {_mask_valid.sum()}')

# ─────────────────────────────────────────────────────────────────────────────
# Construction des folds (stratifiés par image — catégorie dominante)
# ─────────────────────────────────────────────────────────────────────────────
_imgs_uniq  = np.unique(_imgs)
_cat_dom    = np.array([
    int(_sp_mode(_y[_imgs == _img]).mode)
    for _img in _imgs_uniq
])
_skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(_imgs_uniq, _cat_dom))


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions métriques (protocole identique à compare_descriptors_meb.py
# et analyze_vlad_meb.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _pca_l2(X, n_comp=PCA_DIM, seed=SEED):
    """PCA-min(n_comp, dim)d → L2-normalise."""
    n = min(n_comp, X.shape[1])
    Xp = PCA(n_components=n, random_state=seed).fit_transform(X)
    norms = np.linalg.norm(Xp, axis=1, keepdims=True)
    return Xp / np.where(norms < 1e-8, 1.0, norms)


def compute_lp(X):
    """Linear Probing : balanced accuracy, 5-fold by image, PCA-50d, L2-norm, StandardScaler."""
    accs = []
    for _tr_i, _te_i in _FOLDS:
        _tr_imgs = _imgs_uniq[_tr_i]
        _te_imgs = _imgs_uniq[_te_i]
        _m_tr = np.isin(_imgs, _tr_imgs)
        _m_te = np.isin(_imgs, _te_imgs)

        n = min(PCA_DIM, X.shape[1])
        _pca = PCA(n_components=n, random_state=SEED)
        _X_tr = _pca.fit_transform(X[_m_tr])
        _X_te = _pca.transform(X[_m_te])

        _sc = StandardScaler()
        _X_tr = _sc.fit_transform(_X_tr)
        _X_te = _sc.transform(_X_te)

        _clf = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=SEED, n_jobs=-1,
        )
        _clf.fit(_X_tr, _y[_m_tr])
        accs.append(balanced_accuracy_score(_y[_m_te], _clf.predict(_X_te)))
    return float(np.mean(accs)), float(np.std(accs))


def compute_fisher(X):
    """Fisher J balancé sur PCA-50d."""
    n = min(PCA_DIM, X.shape[1])
    X50  = PCA(n_components=n, random_state=SEED).fit_transform(X)
    mu   = X50.mean(axis=0)
    D    = X50.shape[1]
    S_B  = np.zeros((D, D))
    S_W  = np.zeros((D, D))
    for c in _CATS_VALID:
        mask  = _y == c
        N_c   = mask.sum()
        mu_c  = X50[mask].mean(axis=0)
        diff  = (mu_c - mu).reshape(-1, 1)
        S_B  += diff @ diff.T
        dc    = X50[mask] - mu_c
        S_W  += (1.0 / N_c) * (dc.T @ dc)
    return float(np.trace(S_B) / (np.trace(S_W) + 1e-10))


def compute_tau(X):
    """τ cross/intra : cosine sim cross-image vs same-image, macro sur catégories.
    Calculé sur PCA-50d L2-normé (idem analyze_vlad_meb.ipynb)."""
    Xn = _pca_l2(X)
    cross_vals, intra_vals = [], []
    for c in _CATS_VALID:
        mask_c = _y == c
        Xc     = Xn[mask_c]
        imgs_c = _imgs[mask_c]
        N_c    = Xc.shape[0]

        sim   = Xc @ Xc.T
        upper = np.triu(np.ones((N_c, N_c), dtype=bool), k=1)
        m_cross = (imgs_c[:, None] != imgs_c[None, :]) & upper
        m_intra = (imgs_c[:, None] == imgs_c[None, :]) & upper

        if m_cross.any():
            cross_vals.append(float(sim[m_cross].mean()))
        if m_intra.any():
            intra_vals.append(float(sim[m_intra].mean()))

    if not cross_vals or not intra_vals:
        return np.nan
    return float(np.mean(cross_vals)) / (float(np.mean(intra_vals)) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Calcul des métriques par source et par bloc
# ─────────────────────────────────────────────────────────────────────────────
SOURCES = {
    'PatchTagger': DB_PT,
    'Ouassim':     DB_OUA,
}

results = {}   # results[source][block] = {'lp':..., 'lp_std':..., 'fisher':..., 'tau':...}

for src_name, db_path in SOURCES.items():
    results[src_name] = {}
    print(f'\n=== Source : {src_name} ({db_path.name}) ===')

    with h5py.File(db_path, 'r') as h5:
        for blk in BLOCKS_EVAL:
            if blk not in h5['features']:
                print(f'  {blk} ABSENT dans {db_path.name} — ignoré')
                continue
            print(f'  {blk}...', end='', flush=True)
            X_all = h5['features'][blk][:].astype(np.float32)
            X = X_all[_mask_valid]

            lp_mean, lp_std = compute_lp(X)
            fisher           = compute_fisher(X)
            tau              = compute_tau(X)

            results[src_name][blk] = {
                'dim':    X.shape[1],
                'lp':     lp_mean * 100,
                'lp_std': lp_std * 100,
                'fisher': fisher,
                'tau':    tau,
            }
            print(f'  LP={lp_mean*100:.1f}%  Fisher={fisher:.2f}  τ={tau:.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Tableau comparatif & verdict
# ─────────────────────────────────────────────────────────────────────────────
baseline = 100.0 / len(_CATS_VALID)

lines = [
    '=' * 90,
    'ROBUSTESSE À LA SOURCE D\'IMAGES — block_0 vs autres blocs',
    f'PatchTagger (RGB traitées)  vs  Ouassim (grayscale brutes)',
    f'Protocole : PCA-{PCA_DIM}d, L2-norm, {N_FOLDS}-fold stratifié par image, class_weight balanced',
    f'Catégories : {[CATEGORIES[c] for c in _CATS_VALID]}  |  Patches : {_mask_valid.sum()}',
    f'Baseline aléatoire : {baseline:.1f}%',
    '=' * 90,
    '',
    f'{"Bloc":<14} {"Source":<14} {"Dim":>5} │ {"LP (%)":>12} │ {"Fisher J":>10} │ {"τ cross/intra":>14}',
    '─' * 90,
]

for blk in BLOCKS_EVAL:
    for src in ['PatchTagger', 'Ouassim']:
        if blk not in results.get(src, {}):
            continue
        r = results[src][blk]
        lines.append(
            f'{blk:<14} {src:<14} {r["dim"]:>5} │ '
            f'{r["lp"]:>6.1f} ± {r["lp_std"]:>4.1f}   │ '
            f'{r["fisher"]:>10.2f} │ '
            f'{r["tau"]:>14.3f}'
        )
    lines.append('─' * 90)

# Verdict par bloc
lines += ['', 'VERDICT PAR BLOC', '─' * 60]
for blk in BLOCKS_EVAL:
    pt = results.get('PatchTagger', {}).get(blk)
    ou = results.get('Ouassim', {}).get(blk)
    if pt is None or ou is None:
        continue
    delta_lp = ou['lp'] - pt['lp']
    stable   = abs(delta_lp) < 5.0
    best_pt  = (pt['lp'] == max(results['PatchTagger'][b]['lp']
                                for b in BLOCKS_EVAL if b in results['PatchTagger']))
    best_ou  = (ou['lp'] == max(results['Ouassim'][b]['lp']
                                for b in BLOCKS_EVAL if b in results['Ouassim']))
    lines.append(
        f'{blk:<14}  ΔLP={delta_lp:+.1f}%  '
        f'{"stable" if stable else "ΔLARGE"}  '
        f'{"★ meilleur PatchTagger" if best_pt else ""}'
        f'{"★ meilleur Ouassim" if best_ou else ""}'
    )

# Verdict global
best_pt_blk  = max(results['PatchTagger'], key=lambda b: results['PatchTagger'][b]['lp'])
best_ou_blk  = max(results['Ouassim'],     key=lambda b: results['Ouassim'][b]['lp'])
robust       = best_pt_blk == best_ou_blk

lines += [
    '',
    '─' * 60,
    f'Meilleur bloc PatchTagger : {best_pt_blk}  (LP={results["PatchTagger"][best_pt_blk]["lp"]:.1f}%)',
    f'Meilleur bloc Ouassim     : {best_ou_blk}  (LP={results["Ouassim"][best_ou_blk]["lp"]:.1f}%)',
    '',
    f'ROBUSTESSE : {"✓ OUI — même bloc optimal sur les deux sources" if robust else "✗ NON — le meilleur bloc change selon la source"}',
    f'→ {"Conclusions transférables aux images Ouassim." if robust else "Analyser les différences entre sources."}',
    '=' * 90,
]

tableau = '\n'.join(lines)
print('\n' + tableau)

with open(OUTPUT_DIR / 'robustness_table.txt', 'w') as _f:
    _f.write(tableau + '\n')

with open(OUTPUT_DIR / 'robustness_results.json', 'w') as _f:
    json.dump(results, _f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# Figure — LP et Fisher comparés
# ─────────────────────────────────────────────────────────────────────────────
_blks_ok = [b for b in BLOCKS_EVAL
            if b in results['PatchTagger'] and b in results['Ouassim']]
x = np.arange(len(_blks_ok))
w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = {'PatchTagger': '#1B4F72', 'Ouassim': '#E67E22'}

for ax, metric, ylabel, title in [
    (axes[0], 'lp',     'Balanced Accuracy (%)', 'Linear Probing'),
    (axes[1], 'fisher', 'Fisher J',               'Fisher Criterion'),
    (axes[2], 'tau',    'τ cross/intra',           'τ cross/intra'),
]:
    for i, src in enumerate(['PatchTagger', 'Ouassim']):
        vals = [results[src][b][metric] for b in _blks_ok]
        errs = ([results[src][b]['lp_std'] for b in _blks_ok]
                if metric == 'lp' else None)
        ax.bar(x + i * w - w / 2, vals, w,
               label=src, color=colors[src],
               yerr=errs, capsize=4, alpha=0.88)
    if metric == 'lp':
        ax.axhline(baseline, color='red', ls=':', lw=1.2,
                   label=f'Baseline {baseline:.1f}%')
    ax.set_xticks(x)
    ax.set_xticklabels(_blks_ok, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

plt.suptitle(
    'Robustesse à la source d\'images — PatchTagger vs Ouassim\n'
    f'Blocs évalués : {", ".join(_blks_ok)}',
    fontsize=11,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'robustness_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\nFichiers générés dans {OUTPUT_DIR} :')
for fname in ['robustness_table.txt', 'robustness_results.json', 'robustness_comparison.png']:
    p = OUTPUT_DIR / fname
    print(f'  {"✓" if p.exists() else "✗"}  {fname}')
