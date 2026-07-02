"""
Q1 : Les patches outliers (loin de leur groupe dans l'espace features) sont-ils
     les patches atypiquement SOMBRES ou CLAIRS de leur catégorie ?

Pipeline :
  1. Charger features[KEY] + métadonnées depuis datah5/*.h5
  2. PCA-50d + L2-norm → silhouette individuel par patch
  3. Intensité (mean/std pixel) depuis les images brutes
  4. Comparaison outliers vs non-outliers PAR CATÉGORIE (Mann-Whitney)
  5. Corrélation globale silhouette ↔ écart d'intensité
  6. 4 plots + synthèse txt/md

Verdict Q1 : l'intensité explique-t-elle les outliers ?
"""

# ============================================================
#  PARAMÈTRES — MODIFIER ICI
# ============================================================
H5_DIR        = 'data/feature_database'   # dossier contenant les *.h5
IMG_DIR       = 'Image_Ouassim'           # images brutes Ouassim
OUTPUT_DIR    = 'output_ouassim/q1_intensite'
KEY           = 'stage_2_fpn'
SIL_THRESHOLD = 0.0          # sil < SEUIL → outlier
SEED          = 42
PCA_DIM       = 50
CATS_EXCLUDE  = [2, 8, 10, 11, 12, 13]
MIN_N         = 30           # ignorer catégories avec < MIN_N patches
# ============================================================

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.decomposition import PCA
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(SEED)

_q1_ROOT   = Path(__file__).resolve().parent.parent
_q1_H5DIR  = _q1_ROOT / H5_DIR
_q1_IMGDIR = _q1_ROOT / IMG_DIR
_q1_OUTDIR = _q1_ROOT / OUTPUT_DIR
_q1_OUTDIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 0. Config catégories
# ──────────────────────────────────────────────────────────────────────────────
_q1_cfg_path = _q1_ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
if _q1_cfg_path.exists():
    with open(_q1_cfg_path) as _f:
        _q1_cfg = json.load(_f)
    _q1_CATNAMES = {int(k): v['name']
                    for k, v in _q1_cfg['available_categories'].items()}
else:
    _q1_CATNAMES = {}

def _q1_catname(cid):
    return _q1_CATNAMES.get(cid, f'Cat_{cid}')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Charger la base HDF5
# ──────────────────────────────────────────────────────────────────────────────
_q1_h5_files = sorted(_q1_H5DIR.glob('*.h5'))
if not _q1_h5_files:
    raise FileNotFoundError(f'Aucun .h5 dans {_q1_H5DIR}')

# Préférer la base Ouassim
_q1_H5 = next((p for p in _q1_h5_files if 'ouassim' in p.name.lower()),
               _q1_h5_files[0])
print(f'Base HDF5 : {_q1_H5.name}')

with h5py.File(_q1_H5, 'r') as _h5:
    _q1_all_names = np.array([
        n.decode('utf-8') if isinstance(n, (bytes, np.bytes_)) else str(n)
        for n in _h5['metadata/image_names'][:]
    ])
    _q1_all_cats = _h5['metadata/category_ids'][:].astype(int)
    _q1_all_pos  = _h5['metadata/positions'][:].astype(float)  # (N,4) x_min,y_min,x_max,y_max
    _q1_avail    = list(_h5['features'].keys())
    if KEY not in _q1_avail:
        raise KeyError(f'KEY={KEY!r} absent. Disponibles : {_q1_avail}')
    _q1_X_raw = _h5['features'][KEY][:].astype(np.float32)

print(f'Patches totaux : {len(_q1_all_cats)}')
print(f'Représentations disponibles : {_q1_avail}')

# ──────────────────────────────────────────────────────────────────────────────
# 2. Filtrage catégories valides
# ──────────────────────────────────────────────────────────────────────────────
_q1_cats_all = sorted(set(_q1_all_cats.tolist()) - set(CATS_EXCLUDE))
_q1_mask_v   = np.isin(_q1_all_cats, _q1_cats_all)

_q1_y     = _q1_all_cats[_q1_mask_v]
_q1_names = _q1_all_names[_q1_mask_v]
_q1_pos   = _q1_all_pos[_q1_mask_v]
_q1_X     = _q1_X_raw[_q1_mask_v]

# Garder uniquement les catégories avec >= MIN_N patches
_q1_cats_valid = [c for c in _q1_cats_all if (_q1_y == c).sum() >= MIN_N]
_q1_mask_v2    = np.isin(_q1_y, _q1_cats_valid)
_q1_y     = _q1_y[_q1_mask_v2]
_q1_names = _q1_names[_q1_mask_v2]
_q1_pos   = _q1_pos[_q1_mask_v2]
_q1_X     = _q1_X[_q1_mask_v2]

print(f'\nCatégories valides (>= {MIN_N} patches) :')
for _c in _q1_cats_valid:
    print(f'  {_c:3d} {_q1_catname(_c):25} : {(_q1_y==_c).sum():4d} patches')

# ──────────────────────────────────────────────────────────────────────────────
# 3. PCA-50d + L2-normalisation
# ──────────────────────────────────────────────────────────────────────────────
_q1_n_comp = min(PCA_DIM, _q1_X.shape[1])
_q1_pca    = PCA(n_components=_q1_n_comp, random_state=SEED)
_q1_X50    = _q1_pca.fit_transform(_q1_X)
_q1_norms  = np.linalg.norm(_q1_X50, axis=1, keepdims=True)
_q1_X50n   = _q1_X50 / np.where(_q1_norms < 1e-8, 1.0, _q1_norms)
print(f'\nPCA : {_q1_X.shape[1]}d → {_q1_n_comp}d  '
      f'(variance expliquée : {_q1_pca.explained_variance_ratio_.sum()*100:.1f}%)')

# ──────────────────────────────────────────────────────────────────────────────
# 4. Centroïdes et silhouette individuel
# ──────────────────────────────────────────────────────────────────────────────
_q1_centroids = {}
for _c in _q1_cats_valid:
    _mc = _q1_y == _c
    _mu = _q1_X50n[_mc].mean(axis=0)
    _q1_centroids[_c] = _mu / (np.linalg.norm(_mu) + 1e-8)

def _q1_cos_dist(vec, ref):
    return float(1.0 - np.dot(vec, ref))

_q1_N        = len(_q1_y)
_q1_sil      = np.zeros(_q1_N)
_q1_a_dist   = np.zeros(_q1_N)
_q1_b_dist   = np.zeros(_q1_N)
_q1_ncat     = np.zeros(_q1_N, dtype=int)

for _i, (_vec, _ci) in enumerate(zip(_q1_X50n, _q1_y)):
    _a = _q1_cos_dist(_vec, _q1_centroids[_ci])
    _others = [(c, _q1_cos_dist(_vec, _q1_centroids[c]))
               for c in _q1_cats_valid if c != _ci]
    _bc, _b = min(_others, key=lambda x: x[1])
    _q1_a_dist[_i] = _a
    _q1_b_dist[_i] = _b
    _q1_ncat[_i]   = _bc
    _q1_sil[_i]    = (_b - _a) / (max(_a, _b) + 1e-10)

_q1_is_outlier = _q1_sil < SIL_THRESHOLD

print(f'\nSilhouette individuel (KEY={KEY}) :')
print(f'  Moyen global : {_q1_sil.mean():+.3f}')
print(f'  Outliers (sil < {SIL_THRESHOLD}) : {_q1_is_outlier.sum()} / {_q1_N} '
      f'({100*_q1_is_outlier.mean():.1f}%)')

# ──────────────────────────────────────────────────────────────────────────────
# 5. Intensité des patches depuis les images brutes
# ──────────────────────────────────────────────────────────────────────────────
print('\nCalcul des intensités patch par patch...')
_q1_img_cache = {}   # img_name → np.array (H, W) uint8

def _q1_get_img(img_name):
    if img_name not in _q1_img_cache:
        _p = _q1_IMGDIR / img_name
        if not _p.exists():
            _q1_img_cache[img_name] = None
        else:
            _q1_img_cache[img_name] = np.array(Image.open(_p).convert('L'))
    return _q1_img_cache[img_name]

_q1_int_mean = np.full(_q1_N, np.nan)
_q1_int_std  = np.full(_q1_N, np.nan)
_q1_missing  = 0

for _i in range(_q1_N):
    _img = _q1_get_img(_q1_names[_i])
    if _img is None:
        _q1_missing += 1
        continue
    _H, _W = _img.shape
    _xmin, _ymin, _xmax, _ymax = [int(v) for v in _q1_pos[_i]]
    _xmin = max(0, _xmin); _ymin = max(0, _ymin)
    _xmax = min(_W, _xmax); _ymax = min(_H, _ymax)
    if _xmax <= _xmin or _ymax <= _ymin:
        continue
    _crop = _img[_ymin:_ymax, _xmin:_xmax].astype(np.float32)
    _q1_int_mean[_i] = _crop.mean()
    _q1_int_std[_i]  = _crop.std()

if _q1_missing:
    print(f'  ⚠ {_q1_missing} images manquantes ignorées')
print(f'  {(~np.isnan(_q1_int_mean)).sum()} patches avec intensité calculée')

# Masque : patches avec intensité valide
_q1_ok = ~np.isnan(_q1_int_mean)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Analyse PAR CATÉGORIE : outliers vs non-outliers en intensité
# ──────────────────────────────────────────────────────────────────────────────
print(f'\n{"─"*85}')
_header = (f'{"Cat":>3}  {"Nom":<22}  {"N_out":>5}  {"N_ok":>5}  '
           f'{"μ_out":>7}  {"μ_ok":>7}  {"Δ":>7}  {"p-MW":>8}  {"signif":>7}')
print(_header)
print(f'{"─"*85}')

_q1_per_cat = {}   # c → dict avec les stats

for _c in _q1_cats_valid:
    _mc      = (_q1_y == _c) & _q1_ok
    _mc_out  = _mc & _q1_is_outlier
    _mc_norm = _mc & ~_q1_is_outlier

    _int_out  = _q1_int_mean[_mc_out]
    _int_norm = _q1_int_mean[_mc_norm]

    _mu_out  = _int_out.mean()  if len(_int_out)  > 0 else np.nan
    _mu_norm = _int_norm.mean() if len(_int_norm) > 0 else np.nan
    _delta   = _mu_out - _mu_norm if not np.isnan(_mu_out) else np.nan

    # Mann-Whitney U (non paramétrique)
    if len(_int_out) >= 3 and len(_int_norm) >= 3:
        _stat, _pval = stats.mannwhitneyu(_int_out, _int_norm,
                                           alternative='two-sided')
        _sig = '***' if _pval < 0.001 else ('**' if _pval < 0.01 else
               ('*' if _pval < 0.05 else 'ns'))
    else:
        _pval = np.nan
        _sig  = 'n/a'

    # Z-score d'intensité des outliers par rapport à la distribution de la catégorie
    _all_int_c = _q1_int_mean[_mc]
    _mu_c = _all_int_c.mean() if len(_all_int_c) > 0 else np.nan
    _sd_c = _all_int_c.std()  if len(_all_int_c) > 1 else np.nan
    _z_outliers = ((_int_out - _mu_c) / (_sd_c + 1e-6)) if (
        len(_int_out) > 0 and not np.isnan(_mu_c)) else np.array([np.nan])

    _q1_per_cat[_c] = dict(
        int_out=_int_out, int_norm=_int_norm,
        mu_out=_mu_out, mu_norm=_mu_norm, delta=_delta,
        pval=_pval, sig=_sig,
        mu_c=_mu_c, sd_c=_sd_c, z_outliers=_z_outliers,
        n_out=len(_int_out), n_norm=len(_int_norm)
    )

    _mu_out_s  = f'{_mu_out:7.1f}' if not np.isnan(_mu_out)  else '    nan'
    _mu_norm_s = f'{_mu_norm:7.1f}' if not np.isnan(_mu_norm) else '    nan'
    _delta_s   = f'{_delta:+7.1f}'  if not np.isnan(_delta)   else '    nan'
    _pval_s    = f'{_pval:.2e}' if not np.isnan(_pval) else '     nan'
    print(f'{_c:>3}  {_q1_catname(_c):<22}  {len(_int_out):>5}  '
          f'{len(_int_norm):>5}  {_mu_out_s}  {_mu_norm_s}  '
          f'{_delta_s}  {_pval_s}  {_sig:>7}')

print(f'{"─"*85}')

# ──────────────────────────────────────────────────────────────────────────────
# 7. Corrélation globale silhouette ↔ écart d'intensité
# ──────────────────────────────────────────────────────────────────────────────

# Calcul de l'écart d'intensité normalisé par catégorie
_q1_int_dev_lum = np.full(_q1_N, np.nan)   # |intensité - moyenne_catégorie|
_q1_int_dev_std = np.full(_q1_N, np.nan)   # |std_patch - moyenne_std_catégorie|

for _c in _q1_cats_valid:
    _mc = (_q1_y == _c) & _q1_ok
    if _mc.sum() < 2:
        continue
    _mu_c = _q1_int_mean[_mc].mean()
    _mu_s = _q1_int_std[_mc].mean()
    _q1_int_dev_lum[_mc] = np.abs(_q1_int_mean[_mc] - _mu_c)
    _q1_int_dev_std[_mc] = np.abs(_q1_int_std[_mc]  - _mu_s)

_q1_corr_mask = _q1_ok & ~np.isnan(_q1_int_dev_lum) & ~np.isnan(_q1_int_dev_std)

_sil_corr  = _q1_sil[_q1_corr_mask]
_dev_lum   = _q1_int_dev_lum[_q1_corr_mask]
_dev_ctr   = _q1_int_dev_std[_q1_corr_mask]

r_lum, p_lum = stats.spearmanr(_dev_lum, _sil_corr)
r_ctr, p_ctr = stats.spearmanr(_dev_ctr, _sil_corr)
r_lum_p, p_lum_p = stats.pearsonr(_dev_lum, _sil_corr)

print(f'\nCorrélation silhouette ↔ écart d\'intensité (N={_q1_corr_mask.sum()}) :')
print(f'  Spearman (|lum - μ|) : ρ = {r_lum:+.3f}  p = {p_lum:.2e}')
print(f'  Spearman (|ctr - μ|) : ρ = {r_ctr:+.3f}  p = {p_ctr:.2e}')
print(f'  Pearson  (|lum - μ|) : r = {r_lum_p:+.3f}  p = {p_lum_p:.2e}')

# ──────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ──────────────────────────────────────────────────────────────────────────────
_q1_NCATS = len(_q1_cats_valid)
_q1_COLORS = {'outlier': '#e74c3c', 'normal': '#3498db'}

# --- Plot 1 : boxplot intensité outliers vs non-outliers par catégorie --------
_fig1, _axes1 = plt.subplots(1, _q1_NCATS,
                               figsize=(max(12, 2.8*_q1_NCATS), 5),
                               sharey=False)
if _q1_NCATS == 1:
    _axes1 = [_axes1]

for _ci, _c in enumerate(_q1_cats_valid):
    _ax    = _axes1[_ci]
    _d     = _q1_per_cat[_c]
    _data  = [_d['int_out'], _d['int_norm']]
    _labs  = [f'Outlier\n(N={_d["n_out"]})', f'Normal\n(N={_d["n_norm"]})']
    _cols  = [_q1_COLORS['outlier'], _q1_COLORS['normal']]

    _bp = _ax.boxplot(_data, labels=_labs, patch_artist=True,
                      widths=0.55, notch=False,
                      medianprops=dict(color='black', linewidth=2))
    for _patch, _col in zip(_bp['boxes'], _cols):
        _patch.set_facecolor(_col)
        _patch.set_alpha(0.75)

    # Ligne moyenne catégorie
    if not np.isnan(_d['mu_c']):
        _ax.axhline(_d['mu_c'], color='grey', linestyle='--',
                    linewidth=1, alpha=0.7, label=f'μ_cat={_d["mu_c"]:.0f}')

    # Annotation p-value
    _ax.set_title(
        f'{_q1_catname(_c)[:18]}\np={_d["pval"]:.2e} {_d["sig"]}',
        fontsize=8, fontweight='bold'
    )
    _ax.set_ylabel('Intensité (0–255)' if _ci == 0 else '', fontsize=8)
    _ax.tick_params(labelsize=8)
    _ax.grid(axis='y', alpha=0.3)

_fig1.suptitle(
    f'Intensité moyenne — Outliers vs Non-outliers par catégorie  [{KEY}]\n'
    f'Outlier = sil < {SIL_THRESHOLD}',
    fontsize=11, fontweight='bold'
)
_fig1.tight_layout()
_save1 = _q1_OUTDIR / 'intensite_outliers_vs_normaux.png'
_fig1.savefig(_save1, dpi=150, bbox_inches='tight')
plt.close(_fig1)
print(f'\n✓ Plot 1 → {_save1.name}')

# --- Plot 2 : scatter silhouette ↔ écart d'intensité -------------------------
_fig2, (_ax2a, _ax2b) = plt.subplots(1, 2, figsize=(13, 5))

for _ax2, _xvals, _xlabel, _r, _p in [
    (_ax2a, _dev_lum, '|intensité – μ_catégorie|  (luminosité)', r_lum, p_lum),
    (_ax2b, _dev_ctr, '|contraste – μ_catégorie|  (σ pixel)',    r_ctr, p_ctr),
]:
    # Couleur par catégorie
    _cmap_cats = plt.get_cmap('tab10')
    for _ci2, _c2 in enumerate(_q1_cats_valid):
        _m2 = _q1_y[_q1_corr_mask] == _c2
        _ax2.scatter(_xvals[_m2], _sil_corr[_m2],
                     s=8, alpha=0.45, color=_cmap_cats(_ci2 % 10),
                     label=_q1_catname(_c2)[:14])
    # Ligne de tendance
    _coef = np.polyfit(_xvals, _sil_corr, 1)
    _xfit = np.linspace(_xvals.min(), _xvals.max(), 200)
    _ax2.plot(_xfit, np.polyval(_coef, _xfit),
              color='black', linewidth=1.8, linestyle='--', label='tendance')
    _ax2.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    _ax2.set_xlabel(_xlabel, fontsize=9)
    _ax2.set_ylabel('Silhouette individuel', fontsize=9)
    _ax2.set_title(
        f'Silhouette ↔ écart intensité\nSpearman ρ={_r:+.3f}  p={_p:.2e}',
        fontsize=10, fontweight='bold'
    )
    _ax2.grid(alpha=0.25)
    _ax2.legend(fontsize=7, markerscale=2, ncol=2)

_fig2.suptitle(
    f'Corrélation silhouette ↔ atypicité en intensité  [{KEY}]\n'
    '(corrélation négative = outliers features = patches atypiques en intensité)',
    fontsize=10, fontweight='bold'
)
_fig2.tight_layout()
_save2 = _q1_OUTDIR / 'silhouette_vs_ecart_intensite.png'
_fig2.savefig(_save2, dpi=150, bbox_inches='tight')
plt.close(_fig2)
print(f'✓ Plot 2 → {_save2.name}')

# --- Plot 3 : distribution intensité par catégorie, outliers marqués ----------
_ncols3 = min(4, _q1_NCATS)
_nrows3 = int(np.ceil(_q1_NCATS / _ncols3))
_fig3, _axes3 = plt.subplots(_nrows3, _ncols3,
                               figsize=(4.5*_ncols3, 3.5*_nrows3),
                               squeeze=False)

for _ci3, _c3 in enumerate(_q1_cats_valid):
    _ax3 = _axes3[_ci3 // _ncols3, _ci3 % _ncols3]
    _mc3 = (_q1_y == _c3) & _q1_ok
    _mc3_out  = _mc3 & _q1_is_outlier
    _mc3_norm = _mc3 & ~_q1_is_outlier

    _int_all  = _q1_int_mean[_mc3]
    _int_out3 = _q1_int_mean[_mc3_out]
    _int_nrm3 = _q1_int_mean[_mc3_norm]

    _bins3 = np.linspace(_int_all.min()-1, _int_all.max()+1, 30) if len(_int_all) else np.linspace(0,255,30)

    if len(_int_nrm3):
        _ax3.hist(_int_nrm3, bins=_bins3, color=_q1_COLORS['normal'],
                  alpha=0.6, label=f'Normal ({len(_int_nrm3)})')
    if len(_int_out3):
        _ax3.hist(_int_out3, bins=_bins3, color=_q1_COLORS['outlier'],
                  alpha=0.75, label=f'Outlier ({len(_int_out3)})')
    if not np.isnan(_q1_per_cat[_c3]['mu_c']):
        _ax3.axvline(_q1_per_cat[_c3]['mu_c'], color='black',
                     linewidth=1.5, linestyle='--', label=f'μ={_q1_per_cat[_c3]["mu_c"]:.0f}')

    _ax3.set_title(
        f'{_q1_catname(_c3)[:20]}\np={_q1_per_cat[_c3]["pval"]:.2e} {_q1_per_cat[_c3]["sig"]}',
        fontsize=8
    )
    _ax3.set_xlabel('Intensité (0–255)', fontsize=7)
    _ax3.set_ylabel('N patches', fontsize=7)
    _ax3.legend(fontsize=7)
    _ax3.grid(alpha=0.25)

# Masquer axes vides
for _ci3 in range(_q1_NCATS, _nrows3 * _ncols3):
    _axes3[_ci3 // _ncols3, _ci3 % _ncols3].axis('off')

_fig3.suptitle(
    f'Distribution intensité par catégorie  [{KEY}]\nBleu=non-outlier · Rouge=outlier · Noir=μ catégorie',
    fontsize=11, fontweight='bold'
)
_fig3.tight_layout()
_save3 = _q1_OUTDIR / 'distribution_intensite_par_cat.png'
_fig3.savefig(_save3, dpi=150, bbox_inches='tight')
plt.close(_fig3)
print(f'✓ Plot 3 → {_save3.name}')

# --- Plot 4 : heatmap z-score intensité outliers vs non-outliers --------------
# Ligne = catégorie, colonne = groupe (outlier / normal)
# Valeur = z-score moyen d'intensité du groupe par rapport à μ_catégorie

_q1_hm_data  = np.full((_q1_NCATS, 2), np.nan)
_q1_hm_cats  = []
for _ci4, _c4 in enumerate(_q1_cats_valid):
    _d4 = _q1_per_cat[_c4]
    if not np.isnan(_d4['sd_c']) and _d4['sd_c'] > 0:
        if len(_d4['int_out']) > 0 and not np.isnan(_d4['mu_out']):
            _q1_hm_data[_ci4, 0] = (_d4['mu_out'] - _d4['mu_c']) / _d4['sd_c']
        if len(_d4['int_norm']) > 0 and not np.isnan(_d4['mu_norm']):
            _q1_hm_data[_ci4, 1] = (_d4['mu_norm'] - _d4['mu_c']) / _d4['sd_c']
    _q1_hm_cats.append(_q1_catname(_c4)[:20])

_fig4, _ax4 = plt.subplots(figsize=(5, max(4, 0.6 * _q1_NCATS + 2)))
_vmax4 = max(np.nanmax(np.abs(_q1_hm_data)), 0.5)
_im4 = _ax4.imshow(_q1_hm_data, cmap='RdBu_r', aspect='auto',
                    vmin=-_vmax4, vmax=_vmax4)
plt.colorbar(_im4, ax=_ax4, label='Z-score intensité')

_ax4.set_xticks([0, 1])
_ax4.set_xticklabels(['Outliers\n(sil<0)', 'Non-outliers\n(sil≥0)'], fontsize=10)
_ax4.set_yticks(range(_q1_NCATS))
_ax4.set_yticklabels(_q1_hm_cats, fontsize=9)
_ax4.set_title(
    f'Z-score d\'intensité par catégorie et groupe\n'
    f'(rouge = plus sombre que μ_cat, bleu = plus clair)\n[{KEY}]',
    fontsize=10, fontweight='bold'
)

# Annoter les valeurs
for _ri in range(_q1_NCATS):
    for _cj in range(2):
        _v = _q1_hm_data[_ri, _cj]
        if not np.isnan(_v):
            _ax4.text(_cj, _ri, f'{_v:+.2f}', ha='center', va='center',
                      fontsize=8, color='white' if abs(_v) > _vmax4*0.6 else 'black')

_fig4.tight_layout()
_save4 = _q1_OUTDIR / 'heatmap_zscore_intensite.png'
_fig4.savefig(_save4, dpi=150, bbox_inches='tight')
plt.close(_fig4)
print(f'✓ Plot 4 → {_save4.name}')

# ──────────────────────────────────────────────────────────────────────────────
# 9. VERDICT Q1
# ──────────────────────────────────────────────────────────────────────────────
_q1_sig_cats  = [c for c in _q1_cats_valid
                 if _q1_per_cat[c]['pval'] < 0.05 and not np.isnan(_q1_per_cat[c]['pval'])]
_q1_nsig_cats = [c for c in _q1_cats_valid if c not in _q1_sig_cats]

# Corrélation forte si |ρ| > 0.2 avec p significatif
_q1_corr_strong = abs(r_lum) > 0.2 and p_lum < 0.05
_q1_corr_mod    = abs(r_lum) > 0.1 and p_lum < 0.05

if _q1_corr_strong and len(_q1_sig_cats) >= len(_q1_cats_valid) * 0.5:
    _q1_verdict = 'OUI NET'
    _q1_verdict_detail = (
        'L\'intensité explique significativement les outliers. '
        'La corrélation est forte et la majorité des catégories montrent une différence significative. '
        'Passer à Q2 : les features encodent-elles l\'intensité ?'
    )
elif _q1_corr_mod or len(_q1_sig_cats) >= 2:
    _q1_verdict = 'PARTIEL'
    _sig_names = ', '.join(_q1_catname(c) for c in _q1_sig_cats)
    _nsig_names = ', '.join(_q1_catname(c) for c in _q1_nsig_cats)
    _q1_verdict_detail = (
        f'L\'intensité explique partiellement les outliers.\n'
        f'Catégories sensibles (p<0.05) : {_sig_names or "aucune"}.\n'
        f'Catégories non sensibles : {_nsig_names or "aucune"}.\n'
        'Pour les catégories sensibles, explorer Q2. Pour les autres, le problème est ailleurs.'
    )
else:
    _q1_verdict = 'NON'
    _q1_verdict_detail = (
        'L\'intensité N\'explique PAS les outliers. '
        'La corrélation est faible et aucune catégorie ne montre de différence significative. '
        'Le problème vient d\'ailleurs (structure de texture, domain shift).'
    )

print(f'\n{"═"*60}')
print(f'  VERDICT Q1 : {_q1_verdict}')
print(f'{"═"*60}')
print(_q1_verdict_detail)
print(f'\n  Corrélation Spearman (luminosité) : ρ={r_lum:+.3f}  p={p_lum:.2e}')
print(f'  Corrélation Spearman (contraste)  : ρ={r_ctr:+.3f}  p={p_ctr:.2e}')
print(f'  Catégories avec diff. significative : {len(_q1_sig_cats)}/{len(_q1_cats_valid)}')
print(f'{"═"*60}')

# ──────────────────────────────────────────────────────────────────────────────
# 10. Synthèse .txt + .md
# ──────────────────────────────────────────────────────────────────────────────
_q1_lines_txt = [
    f'Q1 — INTENSITÉ ET OUTLIERS  [{KEY}]',
    f'SIL_THRESHOLD={SIL_THRESHOLD}   PCA_DIM={PCA_DIM}   N_total={_q1_N}',
    '',
    f'Outliers : {_q1_is_outlier.sum()} / {_q1_N}  ({100*_q1_is_outlier.mean():.1f}%)',
    '',
    '── RÉSULTATS PAR CATÉGORIE ──────────────────────────────────────────────',
    f'{"Cat":<5} {"Nom":<24} {"N_out":>5} {"N_ok":>5} {"μ_out":>7} '
    f'{"μ_ok":>7} {"Δ":>7} {"p-MW":>10} {"sig":>5} {"sombre?":>8}',
    '─' * 85,
]

for _c in _q1_cats_valid:
    _d = _q1_per_cat[_c]
    _sombre = ''
    if not np.isnan(_d['delta']):
        _sombre = 'SOMBRE' if _d['delta'] < -3 else ('CLAIR' if _d['delta'] > 3 else 'similaire')
    _mu_out_s  = f'{_d["mu_out"]:7.1f}' if not np.isnan(_d["mu_out"])  else '    nan'
    _mu_norm_s = f'{_d["mu_norm"]:7.1f}' if not np.isnan(_d["mu_norm"]) else '    nan'
    _delta_s   = f'{_d["delta"]:+7.1f}'  if not np.isnan(_d["delta"])   else '    nan'
    _pval_s    = f'{_d["pval"]:.2e}' if not np.isnan(_d["pval"]) else '       nan'
    _q1_lines_txt.append(
        f'{_c:<5} {_q1_catname(_c):<24} {_d["n_out"]:>5} {_d["n_norm"]:>5} '
        f'{_mu_out_s} {_mu_norm_s} {_delta_s} {_pval_s} {_d["sig"]:>5} {_sombre:>8}'
    )

_q1_lines_txt += [
    '',
    '── CORRÉLATION GLOBALE ──────────────────────────────────────────────────',
    f'Spearman silhouette ↔ |lum - μ_cat| : ρ = {r_lum:+.3f}  p = {p_lum:.2e}',
    f'Spearman silhouette ↔ |ctr - μ_cat| : ρ = {r_ctr:+.3f}  p = {p_ctr:.2e}',
    f'Pearson  silhouette ↔ |lum - μ_cat| : r = {r_lum_p:+.3f}  p = {p_lum_p:.2e}',
    '(corrélation négative = outliers features = patches atypiques en intensité)',
    '',
    '── VERDICT Q1 ───────────────────────────────────────────────────────────',
    f'VERDICT : {_q1_verdict}',
    '',
    _q1_verdict_detail,
    '',
    f'Catégories sensibles   (p<0.05) : {", ".join(_q1_catname(c) for c in _q1_sig_cats) or "aucune"}',
    f'Catégories non sensibles        : {", ".join(_q1_catname(c) for c in _q1_nsig_cats) or "aucune"}',
    '',
    '── RECOMMANDATION ───────────────────────────────────────────────────────',
]

if _q1_verdict == 'OUI NET':
    _q1_lines_txt.append(
        '→ Passer à Q2 : les features SAM2 encodent-elles l\'intensité absolue ?\n'
        '   (Si oui : l\'intensité contamine l\'espace features → Q3 normalisation)'
    )
elif _q1_verdict == 'PARTIEL':
    _q1_lines_txt.append(
        '→ Pour les catégories sensibles : passer à Q2 (features = intensité ?).\n'
        '→ Pour les autres : explorer d\'autres causes (structure fine, domain shift,\n'
        '   artefacts d\'acquisition, annotations de frontière).'
    )
else:
    _q1_lines_txt.append(
        '→ L\'intensité seule n\'explique pas les outliers.\n'
        '→ Explorer : structure fine de texture (fréquence spatiale),\n'
        '   patches de frontière entre catégories, artefacts d\'acquisition.'
    )

_save_txt = _q1_OUTDIR / 'q1_intensite_outliers.txt'
(_save_txt).write_text('\n'.join(_q1_lines_txt), encoding='utf-8')
print(f'\n✓ Synthèse txt → {_save_txt.name}')

# ── Markdown ──────────────────────────────────────────────────────────────────
_q1_md_rows = ['| Cat | Nom | N_out | N_ok | μ_out | μ_ok | Δ | p-MW | sig |',
               '|-----|-----|------:|-----:|------:|-----:|--:|-----:|-----|']
for _c in _q1_cats_valid:
    _d = _q1_per_cat[_c]
    _mu_o = f'{_d["mu_out"]:.1f}' if not np.isnan(_d["mu_out"]) else '—'
    _mu_n = f'{_d["mu_norm"]:.1f}' if not np.isnan(_d["mu_norm"]) else '—'
    _dl   = f'{_d["delta"]:+.1f}' if not np.isnan(_d["delta"]) else '—'
    _pv   = f'{_d["pval"]:.2e}' if not np.isnan(_d["pval"]) else '—'
    _q1_md_rows.append(
        f'| {_c} | {_q1_catname(_c)} | {_d["n_out"]} | {_d["n_norm"]} '
        f'| {_mu_o} | {_mu_n} | {_dl} | {_pv} | {_d["sig"]} |'
    )

_q1_md = f"""# Q1 — Intensité et outliers

**Configuration** : KEY=`{KEY}` · SIL_THRESHOLD={SIL_THRESHOLD} · PCA={PCA_DIM}d · N={_q1_N} patches

## Résultats par catégorie

{chr(10).join(_q1_md_rows)}

## Corrélation globale silhouette ↔ écart d'intensité

| Métrique | ρ / r | p-value |
|----------|------:|--------:|
| Spearman (luminosité) | {r_lum:+.3f} | {p_lum:.2e} |
| Spearman (contraste σ) | {r_ctr:+.3f} | {p_ctr:.2e} |
| Pearson (luminosité) | {r_lum_p:+.3f} | {p_lum_p:.2e} |

> Corrélation négative = outliers features = patches atypiques en intensité → Q1 confirmé.

## Verdict Q1 : **{_q1_verdict}**

{_q1_verdict_detail}

- Catégories sensibles (p<0.05) : {', '.join('**'+_q1_catname(c)+'**' for c in _q1_sig_cats) or '_aucune_'}
- Catégories non sensibles : {', '.join(_q1_catname(c) for c in _q1_nsig_cats) or '_aucune_'}

## Plots générés

| Fichier | Contenu |
|---------|---------|
| `intensite_outliers_vs_normaux.png` | Boxplot intensité outliers vs non-outliers par catégorie |
| `silhouette_vs_ecart_intensite.png` | Scatter silhouette ↔ écart d'intensité + tendance |
| `distribution_intensite_par_cat.png` | Distribution intensité avec outliers marqués |
| `heatmap_zscore_intensite.png` | Z-score d'intensité outliers vs non-outliers |
"""

_save_md = _q1_OUTDIR / 'q1_intensite_outliers.md'
_save_md.write_text(_q1_md, encoding='utf-8')
print(f'✓ Synthèse md  → {_save_md.name}')
print(f'\nTous les fichiers dans : {_q1_OUTDIR.relative_to(_q1_ROOT)}')
