#!/usr/bin/env python3
"""
boundary_traversal_meb.py
Analyse comment block_0 de TextureSAM se comporte en traversant les
frontières texturales naturelles dans les images MEB.
"""
import os, sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
ROOT         = Path('/home/aidouni/meb_texture_seg')
DB_PATH      = ROOT / 'data/feature_database/database_meb.h5'
IMG_DIR      = ROOT / 'PatchTagger_Output' / 'full_images'
CHECKPOINT   = 'checkpoints/sam2.1_hiera_small_1.pt'
SAM2_DIR     = ROOT / 'TextureSAM' / 'sam2'
OUTPUT_DIR   = ROOT / 'outputs' / 'boundary_analysis'

SEED         = 42
N_BOUNDARIES = 5
N_STEPS      = 11
PCA_DIM      = 10
REG_COVAR    = 1e-1
PERCENTILE   = 5
ADJ_TOL      = 10

CATS_EXCLUDE = [2, 8, 10, 11, 12, 13]
CATEGORIES   = {1:'Homogène', 3:'Faisceaux', 4:'Filaments',
                5:'Strat.rect.', 6:'Strat.sin.', 7:'Granuleux', 9:'Trou'}
CAT_COLORS   = {1:'#4CAF50', 3:'#2196F3', 4:'#9C27B0',
                5:'#FF9800', 6:'#F44336', 7:'#00BCD4', 9:'#795548'}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Lecture HDF5
# ─────────────────────────────────────────────────────────────────────────────
print('Lecture HDF5...')
with h5py.File(DB_PATH, 'r') as _h5:
    _trav_NAMES = _h5['metadata/image_names'][:]
    _trav_CATS  = _h5['metadata/category_ids'][:].astype(int)
    _trav_POS   = _h5['metadata/positions'][:].astype(int)
    _trav_X_all = _h5['features/block_0'][:].astype(np.float32)

_trav_mask_valid = ~np.isin(_trav_CATS, CATS_EXCLUDE)
_trav_X_valid    = _trav_X_all[_trav_mask_valid]
_trav_y_valid    = _trav_CATS[_trav_mask_valid]
_trav_CATS_VALID = sorted(set(_trav_y_valid.tolist()))
print(f'  {len(_trav_X_valid)} patches valides, catégories : {_trav_CATS_VALID}')

# ─────────────────────────────────────────────────────────────────────────────
# Chargement du modèle — import direct depuis build_feature_database.py
# (même encodeur, même checkpoint, mêmes hooks que la construction de la base)
# ─────────────────────────────────────────────────────────────────────────────
print('Chargement modèle...')
try:
    sys.path.insert(0, str(ROOT))
    from build_feature_database import (
        load_model             as _bfdb_load_model,
        register_hooks         as _bfdb_register_hooks,
        remove_hooks           as _bfdb_remove_hooks,
        preprocess             as _bfdb_preprocess,
        extract_patch_features as _bfdb_extract_patch,
    )
except ImportError as _e:
    print(f'[ERREUR] Import build_feature_database impossible.')
    print(f'  sys.path tenté : {str(ROOT)}')
    print(f'  Détail : {_e}')
    sys.exit(1)

_trav_device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_trav_encoder  = _bfdb_load_model(ROOT / CHECKPOINT, str(_trav_device))
_trav_captured, _trav_handles = _bfdb_register_hooks(_trav_encoder)
print(f'  Encodeur chargé ({sum(p.numel() for p in _trav_encoder.parameters())/1e6:.1f}M params)')


def _trav_l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# PCA + GMM — entraînés sur TOUS les patches valides
# ─────────────────────────────────────────────────────────────────────────────
print('Entraînement PCA + GMM...')
_trav_pca    = PCA(n_components=PCA_DIM, random_state=SEED)
_trav_X_pca  = _trav_pca.fit_transform(_trav_X_valid)
_trav_X_norm = normalize(_trav_X_pca, norm='l2')

_trav_gmms       = {}
_trav_thresholds = {}
_trav_centroids  = {}

for _c in _trav_CATS_VALID:
    _mask_c = _trav_y_valid == _c
    _feat_c = _trav_X_norm[_mask_c]
    _trav_centroids[_c] = _trav_l2(_feat_c.mean(axis=0))
    _gmm = GaussianMixture(n_components=1, covariance_type='full',
                            reg_covar=REG_COVAR, random_state=SEED)
    _gmm.fit(_feat_c)
    _trav_gmms[_c]       = _gmm
    _trav_thresholds[_c] = float(np.percentile(_gmm.score_samples(_feat_c), PERCENTILE))

print(f'  GMM entraînés pour {_trav_CATS_VALID}')

# ─────────────────────────────────────────────────────────────────────────────
# Recalcul des frontières depuis HDF5 + sélection prioritisée
# ─────────────────────────────────────────────────────────────────────────────
print('Calcul des frontières...')

_trav_PAIR_PRIORITY = {
    (7, 9): 100, (1, 9): 95,  (6, 7): 90, (5, 7): 88,
    (1, 7): 85,  (3, 9): 80,  (5, 9): 80, (6, 9): 80,
    (4, 7): 75,  (4, 9): 75,  (3, 7): 70, (1, 3): 65,
    (3, 6): 60,  (3, 5): 60,  (4, 6): 60, (4, 5): 55,
    (5, 6): 50,  (1, 5): 70,  (1, 6): 70,
    (3, 4): 0,   # continuum — SKIP
}

_trav_by_img_all = defaultdict(list)
for _i in range(len(_trav_NAMES)):
    if _trav_CATS[_i] not in CATS_EXCLUDE:
        _trav_by_img_all[_trav_NAMES[_i]].append(_i)

_trav_all_bnd = []
for _nm, _idxs in _trav_by_img_all.items():
    for _a in range(len(_idxs)):
        for _b in range(_a + 1, len(_idxs)):
            _ia, _ib   = _idxs[_a], _idxs[_b]
            _ca, _cb   = int(_trav_CATS[_ia]), int(_trav_CATS[_ib])
            if _ca == _cb:
                continue
            _x1a,_y1a,_x2a,_y2a = _trav_POS[_ia]
            _x1b,_y1b,_x2b,_y2b = _trav_POS[_ib]
            _y_ov = _y1a < _y2b and _y1b < _y2a
            _x_ov = _x1a < _x2b and _x1b < _x2a
            _h_cl = abs(_x2a-_x1b) <= ADJ_TOL or abs(_x2b-_x1a) <= ADJ_TOL
            _v_cl = abs(_y2a-_y1b) <= ADJ_TOL or abs(_y2b-_y1a) <= ADJ_TOL
            _adj  = 'H' if (_h_cl and _y_ov) else ('V' if (_v_cl and _x_ov) else None)
            if _adj:
                _pair = (min(_ca, _cb), max(_ca, _cb))
                _trav_all_bnd.append({
                    'image': _nm, 'idx_a': _ia, 'idx_b': _ib,
                    'cat_a': _ca, 'cat_b': _cb, 'pair': _pair,
                })

print(f'  {len(_trav_all_bnd)} frontières trouvées au total')

# Priorité : une par paire de catégories, ordonnée par score
_trav_scored = sorted(
    [(r, _trav_PAIR_PRIORITY.get(r['pair'], 20))
     for r in _trav_all_bnd
     if _trav_PAIR_PRIORITY.get(r['pair'], 20) > 0
     and (IMG_DIR / r['image'].decode()).exists()],
    key=lambda x: -x[1],
)

_trav_selected  = []
_trav_seen_pairs = {}
for _r, _pr in _trav_scored:
    _p = _r['pair']
    if _p not in _trav_seen_pairs:
        _trav_selected.append(_r)
        _trav_seen_pairs[_p] = 1
    if len(_trav_selected) >= N_BOUNDARIES:
        break
# fill remaining slots (allow ≤ 2 per pair)
if len(_trav_selected) < N_BOUNDARIES:
    for _r, _pr in _trav_scored:
        if _r not in _trav_selected and _trav_seen_pairs.get(_r['pair'], 0) < 2:
            _trav_selected.append(_r)
            _trav_seen_pairs[_r['pair']] = _trav_seen_pairs.get(_r['pair'], 0) + 1
        if len(_trav_selected) >= N_BOUNDARIES:
            break

print(f'  {len(_trav_selected)} frontières sélectionnées :')
for _r in _trav_selected:
    print(f'    {CATEGORIES[_r["cat_a"]]:<15} ↔ {CATEGORIES[_r["cat_b"]]:<15}'
          f'  img={_r["image"].decode()[:55]}')

# ─────────────────────────────────────────────────────────────────────────────
# Forward passes — un par image source
# ─────────────────────────────────────────────────────────────────────────────
print('\nForward passes...')
_trav_by_img_sel = defaultdict(list)
for _r in _trav_selected:
    _trav_by_img_sel[_r['image']].append(_r)

_trav_feat_maps  = {}  # nm → (H_feat, W_feat, C)
_trav_img_sizes  = {}  # nm → (orig_W, orig_H)
_trav_img_arrays = {}  # nm → (H, W, 3) uint8

try:
    for _nm in _trav_by_img_sel:
        try:
            _img_path = IMG_DIR / _nm.decode()
            _trav_img_arrays[_nm] = np.array(Image.open(_img_path).convert('RGB'))
            _trav_captured.clear()
            _trav_tensor, _orig_H, _orig_W = _bfdb_preprocess(
                _img_path, str(_trav_device))
            with torch.no_grad():
                _trav_encoder(_trav_tensor)
            _trav_feat_maps[_nm] = _trav_captured['block_0'][0].cpu().numpy()
            _trav_img_sizes[_nm] = (_orig_W, _orig_H)  # convention (W, H)
            _Hf, _Wf, _Cf = _trav_feat_maps[_nm].shape
            print(f'  {_nm.decode()[:60]} → feat_map {_Hf}×{_Wf}×{_Cf}')
        except FileNotFoundError as _e:
            print(f'  SKIP (image manquante) : {_e}')

    # ── Assertion de validation sur 3 patches témoins via extract_patch_features ──
    # Même pipeline exact que build_feature_database.py : forward pass → hook → extract
    print('\nAssertion cosine > 0.98 sur 3 patches témoins...')
    _trav_val_checked = 0
    _trav_val_failed  = 0
    for _nm_v, _bnds_v in _trav_by_img_sel.items():
        if _nm_v not in _trav_feat_maps or _trav_val_checked >= 3:
            continue
        _trav_gi_v   = _bnds_v[0]['idx_a']
        _x1v, _y1v, _x2v, _y2v = _trav_POS[_trav_gi_v]
        _patch_v = {'x_min': int(_x1v), 'y_min': int(_y1v),
                    'x_max': int(_x2v), 'y_max': int(_y2v)}
        _trav_captured.clear()
        _trav_tensor_v, _orig_Hv, _orig_Wv = _bfdb_preprocess(
            IMG_DIR / _nm_v.decode(), str(_trav_device))
        with torch.no_grad():
            _trav_encoder(_trav_tensor_v)
        _feats_v = _bfdb_extract_patch(_trav_captured, _patch_v, _orig_Hv, _orig_Wv)
        _f96_vn  = _feats_v['block_0']
        _cos_v   = float(np.dot(_f96_vn, _trav_X_all[_trav_gi_v]))
        _verdict = '✓ OK' if _cos_v > 0.98 else '✗ ECHEC'
        print(f'  patch idx={_trav_gi_v}  img={_nm_v.decode()[:50]}')
        print(f'  cosine(base, réextrait)={_cos_v:.4f}  {_verdict}')
        if _cos_v <= 0.98:
            _trav_val_failed += 1
        _trav_val_checked += 1

    if _trav_val_checked == 0:
        raise RuntimeError("Aucun patch témoin disponible pour l'assertion.")
    if _trav_val_failed > 0:
        raise RuntimeError(
            f'Extraction invalide : {_trav_val_failed}/{_trav_val_checked} patches '
            f'ont cosine < 0.98. Vérifier modèle/checkpoint.')

    # ─────────────────────────────────────────────────────────────────────────
    # Traversal + analyse par frontière
    # ─────────────────────────────────────────────────────────────────────────
    print('\nTraversée...')
    _trav_t_vals = np.linspace(0., 1., N_STEPS)
    _trav_dt     = _trav_t_vals[1] - _trav_t_vals[0]
    _trav_results = []

    for _r in _trav_selected:
        _nm = _r['image']
        if _nm not in _trav_feat_maps:
            continue

        _feat_map        = _trav_feat_maps[_nm]
        _orig_W, _orig_H = _trav_img_sizes[_nm]
        _H_feat, _W_feat = _feat_map.shape[:2]
        _sx = _W_feat / _orig_W
        _sy = _H_feat / _orig_H

        _ia, _ib     = _r['idx_a'], _r['idx_b']
        _cat_A, _cat_B = _r['cat_a'], _r['cat_b']

        _x1a,_y1a,_x2a,_y2a = _trav_POS[_ia]
        _x1b,_y1b,_x2b,_y2b = _trav_POS[_ib]
        _cxA = (_x1a + _x2a) / 2.0;  _cyA = (_y1a + _y2a) / 2.0
        _cxB = (_x1b + _x2b) / 2.0;  _cyB = (_y1b + _y2b) / 2.0
        # Demi-taille de fenêtre = région d'un patch projeté (cohérent avec build_feature_database.py)
        _half_fx = max(1, int((_x2a - _x1a) * _sx / 2))
        _half_fy = max(1, int((_y2a - _y1a) * _sy / 2))

        # Segment de référence dans l'espace PCA-10d L2-norm
        _vA = _trav_l2(_trav_pca.transform(_trav_X_all[_ia:_ia+1])[0])
        _vB = _trav_l2(_trav_pca.transform(_trav_X_all[_ib:_ib+1])[0])
        _seg_dir = _trav_l2(_vB - _vA)

        _centA = _trav_centroids[_cat_A]
        _centB = _trav_centroids[_cat_B]

        _sim_A    = np.full(N_STEPS, np.nan)
        _sim_B    = np.full(N_STEPS, np.nan)
        _dist_seg = np.full(N_STEPS, np.nan)
        _gmm_pred = np.full(N_STEPS, -2, dtype=int)   # -2 = pas de données
        _feats_10 = [np.zeros(PCA_DIM)] * N_STEPS     # pour Figure 2

        for _ti, _t in enumerate(_trav_t_vals):
            _px = _cxA + _t * (_cxB - _cxA)
            _py = _cyA + _t * (_cyB - _cyA)
            # Mapping direct orig→feature map (int=floor, idem build_feature_database.py)
            _fx_f = _px * _sx
            _fy_f = _py * _sy
            _fx1t = max(0, int(_fx_f - _half_fx))
            _fx2t = min(_W_feat, max(_fx1t + 1, int(_fx_f + _half_fx)))
            _fy1t = max(0, int(_fy_f - _half_fy))
            _fy2t = min(_H_feat, max(_fy1t + 1, int(_fy_f + _half_fy)))
            _win = _feat_map[_fy1t:_fy2t, _fx1t:_fx2t, :]
            if _win.size == 0:
                continue
            _f96 = _win.reshape(-1, _win.shape[-1]).mean(axis=0)
            # L2-norm avant PCA (la base HDF5 stocke des vecteurs déjà L2-normés)
            _f96_n = _trav_l2(_f96)
            _f10 = _trav_l2(_trav_pca.transform(_f96_n.reshape(1,-1))[0])

            _sim_A[_ti]    = np.dot(_f10, _centA)
            _sim_B[_ti]    = np.dot(_f10, _centB)
            _diff          = _f10 - _vA
            _dist_seg[_ti] = np.linalg.norm(_diff - np.dot(_diff, _seg_dir)*_seg_dir)

            _lps   = {_c: _trav_gmms[_c].score_samples([_f10])[0] for _c in _trav_gmms}
            _cstar = max(_lps, key=_lps.get)
            _gmm_pred[_ti] = _cstar if _lps[_cstar] >= _trav_thresholds[_cstar] else -1

            _feats_10[_ti] = _f10

        _trav_results.append({
            'record': _r,
            'cat_A': _cat_A, 'cat_B': _cat_B,
            'sim_A': _sim_A, 'sim_B': _sim_B,
            'dist_seg': _dist_seg, 'gmm_pred': _gmm_pred,
            'feats_10': _feats_10,
            'cxA': _cxA, 'cyA': _cyA, 'cxB': _cxB, 'cyB': _cyB,
        })
        print(f'  {CATEGORIES[_cat_A]:<14} ↔ {CATEGORIES[_cat_B]:<14}  OK')

    _N_RES = len(_trav_results)
    if _N_RES == 0:
        print('Aucune frontière analysée — vérifier IMG_DIR.')
        raise SystemExit(0)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1 — courbes sim_A/sim_B + dist_seg par frontière
    # ─────────────────────────────────────────────────────────────────────────
    print('\nFigure 1...')
    _trav_fig1, _trav_ax1 = plt.subplots(
        2, _N_RES, figsize=(4.0 * _N_RES, 7.0),
        gridspec_kw={'height_ratios': [3, 1]},
    )
    if _N_RES == 1:
        _trav_ax1 = _trav_ax1.reshape(2, 1)

    for _bi, _res in enumerate(_trav_results):
        _axS = _trav_ax1[0, _bi]
        _axD = _trav_ax1[1, _bi]
        _cA, _cB = _res['cat_A'], _res['cat_B']
        _colA, _colB = CAT_COLORS.get(_cA,'#333'), CAT_COLORS.get(_cB,'#888')

        # courbes sim
        _axS.plot(_trav_t_vals, _res['sim_A'], color=_colA, lw=2,
                   label=f'sim_A  ({CATEGORIES[_cA]})')
        _axS.plot(_trav_t_vals, _res['sim_B'], color=_colB, lw=2, ls='--',
                   label=f'sim_B  ({CATEGORIES[_cB]})')
        _axS.axvline(0.5, color='k', lw=0.8, ls=':', alpha=0.5)
        _axS.set_xlim(0, 1)
        _axS.set_ylim(-0.15, 1.05)
        _axS.set_xticks([0, 0.5, 1])
        _axS.set_title(f'{CATEGORIES[_cA]}\n↔ {CATEGORIES[_cB]}', fontsize=9, pad=3)
        _axS.legend(fontsize=7, loc='upper right')
        if _bi == 0:
            _axS.set_ylabel('Similarité cosine', fontsize=8)

        # bande GMM (couleur catégorie ou gris=inconnu)
        for _ti in range(N_STEPS):
            _pred = _res['gmm_pred'][_ti]
            if _pred == -2:
                continue
            _cg = CAT_COLORS.get(_pred, '#303030') if _pred >= 0 else '#303030'
            _t0 = max(0., _trav_t_vals[_ti] - _trav_dt/2)
            _t1 = min(1., _trav_t_vals[_ti] + _trav_dt/2)
            _axS.axvspan(_t0, _t1, ymin=0, ymax=0.07, alpha=0.7, color=_cg, zorder=1)

        # courbe distance
        _axD.plot(_trav_t_vals, _res['dist_seg'], color='#555', lw=1.5)
        _axD.fill_between(_trav_t_vals, 0, np.nan_to_num(_res['dist_seg']),
                           alpha=0.25, color='#555')
        _axD.axvline(0.5, color='k', lw=0.8, ls=':', alpha=0.5)
        _axD.set_xlim(0, 1)
        _axD.set_xlabel('t (A→B)', fontsize=8)
        if _bi == 0:
            _axD.set_ylabel('dist⊥ seg', fontsize=8)
        _axD.set_xticks([0, 0.5, 1])

    _trav_fig1.suptitle(
        'Traversée des frontières texturales — block_0 TextureSAM\n'
        'Bande colorée = décision GMM  (couleur=catégorie, gris foncé=inconnu)',
        fontsize=10,
    )
    _trav_fig1.tight_layout(rect=[0, 0, 1, 0.94])
    _trav_fig1.savefig(OUTPUT_DIR / 'boundary_traversal_curves.png',
                        dpi=150, bbox_inches='tight')
    plt.close(_trav_fig1)
    print('  Saved: boundary_traversal_curves.png')

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2 — trajectoires dans l'espace PCA-2d
    # ─────────────────────────────────────────────────────────────────────────
    print('Figure 2...')
    _trav_pca2      = PCA(n_components=2, random_state=SEED)
    _trav_X_norm_2d = _trav_pca2.fit_transform(_trav_X_norm)

    _trav_fig2, _trav_ax2 = plt.subplots(figsize=(9, 7))

    for _c in _trav_CATS_VALID:
        _mask_c = _trav_y_valid == _c
        _pts    = _trav_X_norm_2d[_mask_c]
        _trav_ax2.scatter(_pts[:,0], _pts[:,1], c=CAT_COLORS.get(_c,'#999'),
                           s=5, alpha=0.18, label=CATEGORIES.get(_c,'?'), zorder=1)

    _trav_cmap_tr = plt.colormaps.get_cmap('tab10')
    for _bi, _res in enumerate(_trav_results):
        _pts_10 = np.vstack(_res['feats_10'])         # (N_STEPS, PCA_DIM)
        _pts_2d = _trav_pca2.transform(_pts_10)       # (N_STEPS, 2)
        _col    = _trav_cmap_tr(_bi / max(_N_RES - 1, 1))
        _lbl    = f'{CATEGORIES[_res["cat_A"]]} ↔ {CATEGORIES[_res["cat_B"]]}'
        _trav_ax2.plot(_pts_2d[:,0], _pts_2d[:,1], '-o',
                        color=_col, lw=2, ms=5, zorder=5, label=_lbl)
        _trav_ax2.annotate('A', _pts_2d[0],  fontsize=7, color=_col,
                             xytext=(3,3),  textcoords='offset points')
        _trav_ax2.annotate('B', _pts_2d[-1], fontsize=7, color=_col,
                             xytext=(3,3),  textcoords='offset points')

    _var = _trav_pca2.explained_variance_ratio_
    _trav_ax2.set_xlabel(f'PC1 ({_var[0]*100:.1f}%)', fontsize=10)
    _trav_ax2.set_ylabel(f'PC2 ({_var[1]*100:.1f}%)', fontsize=10)
    _trav_ax2.set_title(
        'Trajectoires de traversée dans l\'espace PCA-2 (features block_0 L2-norm)\n'
        'Fond = nuage de patches d\'entraînement  ·  A→B = sens de la traversée',
        fontsize=10,
    )
    _trav_ax2.legend(fontsize=7, ncol=2, loc='best', framealpha=0.85)
    _trav_fig2.tight_layout()
    _trav_fig2.savefig(OUTPUT_DIR / 'boundary_traversal_pca2.png',
                        dpi=150, bbox_inches='tight')
    plt.close(_trav_fig2)
    print('  Saved: boundary_traversal_pca2.png')

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3 — image + segment diagnostique
    # ─────────────────────────────────────────────────────────────────────────
    print('Figure 3...')
    _trav_n_diag = min(2, _N_RES)
    _trav_fig3, _trav_ax3 = plt.subplots(
        1, _trav_n_diag, figsize=(7.5 * _trav_n_diag, 5.5))
    if _trav_n_diag == 1:
        _trav_ax3 = [_trav_ax3]

    for _bi in range(_trav_n_diag):
        _res    = _trav_results[_bi]
        _nm     = _res['record']['image']
        _ax     = _trav_ax3[_bi]
        _orig_W, _orig_H = _trav_img_sizes.get(_nm, (1280, 768))

        if _nm in _trav_img_arrays:
            _ax.imshow(_trav_img_arrays[_nm])
        else:
            _ax.set_xlim(0, _orig_W)
            _ax.set_ylim(_orig_H, 0)

        # ligne du segment
        _ax.plot([_res['cxA'], _res['cxB']], [_res['cyA'], _res['cyB']],
                  '-', color='white', lw=1.5, alpha=0.65, zorder=4)

        # points de traversée colorés par décision GMM
        for _ti, _t in enumerate(_trav_t_vals):
            _px = _res['cxA'] + _t * (_res['cxB'] - _res['cxA'])
            _py = _res['cyA'] + _t * (_res['cyB'] - _res['cyA'])
            _pred = _res['gmm_pred'][_ti]
            _cpt  = CAT_COLORS.get(_pred, '#303030') if _pred >= 0 else '#303030'
            _ms   = 10 if (_ti == 0 or _ti == N_STEPS-1) else 6
            _mk   = 'D' if (_ti == 0 or _ti == N_STEPS-1) else 'o'
            _ax.plot(_px, _py, _mk, color=_cpt, ms=_ms,
                      markeredgecolor='white', markeredgewidth=0.8, zorder=6)

        # rectangles des deux patches
        for _idx_key, _cat_key in [('idx_a', 'cat_A'), ('idx_b', 'cat_B')]:
            _idx = _res['record'][_idx_key]
            _cat = _res[_cat_key]
            _col = CAT_COLORS.get(_cat, '#ffffff')
            _x1,_y1,_x2,_y2 = _trav_POS[_idx]
            _rect = plt.Rectangle((_x1, _y1), _x2-_x1, _y2-_y1,
                                    lw=2, edgecolor=_col,
                                    facecolor=_col, alpha=0.18, zorder=3)
            _ax.add_patch(_rect)
            _ax.text(_x1+3, _y1+13, CATEGORIES[_cat], fontsize=7,
                      color='white', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.1',
                                facecolor=_col, alpha=0.75), zorder=7)

        _ax.set_title(
            f'{CATEGORIES[_res["cat_A"]]} ↔ {CATEGORIES[_res["cat_B"]]}',
            fontsize=10)
        _ax.set_xlabel(_nm.decode()[:65], fontsize=7)
        _ax.axis('off')

    _trav_fig3.suptitle(
        'Images sources avec chemin de traversée\n'
        'Losanges = extrémités A/B  ·  points = positions intermédiaires  '
        '·  couleur = décision GMM',
        fontsize=9,
    )
    _trav_fig3.tight_layout()
    _trav_fig3.savefig(OUTPUT_DIR / 'boundary_traversal_diagnostic.png',
                        dpi=150, bbox_inches='tight')
    plt.close(_trav_fig3)
    print('  Saved: boundary_traversal_diagnostic.png')

    # ─────────────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────────────
    print('\n=== VERDICT ===')
    _trav_verdict = [
        'Traversée des frontières texturales — Résultats\n',
        f'N_BOUNDARIES={N_BOUNDARIES}  N_STEPS={N_STEPS}  '
        f'PCA_DIM={PCA_DIM}  REG_COVAR={REG_COVAR}  PERCENTILE={PERCENTILE}\n\n',
    ]

    _trav_n_direct  = 0
    _trav_n_cross   = 0

    for _bi, _res in enumerate(_trav_results):
        _cA, _cB      = _res['cat_A'], _res['cat_B']
        _valid_mask   = ~np.isnan(_res['sim_A'])
        _n_valid      = int(_valid_mask.sum())
        _n_unk        = int(np.sum(_res['gmm_pred'] == -1))
        _max_dist     = float(np.nanmax(_res['dist_seg']))
        _mean_dist    = float(np.nanmean(_res['dist_seg']))

        # point de croisement sim_A = sim_B
        _diff_valid   = (_res['sim_A'] - _res['sim_B'])[_valid_mask]
        _t_valid      = _trav_t_vals[_valid_mask]
        _sign_chg     = np.where(np.diff(np.sign(_diff_valid)))[0]
        _crossover    = float(_t_valid[_sign_chg[0]]) if len(_sign_chg) > 0 else None

        _is_direct  = _max_dist < 0.5
        _is_clean   = _crossover is not None and 0.3 <= _crossover <= 0.7
        if _is_direct:  _trav_n_direct += 1
        if _crossover is not None:  _trav_n_cross += 1

        _cross_str  = f't={_crossover:.2f}' if _crossover is not None else 'aucun'
        _traj_str   = 'interpolation directe' if _is_direct else 'DÉTOUR (zone intermédiaire ?)'
        _trans_str  = 'transition NETTE au centre' if _is_clean else 'transition décalée ou absente'

        _line = (
            f'Frontière {_bi+1} : {CATEGORIES[_cA]} ↔ {CATEGORIES[_cB]}\n'
            f'  dist_seg   max={_max_dist:.3f}  mean={_mean_dist:.3f}'
            f'   → {_traj_str}\n'
            f'  GMM inconnus : {_n_unk}/{_n_valid} positions'
            f'   croisement sim_A/sim_B : {_cross_str}\n'
            f'  → {_trans_str}\n'
        )
        print(_line)
        _trav_verdict.append(_line)

    _trav_summary = (
        f'--- Résumé global ({_N_RES} frontières) ---\n'
        f'Trajectoires directes (max dist_seg < 0.5)  : {_trav_n_direct}/{_N_RES}\n'
        f'Frontières avec croisement sim_A/sim_B      : {_trav_n_cross}/{_N_RES}\n'
        f'Interprétation : block_0 '
        + ('interpole proprement les transitions texturales'
           if _trav_n_direct == _N_RES else
           'montre des détours à certaines frontières — possible "3e texture" intermédiaire')
        + '\n'
    )
    print(_trav_summary)
    _trav_verdict.append(_trav_summary)

    with open(OUTPUT_DIR / 'boundary_traversal_verdict.txt', 'w') as _vf:
        _vf.writelines(_trav_verdict)
    print('  Saved: boundary_traversal_verdict.txt')

finally:
    _bfdb_remove_hooks(_trav_handles)
    print('\nHooks retirés.')
