#!/usr/bin/env python3
"""
debug_boundary_extraction.py
Diagnostique la cause du bug "GMM inconnu sur toutes les positions" dans
boundary_traversal_meb.py :  tous les 11 pas classés "inconnu", dist_seg > 1.0.
"""
import os, sys, tempfile, zipfile
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
ROOT           = Path('/home/aidouni/meb_texture_seg')
DB_PATH        = ROOT / 'data/feature_database/database_meb.h5'
IMG_DIR        = ROOT / 'Image_Ouassim'
CHECKPOINT     = 'checkpoints/sam2.1_hiera_small_1.pt'
SAM2_DIR       = ROOT / 'TextureSAM' / 'sam2'
OUTPUT_DIR     = ROOT / 'outputs' / 'boundary_analysis' / 'debug'

N_TEST_PATCHES = 5
PCA_DIM        = 10
REG_COVAR      = 1e-1
PERCENTILE     = 5
CATS_EXCLUDE   = [2, 8, 10, 11, 12, 13]
CATEGORIES     = {1:'Homogène', 3:'Faisceaux', 4:'Filaments',
                  5:'Strat.rect.', 6:'Strat.sin.', 7:'Granuleux', 9:'Trou'}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lecture HDF5
# ─────────────────────────────────────────────────────────────────────────────
print('Lecture HDF5...')
with h5py.File(DB_PATH, 'r') as _h5:
    _dbg_NAMES = _h5['metadata/image_names'][:]
    _dbg_CATS  = _h5['metadata/category_ids'][:].astype(int)
    _dbg_POS   = _h5['metadata/positions'][:].astype(int)
    _dbg_X_all = _h5['features/block_0'][:].astype(np.float32)

_dbg_mask_valid = ~np.isin(_dbg_CATS, CATS_EXCLUDE)
_dbg_X_valid    = _dbg_X_all[_dbg_mask_valid]
_dbg_y_valid    = _dbg_CATS[_dbg_mask_valid]
_dbg_idx_valid  = np.where(_dbg_mask_valid)[0]
_dbg_CATS_VALID = sorted(set(_dbg_y_valid.tolist()))
print(f'  {len(_dbg_X_valid)} patches valides')

# Un patch par catégorie (image disponible)
_dbg_test_patches = []
for _c in _dbg_CATS_VALID:
    for _gi in _dbg_idx_valid[_dbg_y_valid == _c]:
        if (IMG_DIR / _dbg_NAMES[_gi].decode()).exists():
            _dbg_test_patches.append(int(_gi))
            break
    if len(_dbg_test_patches) >= N_TEST_PATCHES:
        break

print(f'  Patches test : {_dbg_test_patches}')

# ─────────────────────────────────────────────────────────────────────────────
# PCA + GMM — identiques à boundary_traversal_meb.py
# ─────────────────────────────────────────────────────────────────────────────
print('PCA + GMM...')
_dbg_pca    = PCA(n_components=PCA_DIM, random_state=42)
_dbg_X_pca  = _dbg_pca.fit_transform(_dbg_X_valid)
_dbg_X_norm = normalize(_dbg_X_pca, norm='l2')

_dbg_gmms       = {}
_dbg_thresholds = {}
_dbg_centroids  = {}

for _c in _dbg_CATS_VALID:
    _mc = _dbg_y_valid == _c
    _fc = _dbg_X_norm[_mc]
    _cen = _fc.mean(axis=0)
    _dbg_centroids[_c] = _cen / (np.linalg.norm(_cen) + 1e-8)
    _gmm = GaussianMixture(n_components=1, covariance_type='full',
                            reg_covar=REG_COVAR, random_state=42)
    _gmm.fit(_fc)
    _dbg_gmms[_c]       = _gmm
    _dbg_thresholds[_c] = float(np.percentile(_gmm.score_samples(_fc), PERCENTILE))

# ─────────────────────────────────────────────────────────────────────────────
# Chargement modèle
# ─────────────────────────────────────────────────────────────────────────────
print('Chargement modèle...')
sys.path.insert(0, str(SAM2_DIR / 'sam2'))
os.chdir(ROOT)
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir=str(SAM2_DIR / 'sam2' / 'configs'),
    version_base='1.2',
)
from sam2.build_sam import build_sam2

_dbg_device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_dbg_IMG_SIZE = 1024
_dbg_MEAN     = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_dbg_STD      = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _dbg_load_ckpt(ckpt_path):
    _p = Path(ckpt_path)
    if _p.is_file():
        _sd = torch.load(_p, map_location='cpu', weights_only=False)
        return _sd.get('model', _sd)
    _arch = _p / 'archive' if (_p / 'archive').is_dir() else _p
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as _tmp:
        _tmp_path = _tmp.name
    with zipfile.ZipFile(_tmp_path, 'w', compression=zipfile.ZIP_STORED) as _zf:
        for _fp in _arch.rglob('*'):
            if _fp.is_file():
                _info = zipfile.ZipInfo(str(_fp.relative_to(_arch.parent)))
                _info.date_time = (1980, 1, 1, 0, 0, 0)
                with open(_fp, 'rb') as _fh:
                    _zf.writestr(_info, _fh.read())
    _sd = torch.load(_tmp_path, map_location='cpu', weights_only=False)
    os.unlink(_tmp_path)
    return _sd.get('model', _sd)


_dbg_cap  = {}
_dbg_KEY  = 'block0'
_dbg_base = str(ROOT / 'checkpoints' / 'sam2.1_hiera_small')
_dbg_model = build_sam2('sam2.1/sam2.1_hiera_s.yaml', ckpt_path=None,
                          device=_dbg_device, apply_postprocessing=False)
_dbg_model.load_state_dict(_dbg_load_ckpt(_dbg_base), strict=False)
_dbg_ck = str(ROOT / CHECKPOINT)
if Path(_dbg_ck).resolve() != Path(_dbg_base).resolve():
    _m, _u = _dbg_model.load_state_dict(_dbg_load_ckpt(_dbg_ck), strict=False)
    print(f'  Fine-tuned : missing={len(_m)} unexpected={len(_u)}')
_dbg_model.eval()

_dbg_hook = _dbg_model.image_encoder.trunk.blocks[0].register_forward_hook(
    lambda m, i, o: _dbg_cap.update({_dbg_KEY: o.detach()})
)


def _dbg_preprocess(img_pil):
    _img = img_pil.convert('RGB').resize((_dbg_IMG_SIZE, _dbg_IMG_SIZE), Image.BILINEAR)
    _x = torch.from_numpy(np.array(_img)).float() / 255.
    _x = _x.permute(2, 0, 1)
    _x = (_x - _dbg_MEAN) / _dbg_STD
    return _x.unsqueeze(0).to(_dbg_device)


def _dbg_get_feat_map(img_pil):
    _dbg_cap.clear()
    with torch.no_grad():
        _dbg_model.image_encoder(_dbg_preprocess(img_pil))
    return _dbg_cap[_dbg_KEY][0].cpu().numpy()


def _dbg_cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.

def _dbg_l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


try:
    # ── Charger les images une fois
    _dbg_feat_maps  = {}
    _dbg_img_sizes  = {}
    _dbg_img_arrays = {}

    for _gi in _dbg_test_patches:
        _nm = _dbg_NAMES[_gi]
        if _nm not in _dbg_feat_maps:
            _img = Image.open(IMG_DIR / _nm.decode()).convert('RGB')
            _dbg_img_sizes[_nm]  = _img.size          # (W, H)
            _dbg_img_arrays[_nm] = np.array(_img)
            _dbg_feat_maps[_nm]  = _dbg_get_feat_map(_img)
            _Hf, _Wf, _Cf = _dbg_feat_maps[_nm].shape
            print(f'  {_nm.decode()[:60]}  feat_map={_Hf}×{_Wf}×{_Cf}')

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1 — Re-extraire le vecteur d'un patch et comparer à la base HDF5
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '='*68)
    print('TEST 1 — Reproduction de l\'extraction vs vecteurs stockés dans la base')
    print('='*68)
    print('Si cosine ≈ 1.0 → extraction correcte. Si cosine << 1.0 → mapping faux.\n')

    _dbg_res1 = []   # (gi, cos_3x3, cos_full, cos_single)

    for _gi in _dbg_test_patches:
        _nm       = _dbg_NAMES[_gi]
        _cat      = int(_dbg_CATS[_gi])
        _x1,_y1,_x2,_y2 = _dbg_POS[_gi]
        _feat_ref = _dbg_X_all[_gi]
        _fm       = _dbg_feat_maps[_nm]
        _orig_W, _orig_H = _dbg_img_sizes[_nm]
        _Hf, _Wf  = _fm.shape[:2]

        _sx = _Wf / _orig_W     # = 256/1280 = 0.2
        _sy = _Hf / _orig_H     # = 256/768 ≈ 0.333

        # Centre du patch → position feature map
        _cx = (_x1 + _x2) / 2.;  _cy = (_y1 + _y2) / 2.
        _fx = int(round(_cx * _sx));  _fy = int(round(_cy * _sy))

        # Région complète du patch dans la feature map
        _fx1 = max(0, int(round(_x1 * _sx)));  _fx2 = min(_Wf, int(round(_x2 * _sx)))
        _fy1 = max(0, int(round(_y1 * _sy)));  _fy2 = min(_Hf, int(round(_y2 * _sy)))
        if _fx2 <= _fx1: _fx2 = _fx1 + 1
        if _fy2 <= _fy1: _fy2 = _fy1 + 1

        # Extraction 1 × 1 (single cell)
        _f_single = _fm[_fy, _fx, :]

        # Extraction 3 × 3 (comme boundary_traversal_meb.py)
        _win3 = _fm[max(0,_fy-1):min(_Hf,_fy+2), max(0,_fx-1):min(_Wf,_fx+2), :]
        _f_3x3 = _win3.reshape(-1, _win3.shape[-1]).mean(axis=0)

        # Extraction pleine région du patch
        _win_f = _fm[_fy1:_fy2, _fx1:_fx2, :]
        _f_full = _win_f.reshape(-1, _win_f.shape[-1]).mean(axis=0)

        _c_single = _dbg_cos(_feat_ref, _f_single)
        _c_3x3    = _dbg_cos(_feat_ref, _f_3x3)
        _c_full   = _dbg_cos(_feat_ref, _f_full)
        _dbg_res1.append((_gi, _c_single, _c_3x3, _c_full))

        _v3  = '✓' if _c_3x3 > 0.9 else ('~' if _c_3x3 > 0.7 else '✗')
        _vf  = '✓' if _c_full > 0.9 else ('~' if _c_full > 0.7 else '✗')

        print(f'[{_gi}] {CATEGORIES[_cat]:<14}  img={_nm.decode()[:40]}')
        print(f'      pos=[{_x1},{_y1},{_x2},{_y2}]  '
              f'fm_region=cols[{_fx1}:{_fx2}] rows[{_fy1}:{_fy2}]  '
              f'({_fx2-_fx1}×{_fy2-_fy1} cells)')
        print(f'      sx={_sx:.4f}  sy={_sy:.4f}  '
              f'centre→feat({_fx},{_fy})')
        print(f'      cosine(base, 1×1)    = {_c_single:.4f}')
        print(f'      cosine(base, 3×3)    = {_c_3x3:.4f}  {_v3}  ← méthode traversal')
        print(f'      cosine(base, full)   = {_c_full:.4f}  {_vf}')
        print()

    _all_3x3 = [r[2] for r in _dbg_res1]
    _all_full = [r[3] for r in _dbg_res1]
    print(f'Résumé cosines  3×3  : {np.mean(_all_3x3):.3f} ± {np.std(_all_3x3):.3f}'
          f'  [min={min(_all_3x3):.3f}  max={max(_all_3x3):.3f}]')
    print(f'Résumé cosines full  : {np.mean(_all_full):.3f} ± {np.std(_all_full):.3f}'
          f'  [min={min(_all_full):.3f}  max={max(_all_full):.3f}]')

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2 — Trace détaillée du mapping pour 1 patch
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '='*68)
    print('TEST 2 — Trace du mapping image→feature map (patch #0)')
    print('='*68)

    _gi0 = _dbg_test_patches[0]
    _nm0 = _dbg_NAMES[_gi0]
    _cat0 = int(_dbg_CATS[_gi0])
    _x1,_y1,_x2,_y2 = _dbg_POS[_gi0]
    _orig_W0, _orig_H0 = _dbg_img_sizes[_nm0]
    _fm0 = _dbg_feat_maps[_nm0]
    _Hf0, _Wf0 = _fm0.shape[:2]

    print(f'  Patch  : [{_x1},{_y1},{_x2},{_y2}]  {CATEGORIES[_cat0]}')
    print(f'  Image orig : {_orig_W0}×{_orig_H0}')
    print(f'  Feature map : {_Hf0}×{_Wf0}×{_fm0.shape[2]}')
    print()
    print(f'  Scale X : {_Wf0}/{_orig_W0} = {_Wf0/_orig_W0:.6f}  (≠ Y)')
    print(f'  Scale Y : {_Hf0}/{_orig_H0} = {_Hf0/_orig_H0:.6f}')
    print()
    _sx0 = _Wf0/_orig_W0; _sy0 = _Hf0/_orig_H0
    _cx0, _cy0 = (_x1+_x2)/2., (_y1+_y2)/2.
    _fx0, _fy0 = int(round(_cx0*_sx0)), int(round(_cy0*_sy0))
    print(f'  x : [{_x1}→{_x1*_sx0:.2f}  {_x2}→{_x2*_sx0:.2f}]  '
          f'cols [{int(round(_x1*_sx0))}:{int(round(_x2*_sx0))}]')
    print(f'  y : [{_y1}→{_y1*_sy0:.2f}  {_y2}→{_y2*_sy0:.2f}]  '
          f'rows [{int(round(_y1*_sy0))}:{int(round(_y2*_sy0))}]')
    print(f'  Centre image ({_cx0:.1f},{_cy0:.1f}) → feature map ({_fx0},{_fy0})')
    print()

    # Alternative : scale uniforme 256/1024 = 0.25 (bug fréquent)
    _sx_u = 256/1024; _sy_u = 256/1024
    _fx_u = int(round(_cx0*_sx_u)); _fy_u = int(round(_cy0*_sy_u))
    print(f'  [ALTERNATIVE scale uniforme 0.25] centre → ({_fx_u},{_fy_u})')
    _win_u = _fm0[max(0,_fy_u-1):min(_Hf0,_fy_u+2),
                   max(0,_fx_u-1):min(_Wf0,_fx_u+2), :]
    _f_u   = _win_u.reshape(-1, _win_u.shape[-1]).mean(axis=0)
    print(f'  cosine(base, 3×3 @ scale_unif) = {_dbg_cos(_dbg_X_all[_gi0], _f_u):.4f}')

    # Alternative : scale 1/4 sans tenir compte du resize (direct 1024→256)
    # i.e. supposer que l'image est déjà 1024×1024 en coordonnées
    _sx_r = 256/_orig_W0 / (1024/_orig_W0)  # = 256/1024 = 0.25
    print(f'  [ALTERNATIVE scale 1/4 direct]  identique → ({_fx_u},{_fy_u})')
    print()
    print(f'  → Delta position due au choix de scale :')
    print(f'     correct  ({_fx0},{_fy0}) vs uniforme ({_fx_u},{_fy_u})')
    print(f'     différence = ({abs(_fx0-_fx_u)}, {abs(_fy0-_fy_u)}) cells')

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3 — Taille de fenêtre optimale
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '='*68)
    print('TEST 3 — Quelle taille de fenêtre reproduit le mieux la base ?')
    print('='*68)
    print(f'{"patch":<6}  {"cat":<14}  '
          f'{"1×1":>6}  {"3×3":>6}  {"5×5":>6}  {"7×7":>6}  {"9×9":>6}  {"full":>6}')

    _best_counts = {}
    for _gi in _dbg_test_patches:
        _nm  = _dbg_NAMES[_gi]
        _cat = int(_dbg_CATS[_gi])
        _x1,_y1,_x2,_y2 = _dbg_POS[_gi]
        _fr  = _dbg_X_all[_gi]
        _fm  = _dbg_feat_maps[_nm]
        _oW, _oH = _dbg_img_sizes[_nm]
        _Hf, _Wf = _fm.shape[:2]
        _sx, _sy = _Wf/_oW, _Hf/_oH
        _cx, _cy = (_x1+_x2)/2., (_y1+_y2)/2.
        _fx, _fy = int(round(_cx*_sx)), int(round(_cy*_sy))
        _fx1r = max(0, int(round(_x1*_sx)));  _fx2r = min(_Wf, int(round(_x2*_sx)))
        _fy1r = max(0, int(round(_y1*_sy)));  _fy2r = min(_Hf, int(round(_y2*_sy)))
        if _fx2r <= _fx1r: _fx2r = _fx1r+1
        if _fy2r <= _fy1r: _fy2r = _fy1r+1

        _scores = {}
        for _r in [0, 1, 2, 3, 4]:  # radii → windows 1×1 … 9×9
            _w = _fm[max(0,_fy-_r):min(_Hf,_fy+_r+1),
                      max(0,_fx-_r):min(_Wf,_fx+_r+1), :]
            _f = _w.reshape(-1, _w.shape[-1]).mean(axis=0)
            _scores[f'{2*_r+1}×{2*_r+1}'] = _dbg_cos(_fr, _f)
        _wf = _fm[_fy1r:_fy2r, _fx1r:_fx2r, :]
        _scores['full'] = _dbg_cos(_fr, _wf.reshape(-1, _wf.shape[-1]).mean(axis=0))

        _best = max(_scores, key=_scores.get)
        _best_counts[_best] = _best_counts.get(_best, 0) + 1

        _row = f'{_gi:<6}  {CATEGORIES[_cat]:<14}'
        for _k in ['1×1','3×3','5×5','7×7','9×9','full']:
            _v = _scores[_k]
            _mark = '*' if _k == _best else ' '
            _row += f'  {_v:.3f}{_mark}'
        print(_row)

    print(f'\n  Meilleure fenêtre par fréquence : {_best_counts}')

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4 — Vérifier la chaîne PCA/GMM sur vecteurs de la base
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '='*68)
    print('TEST 4 — Chaîne PCA/GMM sur vecteurs stockés dans la base HDF5')
    print('='*68)
    print('Attendu : tous reconnus (≥ seuil PERCENTILE=5 de leur propre distribution)\n')

    _dbg_t4_ok = 0
    for _gi in _dbg_test_patches:
        _cat  = int(_dbg_CATS[_gi])
        _fref = _dbg_X_all[_gi]
        _f10  = _dbg_l2(_dbg_pca.transform(_fref.reshape(1,-1))[0])
        _sim  = float(np.dot(_f10, _dbg_centroids[_cat]))
        _lps  = {_c: _dbg_gmms[_c].score_samples([_f10])[0] for _c in _dbg_gmms}
        _cstar = max(_lps, key=_lps.get)
        _recog = _lps[_cstar] >= _dbg_thresholds[_cstar]
        _ok_lbl = '✓ reconnu' if _recog else '✗ INCONNU'
        _cname  = CATEGORIES[_cstar] if _cstar != _cat else CATEGORIES[_cat]
        _match  = '(CORRECT)' if _cstar == _cat else f'→ prédit {_cname} !'
        if _recog: _dbg_t4_ok += 1
        print(f'  [{_gi}] {CATEGORIES[_cat]:<14}  '
              f'logp={_lps[_cstar]:.2f}  seuil={_dbg_thresholds[_cstar]:.2f}  '
              f'{_ok_lbl}  {_match}  sim_centroïde={_sim:.3f}')

    print(f'\n  {_dbg_t4_ok}/{len(_dbg_test_patches)} patches de la base reconnus par le GMM')

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4b — Même chose MAIS avec les features réextraites (3×3 et full)
    # ─────────────────────────────────────────────────────────────────────────
    print('\n  TEST 4b — GMM sur features réextraites')
    print(f'  {"patch":<6}  {"cat":<14}  '
          f'{"3×3→reconnu?":>14}  {"full→reconnu?":>14}')

    _dbg_t4b_3x3 = 0; _dbg_t4b_full = 0
    for _gi in _dbg_test_patches:
        _nm   = _dbg_NAMES[_gi]
        _cat  = int(_dbg_CATS[_gi])
        _x1,_y1,_x2,_y2 = _dbg_POS[_gi]
        _fm   = _dbg_feat_maps[_nm]
        _oW, _oH = _dbg_img_sizes[_nm]
        _Hf, _Wf = _fm.shape[:2]
        _sx, _sy = _Wf/_oW, _Hf/_oH
        _cx, _cy = (_x1+_x2)/2., (_y1+_y2)/2.
        _fxc, _fyc = int(round(_cx*_sx)), int(round(_cy*_sy))
        _fx1r = max(0,int(round(_x1*_sx))); _fx2r = min(_Wf,int(round(_x2*_sx)))
        _fy1r = max(0,int(round(_y1*_sy))); _fy2r = min(_Hf,int(round(_y2*_sy)))
        if _fx2r<=_fx1r: _fx2r=_fx1r+1
        if _fy2r<=_fy1r: _fy2r=_fy1r+1

        _w3  = _fm[max(0,_fyc-1):min(_Hf,_fyc+2), max(0,_fxc-1):min(_Wf,_fxc+2), :]
        _f3  = _dbg_l2(_dbg_pca.transform(
                   _w3.reshape(-1,_w3.shape[-1]).mean(axis=0).reshape(1,-1))[0])
        _wfl = _fm[_fy1r:_fy2r, _fx1r:_fx2r, :]
        _ffl = _dbg_l2(_dbg_pca.transform(
                   _wfl.reshape(-1,_wfl.shape[-1]).mean(axis=0).reshape(1,-1))[0])

        def _gmm_recog(v):
            lps = {c: _dbg_gmms[c].score_samples([v])[0] for c in _dbg_gmms}
            cs  = max(lps, key=lps.get)
            return lps[cs] >= _dbg_thresholds[cs], cs, lps[cs], _dbg_thresholds[cs]

        _r3, _c3, _lp3, _th3 = _gmm_recog(_f3)
        _rf, _cf, _lpf, _thf = _gmm_recog(_ffl)
        if _r3: _dbg_t4b_3x3 += 1
        if _rf: _dbg_t4b_full += 1

        _s3  = f'{"✓" if _r3 else "✗"}  logp={_lp3:.1f}≥{_th3:.1f}={_r3}'
        _sf  = f'{"✓" if _rf else "✗"}  logp={_lpf:.1f}≥{_thf:.1f}={_rf}'
        print(f'  [{_gi}] {CATEGORIES[_cat]:<14}  {_s3:>20}  {_sf:>20}')

    print(f'\n  Reconnus (3×3)   : {_dbg_t4b_3x3}/{len(_dbg_test_patches)}')
    print(f'  Reconnus (full)  : {_dbg_t4b_full}/{len(_dbg_test_patches)}')

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 5 — Figure diagnostique : mapping visuel
    # ─────────────────────────────────────────────────────────────────────────
    print('\nFigure diagnostique...')
    _n_diag = min(2, len(_dbg_test_patches))
    _fig5, _axes5 = plt.subplots(2, _n_diag, figsize=(7*_n_diag, 9))
    if _n_diag == 1:
        _axes5 = np.array([[_axes5[0]], [_axes5[1]]])

    for _bi in range(_n_diag):
        _gi   = _dbg_test_patches[_bi]
        _nm   = _dbg_NAMES[_gi]
        _cat  = int(_dbg_CATS[_gi])
        _x1,_y1,_x2,_y2 = _dbg_POS[_gi]
        _fm   = _dbg_feat_maps[_nm]
        _oW, _oH = _dbg_img_sizes[_nm]
        _Hf, _Wf = _fm.shape[:2]
        _sx, _sy = _Wf/_oW, _Hf/_oH

        _ax_img = _axes5[0, _bi]
        _ax_fm  = _axes5[1, _bi]

        # Image originale + rectangle du patch
        _ax_img.imshow(_dbg_img_arrays[_nm])
        _ax_img.add_patch(plt.Rectangle((_x1,_y1), _x2-_x1, _y2-_y1,
                                          lw=2, edgecolor='#FF4444', facecolor='none'))
        _cx, _cy = (_x1+_x2)/2., (_y1+_y2)/2.
        _ax_img.plot(_cx, _cy, 'y*', ms=12, zorder=5)
        _ax_img.set_title(f'{CATEGORIES[_cat]} [{_x1},{_y1},{_x2},{_y2}]\n'
                           f'img {_oW}×{_oH}', fontsize=8)
        _ax_img.axis('off')

        # Feature map (canal moyen) + région mappée
        _ax_fm.imshow(_fm.mean(axis=-1), cmap='viridis', origin='upper')

        # Région complète du patch (rouge)
        _fx1r = int(round(_x1*_sx)); _fx2r = int(round(_x2*_sx))
        _fy1r = int(round(_y1*_sy)); _fy2r = int(round(_y2*_sy))
        _ax_fm.add_patch(plt.Rectangle((_fx1r, _fy1r), max(1,_fx2r-_fx1r), max(1,_fy2r-_fy1r),
                                         lw=2, edgecolor='#FF4444', facecolor='none',
                                         label='patch region'))
        # Fenêtre 3×3 traversal (jaune)
        _fxc = int(round(_cx*_sx)); _fyc = int(round(_cy*_sy))
        _ax_fm.add_patch(plt.Rectangle((_fxc-1, _fyc-1), 3, 3,
                                         lw=1.5, edgecolor='yellow', facecolor='none',
                                         label='3×3 traversal'))
        _ax_fm.plot(_fxc, _fyc, 'y*', ms=8)

        _c3x3 = _dbg_cos(_dbg_X_all[_gi],
                          _fm[max(0,_fyc-1):min(_Hf,_fyc+2),
                               max(0,_fxc-1):min(_Wf,_fxc+2), :
                              ].reshape(-1,_fm.shape[-1]).mean(axis=0))
        _cfull_loc = _dbg_cos(_dbg_X_all[_gi],
                               _fm[_fy1r:max(_fy1r+1,_fy2r),
                                    _fx1r:max(_fx1r+1,_fx2r), :
                                  ].reshape(-1,_fm.shape[-1]).mean(axis=0))

        _ax_fm.set_title(f'Feature map (mean all C)\n'
                          f'Red=full patch  Yellow=3×3 traversal\n'
                          f'cos(3×3)={_c3x3:.3f}  cos(full)={_cfull_loc:.3f}', fontsize=7)
        _ax_fm.legend(fontsize=6, loc='upper right')

    _fig5.suptitle('TEST 5 — Vérification visuelle du mapping image→feature map', fontsize=10)
    _fig5.tight_layout()
    _fig5.savefig(OUTPUT_DIR / 'debug_patch_mapping.png', dpi=150, bbox_inches='tight')
    plt.close(_fig5)
    print(f'  Saved: {OUTPUT_DIR}/debug_patch_mapping.png')

    # ─────────────────────────────────────────────────────────────────────────
    # CONCLUSION
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '='*68)
    print('CONCLUSION')
    print('='*68)

    _mean3  = float(np.mean(_all_3x3))
    _meanf  = float(np.mean(_all_full))
    _t4_ok  = _dbg_t4_ok == len(_dbg_test_patches)
    _t4b3   = _dbg_t4b_3x3
    _t4bf   = _dbg_t4b_full
    _n      = len(_dbg_test_patches)

    print(f'\n  TEST 1 — cosine(base, 3×3)  = {_mean3:.3f}  |  '
          f'cosine(base, full) = {_meanf:.3f}')
    print(f'  TEST 4 — GMM reconnaît vecteurs base : {_dbg_t4_ok}/{_n}')
    print(f'  TEST 4b— GMM reconnaît 3×3 réextrait : {_t4b3}/{_n}  |  '
          f'full réextrait : {_t4bf}/{_n}')
    print()

    if not _t4_ok:
        print('→ BUG DANS LA CHAÎNE PCA/GMM')
        print('  Les vecteurs de la base elle-même ne sont pas reconnus.')
        print('  Vérifier : PCA refit depuis 0 dans le script, seuil différent,')
        print('             scaler absent vs présent.')
    elif _mean3 < 0.5 and _meanf < 0.5:
        print('→ BUG DANS LE MAPPING DE COORDONNÉES')
        print('  Ni 3×3 ni la pleine région ne reproduisent les vecteurs de la base.')
        print('  Cause probable : inversion x/y, format du tensor (C,H,W) vs (H,W,C),')
        print('  ou le modèle utilisé pour la base est différent de celui ici.')
    elif _meanf > _mean3 + 0.15 and _meanf > 0.7:
        print('→ BUG DE TAILLE DE FENÊTRE')
        print(f'  La pleine région ({_meanf:.3f}) reproduit bien la base,')
        print(f'  mais la fenêtre 3×3 ({_mean3:.3f}) ne la reproduit pas.')
        print('  La traversée doit extraire sur la pleine région du patch (ou une')
        print('  fenêtre proportionnelle au patch_size/4), pas une fenêtre 3×3.')
        print(f'  Reconnus (3×3)={_t4b3}/{_n}  vs  (full)={_t4bf}/{_n}')
        if _t4bf > _t4b3:
            print('  → Utiliser la pleine région corrigerait le bug "tous inconnus".')
    elif _mean3 > 0.85:
        print('→ L\'EXTRACTION 3×3 EST CORRECTE')
        print(f'  cosine moyen = {_mean3:.3f}. Le bug n\'est PAS dans le mapping.')
        print('  Vérifier : PCA/GMM dans le script de traversée (refit ? scaler ?)')
    else:
        print(f'→ RÉSULTATS PARTIELS : cosine 3×3={_mean3:.3f}  full={_meanf:.3f}')
        print('  Consulter les détails TEST 1-3 ci-dessus.')

    print(f'\nFigures enregistrées dans : {OUTPUT_DIR}')

finally:
    _dbg_hook.remove()
    print('\nHook retiré.')
