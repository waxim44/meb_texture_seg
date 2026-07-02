#!/usr/bin/env python3
"""
attention_texture_coherence.py
Teste si l'attention GLOBALE de TextureSAM (blocks 7, 10, 13) regroupe
les patches de même texture. Mesure le ratio intra/inter par texture et
par (block, head). 3 blocks × 4 heads = 12 configurations.
"""

import json, sys, warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom as _scipy_zoom

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
_att_ROOT        = Path(__file__).resolve().parents[1]
_att_H5_PATH     = _att_ROOT / 'data' / 'feature_database' / 'database_meb_ouassim.h5'
_att_IMG_DIR     = _att_ROOT / 'Image_Ouassim'
_att_CKPT        = _att_ROOT / 'checkpoints' / 'sam2.1_hiera_small_1.pt'
_att_CFG_PATH    = _att_ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
_att_OUTPUT_DIR  = _att_ROOT / 'output_ouassim' / 'attention_coherence'
_att_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_att_SEED          = 42
_att_N_IMAGES      = 4
_att_GLOBAL_BLOCKS = [7, 10, 13]
_att_N_HEADS       = 4
_att_FEAT_H        = 64
_att_FEAT_W        = 64
_att_N_POS         = _att_FEAT_H * _att_FEAT_W   # 4096
_att_HEAD_DIM      = 96   # 384 dim / 4 heads
_att_CATS_EXCL     = {2, 8, 10, 11, 12, 13}
_att_MIN_PATCHES   = 3    # patches minimum d'une texture dans une image pour l'inclure

np.random.seed(_att_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Config + métadonnées
# ─────────────────────────────────────────────────────────────────────────────
with open(_att_CFG_PATH) as _f:
    _att_cfg = json.load(_f)
_att_CATEGORIES = {int(k): v['name'] for k, v in _att_cfg['available_categories'].items()}
_att_CAT_COLORS  = {int(k): v.get('color', '#888888')
                    for k, v in _att_cfg['available_categories'].items()}

with h5py.File(_att_H5_PATH, 'r') as _h5:
    _att_all_names = np.array([
        n.decode('utf-8') if isinstance(n, (bytes, np.bytes_)) else str(n)
        for n in _h5['metadata/image_names'][:]
    ])
    _att_all_cats  = _h5['metadata/category_ids'][:].astype(int)
    _att_all_pos   = _h5['metadata/positions'][:]  # (N,4): x_min,y_min,x_max,y_max

_att_CATS_VALID = sorted(
    int(c) for c in np.unique(_att_all_cats)
    if int(c) not in _att_CATS_EXCL
    and (_att_all_cats == int(c)).sum() >= 10
)
_att_N_CATS = len(_att_CATS_VALID)
print(f'Catégories valides : {[_att_CATEGORIES[c] for c in _att_CATS_VALID]}')

# ─────────────────────────────────────────────────────────────────────────────
# Sélection des N_IMAGES images les plus riches en textures distinctes
# ─────────────────────────────────────────────────────────────────────────────
_att_img_info = {}
for _img_name in np.unique(_att_all_names):
    if not (_att_IMG_DIR / _img_name).exists():
        continue
    _mask_img  = _att_all_names == _img_name
    _cats_here = _att_all_cats[_mask_img]
    _n_cats    = sum(
        1 for c in _att_CATS_VALID
        if (_cats_here == c).sum() >= _att_MIN_PATCHES
    )
    _n_patches = int(np.isin(_cats_here, _att_CATS_VALID).sum())
    if _n_cats >= 2:
        _att_img_info[_img_name] = (_n_cats, _n_patches)

if not _att_img_info:
    raise RuntimeError('Aucune image avec ≥2 textures valides trouvée dans Image_Ouassim')

_att_selected = sorted(_att_img_info, key=lambda x: _att_img_info[x], reverse=True)
_att_selected = _att_selected[:_att_N_IMAGES]
print(f'\nImages sélectionnées ({_att_N_IMAGES}):')
for _i in _att_selected:
    _nc, _np = _att_img_info[_i]
    print(f'  {_i}: {_nc} textures, {_np} patches')

# ─────────────────────────────────────────────────────────────────────────────
# Chargement modèle + hooks QKV sur les blocks globaux
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(_att_ROOT))
from build_feature_database import load_model as _bfdb_load_model
from build_feature_database import preprocess as _bfdb_preprocess

_att_device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\nChargement modèle...')
_att_encoder = _bfdb_load_model(_att_CKPT, str(_att_device))

# Hook sur la couche qkv de chaque block global : capture Q et K
# Output de qkv: (B, N_pos, 3*dim_out) = (1, 4096, 1152)
# Reshape → (B, N_pos, 3, heads, head_dim) → unbind → q, k, v
_att_qkv_cache   = {}   # {block_idx: (Q, K)} tensors CPU (1, heads, N_pos, head_dim)
_att_qkv_handles = []

for _bi in _att_GLOBAL_BLOCKS:
    def _make_hook(_bidx, _nheads=_att_N_HEADS, _hdim=_att_HEAD_DIM):
        def _hook(module, inp, out):
            # Pour blocks globaux (window_size=0), input est (B,H,W,C)
            # → qkv output est (B,H,W,3*C_out) : 4D
            _od = out.detach()
            if _od.dim() == 4:
                B, H, W, _ = _od.shape
                N = H * W
                _od = _od.reshape(B, N, -1)   # → (B, N, 3*C_out)
            elif _od.dim() == 3:
                B, N, _ = _od.shape
            else:
                return
            qkv = _od.reshape(B, N, 3, _nheads, _hdim)
            q   = qkv[:, :, 0].transpose(1, 2)   # (B, heads, N, head_dim)
            k   = qkv[:, :, 1].transpose(1, 2)
            _att_qkv_cache[_bidx] = (q.cpu(), k.cpu())
        return _hook
    _att_qkv_handles.append(
        _att_encoder.trunk.blocks[_bi].attn.qkv.register_forward_hook(_make_hook(_bi))
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _att_to_grid(x_min, y_min, x_max, y_max, orig_H, orig_W):
    """Coordonnées image → position linéaire dans la grille 64×64."""
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    fx = min(_att_FEAT_W - 1, max(0, int(cx * _att_FEAT_W / orig_W)))
    fy = min(_att_FEAT_H - 1, max(0, int(cy * _att_FEAT_H / orig_H)))
    return fy * _att_FEAT_W + fx


def _att_attn_row(Q, K, pos_q, head_h):
    """
    Ligne d'attention de la position pos_q, head head_h.
    Q, K: (heads, N_pos, head_dim) tenseurs CPU.
    Retourne numpy (N_pos,) = softmax(q·Kᵀ/√d).
    """
    scale  = _att_HEAD_DIM ** -0.5
    q_vec  = Q[head_h, pos_q, :]     # (head_dim,)
    k_mat  = K[head_h, :, :]         # (N_pos, head_dim)
    logits = (q_vec @ k_mat.T) * scale   # (N_pos,)
    return F.softmax(logits, dim=0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale : calcul des ratios intra/inter
# ─────────────────────────────────────────────────────────────────────────────
# _att_all_ratios[cat][block][head] = liste de ratios moyens (un par image)
_att_all_ratios = {
    c: {b: {h: [] for h in range(_att_N_HEADS)} for b in _att_GLOBAL_BLOCKS}
    for c in _att_CATS_VALID
}

# Pour les plots visuels : stocker les données de la meilleure image
_att_visual    = None   # sera rempli ci-dessous

try:
    for _img_name in _att_selected:
        print(f'\n── {_img_name} ──')
        _img_path = _att_IMG_DIR / _img_name

        # Métadonnées patches de cette image
        _m       = _att_all_names == _img_name
        _cats_i  = _att_all_cats[_m]
        _pos_i   = _att_all_pos[_m]
        _mv      = np.isin(_cats_i, _att_CATS_VALID)
        _cats_i, _pos_i = _cats_i[_mv], _pos_i[_mv]
        if len(_cats_i) == 0:
            continue

        # Forward pass
        _att_qkv_cache.clear()
        _tensor, _oH, _oW = _bfdb_preprocess(_img_path, str(_att_device))
        with torch.no_grad():
            _att_encoder(_tensor)

        # Positions grille 64×64 pour chaque patch
        _gpos = np.array([
            _att_to_grid(_pos_i[_pi, 0], _pos_i[_pi, 1],
                         _pos_i[_pi, 2], _pos_i[_pi, 3], _oH, _oW)
            for _pi in range(len(_cats_i))
        ])

        # Textures présentes avec assez de patches
        _cats_ok = [c for c in _att_CATS_VALID
                    if (_cats_i == c).sum() >= _att_MIN_PATCHES]
        print(f'  {len(_cats_i)} patches, textures présentes: '
              f'{[_att_CATEGORIES[c][:10] for c in _cats_ok]}')

        # Stocker pour le plot visuel (image avec le plus de textures)
        if _att_visual is None and len(_cats_ok) >= 2:
            _att_visual = {
                'img_name':  _img_name,
                'img_path':  _img_path,
                'orig_H':    _oH, 'orig_W': _oW,
                'cats_i':    _cats_i.copy(),
                'pos_i':     _pos_i.copy(),
                'gpos':      _gpos.copy(),
                'cats_ok':   _cats_ok,
                'qkv_cache': {k: (Q.clone(), K.clone())
                              for k, (Q, K) in _att_qkv_cache.items()},
            }

        # Calcul des ratios
        for _bi in _att_GLOBAL_BLOCKS:
            if _bi not in _att_qkv_cache:
                continue
            _Q, _K = _att_qkv_cache[_bi]
            _Q, _K = _Q[0], _K[0]   # (heads, N_pos, head_dim)

            for _cat in _cats_ok:
                _pc     = _gpos[_cats_i == _cat]
                _po     = _gpos[_cats_i != _cat]
                if len(_pc) < 2 or len(_po) == 0:
                    continue

                for _h in range(_att_N_HEADS):
                    _ratios_h = []
                    for _pq in _pc:
                        _row   = _att_attn_row(_Q, _K, int(_pq), _h)
                        _intra_pos = [p for p in _pc if p != _pq]
                        if not _intra_pos:
                            continue
                        _intra = _row[np.array(_intra_pos)].mean()
                        _inter = _row[_po].mean()
                        if _inter > 1e-12:
                            _ratios_h.append(float(_intra / _inter))

                    if _ratios_h:
                        _r = float(np.mean(_ratios_h))
                        _att_all_ratios[_cat][_bi][_h].append(_r)
                        print(f'    B{_bi} H{_h} {_att_CATEGORIES[_cat][:14]:14}: {_r:.3f}')

finally:
    for _hd in _att_qkv_handles:
        _hd.remove()
    print('\nHooks retirés.')

# ─────────────────────────────────────────────────────────────────────────────
# Agrégation : matrice ratio [N_cats × 12 configs]
# ─────────────────────────────────────────────────────────────────────────────
_att_CONFIG_LABELS = [f'B{b}H{h}' for b in _att_GLOBAL_BLOCKS for h in range(_att_N_HEADS)]
_att_N_CONFIGS     = len(_att_CONFIG_LABELS)

_att_ratio_matrix = np.full((_att_N_CATS, _att_N_CONFIGS), np.nan)

for _ci, _cat in enumerate(_att_CATS_VALID):
    for _ki, (_bi, _h) in enumerate(
        [(b, h) for b in _att_GLOBAL_BLOCKS for h in range(_att_N_HEADS)]
    ):
        _vals = _att_all_ratios[_cat][_bi][_h]
        if _vals:
            _att_ratio_matrix[_ci, _ki] = float(np.mean(_vals))

_att_CAT_LABELS = [_att_CATEGORIES[c] for c in _att_CATS_VALID]

print('\n── Ratio intra/inter moyen (moy. images) ──')
print(f'{"Texture":<24}', end='')
for _cl in _att_CONFIG_LABELS:
    print(f'{_cl:>8}', end='')
print()
for _ci, _cat in enumerate(_att_CATS_VALID):
    print(f'{_att_CATEGORIES[_cat]:<24}', end='')
    for _ki in range(_att_N_CONFIGS):
        _v = _att_ratio_matrix[_ci, _ki]
        print(f'{_v:8.3f}' if not np.isnan(_v) else '     NaN', end='')
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Heatmap ratio par texture × config block-head
# ─────────────────────────────────────────────────────────────────────────────
print('\nPlot 1 — ratio_heatmap.png')

_att_valid_rows = [_ci for _ci in range(_att_N_CATS)
                   if not np.all(np.isnan(_att_ratio_matrix[_ci]))]
_att_mat_plot   = _att_ratio_matrix[_att_valid_rows]
_att_labels_plot = [_att_CAT_LABELS[_ci] for _ci in _att_valid_rows]

# Trier textures par ratio moyen décroissant
_att_row_order  = np.argsort(-np.nanmean(_att_mat_plot, axis=1))
_att_mat_plot   = _att_mat_plot[_att_row_order]
_att_labels_plot = [_att_labels_plot[i] for i in _att_row_order]

_att_vmin = max(0.0, np.nanmin(_att_mat_plot) * 0.9)
_att_vmax = np.nanmax(_att_mat_plot) * 1.05

fig1, ax1 = plt.subplots(figsize=(12, max(4, len(_att_labels_plot) * 0.55 + 2)))
_im1 = ax1.imshow(_att_mat_plot, aspect='auto',
                   cmap='RdYlGn', vmin=_att_vmin, vmax=_att_vmax)
plt.colorbar(_im1, ax=ax1, label='Ratio intra/inter (>1 = regroupe la texture)')

# Annotations cellules
for _ri in range(_att_mat_plot.shape[0]):
    for _ci in range(_att_mat_plot.shape[1]):
        _v = _att_mat_plot[_ri, _ci]
        if not np.isnan(_v):
            _col = 'white' if _v > (_att_vmax * 0.75) or _v < 0.8 else 'black'
            ax1.text(_ci, _ri, f'{_v:.2f}', ha='center', va='center',
                     fontsize=7.5, color=_col)

# Séparateurs blocks
for _sep in [4, 8]:
    ax1.axvline(_sep - 0.5, color='black', linewidth=1.5, linestyle='-')

ax1.set_xticks(range(_att_N_CONFIGS))
ax1.set_xticklabels(_att_CONFIG_LABELS, rotation=45, ha='right', fontsize=9)
ax1.set_yticks(range(len(_att_labels_plot)))
ax1.set_yticklabels(_att_labels_plot, fontsize=10)

# Étiquettes de blocks en haut
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks([1.5, 5.5, 9.5])
ax1_top.set_xticklabels([f'Block {b}' for b in _att_GLOBAL_BLOCKS], fontsize=10)

ax1.set_title(
    'Ratio attention intra/inter par texture et par (block, head)\n'
    'Vert > 1 = l\'attention regroupe la texture ✓   |   Rouge < 1 = dispersion',
    fontsize=11, fontweight='bold', pad=15
)
ax1.axhline(-0.5, color='grey', lw=0.5)
fig1.tight_layout()
fig1.savefig(_att_OUTPUT_DIR / 'ratio_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Cartes d'attention visuelles (patch → quoi regarde-t-il ?)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 2 — attention_maps_visuel.png')

if _att_visual is not None:
    _v = _att_visual

    # Trouver le meilleur (block, head) pour chaque texture
    def _att_best_config(cat_id):
        """Retourne (block_idx, head_idx) avec le meilleur ratio pour cat_id."""
        if cat_id not in _att_CATS_VALID:
            return _att_GLOBAL_BLOCKS[0], 0
        _ci = _att_CATS_VALID.index(cat_id)
        _bh_ratios = []
        for _bi in _att_GLOBAL_BLOCKS:
            for _h in range(_att_N_HEADS):
                _vals = _att_all_ratios[cat_id][_bi][_h]
                _r    = float(np.mean(_vals)) if _vals else 0.0
                _bh_ratios.append((_r, _bi, _h))
        _bh_ratios.sort(reverse=True)
        return _bh_ratios[0][1], _bh_ratios[0][2]

    # Sélectionner 2 textures exemples : la meilleure et la plus difficile
    _best_ratio_per_cat = {}
    for _cat in _att_CATS_VALID:
        _best_r = 0.0
        for _bi in _att_GLOBAL_BLOCKS:
            for _h in range(_att_N_HEADS):
                _vals = _att_all_ratios[_cat][_bi][_h]
                if _vals:
                    _best_r = max(_best_r, float(np.mean(_vals)))
        _best_ratio_per_cat[_cat] = _best_r

    # Textures présentes dans l'image visuelle
    _cats_avail = [c for c in _v['cats_ok']
                   if (_v['cats_i'] == c).sum() >= _att_MIN_PATCHES]

    _cats_avail_sorted = sorted(_cats_avail,
                                key=lambda c: _best_ratio_per_cat.get(c, 0),
                                reverse=True)
    _example_cats = _cats_avail_sorted[:2] if len(_cats_avail_sorted) >= 2 \
                    else _cats_avail_sorted

    # Charger image originale
    _img_pil  = Image.open(_v['img_path']).convert('L')
    _img_arr  = np.array(_img_pil)

    n_ex = len(_example_cats)
    fig2, axes2 = plt.subplots(n_ex, 2, figsize=(14, 5 * n_ex))
    if n_ex == 1:
        axes2 = axes2[np.newaxis, :]

    for _ei, _ecat in enumerate(_example_cats):
        _ecat_mask = _v['cats_i'] == _ecat
        _ecat_pos  = _v['gpos'][_ecat_mask]
        _other_pos = _v['gpos'][~_ecat_mask]
        _ecat_name = _att_CATEGORIES[_ecat]

        # Choisir la query : le patch de _ecat avec le plus de voisins
        _pq_idx = 0  # premier patch de la catégorie
        _pq     = int(_ecat_pos[_pq_idx])

        # Meilleur (block, head)
        _best_b, _best_h = _att_best_config(_ecat)
        _best_ratio_v    = _best_ratio_per_cat.get(_ecat, 0.0)

        if _best_b not in _v['qkv_cache']:
            axes2[_ei, 0].set_visible(False)
            axes2[_ei, 1].set_visible(False)
            continue

        _Q, _K = _v['qkv_cache'][_best_b]
        _Q, _K = _Q[0], _K[0]

        # Ligne d'attention du patch query → N_pos
        _row = _att_attn_row(_Q, _K, _pq, _best_h)
        _att_map_64 = _row.reshape(_att_FEAT_H, _att_FEAT_W)

        # Upscale 64→image
        _zy = _v['orig_H'] / _att_FEAT_H
        _zx = _v['orig_W'] / _att_FEAT_W
        _att_map_full = _scipy_zoom(_att_map_64, (_zy, _zx), order=1)
        _att_map_full = (_att_map_full - _att_map_full.min()) / \
                        (np.ptp(_att_map_full) + 1e-12)

        # Convertir positions grille → coord image (centre)
        def _gp2xy(gp):
            _fy, _fx = divmod(int(gp), _att_FEAT_W)
            return (_fx + 0.5) * _v['orig_W'] / _att_FEAT_W, \
                   (_fy + 0.5) * _v['orig_H'] / _att_FEAT_H

        # --- Panel gauche : image avec toutes positions catégories
        ax_l = axes2[_ei, 0]
        ax_l.imshow(_img_arr, cmap='gray', aspect='auto')
        for _cat2 in _cats_avail:
            _m2  = _v['cats_i'] == _cat2
            _gp2 = _v['gpos'][_m2]
            _col = _att_CAT_COLORS.get(_cat2, '#888888')
            for _gpp in _gp2:
                _xx, _yy = _gp2xy(_gpp)
                _mk  = 'o' if _cat2 == _ecat else 's'
                _sz  = 80 if _cat2 == _ecat else 40
                ax_l.scatter(_xx, _yy, c=_col, s=_sz, marker=_mk,
                              edgecolors='white', linewidths=0.8, zorder=5)
        # Marquer le patch query
        _qx, _qy = _gp2xy(_pq)
        ax_l.scatter(_qx, _qy, c='yellow', s=200, marker='*',
                     edgecolors='black', linewidths=1.5, zorder=10,
                     label='Query patch')
        ax_l.set_title(f'{_ecat_name}\nPositions patches (★ = query)',
                       fontsize=9, fontweight='bold')
        ax_l.axis('off')

        # --- Panel droit : carte d'attention
        ax_r = axes2[_ei, 1]
        ax_r.imshow(_img_arr, cmap='gray', aspect='auto', alpha=0.5)
        ax_r.imshow(_att_map_full, cmap='hot', alpha=0.5, aspect='auto',
                    vmin=0, vmax=1)
        # Marquer patches de même texture
        for _gpp in _ecat_pos:
            _xx, _yy = _gp2xy(_gpp)
            _mk = '★' if _gpp == _pq else '●'
            ax_r.scatter(_xx, _yy, c='cyan', s=100 if _gpp != _pq else 200,
                         marker='*' if _gpp == _pq else 'o',
                         edgecolors='blue', linewidths=1.0, zorder=10)
        ax_r.set_title(
            f'Carte d\'attention — B{_best_b}H{_best_h}\n'
            f'ratio intra/inter = {_best_ratio_v:.3f}',
            fontsize=9, fontweight='bold'
        )
        ax_r.axis('off')

    fig2.suptitle(
        f'Attention globale — {_v["img_name"]}\n'
        'Zones chaudes = ce que le patch regarde. ● = patches de même texture.',
        fontsize=11, fontweight='bold'
    )
    fig2.tight_layout()
    fig2.savefig(_att_OUTPUT_DIR / 'attention_maps_visuel.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('  Saved.')
else:
    print('  Pas de données visuelles disponibles — skipped.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Meilleur ratio par texture (sur les 12 configs)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 3 — ratio_par_texture.png')

_att_best_ratio = {}
_att_best_config_per_cat = {}
for _cat in _att_CATS_VALID:
    _best_r, _best_b, _best_h = 0.0, _att_GLOBAL_BLOCKS[0], 0
    for _bi in _att_GLOBAL_BLOCKS:
        for _h in range(_att_N_HEADS):
            _vals = _att_all_ratios[_cat][_bi][_h]
            if _vals:
                _r = float(np.mean(_vals))
                if _r > _best_r:
                    _best_r, _best_b, _best_h = _r, _bi, _h
    _att_best_ratio[_cat] = _best_r
    _att_best_config_per_cat[_cat] = (_best_b, _best_h)

_att_sorted_cats = sorted(_att_CATS_VALID, key=lambda c: _att_best_ratio[c], reverse=True)

fig3, ax3 = plt.subplots(figsize=(11, 5))
_colors3 = [_att_CAT_COLORS.get(c, '#888888') for c in _att_sorted_cats]
_bars3   = ax3.bar(range(_att_N_CATS),
                    [_att_best_ratio[c] for c in _att_sorted_cats],
                    color=_colors3, edgecolor='white', linewidth=0.5, alpha=0.88)

ax3.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
            label='Ratio = 1 (aléatoire)')

# Annotations au dessus des barres
for _xi, _cat in enumerate(_att_sorted_cats):
    _b, _h = _att_best_config_per_cat[_cat]
    _r     = _att_best_ratio[_cat]
    ax3.text(_xi, _r + 0.01, f'B{_b}H{_h}', ha='center', va='bottom',
             fontsize=8, color='black', fontweight='bold')

ax3.set_xticks(range(_att_N_CATS))
ax3.set_xticklabels([_att_CATEGORIES[c] for c in _att_sorted_cats],
                     rotation=30, ha='right', fontsize=9)
ax3.set_ylabel('Meilleur ratio intra/inter', fontsize=11)
ax3.set_title(
    'Meilleur ratio attention intra/inter par texture (max sur 12 configs)\n'
    '> 1 : l\'attention globale regroupe la texture ✓',
    fontsize=11, fontweight='bold'
)
ax3.legend(fontsize=9)
ax3.set_ylim(0, max(_att_best_ratio.values()) * 1.15 + 0.05)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig(_att_OUTPUT_DIR / 'ratio_par_texture.png', dpi=150)
plt.close(fig3)
print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Têtes spécialisées : ratio moyen par (block, head)
# ─────────────────────────────────────────────────────────────────────────────
print('Plot 4 — heads_specialisees.png')

_att_head_mean = np.zeros((_att_N_CONFIGS,))
_att_head_std  = np.zeros((_att_N_CONFIGS,))
_att_configs   = [(b, h) for b in _att_GLOBAL_BLOCKS for h in range(_att_N_HEADS)]

for _ki, (_bi, _h) in enumerate(_att_configs):
    _all_r = []
    for _cat in _att_CATS_VALID:
        _all_r.extend(_att_all_ratios[_cat][_bi][_h])
    _att_head_mean[_ki] = np.mean(_all_r) if _all_r else 0.0
    _att_head_std[_ki]  = np.std(_all_r)  if _all_r else 0.0

# Couleurs par block
_att_blk_colors = {7: '#3498db', 10: '#e74c3c', 13: '#2ecc71'}
_colors4 = [_att_blk_colors[b] for b, h in _att_configs]

fig4, ax4 = plt.subplots(figsize=(12, 5))
_bars4 = ax4.bar(range(_att_N_CONFIGS), _att_head_mean, yerr=_att_head_std,
                  color=_colors4, edgecolor='white', linewidth=0.5,
                  alpha=0.88, capsize=4, error_kw={'linewidth': 1.5})

ax4.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
            label='Ratio = 1 (aléatoire)')

ax4.set_xticks(range(_att_N_CONFIGS))
ax4.set_xticklabels(_att_CONFIG_LABELS, rotation=45, ha='right', fontsize=10)
ax4.set_ylabel('Ratio intra/inter moyen (toutes textures)', fontsize=11)
ax4.set_title(
    'Têtes "texture-aware" : ratio moyen par (block, head)\n'
    '> 1 : cette tête regroupe globalement les textures',
    fontsize=11, fontweight='bold'
)

# Légende blocks
from matplotlib.patches import Patch
_leg4 = [Patch(facecolor=_att_blk_colors[b], label=f'Block {b}')
         for b in _att_GLOBAL_BLOCKS]
ax4.legend(handles=_leg4 + [plt.Line2D([0], [0], color='red', ls='--',
                                         label='Ratio=1')],
            fontsize=9, loc='upper right')

# Annotations valeurs
for _ki, (_mv, _sv) in enumerate(zip(_att_head_mean, _att_head_std)):
    ax4.text(_ki, _mv + _sv + 0.01, f'{_mv:.3f}', ha='center',
             va='bottom', fontsize=7.5, fontweight='bold')

ax4.set_ylim(0, max(_att_head_mean) * 1.2 + 0.05)

# Séparateurs blocks
for _sep in [4, 8]:
    ax4.axvline(_sep - 0.5, color='black', linewidth=1.5, linestyle='--', alpha=0.4)

ax4.grid(axis='y', alpha=0.3)
fig4.tight_layout()
fig4.savefig(_att_OUTPUT_DIR / 'heads_specialisees.png', dpi=150)
plt.close(fig4)
print('  Saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Résumé console
# ─────────────────────────────────────────────────────────────────────────────
_att_best_head_idx  = int(np.argmax(_att_head_mean))
_att_best_head_lbl  = _att_CONFIG_LABELS[_att_best_head_idx]
_att_best_head_mean = _att_head_mean[_att_best_head_idx]

_att_cats_grouped   = [c for c in _att_CATS_VALID if _att_best_ratio.get(c, 0) > 1.0]
_att_cats_not       = [c for c in _att_CATS_VALID if _att_best_ratio.get(c, 0) <= 1.0]

print('\n' + '=' * 66)
print('RÉSUMÉ — COHÉRENCE TEXTURALE DANS L\'ATTENTION GLOBALE')
print('=' * 66)
print(f'Blocks analysés : {_att_GLOBAL_BLOCKS} (attention globale, 64×64, 4 heads)')
print(f'Images utilisées : {len(_att_selected)}')
print()
print('RATIO INTRA/INTER PAR TEXTURE (meilleur sur 12 configs):')
for _cat in _att_sorted_cats:
    _r    = _att_best_ratio[_cat]
    _b, _h = _att_best_config_per_cat[_cat]
    _mark = '✓' if _r > 1.0 else '✗'
    print(f'  {_mark} {_att_CATEGORIES[_cat]:<22} ratio={_r:.3f}  (B{_b}H{_h})')
print()
print(f'Textures REGROUPÉES par l\'attention (ratio>1) : {len(_att_cats_grouped)}/{_att_N_CATS}')
for _c in _att_cats_grouped:
    print(f'  → {_att_CATEGORIES[_c]}')
print()
print(f'Tête la plus texture-aware : {_att_best_head_lbl} (ratio moyen={_att_best_head_mean:.3f})')
print('=' * 66)

# ─────────────────────────────────────────────────────────────────────────────
# Génération du fichier Markdown
# ─────────────────────────────────────────────────────────────────────────────
print('\nGénération attention_texture_coherence.md...')

_att_md_lines = [
    '# Analyse de cohérence texturale dans l\'attention globale de TextureSAM',
    '',
    '## Objectif',
    '',
    'Tester si l\'attention globale de TextureSAM regroupe les patches de même texture :',
    'un patch de catégorie *c* regarde-t-il préférentiellement les autres patches de la même catégorie *c* ?',
    'La métrique centrale est le **ratio intra/inter** = attention moyenne vers patches de même texture',
    '/ attention moyenne vers patches d\'autres textures. Un ratio > 1 indique un regroupement.',
    '',
    '## Pourquoi attention globale seulement',
    '',
    'Dans Hiera Small (SAM2), les blocks à **fenêtre locale** (window_size > 0) ne peuvent voir',
    'que leur voisinage immédiat. Un patch Granuleux ne voit pas les autres patches Granuleux',
    'situés à l\'autre bout de l\'image. Seuls les **blocks 7, 10 et 13** ont `window_size = 0`',
    '(attention globale) : chaque position peut attendre toutes les 4096 positions de la grille 64×64.',
    '',
    f'**Configuration** : 3 blocks × 4 heads = {_att_N_CONFIGS} configurations par texture.',
    '',
    '## Démarche',
    '',
    '1. **Extraction des poids Q, K** : hook sur la couche `qkv` (Linear) de chaque block global.',
    '   Output shape : `(B, 4096, 1152)` → reshape → Q, K de forme `(1, 4, 4096, 96)`.',
    '',
    '2. **Mapping patches → grille 64×64** : pour un patch annoté `(x_min, y_min, x_max, y_max)`',
    '   en coordonnées image originale `(orig_H, orig_W)`, le centre est converti en position',
    '   `(fy, fx)` dans la grille 64×64 avec `scale = 64 / orig_W` et `scale = 64 / orig_H`.',
    '',
    '3. **Calcul du ratio intra/inter** : pour chaque query patch `q` de texture *c* :',
    '   ```',
    '   attn_row = softmax(Q[q] · Kᵀ / √96)   [4096 valeurs — softmax sur TOUT le contexte]',
    '   intra = mean(attn_row[p] for p in patches_de_texture_c, p ≠ q)',
    '   inter = mean(attn_row[p] for p in patches_d\'autres_textures)',
    '   ratio = intra / inter',
    '   ```',
    '',
    '## Résultats',
    '',
    f'**Images analysées** : {len(_att_selected)} images Ouassim à {_att_N_CATS} textures distinctes.',
    '',
    '### Ratio meilleur par texture (max sur 12 configs)',
    '',
    '| Texture | Meilleur ratio | (Block, Head) | Regroupée ? |',
    '|---------|---------------|---------------|------------|',
]

for _cat in _att_sorted_cats:
    _r    = _att_best_ratio[_cat]
    _b, _h = _att_best_config_per_cat[_cat]
    _mark = '✓ oui' if _r > 1.0 else '✗ non'
    _att_md_lines.append(
        f'| {_att_CATEGORIES[_cat]} | {_r:.3f} | B{_b}H{_h} | {_mark} |'
    )

_att_md_lines += [
    '',
    '### Têtes les plus "texture-aware"',
    '',
    '| Config | Ratio moyen (toutes textures) |',
    '|--------|-------------------------------|',
]
_head_sorted_idx = np.argsort(-_att_head_mean)
for _ki in _head_sorted_idx[:6]:
    _bl, _hl = _att_configs[_ki]
    _att_md_lines.append(
        f'| B{_bl}H{_hl} | {_att_head_mean[_ki]:.3f} ± {_att_head_std[_ki]:.3f} |'
    )

# Analyse facile vs difficile
_att_easy_cats  = [1, 6]   # Homogène, Granuleux
_att_hard_cats  = [4]       # Stratifié rectiligne
_easy_ratios    = [_att_best_ratio.get(c, 0) for c in _att_easy_cats if c in _att_CATS_VALID]
_hard_ratios    = [_att_best_ratio.get(c, 0) for c in _att_hard_cats if c in _att_CATS_VALID]
_easy_mean      = float(np.mean(_easy_ratios)) if _easy_ratios else float('nan')
_hard_mean      = float(np.mean(_hard_ratios)) if _hard_ratios else float('nan')

_att_md_lines += [
    '',
    f'**Textures faciles** (Homogène, Granuleux) : ratio moyen = **{_easy_mean:.3f}**',
    f'**Textures difficiles** (Stratifié rectiligne) : ratio moyen = **{_hard_mean:.3f}**',
    '',
    '## Conclusion',
    '',
]

_n_grouped = len(_att_cats_grouped)
if _n_grouped >= _att_N_CATS // 2:
    _att_md_lines.append(
        f'L\'attention globale **regroupe {_n_grouped}/{_att_N_CATS} textures** (ratio > 1).'
    )
    _att_md_lines.append(
        'Le modèle encode partiellement la cohérence texturale même sans supervision explicite.'
    )
else:
    _att_md_lines.append(
        f'L\'attention globale regroupe seulement **{_n_grouped}/{_att_N_CATS} textures** (ratio > 1),'
    )
    _att_md_lines.append(
        'suggérant que l\'attention encode surtout la **structure spatiale / position**,'
    )
    _att_md_lines.append(
        'plutôt que la texture sémantique des patches MEB Ouassim.'
    )

_att_md_lines += [
    '',
    f'- **Tête la plus texture-aware** : `{_att_best_head_lbl}` (ratio moyen = {_att_best_head_mean:.3f})',
    f'- **Textures regroupées** : {", ".join(_att_CATEGORIES[c] for c in _att_cats_grouped) or "aucune"}',
    f'- **Textures non regroupées** : {", ".join(_att_CATEGORIES[c] for c in _att_cats_not) or "aucune"}',
    '',
    '**Piste** : les têtes montrant ratio > 1 pourraient être exploitées comme signal de segmentation',
    'faiblement supervisé, en propageant l\'attention d\'un patch annoté vers ses voisins texturaux.',
    '',
    '**Lien avec les difficultés observées** : si Stratifié rectiligne a un faible ratio, l\'attention',
    'ne les distingue pas → cohérent avec le recall de 23% observé sur images Ouassim (block_4 LP).',
    '',
    '## Fichiers générés',
    '',
    f'- `ratio_heatmap.png` — heatmap textures × 12 configs (vert > 1 = regroupe)',
    f'- `attention_maps_visuel.png` — cartes d\'attention sur exemples',
    f'- `ratio_par_texture.png` — meilleur ratio par texture',
    f'- `heads_specialisees.png` — ratio moyen par (block, head)',
]

_att_md_text = '\n'.join(_att_md_lines)
with open(_att_OUTPUT_DIR / 'attention_texture_coherence.md', 'w') as _fmd:
    _fmd.write(_att_md_text + '\n')
print('  Saved.')

print(f'\nFichiers dans {_att_OUTPUT_DIR}:')
for _fn in ['ratio_heatmap.png', 'attention_maps_visuel.png',
            'ratio_par_texture.png', 'heads_specialisees.png',
            'attention_texture_coherence.md']:
    _p = _att_OUTPUT_DIR / _fn
    print(f'  {"✓" if _p.exists() else "✗"}  {_fn}')
