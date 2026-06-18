#!/usr/bin/env python3
"""
explore_boundaries_meb.py
Explorer la base HDF5 pour identifier les frontières texturales NATURELLES :
patches adjacents de catégories différentes dans la même image.
Étape de reconnaissance — aucun forward pass, aucune extraction de features.
"""

import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paramètres ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DB_PATH      = ROOT / 'data' / 'feature_database' / 'database_meb.h5'
CFG_PATH     = ROOT / 'PatchTagger_Output' / 'config' / 'config.json'
IMG_DIR      = ROOT / 'Image_Ouassim'
OUTPUT_DIR   = ROOT / 'outputs' / 'boundary_analysis'
ADJ_TOL                = 10   # tolérance d'adjacence en pixels
MIN_BOUNDARIES_PER_PAIR = 3   # seuil "exploitable" pour l'analyse
CATS_EXCLUDE           = [2, 8, 10, 11, 12, 13]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Résultats → {OUTPUT_DIR}')

# ── Config ─────────────────────────────────────────────────────────────────────
with open(CFG_PATH) as _f:
    _bnd_cfg = json.load(_f)
CATEGORIES = {int(k): v['name']  for k, v in _bnd_cfg['available_categories'].items()}
CAT_COLORS = {int(k): v['color'] for k, v in _bnd_cfg['available_categories'].items()}


def _bnd_hex_rgba(hex_color, alpha=0.45):
    h = hex_color.lstrip('#')
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, alpha)


# ── HDF5 ───────────────────────────────────────────────────────────────────────
with h5py.File(DB_PATH, 'r') as _h5:
    _bnd_IMAGE_NAMES  = _h5['metadata/image_names'][:]
    _bnd_POSITIONS    = _h5['metadata/positions'][:].astype(int)   # (N,4) x1,y1,x2,y2
    _bnd_CATEGORY_IDS = _h5['metadata/category_ids'][:].astype(int)

_bnd_EXCL_SET = set(CATS_EXCLUDE)
_bnd_CATS_VALID = sorted(
    int(c) for c in np.unique(_bnd_CATEGORY_IDS)
    if int(c) not in _bnd_EXCL_SET
)
_bnd_cat2idx = {c: i for i, c in enumerate(_bnd_CATS_VALID)}
print(f'Catégories valides ({len(_bnd_CATS_VALID)}) : '
      f'{[CATEGORIES[c] for c in _bnd_CATS_VALID]}')

_bnd_valid_mask = ~np.isin(_bnd_CATEGORY_IDS, list(_bnd_EXCL_SET))
print(f'Patches valides : {_bnd_valid_mask.sum()} / {len(_bnd_valid_mask)}')

# ── Grouper par image ───────────────────────────────────────────────────────────
_bnd_by_img = defaultdict(list)   # img_name → [global_idx, ...]
for _i in np.where(_bnd_valid_mask)[0]:
    _bnd_by_img[_bnd_IMAGE_NAMES[_i]].append(int(_i))


def _bnd_adj_type(pos_a, pos_b, tol):
    """
    Tester si deux patches sont adjacents (bords quasi-contigus).
    Retourne 'H' (horizontal), 'V' (vertical), ou None.
    """
    x1a, y1a, x2a, y2a = pos_a
    x1b, y1b, x2b, y2b = pos_b
    # Adjacence horizontale : bords X proches ET chevauchement Y
    y_overlap = y1a < y2b and y1b < y2a
    h_close   = abs(x2a - x1b) <= tol or abs(x2b - x1a) <= tol
    if h_close and y_overlap:
        return 'H'
    # Adjacence verticale : bords Y proches ET chevauchement X
    x_overlap = x1a < x2b and x1b < x2a
    v_close   = abs(y2a - y1b) <= tol or abs(y2b - y1a) <= tol
    if v_close and x_overlap:
        return 'V'
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Statistiques générales par image
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 1 — Statistiques par image ===')

_bnd_img_stats = {}
for _nm, _idxs in _bnd_by_img.items():
    _cats = sorted(set(_bnd_CATEGORY_IDS[_idxs].tolist()))
    _bnd_img_stats[_nm] = {
        'n_patches'  : len(_idxs),
        'n_cats'     : len(_cats),
        'cats'       : _cats,
        'idxs'       : _idxs,
    }

_imgs_multi = [(nm, s) for nm, s in _bnd_img_stats.items() if s['n_cats'] >= 2]
print(f'Images avec patches valides        : {len(_bnd_by_img)}')
print(f'Images avec ≥ 2 catégories valides : {len(_imgs_multi)}')
print(f'\nTop 8 images (diversité catégories) :')
for _nm, _s in sorted(_imgs_multi, key=lambda x: (-x[1]['n_cats'], -x[1]['n_patches']))[:8]:
    _cats_str = ', '.join(CATEGORIES.get(_c,'?') for _c in _s['cats'])
    print(f'  {_nm.decode()[:55]:<55}  '
          f'cats={_s["n_cats"]}  patches={_s["n_patches"]}')
    print(f'    → {_cats_str}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2+3 — Détecter les paires adjacentes inter-catégories
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étapes 2+3 — Adjacence et frontières inter-catégories ===')

# Enregistrement de chaque frontière
_bnd_records = []   # liste de dicts

for _nm, _idxs in _bnd_by_img.items():
    if len(_idxs) < 2:
        continue
    for _ai in range(len(_idxs)):
        for _bi in range(_ai + 1, len(_idxs)):
            _ia, _ib = _idxs[_ai], _idxs[_bi]
            _ca, _cb = int(_bnd_CATEGORY_IDS[_ia]), int(_bnd_CATEGORY_IDS[_ib])
            if _ca == _cb:
                continue   # même catégorie, pas une frontière
            _t = _bnd_adj_type(
                _bnd_POSITIONS[_ia], _bnd_POSITIONS[_ib], ADJ_TOL
            )
            if _t is not None:
                _bnd_records.append({
                    'image'   : _nm,
                    'idx_a'   : _ia,
                    'idx_b'   : _ib,
                    'cat_a'   : _ca,
                    'cat_b'   : _cb,
                    'pair'    : (min(_ca, _cb), max(_ca, _cb)),
                    'pos_a'   : tuple(_bnd_POSITIONS[_ia]),
                    'pos_b'   : tuple(_bnd_POSITIONS[_ib]),
                    'adj_type': _t,
                })

print(f'Frontières inter-catégories trouvées : {len(_bnd_records)}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Inventaire des frontières
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 4 — Inventaire ===')

# Matrice paires de catégories
_bnd_pair_counts = defaultdict(int)
_bnd_pair_adj    = defaultdict(lambda: {'H': 0, 'V': 0})
for _r in _bnd_records:
    _bnd_pair_counts[_r['pair']] += 1
    _bnd_pair_adj[_r['pair']][_r['adj_type']] += 1

# Frontières par image
_bnd_per_img = defaultdict(int)
for _r in _bnd_records:
    _bnd_per_img[_r['image']] += 1

# Matrice catégorie × catégorie
_n = len(_bnd_CATS_VALID)
_bnd_mat = np.zeros((_n, _n), dtype=int)
for (_a, _b), _cnt in _bnd_pair_counts.items():
    if _a in _bnd_cat2idx and _b in _bnd_cat2idx:
        _i, _j = _bnd_cat2idx[_a], _bnd_cat2idx[_b]
        _bnd_mat[_i, _j] = _cnt
        _bnd_mat[_j, _i] = _cnt

print('\nMatrice de frontières (paires catégories valides) :')
_cat_abbr = {c: CATEGORIES[c][:10] for c in _bnd_CATS_VALID}
_header = f"{'':>12}" + ''.join(f'{_cat_abbr[c]:>12}' for c in _bnd_CATS_VALID)
print(_header)
for _ci, _ca in enumerate(_bnd_CATS_VALID):
    _row = f'{_cat_abbr[_ca]:>12}'
    for _cj, _cb in enumerate(_bnd_CATS_VALID):
        _v = _bnd_mat[_ci, _cj]
        _row += f'{"---":>12}' if _ci == _cj else f'{_v:>12}'
    print(_row)

print(f'\nFrontières par image (top 8) :')
for _nm, _n_bnd in sorted(_bnd_per_img.items(), key=lambda x: -x[1])[:8]:
    print(f'  {_nm.decode()[:55]:<55}  {_n_bnd} frontières')

print(f'\nPaires exploitables (≥ {MIN_BOUNDARIES_PER_PAIR} frontières) :')
_bnd_exploitable = {
    _pair: _cnt
    for _pair, _cnt in sorted(_bnd_pair_counts.items(), key=lambda x: -x[1])
    if _cnt >= MIN_BOUNDARIES_PER_PAIR
}
for (_a, _b), _cnt in _bnd_exploitable.items():
    _h = _bnd_pair_adj[(_a,_b)]['H']
    _v = _bnd_pair_adj[(_a,_b)]['V']
    print(f'  {CATEGORIES[_a]:<25} ↔ {CATEGORIES[_b]:<25}  '
          f'n={_cnt}  (H={_h}, V={_v})')

_bnd_not_enough = {
    (_a, _b): _cnt
    for (_a, _b), _cnt in _bnd_pair_counts.items()
    if _cnt < MIN_BOUNDARIES_PER_PAIR
}
if _bnd_not_enough:
    print(f'\nPaires insuffisantes (< {MIN_BOUNDARIES_PER_PAIR} frontières) :')
    for (_a, _b), _cnt in sorted(_bnd_not_enough.items(), key=lambda x: -x[1]):
        print(f'  {CATEGORIES[_a]:<25} ↔ {CATEGORIES[_b]:<25}  n={_cnt}')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Visualisation (images les plus riches en frontières)
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 5 — Visualisation ===')

_bnd_top_imgs = sorted(_bnd_per_img.keys(), key=lambda _nm: -_bnd_per_img[_nm])[:4]

# Regrouper les frontières par image pour la figure
_bnd_rec_by_img = defaultdict(list)
for _r in _bnd_records:
    _bnd_rec_by_img[_r['image']].append(_r)

_n_panels = len(_bnd_top_imgs)
_ncols    = min(2, _n_panels)
_nrows    = (_n_panels + _ncols - 1) // _ncols
fig, axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 8, _nrows * 7))
axes_flat = np.array(axes).flatten() if _n_panels > 1 else [axes]

for _pi, _nm in enumerate(_bnd_top_imgs):
    _ax = axes_flat[_pi]

    # Charger l'image originale
    try:
        _img_gray = np.array(Image.open(IMG_DIR / _nm.decode()).convert('L'))
        _ax.imshow(_img_gray, cmap='gray', aspect='auto')
    except Exception:
        _ax.set_facecolor('#222222')

    # Dessiner tous les patches valides de cette image (colorés)
    for _idx in _bnd_img_stats.get(_nm, {}).get('idxs', []):
        _c = int(_bnd_CATEGORY_IDS[_idx])
        _x1, _y1, _x2, _y2 = _bnd_POSITIONS[_idx]
        _rect = plt.Rectangle(
            (_x1, _y1), _x2 - _x1, _y2 - _y1,
            linewidth=0.6,
            edgecolor=_bnd_hex_rgba(CAT_COLORS.get(_c, '#808080'), 1.0)[:3],
            facecolor=_bnd_hex_rgba(CAT_COLORS.get(_c, '#808080'), 0.35),
        )
        _ax.add_patch(_rect)

    # Dessiner les frontières détectées (ligne entre centres)
    for _r in _bnd_rec_by_img[_nm]:
        _x1a,_y1a,_x2a,_y2a = _r['pos_a']
        _x1b,_y1b,_x2b,_y2b = _r['pos_b']
        _cxa = (_x1a + _x2a) / 2;  _cya = (_y1a + _y2a) / 2
        _cxb = (_x1b + _x2b) / 2;  _cyb = (_y1b + _y2b) / 2
        # Ligne blanche avec contour noir pour la lisibilité
        _ax.plot([_cxa, _cxb], [_cya, _cyb], '-', color='black',  lw=3.5, zorder=4)
        _ax.plot([_cxa, _cxb], [_cya, _cyb], '-', color='yellow', lw=1.8, zorder=5)
        # Marqueur au point médian
        _mx, _my = (_cxa+_cxb)/2, (_cya+_cyb)/2
        _ax.plot(_mx, _my, 'o', color='yellow', ms=5, zorder=6)

    _ax.set_title(
        f'{_nm.decode()[:45]}\n'
        f'{_bnd_per_img[_nm]} frontières  ·  '
        f'{_bnd_img_stats[_nm]["n_cats"]} catégories  ·  '
        f'{_bnd_img_stats[_nm]["n_patches"]} patches',
        fontsize=8,
    )
    _ax.axis('off')

# Légende catégories
_handles = [
    mpatches.Patch(color=CAT_COLORS[_c], label=CATEGORIES[_c])
    for _c in _bnd_CATS_VALID if _c in CAT_COLORS
]
_handles.append(mlines.Line2D([], [], color='yellow', lw=2, label='Frontière inter-cat.'))
fig.legend(handles=_handles, loc='lower center', ncol=min(len(_handles), 5),
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

# Masquer les panneaux vides
for _pi in range(_n_panels, len(axes_flat)):
    axes_flat[_pi].axis('off')

fig.suptitle(
    f'Frontières texturales naturelles — {len(_bnd_records)} frontières inter-catégories\n'
    f'ligne jaune = frontière entre patches adjacents de catégories différentes',
    fontsize=10,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boundary_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: boundary_visualization.png')

# ─────────────────────────────────────────────────────────────────────────────
# Étape 6 — Rapport txt + recommandation finale
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== Étape 6 — Rapport + recommandation ===')

_bnd_total_boundary_patches = len(set(
    _r['idx_a'] for _r in _bnd_records
) | set(
    _r['idx_b'] for _r in _bnd_records
))

_bnd_report = []
_bnd_report.append('=' * 72)
_bnd_report.append('INVENTAIRE DES FRONTIÈRES TEXTURALES — database_meb.h5')
_bnd_report.append(f'ADJ_TOL={ADJ_TOL}px  ·  MIN_PER_PAIR={MIN_BOUNDARIES_PER_PAIR}')
_bnd_report.append('=' * 72)
_bnd_report.append('')
_bnd_report.append('── STATISTIQUES GÉNÉRALES ──')
_bnd_report.append(f'  Images avec patches valides        : {len(_bnd_by_img)}')
_bnd_report.append(f'  Images avec ≥ 2 catégories valides : {len(_imgs_multi)}')
_bnd_report.append(f'  Total frontières inter-catégories  : {len(_bnd_records)}')
_bnd_report.append(f'  Patches impliqués dans une frontière: {_bnd_total_boundary_patches}')
_bnd_report.append(f'  Paires distinctes de catégories    : {len(_bnd_pair_counts)}')
_bnd_report.append('')
_bnd_report.append('── MATRICE DE FRONTIÈRES (paire A ↔ B) ──')
_bnd_report.append('')
_bnd_report.append(_header)
for _ci, _ca in enumerate(_bnd_CATS_VALID):
    _row = f'{_cat_abbr[_ca]:>12}'
    for _cj, _cb in enumerate(_bnd_CATS_VALID):
        _v = _bnd_mat[_ci, _cj]
        _row += f'{"---":>12}' if _ci == _cj else f'{_v:>12}'
    _bnd_report.append(_row)
_bnd_report.append('')
_bnd_report.append('── FRONTIÈRES PAR IMAGE ──')
for _nm, _n_bnd in sorted(_bnd_per_img.items(), key=lambda x: -x[1]):
    _s = _bnd_img_stats[_nm]
    _cats_str = '+'.join(str(_c) for _c in _s['cats'])
    _bnd_report.append(
        f'  {_nm.decode()[:55]:<55}  n_bnd={_n_bnd:>3}  cats=[{_cats_str}]'
    )
_bnd_report.append('')
_bnd_report.append('── LISTE DÉTAILLÉE DES FRONTIÈRES ──')
_bnd_report.append(
    f'  {"image":<45}  {"catA":<22}  {"catB":<22}  {"type":>4}  '
    f'{"posA":>20}  {"posB":>20}'
)
_bnd_report.append('  ' + '-' * 138)
for _r in sorted(_bnd_records, key=lambda x: (x['image'], x['cat_a'], x['cat_b'])):
    _bnd_report.append(
        f'  {_r["image"].decode()[:45]:<45}  '
        f'{CATEGORIES[_r["cat_a"]]:<22}  {CATEGORIES[_r["cat_b"]]:<22}  '
        f'{"  "+_r["adj_type"]:>4}  '
        f'{str(_r["pos_a"]):>20}  {str(_r["pos_b"]):>20}'
    )
_bnd_report.append('')
_bnd_report.append('── RECOMMANDATION ──')
_bnd_report.append('')
_bnd_report.append(f'Paires EXPLOITABLES (≥ {MIN_BOUNDARIES_PER_PAIR} frontières) :')
_bnd_report.append('')
for (_a, _b), _cnt in sorted(_bnd_exploitable.items(), key=lambda x: -x[1]):
    _h = _bnd_pair_adj[(_a,_b)]['H']
    _v = _bnd_pair_adj[(_a,_b)]['V']
    _bnd_report.append(
        f'  {CATEGORIES[_a]:<25} ↔ {CATEGORIES[_b]:<25}  '
        f'n={_cnt:>3}  (H={_h}, V={_v})'
    )
_bnd_report.append('')
_bnd_report.append(f'Paires INSUFFISANTES (< {MIN_BOUNDARIES_PER_PAIR} frontières) :')
for (_a, _b), _cnt in sorted(_bnd_not_enough.items(), key=lambda x: -x[1]):
    _bnd_report.append(f'  {CATEGORIES[_a]:<25} ↔ {CATEGORIES[_b]:<25}  n={_cnt}')
_bnd_report.append('')
_bnd_report.append('CONCLUSION :')
_bnd_report.append(
    f'  → {len(_bnd_exploitable)} paires exploitables pour le test de frontière.'
)
_bnd_report.append(
    f'  → {len(_bnd_records)} paires de patches adjacents inter-catégories en tout.'
)
_bnd_report.append(
    f'  → {_bnd_total_boundary_patches} patches individuels impliqués dans ≥ 1 frontière.'
)
_bnd_report.append(
    f'  → Suggestion : pour chaque paire exploitable, utiliser TOUTES les frontières'
)
_bnd_report.append(
    f'    disponibles (pas de sous-échantillonnage vu la rareté).'
)
_bnd_report.append(
    f'  → À ÉVITER : créer des frontières artificielles par collage — les vraies'
)
_bnd_report.append(
    f'    frontières capturent la transition biologique réelle.'
)

_bnd_txt = '\n'.join(_bnd_report)
with open(OUTPUT_DIR / 'boundary_inventory.txt', 'w') as _f:
    _f.write(_bnd_txt)
print('Saved: boundary_inventory.txt')

# ── Résumé console final ───────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('RÉSUMÉ FINAL — FRONTIÈRES TEXTURALES NATURELLES')
print('=' * 60)
print(f'  Total frontières inter-catégories : {len(_bnd_records)}')
print(f'  Patches impliqués                 : {_bnd_total_boundary_patches}')
print(f'  Paires exploitables               : {len(_bnd_exploitable)}')
print()
print('TOP PAIRES (par nombre de frontières) :')
for (_a, _b), _cnt in sorted(_bnd_exploitable.items(), key=lambda x: -x[1]):
    print(f'  {CATEGORIES[_a]:<25} ↔ {CATEGORIES[_b]:<25}  n={_cnt}')
print()
print('IMAGES LES PLUS RICHES :')
for _nm, _n_bnd in sorted(_bnd_per_img.items(), key=lambda x: -x[1])[:5]:
    print(f'  {_nm.decode()[:55]}  {_n_bnd} frontières')
print()
print('PROCHAINE ÉTAPE SUGGÉRÉE :')
print('  → Extraire les features block_0 des patches adjacents')
print('  → Analyser les profils de transition (gradient de features)')
print('  → Tester si block_0 "voit" une vraie rupture à la frontière')

print(f'\n=== Fichiers dans {OUTPUT_DIR} ===')
for _p in sorted(OUTPUT_DIR.iterdir()):
    print(f'  {_p.name}')
