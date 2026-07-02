#!/usr/bin/env python3
"""
Phase 1 — Détection d'outliers (MAD, multi-block, sans modèle entraîné)
Sorties → outliers_detection_MAD/
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct, os, csv, sys
from pathlib import Path
from collections import defaultdict, Counter

from sklearn.neighbors import NearestNeighbors

# ─── CONFIG ──────────────────────────────────────────────────────────────────

ROOT       = Path('/home/aidouni/meb_texture_seg')
H5_PATH    = ROOT / 'data/feature_database/database_meb_ouassim.h5'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUTPUT_DIR = ROOT / 'outliers_detection_MAD'

TEXTURES   = [1, 3, 4, 5, 6, 7, 9]
TNAMES     = {1:'Tot.homogène', 3:'Faisceaux', 4:'Filaments', 5:'Strat.rect',
              6:'Strat.sin',    7:'Granuleux',  9:'Trou'}
TDIR       = {1:'1_Tot.homogene', 3:'3_Faisceaux', 4:'4_Filaments',
              5:'5_Strat.rect',   6:'6_Strat.sin', 7:'7_Granuleux', 9:'9_Trou'}
MAD_PEU_FIABLE_SET = {1, 4}  # N < 50

BLOCKS_VOTE = ['stage_1_fpn', 'stage_2_fpn', 'block_4', 'block_7', 'stage_3_fpn', 'stage_4_fpn']
BLOCK_STADE = {
    'stage_1_fpn' : 'précoce',
    'stage_2_fpn' : 'intermédiaire',
    'block_4'     : 'intermédiaire',
    'block_7'     : 'intermédiaire',
    'stage_3_fpn' : 'intermédiaire',
    'stage_4_fpn' : 'tardif',
}

k              = 10
SEUIL_MAD_AB   = 2.5    # ~2.5σ pour A et B (était 3.0 → trop conservateur pour petites classes)
SEUIL_MAD_C    = 3.5    # ~3.5σ pour C (était 3.0 → trop bruité à la frontière)
FRACTION_BLOCK = 4 / 6
CONSENSUS_MIN  = 1
eps            = 1e-8
IMG_H, IMG_W   = 768, 1280
PATCH_SIZE     = 128

# ─── TIFF READER (sans PIL) ───────────────────────────────────────────────────

def read_tiff_gray(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        data = f.read()
    bo  = '<' if data[:2] == b'II' else '>'
    ifd = struct.unpack(bo + 'I', data[4:8])[0]
    pos = ifd
    n   = struct.unpack(bo + 'H', data[pos:pos+2])[0]; pos += 2
    tags = {}
    for _ in range(n):
        e    = data[pos:pos+12]; pos += 12
        tag, dtype, count = struct.unpack(bo + 'HHI', e[:8])
        v    = e[8:12]
        if   dtype == 3: v = struct.unpack(bo + 'H', v[:2])[0]
        elif dtype == 4: v = struct.unpack(bo + 'I', v)[0]
        tags[tag] = v
    w, h = tags[256], tags[257]
    strip_offset = tags[273]
    with open(path, 'rb') as f:
        f.seek(strip_offset)
        raw = np.frombuffer(f.read(h * w), dtype=np.uint8).reshape(h, w)
    return raw

_img_cache: dict = {}

def load_image(img_name: str) -> np.ndarray:
    if img_name not in _img_cache:
        _img_cache[img_name] = read_tiff_gray(IMG_DIR / img_name)
    return _img_cache[img_name]

# ─── KNN HELPERS ─────────────────────────────────────────────────────────────

def knn_within_class(feats_c: np.ndarray, k_eff: int) -> np.ndarray:
    """Mean cosine distance to k nearest same-class neighbours (self excluded)."""
    Nc = len(feats_c)
    if Nc <= 1:
        return np.zeros(Nc)
    k_use = min(k_eff, Nc - 1)
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric='cosine', algorithm='brute')
    nn.fit(feats_c)
    dists, _ = nn.kneighbors(feats_c)
    # column 0 is self (distance ~0), skip it
    return dists[:, 1:k_use + 1].mean(axis=1)


def knn_purity_and_penche(feats_all: np.ndarray, labels_all: np.ndarray,
                           k_eff: int):
    """
    Purity = fraction of k nearest (all classes, self excluded) with same label.
    penche_vers = majority OTHER-class label among those k neighbours.
    Returns purity (N,), penche_vers (N,) with -1 if all neighbours same class.
    """
    N     = len(feats_all)
    k_use = min(k_eff, N - 1)
    nn    = NearestNeighbors(n_neighbors=k_use + 1, metric='cosine', algorithm='brute')
    nn.fit(feats_all)
    _, indices = nn.kneighbors(feats_all)

    purity      = np.zeros(N)
    penche_vers = np.full(N, -1, dtype=int)
    for i in range(N):
        nbrs       = indices[i, 1:k_use + 1]
        nbr_labels = labels_all[nbrs]
        same       = np.sum(nbr_labels == labels_all[i])
        purity[i]  = same / k_use
        others     = nbr_labels[nbr_labels != labels_all[i]]
        if len(others) > 0:
            penche_vers[i] = Counter(others.tolist()).most_common(1)[0][0]
    return purity, penche_vers


def knn_purity_only(feats_all: np.ndarray, labels_all: np.ndarray,
                    k_eff: int) -> np.ndarray:
    N     = len(feats_all)
    k_use = min(k_eff, N - 1)
    nn    = NearestNeighbors(n_neighbors=k_use + 1, metric='cosine', algorithm='brute')
    nn.fit(feats_all)
    _, indices = nn.kneighbors(feats_all)
    purity = np.zeros(N)
    for i in range(N):
        nbrs      = indices[i, 1:k_use + 1]
        same      = np.sum(labels_all[nbrs] == labels_all[i])
        purity[i] = same / k_use
    return purity

# ─── MAD SCORING ─────────────────────────────────────────────────────────────

def mad_scores_per_class(values: np.ndarray, labels: np.ndarray,
                         textures, warn_tag: str) -> tuple:
    """
    Returns scores (N,) and low_mad_classes (set of texture ids with MAD < 1e-6).
    """
    scores    = np.zeros(len(values))
    low_mad   = set()
    for t in textures:
        mask = labels == t
        if mask.sum() == 0:
            continue
        v   = values[mask]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad < 1e-6:
            low_mad.add(t)
            print(f"  ⚠ MAD~0 pour {warn_tag} classe {t} ({TNAMES[t]}) — scores non fiables")
        scores[mask] = (v - med) / (1.4826 * mad + eps)
    return scores, low_mad

# ─── VISUALIZATION ───────────────────────────────────────────────────────────

def save_patch_png(out_path: Path, img_name: str, col: int, row: int,
                   title_left: str, title_right: str, suptitle: str) -> None:
    img  = load_image(img_name)
    crop = img[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                             gridspec_kw={'width_ratios': [1, 2]})

    # LEFT — crop
    axes[0].imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[0].set_title(title_left, fontsize=7, pad=3)
    axes[0].axis('off')

    # RIGHT — full image + red rectangle
    axes[1].imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    rect = mpatches.Rectangle((col, row), PATCH_SIZE, PATCH_SIZE,
                               linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title(title_right, fontsize=7, pad=3)
    axes[1].axis('off')

    fig.suptitle(suptitle, fontsize=7, y=1.0)
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)


def save_grid_png(out_path: Path, entries: list, texture: int,
                  cols_grid: int = 4) -> None:
    """Grid of outlier crops, sorted by consensus then frac."""
    n = len(entries)
    if n == 0:
        return
    rows_grid = (n + cols_grid - 1) // cols_grid
    fig, axes = plt.subplots(rows_grid, cols_grid,
                             figsize=(cols_grid * 3, rows_grid * 3.5))
    axes = np.array(axes).reshape(rows_grid, cols_grid)

    for idx, e in enumerate(entries):
        r, c_ax = divmod(idx, cols_grid)
        ax = axes[r, c_ax]
        img  = load_image(e['image_source'])
        crop = img[e['row']:e['row'] + PATCH_SIZE, e['col']:e['col'] + PATCH_SIZE]
        ax.imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        label = (f"cons={e['consensus']} | {e['metriques_declenchees']}\n"
                 f"vote={e['dom_stade']} | ↗{TNAMES.get(e['penche_vers'], '?')}\n"
                 f"{e['image_source'][:30]}")
        ax.set_title(label, fontsize=5, pad=2)
        ax.axis('off')

    for idx in range(n, rows_grid * cols_grid):
        r, c_ax = divmod(idx, cols_grid)
        axes[r, c_ax].axis('off')

    fig.suptitle(f"Outliers ≥2 — {TNAMES[texture]}", fontsize=9)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    log_lines = []

    def log(msg=''):
        print(msg)
        log_lines.append(msg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in TEXTURES:
        (OUTPUT_DIR / TDIR[t] / 'outliers').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / TDIR[t] / 'reference').mkdir(parents=True, exist_ok=True)

    # ── ÉTAPE 0 : Chargement + contrôles ─────────────────────────────────────
    log("=" * 70)
    log("ÉTAPE 0 — Chargement + contrôles d'intégrité")
    log("=" * 70)

    with h5py.File(H5_PATH, 'r') as f:
        all_cat_ids    = f['metadata']['category_ids'][:]
        all_img_names  = np.array([x.decode() for x in f['metadata']['image_names'][:]])
        all_positions  = f['metadata']['positions'][:]  # [col1, row1, col2, row2]
        block_feats    = {b: f['features'][b][:] for b in BLOCKS_VOTE}

    N_total = len(all_cat_ids)

    # Vérifier que tous les blocks ont le même N
    block_Ns = {b: block_feats[b].shape[0] for b in BLOCKS_VOTE}
    if len(set(block_Ns.values())) != 1:
        log("STOP — Les blocks n'ont pas le même nombre de patches!")
        for b, n in block_Ns.items():
            log(f"  {b}: {n}")
        sys.exit(1)
    log(f"OK — Tous les 6 blocks : {N_total} patches dans le même ordre.")

    # Filtrer sur TEXTURES
    tex_mask  = np.isin(all_cat_ids, TEXTURES)
    tex_idx   = np.where(tex_mask)[0]   # indices globaux H5

    cat_ids   = all_cat_ids[tex_idx]
    img_names = all_img_names[tex_idx]
    positions = all_positions[tex_idx]   # [col1, row1, col2, row2]
    N         = len(tex_idx)

    cols = positions[:, 0].astype(int)
    rows = positions[:, 1].astype(int)

    # Contrôle de bornes
    out_of_bounds = []
    for i in range(N):
        c, r = cols[i], rows[i]
        if c + PATCH_SIZE > IMG_W or r + PATCH_SIZE > IMG_H:
            out_of_bounds.append((tex_idx[i], cat_ids[i], img_names[i], r, c))
    if out_of_bounds:
        log(f"⚠ {len(out_of_bounds)} patches dépassent les bords de l'image :")
        for gidx, t, img, r, c in out_of_bounds[:10]:
            log(f"  global_idx={gidx} texture={t} img={img} row={r} col={c}")
    else:
        log("OK — Aucun patch hors-image.")

    log("\nN patches par texture :")
    for t in TEXTURES:
        n = np.sum(cat_ids == t)
        fiab = "⚠ MAD_PEU_FIABLE" if t in MAD_PEU_FIABLE_SET else ""
        log(f"  {TNAMES[t]:20s} (t={t}) : N={n:3d}  {fiab}")

    # Features filtrées
    feats = {b: block_feats[b][tex_idx] for b in BLOCKS_VOTE}

    # ── ÉTAPE 1 & 2 : Métriques A et B par block ──────────────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPES 1 & 2 — Métriques A (densité classe) et B (pureté voisinage)")
    log("=" * 70)

    A_raw = {b: np.zeros(N) for b in BLOCKS_VOTE}  # mean dist within class
    B_raw = {b: np.zeros(N) for b in BLOCKS_VOTE}  # purity all classes

    # penche_vers calculé une seule fois sur stage_2_fpn
    penche_vers_global = np.full(N, -1, dtype=int)

    for b in BLOCKS_VOTE:
        log(f"\n  Block : {b}")
        F = feats[b]

        # Metric A : kNN within class
        for t in TEXTURES:
            mask_t  = cat_ids == t
            idx_t   = np.where(mask_t)[0]
            Nc      = mask_t.sum()
            k_use   = min(k, Nc - 1)
            if k_use < 1:
                log(f"    ⚠ texture {t} : Nc={Nc}, pas de voisins disponibles — A=0")
                continue
            if k_use < k:
                log(f"    ⚠ texture {t} : Nc={Nc} < k+1, k réduit à {k_use}")
            A_raw[b][mask_t] = knn_within_class(F[mask_t], k)

        # Metric B : kNN purity (tous patches TEXTURES)
        if b == 'stage_2_fpn':
            purity, pv = knn_purity_and_penche(F, cat_ids, k)
            B_raw[b]          = purity
            penche_vers_global = pv
            log(f"    penche_vers calculé sur {b}")
        else:
            B_raw[b] = knn_purity_only(F, cat_ids, k)

        log(f"    A mean={A_raw[b].mean():.4f}  B mean={B_raw[b].mean():.4f}")

    # ── ÉTAPE 3 : Métrique C (intensité pixels) ────────────────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPE 3 — Métrique C (intensité pixels, hors-block)")
    log("=" * 70)

    C_int = np.zeros(N)
    C_uni = np.zeros(N)

    unique_imgs = list(set(img_names))
    img_means   = {}
    for img_name in unique_imgs:
        img = load_image(img_name)
        img_means[img_name] = float(img.mean())

    for i in range(N):
        img  = load_image(img_names[i])
        crop = img[rows[i]:rows[i] + PATCH_SIZE, cols[i]:cols[i] + PATCH_SIZE]
        C_int[i] = float(crop.mean()) - img_means[img_names[i]]
        C_uni[i] = float(crop.std())

    log(f"  C_int : mean={C_int.mean():.2f}  std={C_int.std():.2f}")
    log(f"  C_uni : mean={C_uni.mean():.2f}  std={C_uni.std():.2f}")

    # ── ÉTAPE 4 : Scores MAD par classe, avec direction ───────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPE 4 — Scores MAD par classe")
    log("=" * 70)

    score_A = {b: np.zeros(N) for b in BLOCKS_VOTE}
    score_B = {b: np.zeros(N) for b in BLOCKS_VOTE}

    for b in BLOCKS_VOTE:
        s_A, _ = mad_scores_per_class(A_raw[b], cat_ids, TEXTURES, f"A/{b}")
        s_B, _ = mad_scores_per_class(B_raw[b], cat_ids, TEXTURES, f"B/{b}")
        score_A[b] = s_A
        score_B[b] = s_B

    score_C_int, _ = mad_scores_per_class(C_int, cat_ids, TEXTURES, "C_int")
    score_C_uni, _ = mad_scores_per_class(C_uni, cat_ids, TEXTURES, "C_uni")

    # Extrêmes par block
    extr_A_block = {b: score_A[b] >  SEUIL_MAD_AB for b in BLOCKS_VOTE}
    extr_B_block = {b: score_B[b] < -SEUIL_MAD_AB for b in BLOCKS_VOTE}

    # Métrique C : bilatérale
    extr_C = (np.abs(score_C_int) > SEUIL_MAD_C) | (np.abs(score_C_uni) > SEUIL_MAD_C)

    # ── ÉTAPE 5 : Consensus niveau 1 (inter-blocks) ───────────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPE 5 — Consensus niveau 1 (inter-blocks)")
    log("=" * 70)

    n_blocks_A = np.sum([extr_A_block[b].astype(int) for b in BLOCKS_VOTE], axis=0)
    n_blocks_B = np.sum([extr_B_block[b].astype(int) for b in BLOCKS_VOTE], axis=0)

    frac_A = n_blocks_A / len(BLOCKS_VOTE)
    frac_B = n_blocks_B / len(BLOCKS_VOTE)

    extr_A = frac_A >= FRACTION_BLOCK
    extr_B = frac_B >= FRACTION_BLOCK

    # Vote par stade (pour chaque patch : combien de blocks par stade ont voté extrême A ou B)
    stades = ['précoce', 'intermédiaire', 'tardif']
    vote_stade = {s: np.zeros(N, dtype=int) for s in stades}
    for b in BLOCKS_VOTE:
        s    = BLOCK_STADE[b]
        voted = extr_A_block[b] | extr_B_block[b]
        vote_stade[s] += voted.astype(int)

    log(f"  extr_A (≥{FRACTION_BLOCK:.0%} blocks) : {extr_A.sum()} patches")
    log(f"  extr_B (≥{FRACTION_BLOCK:.0%} blocks) : {extr_B.sum()} patches")

    # ── ÉTAPE 6 : Consensus niveau 2 (inter-métriques) ────────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPE 6 — Consensus niveau 2 (inter-métriques)")
    log("=" * 70)

    consensus = extr_A.astype(int) + extr_B.astype(int) + extr_C.astype(int)
    is_outlier = consensus >= CONSENSUS_MIN

    log(f"  Outliers (consensus≥{CONSENSUS_MIN}) : {is_outlier.sum()} / {N} patches")

    def metriques_declenchees(i):
        parts = []
        if extr_A[i]: parts.append('A')
        if extr_B[i]: parts.append('B')
        if extr_C[i]: parts.append('C')
        return '+'.join(parts) if parts else 'aucune'

    def dom_stade(i):
        """Stade dominant dans les votes A|B pour le patch i."""
        counts = {s: vote_stade[s][i] for s in stades}
        best   = max(counts, key=counts.get)
        total  = n_blocks_A[i] + n_blocks_B[i]
        return best, counts[best], total

    # ── ÉTAPE 7 : Robustesse k (sur stage_2_fpn) ──────────────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPE 7 — Robustesse au choix de k (sur stage_2_fpn)")
    log("=" * 70)

    F_ref = feats['stage_2_fpn']
    k_vals = [8, 10, 15]
    extr_A_k = {}
    extr_B_k = {}

    for kv in k_vals:
        A_kv = np.zeros(N)
        for t in TEXTURES:
            mask_t = cat_ids == t
            if mask_t.sum() <= 1:
                continue
            A_kv[mask_t] = knn_within_class(F_ref[mask_t], kv)
        B_kv = knn_purity_only(F_ref, cat_ids, kv)

        s_A_kv, _ = mad_scores_per_class(A_kv, cat_ids, TEXTURES, f"A/k={kv}")
        s_B_kv, _ = mad_scores_per_class(B_kv, cat_ids, TEXTURES, f"B/k={kv}")

        extr_A_k[kv] = (s_A_kv >  SEUIL_MAD_AB)
        extr_B_k[kv] = (s_B_kv < -SEUIL_MAD_AB)

    # Stabilité : patches extrêmes A dans les 3 k
    for metric_name, extr_k_dict in [('A', extr_A_k), ('B', extr_B_k)]:
        sets = [set(np.where(extr_k_dict[kv])[0]) for kv in k_vals]
        union = sets[0] | sets[1] | sets[2]
        inter = sets[0] & sets[1] & sets[2]
        pct   = 100 * len(inter) / len(union) if union else 100.0
        flag  = "✓ k non-critique" if pct >= 80 else "⚠ FRAGILE"
        log(f"  Métrique {metric_name} : stable={len(inter)}/{len(union)} = {pct:.1f}%  {flag}")
        for kv in k_vals:
            log(f"    k={kv:2d} → {len(extr_k_dict[kv].nonzero()[0])} extrêmes")

    robustesse_msg = []
    for metric_name, extr_k_dict in [('A', extr_A_k), ('B', extr_B_k)]:
        sets  = [set(np.where(extr_k_dict[kv])[0]) for kv in k_vals]
        union = sets[0] | sets[1] | sets[2]
        inter = sets[0] & sets[1] & sets[2]
        pct   = 100 * len(inter) / len(union) if union else 100.0
        flag  = "k non-critique ✓" if pct >= 80 else "FRAGILE ⚠"
        robustesse_msg.append(f"  Métrique {metric_name} (stage_2_fpn) : stable {pct:.1f}% → {flag}")

    # ── ÉTAPE 8 : Sorties ─────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("ÉTAPE 8 — Génération des sorties")
    log("=" * 70)

    # Déterminer images concentrées : >50% patches de la texture dans l'image sont outliers≥2
    concentrated_imgs = {}  # (texture, img_name) → bool
    for t in TEXTURES:
        mask_t = cat_ids == t
        imgs_t = img_names[mask_t]
        cons_t = consensus[mask_t]
        for img_name in set(imgs_t):
            img_mask = imgs_t == img_name
            total_in_img = img_mask.sum()
            outliers_in_img = (cons_t[img_mask] >= 2).sum()
            concentrated_imgs[(t, img_name)] = (outliers_in_img / total_in_img > 0.5)

    # ── outliers.csv ──────────────────────────────────────────────────────────
    outlier_rows = []
    for i in range(N):
        if consensus[i] < 1:
            continue
        stade_name, stade_cnt, total_votes = dom_stade(i)
        pv_label = TNAMES.get(penche_vers_global[i], '?') if penche_vers_global[i] >= 0 else '—'
        metr     = metriques_declenchees(i)
        fiab     = "MAD_PEU_FIABLE" if cat_ids[i] in MAD_PEU_FIABLE_SET else "OK"
        involves_C = 'C' in metr
        a_and_b_only = (extr_A[i] and extr_B[i] and not extr_C[i])

        outlier_rows.append({
            'patch_id'            : int(tex_idx[i]),
            'texture'             : int(cat_ids[i]),
            'texture_nom'         : TNAMES[cat_ids[i]],
            'image_source'        : img_names[i],
            'col'                 : cols[i],
            'row'                 : rows[i],
            'frac_A'              : f"{frac_A[i]:.3f}",
            'frac_B'              : f"{frac_B[i]:.3f}",
            'vote_précoce'        : vote_stade['précoce'][i],
            'vote_intermédiaire'  : vote_stade['intermédiaire'][i],
            'vote_tardif'         : vote_stade['tardif'][i],
            'score_C_int'         : f"{score_C_int[i]:.3f}",
            'score_C_uni'         : f"{score_C_uni[i]:.3f}",
            'penche_vers'         : pv_label,
            'consensus'           : int(consensus[i]),
            'metriques_declenchees': metr,
            'fiabilite_MAD'       : fiab,
            'C_implique'          : involves_C,
            'A_B_seuls'           : a_and_b_only,
        })

    outlier_rows.sort(key=lambda r: (-r['consensus'],
                                      -float(r['frac_A']) - float(r['frac_B'])))

    with open(OUTPUT_DIR / 'outliers.csv', 'w', newline='', encoding='utf-8') as f:
        if outlier_rows:
            writer = csv.DictWriter(f, fieldnames=list(outlier_rows[0].keys()))
            writer.writeheader()
            writer.writerows(outlier_rows)
    log(f"  outliers.csv : {len(outlier_rows)} entrées (consensus≥1)")

    # Index par patch_id pour accès rapide
    outlier_by_idx = {r['patch_id']: r for r in outlier_rows}

    # ── references.csv ────────────────────────────────────────────────────────
    ref_rows = []
    for t in TEXTURES:
        mask_t    = cat_ids == t
        idx_t     = np.where(mask_t)[0]
        cons_t    = consensus[mask_t]
        n_out2    = int((cons_t >= 2).sum())
        N_REF     = min(10, n_out2)

        non_out   = np.where(cons_t == 0)[0]
        if N_REF == 0 or len(non_out) == 0:
            continue

        # conformité = frac_A + frac_B + (1 si extr_C)
        conf_t    = (frac_A[mask_t] + frac_B[mask_t]
                     + extr_C[mask_t].astype(float))
        conf_nonout = conf_t[non_out]
        # tri par conformité croissante (plus petit = plus conforme)
        order     = non_out[np.argsort(conf_nonout)]

        selected  = []
        for j in order:
            gidx   = idx_t[j]
            img_n  = img_names[gidx]
            conc   = concentrated_imgs.get((t, img_n), False)
            selected.append({
                'patch_id'       : int(tex_idx[gidx]),
                'texture'        : int(t),
                'texture_nom'    : TNAMES[t],
                'image_source'   : img_n,
                'col'            : cols[gidx],
                'row'            : rows[gidx],
                'conformite'     : f"{conf_t[j]:.3f}",
                'image_concentree': conc,
                '_local_idx'     : gidx,
            })
            if len(selected) >= N_REF:
                break

        ref_rows.extend(selected)

    with open(OUTPUT_DIR / 'references.csv', 'w', newline='', encoding='utf-8') as f:
        if ref_rows:
            fields = [k for k in ref_rows[0].keys() if not k.startswith('_')]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in ref_rows:
                writer.writerow({k: v for k, v in row.items() if not k.startswith('_')})
    log(f"  references.csv : {len(ref_rows)} entrées")

    # ── PNGs individuels ──────────────────────────────────────────────────────
    log("\n  Génération des PNGs individuels...")
    n_png = 0

    # Outliers (consensus ≥ 2)
    for r in outlier_rows:
        if r['consensus'] < 2:
            continue
        t    = r['texture']
        i_local = np.where(tex_idx == r['patch_id'])[0][0]
        stade_name, stade_cnt, total_votes = dom_stade(i_local)
        vote_str = f"{stade_name[:6]} {stade_cnt}/{total_votes}"
        pv       = r['penche_vers']

        title_left  = (f"{TNAMES[t]} | cons={r['consensus']} | "
                       f"métr={r['metriques_declenchees']}")
        title_right = (f"vote={vote_str} | penche→{pv} | "
                       f"{r['image_source']}")
        suptitle    = title_left + " | " + title_right

        fname = (f"patch_{r['patch_id']}_cons{r['consensus']}"
                 f"_{r['metriques_declenchees'].replace('+','')}.png")
        out_path = OUTPUT_DIR / TDIR[t] / 'outliers' / fname
        save_patch_png(out_path, r['image_source'],
                       r['col'], r['row'],
                       title_left, title_right, suptitle)
        n_png += 1

    # Références
    for r in ref_rows:
        t    = r['texture']
        title_left  = f"{TNAMES[t]} | conformité={r['conformite']}"
        title_right = r['image_source']
        suptitle    = f"{title_left} | {title_right}"
        fname       = f"patch_{r['patch_id']}_conform.png"
        out_path    = OUTPUT_DIR / TDIR[t] / 'reference' / fname
        save_patch_png(out_path, r['image_source'],
                       r['col'], r['row'],
                       title_left, title_right, suptitle)
        n_png += 1

    log(f"  {n_png} PNGs générés")

    # ── Grilles par texture ───────────────────────────────────────────────────
    log("\n  Génération des grilles par texture...")
    for t in TEXTURES:
        mask_t   = cat_ids == t
        idx_t    = np.where(mask_t)[0]
        entries  = []
        for j in idx_t:
            if consensus[j] < 2:
                continue
            stade_name, stade_cnt, total_votes = dom_stade(j)
            entries.append({
                'image_source'       : img_names[j],
                'col'                : cols[j],
                'row'                : rows[j],
                'consensus'          : int(consensus[j]),
                'metriques_declenchees': metriques_declenchees(j),
                'dom_stade'          : stade_name,
                'penche_vers'        : int(penche_vers_global[j]),
                'frac_A'             : frac_A[j],
                'frac_B'             : frac_B[j],
            })
        # tri par consensus décroissant puis frac décroissant
        entries.sort(key=lambda e: (-e['consensus'], -(e['frac_A'] + e['frac_B'])))
        if entries:
            grid_path = OUTPUT_DIR / TDIR[t] / f"grille_outliers_{TNAMES[t]}.png"
            save_grid_png(grid_path, entries, t)

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("SORTIES — RÉSUMÉ")
    log("=" * 70)

    # 1) Sensibilité
    log("\n1) SENSIBILITÉ")
    log(f"  {'Texture':<20} {'N':>4}  {'MAD':>12}  {'≥1':>5}  {'≥2':>5}  {'≥3':>5}")
    log("  " + "-" * 60)
    for t in TEXTURES:
        mask_t = cat_ids == t
        n_t    = mask_t.sum()
        cons_t = consensus[mask_t]
        fiab   = "MAD_PEU_FIABLE" if t in MAD_PEU_FIABLE_SET else "OK"
        log(f"  {TNAMES[t]:<20} {n_t:>4}  {fiab:>12}  "
            f"{(cons_t>=1).sum():>5}  {(cons_t>=2).sum():>5}  {(cons_t>=3).sum():>5}")

    # 2) Par métrique
    log("\n2) PAR MÉTRIQUE (extrêmes niveau 1)")
    log(f"  A seule (extr_A and not extr_B and not extr_C) : "
        f"{(extr_A & ~extr_B & ~extr_C).sum()}")
    log(f"  B seule (extr_B and not extr_A and not extr_C) : "
        f"{(extr_B & ~extr_A & ~extr_C).sum()}")
    log(f"  C seule (extr_C and not extr_A and not extr_B) : "
        f"{(extr_C & ~extr_A & ~extr_B).sum()}")
    log(f"  A+B (sans C)                                   : "
        f"{(extr_A & extr_B & ~extr_C).sum()}")
    log(f"  A+C (sans B)                                   : "
        f"{(extr_A & extr_C & ~extr_B).sum()}")
    log(f"  B+C (sans A)                                   : "
        f"{(extr_B & extr_C & ~extr_A).sum()}")
    log(f"  A+B+C                                          : "
        f"{(extr_A & extr_B & extr_C).sum()}")
    log(f"  Total extrêmes A : {extr_A.sum()}  B : {extr_B.sum()}  C : {extr_C.sum()}")

    # 3) Répartition par stade des outliers ≥2
    log("\n3) RÉPARTITION PAR STADE (outliers≥2)")
    out2_mask = consensus >= 2
    if out2_mask.sum() > 0:
        for s in stades:
            vals = vote_stade[s][out2_mask]
            log(f"  {s:>15s} : mean votes={vals.mean():.2f}  max={vals.max()}")
        n_suspect = ((vote_stade['précoce'] >= 1) & ~(vote_stade['intermédiaire'] >= 1)
                     & out2_mask).sum()
        log(f"  ⚠ Suspects (votés uniquement précoce) : {n_suspect}")
    else:
        log("  Aucun outlier≥2.")

    # 4) Concentration par image
    log("\n4) CONCENTRATION PAR IMAGE (outliers≥2)")
    for t in TEXTURES:
        mask_t  = cat_ids == t
        imgs_t  = img_names[mask_t]
        cons_t  = consensus[mask_t]
        out2_t  = cons_t >= 2
        if out2_t.sum() == 0:
            continue
        log(f"  {TNAMES[t]} :")
        ctr = Counter(imgs_t[out2_t])
        for img_n, cnt in ctr.most_common():
            total_in_img = (imgs_t == img_n).sum()
            conc = "⚠ CONCENTRÉE" if concentrated_imgs.get((t, img_n), False) else ""
            log(f"    {img_n} : {cnt}/{total_in_img} outliers≥2  {conc}")

    # 5) Confusion (penche_vers des outliers≥2)
    log("\n5) CONFUSION (penche_vers, outliers≥2)")
    out2_mask = consensus >= 2
    pv_out2   = penche_vers_global[out2_mask]
    for t in TEXTURES:
        mask_t_out2 = out2_mask & (cat_ids == t)
        if mask_t_out2.sum() == 0:
            continue
        pv_t = penche_vers_global[mask_t_out2]
        ctr  = Counter(pv_t[pv_t >= 0].tolist())
        log(f"  {TNAMES[t]} (n={mask_t_out2.sum()}) :")
        for pv_id, cnt in ctr.most_common():
            log(f"    → {TNAMES.get(pv_id, f'id={pv_id}'):20s} : {cnt}x")

    # 6) Robustesse k
    log("\n6) ROBUSTESSE k (stage_2_fpn)")
    for msg in robustesse_msg:
        log(msg)

    # ── Écriture summary.txt ──────────────────────────────────────────────────
    with open(OUTPUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log(f"\n✓ Tout sauvegardé dans {OUTPUT_DIR}")
    log("  ⚠ ON S'ARRÊTE ICI — inspection avant retrait.")

if __name__ == '__main__':
    main()
