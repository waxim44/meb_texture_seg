#!/usr/bin/env python3
"""
Inspection visuelle des détections A (isolement géométrique) et C (intensité).
Réutilise les scores MAD de la Phase 1. AUCUN retrait.
Sorties → inspection_A_C/
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import struct, csv, sys
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ─── CONFIG (doit correspondre à Phase 1) ────────────────────────────────────

ROOT       = Path('/home/aidouni/meb_texture_seg')
H5_PATH    = ROOT / 'data/feature_database/database_meb_ouassim.h5'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUT_DIR    = ROOT / 'inspection_A_C'

TEXTURES   = [1, 3, 4, 5, 6, 7, 9]
TNAMES     = {1:'Tot.homogène', 3:'Faisceaux', 4:'Filaments', 5:'Strat.rect',
              6:'Strat.sin',    7:'Granuleux',  9:'Trou'}

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
SEUIL_MAD_AB   = 2.5
SEUIL_MAD_C    = 3.5
FRACTION_BLOCK = 4 / 6
eps            = 1e-8
IMG_H, IMG_W   = 768, 1280
PATCH_SIZE     = 128

# Référence LP : block le plus informatif (utilisé pour LOIO)
LP_BLOCK   = 'stage_2_fpn'
LP_PCA_DIM = 50
LP_C       = 1.0

N_REFS     = 5   # référence patches dans les visuels

# ─── TIFF READER ─────────────────────────────────────────────────────────────

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
        tag, dtype, _ = struct.unpack(bo + 'HHI', e[:8])
        v = e[8:12]
        if   dtype == 3: v = struct.unpack(bo + 'H', v[:2])[0]
        elif dtype == 4: v = struct.unpack(bo + 'I', v)[0]
        tags[tag] = v
    w, h = tags[256], tags[257]
    with open(path, 'rb') as f:
        f.seek(tags[273])
        raw = np.frombuffer(f.read(h * w), dtype=np.uint8).reshape(h, w)
    return raw

_img_cache: dict = {}
def load_image(name: str) -> np.ndarray:
    if name not in _img_cache:
        _img_cache[name] = read_tiff_gray(IMG_DIR / name)
    return _img_cache[name]

# ─── KNN + MAD (identique Phase 1) ──────────────────────────────────────────

def knn_within_class(feats_c, k_eff):
    Nc    = len(feats_c)
    if Nc <= 1: return np.zeros(Nc)
    k_use = min(k_eff, Nc - 1)
    nn    = NearestNeighbors(n_neighbors=k_use + 1, metric='cosine', algorithm='brute')
    nn.fit(feats_c)
    dists, _ = nn.kneighbors(feats_c)
    return dists[:, 1:k_use + 1].mean(axis=1)

def knn_purity_only(feats_all, labels_all, k_eff):
    N     = len(feats_all)
    k_use = min(k_eff, N - 1)
    nn    = NearestNeighbors(n_neighbors=k_use + 1, metric='cosine', algorithm='brute')
    nn.fit(feats_all); _, indices = nn.kneighbors(feats_all)
    purity = np.zeros(N)
    for i in range(N):
        nbrs = indices[i, 1:k_use + 1]
        purity[i] = np.sum(labels_all[nbrs] == labels_all[i]) / k_use
    return purity

def mad_scores_per_class(values, labels):
    scores = np.zeros(len(values))
    for t in TEXTURES:
        mask = labels == t
        if not mask.any(): continue
        v   = values[mask]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        scores[mask] = (v - med) / (1.4826 * mad + eps)
    return scores

# ─── LOIO MULTICLASSE par image ───────────────────────────────────────────────

def loio_correct_proba(feats, labels, img_stems):
    """
    Pour chaque patch : P(true_class | features), estimée LOIO par image.
    Retourne correct_proba (N,) entre 0 et 1.
    """
    N         = len(feats)
    proba_out = np.zeros(N)
    unique_stems = sorted(set(img_stems))

    for stem_test in unique_stems:
        idx_te = np.where(img_stems == stem_test)[0]
        idx_tr = np.where(img_stems != stem_test)[0]
        if len(idx_tr) == 0: continue

        X_tr, X_te = feats[idx_tr], feats[idx_te]
        y_tr        = labels[idx_tr]
        y_te        = labels[idx_te]

        if len(np.unique(y_tr)) < 2: continue

        if X_tr.shape[1] > LP_PCA_DIM:
            pca  = PCA(n_components=LP_PCA_DIM, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=LP_C, class_weight='balanced',
                                 max_iter=1000, solver='lbfgs', random_state=42)
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)
        classes  = list(clf.classes_)

        for j, gi in enumerate(idx_te):
            true_cls = y_te[j]
            if true_cls in classes:
                proba_out[gi] = proba_te[j, classes.index(true_cls)]
    return proba_out

# ─── VISUALISATION ───────────────────────────────────────────────────────────

def three_panel(out_path, patch_info, ref_patches, extra_title=''):
    """
    3 colonnes : (1) crop du patch  (2) image entière + rect  (3) grille références.
    patch_info : dict avec image_source, col, row, title
    ref_patches : liste de dicts {image_source, col, row}
    """
    n_refs = len(ref_patches)
    fig    = plt.figure(figsize=(14, 5))
    gs     = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 2], figure=fig)

    # — Panneau gauche : crop du patch
    ax_crop = fig.add_subplot(gs[0])
    img  = load_image(patch_info['image_source'])
    crop = img[patch_info['row']:patch_info['row'] + PATCH_SIZE,
                patch_info['col']:patch_info['col'] + PATCH_SIZE]
    ax_crop.imshow(crop, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax_crop.set_title(patch_info['title'], fontsize=6.5, pad=3)
    ax_crop.axis('off')

    # — Panneau centre : image entière + rectangle rouge
    ax_img = fig.add_subplot(gs[1])
    ax_img.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto')
    rect = mpatches.Rectangle(
        (patch_info['col'], patch_info['row']), PATCH_SIZE, PATCH_SIZE,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax_img.add_patch(rect)
    ax_img.set_title(patch_info['image_source'], fontsize=6, pad=3)
    ax_img.axis('off')

    # — Panneau droit : grille de référence (N_REFS patches)
    ax_ref = fig.add_subplot(gs[2])
    ax_ref.axis('off')
    n_cols_ref = min(n_refs, 5)
    gs_ref     = gridspec.GridSpecFromSubplotSpec(
        1, n_cols_ref, subplot_spec=gs[2], wspace=0.05
    )
    for j, ref in enumerate(ref_patches[:n_cols_ref]):
        ax_r  = fig.add_subplot(gs_ref[j])
        img_r = load_image(ref['image_source'])
        crop_r = img_r[ref['row']:ref['row'] + PATCH_SIZE,
                       ref['col']:ref['col'] + PATCH_SIZE]
        ax_r.imshow(crop_r, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax_r.set_title(f"ref{j+1}", fontsize=5, pad=1)
        ax_r.axis('off')

    if extra_title:
        fig.suptitle(extra_title, fontsize=7, y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=85, bbox_inches='tight')
    plt.close(fig)


def score_distribution_plot(out_path, score_A_tex, score_C_tex, extr_A_mask,
                             extr_C_mask, labels, n_cols=4):
    """
    Distributions des scores A (score_A par texture) et C_int,
    avec les flaggés marqués en rouge.
    """
    n_rows = (2 * len(TEXTURES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.5, n_rows * 2.5))
    axes = np.array(axes).flatten()
    ax_idx = 0

    for t in TEXTURES:
        mask = labels == t
        # Score A
        ax = axes[ax_idx]; ax_idx += 1
        vals = score_A_tex[mask]
        flag = extr_A_mask[mask]
        ax.hist(vals[~flag], bins=20, color='steelblue', alpha=0.7)
        if flag.any():
            ax.scatter(vals[flag], np.zeros(flag.sum()) + 0.5,
                       color='red', zorder=5, s=40)
        ax.axvline(SEUIL_MAD_AB, color='red', lw=1, ls='--', alpha=0.6)
        ax.set_title(f'{TNAMES[t]} — score_A  (n_flag={flag.sum()})', fontsize=7)
        ax.set_xlabel('score MAD', fontsize=6)

        # Score C_int
        ax = axes[ax_idx]; ax_idx += 1
        vals_c = score_C_tex[mask]
        flag_c = extr_C_mask[mask]
        ax.hist(vals_c[~flag_c], bins=20, color='teal', alpha=0.7)
        if flag_c.any():
            ax.scatter(vals_c[flag_c], np.zeros(flag_c.sum()) + 0.5,
                       color='red', zorder=5, s=40)
        ax.axvline(SEUIL_MAD_C, color='red', lw=1, ls='--', alpha=0.6)
        ax.axvline(-SEUIL_MAD_C, color='red', lw=1, ls='--', alpha=0.6)
        ax.set_title(f'{TNAMES[t]} — score_C_int  (n_flag={flag_c.sum()})', fontsize=7)
        ax.set_xlabel('score MAD', fontsize=6)

    for i in range(ax_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close(fig)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    log_lines = []
    def log(msg=''):
        print(msg); log_lines.append(msg)

    (OUT_DIR / 'partie1_A').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'partie2_C').mkdir(parents=True, exist_ok=True)

    # ── Chargement H5 ────────────────────────────────────────────────────────
    log("Chargement H5...")
    with h5py.File(H5_PATH, 'r') as f:
        all_cat   = f['metadata']['category_ids'][:]
        all_imgs  = np.array([x.decode() for x in f['metadata']['image_names'][:]])
        all_pos   = f['metadata']['positions'][:]
        feats_all = {b: f['features'][b][:] for b in BLOCKS_VOTE}

    tex_mask  = np.isin(all_cat, TEXTURES)
    tex_idx   = np.where(tex_mask)[0]
    N         = len(tex_idx)
    cat_ids   = all_cat[tex_idx]
    img_names = all_imgs[tex_idx]
    positions = all_pos[tex_idx]
    cols      = positions[:, 0].astype(int)
    rows      = positions[:, 1].astype(int)
    feats     = {b: feats_all[b][tex_idx] for b in BLOCKS_VOTE}
    img_stems = np.array([n.replace('.tif', '') for n in img_names])

    log(f"  {N} patches, {len(TEXTURES)} textures")

    # ── Recalcul métriques (Phase 1 identique) ────────────────────────────────
    log("\nRecalcul métriques A, B, C...")

    A_raw = {b: np.zeros(N) for b in BLOCKS_VOTE}
    B_raw = {b: np.zeros(N) for b in BLOCKS_VOTE}

    for b in BLOCKS_VOTE:
        for t in TEXTURES:
            mask_t = cat_ids == t
            if mask_t.sum() <= 1: continue
            A_raw[b][mask_t] = knn_within_class(feats[b][mask_t], k)
        B_raw[b] = knn_purity_only(feats[b], cat_ids, k)

    # Pixel stats
    img_means = {}
    for name in set(img_names):
        img_means[name] = float(load_image(name).mean())

    C_int = np.zeros(N)
    C_uni = np.zeros(N)
    for i in range(N):
        img  = load_image(img_names[i])
        crop = img[rows[i]:rows[i] + PATCH_SIZE, cols[i]:cols[i] + PATCH_SIZE]
        C_int[i] = float(crop.mean()) - img_means[img_names[i]]
        C_uni[i] = float(crop.std())

    # Scores MAD par classe
    score_A_per_block = {}
    for b in BLOCKS_VOTE:
        score_A_per_block[b] = mad_scores_per_class(A_raw[b], cat_ids)

    # Score A de référence : stage_2_fpn (pour les distributions et stabilité)
    score_A_ref = score_A_per_block['stage_2_fpn']
    score_C_int = mad_scores_per_class(C_int, cat_ids)
    score_C_uni = mad_scores_per_class(C_uni, cat_ids)

    # Extrêmes par block
    extr_A_block = {b: score_A_per_block[b] >  SEUIL_MAD_AB for b in BLOCKS_VOTE}
    extr_B_block = {b: mad_scores_per_class(B_raw[b], cat_ids) < -SEUIL_MAD_AB
                    for b in BLOCKS_VOTE}
    extr_C       = (np.abs(score_C_int) > SEUIL_MAD_C) | (np.abs(score_C_uni) > SEUIL_MAD_C)

    # Consensus niveau 1 inter-blocks
    n_blocks_A = np.sum([extr_A_block[b].astype(int) for b in BLOCKS_VOTE], axis=0)
    n_blocks_B = np.sum([extr_B_block[b].astype(int) for b in BLOCKS_VOTE], axis=0)
    frac_A     = n_blocks_A / len(BLOCKS_VOTE)
    frac_B     = n_blocks_B / len(BLOCKS_VOTE)
    extr_A     = frac_A >= FRACTION_BLOCK
    extr_B     = frac_B >= FRACTION_BLOCK

    # Vote par stade
    stades = ['précoce', 'intermédiaire', 'tardif']
    vote_stade = {s: np.zeros(N, dtype=int) for s in stades}
    for b in BLOCKS_VOTE:
        s = BLOCK_STADE[b]
        vote_stade[s] += (extr_A_block[b] | extr_B_block[b]).astype(int)

    log(f"  extr_A : {extr_A.sum()}  extr_B : {extr_B.sum()}  extr_C : {extr_C.sum()}")
    log(f"  A∩C : {(extr_A & extr_C).sum()}  (0 = orthogonales)")

    # ── Références par texture (plus conformes) ───────────────────────────────
    refs_by_texture = {}
    for t in TEXTURES:
        mask_t = (cat_ids == t) & ~extr_A & ~extr_C
        idx_t  = np.where(mask_t)[0]
        if len(idx_t) == 0:
            refs_by_texture[t] = []
            continue
        # Tri par frac_A + frac_C ascendant (plus conformes d'abord)
        conf = frac_A[idx_t] + (extr_C[idx_t].astype(float))
        order = idx_t[np.argsort(conf)][:N_REFS]
        refs_by_texture[t] = [
            {'image_source': img_names[j], 'col': cols[j], 'row': rows[j]}
            for j in order
        ]

    # ── PARTIE 1 — Patches A (isolement géométrique) ─────────────────────────
    log("\n" + "="*60)
    log("PARTIE 1 — Patches flaggés par A (isolement géométrique)")
    log("="*60)

    a_patches = np.where(extr_A)[0]
    log(f"  {len(a_patches)} patches")

    a_summary = []
    for i in sorted(a_patches):
        t        = int(cat_ids[i])
        fa       = frac_A[i]
        sa       = score_A_ref[i]
        dom_stade_name = max(stades, key=lambda s: vote_stade[s][i])
        dom_stade_cnt  = vote_stade[dom_stade_name][i]
        vote_str = f"{dom_stade_name} ({dom_stade_cnt}/6)"
        img_n    = img_names[i]

        title = (f"{TNAMES[t]} | score_A={sa:.2f} | frac_A={fa:.2f} ({n_blocks_A[i]}/6 blocks)"
                 f" | vote={vote_str} | img={img_n}")

        patch_info = {
            'image_source': img_n,
            'col': cols[i], 'row': rows[i],
            'title': f"{TNAMES[t]} | score_A={sa:.2f} | frac={n_blocks_A[i]}/6 | {vote_str}",
        }
        fname    = f"A_pid{tex_idx[i]}_{TNAMES[t].replace('.','').replace(' ','_')}_frac{n_blocks_A[i]}.png"
        out_path = OUT_DIR / 'partie1_A' / fname
        three_panel(out_path, patch_info, refs_by_texture[t],
                    extra_title=title)

        a_summary.append({
            'patch_id': int(tex_idx[i]),
            'texture': t, 'texture_nom': TNAMES[t],
            'image': img_n,
            'col': cols[i], 'row': rows[i],
            'score_A_stage2': round(float(sa), 3),
            'frac_A': round(float(fa), 3),
            'n_blocks_A': int(n_blocks_A[i]),
            'dom_stade': dom_stade_name,
            'vote_précoce': int(vote_stade['précoce'][i]),
            'vote_intermédiaire': int(vote_stade['intermédiaire'][i]),
            'vote_tardif': int(vote_stade['tardif'][i]),
        })
        log(f"  [{TNAMES[t]}] score_A={sa:.2f} frac={n_blocks_A[i]}/6 "
            f"vote_dom={vote_str} img={img_n}")

    # ── PARTIE 2 — Patches C (intensité relative) ─────────────────────────────
    log("\n" + "="*60)
    log("PARTIE 2 — Patches flaggés par C (intensité relative)")
    log("="*60)

    c_patches = np.where(extr_C)[0]
    log(f"  {len(c_patches)} patches")

    c_summary = []
    for i in sorted(c_patches):
        t       = int(cat_ids[i])
        sc_int  = score_C_int[i]
        sc_uni  = score_C_uni[i]
        int_p   = float(load_image(img_names[i])[rows[i]:rows[i]+PATCH_SIZE,
                                                  cols[i]:cols[i]+PATCH_SIZE].mean())
        int_img = img_means[img_names[i]]
        delta   = int_p - int_img
        sens    = "SOMBRE" if delta < 0 else "CLAIR"
        img_n   = img_names[i]
        title = (f"{TNAMES[t]} | int_patch={int_p:.1f} | int_image={int_img:.1f} "
                 f"| C_int={delta:+.1f} ({sens})"
                 f" | scoreMAD_C_int={sc_int:.2f} | scoreMAD_C_uni={sc_uni:.2f}")

        patch_info = {
            'image_source': img_n,
            'col': cols[i], 'row': rows[i],
            'title': (f"{TNAMES[t]} | Δint={delta:+.1f}px ({sens}) | "
                      f"sC_int={sc_int:.2f} sC_uni={sc_uni:.2f}"),
        }
        fname    = (f"C_pid{tex_idx[i]}_{TNAMES[t].replace('.','').replace(' ','_')}"
                    f"_{sens.lower()}.png")
        out_path = OUT_DIR / 'partie2_C' / fname
        three_panel(out_path, patch_info, refs_by_texture[t],
                    extra_title=title)

        c_summary.append({
            'patch_id': int(tex_idx[i]),
            'texture': t, 'texture_nom': TNAMES[t],
            'image': img_n,
            'col': cols[i], 'row': rows[i],
            'int_patch': round(int_p, 1),
            'int_image': round(int_img, 1),
            'C_int': round(delta, 2),
            'score_C_int': round(float(sc_int), 3),
            'score_C_uni': round(float(sc_uni), 3),
            'sens': sens,
        })
        log(f"  [{TNAMES[t]}] C_int={delta:+.1f} ({sens}) "
            f"sMAD_int={sc_int:.2f} sMAD_uni={sc_uni:.2f} img={img_n}")

    # ── PARTIE 3A — Stabilité (seuil 2.5 → 2.0) ──────────────────────────────
    log("\n" + "="*60)
    log("PARTIE 3A — Stabilité MAD (2.5 → 2.0)")
    log("="*60)

    for metric_name, raw_dict in [('A', A_raw)]:
        set_25 = set(np.where(extr_A)[0].tolist())
        # Recalcul à 2.0
        extr_A_20_blocks = {}
        for b in BLOCKS_VOTE:
            s = mad_scores_per_class(raw_dict[b], cat_ids)
            extr_A_20_blocks[b] = s > 2.0
        n_blocks_A20 = np.sum([extr_A_20_blocks[b].astype(int) for b in BLOCKS_VOTE], axis=0)
        extr_A_20    = (n_blocks_A20 / len(BLOCKS_VOTE)) >= FRACTION_BLOCK
        set_20       = set(np.where(extr_A_20)[0].tolist())

        stable = set_25 & set_20
        new_20 = set_20 - set_25
        lost   = set_25 - set_20   # ne devrait pas arriver (2.0 < 2.5 → plus large)

        pct_stable = 100 * len(stable) / len(set_20) if set_20 else 100.0
        flag       = "signal stable ✓" if pct_stable >= 80 else "fragile ⚠"
        log(f"  Métrique A | seuil=2.5 : {len(set_25)}  seuil=2.0 : {len(set_20)}")
        log(f"    Stables dans les 2 : {len(stable)} / {len(set_20)} = {pct_stable:.1f}%  → {flag}")
        log(f"    Nouveaux à 2.0 : {len(new_20)} (patches borderline)")

        for j in sorted(new_20)[:5]:
            log(f"      pid={tex_idx[j]} {TNAMES[cat_ids[j]]} frac20={n_blocks_A20[j]}/6"
                f" img={img_names[j]}")

    # Stabilité C (3.5 → 3.0)
    extr_C_30 = (np.abs(score_C_int) > 3.0) | (np.abs(score_C_uni) > 3.0)
    set_C35   = set(np.where(extr_C)[0].tolist())
    set_C30   = set(np.where(extr_C_30)[0].tolist())
    stable_C  = set_C35 & set_C30
    new_C30   = set_C30 - set_C35
    pct_C     = 100 * len(stable_C) / len(set_C30) if set_C30 else 100.0
    log(f"\n  Métrique C | seuil=3.5 : {len(set_C35)}  seuil=3.0 : {len(set_C30)}")
    log(f"    Stables : {len(stable_C)}/{len(set_C30)} = {pct_C:.1f}%")
    log(f"    Abandonnés en montant de 3.0→3.5 : {len(new_C30)} (borderlines éliminés)")

    # ── PARTIE 3B — Distribution des scores ──────────────────────────────────
    log("\n  Génération distribution_scores.png...")
    score_distribution_plot(
        OUT_DIR / 'distribution_scores.png',
        score_A_ref, score_C_int, extr_A, extr_C, cat_ids
    )

    # ── PARTIE 3C — Comparaison écart score vs seuil (A seulement) ───────────
    log("\n  Écart score_A des flaggés vs médiane de leur classe :")
    for t in TEXTURES:
        mask_t  = cat_ids == t
        flagged = extr_A & mask_t
        if not flagged.any(): continue
        med_cls = np.median(score_A_ref[mask_t])
        scores_flag = score_A_ref[flagged]
        log(f"  {TNAMES[t]} : médiane_classe={med_cls:.2f}  "
            f"scores_flaggés=[{','.join(f'{x:.2f}' for x in sorted(scores_flag,reverse=True))}]"
            f"  Δ_min={scores_flag.min()-med_cls:.2f}")

    # ── PARTIE 4 — Croisement LP (LOIO) ──────────────────────────────────────
    log("\n" + "="*60)
    log("PARTIE 4 — Croisement avec LP LOIO (pires patches)")
    log("="*60)
    log(f"  Calcul LOIO multiclasse sur {LP_BLOCK}...")

    lp_proba = loio_correct_proba(feats[LP_BLOCK], cat_ids, img_stems)

    # "pire" = proba correcte basse
    # Seuil "mal classé" : proba < 0.5 (chances < coin flip)
    LP_SEUIL  = 0.5
    lp_mauvais = lp_proba < LP_SEUIL

    log(f"  Patches mal classés par LP (proba<{LP_SEUIL}): {lp_mauvais.sum()}")

    # Croisement
    a_et_lp  = extr_A & lp_mauvais
    c_et_lp  = extr_C & lp_mauvais
    a_pas_lp = extr_A & ~lp_mauvais
    c_pas_lp = extr_C & ~lp_mauvais

    log(f"\n  A flaggés ({extr_A.sum()}) :")
    log(f"    Aussi mal classés LP : {a_et_lp.sum()}  ({100*a_et_lp.sum()/max(extr_A.sum(),1):.0f}%)"
        f"  → outliers RÉELS (A et LP concordent)")
    log(f"    Bien classés par LP  : {a_pas_lp.sum()} ({100*a_pas_lp.sum()/max(extr_A.sum(),1):.0f}%)"
        f"  → atypiques géométriques BÉNINS (LP les classe bien)")

    log(f"\n  C flaggés ({extr_C.sum()}) :")
    log(f"    Aussi mal classés LP : {c_et_lp.sum()}  ({100*c_et_lp.sum()/max(extr_C.sum(),1):.0f}%)"
        f"  → intensité ET classification anormales")
    log(f"    Bien classés par LP  : {c_pas_lp.sum()} ({100*c_pas_lp.sum()/max(extr_C.sum(),1):.0f}%)"
        f"  → anomalie d'intensité BÉNIGNE (LP robuste)")

    # CSV croisement
    cross_rows = []
    for i in range(N):
        if not (extr_A[i] or extr_C[i]): continue
        cross_rows.append({
            'patch_id'      : int(tex_idx[i]),
            'texture'       : int(cat_ids[i]),
            'texture_nom'   : TNAMES[cat_ids[i]],
            'image'         : img_names[i],
            'col'           : cols[i], 'row': rows[i],
            'extr_A'        : int(extr_A[i]),
            'extr_C'        : int(extr_C[i]),
            'frac_A'        : round(float(frac_A[i]), 3),
            'score_C_int'   : round(float(score_C_int[i]), 3),
            'loio_proba'    : round(float(lp_proba[i]), 3),
            'lp_mal_classe' : int(lp_mauvais[i]),
            'verdict'       : (
                'RÉEL (A+LP)' if extr_A[i] and lp_mauvais[i] else
                'RÉEL (C+LP)' if extr_C[i] and lp_mauvais[i] else
                'BÉNIN (A géom)' if extr_A[i] else
                'BÉNIN (C intens)'
            ),
        })

    with open(OUT_DIR / 'croisement_LP.csv', 'w', newline='', encoding='utf-8') as f:
        if cross_rows:
            w = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
            w.writeheader(); w.writerows(cross_rows)

    # Stats par texture
    log("\n  Par texture :")
    log(f"  {'Texture':<15} {'N_A':>4} {'N_C':>4} {'A+LP':>6} {'C+LP':>6}"
        f" {'lp_proba_moy_A':>14}")
    for t in TEXTURES:
        mask_t = cat_ids == t
        a_t    = extr_A & mask_t
        c_t    = extr_C & mask_t
        if not a_t.any() and not c_t.any(): continue
        lp_a_mean = lp_proba[a_t].mean() if a_t.any() else float('nan')
        log(f"  {TNAMES[t]:<15} {a_t.sum():>4} {c_t.sum():>4}"
            f" {(a_t&lp_mauvais).sum():>6} {(c_t&lp_mauvais).sum():>6}"
            f" {lp_a_mean:>14.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("VERDICT FACTUEL")
    log("="*60)

    # Vote stade pour A
    a_idx     = np.where(extr_A)[0]
    n_precoce = (vote_stade['précoce'][a_idx] > vote_stade['intermédiaire'][a_idx]).sum()
    n_interm  = (vote_stade['intermédiaire'][a_idx] >= vote_stade['précoce'][a_idx]).sum()
    log(f"\nA ({len(a_idx)} patches) :")
    log(f"  Dominante INTERMÉDIAIRE (vrais outliers texture) : {n_interm}")
    log(f"  Dominante PRÉCOCE (artefacts position)          : {n_precoce}")

    # Concentration par image (A)
    ctr_A = Counter(img_names[i] for i in a_idx)
    conc_A = [(img, cnt) for img, cnt in ctr_A.most_common() if cnt >= 3]
    log(f"  Images concentrées (≥3 outliers A) : {len(conc_A)}")
    for img_n, cnt in conc_A:
        log(f"    {img_n} : {cnt} patches A")

    # C
    c_idx = np.where(extr_C)[0]
    ctr_C = Counter(img_names[i] for i in c_idx)
    conc_C = [(img, cnt) for img, cnt in ctr_C.most_common() if cnt >= 3]
    log(f"\nC ({len(c_idx)} patches) :")
    log(f"  Images concentrées (≥3 outliers C) : {len(conc_C)}")
    for img_n, cnt in conc_C:
        log(f"    {img_n} : {cnt} patches C")

    # LP concordance
    log(f"\nCroisement LP (LOIO, {LP_BLOCK}) :")
    log(f"  A concordant avec LP : {a_et_lp.sum()}/{extr_A.sum()}")
    log(f"  C concordant avec LP : {c_et_lp.sum()}/{extr_C.sum()}")

    verdict_A = (
        "SIGNAL — vrais cas limites géométriques"
        if a_et_lp.sum() >= 0.4 * extr_A.sum()
        else "BRUIT — A capte des patches que LP classe bien (atypie bénigne)"
    )
    verdict_C = (
        "SIGNAL — intensité corrèle avec difficulté LP"
        if c_et_lp.sum() >= 0.4 * extr_C.sum()
        else "BRUIT — C capte intensité, LP robuste à l'intensité (bénin)"
    )
    log(f"\n  Verdict A : {verdict_A}")
    log(f"  Verdict C : {verdict_C}")
    log("\n⚠ Inspection seule. Aucun retrait.")

    with open(OUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log(f"\n✓ Sorties dans {OUT_DIR}")

if __name__ == '__main__':
    main()
