"""
DIAGNOSTIC VISUEL — Pires patches par texture (LOIO honnête)
Meilleur (checkpoint, block) depuis Q1.  Aucune suppression.  Support visuel uniquement.
"""

import sys, re, struct, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT       = Path('/home/aidouni/meb_texture_seg')
PATCH_ROOT = ROOT / 'PatchTagger_Output/patches'
IMG_DIR    = ROOT / 'Image_Ouassim'
FEAT_CACHE = ROOT / 'output_ouassim/compare_checkpoints'
Q1_JSON    = FEAT_CACHE / 'results_q1.json'
OUT_DIR    = ROOT / 'output_ouassim/outlier_inspection'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTURES  = [1, 3, 4, 5, 6, 7, 9]
TNAMES    = {1:'Tot.homogène', 3:'Faisceaux', 4:'Filaments', 5:'Strat.rect',
             6:'Strat.sin', 7:'Granuleux', 9:'Trou'}
TSHORT    = {1:'Homog', 3:'Faisc', 4:'Filam', 5:'S.rect',
             6:'S.sin', 7:'Granu', 9:'Trou'}
TUNIFORM  = {1, 9}
SEED      = 42
PCA_DIM   = 50
N_PIRES   = 10
N_REF     = 5
C_LR      = 1.0

# block commun pour le LP multiclasse (→ "penche vers")
MULTI_CKPT  = 'ft_1.0'
MULTI_REP   = 'block_6'


# ─── Cache features ───────────────────────────────────────────────────────────

def load_feat(ckpt_name, rep):
    """Charge (816, D) depuis le cache .npy de compare_checkpoints."""
    p = FEAT_CACHE / f'feat_{ckpt_name}_{rep}.npy'
    if not p.exists():
        raise FileNotFoundError(f"Cache manquant : {p.name}")
    return np.load(str(p))


# ─── Meilleur (ckpt, block) par texture depuis Q1 ────────────────────────────

def best_rep_per_texture(q1):
    """q1[ckpt][rep][str(t)] = {mean, std, n_folds}"""
    ckpts = list(q1.keys())
    reps  = list(q1[ckpts[0]].keys())
    best  = {}
    for t in TEXTURES:
        tk = str(t)
        best_ckpt, best_rep, best_score = 'ft_1.0', MULTI_REP, -1.0
        for ck in ckpts:
            for rep in reps:
                m = q1[ck][rep].get(tk, {}).get('mean', 0.0)
                if m > best_score:
                    best_score, best_ckpt, best_rep = m, ck, rep
        best[t] = (best_ckpt, best_rep, best_score)
    return best


# ─── I/O patches ──────────────────────────────────────────────────────────────

def read_tiff_gray(path):
    with open(path, 'rb') as f: data = f.read()
    bo  = '<' if data[:2] == b'II' else '>'
    ifd = struct.unpack(bo+'I', data[4:8])[0]
    pos = ifd
    n   = struct.unpack(bo+'H', data[pos:pos+2])[0]; pos += 2
    tags = {}
    for _ in range(n):
        e = data[pos:pos+12]; pos += 12
        tag, dtype, _ = struct.unpack(bo+'HHI', e[:8]); v = e[8:12]
        if dtype == 3: v = struct.unpack(bo+'H', v[:2])[0]
        elif dtype == 4: v = struct.unpack(bo+'I', v)[0]
        tags[tag] = v
    w, h = tags[256], tags[257]
    with open(path, 'rb') as f:
        f.seek(tags[273])
        raw = np.frombuffer(f.read(h*w), dtype=np.uint8).reshape(h, w)
    return raw


def parse_patches():
    pat = re.compile(r'^(.+)_\((\d+)_(\d+)\)$')
    result = []
    for t in TEXTURES:
        for f in sorted((PATCH_ROOT/str(t)).iterdir()):
            if f.suffix != '.tif' or '_cp_masks_' in f.name: continue
            m = pat.match(f.stem)
            if m:
                result.append({'texture': t, 'stem': m.group(1),
                               'row': int(m.group(2)), 'col': int(m.group(3))})
    return result


def load_patch_crop(stem, row, col):
    """Renvoie ndarray (128,128) uint8 — crop depuis l'image source."""
    img = read_tiff_gray(IMG_DIR / (stem + '.tif'))
    py, px = row * 128, col * 128
    return img[py:py+128, px:px+128]


def load_img_thumb(stem, row, col, thumb_w=180):
    """Image complète redimensionnée avec rectangle rouge."""
    img = read_tiff_gray(IMG_DIR / (stem + '.tif'))
    H, W = img.shape
    scale = thumb_w / W
    th, tw = int(H * scale), thumb_w
    thumb = np.array(Image.fromarray(img).resize((tw, th), Image.BILINEAR))
    # rectangle patch
    py1, px1 = int(row*128*scale), int(col*128*scale)
    py2, px2 = int((row+1)*128*scale), int((col+1)*128*scale)
    py1, px1 = max(0, py1), max(0, px1)
    py2, px2 = min(th-1, py2), min(tw-1, px2)
    return thumb, (py1, px1, py2, px2)


# ─── LOIO one-vs-rest : proba par patch ───────────────────────────────────────

def loio_proba_ovr(X_all, patches_meta, texture_c):
    """
    Retourne probas[i] = proba que patch i soit de texture_c
    (estimée dans son fold LOIO propre : image i retirée du train).
    probas[i] = NaN si le patch n'appartient pas à texture_c.
    """
    N = len(patches_meta)
    probas = np.full(N, np.nan)
    stems  = sorted(set(p['stem'] for p in patches_meta))

    for stem_test in stems:
        idx_te = [i for i, p in enumerate(patches_meta) if p['stem'] == stem_test]
        idx_tr = [i for i, p in enumerate(patches_meta) if p['stem'] != stem_test]
        if not idx_tr: continue

        X_tr = X_all[idx_tr]; X_te = X_all[idx_te]
        y_tr = np.array([1 if patches_meta[i]['texture'] == texture_c else 0
                         for i in idx_tr], dtype=np.int32)

        if len(np.unique(y_tr)) < 2: continue

        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=C_LR, class_weight='balanced',
                                 max_iter=1000, solver='lbfgs', random_state=SEED)
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)
        pos_idx  = list(clf.classes_).index(1)
        for j, gi in enumerate(idx_te):
            probas[gi] = proba_te[j, pos_idx]

    return probas


# ─── LOIO multiclasse : "penche vers" ─────────────────────────────────────────

def loio_multiclass(X_all, patches_meta):
    """
    Retourne pred[i] = texture prédite par LP multiclasse LOIO.
    Entraîné sur TOUTES les textures, block commun.
    """
    N      = len(patches_meta)
    pred   = np.full(N, -1, dtype=np.int32)
    stems  = sorted(set(p['stem'] for p in patches_meta))
    labels = np.array([p['texture'] for p in patches_meta], dtype=np.int32)

    for stem_test in stems:
        idx_te = [i for i, p in enumerate(patches_meta) if p['stem'] == stem_test]
        idx_tr = [i for i, p in enumerate(patches_meta) if p['stem'] != stem_test]
        if not idx_tr: continue

        X_tr = X_all[idx_tr]; X_te = X_all[idx_te]
        y_tr = labels[idx_tr]

        if len(np.unique(y_tr)) < 2: continue

        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=C_LR, class_weight='balanced',
                                 max_iter=1000, solver='lbfgs', random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_pred_te = clf.predict(X_te)
        for j, gi in enumerate(idx_te):
            pred[gi] = y_pred_te[j]

    return pred


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_texture(texture_c, pires, refs, best_ckpt, best_rep, best_recall, out_path):
    """
    pires / refs : liste de dicts
      {idx, stem, row, col, proba_ovr, penche_vers, texture}
    """
    tag   = '[U]' if texture_c in TUNIFORM else '[S]'
    ncols = 5
    n_p   = len(pires)
    n_r   = len(refs)
    n_rows_p = (n_p + ncols - 1) // ncols   # 1 ou 2
    total_rows = n_rows_p + (1 if n_r > 0 else 0)

    fig = plt.figure(figsize=(ncols * 3.2, total_rows * 3.8))
    fig.suptitle(
        f"t{texture_c} {tag} {TNAMES[texture_c]}\n"
        f"Meilleur : {best_ckpt} | {best_rep} | recall LOIO = {best_recall:.3f}\n"
        f"ROUGE = pires  VERT = référence bien classée",
        fontsize=9, fontweight='bold', y=1.01
    )

    def draw_cell(ax, entry, border_color):
        crop  = load_patch_crop(entry['stem'], entry['row'], entry['col'])
        thumb, rect = load_img_thumb(entry['stem'], entry['row'], entry['col'])

        ax.imshow(crop, cmap='gray', vmin=0, vmax=255, aspect='equal')
        ax.axis('off')

        # Inset : image entière avec rectangle
        inset = ax.inset_axes([0.0, 0.0, 0.35, 0.35])
        inset.imshow(thumb, cmap='gray', vmin=0, vmax=255)
        py1, px1, py2, px2 = rect
        from matplotlib.patches import Rectangle
        inset.add_patch(Rectangle((px1, py1), px2-px1, py2-py1,
                                   linewidth=1.5, edgecolor='red',
                                   facecolor='none'))
        inset.axis('off')

        penche_str = TSHORT.get(entry['penche_vers'], '?') if entry.get('penche_vers') else '?'
        title = (f"p={entry['proba_ovr']:.2f}  →{penche_str}\n"
                 f"{entry['stem'][-18:]}  ({entry['row']},{entry['col']})")
        ax.set_title(title, fontsize=6.5, color=border_color, pad=2)

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color); spine.set_linewidth(2.5)
            spine.set_visible(True)

    # Pires
    for i, entry in enumerate(pires):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(total_rows, ncols, r * ncols + c + 1)
        draw_cell(ax, entry, 'red')

    # Cellules vides dans les lignes pires
    for i in range(n_p, n_rows_p * ncols):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(total_rows, ncols, r * ncols + c + 1)
        ax.axis('off')

    # Références
    for j, entry in enumerate(refs):
        ax = fig.add_subplot(total_rows, ncols, n_rows_p * ncols + j + 1)
        draw_cell(ax, entry, 'green')
    for j in range(n_r, ncols):
        ax = fig.add_subplot(total_rows, ncols, n_rows_p * ncols + j + 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"    → {out_path.name}")


# ─── Résumé texte ─────────────────────────────────────────────────────────────

def print_summary(texture_c, pires, refs, pred_multi, patches_meta):
    print(f"\n  t{texture_c} {TNAMES[texture_c]} ─────────────────────────")

    # Concentration sur une image
    stem_count = Counter(p['stem'] for p in pires)
    print(f"  Pires viennent de {len(stem_count)} image(s) distincte(s) :")
    for stem, cnt in stem_count.most_common(3):
        print(f"    {cnt}/{len(pires)}  {stem}")
    if len(stem_count) == 1:
        print("    → CONCENTRATION : tous les pires sur 1 seule image")

    # Confusion dominante (penche vers)
    penche_list = [p['penche_vers'] for p in pires if p.get('penche_vers') not in (None, -1)]
    if penche_list:
        dom_class, dom_cnt = Counter(penche_list).most_common(1)[0]
        print(f"  Confusion dominante : penchent vers t{dom_class} "
              f"{TNAMES.get(dom_class,'?')} ({dom_cnt}/{len(penche_list)})")

    # proba moyenne pires vs refs
    pm = np.mean([p['proba_ovr'] for p in pires])
    rm = np.mean([p['proba_ovr'] for p in refs]) if refs else float('nan')
    print(f"  Proba OvR — pires={pm:.3f}  refs={rm:.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═══ DIAGNOSTIC VISUEL — Pires patches LOIO ═══\n")

    q1         = json.load(open(Q1_JSON))
    best_by_t  = best_rep_per_texture(q1)
    patches_meta = parse_patches()
    N = len(patches_meta)

    print("Meilleur (ckpt, block) par texture depuis Q1 :")
    for t in TEXTURES:
        ck, rep, sc = best_by_t[t]
        tag = '[U]' if t in TUNIFORM else '[S]'
        note = '  [non-éval LOIO, best-guess]' if t == 5 else ''
        print(f"  t{t} {tag} {TNAMES[t]:<15}: {ck} | {rep:<18} recall={sc:.3f}{note}")

    # ── Chargement features multiclasse (block commun) ──────────────────────
    print(f"\nChargement features multiclasse ({MULTI_CKPT} | {MULTI_REP})...")
    X_multi = load_feat(MULTI_CKPT, MULTI_REP)
    print(f"  shape={X_multi.shape}")

    print(f"LOIO multiclasse ({MULTI_REP}) → 'penche vers'...")
    pred_multi = loio_multiclass(X_multi, patches_meta)

    # ── Par texture : LOIO OvR + proba + sélection pires/refs ───────────────
    all_entries = []   # pour référence croisée

    for t in TEXTURES:
        ck, rep, best_recall = best_by_t[t]
        print(f"\nTexture t{t} {TNAMES[t]}  ({ck} | {rep})...")

        X = load_feat(ck, rep)
        probas = loio_proba_ovr(X, patches_meta, t)

        # Collecte des patches de cette texture avec leur proba
        entries = []
        for i, p in enumerate(patches_meta):
            if p['texture'] != t: continue
            if np.isnan(probas[i]): continue
            entries.append({
                'idx':         i,
                'stem':        p['stem'],
                'row':         p['row'],
                'col':         p['col'],
                'texture':     t,
                'proba_ovr':   float(probas[i]),
                'penche_vers': int(pred_multi[i]) if pred_multi[i] != -1 else None,
                'ckpt':        ck,
                'rep':         rep,
            })

        all_entries.extend(entries)

        entries_sorted = sorted(entries, key=lambda e: e['proba_ovr'])
        pires = entries_sorted[:N_PIRES]
        refs  = entries_sorted[-N_REF:][::-1]

        # Plot
        fname = OUT_DIR / f't{t}_{TNAMES[t].replace(".","").replace(" ","_")}_pires.png'
        plot_texture(t, pires, refs, ck, rep, best_recall, fname)

        # Résumé texte
        print_summary(t, pires, refs, pred_multi, patches_meta)

    # ── Tableau récap ─────────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("TABLEAU — PIRES patches par texture")
    print("═"*70)
    print(f"{'t':>3}  {'Texture':<15}  {'Proba':>6}  {'Penche vers':<15}  "
          f"{'Image':<25}  {'(row,col)'}")
    print("─"*70)

    for t in TEXTURES:
        ck, rep, _ = best_by_t[t]
        X      = load_feat(ck, rep)
        probas = loio_proba_ovr(X, patches_meta, t)
        entries = sorted(
            [{'idx': i, 'stem': p['stem'], 'row': p['row'], 'col': p['col'],
              'proba_ovr': float(probas[i]),
              'penche_vers': int(pred_multi[i]) if pred_multi[i] != -1 else None}
             for i, p in enumerate(patches_meta)
             if p['texture'] == t and not np.isnan(probas[i])],
            key=lambda e: e['proba_ovr']
        )
        for e in entries[:5]:
            penche = TNAMES.get(e['penche_vers'], '?') if e['penche_vers'] else '?'
            print(f"  {t:>2}  {TNAMES[t]:<15}  {e['proba_ovr']:>6.3f}  "
                  f"{penche:<15}  {e['stem'][-25:]:<25}  ({e['row']},{e['col']})")
        print()

    print(f"\n  → Sorties : {OUT_DIR}/")
    print("═══ Terminé ═══")


if __name__ == '__main__':
    main()
