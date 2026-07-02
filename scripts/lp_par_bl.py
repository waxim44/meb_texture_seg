#!/usr/bin/env python3
"""
Séparabilité LP LOIO propre — par texture, par bloc, TOUS les blocs.
Protocole sain et robuste :
  - LOIO PAR IMAGE (jamais des patches d'une même image en train ET test)
  - PCA fittée sur le TRAIN du fold uniquement, appliquée au test
  - LP régularisé, class_weight='balanced'
  - recall calculé sur le TEST (image retirée), jamais sur le train
  - one-vs-rest par texture
Sortie : recall (moyenne ± std LOIO) de chaque texture pour chaque bloc.
"""

import h5py
import numpy as np
import csv
import time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ROOT     = Path('/home/aidouni/meb_texture_seg')
H5_PATH  = ROOT / 'data/feature_database/database_meb_ouassim.h5'
OUT_CSV  = ROOT / 'lp_loio_par_texture_par_block.csv'

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1:'Tot.homogène', 3:'Faisceaux', 4:'Filaments', 5:'Strat.rect',
            6:'Strat.sin', 7:'Granuleux', 9:'Trou'}

PCA_DIM  = 50
LP_C     = 1.0          # régularisation (capacité limitée, évite la mémorisation)
SEED     = 42

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
with h5py.File(H5_PATH, 'r') as f:
    all_cat  = f['metadata']['category_ids'][:]
    all_imgs = np.array([x.decode() for x in f['metadata']['image_names'][:]])
    # tous les blocs disponibles dans le H5
    BLOCKS   = list(f['features'].keys())
    feats_all = {b: f['features'][b][:] for b in BLOCKS}

# garder seulement les patches des textures étudiées
mask    = np.isin(all_cat, TEXTURES)
idx     = np.where(mask)[0]
cats    = all_cat[idx]
imgs    = all_imgs[idx]
feats   = {b: feats_all[b][idx] for b in BLOCKS}

print(f"{len(idx)} patches, {len(TEXTURES)} textures, {len(BLOCKS)} blocs")
for t in TEXTURES:
    n_img = len(set(imgs[cats == t]))
    print(f"  {TNAMES[t]:<15} N={ (cats==t).sum():>4}  images={n_img}")

# ─── LP LOIO ONE-VS-REST ──────────────────────────────────────────────────────
def loio_recall_ovr(X, y, images, texture):
    """
    Recall LOIO one-vs-rest pour une texture donnée.
    Retourne : liste des recalls (un par image contenant la texture).
    """
    y_bin  = (y == texture).astype(int)
    scores = []
    for img_test in sorted(set(images)):
        te = images == img_test
        tr = images != img_test
        # l'image test doit contenir des patches de la texture, sinon recall indéfini
        if y_bin[te].sum() == 0:
            continue
        # le train doit avoir les deux classes
        if len(np.unique(y_bin[tr])) < 2:
            continue

        X_tr, X_te = X[tr], X[te]
        # PCA fit sur TRAIN uniquement
        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=LP_C, class_weight='balanced',
                                 max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_bin[tr])
        pred = clf.predict(X_te)
        # recall de la classe positive (texture) sur le TEST
        rec = recall_score(y_bin[te], pred, pos_label=1, zero_division=0)
        scores.append(rec)
    return scores

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────
results = {}   # results[block][texture] = (mean, std, n_folds)

t0 = time.time()
for b_idx, b in enumerate(BLOCKS, 1):
    X = feats[b]
    results[b] = {}
    b_start = time.time()
    print(f"\n[{b_idx}/{len(BLOCKS)}] {b} ...", flush=True)
    for t in TEXTURES:
        scores = loio_recall_ovr(X, cats, imgs, t)
        if scores:
            m, s = np.mean(scores), np.std(scores)
            results[b][t] = (m, s, len(scores))
            print(f"  {TNAMES[t]:<15} recall={m:.3f} ± {s:.3f}  ({len(scores)} folds)", flush=True)
        else:
            results[b][t] = (float('nan'), float('nan'), 0)
            print(f"  {TNAMES[t]:<15} —", flush=True)
    print(f"  → bloc terminé en {time.time()-b_start:.1f}s  (total {time.time()-t0:.0f}s)", flush=True)

# ─── AFFICHAGE ────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("RECALL LP LOIO one-vs-rest — moyenne (± std) par texture et par bloc")
print("="*90)

# en-tête
header = f"{'Bloc':<16}" + "".join(f"{TNAMES[t][:9]:>11}" for t in TEXTURES)
print(header)
print("-"*len(header))
for b in BLOCKS:
    row = f"{b:<16}"
    for t in TEXTURES:
        m, s, n = results[b][t]
        row += f"{m:>6.2f}±{s:>3.2f}" if not np.isnan(m) else f"{'—':>11}"
    print(row)

# ─── CSV ──────────────────────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['block', 'texture', 'texture_nom', 'recall_mean', 'recall_std', 'n_folds'])
    for b in BLOCKS:
        for t in TEXTURES:
            m, s, n = results[b][t]
            w.writerow([b, t, TNAMES[t],
                        round(m, 4) if not np.isnan(m) else '',
                        round(s, 4) if not np.isnan(s) else '', n])

# ─── MEILLEUR BLOC PAR TEXTURE ────────────────────────────────────────────────
print("\n" + "="*90)
print("MEILLEUR BLOC PAR TEXTURE (recall test le plus haut, std bas si ex-æquo)")
print("="*90)
for t in TEXTURES:
    best_b, best_m, best_s = None, -1, None
    for b in BLOCKS:
        m, s, n = results[b][t]
        if np.isnan(m):
            continue
        # sélection : recall haut, puis std bas
        if m > best_m or (m == best_m and s < best_s):
            best_b, best_m, best_s = b, m, s
    print(f"  {TNAMES[t]:<15} → {best_b:<16} recall={best_m:.3f} ± {best_s:.3f}")

print(f"\n✓ CSV écrit : {OUT_CSV}")