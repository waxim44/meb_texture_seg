#!/usr/bin/env python3
"""
Meilleur bloc SAM par texture — Recall LP LOIO
Protocole strict : LOIO par image, PCA(50) train-only, LR balanced, recall test.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

H5   = Path("data/feature_database/database_meb_ouassim.h5")
OUT  = Path("output_ouassim")

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1:"Tot.homogène", 3:"Faisceaux", 4:"Filaments", 5:"Strat.rect",
            6:"Strat.sin",   7:"Granuleux", 9:"Trou"}
PCA_DIM  = 50
C        = 1.0
SEED     = 42

# ─── Chargement ───────────────────────────────────────────────────────────────
print("Chargement H5...")
with h5py.File(H5, "r") as f:
    all_cat   = f["metadata"]["category_ids"][:]
    all_imgs  = np.array([x.decode() for x in f["metadata"]["image_names"][:]])
    BLOCKS    = sorted(f["features"].keys())
    feats     = {b: f["features"][b][:] for b in BLOCKS}

mask = np.isin(all_cat, TEXTURES)
cat_ids   = all_cat[mask]
img_names = all_imgs[mask]
feats     = {b: feats[b][mask] for b in BLOCKS}
stems     = np.array([n.replace(".tif","") for n in img_names])

print(f"  {mask.sum()} patches | {len(BLOCKS)} blocs")

# ─── LP LOIO ──────────────────────────────────────────────────────────────────
def loio_recall(X, y_bin, stems):
    """LOIO par image, PCA train-only, LR balanced → recalls list."""
    recalls = []
    for stem in sorted(set(stems)):
        te = stems == stem
        tr = ~te
        if y_bin[te].sum() == 0:        # pas de positifs dans le test → skip
            continue
        if len(np.unique(y_bin[tr])) < 2:
            continue
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]
        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(C=C, class_weight="balanced",
                                 max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        tp = int(((pred==1)&(y_te==1)).sum())
        fn = int(((pred==0)&(y_te==1)).sum())
        recalls.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
    return np.array(recalls)

print("\nCal LP LOIO (tous blocs × toutes textures) ...")
results = {}   # results[texture][bloc] = (mean, std, n_folds)

for t in TEXTURES:
    y_bin = (cat_ids == t).astype(int)
    results[t] = {}
    for b in BLOCKS:
        r = loio_recall(feats[b], y_bin, stems)
        results[t][b] = (float(r.mean()) if len(r) else 0.0,
                         float(r.std())  if len(r) else 0.0,
                         len(r))
    best = max(BLOCKS, key=lambda b: (results[t][b][0], -results[t][b][1]))
    m, s, n = results[t][best]
    print(f"  t{t} {TNAMES[t]:<15} → {best:<14} recall={m:.3f}±{s:.3f}  ({n} folds)")

# ─── Sélection meilleur bloc ─────────────────────────────────────────────────
print("\nNOTE : meilleur bloc sélectionné a posteriori sur tous les blocs "
      "→ biais optimiste (max sur 20 blocs).")

best_per_tex = {}
for t in TEXTURES:
    best_b = max(BLOCKS, key=lambda b: (results[t][b][0], -results[t][b][1]))
    m, s, n = results[t][best_b]
    best_per_tex[t] = {"bloc": best_b, "mean": m, "std": s, "n_folds": n}

# ─── Tableau texte ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Texture':<16} {'Meilleur bloc':<16} {'Recall':>8} {'±std':>7} {'n_folds':>8}")
print("-"*70)
for t in sorted(best_per_tex, key=lambda t: -best_per_tex[t]["mean"]):
    d = best_per_tex[t]
    print(f"{TNAMES[t]:<16} {d['bloc']:<16} {d['mean']:>8.3f} {d['std']:>7.3f} {d['n_folds']:>8}")
print("="*70)

# ─── Plot ────────────────────────────────────────────────────────────────────
order = sorted(best_per_tex, key=lambda t: -best_per_tex[t]["mean"])
tex_labels = [TNAMES[t] for t in order]
means  = [best_per_tex[t]["mean"] for t in order]
stds   = [best_per_tex[t]["std"]  for t in order]
blocs  = [best_per_tex[t]["bloc"] for t in order]
colors = ["#2ecc71" if m>=0.80 else "#f39c12" if m>=0.60 else "#e74c3c"
          for m in means]

x = np.arange(len(order))

# Barres d'erreur asymétriques clippées à [0, 1]
means_arr = np.array(means)
stds_arr  = np.array(stds)
err_lo = np.minimum(stds_arr, means_arr)           # ne descend pas sous 0
err_hi = np.minimum(stds_arr, 1.0 - means_arr)     # ne monte pas au-dessus de 1
yerr = [err_lo, err_hi]

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors,
              edgecolor="white", linewidth=0.8,
              error_kw=dict(lw=1.5, capthick=1.5, ecolor="black"))

for bar, bloc, hi in zip(bars, blocs, err_hi):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + hi + 0.025,
            bloc, ha="center", va="bottom", fontsize=8, color="#333")

ax.set_xticks(x)
ax.set_xticklabels(tex_labels, fontsize=11)
ax.set_ylabel("Recall LOIO (one-vs-rest)", fontsize=11)
ax.set_ylim(0, 1.18)
ax.axhline(0.5, color="gray", lw=0.9, ls="--", alpha=0.6)
ax.set_title("Meilleur bloc SAM par texture — Recall LP LOIO", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
out = OUT / "meilleur_bloc_par_texture.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot sauvé : {out}")
