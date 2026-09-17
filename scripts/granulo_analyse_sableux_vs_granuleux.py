from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/granulometrie_exploration"
IMG_DIR = ROOT / "Image_Ouassim"

METRICS = ["D50", "D10", "D90", "Span", "Skewness", "sigma_local"]
CG, CS = "#b55e07", "#1f77b4"
RNG = np.random.default_rng(0)

df = pd.read_csv(OUT / "granulo_patches.csv")
G = df[df.id == 7].reset_index(drop=True)
S = df[df.id == 8].reset_index(drop=True)
print(f"Granuleux={len(G)}  Sableux={len(S)}")


def overlapping_coefficient(a, b, nbins=40):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, nbins + 1)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    w = np.diff(bins)
    return float(np.sum(np.minimum(pa, pb) * w))


def rank_biserial(a, b):
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    delta = 2.0 * U / (n1 * n2) - 1.0
    return U, p, delta


rows = []
for m in METRICS:
    for name, d in [("Granuleux", G[m]), ("Sableux", S[m])]:
        rows.append(dict(metrique=m, texture=name, moyenne=d.mean(), mediane=d.median(),
                         ecart_type=d.std(), q25=d.quantile(.25), q75=d.quantile(.75),
                         min=d.min(), max=d.max()))
desc = pd.DataFrame(rows)
desc.to_csv(OUT / "stats_descriptives.csv", index=False)
print("\n=== Etape 2B : stats descriptives ===")
print(desc.round(3).to_string(index=False))


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.ravel(), METRICS):
    lo, hi = df[m].min(), df[m].max()
    bins = np.linspace(lo, hi, 36)
    ax.hist(G[m], bins=bins, color=CG, alpha=0.55, label=f"Granuleux (n={len(G)})", density=True)
    ax.hist(S[m], bins=bins, color=CS, alpha=0.55, label=f"Sableux (n={len(S)})", density=True)
    ax.axvline(G[m].median(), color=CG, ls="--", lw=1.5)
    ax.axvline(S[m].median(), color=CS, ls="--", lw=1.5)
    ax.set_title(m); ax.set_xlabel("rayon (px)" if m in ("D10", "D50", "D90") else m)
    ax.set_ylabel("densité"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
fig.suptitle("Granulométrie — distributions Granuleux vs Sableux", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "fig_histogrammes_metriques.png", dpi=140)
plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(df.D50.min(), df.D50.max(), 40)
ax.hist(G.D50, bins=bins, color=CG, alpha=0.55, label=f"Granuleux (n={len(G)})", density=True)
ax.hist(S.D50, bins=bins, color=CS, alpha=0.55, label=f"Sableux (n={len(S)})", density=True)
ax.axvline(G.D50.median(), color=CG, ls="--", lw=2, label=f"médiane Granuleux = {G.D50.median():.2f}")
ax.axvline(S.D50.median(), color=CS, ls="--", lw=2, label=f"médiane Sableux = {S.D50.median():.2f}")
ax.set_xlabel("D50 (rayon, px)"); ax.set_ylabel("densité")
ax.spines[["top", "right"]].set_visible(False); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig_D50_histogramme.png", dpi=140)
plt.close(fig)


print("\n=== Etape 3 : tests de separation (Granuleux vs Sableux) ===")
res = []
for m in METRICS:
    a, b = G[m].values, S[m].values
    U, p, delta = rank_biserial(a, b)
    ovl = overlapping_coefficient(a, b)
    res.append(dict(metrique=m, p_mann_whitney=p, taille_effet_rb=delta,
                    abs_effet=abs(delta), recouvrement_OVL=ovl,
                    moy_Granuleux=a.mean(), moy_Sableux=b.mean()))
tests = pd.DataFrame(res).sort_values("abs_effet", ascending=False)
tests.to_csv(OUT / "tests_separation.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
print(tests.drop(columns="abs_effet").to_string(index=False))


print("\n=== Etape 4 : SVM lineaire balanced (6 metriques standardisees) ===")
X = df[METRICS].values
y = (df.id.values == 7).astype(int)
print(f"  classes : Granuleux={y.sum()}  Sableux={(1-y).sum()}  (class_weight='balanced' OBLIGATOIRE)")

pipe = make_pipeline(StandardScaler(),
                     SVC(kernel="linear", class_weight="balanced", random_state=0))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_validate(pipe, X, y, cv=cv,
                        scoring=["roc_auc", "balanced_accuracy"], return_train_score=False)
auc_m, auc_s = scores["test_roc_auc"].mean(), scores["test_roc_auc"].std()
bacc_m, bacc_s = scores["test_balanced_accuracy"].mean(), scores["test_balanced_accuracy"].std()
print(f"  AUC (5-fold stratifie)         : {auc_m:.3f} ± {auc_s:.3f}")
print(f"  balanced accuracy (complement) : {bacc_m:.3f} ± {bacc_s:.3f}")
print(f"  folds AUC : {np.round(scores['test_roc_auc'], 3).tolist()}")

pipe.fit(X, y)
coef = pipe.named_steps["svc"].coef_.ravel()
imp = pd.DataFrame({"metrique": METRICS, "coef_SVM_standardise": coef,
                    "abs": np.abs(coef)}).sort_values("abs", ascending=False)
imp.to_csv(OUT / "svm_coefficients.csv", index=False)
print("  coefficients (espace standardise, +->Granuleux) :")
print(imp.drop(columns="abs").to_string(index=False))

with open(OUT / "svm_resume.txt", "w") as fh:
    fh.write(f"SVM lineaire, class_weight='balanced', StandardScaler, 5-fold stratifie\n")
    fh.write(f"AUC = {auc_m:.3f} +/- {auc_s:.3f}\n")
    fh.write(f"balanced_accuracy = {bacc_m:.3f} +/- {bacc_s:.3f}\n")
    fh.write(f"folds AUC = {np.round(scores['test_roc_auc'],3).tolist()}\n")


def load_crop(r):
    big = np.array(Image.open(IMG_DIR / r["image"]).convert("L"))
    return big[int(r.y_min):int(r.y_max), int(r.x_min):int(r.x_max)]


fig, axes = plt.subplots(2, 3, figsize=(11, 8))
gi = RNG.choice(len(G), 3, replace=False)
si = RNG.choice(len(S), 3, replace=False)
for j, k in enumerate(gi):
    r = G.iloc[k]; axes[0, j].imshow(load_crop(r), cmap="gray")
    axes[0, j].set_title(f"Granuleux  D50={r.D50:.2f}", color=CG, fontsize=10)
    axes[0, j].axis("off")
for j, k in enumerate(si):
    r = S.iloc[k]; axes[1, j].imshow(load_crop(r), cmap="gray")
    axes[1, j].set_title(f"Sableux  D50={r.D50:.2f}", color=CS, fontsize=10)
    axes[1, j].axis("off")
fig.suptitle("Étape 5A — crops représentatifs", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig_grille_crops.png", dpi=140)
plt.close(fig)


lo = max(G.D50.min(), S.D50.min())
hi = min(G.D50.max(), S.D50.max())
Gz = G[(G.D50 >= lo) & (G.D50 <= hi)]
Sz = S[(S.D50 >= lo) & (S.D50 <= hi)]
print(f"\n=== Etape 5B : zone de recouvrement D50 = [{lo:.2f}, {hi:.2f}] px ===")
print(f"  Granuleux dans la zone : {len(Gz)}/{len(G)}   Sableux dans la zone : {len(Sz)}/{len(S)}")


diffs = np.abs(Gz.D50.values[:, None] - Sz.D50.values[None, :])
ig, is_ = np.unravel_index(np.argmin(diffs), diffs.shape)
pair_close = (Gz.iloc[ig], Sz.iloc[is_])

pair_far = (Gz.loc[Gz.D50.idxmax()], Sz.loc[Sz.D50.idxmin()])

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
for col, (rg, rs) in enumerate([pair_close, pair_far]):
    axes[0, col].imshow(load_crop(rg), cmap="gray")
    axes[0, col].set_title(f"Granuleux  D50={rg.D50:.2f}", color=CG, fontsize=12)
    axes[0, col].axis("off")
    axes[1, col].imshow(load_crop(rs), cmap="gray")
    axes[1, col].set_title(f"Sableux  D50={rs.D50:.2f}", color=CS, fontsize=12)
    axes[1, col].axis("off")
fig.tight_layout(); fig.savefig(OUT / "fig_paires_recouvrement.png", dpi=140)
fig.savefig(OUT / "fig_paires_recouvrement.pdf")
plt.close(fig)

print(f"  paire chevauchante  : Granuleux uid={int(pair_close[0].patch_uid)} D50={pair_close[0].D50:.2f}"
      f" | Sableux uid={int(pair_close[1].patch_uid)} D50={pair_close[1].D50:.2f}"
      f" | |ΔD50|={abs(pair_close[0].D50-pair_close[1].D50):.3f}")
print(f"  paire eloignee (zone): Granuleux uid={int(pair_far[0].patch_uid)} D50={pair_far[0].D50:.2f}"
      f" | Sableux uid={int(pair_far[1].patch_uid)} D50={pair_far[1].D50:.2f}"
      f" | |ΔD50|={abs(pair_far[0].D50-pair_far[1].D50):.3f}")

print("\nFigures ecrites dans", OUT)
for p in sorted(OUT.glob("fig_*.png")):
    print("  ", p.name)
