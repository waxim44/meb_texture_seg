from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/variance_sweep"

METRICS = ["D50", "D10", "D90", "Span", "Skewness", "sigma_local"]


FIG_METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
LEVELS = ["50%", "75%", "95%", "brut (100%)"]

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


r2m = pd.read_csv(OUT / "R2_mean_par_niveau.csv", index_col=0).loc[LEVELS]
r2s = pd.read_csv(OUT / "R2_std_par_niveau.csv", index_col=0).loc[LEVELS]
rmsem = pd.read_csv(OUT / "RMSE_mean_par_niveau.csv", index_col=0).loc[LEVELS]
rmses = pd.read_csv(OUT / "RMSE_std_par_niveau.csv", index_col=0).loc[LEVELS]
medm = pd.read_csv(OUT / "MedAE_mean_par_niveau.csv", index_col=0).loc[LEVELS]
meds = pd.read_csv(OUT / "MedAE_std_par_niveau.csv", index_col=0).loc[LEVELS]
ncomp = pd.read_csv(OUT / "n_components_par_niveau.csv", index_col=0).loc[LEVELS]

x = np.arange(len(LEVELS))
cmap = plt.get_cmap("tab10")


fig, ax = plt.subplots(figsize=(9, 6))
for k, m in enumerate(FIG_METRICS):
    lw = 3.2 if m == "D50" else 1.9
    ax.errorbar(x, r2m[m].values, yerr=r2s[m].values, fmt="-o", ms=6, lw=lw,
                capsize=4, color=cmap(METRICS.index(m)), label=m)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(LEVELS)
ax.set_xlabel("Variance PCA conservée")
ax.set_ylabel("R²")
ax.set_title("Granulométrie prédite — R² vs niveau de variance PCA")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
save(fig, "fig_R2_vs_variance")


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.ravel(), FIG_METRICS):
    ax.errorbar(x, r2m[m].values, yerr=r2s[m].values, fmt="-o", ms=6, lw=2,
                capsize=4, color=cmap(METRICS.index(m)))
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(LEVELS, fontsize=11)
    ax.set_title(m)
    ax.set_ylabel("R²")
axes.ravel()[-1].axis("off")
fig.suptitle("R² vs niveau de variance PCA — par métrique", fontsize=16)
fig.tight_layout()
save(fig, "fig_R2_grille_par_metrique")


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.ravel(), FIG_METRICS):
    ax.errorbar(x, rmsem[m].values, yerr=rmses[m].values, fmt="-o", ms=6, lw=2,
                capsize=4, color=cmap(METRICS.index(m)))
    ax.set_xticks(x); ax.set_xticklabels(LEVELS, fontsize=11)
    ax.set_title(m)
    ax.set_ylabel("RMSE")
axes.ravel()[-1].axis("off")
fig.suptitle("RMSE vs niveau de variance PCA — par métrique", fontsize=16)
fig.tight_layout()
save(fig, "fig_RMSE_grille_par_metrique")


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.ravel(), FIG_METRICS):
    ax.errorbar(x, medm[m].values, yerr=meds[m].values, fmt="-o", ms=6, lw=2,
                capsize=4, color=cmap(METRICS.index(m)))
    ax.set_xticks(x); ax.set_xticklabels(LEVELS, fontsize=11)
    ax.set_title(m)
    ax.set_ylabel("MedAE")
axes.ravel()[-1].axis("off")
fig.suptitle("MedAE vs niveau de variance PCA — par métrique", fontsize=16)
fig.tight_layout()
save(fig, "fig_MedAE_grille_par_metrique")


fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(x, ncomp["n_components_mean"].values, yerr=ncomp["n_components_std"].values,
       color="#333333", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(LEVELS)
ax.set_ylabel("Nombre de composantes PCA retenues")
ax.set_title("Dimension retenue vs niveau de variance (block_6)")
ax.set_yscale("log")
save(fig, "fig_n_components_vs_variance")

print("Figures ecrites dans", OUT)
for p in sorted(OUT.glob("fig_*.pdf")):
    print("  ", p.stem)
