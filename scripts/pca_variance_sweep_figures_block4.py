from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/variance_sweep_block4"

METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
LEVELS = ["50%", "75%", "95%", "brut (100%)"]
x = np.arange(len(LEVELS))
cmap = plt.get_cmap("tab10")
METRIC_COLOR = {m: cmap(i) for i, m in enumerate(METRICS)}

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
    "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

r2m = pd.read_csv(OUT / "R2_mean_par_niveau.csv", index_col=0).reindex(LEVELS)
r2s = pd.read_csv(OUT / "R2_std_par_niveau.csv", index_col=0).reindex(LEVELS)
rmsem = pd.read_csv(OUT / "RMSE_mean_par_niveau.csv", index_col=0).reindex(LEVELS)
rmses = pd.read_csv(OUT / "RMSE_std_par_niveau.csv", index_col=0).reindex(LEVELS)
medm = pd.read_csv(OUT / "MedAE_mean_par_niveau.csv", index_col=0).reindex(LEVELS)
meds = pd.read_csv(OUT / "MedAE_std_par_niveau.csv", index_col=0).reindex(LEVELS)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 6.5))
for m in METRICS:
    lw = 3.4 if m == "D50" else 2.0
    ax.errorbar(x, r2m[m].values, yerr=r2s[m].values, fmt="-o", ms=7, lw=lw,
                capsize=4, color=METRIC_COLOR[m], label=m)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(LEVELS)
ax.set_xlabel("Variance PCA conservée")
ax.set_ylabel("R²")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
save(fig, "fig_R2_vs_variance")


for values_mean, values_std, ylabel, fname in [
    (rmsem, rmses, "RMSE", "fig_RMSE_grille_par_metrique"),
    (medm, meds, "MedAE", "fig_MedAE_grille_par_metrique"),
]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, m in zip(axes.ravel(), METRICS):
        ax.errorbar(x, values_mean[m].values, yerr=values_std[m].values, fmt="-o",
                    ms=7, lw=2.2, capsize=4, color=METRIC_COLOR[m])
        ax.set_xticks(x); ax.set_xticklabels(LEVELS, fontsize=12)
        ax.set_title(m)
        ax.set_ylabel(ylabel)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    save(fig, fname)

print("Figures ecrites dans", OUT)
for p in sorted(OUT.glob("fig_*.png")):
    print("  ", p.name)
