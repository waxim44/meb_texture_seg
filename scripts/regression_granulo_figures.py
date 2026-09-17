from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie"

METRICS = ["D50", "D10", "D90", "Span", "Skewness", "sigma_local"]


FIG_METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
BLOCKS = [f"block_{i}" for i in range(16)] + [f"stage_{i}_fpn" for i in (1, 2, 3, 4)]
XT = [str(i) for i in range(16)] + ["s1", "s2", "s3", "s4"]
CG, CS = "#b55e07", "#1f77b4"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


r2 = pd.read_csv(OUT / "R2_ridge_metrique_x_bloc.csv", index_col=0)
r2svr = pd.read_csv(OUT / "R2_svr_metrique_x_bloc.csv", index_col=0)
summ = pd.read_csv(OUT / "resume_meilleur_bloc.csv", index_col=0)


x = np.arange(len(BLOCKS))
fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.get_cmap("tab10")
for k, m in enumerate(FIG_METRICS):
    lw, z = (3.4, 5) if m == "D50" else (1.9, 3)
    ax.plot(x, r2.loc[m, BLOCKS].values, "-o", ms=5, lw=lw,
            color=cmap(METRICS.index(m)), label=m, zorder=z)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(XT)
ax.set_xlabel("Bloc")
ax.set_ylabel("R²")
ax.set_title("Granulométrie prédite depuis le latent — R² par couche")
ax.set_ylim(-0.02, 0.82)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
save(fig, "fig_profil_R2_par_couche")


oof_path = OUT / "oof_ridge.npz"
if oof_path.exists():
    d = np.load(oof_path, allow_pickle=True)
    best = str(d["best_block_D50"])
    Y = d["Y"]; ids = d["id"]
    j = METRICS.index("D50")
    pred = d[best][:, j]; true = Y[:, j]

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    lo = min(true.min(), pred.min()) - 0.5
    hi = max(true.max(), pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.3, zorder=1)
    ax.scatter(true[ids == 7], pred[ids == 7], s=32, color=CG, alpha=0.75, label="Granuleux", edgecolor="none")
    ax.scatter(true[ids == 8], pred[ids == 8], s=32, color=CS, alpha=0.75, label="Sableux", edgecolor="none")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("D50 réel  (px)")
    ax.set_ylabel("D50 prédit  (px)")
    ax.set_title(f"D50 — prédit vs réel  ({best})")
    ax.legend(loc="upper left")
    save(fig, "fig_predit_vs_reel_D50")
else:
    print(f"!!! {oof_path} absent -- figure 'predit vs reel D50' non regeneree (PNG existant conserve tel quel).")


best_blk = summ["meilleur_bloc_Ridge"].to_dict()
rid = [r2.loc[m, best_blk[m]] for m in FIG_METRICS]
svr = [r2svr.loc[m, best_blk[m]] for m in FIG_METRICS]
xm = np.arange(len(FIG_METRICS)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(xm - w/2, rid, w, color="#333333", label="Ridge")
ax.bar(xm + w/2, svr, w, color="#c1121f", label="SVR RBF")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(xm)
ax.set_xticklabels([f"{m}\n({best_blk[m]})" for m in FIG_METRICS], fontsize=11)
ax.set_ylabel("R²")
ax.set_ylim(0, 0.86)
ax.set_title("Ridge vs SVR RBF")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
save(fig, "fig_ridge_vs_svr")

print("Figures ecrites (PDF+PNG) dans", OUT)
for p in sorted(OUT.glob("fig_*.pdf")):
    print("  ", p.stem)
