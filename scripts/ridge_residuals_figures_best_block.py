from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/ridge_residuals"
R2_PATH = ROOT / "outputs/rsa_granulometrie/ridge_w_alignment/r2_par_metrique_bloc.csv"

METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
CG, CS = "#b55e07", "#1f77b4"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

res = pd.read_csv(OUT / "residus_oof.csv")
images_sorted = sorted(res.image.unique())
img_to_idx = {im: i for i, im in enumerate(images_sorted)}

r2 = pd.read_csv(R2_PATH)
best_block = r2.loc[r2.groupby("metric")["R2_mean"].idxmax()].set_index("metric")["block"].to_dict()
print("Meilleur bloc par metrique :", best_block)

for m in METRICS:
    block = best_block[m]
    sub = res[(res.block == block) & (res.metric == m)].copy()
    pred, true, resid = sub.y_pred_oof.values, sub.y_true.values, sub.residual.values
    tex = sub.texture.values
    img_idx = sub.image.map(img_to_idx).values

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))


    ax = axes[0, 0]
    ax.scatter(pred, resid, s=22, color="#555555", alpha=0.55, edgecolor="none")
    order = np.argsort(pred)
    sm = lowess(resid[order], pred[order], frac=0.4, return_sorted=True)
    ax.plot(sm[:, 0], sm[:, 1], color="#c1121f", lw=2.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(f"{m} prédit"); ax.set_ylabel("Résidu (réel − prédit)")


    ax = axes[0, 1]
    ax.scatter(true, pred, s=22, color="#555555", alpha=0.55, edgecolor="none")
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.3)
    ax.set_xlabel(f"{m} réel"); ax.set_ylabel(f"{m} prédit")
    ax.set_aspect("equal")


    ax = axes[1, 0]
    for t, c in [("Granuleux", CG), ("Sableux", CS)]:
        mask = tex == t
        ax.scatter(pred[mask], resid[mask], s=22, color=c, alpha=0.65, edgecolor="none", label=t)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(f"{m} prédit"); ax.set_ylabel("Résidu")
    ax.legend(loc="upper right")


    ax = axes[1, 1]
    sc = ax.scatter(pred, resid, c=img_idx, cmap="turbo", s=22, alpha=0.75, edgecolor="none")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(f"{m} prédit"); ax.set_ylabel("Résidu")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("image (indice)")

    fig.suptitle(f"{m} — meilleur bloc ({block})", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_residus_best_{m}.png", bbox_inches="tight")
    fig.savefig(OUT / f"fig_residus_best_{m}.pdf", bbox_inches="tight")
    plt.close(fig)

print("Figures ecrites dans", OUT)
for p in sorted(OUT.glob("fig_residus_best_*.png")):
    print("  ", p.name)
