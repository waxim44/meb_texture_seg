import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/dinov2_comparison"
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
CG, CS = "#b55e07", "#1f77b4"
PRIMARY_MODEL = "dinov2_vits14_reg"
BEST_LAYER = "layer_03"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

with open(OUT / "oof_dinov2.pkl", "rb") as fh:
    oof_dinov2 = pickle.load(fh)

res_sam = pd.read_csv(ROOT / "outputs/rsa_granulometrie/ridge_residuals/residus_oof.csv")


rec = pd.read_csv(OUT / "ridge_dinov2_par_couche.csv")
primary = rec[(rec.model == PRIMARY_MODEL) & (rec.layer == BEST_LAYER)].set_index("metric")


sam_r2 = pd.read_csv(ROOT / "outputs/rsa_granulometrie/ridge_w_alignment/r2_par_metrique_bloc.csv")
sam_b4 = sam_r2[sam_r2.block == "block_4"].set_index("metric")


sam_err = res_sam[res_sam.block == "block_4"].copy()
sam_err_agg = sam_err.groupby(["metric", "fold"]).apply(
    lambda g: pd.Series({"rmse": np.sqrt((g.residual**2).mean()), "medae": g.residual.abs().median()})
).groupby("metric").mean()

print(f"=== Comparaison R2/RMSE/MedAE (moyenne +/- std sur 5 folds, MEME agregation des deux cotes) ===")
print(f"    DINOv2 : {PRIMARY_MODEL}, {BEST_LAYER}  |  TextureSAM : block_4")
comp = pd.DataFrame({
    "R2_DINOv2": primary["R2_mean"], "R2_DINOv2_std": primary["R2_std"],
    "RMSE_DINOv2": primary["RMSE_mean"], "MedAE_DINOv2": primary["MedAE_mean"],
    "R2_TextureSAM": sam_b4["R2_mean"], "R2_TextureSAM_std": sam_b4["R2_std"],
    "RMSE_TextureSAM": sam_err_agg["rmse"], "MedAE_TextureSAM": sam_err_agg["medae"],
}).loc[METRICS]
comp["ecart_R2_DINOv2_moins_SAM"] = comp["R2_DINOv2"] - comp["R2_TextureSAM"]
print(comp.round(3).to_string())
comp.to_csv(OUT / "comparaison_dinov2_vs_sam_primaire.csv")


for m in METRICS:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))


    y_true, y_pred, tex = oof_dinov2[(PRIMARY_MODEL, BEST_LAYER, m)]
    ax = axes[0]
    for t, c in [("Granuleux", CG), ("Sableux", CS)]:
        mask = tex == t
        ax.scatter(y_true[mask], y_pred[mask], s=22, color=c, alpha=0.65, edgecolor="none", label=t)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(f"{m} réel"); ax.set_ylabel(f"{m} prédit")
    ax.set_title(f"DINOv2 ({BEST_LAYER})")
    ax.legend(loc="upper left")


    sub = res_sam[(res_sam.block == "block_4") & (res_sam.metric == m)]
    ax = axes[1]
    for t, c in [("Granuleux", CG), ("Sableux", CS)]:
        mask = sub.texture == t
        ax.scatter(sub.y_true[mask], sub.y_pred_oof[mask], s=22, color=c, alpha=0.65,
                   edgecolor="none", label=t)
    lo2 = min(sub.y_true.min(), sub.y_pred_oof.min()); hi2 = max(sub.y_true.max(), sub.y_pred_oof.max())
    ax.plot([lo2, hi2], [lo2, hi2], "--", color="black", lw=1.3)
    ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2); ax.set_aspect("equal")
    ax.set_xlabel(f"{m} réel"); ax.set_ylabel(f"{m} prédit")
    ax.set_title("TextureSAM (block_4)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / f"fig_predit_reel_{m}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig_predit_reel_{m}.png", bbox_inches="tight")
    plt.close(fig)

print("\nFigures ecrites :", [f"fig_predit_reel_{m}.png" for m in METRICS])
