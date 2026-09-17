from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/mlp_vs_ridge"

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]

rec = pd.read_csv(OUT / "resultats_bruts.csv")


mlp = rec[rec.model.str.startswith("MLP")].copy()
mlp_per_fold = (mlp.groupby(["block", "metric", "dropout", "fold"])[["r2", "rmse", "medae"]]
                .mean().reset_index())
mlp_seed_std = (mlp.groupby(["block", "metric", "dropout", "fold"])[["r2", "rmse", "medae"]]
                .std().reset_index().rename(columns={"r2": "r2_seedstd", "rmse": "rmse_seedstd", "medae": "medae_seedstd"}))

ridge = rec[rec.model == "Ridge"][["block", "metric", "fold", "r2", "rmse", "medae"]].copy()


def agg(df, group_cols):
    return df.groupby(group_cols)[["r2", "rmse", "medae"]].agg(["mean", "std"])

ridge_agg = agg(ridge, ["block", "metric"])
mlp_agg = agg(mlp_per_fold, ["block", "metric", "dropout"])
seed_stability = mlp_seed_std.groupby(["block", "metric", "dropout"])[["r2_seedstd"]].mean()


rows = []
for b in BLOCKS:
    for m in METRICS:
        r2r = ridge_agg.loc[(b, m), ("r2", "mean")]; r2r_s = ridge_agg.loc[(b, m), ("r2", "std")]
        rmr = ridge_agg.loc[(b, m), ("rmse", "mean")]; rmr_s = ridge_agg.loc[(b, m), ("rmse", "std")]
        mer = ridge_agg.loc[(b, m), ("medae", "mean")]; mer_s = ridge_agg.loc[(b, m), ("medae", "std")]
        row = dict(block=b, metric=m,
                   R2_Ridge=r2r, R2_Ridge_std=r2r_s,
                   RMSE_Ridge=rmr, RMSE_Ridge_std=rmr_s,
                   MedAE_Ridge=mer, MedAE_Ridge_std=mer_s)
        for p in [0.3, 0.5]:
            r2m_ = mlp_agg.loc[(b, m, p), ("r2", "mean")]; r2m_s = mlp_agg.loc[(b, m, p), ("r2", "std")]
            rmm_ = mlp_agg.loc[(b, m, p), ("rmse", "mean")]; rmm_s = mlp_agg.loc[(b, m, p), ("rmse", "std")]
            mem_ = mlp_agg.loc[(b, m, p), ("medae", "mean")]; mem_s = mlp_agg.loc[(b, m, p), ("medae", "std")]
            seedstd = seed_stability.loc[(b, m, p), "r2_seedstd"]
            row[f"R2_MLP{p}"] = r2m_; row[f"R2_MLP{p}_std"] = r2m_s
            row[f"RMSE_MLP{p}"] = rmm_; row[f"RMSE_MLP{p}_std"] = rmm_s
            row[f"MedAE_MLP{p}"] = mem_; row[f"MedAE_MLP{p}_std"] = mem_s
            row[f"R2_MLP{p}_seedstd"] = seedstd
        rows.append(row)

table = pd.DataFrame(rows)
table["gap_best_MLP_minus_Ridge_R2"] = table[["R2_MLP0.3", "R2_MLP0.5"]].max(axis=1) - table["R2_Ridge"]
table.to_csv(OUT / "comparaison_mlp_ridge.csv", index=False)

pd.set_option("display.width", 220)
disp_cols = ["block", "metric", "R2_Ridge", "R2_MLP0.3", "R2_MLP0.5", "gap_best_MLP_minus_Ridge_R2"]
print(table[disp_cols].round(3).to_string(index=False))


plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

for b in BLOCKS:
    sub = table[table.block == b].set_index("metric").loc[METRICS]
    x = np.arange(len(METRICS)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w, sub["R2_Ridge"], w, yerr=sub["R2_Ridge_std"], capsize=3,
           color="#333333", label="Ridge")
    ax.bar(x, sub["R2_MLP0.3"], w, yerr=sub["R2_MLP0.3_std"], capsize=3,
           color="#1f77b4", label="MLP (dropout 0.3)")
    ax.bar(x + w, sub["R2_MLP0.5"], w, yerr=sub["R2_MLP0.5_std"], capsize=3,
           color="#c1121f", label="MLP (dropout 0.5)")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(METRICS)
    ax.set_ylabel("R²")
    ax.set_title(f"Ridge vs MLP — {b}")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(OUT / f"fig_ridge_vs_mlp_{b}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig_ridge_vs_mlp_{b}.png", bbox_inches="tight")
    plt.close(fig)

print("\nFigures ecrites :", [f"fig_ridge_vs_mlp_{b}.png" for b in BLOCKS])
