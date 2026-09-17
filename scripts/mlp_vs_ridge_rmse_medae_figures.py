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

table = pd.read_csv(OUT / "comparaison_mlp_ridge.csv")

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

def plot_metric(err_name, ylabel):
    for b in BLOCKS:
        sub = table[table.block == b].set_index("metric").loc[METRICS]
        x = np.arange(len(METRICS)); w = 0.26
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x - w, sub[f"{err_name}_Ridge"], w, yerr=sub[f"{err_name}_Ridge_std"], capsize=3,
               color="#333333", label="Ridge")
        ax.bar(x, sub[f"{err_name}_MLP0.3"], w, yerr=sub[f"{err_name}_MLP0.3_std"], capsize=3,
               color="#1f77b4", label="MLP (dropout 0.3)")
        ax.bar(x + w, sub[f"{err_name}_MLP0.5"], w, yerr=sub[f"{err_name}_MLP0.5_std"], capsize=3,
               color="#c1121f", label="MLP (dropout 0.5)")
        ax.set_xticks(x); ax.set_xticklabels(METRICS)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Ridge vs MLP — {b}")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        fig.savefig(OUT / f"fig_{err_name.lower()}_ridge_vs_mlp_{b}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"fig_{err_name.lower()}_ridge_vs_mlp_{b}.png", bbox_inches="tight")
        plt.close(fig)


plot_metric("RMSE", "RMSE")
plot_metric("MedAE", "MedAE")
print("Figures ecrites :", [f"fig_{e}_ridge_vs_mlp_{b}.png" for e in ['rmse', 'medae'] for b in BLOCKS])


rows = []
for _, row in table.iterrows():
    best_mlp_rmse = min(row["RMSE_MLP0.3"], row["RMSE_MLP0.5"])
    best_mlp_medae = min(row["MedAE_MLP0.3"], row["MedAE_MLP0.5"])
    rows.append(dict(
        block=row.block, metric=row.metric,
        RMSE_Ridge=row.RMSE_Ridge, RMSE_MLP_best=best_mlp_rmse,
        gagnant_RMSE="Ridge" if row.RMSE_Ridge < best_mlp_rmse else "MLP",
        MedAE_Ridge=row.MedAE_Ridge, MedAE_MLP_best=best_mlp_medae,
        gagnant_MedAE="Ridge" if row.MedAE_Ridge < best_mlp_medae else "MLP",
    ))
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "resume_rmse_medae_ridge_vs_mlp.csv", index=False)

print("\n=== Qui gagne (RMSE / MedAE, plus petit = mieux) ===")
print(summary[["block", "metric", "RMSE_Ridge", "RMSE_MLP_best", "gagnant_RMSE",
                "MedAE_Ridge", "MedAE_MLP_best", "gagnant_MedAE"]].round(3).to_string(index=False))
print(f"\nRidge gagne en RMSE : {(summary.gagnant_RMSE == 'Ridge').sum()}/{len(summary)}")
print(f"Ridge gagne en MedAE : {(summary.gagnant_MedAE == 'Ridge').sum()}/{len(summary)}")
