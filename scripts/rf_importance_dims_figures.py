from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/rf_gbm"

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
TOP_N = 12

arr = np.load(OUT / "importances_arrays.npz")

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

rows_summary = []

for block in BLOCKS:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric in zip(axes.ravel(), METRICS):

        imp = (arr[f"impurity__{block}__{metric}__sqrt"] +
               arr[f"impurity__{block}__{metric}__third"]) / 2
        w_ridge = np.abs(arr[f"ridge_w__{block}__{metric}"])
        ridge_top10 = set(np.argsort(-w_ridge)[:10].tolist())

        order = np.argsort(-imp)[:TOP_N]
        vals = imp[order]
        colors = ["#c1121f" if d in ridge_top10 else "#1f77b4" for d in order]

        y = np.arange(TOP_N)
        ax.barh(y, vals[::-1], color=[colors[::-1][i] for i in range(TOP_N)])
        ax.set_yticks(y)
        ax.set_yticklabels([f"dim {d}" for d in order[::-1]], fontsize=10)
        ax.set_xlabel("importance (impureté)")
        ax.set_title(metric)

        for d in order:
            rows_summary.append(dict(block=block, metric=metric, dim=int(d),
                                      importance_impurete=float(imp[d]),
                                      dans_top10_ridge=d in ridge_top10))

    axes.ravel()[-1].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
               plt.Rectangle((0, 0), 1, 1, color="#c1121f")]
    axes.ravel()[-1].legend(handles, ["dimension hors top-10 |w| Ridge", "dimension aussi dans le top-10 |w| Ridge"],
                             loc="center", fontsize=12, frameon=False)
    fig.suptitle(f"Dimensions les plus utilisées pour les découpages — {block}", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_dims_decoupage_{block}.png", bbox_inches="tight")
    plt.close(fig)

print("Figures ecrites :", [f"fig_dims_decoupage_{b}.png" for b in BLOCKS])

summary = pd.DataFrame(rows_summary)
summary.to_csv(OUT / "dims_decoupage_top12.csv", index=False)
print("Ecrit :", OUT / "dims_decoupage_top12.csv")


print("\n=== Dimensions recurrentes (top-12 sur plusieurs metriques), par bloc ===")
recurrence_rows = []
for block in BLOCKS:
    sub = summary[summary.block == block]
    counts = sub.groupby("dim").size().sort_values(ascending=False)
    recurrent = counts[counts >= 2]
    print(f"\n{block} : {len(recurrent)} dimensions apparaissent dans le top-12 d'au moins 2 metriques")
    for dim, n in recurrent.items():
        metrics_concerned = sorted(sub[sub.dim == dim].metric.tolist())
        recurrence_rows.append(dict(block=block, dim=int(dim), n_metriques=int(n),
                                     metriques=",".join(metrics_concerned)))
    if len(recurrent):
        print(recurrent.head(10).to_string())

recurrence_df = pd.DataFrame(recurrence_rows)
recurrence_df.to_csv(OUT / "dims_recurrentes_par_bloc.csv", index=False)
print("\nEcrit :", OUT / "dims_recurrentes_par_bloc.csv")
