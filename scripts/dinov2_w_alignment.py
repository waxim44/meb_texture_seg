from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
H5_PATH = ROOT / "data/feature_database/dinov2_dinov2_vits14_reg_iso.h5"
OUT = ROOT / "outputs/rsa_granulometrie/dinov2_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = "dinov2_vits14_reg"
BEST_LAYER = "layer_03"
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5

df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
Y = df[METRICS].values.astype(np.float64)
groups = df.image.values

with h5py.File(H5_PATH, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all()
    X = f["features"][BEST_LAYER][:][uid].astype(np.float64)
print(f"Bloc/couche : {MODEL_LABEL} / {BEST_LAYER}  (dim={X.shape[1]})")

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))


def unit(w):
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w


w_main = {}
w_folds = {}
for j, m in enumerate(METRICS):
    y = Y[:, j]
    scaler_all = StandardScaler().fit(X)
    ridge_all = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler_all.transform(X), y)
    w_main[m] = unit(ridge_all.coef_)

    w_list = []
    for tr, te in folds:
        scaler = StandardScaler().fit(X[tr])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(X[tr]), y[tr])
        w_list.append(unit(ridge.coef_))
    w_folds[m] = w_list


stability_rows = []
for m in METRICS:
    wl = w_folds[m]
    pair_cos = [float(np.dot(wl[i], wl[k])) for i in range(len(wl)) for k in range(i + 1, len(wl))]
    stability_rows.append(dict(metric=m, cos_moyen_entre_folds=np.mean(pair_cos),
                                cos_min=np.min(pair_cos), cos_max=np.max(pair_cos)))
stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv(OUT / "stabilite_w_dinov2.csv", index=False)
print("\nStabilite des w entre folds :")
print(stability_df.round(3).to_string(index=False))


cos_mat = np.zeros((len(METRICS), len(METRICS)))
for i, mi in enumerate(METRICS):
    for k, mk in enumerate(METRICS):
        cos_mat[i, k] = float(np.dot(w_main[mi], w_main[mk]))
pd.DataFrame(cos_mat, index=METRICS, columns=METRICS).to_csv(OUT / "cosinus_w_dinov2.csv")


corr_ref = pd.read_csv(ROOT / "outputs/rsa_granulometrie/ridge_w_alignment/correlation_pearson_metriques.csv",
                        index_col=0)

print("\nCosinus(w) DINOv2 :")
print(pd.DataFrame(cos_mat, index=METRICS, columns=METRICS).round(3).to_string())
print("\nCorrelation Pearson des metriques (reutilisee, identique pour tous les encodeurs) :")
print(corr_ref.round(3).to_string())


size_metrics = ["D50", "D90", "Span", "Skewness"]
idx = {m: i for i, m in enumerate(METRICS)}
sigma_align = [abs(cos_mat[idx["sigma_local"]][idx[c]]) for c in size_metrics]
print(f"\n|cos(w_sigma_local, w_taille)| : {[round(v,3) for v in sigma_align]}  "
      f"(moyenne={np.mean(sigma_align):.3f})")
print("Rappel TextureSAM (block_4/6/9) : moyenne 0.157 / 0.086 / 0.042 "
      "(orthogonalite croissante avec la profondeur)")


plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "figure.dpi": 200, "savefig.dpi": 200})

def heatmap(ax, M, labels, title):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = M[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=13)
    return im

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
fig.subplots_adjust(wspace=0.55)
heatmap(axes[0], cos_mat, METRICS, "cosinus(w_i, w_j)")
im1 = heatmap(axes[1], corr_ref.values, METRICS, "corrélation(métrique_i, métrique_j)")
cbar = fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
cbar.set_label("valeur signée")
fig.suptitle(f"{MODEL_LABEL} ({BEST_LAYER})", y=1.02, fontsize=18)
fig.savefig(OUT / "fig_alignement_dinov2.png", bbox_inches="tight")
plt.close(fig)
print("\nFigure ecrite : fig_alignement_dinov2.png")
