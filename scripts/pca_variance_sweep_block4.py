from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/variance_sweep_block4"
OUT.mkdir(parents=True, exist_ok=True)

BLOCK = "block_4"
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
LEVELS = [0.50, 0.75, 0.95, "brut"]
LEVEL_LABELS = {0.50: "50%", 0.75: "75%", 0.95: "95%", "brut": "brut (100%)"}
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5

df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
Y = df[METRICS].values.astype(np.float64)
groups = df.image.values

with h5py.File(H5, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all()
    X = f["features"][BLOCK][:][uid].astype(np.float64)

print(f"Bloc={BLOCK}  dim_brute={X.shape[1]}  n_patchs={len(df)}")

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))

records = []
n_components_records = []
for fk, (tr, te) in enumerate(folds):
    Xtr_raw, Xte_raw = X[tr], X[te]
    Ytr, Yte = Y[tr], Y[te]
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    for level in LEVELS:
        if level == "brut":
            Ztr, Zte = Xtr_s, Xte_s
            n_comp = Xtr_s.shape[1]
        else:
            pca = PCA(n_components=level, svd_solver="full", random_state=0).fit(Xtr_s)
            Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
            n_comp = Ztr.shape[1]
        n_components_records.append({"fold": fk, "level": LEVEL_LABELS[level], "n_components": n_comp})

        for j, m in enumerate(METRICS):
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Ztr, Ytr[:, j])
            pred = ridge.predict(Zte)
            records.append({"fold": fk, "level": LEVEL_LABELS[level], "metric": m,
                             "r2": r2_score(Yte[:, j], pred),
                             "rmse": np.sqrt(mean_squared_error(Yte[:, j], pred)),
                             "medae": median_absolute_error(Yte[:, j], pred),
                             "n_components": n_comp})
    print(f"fold {fk} termine")

rec = pd.DataFrame(records)
ncomp = pd.DataFrame(n_components_records)
rec.to_csv(OUT / "resultats_par_fold.csv", index=False)

level_order = [LEVEL_LABELS[l] for l in LEVELS]
r2_mean = rec.pivot_table(index="level", columns="metric", values="r2", aggfunc="mean").reindex(level_order)[METRICS]
r2_std = rec.pivot_table(index="level", columns="metric", values="r2", aggfunc="std").reindex(level_order)[METRICS]
rmse_mean = rec.pivot_table(index="level", columns="metric", values="rmse", aggfunc="mean").reindex(level_order)[METRICS]
rmse_std = rec.pivot_table(index="level", columns="metric", values="rmse", aggfunc="std").reindex(level_order)[METRICS]
medae_mean = rec.pivot_table(index="level", columns="metric", values="medae", aggfunc="mean").reindex(level_order)[METRICS]
medae_std = rec.pivot_table(index="level", columns="metric", values="medae", aggfunc="std").reindex(level_order)[METRICS]
ncomp_mean = ncomp.groupby("level")["n_components"].mean().reindex(level_order)
ncomp_std = ncomp.groupby("level")["n_components"].std().reindex(level_order)

r2_mean.to_csv(OUT / "R2_mean_par_niveau.csv"); r2_std.to_csv(OUT / "R2_std_par_niveau.csv")
rmse_mean.to_csv(OUT / "RMSE_mean_par_niveau.csv"); rmse_std.to_csv(OUT / "RMSE_std_par_niveau.csv")
medae_mean.to_csv(OUT / "MedAE_mean_par_niveau.csv"); medae_std.to_csv(OUT / "MedAE_std_par_niveau.csv")
pd.DataFrame({"n_components_mean": ncomp_mean, "n_components_std": ncomp_std}).to_csv(OUT / "n_components_par_niveau.csv")

print("\n=== R2 moyen (5 folds) -- niveau x metrique (block_4) ===")
print(r2_mean.round(3).to_string())
print("\nEcrit dans", OUT)
