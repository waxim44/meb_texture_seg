import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ["D50", "D10", "D90", "Span", "Skewness", "sigma_local"]
BLOCKS = [f"block_{i}" for i in range(16)] + [f"stage_{i}_fpn" for i in (1, 2, 3, 4)]
N_SPLITS = 5
PCA_VAR = 0.95
RIDGE_ALPHAS = np.logspace(-3, 4, 30)
SVR_GRID = {"estimator__C": [1, 10, 100], "estimator__gamma": ["scale", 0.1]}


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
Y = df[METRICS].values.astype(np.float64)
groups = df.image.values
strat = df.id.values

with h5py.File(H5, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all(),\
        "desalignement latent <-> granulometrie"
    LAT = {b: f["features"][b][:][uid].astype(np.float64) for b in BLOCKS}

print(f"Patchs : {len(df)}  (Granuleux={int((df.id==7).sum())}, Sableux={int((df.id==8).sum())})")
print(f"Images distinctes : {df.image.nunique()}   -> StratifiedGroupKFold({N_SPLITS})")
print(f"Latents alignes sur le CSV via patch_uid : OK (id + image_name verifies)\n")

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
folds = list(sgkf.split(np.zeros(len(df)), strat, groups))
for k, (tr, te) in enumerate(folds):
    print(f"  fold {k}: test n={len(te):3d}  G={int((strat[te]==7).sum()):3d}  "
          f"S={int((strat[te]==8).sum()):3d}  images={len(set(groups[te]))}")


def eval_block(X):
    n, d = X.shape
    oof_ridge = np.full_like(Y, np.nan)
    oof_svr = np.full_like(Y, np.nan)
    r2_ridge_f, mae_ridge_f, r2_svr_f, mae_svr_f = [], [], [], []
    n_pca = []

    for tr, te in folds:
        xs = StandardScaler().fit(X[tr])
        Xtr, Xte = xs.transform(X[tr]), xs.transform(X[te])
        pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=0).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        n_pca.append(Ztr.shape[1])

        ys = StandardScaler().fit(Y[tr])
        Ytr_s = ys.transform(Y[tr])


        ridge = MultiOutputRegressor(RidgeCV(alphas=RIDGE_ALPHAS))
        ridge.fit(Ztr, Ytr_s)
        pr = ys.inverse_transform(ridge.predict(Zte))
        oof_ridge[te] = pr
        r2_ridge_f.append([r2_score(Y[te, j], pr[:, j]) for j in range(len(METRICS))])
        mae_ridge_f.append([mean_absolute_error(Y[te, j], pr[:, j]) for j in range(len(METRICS))])


        base = GridSearchCV(SVR(kernel="rbf"),
                            {"C": [1, 10, 100], "gamma": ["scale", 0.1]},
                            cv=3, scoring="r2", n_jobs=-1)
        svr = MultiOutputRegressor(base)
        svr.fit(Ztr, Ytr_s)
        ps = ys.inverse_transform(svr.predict(Zte))
        oof_svr[te] = ps
        r2_svr_f.append([r2_score(Y[te, j], ps[:, j]) for j in range(len(METRICS))])
        mae_svr_f.append([mean_absolute_error(Y[te, j], ps[:, j]) for j in range(len(METRICS))])

    r2_ridge_f = np.array(r2_ridge_f); mae_ridge_f = np.array(mae_ridge_f)
    r2_svr_f = np.array(r2_svr_f); mae_svr_f = np.array(mae_svr_f)
    return dict(
        r2_ridge_mean=r2_ridge_f.mean(0), r2_ridge_std=r2_ridge_f.std(0),
        mae_ridge_mean=mae_ridge_f.mean(0), mae_ridge_std=mae_ridge_f.std(0),
        r2_svr_mean=r2_svr_f.mean(0), r2_svr_std=r2_svr_f.std(0),
        mae_svr_mean=mae_svr_f.mean(0), mae_svr_std=mae_svr_f.std(0),
        r2_ridge_pooled=np.array([r2_score(Y[:, j], oof_ridge[:, j]) for j in range(len(METRICS))]),
        r2_svr_pooled=np.array([r2_score(Y[:, j], oof_svr[:, j]) for j in range(len(METRICS))]),
        n_pca=int(np.mean(n_pca)), oof_ridge=oof_ridge,
    )


results = {}
oof_ridge_by_block = {}
for b in BLOCKS:
    res = eval_block(LAT[b])
    results[b] = res
    oof_ridge_by_block[b] = res["oof_ridge"]
    d50 = METRICS.index("D50")
    print(f"{b:14s} PCA~{res['n_pca']:3d}c  "
          f"R2(D50) Ridge={res['r2_ridge_mean'][d50]:.3f}±{res['r2_ridge_std'][d50]:.3f}  "
          f"SVR={res['r2_svr_mean'][d50]:.3f}  | "
          f"R2 moyen(6) Ridge={res['r2_ridge_mean'].mean():.3f}")


def table(key):
    return pd.DataFrame({b: results[b][key] for b in BLOCKS}, index=METRICS)

r2_ridge = table("r2_ridge_mean"); r2_ridge.to_csv(OUT / "R2_ridge_metrique_x_bloc.csv")
mae_ridge = table("mae_ridge_mean"); mae_ridge.to_csv(OUT / "MAE_ridge_metrique_x_bloc.csv")
r2_svr = table("r2_svr_mean"); r2_svr.to_csv(OUT / "R2_svr_metrique_x_bloc.csv")
table("r2_ridge_std").to_csv(OUT / "R2_ridge_std_metrique_x_bloc.csv")
table("r2_ridge_pooled").to_csv(OUT / "R2_ridge_pooled_metrique_x_bloc.csv")

best_block = {m: r2_ridge.loc[m].idxmax() for m in METRICS}
summary = pd.DataFrame({
    "meilleur_bloc_Ridge": [best_block[m] for m in METRICS],
    "R2_Ridge_best": [r2_ridge.loc[m, best_block[m]] for m in METRICS],
    "R2_Ridge_std_best": [results[best_block[m]]["r2_ridge_std"][METRICS.index(m)] for m in METRICS],
    "MAE_Ridge_best": [mae_ridge.loc[m, best_block[m]] for m in METRICS],
    "R2_SVR_meme_bloc": [r2_svr.loc[m, best_block[m]] for m in METRICS],
    "R2_Ridge_pooled_best": [results[best_block[m]]["r2_ridge_pooled"][METRICS.index(m)] for m in METRICS],
}, index=METRICS)
summary.to_csv(OUT / "resume_meilleur_bloc.csv")

np.savez(OUT / "oof_ridge.npz", **{b: oof_ridge_by_block[b] for b in BLOCKS},
         Y=Y, id=df.id.values, image=df.image.values, best_block_D50=best_block["D50"])

print("\n=== R2 Ridge (moyenne 5 folds) — metrique x bloc ===")
print(r2_ridge.round(3).to_string())
print("\n=== Resume au meilleur bloc (Ridge) ===")
print(summary.round(3).to_string())

with open(OUT / "results_json.json", "w") as fh:
    json.dump({
        "n_patchs": int(len(df)), "n_granuleux": int((df.id == 7).sum()),
        "n_sableux": int((df.id == 8).sum()), "n_images": int(df.image.nunique()),
        "validation": f"StratifiedGroupKFold({N_SPLITS}) groupes=image_name strat=texture",
        "pca_var": PCA_VAR,
        "best_block": best_block,
        "R2_ridge_best": {m: float(r2_ridge.loc[m, best_block[m]]) for m in METRICS},
        "MAE_ridge_best": {m: float(mae_ridge.loc[m, best_block[m]]) for m in METRICS},
        "R2_svr_same_block": {m: float(r2_svr.loc[m, best_block[m]]) for m in METRICS},
    }, fh, indent=2)
print("\nEcrit dans", OUT)
