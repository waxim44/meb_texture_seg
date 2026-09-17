from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import skew as scipy_skew

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/ridge_residuals"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
groups = df.image.values
texture = np.where(df.id.values == 7, "Granuleux", "Sableux")

with h5py.File(H5, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all()
    LAT = {b: f["features"][b][:][uid].astype(np.float64) for b in BLOCKS}

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))
print("Folds GroupKFold(5) -- identiques aux tests precedents (meme code deterministe) :")
for k, (tr, te) in enumerate(folds):
    g = df.iloc[te]
    print(f"  fold {k}: n_test={len(te):3d}  G={int((g.id==7).sum()):3d}  S={int((g.id==8).sum()):3d}")


all_rows = []
r2_check = []
for block in BLOCKS:
    X = LAT[block]
    for m in METRICS:
        j = list(df.columns).index(m) if m in df.columns else None
        y = df[m].values.astype(np.float64)
        pred_oof = np.full(len(df), np.nan)
        fold_id = np.full(len(df), -1)

        for fk, (tr, te) in enumerate(folds):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr, y[tr])
            pred_oof[te] = ridge.predict(Xte)
            fold_id[te] = fk

        assert not np.isnan(pred_oof).any()
        r2_oof = r2_score(y, pred_oof)
        r2_check.append({"block": block, "metric": m, "r2_oof_poole": r2_oof})

        resid = y - pred_oof
        for i in range(len(df)):
            all_rows.append(dict(block=block, metric=m, patch_uid=int(df.patch_uid.iloc[i]),
                                 image=df.image.iloc[i], texture=texture[i], fold=int(fold_id[i]),
                                 y_true=y[i], y_pred_oof=pred_oof[i], residual=resid[i]))
        print(f"{block:10s} {m:12s} R2_oof(poole)={r2_oof:.3f}")

res = pd.DataFrame(all_rows)
res.to_csv(OUT / "residus_oof.csv", index=False)
pd.DataFrame(r2_check).to_csv(OUT / "r2_oof_verif.csv", index=False)


stat_rows = []
top_rows = []
for block in BLOCKS:
    for m in METRICS:
        sub = res[(res.block == block) & (res.metric == m)]
        r = sub.residual.values
        stat_rows.append(dict(block=block, metric=m, n=len(r), mean=r.mean(), std=r.std(),
                              skewness=scipy_skew(r), min=r.min(), max=r.max(),
                              q05=np.quantile(r, 0.05), q95=np.quantile(r, 0.95)))
        top5 = sub.reindex(sub.residual.abs().sort_values(ascending=False).index[:5])
        for _, row in top5.iterrows():
            top_rows.append(dict(block=block, metric=m, patch_uid=row.patch_uid, texture=row.texture,
                                 image=row.image, y_true=row.y_true, y_pred_oof=row.y_pred_oof,
                                 residual=row.residual))

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(OUT / "stats_residus.csv", index=False)
top_df = pd.DataFrame(top_rows)
top_df.to_csv(OUT / "top5_residus_extremes.csv", index=False)

print("\n=== stats residus ===")
print(stats_df.round(3).to_string(index=False))


eta_rows = []
for block in BLOCKS:
    for m in METRICS:
        sub = res[(res.block == block) & (res.metric == m)]
        grand_mean = sub.residual.mean()
        ss_tot = ((sub.residual - grand_mean) ** 2).sum()
        grp = sub.groupby("image")["residual"].agg(["mean", "count"])
        ss_between = (grp["count"] * (grp["mean"] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_tot if ss_tot > 0 else np.nan
        std_per_image = sub.groupby("image")["residual"].std()
        eta_rows.append(dict(block=block, metric=m, eta2_image=eta2,
                             std_intra_image_moyen=std_per_image.mean(),
                             std_global=sub.residual.std(), n_images=sub.image.nunique()))
eta_df = pd.DataFrame(eta_rows)
eta_df.to_csv(OUT / "variance_par_image.csv", index=False)
print("\n=== part de variance des residus expliquee par l'image (eta^2) ===")
print(eta_df.round(3).to_string(index=False))

print("\nEcrit dans", OUT)
