from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

ROOT = Path("/home/aidouni/meb_texture_seg")
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/dinov2_comparison"
OUT.mkdir(parents=True, exist_ok=True)

H5_FILES = {
    "dinov2_vits14_reg": ROOT / "data/feature_database/dinov2_dinov2_vits14_reg_iso.h5",
    "dinov2_vitb14":      ROOT / "data/feature_database/dinov2_dinov2_vitb14_iso.h5",
    "dinov2_vitb14_reg":  ROOT / "data/feature_database/dinov2_dinov2_vitb14_reg_iso.h5",
}
LAYERS = ["layer_03", "layer_06", "layer_09", "layer_12"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5
EXPECTED_FOLD_GS = [(127, 9), (88, 48), (51, 84), (71, 65), (72, 63)]


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
Y = df[METRICS].values.astype(np.float64)
groups = df.image.values
print(f"N patchs = {len(df)}  metriques = {METRICS}")

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))
print("Verification des folds GroupKFold(5) (doivent coincider avec TextureSAM/Ridge) :")
fold_ok = True
for k, (tr, te) in enumerate(folds):
    g_te = df.iloc[te]
    ng, ns = int((g_te.id == 7).sum()), int((g_te.id == 8).sum())
    exp_g, exp_s = EXPECTED_FOLD_GS[k]
    ok = (ng, ns) == (exp_g, exp_s)
    fold_ok &= ok
    print(f"  fold {k}: G={ng:3d} S={ns:3d}  (attendu G={exp_g},S={exp_s})  {'OK' if ok else 'MISMATCH'}")
assert fold_ok, "folds differents du protocole TextureSAM -- ARRET"
print("=> Folds IDENTIQUES au protocole TextureSAM confirmes.\n")


def metrics3(y_true, y_pred):
    return dict(r2=r2_score(y_true, y_pred),
                rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                medae=median_absolute_error(y_true, y_pred))


records = []
oof_store = {}
texture = np.where(df.id.values == 7, "Granuleux", "Sableux")

for model_name, h5_path in H5_FILES.items():
    with h5py.File(h5_path, "r") as f:
        cids = f["metadata/category_ids"][:]
        names = np.array([x.decode() for x in f["metadata/image_names"][:]])
        assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all(),\
            f"desalignement H5<->CSV pour {model_name}"
        LAT = {L: f["features"][L][:][uid].astype(np.float64) for L in LAYERS}

    for L in LAYERS:
        X = LAT[L]
        for j, m in enumerate(METRICS):
            y = Y[:, j]
            pred_oof = np.full(len(df), np.nan)
            for tr, te in folds:
                scaler = StandardScaler().fit(X[tr])
                ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(X[tr]), y[tr])
                pred_oof[te] = ridge.predict(scaler.transform(X[te]))
            assert not np.isnan(pred_oof).any()


            fold_metrics = []
            for tr, te in folds:
                fold_metrics.append(metrics3(y[te], pred_oof[te]))
            r2_mean = np.mean([fm["r2"] for fm in fold_metrics])
            r2_std = np.std([fm["r2"] for fm in fold_metrics])
            rmse_mean = np.mean([fm["rmse"] for fm in fold_metrics])
            rmse_std = np.std([fm["rmse"] for fm in fold_metrics])
            medae_mean = np.mean([fm["medae"] for fm in fold_metrics])
            medae_std = np.std([fm["medae"] for fm in fold_metrics])
            r2_pooled = r2_score(y, pred_oof)

            records.append(dict(model=model_name, layer=L, metric=m,
                                 R2_mean=r2_mean, R2_std=r2_std,
                                 RMSE_mean=rmse_mean, RMSE_std=rmse_std,
                                 MedAE_mean=medae_mean, MedAE_std=medae_std,
                                 R2_pooled=r2_pooled, dim=X.shape[1]))
            oof_store[(model_name, L, m)] = (y.copy(), pred_oof.copy(), texture.copy())

        print(f"  {model_name:20s} {L} : R2(D50)={records[-5]['R2_mean']:.3f}  "
              f"(pooled={records[-5]['R2_pooled']:.3f})")

rec = pd.DataFrame(records)
rec.to_csv(OUT / "ridge_dinov2_par_couche.csv", index=False)
print(f"\nEcrit : {OUT / 'ridge_dinov2_par_couche.csv'}")


best_rows = rec.loc[rec.groupby("metric")["R2_mean"].idxmax()]
print("\n=== Meilleure (modele, couche) par metrique (tous modeles confondus) ===")
print(best_rows[["metric", "model", "layer", "R2_mean"]].to_string(index=False))

primary = rec[rec.model == "dinov2_vits14_reg"]
best_primary_layer = primary.groupby("layer")["R2_mean"].mean().idxmax()
print(f"\nMeilleure couche du modele PRINCIPAL (dinov2_vits14_reg), moyenne des 5 metriques : "
      f"{best_primary_layer}")
best_primary_d50 = primary[primary.metric == "D50"].sort_values("R2_mean", ascending=False).iloc[0]
print(f"Meilleure couche pour D50 (modele principal) : {best_primary_d50.layer} "
      f"(R2={best_primary_d50.R2_mean:.3f})")

import pickle
with open(OUT / "oof_dinov2.pkl", "wb") as fh:
    pickle.dump(oof_store, fh, protocol=4)
print(f"\nOOF sauvegardes : {OUT / 'oof_dinov2.pkl'}")
print("\n[OK] Ridge DINOv2 termine -> dinov2_ridge_outputs.py pour tableau + figures predit/reel.")
