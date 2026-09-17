import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/rf_gbm"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
N_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RNG = 0


EXPECTED_FOLD_GS = [(127, 9), (88, 48), (51, 84), (71, 65), (72, 63)]


RF_PARAMS_SQRT = dict(n_estimators=400, max_depth=10, min_samples_leaf=4,
                       max_features="sqrt", random_state=RNG, n_jobs=-1)
RF_PARAMS_THIRD = dict(n_estimators=400, max_depth=10, min_samples_leaf=4,
                        max_features=1/3, random_state=RNG, n_jobs=-1)


GBM_MAX_ESTIMATORS = 300
GBM_PARAMS = dict(n_estimators=GBM_MAX_ESTIMATORS, max_depth=4, learning_rate=0.05,
                   subsample=0.8, random_state=RNG,
                   validation_fraction=0.15, n_iter_no_change=15, tol=1e-4)


N_PERM_REPEATS = 10


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
Y = df[METRICS].values.astype(np.float64)
groups = df.image.values
print(f"N patchs (Granuleux+Sableux) = {len(df)}  |  metriques = {METRICS}")

with h5py.File(H5, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all()
    LAT = {b: f["features"][b][:][uid].astype(np.float64) for b in BLOCKS}

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))

print("\nVerification des folds GroupKFold(5) (doivent coincider avec Ridge/MLP) :")
fold_ok = True
for k, (tr, te) in enumerate(folds):
    g_te = df.iloc[te]
    ng, ns = int((g_te.id == 7).sum()), int((g_te.id == 8).sum())
    exp_g, exp_s = EXPECTED_FOLD_GS[k]
    ok = (ng, ns) == (exp_g, exp_s)
    fold_ok &= ok
    print(f"  fold {k}: G={ng:3d} S={ns:3d}  (attendu G={exp_g},S={exp_s})  {'OK' if ok else 'MISMATCH !!!'}")
assert fold_ok, "Les folds ne correspondent pas au protocole Ridge/MLP -- comparaison invalidee."
print("=> Folds IDENTIQUES au protocole Ridge / MLP confirmes.\n")


def metrics3(y_true, y_pred):
    return dict(r2=r2_score(y_true, y_pred),
                rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                medae=median_absolute_error(y_true, y_pred))


records = []
rf_importances_impurity = {}
rf_importances_perm = {}
ridge_coefs = {}

t0 = time.time()
for block in BLOCKS:
    X = LAT[block]
    dim = X.shape[1]
    print(f"=== {block} (dim={dim}) ===", flush=True)


    for j, m in enumerate(METRICS):
        scaler_all = StandardScaler().fit(X)
        ridge_all = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler_all.transform(X), Y[:, j])
        w = ridge_all.coef_
        ridge_coefs[(block, m)] = w / (np.linalg.norm(w) + 1e-12)

    perm_acc = {(m, v): np.zeros(dim) for m in METRICS for v in ["sqrt", "third"]}
    imp_acc = {(m, v): [] for m in METRICS for v in ["sqrt", "third"]}

    for fk, (tr, te) in enumerate(folds):
        Xtr, Xte = X[tr], X[te]

        for j, m in enumerate(METRICS):
            ytr, yte = Y[tr, j], Y[te, j]


            scaler = StandardScaler().fit(Xtr)
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(Xtr), ytr)
            pred_ridge = ridge.predict(scaler.transform(Xte))
            records.append(dict(block=block, metric=m, fold=fk, model="Ridge",
                                 **metrics3(yte, pred_ridge)))


            scaler_y = StandardScaler().fit(ytr.reshape(-1, 1))
            ytr_s = scaler_y.transform(ytr.reshape(-1, 1)).ravel()
            svr = SVR(kernel="rbf", C=10, gamma="scale").fit(scaler.transform(Xtr), ytr_s)
            pred_svr = scaler_y.inverse_transform(svr.predict(scaler.transform(Xte)).reshape(-1, 1)).ravel()
            records.append(dict(block=block, metric=m, fold=fk, model="SVR_RBF",
                                 **metrics3(yte, pred_svr)))


            for variant, params in [("sqrt", RF_PARAMS_SQRT), ("third", RF_PARAMS_THIRD)]:
                rf = RandomForestRegressor(**params).fit(Xtr, ytr)
                pred_rf = rf.predict(Xte)
                records.append(dict(block=block, metric=m, fold=fk, model=f"RF_{variant}",
                                     **metrics3(yte, pred_rf)))
                imp_acc[(m, variant)].append(rf.feature_importances_)


                rf.n_jobs = 1
                pim = permutation_importance(rf, Xte, yte, n_repeats=N_PERM_REPEATS,
                                              random_state=RNG, scoring="r2", n_jobs=-1)
                perm_acc[(m, variant)] += pim.importances_mean


            gbm = GradientBoostingRegressor(**GBM_PARAMS).fit(Xtr, ytr)
            pred_gbm_te = gbm.predict(Xte)
            pred_gbm_tr = gbm.predict(Xtr)
            m_te = metrics3(yte, pred_gbm_te)
            m_tr = metrics3(ytr, pred_gbm_tr)
            records.append(dict(block=block, metric=m, fold=fk, model="GBM",
                                 n_estimators_used=gbm.n_estimators_, **m_te))
            records.append(dict(block=block, metric=m, fold=fk, model="GBM_train",
                                 n_estimators_used=gbm.n_estimators_, **m_tr))

        print(f"  fold {fk} termine ({time.time()-t0:.0f}s ecoulees)", flush=True)

    for m in METRICS:
        for variant in ["sqrt", "third"]:
            rf_importances_impurity[(block, m, variant)] = np.mean(imp_acc[(m, variant)], axis=0)
            rf_importances_perm[(block, m, variant)] = perm_acc[(m, variant)] / N_SPLITS

rec = pd.DataFrame(records)
rec.to_csv(OUT / "resultats_bruts.csv", index=False)
print(f"\nTemps total : {time.time()-t0:.0f}s. Ecrit : {OUT/'resultats_bruts.csv'}")


TOP_K = 10
imp_rows = []
for block in BLOCKS:
    for m in METRICS:
        w = np.abs(ridge_coefs[(block, m)])
        top_ridge = set(np.argsort(-w)[:TOP_K].tolist())
        for variant in ["sqrt", "third"]:
            imp_i = rf_importances_impurity[(block, m, variant)]
            imp_p = rf_importances_perm[(block, m, variant)]
            top_imp = set(np.argsort(-imp_i)[:TOP_K].tolist())
            top_perm = set(np.argsort(-imp_p)[:TOP_K].tolist())
            overlap_impurity = len(top_ridge & top_imp)
            overlap_perm = len(top_ridge & top_perm)
            imp_rows.append(dict(
                block=block, metric=m, rf_variant=variant,
                top_dims_impurity=",".join(map(str, np.argsort(-imp_i)[:TOP_K])),
                top_dims_permutation=",".join(map(str, np.argsort(-imp_p)[:TOP_K])),
                top_dims_ridge_w=",".join(map(str, np.argsort(-w)[:TOP_K])),
                overlap_impurity_vs_ridge=overlap_impurity,
                overlap_permutation_vs_ridge=overlap_perm,
            ))
imp_df = pd.DataFrame(imp_rows)
imp_df.to_csv(OUT / "importances_top_dims.csv", index=False)
print(f"Ecrit : {OUT/'importances_top_dims.csv'}")


np.savez(OUT / "importances_arrays.npz",
         **{f"impurity__{b}__{m}__{v}": rf_importances_impurity[(b, m, v)]
            for b in BLOCKS for m in METRICS for v in ["sqrt", "third"]},
         **{f"perm__{b}__{m}__{v}": rf_importances_perm[(b, m, v)]
            for b in BLOCKS for m in METRICS for v in ["sqrt", "third"]},
         **{f"ridge_w__{b}__{m}": ridge_coefs[(b, m)] for b in BLOCKS for m in METRICS})

print("\n[OK] etape 1-3 (RF, GBM, SVR, importances) terminees -> lancer rf_gbm_granulo_outputs.py")
