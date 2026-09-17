from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/ridge_w_alignment"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5


THRESH_STRONG = 0.7
THRESH_PARTIAL_LOW = 0.3
THRESH_PARTIAL_HIGH = 0.5
THRESH_INDEP = 0.2
GAP_NOTABLE = 0.3


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

for b in BLOCKS:
    print(f"  {b} : dim={LAT[b].shape[1]}")

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))


def unit(w):
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w


w_main = {}
w_folds = {}
r2_records = []

for block in BLOCKS:
    X = LAT[block]
    for j, m in enumerate(METRICS):
        y = Y[:, j]


        scaler_all = StandardScaler().fit(X)
        Xz_all = scaler_all.transform(X)
        ridge_all = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xz_all, y)
        w_main[(block, m)] = unit(ridge_all.coef_)


        w_list = []
        r2_list = []
        for fk, (tr, te) in enumerate(folds):
            scaler = StandardScaler().fit(X[tr])
            Xz_tr, Xz_te = scaler.transform(X[tr]), scaler.transform(X[te])
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xz_tr, y[tr])
            pred = ridge.predict(Xz_te)
            r2 = r2_score(y[te], pred)
            r2_list.append(r2)
            w_list.append(unit(ridge.coef_))
            r2_records.append(dict(block=block, metric=m, fold=fk, r2=r2))
        w_folds[(block, m)] = w_list

r2_df = pd.DataFrame(r2_records)
r2_summary = r2_df.groupby(["block", "metric"])["r2"].agg(["mean", "std"]).reset_index()
r2_summary.columns = ["block", "metric", "R2_mean", "R2_std"]
r2_summary.to_csv(OUT / "r2_par_metrique_bloc.csv", index=False)
print("\nR2 (GroupKFold(5), moyenne +/- std sur les 5 folds) :")
print(r2_summary.round(3).to_string(index=False))


stability_records = []
for block in BLOCKS:
    for m in METRICS:
        wl = w_folds[(block, m)]
        pair_cos = []
        for i in range(len(wl)):
            for k in range(i + 1, len(wl)):
                pair_cos.append(float(np.dot(wl[i], wl[k])))
        stability_records.append(dict(block=block, metric=m,
                                       cos_moyen_entre_folds=np.mean(pair_cos),
                                       cos_min_entre_folds=np.min(pair_cos),
                                       cos_max_entre_folds=np.max(pair_cos)))
stability_df = pd.DataFrame(stability_records)
stability_df.to_csv(OUT / "stabilite_w_entre_folds.csv", index=False)
print("\nStabilite des w entre les 5 folds (cosinus pairwise, meme metrique/bloc) :")
print(stability_df.round(3).to_string(index=False))

STABILITY_LOW = stability_df["cos_moyen_entre_folds"] < 0.5
if STABILITY_LOW.any():
    print("\n!!! ATTENTION : directions instables entre folds pour :")
    print(stability_df[STABILITY_LOW][["block", "metric", "cos_moyen_entre_folds"]].to_string(index=False))


cos_matrices = {}
for block in BLOCKS:
    M = np.zeros((len(METRICS), len(METRICS)))
    for i, mi in enumerate(METRICS):
        for k, mk in enumerate(METRICS):
            M[i, k] = float(np.dot(w_main[(block, mi)], w_main[(block, mk)]))
    cos_matrices[block] = M
    pd.DataFrame(M, index=METRICS, columns=METRICS).to_csv(OUT / f"cosinus_w_{block}.csv")


corr_pearson = np.zeros((len(METRICS), len(METRICS)))
corr_spearman = np.zeros((len(METRICS), len(METRICS)))
for i, mi in enumerate(METRICS):
    for k, mk in enumerate(METRICS):
        corr_pearson[i, k] = pearsonr(Y[:, i], Y[:, k])[0]
        corr_spearman[i, k] = spearmanr(Y[:, i], Y[:, k])[0]

pd.DataFrame(corr_pearson, index=METRICS, columns=METRICS).to_csv(OUT / "correlation_pearson_metriques.csv")
pd.DataFrame(corr_spearman, index=METRICS, columns=METRICS).to_csv(OUT / "correlation_spearman_metriques.csv")


corr_ref = corr_pearson
print("\nEcart max |Pearson - Spearman| sur les correlations metriques :",
      round(np.max(np.abs(corr_pearson - corr_spearman)), 3))


gap_records = []
for block in BLOCKS:
    Mcos = cos_matrices[block]
    for i, mi in enumerate(METRICS):
        for k, mk in enumerate(METRICS):
            if k <= i:
                continue
            acos, acorr = abs(Mcos[i, k]), abs(corr_ref[i, k])
            gap = acos - acorr
            if abs(gap) >= GAP_NOTABLE:
                kind = ("alignees_mais_peu_correlees" if gap > 0 else
                         "correlees_mais_divergentes")
            else:
                kind = "coherent"
            gap_records.append(dict(block=block, metric_i=mi, metric_j=mk,
                                     cos_signed=Mcos[i, k], corr_signed=corr_ref[i, k],
                                     abs_cos=acos, abs_corr=acorr, gap=gap, type=kind))

gap_df = pd.DataFrame(gap_records)
gap_df.to_csv(OUT / "comparaison_cosinus_vs_correlation.csv", index=False)
notable_df = gap_df[gap_df.type != "coherent"].sort_values("gap", key=abs, ascending=False)
notable_df.to_csv(OUT / "ecarts_notables.csv", index=False)

print(f"\nPaires avec ecart notable (|gap| >= {GAP_NOTABLE}) :")
if len(notable_df):
    print(notable_df[["block", "metric_i", "metric_j", "abs_cos", "abs_corr", "gap", "type"]]
          .round(3).to_string(index=False))
else:
    print("  (aucune)")

print("\n[OK] etapes 1-4 terminees -> lancer ridge_w_alignment_outputs.py pour figures + rapport.txt")
