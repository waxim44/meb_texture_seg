import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/mlp_vs_ridge"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]
DROPOUTS = [0.3, 0.5]
SEEDS = [0, 1, 2]
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5
DEVICE = torch.device("cpu")


HIDDEN = 32
WEIGHT_DECAY = 1e-3
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 20


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
Y = df[METRICS].values.astype(np.float64)
groups = df.image.values

with h5py.File(H5, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all()
    LAT = {b: f["features"][b][:][uid].astype(np.float64) for b in BLOCKS}

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(len(df)), groups=groups))
print("Folds GroupKFold(5) -- IDENTIQUES a ceux du balayage PCA (meme code, meme ordre) :")
for k, (tr, te) in enumerate(folds):
    g = df.iloc[te]
    print(f"  fold {k}: n_test={len(te):3d}  G={int((g.id==7).sum()):3d}  "
          f"S={int((g.id==8).sum()):3d}  images_test={sorted(g.image.unique())[:2]}...")


class MLP1(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xval, yval, p, seed):
    torch.manual_seed(seed)
    model = MLP1(Xtr.shape[1], p=p).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=DEVICE)
    yval_t = torch.tensor(yval, dtype=torch.float32, device=DEVICE)
    n = len(ytr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = nn.functional.mse_loss(model(Xval_t), yval_t).item()
        if vloss < best_val - 1e-6:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def reduce_features(Xtr, Xother_list, use_pca):
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zothers = [scaler.transform(Xo) for Xo in Xother_list]
    if use_pca:
        pca = PCA(n_components=0.95, svd_solver="full", random_state=0).fit(Ztr)
        Ztr = pca.transform(Ztr)
        Zothers = [pca.transform(Zo) for Zo in Zothers]
    return Ztr, Zothers


def metrics3(y_true, y_pred):
    return dict(r2=r2_score(y_true, y_pred),
                rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                medae=median_absolute_error(y_true, y_pred))


records = []
t_start = time.time()

for block in BLOCKS:
    X = LAT[block]
    print(f"\n=== {block} (dim={X.shape[1]}) ===")
    for fk, (tr_full, te) in enumerate(folds):

        Ztr_ridge, (Zte_ridge,) = reduce_features(X[tr_full], [X[te]], use_pca=False)
        for j, m in enumerate(METRICS):
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Ztr_ridge, Y[tr_full, j])
            pred = ridge.predict(Zte_ridge)
            met = metrics3(Y[te, j], pred)
            records.append(dict(block=block, fold=fk, metric=m, model="Ridge",
                                dropout=np.nan, seed=np.nan, **met))


        groups_tr = groups[tr_full]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=fk)
        tr_in_rel, val_in_rel = next(gss.split(np.zeros(len(tr_full)), groups=groups_tr))
        tr_in, val_in = tr_full[tr_in_rel], tr_full[val_in_rel]

        Ztr_in, (Zval_in, Zte_mlp) = reduce_features(X[tr_in], [X[val_in], X[te]], use_pca=False)

        for j, m in enumerate(METRICS):
            yscaler = StandardScaler().fit(Y[tr_in, j:j+1])
            ytr_s = yscaler.transform(Y[tr_in, j:j+1]).ravel()
            yval_s = yscaler.transform(Y[val_in, j:j+1]).ravel()

            for p in DROPOUTS:
                for seed in SEEDS:
                    model = train_mlp(Ztr_in, ytr_s, Zval_in, yval_s, p, seed)
                    with torch.no_grad():
                        pred_s = model(torch.tensor(Zte_mlp, dtype=torch.float32, device=DEVICE)).cpu().numpy()
                    pred = yscaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
                    met = metrics3(Y[te, j], pred)
                    records.append(dict(block=block, fold=fk, metric=m, model=f"MLP_p{p}",
                                        dropout=p, seed=seed, **met))
        print(f"  fold {fk} termine ({time.time()-t_start:.0f}s ecoulees)")

rec = pd.DataFrame(records)
rec.to_csv(OUT / "resultats_bruts.csv", index=False)
print(f"\nTemps total : {time.time()-t_start:.0f}s")
print("Ecrit :", OUT / "resultats_bruts.csv")
