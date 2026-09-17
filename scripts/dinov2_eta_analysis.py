import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path("/home/aidouni/meb_texture_seg")
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
H5_PATH = ROOT / "data/feature_database/dinov2_dinov2_vits14_reg_iso.h5"
IMG_DIR = ROOT / "Image_Ouassim"
OUT = ROOT / "outputs/rsa_granulometrie/dinov2_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "dinov2_vits14_reg"
LAYER_INDEX = 2
LAYER_LABEL = "layer_03"
ORIG_H, ORIG_W = 768, 1280
ISO_H, ISO_W = 756, 1260
ISO_NH, ISO_NW = 54, 90

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] device={device}, modele={MODEL_NAME}, couche={LAYER_LABEL}")


def eta_from_tokens(tokens_by_patch, patch_mask=None):
    idx = np.arange(len(tokens_by_patch)) if patch_mask is None else np.where(patch_mask)[0]
    tb = [tokens_by_patch[i] for i in idx]
    n_p_sub = np.array([t.shape[0] for t in tb])
    N = n_p_sub.sum()
    z_p = np.stack([t.mean(axis=0) for t in tb])
    all_tokens = np.concatenate(tb, axis=0)
    z_bar = all_tokens.sum(axis=0) / N
    ss_intra_p = np.array([((t - zp) ** 2).sum() for t, zp in zip(tb, z_p)])
    ss_intra = ss_intra_p.sum()
    ss_inter_p = n_p_sub * ((z_p - z_bar) ** 2).sum(axis=1)
    ss_inter = ss_inter_p.sum()
    ss_total = ((all_tokens - z_bar) ** 2).sum()
    err = abs(ss_total - (ss_inter + ss_intra)) / ss_total
    eta2 = ss_inter / ss_total
    ss_inter_d = (n_p_sub[:, None] * (z_p - z_bar) ** 2).sum(axis=0)
    ss_total_d = ((all_tokens - z_bar) ** 2).sum(axis=0)
    return dict(N=int(N), n_patchs=len(idx), ss_intra=ss_intra, ss_inter=ss_inter,
                ss_total=ss_total, eta2=eta2, identity_rel_err=err,
                ss_intra_p=ss_intra_p, disp_p=ss_intra_p / n_p_sub,
                ss_inter_d=ss_inter_d, ss_total_d=ss_total_d,
                eta2_d=ss_inter_d / ss_total_d)


print("\n=== tests synthetiques (AVANT donnees reelles) ===")
rng = np.random.default_rng(0)
patch_means = np.array([[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]])
tokens1 = [np.tile(m, (4, 1)) for m in patch_means]
r1 = eta_from_tokens(tokens1)
ok1 = np.isclose(r1["eta2"], 1.0) and np.isclose(r1["ss_intra"], 0.0)
print(f"  Test 1 (intra=0) : eta2={r1['eta2']:.6f}  {'OK' if ok1 else 'ECHEC'}")

common_mean = np.array([2.0, -1.0])
tokens2 = [common_mean + rng.normal(size=(6, 2)) for _ in range(4)]
tokens2 = [common_mean + (t - t.mean(axis=0)) for t in tokens2]
r2 = eta_from_tokens(tokens2)
ok2 = np.isclose(r2["eta2"], 0.0, atol=1e-10)
print(f"  Test 2 (inter=0) : eta2={r2['eta2']:.2e}  {'OK' if ok2 else 'ECHEC'}")

tokens3 = [rng.normal(loc=i * 2.0, size=(rng.integers(1, 15), 5)) for i in range(10)]
r3 = eta_from_tokens(tokens3)
err3 = abs(r3["ss_total"] - (r3["ss_inter"] + r3["ss_intra"])) / r3["ss_total"]
ok3 = err3 < 1e-8
print(f"  Test 3 (identite): erreur={err3:.2e}  {'OK' if ok3 else 'ECHEC'}")
assert ok1 and ok2 and ok3, "Tests synthetiques echoues -- ARRET"
print("  => 3/3 PASSES")


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
texture = np.where(df.id.values == 7, "Granuleux", "Sableux")

with h5py.File(H5_PATH, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    positions = f["metadata/positions"][:]
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all()
    h5_vecs = f["features"][LAYER_LABEL][:][uid].astype(np.float64)
print(f"\n{len(df)} patchs (Granuleux+Sableux), {df.image.nunique()} images. Alignement H5 OK.")


def coord_to_tokens(x_min, y_min, x_max, y_max, nH, nW):
    sx = nW / ORIG_W
    sy = nH / ORIG_H
    tx1 = max(0, int(x_min * sx))
    ty1 = max(0, int(y_min * sy))
    tx2 = min(nW, max(tx1 + 1, int(x_max * sx)))
    ty2 = min(nH, max(ty1 + 1, int(y_max * sy)))
    if tx2 - tx1 < 1:
        tx1 = min(tx1, nW - 1); tx2 = tx1 + 1
    if ty2 - ty1 < 1:
        ty1 = min(ty1, nH - 1); ty2 = ty1 + 1
    return tx1, ty1, tx2, ty2


def preprocess(img_array):
    img = Image.fromarray(img_array)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((ISO_W, ISO_H), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)


print(f"\n[Extraction] chargement {MODEL_NAME}")
model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME, verbose=False).eval().to(device)

tokens_list = [None] * len(df)
images_sorted = sorted(df.image.unique())
img_to_indices = {im: np.where(df.image.values == im)[0] for im in images_sorted}
patch_positions = positions[uid]

for img_name in images_sorted:
    img_arr = np.array(Image.open(IMG_DIR / img_name).convert("L"))
    tensor = preprocess(img_arr).to(device)
    with torch.no_grad():
        inter = model.get_intermediate_layers(tensor, n=[LAYER_INDEX], reshape=False,
                                               return_class_token=False, norm=True)
    tokens_flat = inter[0][0].double()
    D = tokens_flat.shape[-1]
    tokens_grid = tokens_flat.reshape(ISO_NH, ISO_NW, D).cpu().numpy()

    for i in img_to_indices[img_name]:
        x1, y1, x2, y2 = patch_positions[i]
        tx1, ty1, tx2, ty2 = coord_to_tokens(x1, y1, x2, y2, ISO_NH, ISO_NW)
        region = tokens_grid[ty1:ty2, tx1:tx2, :].reshape(-1, D)
        tokens_list[i] = region

print(f"[Extraction] termine pour {len(df)} patchs.")

n_p = np.array([t.shape[0] for t in tokens_list])
print(f"\ndistribution n_p : min={n_p.min()}, mediane={np.median(n_p):.1f}, max={n_p.max()}")
n_le2 = int((n_p <= 2).sum())
print(f"patchs n_p<=2 : {n_le2} / {len(n_p)}")


rng_check = np.random.default_rng(1)
check_idx = rng_check.choice(len(df), size=8, replace=False)
max_abs_diff = max(np.abs(tokens_list[i].mean(axis=0) - h5_vecs[i]).max() for i in check_idx)
print(f"\ncontrole H5 (sans normalisation) : ecart max = {max_abs_diff:.2e}")
H5_OK = max_abs_diff < 1e-4
print(f"=> {'OK' if H5_OK else 'ECHEC -- ARRET'}")
assert H5_OK, "pipeline DINOv2 non reproduit a l'identique -- arret"


r_all = eta_from_tokens(tokens_list)
print(f"\n=== eta^2 (DINOv2 {LAYER_LABEL}) ===")
print(f"  SS_total={r_all['ss_total']:.4e}  SS_inter={r_all['ss_inter']:.4e}  "
      f"SS_intra={r_all['ss_intra']:.4e}  erreur_identite={r_all['identity_rel_err']:.2e}")
assert r_all["identity_rel_err"] < 1e-8, "IDENTITE ECHOUEE -- eta2 non fiable, ARRET"
assert 0 <= r_all["eta2"] <= 1
print(f"  eta2 = {r_all['eta2']:.4f}  (1-eta2 = {1-r_all['eta2']:.4f})")

texture_rows = []
for t in ["Granuleux", "Sableux"]:
    mask = texture == t
    rt = eta_from_tokens(tokens_list, patch_mask=mask)
    assert rt["identity_rel_err"] < 1e-8
    texture_rows.append(dict(texture=t, n_patchs=rt["n_patchs"], eta2=rt["eta2"]))
    print(f"  {t:10s} : n={rt['n_patchs']:3d}  eta2={rt['eta2']:.4f}")

texture_df = pd.DataFrame(texture_rows)
texture_df.to_csv(OUT / "eta2_dinov2_par_texture.csv", index=False)

summary = pd.DataFrame([dict(model=MODEL_NAME, layer=LAYER_LABEL,
                              n_p_min=int(n_p.min()), n_p_median=float(np.median(n_p)),
                              n_p_max=int(n_p.max()), n_le2=n_le2,
                              SS_total=r_all["ss_total"], SS_inter=r_all["ss_inter"],
                              SS_intra=r_all["ss_intra"], erreur_identite=r_all["identity_rel_err"],
                              eta2=r_all["eta2"], un_moins_eta2=1 - r_all["eta2"])])
summary.to_csv(OUT / "resume_eta2_dinov2.csv", index=False)

print("\nComparaison avec TextureSAM (deja etabli) :")
print("  block_4 : Granuleux 0.399 / Sableux 0.424  (global 0.419)")
print("  block_6 : Granuleux 0.345 / Sableux 0.376  (global 0.368)")
print("  block_9 : Granuleux 0.304 / Sableux 0.333  (global 0.327)")
print(f"  DINOv2 {LAYER_LABEL} : Granuleux {texture_df[texture_df.texture=='Granuleux'].eta2.iloc[0]:.3f} / "
      f"Sableux {texture_df[texture_df.texture=='Sableux'].eta2.iloc[0]:.3f}  (global {r_all['eta2']:.3f})")

import pickle
with open(OUT / "eta2_dinov2_resultats.pkl", "wb") as fh:
    pickle.dump(dict(r_all=r_all, texture=texture, n_p=n_p), fh, protocol=4)

print(f"\n[OK] Ecrit dans {OUT}")
