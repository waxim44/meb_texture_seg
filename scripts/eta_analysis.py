import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from scipy.stats import spearmanr

ROOT = Path("/home/aidouni/meb_texture_seg")
sys.path.insert(0, str(ROOT))
from build_feature_database import (build_image_encoder, load_model, register_hooks,
                                     remove_hooks, preprocess, extract_patch_features)

H5_PATH = ROOT / "data/feature_database/database_meb_ouassim.h5"
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
IMG_DIR = ROOT / "Image_Ouassim"
CKPT = ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
OUT = ROOT / "outputs/rsa_granulometrie/eta_analysis"
OUT.mkdir(parents=True, exist_ok=True)

BLOCKS = ["block_4", "block_6", "block_9"]
N_TOL_CONTROL = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] device={device}, checkpoint={CKPT.name}")


def eta_from_tokens(tokens_by_patch):
    n_p = np.array([t.shape[0] for t in tokens_by_patch])
    N = n_p.sum()
    C = tokens_by_patch[0].shape[1]

    z_p = np.stack([t.mean(axis=0) for t in tokens_by_patch])

    all_tokens = np.concatenate(tokens_by_patch, axis=0)
    z_bar = all_tokens.sum(axis=0) / N

    ss_intra_p = np.array([((t - zp) ** 2).sum() for t, zp in zip(tokens_by_patch, z_p)])
    ss_intra = ss_intra_p.sum()

    ss_inter_p = n_p * ((z_p - z_bar) ** 2).sum(axis=1)
    ss_inter = ss_inter_p.sum()

    ss_total = ((all_tokens - z_bar) ** 2).sum()

    eta2 = ss_inter / ss_total if ss_total > 0 else np.nan
    disp_p = ss_intra_p / n_p


    ss_inter_d = (n_p[:, None] * (z_p - z_bar) ** 2).sum(axis=0)
    ss_total_d = ((all_tokens - z_bar) ** 2).sum(axis=0)

    return dict(n_p=n_p, N=N, z_p=z_p, z_bar=z_bar,
                ss_intra=ss_intra, ss_inter=ss_inter, ss_total=ss_total, eta2=eta2,
                ss_intra_p=ss_intra_p, disp_p=disp_p,
                ss_inter_d=ss_inter_d, ss_total_d=ss_total_d)


def run_synthetic_tests():
    print("\n=== ETAPE 3.D -- tests sur donnees synthetiques (AVANT donnees reelles) ===")
    rng = np.random.default_rng(0)


    patch_means = np.array([[0.0, 0.0], [5.0, 5.0], [-3.0, 2.0]])
    tokens1 = [np.tile(m, (4, 1)) for m in patch_means]
    r1 = eta_from_tokens(tokens1)
    ok1 = np.isclose(r1["eta2"], 1.0) and np.isclose(r1["ss_intra"], 0.0)
    print(f"  Test 1 (intra=0, patches distincts) : eta2={r1['eta2']:.6f} "
          f"(attendu 1.0) SS_intra={r1['ss_intra']:.2e} (attendu 0)  {'OK' if ok1 else 'ECHEC'}")


    common_mean = np.array([2.0, -1.0])
    tokens2 = [common_mean + rng.normal(size=(6, 2)) for _ in range(4)]

    tokens2 = [common_mean + (t - t.mean(axis=0)) for t in tokens2]
    r2 = eta_from_tokens(tokens2)
    ok2 = np.isclose(r2["eta2"], 0.0, atol=1e-10) and np.isclose(r2["ss_inter"], 0.0, atol=1e-10)
    print(f"  Test 2 (inter=0, memes moyennes)    : eta2={r2['eta2']:.2e} "
          f"(attendu 0.0) SS_inter={r2['ss_inter']:.2e} (attendu 0)  {'OK' if ok2 else 'ECHEC'}")


    tokens3 = [rng.normal(loc=i * 2.0, size=(rng.integers(1, 15), 5)) for i in range(10)]
    r3 = eta_from_tokens(tokens3)
    err3 = abs(r3["ss_total"] - (r3["ss_inter"] + r3["ss_intra"])) / r3["ss_total"]
    ok3 = err3 < 1e-8
    print(f"  Test 3 (n_p variables, cas general) : erreur relative identite = {err3:.2e}  "
          f"{'OK' if ok3 else 'ECHEC'}")

    all_ok = ok1 and ok2 and ok3
    print(f"  => Tests synthetiques : {'TOUS PASSES' if all_ok else 'ECHEC -- ARRET'}")
    if not all_ok:
        raise SystemExit("Tests synthetiques echoues -- le calcul d'eta2 est probablement bugge. Arret.")
    return all_ok


run_synthetic_tests()


df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
texture = np.where(df.id.values == 7, "Granuleux", "Sableux")

with h5py.File(H5_PATH, "r") as f:
    cids = f["metadata/category_ids"][:]
    names = np.array([x.decode() for x in f["metadata/image_names"][:]])
    positions_h5 = f["metadata/positions"][:]
    assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all(),\
        "desalignement CSV <-> H5 -- ARRET"
    h5_vecs = {b: f["features"][b][:][uid].astype(np.float64) for b in BLOCKS}
print(f"[Etape 0] {len(df)} patchs (Granuleux+Sableux), {df.image.nunique()} images. "
      "Alignement H5 <-> CSV verifie (category_id + image_name).")


patch_dicts = []
for i in range(len(df)):
    x1, y1, x2, y2 = positions_h5[uid[i]].astype(int)
    patch_dicts.append(dict(x_min=x1, y_min=y1, x_max=x2, y_max=y2))


print(f"\n[Extraction] chargement encoder + checkpoint {CKPT.name}")
encoder = load_model(CKPT, device)
captured, handles = register_hooks(encoder)

tokens_by_block = {b: [None] * len(df) for b in BLOCKS}
h5_check_rows = []

images_sorted = sorted(df.image.unique())
img_to_indices = {im: np.where(df.image.values == im)[0] for im in images_sorted}

for img_name in images_sorted:
    img_path = IMG_DIR / img_name
    tensor, orig_H, orig_W = preprocess(img_path, device)
    captured.clear()
    with torch.no_grad():
        encoder(tensor)

    for i in img_to_indices[img_name]:
        p = patch_dicts[i]
        for key in BLOCKS:
            feat_map = captured[key]
            feat = feat_map[0].double()
            H_feat, W_feat, C = feat.shape

            scale_x = W_feat / orig_W
            scale_y = H_feat / orig_H
            fx1 = max(0, int(p["x_min"] * scale_x))
            fy1 = max(0, int(p["y_min"] * scale_y))
            fx2 = min(W_feat, max(fx1 + 1, int(p["x_max"] * scale_x)))
            fy2 = min(H_feat, max(fy1 + 1, int(p["y_max"] * scale_y)))
            if fx2 - fx1 < 1:
                fx1 = min(fx1, W_feat - 1); fx2 = fx1 + 1
            if fy2 - fy1 < 1:
                fy1 = min(fy1, H_feat - 1); fy2 = fy1 + 1

            region = feat[fy1:fy2, fx1:fx2, :].reshape(-1, C).cpu().numpy()
            tokens_by_block[key][i] = region

remove_hooks(handles)
print(f"[Extraction] termine pour {len(df)} patchs x {len(BLOCKS)} blocs.")


n_p_ref = np.array([t.shape[0] for t in tokens_by_block[BLOCKS[0]]])
for b in BLOCKS:
    n_p_b = np.array([t.shape[0] for t in tokens_by_block[b]])
    assert np.array_equal(n_p_b, n_p_ref), f"n_p differe entre blocs pour {b} -- stride inattendu"

print(f"\n[Etape 0] distribution de n_p (identique sur block_4/6/9, meme stride) : "
      f"min={n_p_ref.min()}, mediane={np.median(n_p_ref):.1f}, max={n_p_ref.max()}")
n_le2 = int((n_p_ref <= 2).sum())
print(f"[Etape 0] patchs avec n_p <= 2 : {n_le2} / {len(n_p_ref)}")
if n_le2 > 0:
    print("  -> SS_intra=0 par construction pour ces patchs (token = sa propre moyenne). "
          "eta2 sera rapporte avec ET sans eux si leur proportion est notable.")


print(f"\n=== ETAPE 1a / 3B — controle coherence avec le H5 (bloc {BLOCKS[0]}, "
      f"{N_TOL_CONTROL} patchs) ===")
rng_check = np.random.default_rng(1)
check_idx = rng_check.choice(len(df), size=N_TOL_CONTROL, replace=False)
max_abs_diff = 0.0
for i in check_idx:
    raw_mean = tokens_by_block[BLOCKS[0]][i].mean(axis=0)
    norm = np.linalg.norm(raw_mean)
    z_p_norm = raw_mean / norm if norm > 1e-8 else raw_mean
    h5_vec = h5_vecs[BLOCKS[0]][i]
    diff = np.abs(z_p_norm - h5_vec).max()
    max_abs_diff = max(max_abs_diff, diff)
    h5_check_rows.append(dict(patch_idx=int(i), max_abs_diff=float(diff)))
print(f"  ecart max (L_inf) sur {N_TOL_CONTROL} patchs : {max_abs_diff:.2e}")
H5_CHECK_OK = max_abs_diff < 1e-4
print(f"  => coherence H5 : {'OK' if H5_CHECK_OK else 'ECHEC -- ARRET, pipeline non reproduit a l identique'}")
if not H5_CHECK_OK:
    raise SystemExit("Le pipeline reproduit ne correspond pas au H5 -- verifier coordonnees/pretraitement.")

pd.DataFrame(h5_check_rows).to_csv(OUT / "controle_coherence_h5.csv", index=False)
print("\n[OK] etape 0 / 1a / 3.D verifiees -- lancer eta_analysis_compute.py pour les etapes 2-4.")

import pickle
with open(OUT / "tokens_bruts.pkl", "wb") as fh:
    pickle.dump(dict(tokens_by_block=tokens_by_block, n_p=n_p_ref, texture=texture,
                      image=df.image.values, patch_uid=uid, id=df.id.values), fh, protocol=4)
print(f"Tokens bruts sauvegardes : {OUT / 'tokens_bruts.pkl'} "
      f"({sum(t.nbytes for b in BLOCKS for t in tokens_by_block[b]) / 1e6:.1f} Mo)")
