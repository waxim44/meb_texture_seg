import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/eta_analysis"
BLOCKS = ["block_4", "block_6", "block_9"]

with open(OUT / "tokens_bruts.pkl", "rb") as fh:
    D = pickle.load(fh)
tokens_by_block = D["tokens_by_block"]
n_p = D["n_p"]
texture = D["texture"]
image = D["image"]
P = len(n_p)
print(f"[chargement] {P} patchs, n_p min/median/max = {n_p.min()}/{np.median(n_p):.1f}/{n_p.max()}")


def decompose(tokens_by_patch, patch_mask=None):
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
    err_d = abs(ss_inter_d.sum() - ss_inter) / ss_inter, abs(ss_total_d.sum() - ss_total) / ss_total

    return dict(N=int(N), n_patchs=len(idx), ss_intra=ss_intra, ss_inter=ss_inter,
                ss_total=ss_total, eta2=eta2, identity_rel_err=err,
                ss_intra_p=ss_intra_p, disp_p=ss_intra_p / n_p_sub, idx=idx,
                ss_inter_d=ss_inter_d, ss_total_d=ss_total_d,
                eta2_d=ss_inter_d / ss_total_d, dimwise_check_err=err_d)


print("\n=== ETAPE 2-3A : SS_total = SS_inter + SS_intra, eta2 par bloc ===")
results = {}
summary_rows = []
for b in BLOCKS:
    r = decompose(tokens_by_block[b])
    results[b] = r
    print(f"  {b} : SS_total={r['ss_total']:.4e}  SS_inter={r['ss_inter']:.4e}  "
          f"SS_intra={r['ss_intra']:.4e}  erreur_identite={r['identity_rel_err']:.2e}  "
          f"eta2={r['eta2']:.4f}  (1-eta2={1-r['eta2']:.4f})")
    assert r["identity_rel_err"] < 1e-8, f"IDENTITE ECHOUEE pour {b} -- ARRET, resultat non fiable"
    assert r["ss_intra"] >= 0 and r["ss_inter"] >= 0 and r["ss_total"] >= 0
    assert 0 <= r["eta2"] <= 1
    assert r["N"] == n_p.sum()
    dc0, dc1 = r["dimwise_check_err"]
    assert dc0 < 1e-8 and dc1 < 1e-8, f"decomposition par dimension incoherente pour {b}"
    summary_rows.append(dict(block=b, n_p_min=int(n_p.min()), n_p_median=float(np.median(n_p)),
                              n_p_max=int(n_p.max()), SS_total=r["ss_total"], SS_inter=r["ss_inter"],
                              SS_intra=r["ss_intra"], erreur_identite=r["identity_rel_err"],
                              eta2=r["eta2"], un_moins_eta2=1 - r["eta2"]))
print("  => TOUTES les identites et controles de bon sens sont VERIFIES.")

pd.DataFrame(summary_rows).to_csv(OUT / "resume_eta2_par_bloc.csv", index=False)


print("\n=== ETAPE 4.3 : eta2 par texture (block_4) ===")
texture_rows = []
for b in BLOCKS:
    for t in ["Granuleux", "Sableux"]:
        mask = texture == t
        rt = decompose(tokens_by_block[b], patch_mask=mask)
        assert rt["identity_rel_err"] < 1e-8
        texture_rows.append(dict(block=b, texture=t, n_patchs=rt["n_patchs"],
                                  eta2=rt["eta2"], SS_inter=rt["ss_inter"], SS_intra=rt["ss_intra"]))
        print(f"  {b:10s} {t:10s} : n={rt['n_patchs']:3d}  eta2={rt['eta2']:.4f}")
texture_df = pd.DataFrame(texture_rows)
texture_df.to_csv(OUT / "eta2_par_texture.csv", index=False)


print("\n=== ETAPE 4.5 : correlation disp(p) vs |residu Ridge D50| (block_4) ===")
res_path = ROOT / "outputs/rsa_granulometrie/ridge_residuals/residus_oof.csv"
res = pd.read_csv(res_path)
res_d50_b4 = res[(res.block == "block_4") & (res.metric == "D50")].set_index("patch_uid")

disp_b4 = results["block_4"]["disp_p"]
patch_uid = D["patch_uid"]
df_link = pd.DataFrame(dict(patch_uid=patch_uid, disp=disp_b4, texture=texture, image=image))
df_link = df_link.merge(res_d50_b4[["y_true", "y_pred_oof", "residual"]], on="patch_uid", how="inner")
df_link["abs_residual"] = df_link.residual.abs()
assert len(df_link) == P, "fusion residus incomplete -- verifier alignement patch_uid"

rho, pval = spearmanr(df_link.disp, df_link.abs_residual)
print(f"  Spearman(disp(p), |residu D50|) = {rho:.3f}  (p={pval:.2e}, n={len(df_link)})")
df_link.to_csv(OUT / "dispersion_vs_residus.csv", index=False)


import pickle as pkl
with open(OUT / "resultats_calcul.pkl", "wb") as fh:
    pkl.dump(dict(results=results, texture=texture, image=image, n_p=n_p,
                  patch_uid=patch_uid, spearman_rho=rho, spearman_p=pval), fh, protocol=4)

print(f"\n[OK] etapes 2-4 terminees. Ecrit dans {OUT}")
print("Lancer eta_analysis_figures.py pour les figures + rapport.txt")
