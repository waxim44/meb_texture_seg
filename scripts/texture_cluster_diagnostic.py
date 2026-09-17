from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
CSV = ROOT / "outputs/granulometrie_exploration/granulo_patches.csv"
OUT = ROOT / "outputs/rsa_granulometrie/texture_cluster_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

SPACES = {
    "TextureSAM_block_4": dict(h5=ROOT / "data/feature_database/database_meb_ouassim.h5", key="block_4"),
    "DINOv2_layer_03":    dict(h5=ROOT / "data/feature_database/dinov2_dinov2_vits14_reg_iso.h5", key="layer_03"),
}
CG, CS = "#b55e07", "#1f77b4"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 14,
    "legend.fontsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

df = pd.read_csv(CSV)
df = df[df.id.isin([7, 8])].reset_index(drop=True)
uid = df.patch_uid.values
texture = np.where(df.id.values == 7, "Granuleux", "Sableux")
print(f"N patchs = {len(df)}  (Granuleux={int((texture=='Granuleux').sum())}, "
      f"Sableux={int((texture=='Sableux').sum())})")


def decompose_by_texture(Z, texture):
    mu_global = Z.mean(axis=0)
    ss_total = ((Z - mu_global) ** 2).sum()

    ss_inter = 0.0
    ss_intra = 0.0
    centers = {}
    radii = {}
    for t in ["Granuleux", "Sableux"]:
        mask = texture == t
        n_t = mask.sum()
        mu_t = Z[mask].mean(axis=0)
        centers[t] = mu_t
        ss_inter += n_t * ((mu_t - mu_global) ** 2).sum()
        diffs = Z[mask] - mu_t
        ss_intra += (diffs ** 2).sum()
        radii[t] = np.sqrt((diffs ** 2).sum(axis=1).mean())

    err = abs(ss_total - (ss_inter + ss_intra)) / ss_total
    eta2 = ss_inter / ss_total
    dist_centers = np.linalg.norm(centers["Granuleux"] - centers["Sableux"])
    mean_radius = (radii["Granuleux"] + radii["Sableux"]) / 2
    ratio = dist_centers / mean_radius

    return dict(mu_global=mu_global, ss_inter=ss_inter, ss_intra=ss_intra, ss_total=ss_total,
                eta2=eta2, identity_rel_err=err, centers=centers, radii=radii,
                dist_centers=dist_centers, mean_radius=mean_radius, ratio=ratio)


rows = []
for space_name, spec in SPACES.items():
    with h5py.File(spec["h5"], "r") as f:
        cids = f["metadata/category_ids"][:]
        names = np.array([x.decode() for x in f["metadata/image_names"][:]])
        assert (cids[uid] == df.id.values).all() and (names[uid] == df.image.values).all(),\
            f"desalignement H5<->CSV pour {space_name}"
        Z = f["features"][spec["key"]][:][uid].astype(np.float64)

    print(f"\n=== {space_name} (dim={Z.shape[1]}) ===")
    r = decompose_by_texture(Z, texture)
    print(f"  SS_total={r['ss_total']:.4e}  SS_inter={r['ss_inter']:.4e}  SS_intra={r['ss_intra']:.4e}")
    print(f"  erreur_identite={r['identity_rel_err']:.2e}")
    assert r["identity_rel_err"] < 1e-8, f"IDENTITE ECHOUEE pour {space_name} -- ARRET"
    assert 0 <= r["eta2"] <= 1
    print(f"  eta2_texture = {r['eta2']:.4f}")
    print(f"  distance centres = {r['dist_centers']:.4f}  rayon_G={r['radii']['Granuleux']:.4f}  "
          f"rayon_S={r['radii']['Sableux']:.4f}  ratio={r['ratio']:.4f}")


    pca = PCA(n_components=2, random_state=0)
    emb = pca.fit_transform(Z)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  PCA 2D : variance expliquee = {var_exp*100:.1f}%")

    rows.append(dict(space=space_name, dim=Z.shape[1], eta2_texture=r["eta2"],
                      erreur_identite=r["identity_rel_err"], dist_centres=r["dist_centers"],
                      rayon_Granuleux=r["radii"]["Granuleux"], rayon_Sableux=r["radii"]["Sableux"],
                      ratio=r["ratio"], variance_expliquee_pca2d=var_exp))


    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for t, c in [("Granuleux", CG), ("Sableux", CS)]:
        mask = texture == t
        ax.scatter(emb[mask, 0], emb[mask, 1], s=26, color=c, alpha=0.65, edgecolor="none", label=t)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(space_name.replace("_", " "))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_pca_{space_name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig_pca_{space_name}.png", bbox_inches="tight")
    plt.close(fig)

table = pd.DataFrame(rows)
table.to_csv(OUT / "resume_diagnostic.csv", index=False)
print("\n=== Tableau recapitulatif ===")
print(table.round(4).to_string(index=False))
print(f"\nEcrit dans {OUT}")
