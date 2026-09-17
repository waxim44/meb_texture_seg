import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import cv2
from PIL import Image
from scipy.stats import skew as _scipy_skew

ROOT = Path("/home/aidouni/meb_texture_seg")
H5 = ROOT / "data/feature_database/database_meb_ouassim.h5"
IMG_DIR = ROOT / "Image_Ouassim"
PATCH_DIR = ROOT / "PatchTagger_Output/patches"
OUT = ROOT / "outputs/granulometrie_exploration"
OUT.mkdir(parents=True, exist_ok=True)

R_MAX = 25
NM_PER_PX = None

TEX = {7: "Granuleux", 8: "Sableux"}


def load_meta():
    with h5py.File(H5, "r") as f:
        cids = f["metadata/category_ids"][:]
        names = np.array([x.decode() for x in f["metadata/image_names"][:]])
        pos = f["metadata/positions"][:].astype(int)
    rows = []
    for uid, (c, nm, p) in enumerate(zip(cids, names, pos)):
        if int(c) in TEX:
            rows.append(dict(patch_uid=uid, id=int(c), texture=TEX[int(c)],
                             image=nm, x_min=p[0], y_min=p[1], x_max=p[2], y_max=p[3]))
    return pd.DataFrame(rows)


def crop_of(row, cache):
    nm = row["image"]
    if nm not in cache:
        cache[nm] = np.array(Image.open(IMG_DIR / nm).convert("L"))
    big = cache[nm]
    return big[row["y_min"]:row["y_max"], row["x_min"]:row["x_max"]]


def granulometry(crop):
    g = crop.astype(np.float64)
    S0 = g.sum()
    surf = np.empty(R_MAX + 1)
    surf[0] = S0
    for r in range(1, R_MAX + 1):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(g, cv2.MORPH_OPEN, se, borderType=cv2.BORDER_REPLICATE)
        surf[r] = opened.sum()

    loss = surf[:-1] - surf[1:]
    loss = np.clip(loss, 0, None)
    total = loss.sum()
    if total <= 0:

        return dict(D10=np.nan, D50=np.nan, D90=np.nan, Span=np.nan,
                    Skewness=np.nan, spectrum=loss)
    pdf = loss / total
    cdf = np.cumsum(pdf)
    radii = np.arange(1, R_MAX + 1)

    def dq(q):

        if cdf[0] >= q:
            return float(radii[0])
        if cdf[-1] < q:
            return float(radii[-1])
        i = np.searchsorted(cdf, q)
        c0, c1 = cdf[i - 1], cdf[i]
        r0, r1 = radii[i - 1], radii[i]
        return float(r0 + (q - c0) / (c1 - c0) * (r1 - r0))

    D10, D50, D90 = dq(0.10), dq(0.50), dq(0.90)
    span = (D90 - D10) / D50 if D50 > 0 else np.nan

    mu = np.sum(radii * pdf)
    var = np.sum((radii - mu) ** 2 * pdf)
    sk = np.sum((radii - mu) ** 3 * pdf) / var ** 1.5 if var > 0 else np.nan
    return dict(D10=D10, D50=D50, D90=D90, Span=span, Skewness=sk, spectrum=loss)


def sanity_check(df, cache):
    print("=== Etape 0 : controle crops vs patches/<id>/*.tif ===")
    ok = True
    for tid in (7, 8):
        sub = df[df.id == tid].iloc[[0, len(df[df.id == tid]) // 2]]
        for _, row in sub.iterrows():
            crop = crop_of(row, cache)
            stem = row["image"][:-4]
            col, r = row["x_min"] // 128, row["y_min"] // 128
            ptif = PATCH_DIR / str(tid) / f"{stem}_({r}_{col}).tif"
            if not ptif.exists():
                print(f"  [{TEX[tid]}] {ptif.name} ABSENT"); ok = False; continue
            pim = np.array(Image.open(ptif).convert("L"))
            d = np.abs(crop.astype(int) - pim.astype(int)).mean() if pim.shape == crop.shape else np.nan
            print(f"  [{TEX[tid]:9s}] {row['image']}  pos=({row.x_min},{row.y_min})  "
                  f"{ptif.name}  |crop-patch|_moy={d:.3f}")
            ok = ok and (d == 0)
    print("  -> crops = patches PatchTagger :", "OK" if ok else "ECHEC")
    return ok


def main():
    df = load_meta()
    print(f"Patchs : Granuleux={int((df.id==7).sum())}  Sableux={int((df.id==8).sum())}  "
          f"total={len(df)}")
    print(f"Source crops : {IMG_DIR}  (images MEB originales, niveaux de gris)")
    cache = {}
    if not sanity_check(df, cache):
        print("Controle crops echoue -> arret."); sys.exit(1)

    print(f"\n=== Etape 1 : granulometrie grayscale (R_MAX={R_MAX} px, sans seuil) ===")
    recs = []
    spectra = []
    for k, (_, row) in enumerate(df.iterrows()):
        crop = crop_of(row, cache)
        m = granulometry(crop)
        recs.append({**{c: row[c] for c in
                        ["patch_uid", "id", "texture", "image",
                         "x_min", "y_min", "x_max", "y_max"]},
                     "D50": m["D50"], "D10": m["D10"], "D90": m["D90"],
                     "Span": m["Span"], "Skewness": m["Skewness"],
                     "sigma_local": float(crop.std())})
        spectra.append(m["spectrum"])
        if (k + 1) % 100 == 0 or k + 1 == len(df):
            print(f"  {k+1}/{len(df)}")

    out = pd.DataFrame(recs)
    out.to_csv(OUT / "granulo_patches.csv", index=False)
    np.save(OUT / "granulo_spectra.npy", np.vstack(spectra))
    print("\nCSV :", OUT / "granulo_patches.csv")
    print("Spectres :", OUT / "granulo_spectra.npy", "  (shape", np.vstack(spectra).shape, ")")
    print("\nNaN par colonne :")
    print(out[["D50", "D10", "D90", "Span", "Skewness", "sigma_local"]].isna().sum().to_string())
    print("\nApercu (moyennes par texture) :")
    print(out.groupby("texture")[["D10", "D50", "D90", "Span", "Skewness", "sigma_local"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
