import os
import json
import numpy as np
import h5py
from PIL import Image
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import opening, disk

H5_PATH = "data/feature_database/database_meb_ouassim.h5"
IMG_DIR = "Image_Ouassim"
OUT_DIR = "outputs/rsa_granulometrie"
os.makedirs(OUT_DIR, exist_ok=True)

BLOCK_KEYS = [f"block_{i}" for i in range(16)] + [
    "stage_1_fpn", "stage_2_fpn", "stage_3_fpn", "stage_4_fpn"
]
BLOCK_LABELS = {**{f"block_{i}": f"B{i}" for i in range(16)},
                "stage_1_fpn": "FPN1", "stage_2_fpn": "FPN2",
                "stage_3_fpn": "FPN3", "stage_4_fpn": "FPN4"}

R_MAX = 25
MIN_PATCHES_INTRA = 8

rng = np.random.default_rng(0)


print(f"[H5] chemin utilise : {os.path.abspath(H5_PATH)}")
with h5py.File(H5_PATH, "r") as f:
    cat_ids = f["metadata/category_ids"][:]
    image_names = np.array([s.decode() for s in f["metadata/image_names"][:]])
    positions = f["metadata/positions"][:]
    block0_mean = f["features/block_0"][:].mean()
    feats = {k: f[f"features/{k}"][:] for k in BLOCK_KEYS}

print(f"[H5] mean(block_0) = {block0_mean:.6f}  (info seule, block_0 non privilegie)")

mask7 = cat_ids == 7
N = int(mask7.sum())
idx7 = np.where(mask7)[0]
print(f"[Etape 1] N patchs granuleux (category_id=7) = {N}")

img7 = image_names[idx7]
pos7 = positions[idx7]


uniq_imgs, counts = np.unique(img7, return_counts=True)
order = np.argsort(-counts)
print(f"[Etape 1] {len(uniq_imgs)} images distinctes contiennent des patchs granuleux")
rep_lines = ["image_name,n_patches_granuleux"]
for u, c in zip(uniq_imgs[order], counts[order]):
    rep_lines.append(f"{u},{c}")
with open(f"{OUT_DIR}/repartition_par_image.csv", "w") as fp:
    fp.write("\n".join(rep_lines))
print("  -> ecrit outputs/rsa_granulometrie/repartition_par_image.csv")


ref_img_name = "060525-JPB-MEB-EIHNValves-Ech1-ZigZag20002.tif"
ref_img_path = os.path.join(IMG_DIR, ref_img_name)
ref_img_raw = Image.open(ref_img_path)
ref_arr = np.array(ref_img_raw)
print(f"[Etape 0] verif source : {ref_img_path}, mode={ref_img_raw.mode}, mean={ref_arr.mean():.2f} "
      f"(reference Ouassim brut grayscale, PatchTagger serait RGB contraste ~106 -> ne pas utiliser)")
assert ref_img_raw.mode == "L", "image de reference n'est pas en niveaux de gris -> verifier la source !"
assert 75 < ref_arr.mean() < 95, "mean pixel hors plage Ouassim attendue sur l'image de reference !"


_img_cache = {}
def load_gray(name):
    if name not in _img_cache:
        p = os.path.join(IMG_DIR, name)
        _img_cache[name] = np.array(Image.open(p).convert("L")).astype(np.float64)
    return _img_cache[name]


print(f"[Etape 2] tamisage morphologique r=1..{R_MAX} (ouverture, disque), "
      "surface = somme d'intensite (niveaux de gris) restante")

def granulometry_vector(crop):
    surf0 = crop.sum()
    surfaces = [surf0]
    for r in range(1, R_MAX + 1):
        opened = opening(crop, disk(r))
        surfaces.append(opened.sum())
    surfaces = np.array(surfaces)

    total_loss = surf0 - surfaces[-1]
    if total_loss <= 1e-9:

        cum = np.zeros(R_MAX + 1)
    else:
        cum = (surf0 - surfaces) / total_loss

    radii = np.arange(0, R_MAX + 1)

    def interp_radius(target):

        if cum[-1] < target:
            return float(R_MAX)
        j = np.searchsorted(cum, target)
        if j == 0:
            return 0.0
        r0, r1 = radii[j - 1], radii[j]
        c0, c1 = cum[j - 1], cum[j]
        if c1 == c0:
            return float(r1)
        return float(r0 + (target - c0) / (c1 - c0) * (r1 - r0))

    d10 = interp_radius(0.10)
    d50 = interp_radius(0.50)
    d90 = interp_radius(0.90)
    span = (d90 - d10) / d50 if d50 > 1e-9 else 0.0


    density = np.diff(cum)
    density = np.clip(density, 0, None)
    if density.sum() > 1e-9:
        p = density / density.sum()
        mids = radii[1:]
        mean_r = (p * mids).sum()
        var_r = (p * (mids - mean_r) ** 2).sum()
        std_r = np.sqrt(var_r) if var_r > 1e-9 else 1e-9
        skew = ((p * (mids - mean_r) ** 3).sum()) / (std_r ** 3)
    else:
        skew = 0.0

    a_gran = total_loss / surf0 if surf0 > 1e-9 else 0.0
    sigma_local = float(crop.std())

    return d50, span, a_gran, skew, sigma_local


rows = []
for k in range(N):
    name = img7[k]
    x1, y1, x2, y2 = pos7[k].astype(int)
    img = load_gray(name)
    crop = img[y1:y2, x1:x2]
    rows.append(granulometry_vector(crop))

G = np.array(rows)
metric_names = ["D50_px", "Span", "A_gran", "Skewness", "sigma_local"]

print("[Etape 2] granulometrie calculee pour les", N, "patchs granuleux")


d50_all = G[:, 0]
d50_stats = dict(min=float(d50_all.min()), max=float(d50_all.max()),
                  median=float(np.median(d50_all)), std=float(d50_all.std()))
print(f"[Etape 2.4] D50 (px) : min={d50_stats['min']:.2f} max={d50_stats['max']:.2f} "
      f"median={d50_stats['median']:.2f} std={d50_stats['std']:.2f}")

hist, edges = np.histogram(d50_all, bins=15)
with open(f"{OUT_DIR}/d50_histogram.csv", "w") as fp:
    fp.write("bin_left,bin_right,count\n")
    for c, l, r in zip(hist, edges[:-1], edges[1:]):
        fp.write(f"{l:.3f},{r:.3f},{c}\n")

VARIETY_OK = d50_stats["std"] > 1e-6 and (d50_stats["max"] - d50_stats["min"]) > 1.0
if not VARIETY_OK:
    print("!!! ATTENTION : quasiment aucune variete de granulometrie (D50) parmi les "
          "patchs granuleux -> la RSA n'aura rien a correler. Signale a l'utilisateur.")
else:
    print(f"[Etape 2.4] variete suffisante (std={d50_stats['std']:.2f} px, "
          f"range={d50_stats['max']-d50_stats['min']:.2f} px) -> RSA a un objet.")

np.save(f"{OUT_DIR}/G_physique.npy", G)


G_mean = G.mean(axis=0)
G_std = G.std(axis=0)
G_std[G_std < 1e-9] = 1.0
Gz = (G - G_mean) / G_std

D_phys = squareform(pdist(Gz, metric="euclidean"))
np.save(f"{OUT_DIR}/D_phys.npy", D_phys)
print("[Etape 2] D_phys (N x N, euclidienne sur metriques Z-scorees) calculee")


print("[Etape 3] calcul des distances latentes (cosinus) pour les 20 blocs")
D_lat = {}
for bk in BLOCK_KEYS:
    Z = feats[bk][idx7]
    D_lat[bk] = squareform(pdist(Z, metric="cosine"))
print("[Etape 3] termine")


iu = np.triu_indices(N, k=1)
phys_flat = D_phys[iu]

results = []


img_to_idx = {}
for local_i, name in enumerate(img7):
    img_to_idx.setdefault(name, []).append(local_i)
kept_images = {name: idxs for name, idxs in img_to_idx.items() if len(idxs) >= MIN_PATCHES_INTRA}
excluded_images = {name: idxs for name, idxs in img_to_idx.items() if len(idxs) < MIN_PATCHES_INTRA}
print(f"[Etape 4] images retenues (>= {MIN_PATCHES_INTRA} patchs) pour l'intra : "
      f"{len(kept_images)} / {len(img_to_idx)} (exclues : {len(excluded_images)})")

for bk in BLOCK_KEYS:
    lat_flat = D_lat[bk][iu]
    rs_inter, p_inter = stats.spearmanr(phys_flat, lat_flat)

    rs_per_image = []
    for name, idxs in kept_images.items():
        idxs = np.array(idxs)
        sub_phys = D_phys[np.ix_(idxs, idxs)]
        sub_lat = D_lat[bk][np.ix_(idxs, idxs)]
        n_sub = len(idxs)
        iu_sub = np.triu_indices(n_sub, k=1)
        rs_img, _ = stats.spearmanr(sub_phys[iu_sub], sub_lat[iu_sub])
        if not np.isnan(rs_img):
            rs_per_image.append(rs_img)
    rs_intra = float(np.mean(rs_per_image)) if rs_per_image else float("nan")

    results.append(dict(block=bk, label=BLOCK_LABELS[bk],
                         rs_inter=float(rs_inter), p_inter=float(p_inter),
                         rs_intra=rs_intra, n_images_intra=len(rs_per_image)))
    print(f"  {BLOCK_LABELS[bk]:>5s}  Rs_inter={rs_inter:+.3f} (p={p_inter:.1e})  "
          f"Rs_intra={rs_intra:+.3f}  (n_img={len(rs_per_image)})")

with open(f"{OUT_DIR}/rsa_results.json", "w") as fp:
    json.dump(dict(results=results, d50_stats=d50_stats, variety_ok=VARIETY_OK,
                    N=N, n_images_total=len(img_to_idx),
                    n_images_kept_intra=len(kept_images),
                    n_images_excluded_intra=len(excluded_images),
                    min_patches_intra=MIN_PATCHES_INTRA, R_MAX=R_MAX,
                    excluded_images={k: len(v) for k, v in excluded_images.items()}),
              fp, indent=2)

print(f"[Etape 4] resultats ecrits dans {OUT_DIR}/rsa_results.json")
print("[OK] etapes 0-4 terminees. Lancer rsa_granulometrie_outputs.py pour les figures/tableaux.")
