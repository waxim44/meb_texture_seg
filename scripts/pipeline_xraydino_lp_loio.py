#!/usr/bin/env python3
"""
Pipeline XRay-DINO ViT-L/16 — LP LOIO
Extraction 4 couches intermédiaires (une par BlockChunk) → H5 → LP LOIO → comparaison SAM.
Résolution native 768×1280 (patch_size=16 → 48×80=3840 tokens).
"""
import sys, struct, argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT      = Path("/home/aidouni/meb_texture_seg")
REPO_DIR  = Path("/home/aidouni/.cache/torch/hub/facebookresearch_dinov2_main")
CKPT      = ROOT / "checkpoints/xray_dino_vitl16_pretrained-ad31c2b0.pth"
H5_SAM    = ROOT / "data/feature_database/database_meb_ouassim.h5"
H5_OUT    = ROOT / "data/feature_database/xraydino_vitl16_native.h5"
IMG_DIR   = ROOT / "Image_Ouassim"
OUT_DIR   = ROOT / "output_ouassim/xraydino_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTURES  = [1, 3, 4, 5, 6, 7, 9]
TNAMES    = {1:"Tot.homogène", 3:"Faisceaux", 4:"Filaments", 5:"Strat.rect",
             6:"Strat.sin",   7:"Granuleux", 9:"Trou"}
ORIG_H, ORIG_W = 768, 1280
PATCH_PX       = 128
PATCH_SIZE     = 16          # ViT-L patch size
IMG_H, IMG_W   = 768, 1280  # résolution native (multiple exact de 16)
N_TOKENS_H     = IMG_H // PATCH_SIZE   # 48
N_TOKENS_W     = IMG_W // PATCH_SIZE   # 80
LAYER_NAMES    = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
PCA_DIM, C_LP, SEED = 50, 1.0, 42

MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# Résultats SAM meilleur bloc par texture
SAM_RECALL = {1:1.000, 3:0.790, 4:0.837, 5:0.502, 6:0.524, 7:0.860, 9:0.735}

# ─── Utilitaires ──────────────────────────────────────────────────────────────
def read_tiff_gray(path):
    with open(path, "rb") as f:
        data = f.read()
    bo = "<" if data[:2] == b"II" else ">"
    ifd = struct.unpack(bo+"I", data[4:8])[0]
    pos = ifd
    n   = struct.unpack(bo+"H", data[pos:pos+2])[0]; pos += 2
    tags = {}
    for _ in range(n):
        e = data[pos:pos+12]; pos += 12
        tag, dtype, _ = struct.unpack(bo+"HHI", e[:8]); v = e[8:12]
        if   dtype == 3: v = struct.unpack(bo+"H", v[:2])[0]
        elif dtype == 4: v = struct.unpack(bo+"I", v)[0]
        tags[tag] = v
    w, h = tags[256], tags[257]
    with open(path, "rb") as f:
        f.seek(tags[273])
        return np.frombuffer(f.read(h*w), dtype=np.uint8).reshape(h, w)

def preprocess(img_name):
    img = read_tiff_gray(IMG_DIR / img_name)
    pil = Image.fromarray(img)
    if pil.size != (IMG_W, IMG_H):
        pil = pil.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    t   = torch.from_numpy(np.stack([arr]*3, axis=0))
    return ((t - MEAN_T) / STD_T).unsqueeze(0).cuda()

def coord_to_tokens(x_min, y_min, x_max, y_max):
    """Patch pixel coords → token grid indices."""
    tx0 = int(x_min * N_TOKENS_W / ORIG_W)
    ty0 = int(y_min * N_TOKENS_H / ORIG_H)
    tx1 = max(tx0+1, int(x_max * N_TOKENS_W / ORIG_W))
    ty1 = max(ty0+1, int(y_max * N_TOKENS_H / ORIG_H))
    return ty0, ty1, tx0, tx1

def pool_patch(feat_grid, ty0, ty1, tx0, tx1):
    """feat_grid: (H_tok, W_tok, D) → mean pool sur la région du patch."""
    return feat_grid[ty0:ty1, tx0:tx1, :].reshape(-1, feat_grid.shape[-1]).mean(0)

# ─── LP LOIO ──────────────────────────────────────────────────────────────────
def loio_recall(X, y_bin, stems):
    recalls = []
    for stem in sorted(set(stems)):
        te, tr = stems==stem, stems!=stem
        if y_bin[te].sum()==0 or len(np.unique(y_bin[tr]))<2:
            continue
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]
        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(C=C_LP, class_weight="balanced",
                                 max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        tp = int(((pred==1)&(y_te==1)).sum())
        fn = int(((pred==0)&(y_te==1)).sum())
        recalls.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
    return np.array(recalls)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    # ── Chargement métadonnées SAM H5 ────────────────────────────────────────
    print("Chargement métadonnées SAM H5...")
    with h5py.File(H5_SAM, "r") as f:
        all_cat  = f["metadata"]["category_ids"][:]
        all_imgs = np.array([x.decode() for x in f["metadata"]["image_names"][:]])
        all_pos  = f["metadata"]["positions"][:]

    mask     = np.isin(all_cat, TEXTURES)
    cat_ids  = all_cat[mask]
    img_names= all_imgs[mask]
    positions= all_pos[mask]
    stems    = np.array([n.replace(".tif","") for n in img_names])
    N        = mask.sum()
    print(f"  {N} patches | {len(set(img_names))} images")

    # ── Extraction XRay-DINO ─────────────────────────────────────────────────
    if not args.skip_extract or not H5_OUT.exists():
        print("\nChargement XRay-DINO ViT-L/16...")
        model = torch.hub.load(str(REPO_DIR), "xray_dino_vitl16",
                               source="local", weights=str(CKPT))
        model = model.cuda().eval()
        print("  Modèle OK — embed_dim=1024, patch_size=16, 4 BlockChunks")

        feats = {ln: np.zeros((N, 1024), dtype=np.float32) for ln in LAYER_NAMES}
        unique_imgs = sorted(set(img_names))

        print(f"\nExtraction sur {len(unique_imgs)} images...")
        for ii, img_name in enumerate(unique_imgs):
            idx = np.where(img_names == img_name)[0]
            x   = preprocess(img_name)
            with torch.no_grad():
                layers = model.get_intermediate_layers(
                    x, n=4, reshape=False, return_class_token=False, norm=True)

            for pi, gi in enumerate(idx):
                xmin, ymin, xmax, ymax = positions[gi]
                ty0, ty1, tx0, tx1 = coord_to_tokens(xmin, ymin, xmax, ymax)
                for li, ln in enumerate(LAYER_NAMES):
                    grid = layers[li][0].reshape(N_TOKENS_H, N_TOKENS_W, 1024).cpu().numpy()
                    feats[ln][gi] = pool_patch(grid, ty0, ty1, tx0, tx1)

            if (ii+1) % 10 == 0:
                print(f"  {ii+1}/{len(unique_imgs)}")

        print(f"\nSauvegarde → {H5_OUT}")
        with h5py.File(H5_OUT, "w") as f:
            grp = f.create_group("features")
            for ln in LAYER_NAMES:
                grp.create_dataset(ln, data=feats[ln], compression="gzip")
            mg = f.create_group("metadata")
            mg.create_dataset("category_ids",  data=cat_ids)
            mg.create_dataset("image_names",
                              data=np.array([s.encode() for s in img_names]))
        del model
        torch.cuda.empty_cache()
    else:
        print(f"Skip extraction — chargement {H5_OUT}")
        with h5py.File(H5_OUT, "r") as f:
            feats = {ln: f["features"][ln][:] for ln in LAYER_NAMES}

    # ── LP LOIO ──────────────────────────────────────────────────────────────
    print("\nLP LOIO XRay-DINO (4 couches × 7 textures)...")
    results = {}   # results[texture][layer] = (mean, std, n_folds)

    for t in TEXTURES:
        y_bin = (cat_ids == t).astype(int)
        results[t] = {}
        for ln in LAYER_NAMES:
            r = loio_recall(feats[ln], y_bin, stems)
            results[t][ln] = (float(r.mean()) if len(r) else 0.0,
                              float(r.std())  if len(r) else 0.0,
                              len(r))

        best_ln = max(LAYER_NAMES, key=lambda l: (results[t][l][0], -results[t][l][1]))
        m, s, n = results[t][best_ln]
        print(f"  t{t} {TNAMES[t]:<15} → {best_ln}  recall={m:.3f}±{s:.3f} ({n} folds)")

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Texture':<16} {'XRay best':>12} {'±std':>7} {'SAM best':>10} {'Δ':>8}")
    print("-"*72)
    for t in sorted(results, key=lambda t: -SAM_RECALL[t]):
        best_ln = max(LAYER_NAMES, key=lambda l: (results[t][l][0], -results[t][l][1]))
        m, s, _  = results[t][best_ln]
        delta    = m - SAM_RECALL[t]
        print(f"{TNAMES[t]:<16} {m:>12.3f} {s:>7.3f} {SAM_RECALL[t]:>10.3f} {delta:>+8.3f}")
    print("="*72)

    # ── Plot comparatif ───────────────────────────────────────────────────────
    order   = sorted(TEXTURES, key=lambda t: -SAM_RECALL[t])
    labels  = [TNAMES[t] for t in order]
    sam_v   = [SAM_RECALL[t] for t in order]
    xrd_v   = [max(results[t][l][0] for l in LAYER_NAMES) for t in order]
    xrd_std = [results[t][max(LAYER_NAMES,
                key=lambda l:(results[t][l][0],-results[t][l][1]))][1] for t in order]

    x  = np.arange(len(order))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, sam_v, w, label="TextureSAM (Hiera-S)", color="#5B8DB8",
           edgecolor="white")
    xrd_err_hi = np.minimum(xrd_std, 1.0 - np.array(xrd_v))
    xrd_err_lo = np.minimum(xrd_std, np.array(xrd_v))
    ax.bar(x + w/2, xrd_v, w, label="XRay-DINO (ViT-L/16)", color="#E8843A",
           edgecolor="white",
           yerr=[xrd_err_lo, xrd_err_hi], capsize=4,
           error_kw=dict(lw=1.2, capthick=1.2, ecolor="black"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Recall LOIO (meilleur bloc)", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("TextureSAM vs XRay-DINO ViT-L/16 — Recall LP LOIO par texture",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out = OUT_DIR / "comparison_sam_vs_xraydino.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot sauvé : {out}")

    # ── Heatmap XRay-DINO (4 couches × 7 textures) ───────────────────────────
    mat = np.array([[results[t][ln][0] for t in TEXTURES] for ln in LAYER_NAMES])
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TEXTURES)))
    ax.set_xticklabels([TNAMES[t] for t in TEXTURES], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(LAYER_NAMES, fontsize=9)
    ax.set_title("XRay-DINO ViT-L/16 — Recall LP LOIO par couche et texture", fontsize=11)
    for i in range(4):
        for j, t in enumerate(TEXTURES):
            v = mat[i, j]
            col = "white" if v < 0.35 or v > 0.75 else "#222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=col)
    plt.colorbar(im, ax=ax, label="Recall", fraction=0.03, pad=0.03)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heatmap_xraydino.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap sauvée : {OUT_DIR/'heatmap_xraydino.png'}")

if __name__ == "__main__":
    main()
