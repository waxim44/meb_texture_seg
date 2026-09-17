import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/dinov2_comparison"
OUT.mkdir(parents=True, exist_ok=True)

ORIG_H, ORIG_W = 768, 1280
PATCH_SZ = 128
ISO_H, ISO_W = 756, 1260
ISO_NH, ISO_NW = 54, 90

LAYER_INDICES = [2, 5, 8, 11]
LAYER_LABELS = {2: "layer_03", 5: "layer_06", 8: "layer_09", 11: "layer_12"}

MODELS = ["dinov2_vits14_reg", "dinov2_vitb14", "dinov2_vitb14_reg"]
PRIMARY_MODEL = "dinov2_vits14_reg"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TEST_POSITIONS = [(0, 0), (256, 384), (640, 256), (1152, 640)]

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(img_array):
    img = Image.fromarray(img_array)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((ISO_W, ISO_H), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)


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


print("=" * 78)
print("ETAPE 1 — Modeles DINOv2 disponibles (poids caches localement, ~/.cache/torch/hub)")
print("=" * 78)

arch_rows = []
for name in MODELS:
    m = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    n_params = sum(p.numel() for p in m.parameters())
    row = dict(model=name, embed_dim=m.embed_dim, patch_size=m.patch_size,
               depth=len(m.blocks), n_heads=m.blocks[0].attn.num_heads,
               n_registers=getattr(m, "num_register_tokens", 0), n_params=n_params)
    arch_rows.append(row)
    print(f"  {name:20s}  embed_dim={row['embed_dim']:4d}  patch={row['patch_size']:2d}  "
          f"depth={row['depth']:2d}  heads={row['n_heads']:2d}  "
          f"registres={row['n_registers']}  params={n_params/1e6:.1f}M")
    del m

print(f"\n  Pour reference, TextureSAM Hiera-Small (trunk) : ~34.3M params, "
      f"dim par stage 96/192/384/768 (blocks 4/6/9 -> dim=384).")
print(f"\n  CHOIX DU MODELE PRINCIPAL : {PRIMARY_MODEL}")
print(f"    Raison documentee : embed_dim=384 -- IDENTIQUE a celui des blocs 4/6/9 de")
print(f"    TextureSAM (meme dimensionnalite en sortie, comparaison d'echelle homogene) ;")
print(f"    22.1M params, ordre de grandeur le plus proche des trois variantes du trunk")
print(f"    Hiera-S (34.3M), avec des registres (4) qui evitent les artefacts d'attention")
print(f"    documentes dans le papier DINOv2 (tokens 'poubelle' a haute norme).")
print(f"    dinov2_vitb14(_reg) (768-dim, 86.6M) sont EGALEMENT evalues en complement")
print(f"    (deja extraits, cout marginal nul) pour verifier si un encodeur plus grand")
print(f"    change la conclusion.")

print(f"\n  API d'extraction : model.get_intermediate_layers(x, n=[...], reshape=False,")
print(f"    return_class_token=False, norm=True) -> ne retourne QUE les tokens spatiaux")
print(f"    (return_class_token=False exclut le CLS ; les 4 registres sont des tokens")
print(f"    SEPARES du forward interne, jamais concatenes aux tokens patch retournes")
print(f"    par cette API -- verifie ci-dessous par un controle de shape strict).")
print(f"    Couches extraites (sur 12 blocs, 0-indexees) : {LAYER_INDICES} -> "
      f"{list(LAYER_LABELS.values())} (early=3, mid=6/9, late=12).")

print(f"\n  Pretraitement : normalisation ImageNet standard (mean/std) -- meme convention")
print(f"    que le prepocessing SAM. Resize choisi : ISO {ISO_H}x{ISO_W} (grille "
      f"{ISO_NH}x{ISO_NW}={ISO_NH*ISO_NW} tokens), MULTIPLE DE 14 (patch_size) et qui")
print(f"    PRESERVE le ratio original {ORIG_H}x{ORIG_W} ({ORIG_H/ORIG_W:.4f} == "
      f"{ISO_H/ISO_W:.4f}), contrairement au resize CARRE anisotrope de SAM (1024x1024).")
print(f"    Ce choix est deja documente/teste dans pipeline_dinov2_lp_loio.py.")

import pandas as pd
pd.DataFrame(arch_rows).to_csv(OUT / "architecture_dinov2.csv", index=False)


print("\n" + "=" * 78)
print(f"SANITY CHECK (shape) — {PRIMARY_MODEL}")
print("=" * 78)
model = torch.hub.load("facebookresearch/dinov2", PRIMARY_MODEL, verbose=False).eval().to(device)
dummy = torch.randn(1, 3, ISO_H, ISO_W).to(device)
with torch.no_grad():
    inter = model.get_intermediate_layers(dummy, n=LAYER_INDICES, reshape=False,
                                           return_class_token=False, norm=True)
D = inter[0].shape[-1]
all_ok = True
for i, li in enumerate(LAYER_INDICES):
    ok = inter[i].shape == (1, ISO_NH * ISO_NW, D)
    all_ok &= ok
    print(f"  block[{li}] -> shape {tuple(inter[i].shape)}  attendu (1,{ISO_NH*ISO_NW},{D})  "
          f"{'OK' if ok else 'ECHEC'}")
assert all_ok, "shape inattendue -- CLS/registres probablement melanges aux tokens spatiaux. ARRET."
print(f"  => {ISO_NH*ISO_NW} tokens exactement, dimension {D} -- CLS et {getattr(model,'num_register_tokens',0)} "
      "registres bien EXCLUS des tokens spatiaux retournes.")


print("\n" + "=" * 78)
print("VALIDATION LOCALISATION — carre blanc synthetique (AVANT extraction massive)")
print("=" * 78)

all_ok_coord = True
for col, row in TEST_POSITIONS:
    img = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)
    c2 = min(col + PATCH_SZ, ORIG_W)
    r2 = min(row + PATCH_SZ, ORIG_H)
    img[row:r2, col:c2] = 255

    tensor = preprocess(img).to(device)
    with torch.no_grad():
        inter = model.get_intermediate_layers(tensor, n=LAYER_INDICES, reshape=False,
                                               return_class_token=False, norm=True)

    tokens_flat = inter[-1][0]
    tokens_grid = tokens_flat.reshape(ISO_NH, ISO_NW, -1).cpu().float().numpy()
    norms = np.linalg.norm(tokens_grid, axis=-1)

    tx1, ty1, tx2, ty2 = coord_to_tokens(col, row, c2, r2, ISO_NH, ISO_NW)
    mask = np.zeros((ISO_NH, ISO_NW), dtype=bool)
    mask[ty1:ty2, tx1:tx2] = True
    mean_patch = float(norms[mask].mean())
    mean_bg = float(norms[~mask].mean())
    diff_ratio = abs(mean_patch - mean_bg) / (mean_bg + 1e-9)
    ok = diff_ratio >= 0.01
    all_ok_coord &= ok
    status = "OK" if diff_ratio >= 0.05 else ("FAIBLE" if ok else "FAIL -- DECALAGE")
    print(f"  (col={col:4d},row={row:3d}) tokens[{ty1}:{ty2},{tx1}:{tx2}] "
          f"norme_patch={mean_patch:.3f} norme_bg={mean_bg:.3f} diff={diff_ratio*100:.1f}%  {status}")

print(f"\n  => VALIDATION LOCALISATION : {'FIABLE, on continue' if all_ok_coord else 'ECHEC -- ARRET'}")
assert all_ok_coord, "Decalage de coordonnees detecte -- arret avant extraction massive."

print(f"\n[OK] Etape 1 terminee et documentee. Ecrit : {OUT}/architecture_dinov2.csv")
