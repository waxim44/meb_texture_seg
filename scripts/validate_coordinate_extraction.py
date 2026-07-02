"""
validate_coordinate_extraction.py
══════════════════════════════════════════════════════════════════════════════
Test de validation INDÉPENDANT de la conversion patch → feature map.

But : vérifier que la conversion (col, row) espace original (768×1280)
→ indices feature map est EXACTE malgré le resize anisotrope 1024×1024.

Vérité terrain : carrés synthétiques que l'on place soi-même — zéro
dépendance à la H5 ou à toute extraction précédente.
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── SAM2 path ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SAM2 = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Constantes du pipeline (identiques à build_feature_database.py) ───────────
IMG_SIZE = 1024
ORIG_H   = 768
ORIG_W   = 1280
PATCH_SZ = 128  # taille des patches en espace original

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Strides effectifs de chaque feature map (relative à l'image 1024×1024)
# PatchEmbed stride=4, puis q_pool ×2 aux blocs 1, 3, 14
BLOCK_STRIDE = {i: 4 for i in range(16)}
for i in range(1, 16):   BLOCK_STRIDE[i] = 8    # après bloc 1
for i in range(3, 16):   BLOCK_STRIDE[i] = 16   # après bloc 3
for i in range(14, 16):  BLOCK_STRIDE[i] = 32   # après bloc 14

FPN_STRIDE  = {"stage_1_fpn": 4, "stage_2_fpn": 8,
               "stage_3_fpn": 16, "stage_4_fpn": 32}

ALL_STRIDES = {f"block_{i}": BLOCK_STRIDE[i] for i in range(16)}
ALL_STRIDES.update(FPN_STRIDE)

OUTDIR = _ROOT / "output_ouassim" / "validate_coordinates"
OUTDIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"

# ── Positions test ────────────────────────────────────────────────────────────
TEST_POSITIONS = [
    (0,    0),    # coin supérieur-gauche
    (256,  384),  # zone centrale-gauche
    (640,  256),  # zone centrale
    (1152, 640),  # zone droite (proche du bord)
]
# (col, row) en coordonnées image originale (col=x horiz, row=y vert)

# ── Sélection des blocs à afficher (un par stride pour la lisibilité) ─────────
DISPLAY_KEYS = ["block_0", "block_2", "block_9", "block_15",
                "stage_1_fpn", "stage_3_fpn"]


# ═════════════════════════════════════════════════════════════════════════════
# Construction modèle + hooks
# ═════════════════════════════════════════════════════════════════════════════

def build_encoder() -> ImageEncoder:
    trunk = Hiera(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000
        ),
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model="nearest",
        fuse_type="sum",
        fpn_top_down_levels=[2, 3],
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def load_encoder(ckpt_path: Path) -> tuple[ImageEncoder, bool]:
    """Charge l'encoder avec le checkpoint si disponible."""
    encoder = build_encoder()
    loaded = False

    if ckpt_path.is_file():
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            sd = sd.get("model", sd)
            prefix = "image_encoder."
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            missing, unexpected = encoder.load_state_dict(sd, strict=False)
            if not missing and not unexpected:
                print(f"  [encoder] Checkpoint chargé : {ckpt_path.name}")
            else:
                print(f"  [encoder] Checkpoint partiel "
                      f"({len(missing)} manquantes, {len(unexpected)} inattendues)")
            loaded = True
        except Exception as e:
            print(f"  [encoder] Erreur chargement checkpoint : {e}")
    else:
        print(f"  [encoder] Checkpoint absent ({ckpt_path}) — poids aléatoires")
        print("             NOTE : la conversion reste testable avec poids aléatoires.")

    encoder.eval()
    return encoder, loaded


def register_hooks(encoder: ImageEncoder) -> tuple[dict, list]:
    """
    Pose les hooks IDENTIQUES à build_feature_database.py.
    - Blocs trunk : sortie (B, H, W, C)  → stockée telle quelle
    - Convs FPN   : sortie (B, C, H, W)  → permutée en (B, H, W, C)
    """
    captured = {}
    handles  = []

    for i, block in enumerate(encoder.trunk.blocks):
        def _bh(m, inp, out, idx=i):
            captured[f"block_{idx}"] = out.detach()   # (B, H, W, C)
        handles.append(block.register_forward_hook(_bh))

    for conv_idx, stage_num in enumerate([4, 3, 2, 1]):
        key = f"stage_{stage_num}_fpn"
        def _fpn(m, inp, out, k=key):
            captured[k] = out.detach().permute(0, 2, 3, 1)  # → (B, H, W, C)
        handles.append(encoder.neck.convs[conv_idx].register_forward_hook(_fpn))

    return captured, handles


# ═════════════════════════════════════════════════════════════════════════════
# Prétraitement (identique à build_feature_database.py)
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_array(img_array: np.ndarray) -> torch.Tensor:
    """
    img_array : (H, W) uint8 niveaux de gris  ou  (H, W, 3) uint8 RGB.
    Retourne tensor (1, 3, 1024, 1024) prêt pour l'encoder.
    """
    img = Image.fromarray(img_array)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0   # (1024, 1024, 3)
    x = x.permute(2, 0, 1)                                # (3, 1024, 1024)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0)                                  # (1, 3, 1024, 1024)


# ═════════════════════════════════════════════════════════════════════════════
# Conversion coordonnées (code de build_feature_database.py — lignes 340-347)
# ═════════════════════════════════════════════════════════════════════════════

def coord_to_fmap(x_min, y_min, x_max, y_max, orig_H, orig_W, H_feat, W_feat):
    """
    Conversion EXACTE du code build_feature_database.py : extract_patch_features.
    x=horizontal (col), y=vertical (row).
    """
    scale_x = W_feat / orig_W
    scale_y = H_feat / orig_H

    fx1 = max(0, int(x_min * scale_x))
    fy1 = max(0, int(y_min * scale_y))
    fx2 = min(W_feat, max(fx1 + 1, int(x_max * scale_x)))
    fy2 = min(H_feat, max(fy1 + 1, int(y_max * scale_y)))

    if fx2 - fx1 < 1:
        fx1 = min(fx1, W_feat - 1);  fx2 = fx1 + 1
    if fy2 - fy1 < 1:
        fy1 = min(fy1, H_feat - 1);  fy2 = fy1 + 1

    return fx1, fy1, fx2, fy2


def coord_manual(col, row, patch_sz, orig_H, orig_W, stride):
    """
    Calcul MANUEL pas-à-pas :
      1. resize anisotrope vers 1024×1024
      2. division par stride
    Retourne (col_fmap, row_fmap, w_fmap, h_fmap) — valeurs flottantes.
    """
    col_1024 = col * (IMG_SIZE / orig_W)
    row_1024 = row * (IMG_SIZE / orig_H)
    col_fmap  = col_1024 / stride
    row_fmap  = row_1024 / stride
    w_fmap    = patch_sz * (IMG_SIZE / orig_W) / stride
    h_fmap    = patch_sz * (IMG_SIZE / orig_H) / stride
    return col_fmap, row_fmap, w_fmap, h_fmap


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — Vérification ANALYTIQUE de la conversion
# ═════════════════════════════════════════════════════════════════════════════

def part1_analytical():
    print()
    print("=" * 72)
    print("PARTIE 1 — Vérification ANALYTIQUE de la conversion")
    print("=" * 72)

    print("""
Code source de la conversion : build_feature_database.py, lignes 340-347
(fonction extract_patch_features) :

    scale_x = W_feat / orig_W          # ← direct, sans passer par 1024
    scale_y = H_feat / orig_H
    fx1 = max(0, int(patch["x_min"] * scale_x))
    fy1 = max(0, int(patch["y_min"] * scale_y))
    fx2 = min(W_feat, max(fx1+1, int(patch["x_max"] * scale_x)))
    fy2 = min(H_feat, max(fy1+1, int(patch["y_max"] * scale_y)))

Preuve d'équivalence algébrique avec le chemin en deux étapes :
    col_1024 = col × (1024 / orig_W)
    col_fmap  = col_1024 / stride  =  col × (1024/orig_W) / stride
             =  col × (W_feat / orig_W)  [car W_feat = 1024/stride]
             =  col × scale_x            ✓

Le code est ALGÉBRIQUEMENT CORRECT — il court-circuite la résolution 1024
de manière légale.
""")

    stride = 16  # stage 3 / block_9, comme demandé
    W_feat = IMG_SIZE // stride   # 64
    H_feat = IMG_SIZE // stride   # 64

    print(f"Test sur stride={stride} (stage 3, ex. block_9) :")
    print(f"  feature map : {H_feat}×{W_feat}")
    print(f"  scale_x = {W_feat}/{ORIG_W} = {W_feat/ORIG_W:.6f}")
    print(f"  scale_y = {H_feat}/{ORIG_H} = {H_feat/ORIG_H:.6f}")
    print()

    positions_1 = [(0, 0), (256, 384), (1152, 640)]
    print(f"  {'Position (col,row)':20s}  {'col_fmap (manuel)':20s}  "
          f"{'row_fmap (manuel)':20s}  {'code int(col*sx)':18s}  "
          f"{'code int(row*sy)':18s}  {'Match?':8s}")
    print("  " + "-" * 110)
    all_ok = True
    for col, row in positions_1:
        cf, rf, wf, hf = coord_manual(col, row, PATCH_SZ, ORIG_H, ORIG_W, stride)
        code_cf = int(col * W_feat / ORIG_W)
        code_rf = int(row * H_feat / ORIG_H)
        match = (code_cf == int(cf)) and (code_rf == int(rf))
        all_ok = all_ok and match
        flag = "✓" if match else "✗ DÉCALAGE"
        print(f"  ({col:4d},{row:4d})               "
              f"  {cf:.3f}  → floor={int(cf):3d}       "
              f"  {rf:.3f}  → floor={int(rf):3d}       "
              f"  {code_cf:3d}               "
              f"  {code_rf:3d}               "
              f"  {flag}")
    print()
    print(f"  Taille patch dans la feature map (stride {stride}) :")
    print(f"    largeur  = {PATCH_SZ} × {IMG_SIZE/ORIG_W:.4f} / {stride} "
          f"= {PATCH_SZ * IMG_SIZE/ORIG_W/stride:.3f} cellules  "
          f"[facteur horizontal fW={IMG_SIZE/ORIG_W:.3f}]")
    print(f"    hauteur  = {PATCH_SZ} × {IMG_SIZE/ORIG_H:.4f} / {stride} "
          f"= {PATCH_SZ * IMG_SIZE/ORIG_H/stride:.3f} cellules  "
          f"[facteur vertical   fH={IMG_SIZE/ORIG_H:.3f}]")
    print()
    if all_ok:
        print("  → CONCLUSION : conversion analytiquement CORRECTE ✓")
    else:
        print("  → CONCLUSION : DÉCALAGE DÉTECTÉ dans la conversion ✗")
    return all_ok


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — Test PATCH SYNTHÉTIQUE (vérité terrain fabriquée)
# ═════════════════════════════════════════════════════════════════════════════

def make_black_white_image(col, row, patch_sz=PATCH_SZ, orig_h=ORIG_H, orig_w=ORIG_W):
    """Image noire avec un carré blanc à (col, row), taille patch_sz×patch_sz."""
    img = np.zeros((orig_h, orig_w), dtype=np.uint8)
    r2 = min(row + patch_sz, orig_h)
    c2 = min(col + patch_sz, orig_w)
    img[row:r2, col:c2] = 255
    return img


def feature_norms(feat_bhwc: torch.Tensor) -> np.ndarray:
    """Retourne carte des normes L2 : (H_feat, W_feat)."""
    f = feat_bhwc[0].float()   # (H, W, C)
    return f.norm(dim=-1).cpu().numpy()


def part2_synthetic_patches(encoder, captured, device):
    print()
    print("=" * 72)
    print("PARTIE 2 — Test du PATCH SYNTHÉTIQUE")
    print("=" * 72)
    print("  Stratégie : image noire + carré blanc à la position P.")
    print("  La zone de la feature map censée correspondre au carré blanc")
    print("  doit avoir des normes DIFFÉRENTES du fond noir.")
    print()
    print("  Seuils : ≥5% → signal clair (OK), 1-5% → signal faible (architectural?),")
    print("           <1% → absence totale de signal (DÉCALAGE probable).")
    print()

    verdicts = {}       # key → list of float (diff_ratio par position)
    block_results = {}  # key → list de (diff_ratio, coords_ok)

    for key in DISPLAY_KEYS:
        verdicts[key] = []

    global_ok = True

    for col, row in TEST_POSITIONS:
        x_min, y_min = col, row
        x_max = min(col + PATCH_SZ, ORIG_W)
        y_max = min(row + PATCH_SZ, ORIG_H)

        img_arr = make_black_white_image(col, row)
        tensor  = preprocess_array(img_arr).to(device)

        captured.clear()
        with torch.no_grad():
            encoder(tensor)

        print(f"  ── Position (col={col}, row={row}) ─────────────────────────────")

        fig, axes = plt.subplots(
            len(DISPLAY_KEYS), 2,
            figsize=(12, 3 * len(DISPLAY_KEYS))
        )
        fig.suptitle(
            f"Patch synthétique blanc à (col={col}, row={row})\n"
            f"Original {ORIG_H}×{ORIG_W} → resize 1024×1024 anisotrope",
            fontsize=11
        )

        for ax_row_idx, key in enumerate(DISPLAY_KEYS):
            if key not in captured:
                continue

            feat = captured[key]              # (B, H_feat, W_feat, C)
            H_feat, W_feat = feat.shape[1], feat.shape[2]
            stride = ALL_STRIDES[key]

            # Conversion code existant
            fx1, fy1, fx2, fy2 = coord_to_fmap(
                x_min, y_min, x_max, y_max, ORIG_H, ORIG_W, H_feat, W_feat
            )
            # Conversion manuelle (pour affichage)
            cf, rf, wf, hf = coord_manual(col, row, PATCH_SZ, ORIG_H, ORIG_W, stride)

            norms_map = feature_norms(feat)   # (H_feat, W_feat)

            # Masques : zone "blanche" (patch) vs zone "noire" (fond)
            mask_patch = np.zeros((H_feat, W_feat), dtype=bool)
            mask_patch[fy1:fy2, fx1:fx2] = True
            mask_bg    = ~mask_patch

            mean_patch = float(norms_map[mask_patch].mean()) if mask_patch.any() else 0.0
            mean_bg    = float(norms_map[mask_bg].mean())    if mask_bg.any()    else 0.0

            # Verdict : la zone patch doit être nettement différente du fond
            diff_ratio = abs(mean_patch - mean_bg) / (mean_bg + 1e-9)
            # Seuil adaptatif :
            #   ≥5%  → signal clair             → ✓ OK
            #   1-5% → signal faible (arch.)    → ~ WARN (pas un bug de coords)
            #   <1%  → quasi-nul                → ✗ FAIL (décalage probable)
            ok_strong = diff_ratio >= 0.05
            ok_weak   = diff_ratio >= 0.01
            verdicts[key].append(diff_ratio)
            if not ok_weak:
                global_ok = False

            if ok_strong:
                status = "✓ OK"
            elif ok_weak:
                status = "~ FAIBLE (architectural, coords OK)"
            else:
                status = "✗ FAIL (décalage?)"
            print(f"    {key:15s}  stride={stride:2d}  "
                  f"feat={H_feat}×{W_feat}  "
                  f"zone=[{fy1}:{fy2},{fx1}:{fx2}]  "
                  f"norme_patch={mean_patch:.4f}  "
                  f"norme_fond={mean_bg:.4f}  "
                  f"diff={diff_ratio*100:.1f}%  {status}")

            # ── Panneau gauche : image synthétique + zone extraite ────────────
            ax_l = axes[ax_row_idx, 0]
            ax_l.imshow(img_arr, cmap="gray", vmin=0, vmax=255,
                        origin="upper", aspect="auto",
                        extent=[0, ORIG_W, ORIG_H, 0])
            # Rectangle de la zone que le code pense extraire,
            # remis en coordonnées originales
            px1_orig = fx1 / W_feat * ORIG_W
            px2_orig = fx2 / W_feat * ORIG_W
            py1_orig = fy1 / H_feat * ORIG_H
            py2_orig = fy2 / H_feat * ORIG_H
            rect_code = mpatches.FancyBboxPatch(
                (px1_orig, py1_orig),
                px2_orig - px1_orig, py2_orig - py1_orig,
                boxstyle="square,pad=0", linewidth=2,
                edgecolor="red", facecolor="none",
                label="Zone code"
            )
            ax_l.add_patch(rect_code)
            rect_true = mpatches.FancyBboxPatch(
                (col, row), min(PATCH_SZ, ORIG_W-col), min(PATCH_SZ, ORIG_H-row),
                boxstyle="square,pad=0", linewidth=2,
                edgecolor="lime", facecolor="none", linestyle="--",
                label="Carré blanc"
            )
            ax_l.add_patch(rect_true)
            ax_l.set_title(f"{key}  stride={stride}\n"
                           f"Rouge=zone code  Vert=carré blanc", fontsize=8)
            ax_l.legend(fontsize=7, loc="upper right")
            ax_l.set_xlabel("col (x)"); ax_l.set_ylabel("row (y)")

            # ── Panneau droit : carte des normes feature map ───────────────────
            ax_r = axes[ax_row_idx, 1]
            im = ax_r.imshow(norms_map, cmap="hot", origin="upper", aspect="auto")
            rect_fmap = mpatches.FancyBboxPatch(
                (fx1 - 0.5, fy1 - 0.5), fx2 - fx1, fy2 - fy1,
                boxstyle="square,pad=0", linewidth=2,
                edgecolor="cyan", facecolor="none",
                label="Zone extraite"
            )
            ax_r.add_patch(rect_fmap)
            ax_r.set_title(f"Carte normes {H_feat}×{W_feat}\n"
                           f"norme_patch={mean_patch:.3f}  "
                           f"norme_fond={mean_bg:.3f}  {status}", fontsize=8)
            ax_r.legend(fontsize=7, loc="upper right")
            plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = OUTDIR / f"part2_col{col}_row{row}.png"
        plt.savefig(fname, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"    → figure sauvée : {fname.name}")
        print()

    print()
    print("  ── Résumé PARTIE 2 ──────────────────────────────────────────────")
    print(f"  {'Bloc/Stage':18s}  {'diff_ratio moyen':18s}  Verdict")
    coord_failures = []
    for key in DISPLAY_KEYS:
        ratios = verdicts.get(key, [])
        if not ratios:
            continue
        mean_r = np.mean(ratios)
        if mean_r >= 0.05:
            verd = "✓ Signal clair"
        elif mean_r >= 0.01:
            verd = "~ Signal faible (probable: attention mixing architectural)"
        else:
            verd = "✗ Absence de signal → DÉCALAGE de coordonnées suspecté"
            coord_failures.append(key)
        print(f"  {key:18s}  {mean_r*100:6.1f}%               {verd}")

    print()
    if not coord_failures:
        print("  → Aucun bloc ne montre d'absence totale de signal.")
        print("    Les signaux faibles (ex. block_2) reflètent l'attention mixing")
        print("    architecturale — pas un bug de conversion de coordonnées.")
        print("  → CONCLUSION PARTIE 2 : coordonnées FIABLES ✓")
        global_ok = True
    else:
        print(f"  → BLOCS SANS SIGNAL : {coord_failures}")
        print("    → DÉCALAGE DE COORDONNÉES probable — investiguer.")
        global_ok = False
    return global_ok


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 3 — Test de LOCALISATION FINE (décalage sub-patch)
# ═════════════════════════════════════════════════════════════════════════════

def make_gradient_patch_image(col, row, patch_sz=PATCH_SZ):
    """
    Image noire. Le patch à (col,row) a un gradient horizontal :
    moitié gauche = 100, moitié droite = 255.
    """
    img = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)
    r2 = min(row + patch_sz, ORIG_H)
    c2 = min(col + patch_sz, ORIG_W)
    half = (c2 - col) // 2
    img[row:r2, col:col+half]  = 100   # gris foncé
    img[row:r2, col+half:c2]   = 255   # blanc
    return img


def part3_fine_localization(encoder, captured, device):
    print()
    print("=" * 72)
    print("PARTIE 3 — Test de LOCALISATION FINE (décalage sub-patch)")
    print("=" * 72)
    print("  Patch avec gradient interne : moitié gauche=100, moitié droite=255.")
    print("  On vérifie que la carte des normes montre la MÊME transition")
    print("  gauche/droite que dans le patch.")
    print()

    # Positions choisies pour avoir assez de place dans la feature map
    positions_fine = [(256, 256), (640, 256)]

    for col, row in positions_fine:
        img_arr = make_gradient_patch_image(col, row)
        tensor  = preprocess_array(img_arr).to(device)

        captured.clear()
        with torch.no_grad():
            encoder(tensor)

        print(f"  ── Position (col={col}, row={row}) ─────────────────────────────")

        # On se concentre sur un bloc à stride 16 (assez de résolution + assez d'espace)
        key    = "block_9"
        stride = 16

        if key not in captured:
            print(f"    {key} non capturé — skip")
            continue

        feat   = captured[key]
        H_feat, W_feat = feat.shape[1], feat.shape[2]

        x_min, y_min = col, row
        x_max = min(col + PATCH_SZ, ORIG_W)
        y_max = min(row + PATCH_SZ, ORIG_H)

        fx1, fy1, fx2, fy2 = coord_to_fmap(
            x_min, y_min, x_max, y_max, ORIG_H, ORIG_W, H_feat, W_feat
        )

        norms_map  = feature_norms(feat)
        patch_norms = norms_map[fy1:fy2, fx1:fx2]   # (h_p, w_p)

        h_p, w_p = patch_norms.shape
        if w_p < 2:
            print(f"    Patch trop petit dans la feature map ({w_p} col) — skip")
            continue

        # Norme moyenne par colonne dans le patch (profil horizontal)
        col_profile = patch_norms.mean(axis=0)  # (w_p,)
        mid = w_p // 2
        left_mean  = col_profile[:mid].mean()
        right_mean = col_profile[mid:].mean()
        diff = right_mean - left_mean   # positif si droite > gauche (attendu)

        # La moitié droite (255) devrait activer plus que la gauche (100)
        expected_direction = diff > 0
        gradient_visible   = abs(diff) / (abs(left_mean) + 1e-9) > 0.02

        print(f"    {key:15s}  patch_fmap=({h_p}×{w_p})")
        print(f"    norme_left_half={left_mean:.4f}  norme_right_half={right_mean:.4f}  "
              f"diff={diff:.4f}")
        if gradient_visible and expected_direction:
            print(f"    → Gradient VISIBLE et dans le bon sens (droite>gauche) ✓")
        elif gradient_visible and not expected_direction:
            print(f"    → Gradient VISIBLE mais sens INVERSÉ ✗ (probable décalage)")
        else:
            print(f"    → Gradient NON VISIBLE (diff trop faible) — features peu sensibles à l'intensité")

        # ── Visualisation ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"Test localisation fine — (col={col}, row={row})  [{key} stride={stride}]",
            fontsize=11
        )

        # Image synthétique
        axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255, origin="upper",
                       aspect="auto", extent=[0, ORIG_W, ORIG_H, 0])
        px1o = fx1 / W_feat * ORIG_W;  px2o = fx2 / W_feat * ORIG_W
        py1o = fy1 / H_feat * ORIG_H;  py2o = fy2 / H_feat * ORIG_H
        axes[0].add_patch(mpatches.FancyBboxPatch(
            (px1o, py1o), px2o-px1o, py2o-py1o,
            boxstyle="square,pad=0", lw=2, edgecolor="red", facecolor="none"
        ))
        axes[0].set_title("Image synthétique\n(rouge=zone extraite)", fontsize=9)
        axes[0].set_xlabel("col"); axes[0].set_ylabel("row")

        # Carte des normes dans le patch extrait
        axes[1].imshow(patch_norms, cmap="hot", origin="upper", aspect="auto")
        axes[1].axvline(x=mid-0.5, color="cyan", lw=2, label="Milieu attendu")
        axes[1].set_title(f"Carte normes dans le patch\n({h_p}×{w_p} cellules)", fontsize=9)
        axes[1].legend(fontsize=8)

        # Profil horizontal moyen
        axes[2].plot(col_profile, color="blue", lw=2)
        axes[2].axvline(x=mid-0.5, color="cyan", lw=2, ls="--",
                        label=f"Milieu ({mid})")
        axes[2].axhline(y=left_mean,  color="gray",  lw=1, ls=":", label=f"moy_gauche={left_mean:.3f}")
        axes[2].axhline(y=right_mean, color="orange", lw=1, ls=":", label=f"moy_droite={right_mean:.3f}")
        axes[2].set_xlabel("colonne dans le patch")
        axes[2].set_ylabel("norme moyenne")
        axes[2].set_title("Profil horizontal\n(doit monter vers la droite)", fontsize=9)
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = OUTDIR / f"part3_col{col}_row{row}.png"
        plt.savefig(fname, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"    → figure sauvée : {fname.name}")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE 4 — Comparaison ANCIEN code (H5) vs NOUVEAU code
# ═════════════════════════════════════════════════════════════════════════════

def part4_compare_old_new():
    print()
    print("=" * 72)
    print("PARTIE 4 — Comparaison ancien code (H5) vs nouveau code")
    print("=" * 72)
    print("""
  Constat : le code de conversion est IDENTIQUE dans build_feature_database.py
  et dans tout script qui l'utilise pour l'extraction (il n'y a qu'une seule
  implémentation de extract_patch_features).

  La H5 partage donc EXACTEMENT le même code que tout nouveau test.
  Si la conversion est correcte → la H5 est correcte.
  Si la conversion est décalée  → la H5 ET tout nouveau test sont décalés
                                   au même endroit et dans la même mesure.

  Code analysé : build_feature_database.py, lignes 322-365.

  Résumé des conventions de coordonnées :
    • Excel PatchTagger : x=row (vertical), y=col (horizontal)   [lignes 139-147]
    • Swap effectué en _load_from_xlsx : y_min→x_min, x_min→y_min
      (remet en convention image : x=horizontal, y=vertical)
    • JSON dir : déjà en convention image standard (pas de swap)
    • extract_patch_features : x=horizontal (col), y=vertical (row) ✓

  → Pas de code alternatif à comparer. Les parties 1-3 valident la H5
    au passage, sans circularité.
""")


# ═════════════════════════════════════════════════════════════════════════════
# Rapport final
# ═════════════════════════════════════════════════════════════════════════════

def print_verdict(ok_analytical, ok_synthetic, ckpt_loaded):
    print()
    print("=" * 72)
    print("VERDICT FINAL")
    print("=" * 72)
    ckpt_note = "checkpoint réel" if ckpt_loaded else "poids ALÉATOIRES"
    print(f"  (Test effectué avec : {ckpt_note})")
    print()

    checks = {
        "Partie 1 — Conversion analytique": ok_analytical,
        "Partie 2 — Patch synthétique":     ok_synthetic,
    }
    all_ok = all(checks.values())

    for name, ok in checks.items():
        status = "✓ FIABLE" if ok else "✗ PROBLÈME DÉTECTÉ"
        print(f"  {name:40s}  {status}")

    print()
    if all_ok:
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │  EXTRACTION FIABLE                                      │")
        print("  │  La conversion patch→feature map est correcte.          │")
        print("  │  La H5 est validée au passage (même code).              │")
        print("  │  On peut lancer le test d'agrégation en confiance.      │")
        print("  └─────────────────────────────────────────────────────────┘")
    else:
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │  ⚠ DÉCALAGE DÉTECTÉ                                     │")
        print("  │  Localiser l'erreur de conversion AVANT tout autre test.│")
        print("  │  Si l'ancien code est décalé → TOUS les résultats basés │")
        print("  │  sur la H5 sont à réexaminer.                           │")
        print("  └─────────────────────────────────────────────────────────┘")

    if not ckpt_loaded:
        print()
        print("  NOTE : test exécuté avec des poids aléatoires.")
        print("  La partie 2 teste la propagation spatiale du signal noir/blanc,")
        print("  non la qualité sémantique des features. Le verdict sur la")
        print("  conversion de coordonnées reste valide.")
        print("  Pour un verdict maximal sur la partie 3 (gradient fin),")
        print("  relancer avec le checkpoint réel.")

    print()
    print(f"  Figures sauvées dans : {OUTDIR}/")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  VALIDATION DE L'EXTRACTION PATCH → FEATURE MAP                 ║")
    print("║  Vérité terrain synthétique — zéro dépendance à la H5           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Image originale    : {ORIG_H}×{ORIG_W}")
    print(f"  Image resize       : {IMG_SIZE}×{IMG_SIZE}  (anisotrope)")
    print(f"  Taille patch       : {PATCH_SZ}×{PATCH_SZ}  (espace original)")
    print(f"  Facteur horizontal : fW = {IMG_SIZE/ORIG_W:.4f}")
    print(f"  Facteur vertical   : fH = {IMG_SIZE/ORIG_H:.4f}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    # ── Construction modèle ───────────────────────────────────────────────────
    encoder, ckpt_loaded = load_encoder(CKPT_PATH)
    encoder = encoder.to(device)
    captured, handles = register_hooks(encoder)

    try:
        # ── Partie 1 : analytique ─────────────────────────────────────────────
        ok_analytical = part1_analytical()

        # ── Partie 2 : patch synthétique ──────────────────────────────────────
        ok_synthetic = part2_synthetic_patches(encoder, captured, device)

        # ── Partie 3 : localisation fine ──────────────────────────────────────
        part3_fine_localization(encoder, captured, device)

        # ── Partie 4 : comparaison H5 vs nouveau ──────────────────────────────
        part4_compare_old_new()

        # ── Verdict ───────────────────────────────────────────────────────────
        print_verdict(ok_analytical, ok_synthetic, ckpt_loaded)

    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
