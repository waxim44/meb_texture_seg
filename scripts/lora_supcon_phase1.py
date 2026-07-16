"""
lora_supcon_phase1.py
══════════════════════════════════════════════════════════════════════════════
PHASE 1 — Validation du pipeline d'augmentation (image entière) + formules
de correspondance des coordonnées de patch.

Tests :
  1. PIXEL-PERFECT : pour ≥20 patchs aléatoires × chaque transformation t,
     crop_original transformé par t == crop extrait aux nouvelles coords
     dans l'image transformée (égalité numpy stricte). 100% requis.
  2. VISUEL : 3 exemples (image augmentée + rectangle + crop côte à côte).
  3. BORNES : aucun patch ne sort des bornes après transformation (assert,
     déjà fait dans transform_coords, mais on vérifie ici que ça ne lève pas).
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "lora_supcon"))

from transforms import transform_image, transform_coords, TRANSFORMS  # noqa: E402

IMG_DIR = _ROOT / "Image_Ouassim"
OUTDIR = _ROOT / "lora_supcon" / "phase_1"
OUTDIR.mkdir(parents=True, exist_ok=True)

S = 128
N_RANDOM_PATCHES = 30
SEED = 0


def load_grayscale(path):
    return np.array(Image.open(path))


def random_patches(H, W, n, rng):
    rows = list(range(0, H - S + 1, S))
    cols = list(range(0, W - S + 1, S))
    patches = []
    for _ in range(n):
        patches.append((rng.choice(rows), rng.choice(cols)))
    return patches


def pixel_perfect_test():
    rng = random.Random(SEED)
    files = sorted(glob.glob(str(IMG_DIR / "*.tif")))
    n_total = 0
    n_pass = 0
    failures = []

    for _ in range(N_RANDOM_PATCHES):
        f = rng.choice(files)
        img = load_grayscale(f)
        H, W = img.shape
        row0, col0 = rng.choice(random_patches(H, W, 1, rng))

        for t in TRANSFORMS:
            n_total += 1
            img_t = transform_image(img, t)
            row0p, col0p, Hp, Wp = transform_coords(row0, col0, t, H, W, S)

            crop_orig_transformed = transform_image(
                img[row0:row0 + S, col0:col0 + S], t
            )
            crop_from_new_coords = img_t[row0p:row0p + S, col0p:col0p + S]

            equal = np.array_equal(crop_orig_transformed, crop_from_new_coords)
            if equal:
                n_pass += 1
            else:
                failures.append((f, t, row0, col0, row0p, col0p))

    return n_total, n_pass, failures


def bounds_test():
    """Vérifie qu'aucun patch de la grille ne sort des bornes, pour toutes
    les images et toutes les transformations (couverture exhaustive)."""
    files = sorted(glob.glob(str(IMG_DIR / "*.tif")))
    n_checked = 0
    for f in files:
        img = load_grayscale(f)
        H, W = img.shape
        for row0 in range(0, H - S + 1, S):
            for col0 in range(0, W - S + 1, S):
                for t in TRANSFORMS:
                    transform_coords(row0, col0, t, H, W, S)  # raises if OOB
                    n_checked += 1
    return n_checked


def visual_examples():
    rng = random.Random(SEED + 1)
    files = sorted(glob.glob(str(IMG_DIR / "*.tif")))
    examples = []
    for i in range(3):
        f = rng.choice(files)
        img = load_grayscale(f)
        H, W = img.shape
        row0, col0 = rng.choice(random_patches(H, W, 1, rng))
        t = rng.choice([x for x in TRANSFORMS if x != "identity"])
        examples.append((f, img, row0, col0, t))

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, (f, img, row0, col0, t) in enumerate(examples):
        H, W = img.shape
        img_t = transform_image(img, t)
        row0p, col0p, Hp, Wp = transform_coords(row0, col0, t, H, W, S)

        crop_orig_transformed = transform_image(img[row0:row0 + S, col0:col0 + S], t)
        crop_from_new_coords = img_t[row0p:row0p + S, col0p:col0p + S]

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].add_patch(plt.Rectangle((col0, row0), S, S, edgecolor="red", facecolor="none", linewidth=2))
        axes[i, 0].set_title(f"Original\n{Path(f).name}\nrow0={row0} col0={col0}")

        axes[i, 1].imshow(img_t, cmap="gray")
        axes[i, 1].add_patch(plt.Rectangle((col0p, row0p), S, S, edgecolor="lime", facecolor="none", linewidth=2))
        axes[i, 1].set_title(f"t={t}\nrow0'={row0p} col0'={col0p}")

        diff = np.abs(crop_orig_transformed.astype(int) - crop_from_new_coords.astype(int))
        axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_title(f"|diff| (max={diff.max()})")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = OUTDIR / "visual_examples.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def main():
    lines = ["═" * 70, "PHASE 1 — Rapport de validation", "═" * 70]

    n_total, n_pass, failures = pixel_perfect_test()
    lines.append(f"TEST PIXEL-PERFECT : {n_pass}/{n_total} passent")
    if failures:
        lines.append("Échecs (max 10 affichés) :")
        for fail in failures[:10]:
            lines.append(f"  {fail}")
    ok_pixel = n_pass == n_total

    n_checked = bounds_test()
    lines.append(f"TEST BORNES : {n_checked} (image, patch, t) vérifiés sans dépassement")
    ok_bounds = True  # bounds_test raises on failure, so reaching here means OK

    viz_path = visual_examples()
    lines.append(f"TEST VISUEL : exemples sauvegardés dans {viz_path}")

    lines.append("")
    lines.append("── VALIDATION P1 ──")
    lines.append(f"[{'x' if ok_pixel else ' '}] TEST PIXEL-PERFECT (100% requis)")
    lines.append(f"[x] TEST VISUEL (3 exemples générés — inspection manuelle requise)")
    lines.append(f"[{'x' if ok_bounds else ' '}] Aucun patch hors bornes")
    go = ok_pixel and ok_bounds
    lines.append("")
    lines.append(f"VERDICT : {'GO' if go else 'NO-GO'}")

    report = "\n".join(lines)
    print(report)
    (OUTDIR / "report.txt").write_text(report)
    return go


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
