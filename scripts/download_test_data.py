"""
Prépare les données de test depuis Kaust256 (local) et STMD (clone git).

Usage:
    python scripts/download_test_data.py
"""

import os
import sys
import shutil
import random
import subprocess

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Chemins ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAUST_SRC  = os.path.join(ROOT, "TextureSAM", "Kaust256")
STMD_REPO  = os.path.join(ROOT, "data", "raw", "stmd_repo")
DST_KAUST  = os.path.join(ROOT, "data", "raw", "kaust")
DST_STMD   = os.path.join(ROOT, "data", "raw", "stmd")
PREVIEW_DIR = os.path.join(ROOT, "outputs", "data_preview")
N = 20

STMD_GIT = "https://github.com/mubashar1030/Segmentation_dataset"


# ── Helpers visuels ────────────────────────────────────────────────────────────

def hr(char="─", width=60):
    print(char * width)

def section(title):
    print()
    hr("═")
    print(f"  {title}")
    hr("═")


# ── Copie avec correspondance stem ────────────────────────────────────────────

def copy_matched_sample(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir,
                        img_exts=(".jpg", ".jpeg", ".png"),
                        lbl_exts=(".png", ".jpg"),
                        n=N, seed=42):
    """
    Copie n paires (image, label) choisies aléatoirement.
    La correspondance se fait par stem (nom sans extension).
    Retourne la liste des stems copiés.
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    imgs = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(src_img_dir)
        if os.path.splitext(f)[1].lower() in img_exts
    }
    lbls = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(src_lbl_dir)
        if os.path.splitext(f)[1].lower() in lbl_exts
    }

    common = sorted(set(imgs) & set(lbls))
    if not common:
        raise FileNotFoundError(
            f"Aucune paire image/label trouvée entre\n  {src_img_dir}\n  {src_lbl_dir}"
        )

    random.seed(seed)
    chosen = random.sample(common, min(n, len(common)))

    for stem in chosen:
        shutil.copy2(
            os.path.join(src_img_dir, imgs[stem]),
            os.path.join(dst_img_dir, imgs[stem]),
        )
        shutil.copy2(
            os.path.join(src_lbl_dir, lbls[stem]),
            os.path.join(dst_lbl_dir, lbls[stem]),
        )

    return chosen, imgs, lbls


# ── Vérification d'un dataset ─────────────────────────────────────────────────

def verify_dataset(name, img_dir, lbl_dir):
    section(f"Vérification — {name}")

    imgs = sorted(os.listdir(img_dir)) if os.path.isdir(img_dir) else []
    lbls = sorted(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else []

    has_gt = len(lbls) > 0
    n_imgs = len(imgs)

    sample_img = None
    if imgs:
        sample_path = os.path.join(img_dir, imgs[0])
        sample_img = Image.open(sample_path)
        W, H = sample_img.size
        fmt = os.path.splitext(imgs[0])[1].lower().replace(".", "")
    else:
        H, W, fmt = "?", "?", "?"

    print(f"  → nombre d'images  : {n_imgs}")
    print(f"  → taille           : {H}×{W}")
    print(f"  → format           : {fmt}")
    print(f"  → GT disponible    : {'oui' if has_gt else 'non'}")

    return imgs, lbls


# ── Preview image + GT ────────────────────────────────────────────────────────

def save_preview(name, img_dir, lbl_dir, out_path):
    imgs = sorted(os.listdir(img_dir))
    lbls = sorted(os.listdir(lbl_dir))
    if not imgs:
        print(f"  [WARN] Aucune image dans {img_dir} — preview ignorée")
        return

    img_file = imgs[0]
    stem = os.path.splitext(img_file)[0]

    # Trouver le label correspondant
    lbl_file = None
    for lbl in lbls:
        if os.path.splitext(lbl)[0] == stem:
            lbl_file = lbl
            break

    raw_img = Image.open(os.path.join(img_dir, img_file))
    is_gray = raw_img.mode == "L"
    img = np.array(raw_img.convert("RGB"))

    fig, axes = plt.subplots(1, 2 if lbl_file else 1, figsize=(10, 5))
    fig.suptitle(f"{name} — {img_file}", fontsize=13, fontweight="bold")

    if lbl_file is None:
        axes = [axes]

    axes[0].imshow(img)
    axes[0].set_title("Image", fontsize=11)
    axes[0].axis("off")

    if lbl_file:
        lbl_arr = np.array(Image.open(os.path.join(lbl_dir, lbl_file)))
        # Normaliser pour affichage si valeurs 0/255 ou 0/1
        if lbl_arr.max() <= 1:
            lbl_arr = (lbl_arr * 255).astype(np.uint8)
        axes[1].imshow(lbl_arr, cmap="tab20", interpolation="nearest")
        axes[1].set_title(f"GT — {lbl_file}", fontsize=11)
        axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → preview sauvegardée : {os.path.relpath(out_path, ROOT)}")


# ── Affiche la structure finale ───────────────────────────────────────────────

def print_final_structure():
    section("Structure finale")
    raw = os.path.join(ROOT, "data", "raw")
    print("  data/raw/")
    for ds in ["kaust", "stmd"]:
        ds_path = os.path.join(raw, ds)
        print(f"  ├── {ds}/")
        for sub in ["images", "labels"]:
            sub_path = os.path.join(ds_path, sub)
            if os.path.isdir(sub_path):
                n = len(os.listdir(sub_path))
                print(f"  │   ├── {sub}/    ({n} fichiers)")
            else:
                print(f"  │   ├── {sub}/    (absent)")


# ── ÉTAPE 1 — KAUST256 ───────────────────────────────────────────────────────

def step_kaust():
    section("ÉTAPE 1 — KAUST256 (copie locale)")

    src_img = os.path.join(KAUST_SRC, "images")
    src_lbl = os.path.join(KAUST_SRC, "labeles")   # typo intentionnelle du repo
    dst_img = os.path.join(DST_KAUST, "images")
    dst_lbl = os.path.join(DST_KAUST, "labels")

    chosen, imgs, lbls = copy_matched_sample(src_img, src_lbl, dst_img, dst_lbl)

    sample_path = os.path.join(dst_img, imgs[chosen[0]])
    sample_img  = Image.open(sample_path)
    W, H        = sample_img.size

    print(f"  → {len(chosen)} images copiées vers {os.path.relpath(dst_img, ROOT)}")
    print(f"  → taille des images : {H}×{W}")
    print(f"  → exemple           : {imgs[chosen[0]]}")


# ── ÉTAPE 2 — STMD ───────────────────────────────────────────────────────────

def step_stmd():
    section("ÉTAPE 2 — STMD (clone git)")

    dst_img = os.path.join(DST_STMD, "images")
    dst_lbl = os.path.join(DST_STMD, "labels")

    # Clone
    if os.path.isdir(STMD_REPO):
        print(f"  Repo déjà présent — skip clone : {os.path.relpath(STMD_REPO, ROOT)}")
    else:
        print(f"  Clone → {os.path.relpath(STMD_REPO, ROOT)}")
        os.makedirs(os.path.dirname(STMD_REPO), exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth=1", STMD_GIT, STMD_REPO],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [ERREUR] Clone échoué :\n  {result.stderr.strip()}")
            return False
        print("  Clone réussi.")

    # Déterminer les sous-dossiers disponibles
    candidates_img = ["images_test", "images", "Images"]
    candidates_lbl = ["labels_test", "labels", "Labels", "masks_test", "masks",
                       "groundtruth", "Groundtruth", "ground_truth"]

    src_img = None
    for c in candidates_img:
        p = os.path.join(STMD_REPO, c)
        if os.path.isdir(p) and os.listdir(p):
            src_img = p
            break

    src_lbl = None
    for c in candidates_lbl:
        p = os.path.join(STMD_REPO, c)
        if os.path.isdir(p) and os.listdir(p):
            src_lbl = p
            break

    if src_img is None:
        print(f"  [ERREUR] Dossier images introuvable dans {STMD_REPO}")
        print(f"  Contenu : {os.listdir(STMD_REPO)}")
        return False

    print(f"  Images source : {os.path.relpath(src_img, ROOT)}")

    if src_lbl is None:
        print("  [WARN] Dossier labels introuvable — copie images seules")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        imgs_all = sorted([
            f for f in os.listdir(src_img)
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
        ])
        random.seed(42)
        chosen = random.sample(imgs_all, min(N, len(imgs_all)))
        for f in chosen:
            shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
        print(f"  → {len(chosen)} images copiées (sans GT)")
        return True

    print(f"  Labels source  : {os.path.relpath(src_lbl, ROOT)}")
    chosen, imgs, lbls = copy_matched_sample(src_img, src_lbl, dst_img, dst_lbl)

    sample_path = os.path.join(dst_img, imgs[chosen[0]])
    sample_img  = Image.open(sample_path)
    W, H        = sample_img.size

    print(f"  → {len(chosen)} images copiées vers {os.path.relpath(dst_img, ROOT)}")
    print(f"  → taille des images : {H}×{W}")
    print(f"  → exemple           : {imgs[chosen[0]]}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    hr("═")
    print("  Préparation des données de test")
    hr("═")

    # Étape 1 — Kaust
    step_kaust()

    # Étape 2 — STMD
    stmd_ok = step_stmd()

    # Étape 3 — Vérification
    section("ÉTAPE 3 — Vérification")

    print("\n  KAUST256 :")
    verify_dataset(
        "KAUST256",
        os.path.join(DST_KAUST, "images"),
        os.path.join(DST_KAUST, "labels"),
    )

    if stmd_ok:
        print("\n  STMD :")
        verify_dataset(
            "STMD",
            os.path.join(DST_STMD, "images"),
            os.path.join(DST_STMD, "labels"),
        )

    # Étape 4 — Previews
    section("ÉTAPE 4 — Previews")

    save_preview(
        "KAUST256",
        os.path.join(DST_KAUST, "images"),
        os.path.join(DST_KAUST, "labels"),
        os.path.join(PREVIEW_DIR, "kaust_preview.png"),
    )

    if stmd_ok:
        save_preview(
            "STMD",
            os.path.join(DST_STMD, "images"),
            os.path.join(DST_STMD, "labels"),
            os.path.join(PREVIEW_DIR, "stmd_preview.png"),
        )

    # Structure finale
    print_final_structure()
    print()


if __name__ == "__main__":
    main()
