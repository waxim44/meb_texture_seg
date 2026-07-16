"""
transforms.py
══════════════════════════════════════════════════════════════════════════════
PHASE 1 — Augmentations d'image entière + correspondance des annotations.

Convention : image indexée img[row, col], row ∈ [0, H), col ∈ [0, W),
row = axe vertical (haut→bas), col = axe horizontal (gauche→droite).
Patch top-left (row0, col0), taille S=128 (carré, bornes [row0,row0+S),
[col0,col0+S)).

TRANSFORMS = {"identity", "flipH", "flipV", "rot90", "rot180", "rot270"}
  - flipH  : miroir gauche-droite (flip des colonnes)
  - flipV  : miroir haut-bas (flip des lignes)
  - rot90  : rotation 90° (sens np.rot90(k=1) — antihoraire en convention
             matricielle standard)
  - rot180 : rotation 180°
  - rot270 : rotation 270° (= np.rot90(k=3))
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np

TRANSFORMS = ["identity", "flipH", "flipV", "rot90", "rot180", "rot270"]


def transform_image(img: np.ndarray, t: str) -> np.ndarray:
    """Applique la transformation t à une image entière (H, W) [, C]."""
    if t == "identity":
        return img
    elif t == "flipH":
        return img[:, ::-1, ...]
    elif t == "flipV":
        return img[::-1, :, ...]
    elif t == "rot90":
        return np.rot90(img, k=1, axes=(0, 1))
    elif t == "rot180":
        return np.rot90(img, k=2, axes=(0, 1))
    elif t == "rot270":
        return np.rot90(img, k=3, axes=(0, 1))
    else:
        raise ValueError(f"Transformation inconnue : {t}")


def transform_coords(row0: int, col0: int, t: str, H_img: int, W_img: int, S: int = 128):
    """
    Coin haut-gauche (row0, col0) d'un patch S×S dans une image (H_img, W_img)
    → coin haut-gauche (row0', col0') du même patch dans l'image transformée
    par t. Retourne aussi (H_img', W_img'), la taille de l'image transformée.

    Formules fermées (dérivées de la définition pixel-à-pixel de chaque t,
    cf. docstring module) :
      identity : row0'=row0                col0'=col0
      flipH    : row0'=row0                col0'=W_img-col0-S
      flipV    : row0'=H_img-row0-S        col0'=col0
      rot90    : row0'=W_img-col0-S        col0'=row0
      rot180   : row0'=H_img-row0-S        col0'=W_img-col0-S
      rot270   : row0'=col0                col0'=H_img-row0-S
    """
    if t == "identity":
        row0p, col0p = row0, col0
        Hp, Wp = H_img, W_img
    elif t == "flipH":
        row0p, col0p = row0, W_img - col0 - S
        Hp, Wp = H_img, W_img
    elif t == "flipV":
        row0p, col0p = H_img - row0 - S, col0
        Hp, Wp = H_img, W_img
    elif t == "rot90":
        row0p, col0p = W_img - col0 - S, row0
        Hp, Wp = W_img, H_img
    elif t == "rot180":
        row0p, col0p = H_img - row0 - S, W_img - col0 - S
        Hp, Wp = H_img, W_img
    elif t == "rot270":
        row0p, col0p = col0, H_img - row0 - S
        Hp, Wp = W_img, H_img
    else:
        raise ValueError(f"Transformation inconnue : {t}")

    if not (0 <= row0p <= Hp - S and 0 <= col0p <= Wp - S):
        raise AssertionError(
            f"Patch hors bornes après transform '{t}': row0'={row0p}, col0'={col0p}, "
            f"S={S}, H'={Hp}, W'={Wp}"
        )
    return row0p, col0p, Hp, Wp
