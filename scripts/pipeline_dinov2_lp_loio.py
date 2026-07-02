#!/usr/bin/env python3
"""
pipeline_dinov2_lp_loio.py
═══════════════════════════════════════════════════════════════════
DINOv2 vs TextureSAM — Séparabilité des textures MEB
Protocole LP LOIO identique à SAM pour comparaison directe.

Étapes :
  0 — Sanity check modèle + shapes
  1 — Validation extraction coordonnées (carré blanc synthétique)
  2 — Choix resize documenté (ISO par défaut)
  3 — Extraction features par patch → H5 dédié par modèle
  4 — LP LOIO (protocole identique SAM)
  5 — Sorties + comparaison directe SAM vs DINOv2
═══════════════════════════════════════════════════════════════════
"""

import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from tqdm import tqdm

# ─── CONSTANTES ────────────────────────────────────────────────────────────────
ROOT       = Path('/home/aidouni/meb_texture_seg')
IMG_DIR    = ROOT / 'Image_Ouassim'
H5_SAM     = ROOT / 'data/feature_database/database_meb_ouassim.h5'
H5_OUT_DIR = ROOT / 'data/feature_database'
OUT_DIR    = ROOT / 'output_ouassim/dinov2_comparison'
VAL_DIR    = ROOT / 'output_ouassim/dinov2_validate_coordinates'

ORIG_H, ORIG_W = 768, 1280
PATCH_SZ       = 128

# Resize ISO : préserve ratio 3:5 exactement, multiples de 14
# 756 = 54×14, 1260 = 90×14  →  ratio 756/1260 = 3/5 ✓
ISO_H,  ISO_W  = 756, 1260    # → 54×90 = 4860 tokens
ISO_NH, ISO_NW = 54, 90

# Resize carré : anisotrope comme SAM (518 = 37×14)
SQ_H, SQ_W = 518, 518
SQ_NH = SQ_NW = 37

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {
    1: 'Tot.homogène', 3: 'Faisceaux',  4: 'Filaments',
    5: 'Strat.rect',   6: 'Strat.sin',  7: 'Granuleux',  9: 'Trou',
}

# 0-indexed DINOv2 block indices → étiquettes lisibles (1-indexé)
LAYER_INDICES = [2, 5, 8, 11]   # blocks 3, 6, 9, 12
LAYER_LABELS  = {2: 'layer_03', 5: 'layer_06', 8: 'layer_09', 11: 'layer_12'}

MODELS = ['dinov2_vits14_reg', 'dinov2_vitb14', 'dinov2_vitb14_reg']

PCA_DIM = 50
LP_C    = 1.0
SEED    = 42

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Résultats SAM (meilleur bloc par texture, protocole LP LOIO propre)
SAM_RECALL = {
    1: (1.000, 0.000),   # Tot.homogène
    3: (0.790, 0.278),   # Faisceaux
    4: (0.837, 0.323),   # Filaments
    5: (0.502, 0.498),   # Strat.rect
    6: (0.524, 0.383),   # Strat.sin
    7: (0.860, 0.219),   # Granuleux
    9: (0.735, 0.388),   # Trou
}

TEST_POSITIONS = [(0, 0), (256, 384), (640, 256), (1152, 640)]


# ═══════════════════════════════════════════════════════════════════════════════
# Utilitaires — resize / preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def get_token_grid(mode: str):
    """Retourne (resize_H, resize_W, n_tokens_H, n_tokens_W)."""
    if mode == 'iso':
        return ISO_H, ISO_W, ISO_NH, ISO_NW
    return SQ_H, SQ_W, SQ_NH, SQ_NW


def preprocess(img_array: np.ndarray, mode: str) -> torch.Tensor:
    """
    img_array : (H, W) uint8 grayscale → tensor (1, 3, rH, rW) normalisé ImageNet.
    """
    rH, rW, _, _ = get_token_grid(mode)
    img = Image.fromarray(img_array)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((rW, rH), Image.BILINEAR)   # PIL : (W, H)
    x = torch.from_numpy(np.array(img)).float() / 255.0   # (rH, rW, 3)
    x = x.permute(2, 0, 1)                                # (3, rH, rW)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)                                  # (1, 3, rH, rW)


def coord_to_tokens(x_min, y_min, x_max, y_max, nH: int, nW: int):
    """
    Conversion (x_min, y_min, x_max, y_max) coords originales
    → indices (tx1, ty1, tx2, ty2) dans la grille de tokens.
    Même formule que SAM extract_patch_features.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 0 — Sanity check
# ═══════════════════════════════════════════════════════════════════════════════

def step0_sanity(model, model_name: str, mode: str, device: str):
    print(f"\n{'='*70}")
    print(f"ÉTAPE 0 — Sanity check : {model_name}  (mode={mode})")
    print(f"{'='*70}")

    rH, rW, nH, nW = get_token_grid(mode)
    N_tokens = nH * nW
    print(f"  Resize cible : {rH}×{rW}  →  grille tokens : {nH}×{nW} = {N_tokens} tokens")

    dummy = torch.randn(1, 3, rH, rW).to(device)
    with torch.no_grad():
        intermediates = model.get_intermediate_layers(
            dummy, n=LAYER_INDICES, reshape=False,
            return_class_token=False, norm=True
        )

    D = intermediates[0].shape[-1]
    print(f"  Dimension D  : {D}")
    print()
    print(f"  {'Couche (0-idx)':>18}  {'Étiquette':>12}  {'Shape':>22}  Vérif")
    print(f"  {'-'*65}")
    all_ok = True
    for i, (li, lbl) in enumerate(zip(LAYER_INDICES, LAYER_LABELS.values())):
        t = intermediates[i]
        ok_shape = t.shape == (1, N_tokens, D)
        all_ok = all_ok and ok_shape
        flag = "✓" if ok_shape else "✗ SHAPE INATTENDU"
        print(f"  block[{li:>2}] ({lbl})  {str(t.shape):>22}  {flag}")

    print()
    if all_ok:
        print("  → Sanity check PASSÉ ✓")
    else:
        print("  → Sanity check ÉCHOUÉ ✗ — vérifier la version du modèle")
    return all_ok, D


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Validation de l'extraction coordonnées (carré blanc)
# ═══════════════════════════════════════════════════════════════════════════════

def step1_validate_coords(model, mode: str, device: str) -> bool:
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    rH, rW, nH, nW = get_token_grid(mode)
    print(f"\n{'='*70}")
    print(f"ÉTAPE 1 — Validation coordonnées (carré blanc synthétique)")
    print(f"  Mode resize   : {mode}  ({rH}×{rW} → {nH}×{nW} tokens)")
    print(f"  Taille carré  : {PATCH_SZ}×{PATCH_SZ} px (espace original {ORIG_H}×{ORIG_W})")
    print(f"  Conversion    : scale_x = {nW}/{ORIG_W} = {nW/ORIG_W:.6f}")
    print(f"                  scale_y = {nH}/{ORIG_H} = {nH/ORIG_H:.6f}")
    print(f"{'='*70}\n")

    all_ok = True

    for col, row in TEST_POSITIONS:
        # Image noire + carré blanc
        img = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)
        c2  = min(col + PATCH_SZ, ORIG_W)
        r2  = min(row + PATCH_SZ, ORIG_H)
        img[row:r2, col:c2] = 255

        tensor = preprocess(img, mode).to(device)
        with torch.no_grad():
            intermediates = model.get_intermediate_layers(
                tensor, n=LAYER_INDICES, reshape=False,
                return_class_token=False, norm=True
            )

        # On valide avec la dernière couche (la plus informative)
        tokens_flat = intermediates[-1][0]           # (nH*nW, D)
        tokens_grid = tokens_flat.reshape(nH, nW, -1).cpu().float().numpy()  # (nH, nW, D)
        norms = np.linalg.norm(tokens_grid, axis=-1)                          # (nH, nW)

        tx1, ty1, tx2, ty2 = coord_to_tokens(col, row, c2, r2, nH, nW)

        mask = np.zeros((nH, nW), dtype=bool)
        mask[ty1:ty2, tx1:tx2] = True

        mean_patch = float(norms[mask].mean())   if mask.any()  else 0.0
        mean_bg    = float(norms[~mask].mean())  if (~mask).any() else 0.0
        diff_ratio = abs(mean_patch - mean_bg) / (mean_bg + 1e-9)

        ok_strong = diff_ratio >= 0.05
        ok_weak   = diff_ratio >= 0.01
        if not ok_weak:
            all_ok = False

        if ok_strong:
            status = "✓ OK"
        elif ok_weak:
            status = "~ FAIBLE (attention mixing architectural)"
        else:
            status = "✗ FAIL — DÉCALAGE PROBABLE"

        print(f"  (col={col:4d}, row={row:3d})  →  tokens [{ty1}:{ty2}, {tx1}:{tx2}]  "
              f"patch_norm={mean_patch:.4f}  bg_norm={mean_bg:.4f}  "
              f"diff={diff_ratio*100:.1f}%  {status}")

        # ── Visualisation ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"DINOv2 validation coordonnées — col={col}, row={row}  [mode={mode}]\n"
            f"diff={diff_ratio*100:.1f}%  {status}",
            fontsize=11
        )

        # Gauche : image + zones
        ax = axes[0]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, origin='upper',
                  aspect='auto', extent=[0, ORIG_W, ORIG_H, 0])
        # zone extraite remapée en coords originales
        px1 = tx1 / nW * ORIG_W;  px2 = tx2 / nW * ORIG_W
        py1 = ty1 / nH * ORIG_H;  py2 = ty2 / nH * ORIG_H
        ax.add_patch(mpatches.FancyBboxPatch(
            (px1, py1), px2 - px1, py2 - py1,
            boxstyle='square,pad=0', lw=2, edgecolor='red', facecolor='none',
            label='Zone tokens (code)'
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (col, row), c2 - col, r2 - row,
            boxstyle='square,pad=0', lw=2, edgecolor='lime', facecolor='none',
            ls='--', label='Carré blanc réel'
        ))
        ax.set_title('Image synthétique\nRouge=zone extraite  Vert=carré réel', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlabel('col (x)');  ax.set_ylabel('row (y)')

        # Droite : carte des normes
        ax = axes[1]
        im = ax.imshow(norms, cmap='hot', origin='upper', aspect='auto')
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx1 - 0.5, ty1 - 0.5), tx2 - tx1, ty2 - ty1,
            boxstyle='square,pad=0', lw=2, edgecolor='cyan', facecolor='none',
            label='Zone extraite'
        ))
        ax.set_title(
            f'Carte normes tokens ({nH}×{nW})\n'
            f'norme_patch={mean_patch:.3f}  norme_bg={mean_bg:.3f}  {status}',
            fontsize=9
        )
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = VAL_DIR / f'validate_col{col:04d}_row{row:03d}_{mode}.png'
        plt.savefig(fname, dpi=90, bbox_inches='tight')
        plt.close(fig)
        print(f"    → {fname.name}")

    print()
    if all_ok:
        print("  → CONCLUSION : extraction FIABLE ✓  (pipeline peut continuer)")
    else:
        print("  → ⚠ DÉCALAGE DÉTECTÉ — arrêt, corriger la conversion avant de continuer")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Extraction des features par patch
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_one_image(model, img_array: np.ndarray, positions: np.ndarray,
                       nH: int, nW: int, mode: str, device: str) -> dict:
    """
    Forward une image entière, retourne {layer_label: (N_patches, D)}.
    positions : (N, 4) float [x_min, y_min, x_max, y_max]
    """
    tensor = preprocess(img_array, mode).to(device)

    with torch.no_grad():
        intermediates = model.get_intermediate_layers(
            tensor, n=LAYER_INDICES, reshape=False,
            return_class_token=False, norm=True
        )

    result = {}
    for i, li in enumerate(LAYER_INDICES):
        lbl = LAYER_LABELS[li]
        tokens_flat  = intermediates[i][0]                             # (nH*nW, D)
        tokens_grid  = tokens_flat.reshape(nH, nW, -1).cpu().float().numpy()  # (nH, nW, D)
        D            = tokens_grid.shape[-1]
        patch_feats  = np.zeros((len(positions), D), dtype=np.float32)

        for j, (x_min, y_min, x_max, y_max) in enumerate(positions):
            tx1, ty1, tx2, ty2 = coord_to_tokens(x_min, y_min, x_max, y_max, nH, nW)
            region = tokens_grid[ty1:ty2, tx1:tx2]   # (h, w, D)
            patch_feats[j] = region.reshape(-1, D).mean(axis=0)

        result[lbl] = patch_feats
    return result


def step3_extract(model, model_name: str, mode: str, device: str) -> Path:
    h5_path = H5_OUT_DIR / f'dinov2_{model_name}_{mode}.h5'

    if h5_path.exists():
        print(f"  H5 existant : {h5_path.name}  (supprimez-le pour re-extraire)")
        return h5_path

    # Charger métadonnées SAM (mêmes patches → comparaison directe)
    with h5py.File(H5_SAM, 'r') as f:
        all_cat    = f['metadata']['category_ids'][:]
        all_imgs   = np.array([x.decode() for x in f['metadata']['image_names'][:]])
        all_pos    = f['metadata']['positions'][:]      # (N, 4) [x_min,y_min,x_max,y_max]
        all_cnames = np.array([x.decode() for x in f['metadata']['category_names'][:]])

    N           = len(all_cat)
    unique_imgs = sorted(set(all_imgs))
    rH, rW, nH, nW = get_token_grid(mode)

    print(f"\n{'='*70}")
    print(f"ÉTAPE 3 — Extraction : {model_name}  (mode={mode})")
    print(f"  {N} patches sur {len(unique_imgs)} images")
    print(f"  Grille tokens : {nH}×{nW}  |  resize : {rH}×{rW}")

    # Déduire D via une passe test
    sample_img = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)
    test_pos   = np.array([[0., 0., 128., 128.]])
    test_res   = _extract_one_image(model, sample_img, test_pos, nH, nW, mode, device)
    D = list(test_res.values())[0].shape[-1]
    print(f"  Dimension D  : {D}")

    feats_by_layer = {lbl: np.zeros((N, D), dtype=np.float32)
                      for lbl in LAYER_LABELS.values()}

    t0 = time.time()
    for img_name in tqdm(unique_imgs, desc=f'  Extraction {model_name[:20]}'):
        mask     = all_imgs == img_name
        idx      = np.where(mask)[0]
        pos      = all_pos[idx]

        img_path = IMG_DIR / img_name
        img_pil  = Image.open(img_path)
        img_arr  = np.array(img_pil.convert('L'))   # grayscale uint8

        feats = _extract_one_image(model, img_arr, pos, nH, nW, mode, device)
        for lbl, feat_mat in feats.items():
            feats_by_layer[lbl][idx] = feat_mat

    elapsed = time.time() - t0
    print(f"  Extraction terminée en {elapsed:.1f}s")

    # Sauvegarde H5
    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('metadata')
        meta.create_dataset('category_ids',   data=all_cat)
        meta.create_dataset('category_names', data=np.array([x.encode() for x in all_cnames]))
        meta.create_dataset('image_names',    data=np.array([x.encode() for x in all_imgs]))
        meta.create_dataset('positions',      data=all_pos)

        feat_grp = f.create_group('features')
        for lbl, feat_mat in feats_by_layer.items():
            feat_grp.create_dataset(lbl, data=feat_mat)

        f.attrs['model_name']  = model_name
        f.attrs['resize_mode'] = mode
        f.attrs['resize_H']    = rH
        f.attrs['resize_W']    = rW
        f.attrs['n_tokens_H']  = nH
        f.attrs['n_tokens_W']  = nW
        f.attrs['layers_0idx'] = str(LAYER_INDICES)

    print(f"  Sauvé : {h5_path}")
    return h5_path


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — LP LOIO (protocole identique SAM)
# ═══════════════════════════════════════════════════════════════════════════════

def loio_recall_ovr(X: np.ndarray, y: np.ndarray,
                    images: np.ndarray, texture: int) -> list:
    """
    Recall LOIO one-vs-rest pour une texture.
    Protocole identique à lp_par_bl.py :
      - LOIO par image
      - PCA(50) fitée sur le train uniquement
      - LogisticRegression(C=1, balanced, seed=42)
      - recall classe positive sur le test
    """
    y_bin  = (y == texture).astype(int)
    scores = []
    for img_test in sorted(set(images)):
        te = images == img_test
        tr = ~te
        if y_bin[te].sum() == 0:
            continue
        if len(np.unique(y_bin[tr])) < 2:
            continue

        X_tr, X_te = X[tr], X[te]
        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=LP_C, class_weight='balanced',
                                 max_iter=1000, random_state=SEED, solver='lbfgs')
        clf.fit(X_tr, y_bin[tr])
        pred   = clf.predict(X_te)
        scores.append(recall_score(y_bin[te], pred, pos_label=1, zero_division=0))

    return scores


def step4_lp_loio(h5_path: Path, model_name: str) -> dict:
    """
    Retourne results[layer_label][texture] = (mean, std, n_folds).
    """
    with h5py.File(h5_path, 'r') as f:
        all_cat  = f['metadata']['category_ids'][:]
        all_imgs = np.array([x.decode() for x in f['metadata']['image_names'][:]])
        layer_keys = sorted(f['features'].keys())
        feats_all  = {k: f['features'][k][:] for k in layer_keys}

    mask = np.isin(all_cat, TEXTURES)
    idx  = np.where(mask)[0]
    cats = all_cat[idx]
    imgs = all_imgs[idx]
    feats = {k: feats_all[k][idx] for k in layer_keys}

    print(f"\n  LP LOIO : {model_name}  →  {len(idx)} patches utiles "
          f"({len(TEXTURES)} textures, {len(layer_keys)} couches)")

    results = {}
    t0 = time.time()
    for lbl in layer_keys:
        X = feats[lbl]
        results[lbl] = {}
        for t in TEXTURES:
            scores = loio_recall_ovr(X, cats, imgs, t)
            if scores:
                m, s = np.mean(scores), np.std(scores)
                results[lbl][t] = (m, s, len(scores))
            else:
                results[lbl][t] = (float('nan'), float('nan'), 0)

    elapsed = time.time() - t0
    print(f"  LP LOIO terminé en {elapsed:.1f}s")

    # Affichage tableau couche × texture
    print(f"\n  {'Couche':<12}" + "".join(f"  {TNAMES[t][:9]:>11}" for t in TEXTURES))
    print(f"  {'-'*80}")
    for lbl in layer_keys:
        row = f"  {lbl:<12}"
        for t in TEXTURES:
            m, s, n = results[lbl][t]
            row += f"  {m:>5.3f}±{s:.3f}" if not np.isnan(m) else f"  {'—':>9}"
        print(row)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Sorties + comparaison SAM
# ═══════════════════════════════════════════════════════════════════════════════

def _best_dinov2_per_texture(all_results: dict) -> dict:
    """
    all_results[model_name][layer_label][texture] = (mean, std, n_folds)
    Retourne best[texture] = (mean, std, model_name, layer_label).
    """
    best = {}
    for mn, res in all_results.items():
        for lbl, lr in res.items():
            for t in TEXTURES:
                m, s, _ = lr[t]
                if np.isnan(m):
                    continue
                if t not in best or m > best[t][0] or (m == best[t][0] and s < best[t][1]):
                    best[t] = (m, s, mn, lbl)
    return best


def step5_outputs(all_results: dict, mode: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models_list = list(all_results.keys())

    # ── 1. Tableau par modèle ──────────────────────────────────────────────────
    print(f"\n\n{'#'*80}")
    print("# ÉTAPE 5 — RÉSULTATS DÉTAILLÉS PAR MODÈLE")
    print(f"{'#'*80}")

    for mn in models_list:
        res = all_results[mn]
        layer_keys = sorted(res.keys())
        print(f"\n{'='*72}")
        print(f"Modèle : {mn}  (mode={mode})")
        print(f"{'='*72}")
        print(f"{'Couche':<12}" + "".join(f"  {TNAMES[t][:9]:>11}" for t in TEXTURES))
        print("-" * 72)
        for lbl in layer_keys:
            row = f"{lbl:<12}"
            for t in TEXTURES:
                m, s, n = res[lbl][t]
                row += f"  {m:>5.3f}±{s:.3f}" if not np.isnan(m) else f"  {'—':>9}"
            print(row)

        # Meilleure couche par texture
        print()
        print("  Meilleure couche par texture :")
        for t in TEXTURES:
            best_lbl, best_m, best_s = None, -1.0, None
            for lbl in layer_keys:
                m, s, _ = res[lbl][t]
                if not np.isnan(m) and (m > best_m or (m == best_m and s < best_s)):
                    best_lbl, best_m, best_s = lbl, m, s
            if best_lbl:
                print(f"    {TNAMES[t]:<15} → {best_lbl}  recall={best_m:.3f}±{best_s:.3f}")

    # ── 2. Meilleur modèle par texture ────────────────────────────────────────
    best = _best_dinov2_per_texture(all_results)
    print(f"\n{'='*72}")
    print("Meilleur (modèle, couche) par texture (DINOv2 global) :")
    print(f"{'='*72}")
    for t in TEXTURES:
        if t in best:
            m, s, mn, lbl = best[t]
            print(f"  {TNAMES[t]:<15} → {mn}  {lbl}  recall={m:.3f}±{s:.3f}")

    # ── 3. Comparaison SAM vs DINOv2 ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("COMPARAISON DIRECTE SAM (Hiera Small) vs DINOv2 (best)")
    print(f"  Protocole identique → toute différence vient de l'encodeur")
    print(f"  Mode DINOv2 : {mode}  ({'ISO, ratio préservé' if mode=='iso' else 'carré anisotrope comme SAM'})")
    print(f"{'='*72}")
    print(f"  {'Texture':<15} {'SAM recall':>12} {'DINOv2 best':>13} {'Δ':>8}  Modèle, Couche")
    print(f"  {'-'*75}")

    for t in TEXTURES:
        sam_m, sam_s = SAM_RECALL[t]
        if t in best:
            d_m, d_s, d_mn, d_lbl = best[t]
            delta = d_m - sam_m
            sign  = '+' if delta >= 0 else ''
            win   = 'DINOv2↑' if delta > 0.01 else ('SAM↑' if delta < -0.01 else '≈égal')
            print(f"  {TNAMES[t]:<15} {sam_m:.3f}±{sam_s:.3f}  "
                  f"  {d_m:.3f}±{d_s:.3f}  {sign}{delta:+.3f}  [{win}]  {d_mn}, {d_lbl}")

    # ── 4. Focus textures orientées ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Focus textures ORIENTÉES (Filaments=4, Faisceaux=3, Strat.sin=6)")
    print(f"  Hypothèse : resize ISO (non-déformant) ↔ DINOv2 améliore vs SAM")
    print(f"  (SAM utilisait resize anisotrope 1024×1024 qui déformait l'orientation)")
    print(f"{'='*72}")
    oriented = [3, 4, 6]
    for t in oriented:
        sam_m, sam_s = SAM_RECALL[t]
        if t in best:
            d_m, d_s, d_mn, d_lbl = best[t]
            delta = d_m - sam_m
            sign  = '+' if delta >= 0 else ''
            print(f"  {TNAMES[t]:<15} SAM={sam_m:.3f}  DINOv2={d_m:.3f}  Δ={sign}{delta:+.3f}  "
                  f"{d_mn}, {d_lbl}")

    # ── 5. Comparaison modèles S vs B vs B_reg ─────────────────────────────────
    print(f"\n{'='*72}")
    print("Comparaison modèles : ViT-S vs ViT-B vs ViT-B+registers")
    print(f"{'='*72}")
    for mn in models_list:
        res = all_results[mn]
        # Meilleure couche → moyenne sur toutes les textures
        best_per_t = []
        for t in TEXTURES:
            best_m = -1.0
            for lbl in res:
                m, s, _ = res[lbl][t]
                if not np.isnan(m) and m > best_m:
                    best_m = m
            if best_m >= 0:
                best_per_t.append(best_m)
        avg = np.mean(best_per_t) if best_per_t else float('nan')
        print(f"  {mn:<30}  recall_moyen (best couche) = {avg:.3f}")

    # ── 6. Heatmaps par modèle ────────────────────────────────────────────────
    for mn in models_list:
        res = all_results[mn]
        layer_keys = sorted(res.keys())

        data = np.full((len(TEXTURES), len(layer_keys)), np.nan)
        for j, lbl in enumerate(layer_keys):
            for i, t in enumerate(TEXTURES):
                m, s, _ = res[lbl][t]
                if not np.isnan(m):
                    data[i, j] = m

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(np.nan_to_num(data, nan=0.0), cmap='RdYlGn',
                       vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(len(layer_keys)))
        ax.set_xticklabels(layer_keys, fontsize=9)
        ax.set_yticks(range(len(TEXTURES)))
        ax.set_yticklabels([TNAMES[t] for t in TEXTURES], fontsize=9)
        ax.set_xlabel('Couche DINOv2')
        ax.set_title(f'Recall LP LOIO — {mn}  (mode={mode})', fontsize=11)

        for i in range(len(TEXTURES)):
            for j in range(len(layer_keys)):
                v = data[i, j]
                if not np.isnan(v):
                    color = 'white' if v < 0.3 or v > 0.7 else 'black'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Recall LOIO')
        plt.tight_layout()
        fname = OUT_DIR / f'heatmap_{mn}_{mode}.png'
        plt.savefig(fname, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Heatmap : {fname.name}")

    # ── 7. Figure comparaison SAM vs DINOv2 par texture ───────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    x    = np.arange(len(TEXTURES))
    w    = 0.35
    bars_sam    = [SAM_RECALL[t][0]          for t in TEXTURES]
    errs_sam    = [SAM_RECALL[t][1]          for t in TEXTURES]
    bars_dino   = [best[t][0] if t in best else 0.0 for t in TEXTURES]
    errs_dino   = [best[t][1] if t in best else 0.0 for t in TEXTURES]

    ax.bar(x - w/2, bars_sam,  w, yerr=errs_sam,  label='SAM (Hiera-S)', color='steelblue',
           capsize=4, alpha=0.85)
    ax.bar(x + w/2, bars_dino, w, yerr=errs_dino, label=f'DINOv2 best ({mode})', color='tomato',
           capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([TNAMES[t] for t in TEXTURES], rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Recall LOIO (one-vs-rest)', fontsize=10)
    ax.set_title('Séparabilité LP LOIO — SAM vs DINOv2 (meilleur modèle+couche par texture)',
                 fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fname = OUT_DIR / f'comparison_sam_vs_dinov2_{mode}.png'
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparaison : {fname.name}")

    # ── 8. Verdict factuel ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("VERDICT FACTUEL")
    print(f"{'='*72}")

    gains = [(t, best[t][0] - SAM_RECALL[t][0]) for t in TEXTURES if t in best]
    gains_pos = [(t, g) for t, g in gains if g > 0.01]
    gains_neg = [(t, g) for t, g in gains if g < -0.01]
    ties      = [(t, g) for t, g in gains if abs(g) <= 0.01]

    mean_gain = np.mean([g for _, g in gains]) if gains else float('nan')

    print(f"\n  Gain moyen DINOv2 - SAM : {mean_gain:+.3f}")
    print(f"  DINOv2 gagne   ({len(gains_pos)} textures) : " +
          ", ".join(f"{TNAMES[t]} (+{g:.2f})" for t, g in sorted(gains_pos, key=lambda x: -x[1])))
    print(f"  SAM gagne      ({len(gains_neg)} textures) : " +
          ", ".join(f"{TNAMES[t]} ({g:+.2f})" for t, g in sorted(gains_neg, key=lambda x: x[1])))
    print(f"  Quasi-égalité  ({len(ties)} textures)  : " +
          ", ".join(f"{TNAMES[t]}" for t, _ in ties))

    oriented_gains = [g for t, g in gains if t in [3, 4, 6]]
    print(f"\n  Textures orientées (Filaments, Faisceaux, Strat.sin) — gain moyen : "
          f"{np.mean(oriented_gains):+.3f}")
    if np.mean(oriented_gains) > 0.05:
        print("  → Hypothèse déformation SAM CONFIRMÉE : resize ISO améliore les orientées")
    elif np.mean(oriented_gains) < -0.05:
        print("  → Hypothèse déformation SAM RÉFUTÉE : DINOv2 ISO ne les améliore pas")
    else:
        print("  → Résultat ambigu sur les textures orientées (Δ < 5%)")

    # Variance inter-images (std)
    mean_std_sam  = np.mean([SAM_RECALL[t][1] for t in TEXTURES])
    mean_std_dino = np.mean([best[t][1] for t in TEXTURES if t in best])
    print(f"\n  Std inter-images moyenne — SAM : {mean_std_sam:.3f}  |  DINOv2 : {mean_std_dino:.3f}")
    if mean_std_dino < mean_std_sam - 0.03:
        print("  → Variance inter-images RÉDUITE par DINOv2 → l'encodeur aidait")
    elif mean_std_dino > mean_std_sam + 0.03:
        print("  → Variance inter-images AUGMENTÉE → encodeur moins robuste inter-images")
    else:
        print("  → Variance inter-images STABLE → le problème de variance persiste "
              "(problème de dataset, pas d'encodeur)")

    if mean_gain > 0.05:
        print("\n  INTERPRÉTATION : DINOv2 >> SAM → le goulot était Hiera, "
              "changer d'encodeur débloque (résultat fort)")
    elif mean_gain < -0.02:
        print("\n  INTERPRÉTATION : SAM ≥ DINOv2 → l'encodeur n'est pas le facteur limitant")
    else:
        print("\n  INTERPRÉTATION : DINOv2 ≈ SAM → textures intrinsèquement difficiles, "
              "aucun encodeur générique ne suffit → fine-tuning domaine requis (Cell-DINO / Branche C)")

    print(f"\n{'='*72}")
    print(f"Sorties sauvées dans : {OUT_DIR}")
    print(f"Validation visuelle  : {VAL_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Pipeline DINOv2 LP LOIO vs SAM')
    parser.add_argument('--mode', choices=['iso', 'square'], default='iso',
                        help='Mode resize (défaut: iso — préserve ratio, non-déformant)')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help='Modèles à évaluer (défaut: tous)')
    parser.add_argument('--skip-validate', action='store_true',
                        help='Sauter la validation des coordonnées')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Sauter l extraction (utiliser H5 existants)')
    parser.add_argument('--only-first', action='store_true',
                        help='N évaluer que dinov2_vits14_reg (test rapide)')
    args = parser.parse_args()

    if args.only_first:
        args.models = ['dinov2_vits14_reg']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Pipeline DINOv2 — Séparabilité textures MEB                        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Device      : {device}")
    if device == 'cuda':
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Mode resize : {args.mode}")
    rH, rW, nH, nW = get_token_grid(args.mode)
    print(f"  Cible resize: {rH}×{rW}  →  {nH}×{nW} = {nH*nW} tokens")
    print(f"  Modèles     : {args.models}")
    print()
    print("  ÉTAPE 2 — Choix resize documenté :")
    print(f"    Option ISO (utilisée)   : {ISO_H}×{ISO_W} → {ISO_NH}×{ISO_NW} tokens")
    print(f"    Ratio original          : {ORIG_H}×{ORIG_W} = {ORIG_H/ORIG_W:.4f}")
    print(f"    Ratio ISO               : {ISO_H}×{ISO_W} = {ISO_H/ISO_W:.4f}  ✓ identique")
    print(f"    Option carré (SAM-like) : {SQ_H}×{SQ_W} → {SQ_NH}×{SQ_NW} tokens  (anisotrope)")
    print(f"    → L option ISO teste l hypothèse que la déformation SAM nuisait aux")
    print(f"      textures orientées (Filaments, Faisceaux, Strat.sin).")
    print()

    # Charger premier modèle pour validation
    first_model_name = args.models[0]
    print(f"Chargement {first_model_name} (torch.hub.load) ...", flush=True)
    t_load = time.time()
    model = torch.hub.load('facebookresearch/dinov2', first_model_name, verbose=False)
    model = model.eval().to(device)
    print(f"  Chargé en {time.time()-t_load:.1f}s")

    # Étape 0 : sanity check
    ok_sanity, _ = step0_sanity(model, first_model_name, args.mode, device)
    if not ok_sanity:
        print("⚠ Sanity check échoué — arrêt.")
        sys.exit(1)

    # Étape 1 : validation coordonnées (une seule fois, indépendant du modèle)
    if not args.skip_validate:
        ok_coords = step1_validate_coords(model, args.mode, device)
        if not ok_coords:
            print("⚠ Décalage de coordonnées — arrêt.")
            sys.exit(1)

    # Étapes 3 + 4 : extraction + LP LOIO par modèle
    all_results = {}
    current_model_name = first_model_name

    for mn in args.models:
        # Changer de modèle si nécessaire
        if mn != current_model_name:
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f"\nChargement {mn} ...", flush=True)
            t_load = time.time()
            model = torch.hub.load('facebookresearch/dinov2', mn, verbose=False)
            model = model.eval().to(device)
            print(f"  Chargé en {time.time()-t_load:.1f}s")
            current_model_name = mn
            # Sanity check rapide (shapes)
            step0_sanity(model, mn, args.mode, device)

        # Extraction
        if not args.skip_extract:
            h5_path = step3_extract(model, mn, args.mode, device)
        else:
            h5_path = H5_OUT_DIR / f'dinov2_{mn}_{args.mode}.h5'
            if not h5_path.exists():
                print(f"  ⚠ H5 absent : {h5_path} — lancement extraction")
                h5_path = step3_extract(model, mn, args.mode, device)

        # LP LOIO
        print(f"\n{'='*70}")
        print(f"ÉTAPE 4 — LP LOIO : {mn}")
        results = step4_lp_loio(h5_path, mn)
        all_results[mn] = results

    # Étape 5 : sorties + comparaison
    step5_outputs(all_results, args.mode)

    print("\n✓ Pipeline terminé.")


if __name__ == '__main__':
    main()
