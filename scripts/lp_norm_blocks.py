"""
Linear Probing : séparabilité par texture × normalisation × block Hiera.
Split par image, features BRUTES (pas de PCA, pas de L2-norm).
"""

import sys, os, re, struct, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ─── Chemins ─────────────────────────────────────────────────────────────────
ROOT       = Path('/home/aidouni/meb_texture_seg')
SAM2_DIR   = ROOT / 'TextureSAM' / 'sam2'
CKPT       = ROOT / 'checkpoints' / 'sam2.1_hiera_small_1.pt'
PATCH_ROOT = ROOT / 'PatchTagger_Output' / 'patches'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUT_DIR    = ROOT / 'output_ouassim' / 'lp_norm_blocks'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SAM2_DIR))
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ─── Config ──────────────────────────────────────────────────────────────────
TEXTURES   = [1, 3, 4, 5, 6, 7, 9]
BLOCKS     = list(range(3, 14))   # [3..13], stage 3, stride 16
SEED       = 52   # seed 52 = split le plus équilibré (min 15 patches/texture côté train et test)
IMG_SIZE   = 1024
ORIG_H, ORIG_W = 768, 1280
PATCH_PX   = 128

TEXTURE_NAMES = {
    1: "Homogène", 3: "Faisceaux", 4: "Filaments",
    5: "Strat. rect.", 6: "Strat. sinueux", 7: "Granuleux", 9: "Trou"
}

NORMS = ["baseline", "gamma_0.7", "gamma_1.5", "zscore_image"]

# ImageNet norm (toujours appliqué en dernier)
MEAN_IN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_IN  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Stride effectif par bloc (cumul de patch_embed + q_pool)
# blocks 0     : stride 4  → feature map 256×256
# blocks 1-2   : stride 8  → 128×128
# blocks 3-13  : stride 16 → 64×64   ← nos blocks d'intérêt
# blocks 14-15 : stride 32 → 32×32
def effective_stride(block_idx):
    if block_idx < 1:  return 4
    if block_idx < 3:  return 8
    if block_idx < 14: return 16
    return 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ─── Lecture TIFF grayscale ──────────────────────────────────────────────────
def read_tiff_gray(path):
    with open(path, 'rb') as f:
        data = f.read()
    bo = '<' if data[:2] == b'II' else '>'
    ifd_off = struct.unpack(bo+'I', data[4:8])[0]
    pos = ifd_off
    n = struct.unpack(bo+'H', data[pos:pos+2])[0]; pos += 2
    tags = {}
    for _ in range(n):
        entry = data[pos:pos+12]; pos += 12
        tag, dtype, _ = struct.unpack(bo+'HHI', entry[:8])
        v = entry[8:12]
        if dtype == 3:   v = struct.unpack(bo+'H', v[:2])[0]
        elif dtype == 4: v = struct.unpack(bo+'I', v)[0]
        tags[tag] = v
    w, h = tags[256], tags[257]
    off = tags.get(273)
    with open(path, 'rb') as f:
        f.seek(off)
        raw = np.frombuffer(f.read(h * w), dtype=np.uint8).reshape(h, w)
    return raw

# ─── Application de la transformation image (avant ImageNet) ─────────────────
def apply_norm_transform(I_uint8: np.ndarray, norm_name: str) -> np.ndarray:
    """
    I_uint8 : (H, W) uint8
    Retourne : (H, W) float32 dans [0,1] (avant duplication RGB et ImageNet)
    """
    I = I_uint8.astype(np.float32)
    if norm_name == "baseline":
        x = I / 255.0

    elif norm_name == "gamma_0.7":
        x = np.power(I / 255.0, 0.7)

    elif norm_name == "gamma_1.5":
        x = np.power(I / 255.0, 1.5)

    elif norm_name == "zscore_image":
        mu, sg = I.mean(), I.std()
        z = (I - mu) / (sg + 1e-8)
        z = np.clip(z, -3.0, 3.0)
        x = (z + 3.0) / 6.0          # → [0, 1]

    else:
        raise ValueError(f"Normalisation inconnue : {norm_name}")

    return x.astype(np.float32)

def to_tensor_imagenet(x_hw: np.ndarray) -> torch.Tensor:
    """(H, W) float32 [0,1] → (1,3,H,W) tensor après dup RGB + ImageNet."""
    rgb = np.stack([x_hw, x_hw, x_hw], axis=0)   # (3,H,W)
    t = torch.from_numpy(rgb).float()
    t = (t - MEAN_IN) / STD_IN
    return t.unsqueeze(0)   # (1,3,H,W)

# ─── Construction modèle ─────────────────────────────────────────────────────
def build_model():
    trunk = Hiera(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000
        ),
        d_model=256, backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model='nearest', fuse_type='sum', fpn_top_down_levels=[2, 3],
    )
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
    prefix = 'image_encoder.'
    sd_enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sd_enc, strict=False)
    print(f"  Checkpoint : missing={len(missing)}, unexpected={len(unexpected)}")
    return encoder.to(DEVICE).eval()

def build_hooks(encoder):
    block_feats = {}   # {block_idx: (B, H, W, C) tensor}
    handles = []
    for i in range(len(encoder.trunk.blocks)):
        def _hook(m, inp, out, _i=i):
            block_feats[_i] = out.detach().cpu()
        handles.append(encoder.trunk.blocks[i].register_forward_hook(_hook))
    return block_feats, handles

# ─── Extraction feature pour une patch ───────────────────────────────────────
def extract_patch_feat(block_feat_bhwc: torch.Tensor,
                       row_idx: int, col_idx: int,
                       block_idx: int) -> np.ndarray:
    """
    Extrait le vecteur feature moyen correspondant à une patch de l'image.
    block_feat_bhwc : (1, H_feat, W_feat, C)
    row_idx, col_idx : position dans la grille d'origine (0-indexed)
    block_idx : pour calculer le stride effectif
    Retourne : (C,) float32
    """
    feat = block_feat_bhwc[0].numpy()   # (H_feat, W_feat, C)
    H_feat, W_feat, C = feat.shape
    stride = effective_stride(block_idx)

    # Coordonnées dans l'image 1024×1024 (après resize)
    scale_h = IMG_SIZE / ORIG_H
    scale_w = IMG_SIZE / ORIG_W

    y0 = row_idx * PATCH_PX * scale_h
    x0 = col_idx * PATCH_PX * scale_w
    y1 = y0 + PATCH_PX * scale_h
    x1 = x0 + PATCH_PX * scale_w

    # Coordonnées dans la feature map
    fy0 = int(y0 / stride)
    fy1 = max(int(y1 / stride), fy0 + 1)
    fx0 = int(x0 / stride)
    fx1 = max(int(x1 / stride), fx0 + 1)

    fy0, fy1 = max(0, fy0), min(H_feat, fy1)
    fx0, fx1 = max(0, fx0), min(W_feat, fx1)

    region = feat[fy0:fy1, fx0:fx1, :]   # (h, w, C)
    return region.mean(axis=(0, 1)).astype(np.float32)   # (C,)

# ─── Parsing patches ─────────────────────────────────────────────────────────
def load_patch_list():
    pattern = re.compile(r'^(.+)_\((\d+)_(\d+)\)$')
    patches = []
    for t in TEXTURES:
        for f in sorted((PATCH_ROOT / str(t)).iterdir()):
            if f.suffix != '.tif' or '_cp_masks_' in f.name:
                continue
            m = pattern.match(f.stem)
            if m:
                patches.append({
                    'texture': t,
                    'img_stem': m.group(1),
                    'row': int(m.group(2)),
                    'col': int(m.group(3)),
                    'patch_path': f,
                })
    return patches

# ─── Split train/test par image ───────────────────────────────────────────────
def make_split(patches):
    """Split 70/30 par image, permutation aléatoire (seed optimisé pour équilibre)."""
    img_textures = defaultdict(list)
    for p in patches:
        img_textures[p['img_stem']].append(p['texture'])

    imgs = sorted(img_textures.keys())
    rng  = np.random.default_rng(SEED)
    idx  = rng.permutation(len(imgs))
    n_test     = max(1, int(len(imgs) * 0.30))
    test_imgs  = [imgs[i] for i in idx[:n_test]]
    train_imgs = [imgs[i] for i in idx[n_test:]]
    train_set, test_set = set(train_imgs), set(test_imgs)

    train_p = [p for p in patches if p['img_stem'] in train_set]
    test_p  = [p for p in patches if p['img_stem'] in test_set]
    return train_p, test_p, train_set, test_set

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n═══ Linear Probing : texture × normalisation × block ═══\n")

    patches = load_patch_list()
    print(f"Total patches : {len(patches)}")
    for t in TEXTURES:
        n = sum(1 for p in patches if p['texture'] == t)
        print(f"  texture {t} ({TEXTURE_NAMES[t]:<16}) : {n:>3} patches")

    train_p, test_p, train_set, test_set = make_split(patches)
    print(f"\nSplit : {len(train_set)} images train, {len(test_set)} images test")
    print(f"        {len(train_p)} patches train, {len(test_p)} patches test")

    # Vérification que chaque texture est dans train ET test
    for t in TEXTURES:
        n_tr = sum(1 for p in train_p if p['texture'] == t)
        n_te = sum(1 for p in test_p  if p['texture'] == t)
        print(f"  t{t} ({TEXTURE_NAMES[t]:<16}) : train={n_tr:>3}  test={n_te:>3}")

    print("\n─── Vérification zscore_image (stats avant ImageNet) ───")
    sample_img = read_tiff_gray(IMG_DIR / (list(train_set)[0] + '.tif'))
    sample_r = Image.fromarray(sample_img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    I = np.array(sample_r, dtype=np.float32)
    mu, sg = I.mean(), I.std()
    z = np.clip((I - mu) / (sg + 1e-8), -3, 3)
    x_z = (z + 3) / 6
    print(f"  Après remap [0,1] : min={x_z.min():.4f}  max={x_z.max():.4f}  "
          f"mean={x_z.mean():.4f}  std={x_z.std():.4f}")

    # Construire modèle
    print("\n─── Chargement modèle ───")
    encoder = build_model()
    block_feats, handles = build_hooks(encoder)

    # Unique images à traiter
    all_img_stems = set(p['img_stem'] for p in patches)

    # Résultats : results[norm][block][texture] = {train, test}
    # confusion : confusion[norm][block] = matrix sur test
    results  = {n: {b: {} for b in BLOCKS} for n in NORMS}
    conf_mats = {n: {b: None for b in BLOCKS} for n in NORMS}

    for norm_name in NORMS:
        print(f"\n══ Normalisation : {norm_name} ══")
        t_start = time.time()

        # ── Extraction features : une passe par image ───────────────────────
        # features_cache[img_stem][block_idx] = (C,) vecteur moyen du patch
        # En fait on stocke les features de CHAQUE patch
        patch_feats = []   # liste de {texture, img_stem, block→feat_vector, split}

        for img_stem in sorted(all_img_stems):
            img_path = IMG_DIR / (img_stem + '.tif')
            I_gray = read_tiff_gray(img_path)
            x_01   = apply_norm_transform(I_gray, norm_name)
            # Resize 768×1280 → 1024×1024
            pil_r  = Image.fromarray((x_01 * 255).clip(0, 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
            x_01r  = np.array(pil_r, dtype=np.float32) / 255.0
            tensor = to_tensor_imagenet(x_01r).to(DEVICE)

            block_feats.clear()
            with torch.no_grad():
                _ = encoder(tensor)

            # Toutes les patches de cette image
            img_patches = [p for p in patches if p['img_stem'] == img_stem]
            split_flag  = 'train' if img_stem in train_set else 'test'

            for p in img_patches:
                feat_per_block = {}
                for b in BLOCKS:
                    if b in block_feats:
                        feat_per_block[b] = extract_patch_feat(
                            block_feats[b], p['row'], p['col'], b
                        )
                patch_feats.append({
                    'texture': p['texture'],
                    'feats': feat_per_block,
                    'split': split_flag,
                })

        print(f"  Extraction terminée ({time.time()-t_start:.1f}s)")

        # ── LP par block ────────────────────────────────────────────────────
        for b in BLOCKS:
            # Collecter X, y pour train et test
            X_tr, y_tr, X_te, y_te = [], [], [], []
            for pf in patch_feats:
                if b not in pf['feats']:
                    continue
                v = pf['feats'][b]
                if pf['split'] == 'train':
                    X_tr.append(v); y_tr.append(pf['texture'])
                else:
                    X_te.append(v); y_te.append(pf['texture'])

            X_tr = np.array(X_tr); y_tr = np.array(y_tr)
            X_te = np.array(X_te); y_te = np.array(y_te)

            # LogReg (features BRUTES, pas de L2-norm, pas de PCA)
            clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0,
                                     solver='lbfgs', n_jobs=-1)
            clf.fit(X_tr, y_tr)

            pred_tr = clf.predict(X_tr)
            pred_te = clf.predict(X_te)

            # Recall par texture (= accuracy de la classe)
            for t in TEXTURES:
                mask_tr = y_tr == t
                mask_te = y_te == t
                recall_tr = (pred_tr[mask_tr] == t).mean() if mask_tr.sum() > 0 else np.nan
                recall_te = (pred_te[mask_te] == t).mean() if mask_te.sum() > 0 else np.nan
                results[norm_name][b][t] = {'train': float(recall_tr), 'test': float(recall_te)}

            # Matrice de confusion (test)
            conf_mats[norm_name][b] = confusion_matrix(y_te, pred_te, labels=TEXTURES)

        print(f"  LP terminé ({time.time()-t_start:.1f}s total)")

    # Nettoyer hooks
    for h in handles:
        h.remove()

    # ─── SORTIE 1 : Heatmaps ─────────────────────────────────────────────────
    print("\n─── Génération heatmaps ───")

    fig, axes = plt.subplots(
        len(TEXTURES), 2, figsize=(10, 3 * len(TEXTURES)),
        constrained_layout=True
    )

    for row_idx, t in enumerate(TEXTURES):
        for col_idx, split in enumerate(['train', 'test']):
            ax = axes[row_idx, col_idx]

            # Matrice : (N_blocks, N_norms)
            data = np.zeros((len(BLOCKS), len(NORMS)))
            for bi, b in enumerate(BLOCKS):
                for ni, norm in enumerate(NORMS):
                    val = results[norm][b].get(t, {}).get(split, np.nan)
                    data[bi, ni] = val

            im = ax.imshow(data, vmin=0, vmax=1, aspect='auto', cmap='RdYlGn',
                           interpolation='nearest')
            ax.set_yticks(range(len(BLOCKS)))
            ax.set_yticklabels([str(b) for b in BLOCKS], fontsize=8)
            ax.set_xticks(range(len(NORMS)))
            ax.set_xticklabels(NORMS, fontsize=7, rotation=25, ha='right')
            ax.set_ylabel('Block', fontsize=8)

            title_split = 'Train' if split == 'train' else 'Test'
            ax.set_title(f"{TEXTURE_NAMES[t]}  [{title_split}]", fontsize=9, pad=3)

            # Valeurs dans les cellules
            for bi in range(len(BLOCKS)):
                for ni in range(len(NORMS)):
                    v = data[bi, ni]
                    if not np.isnan(v):
                        ax.text(ni, bi, f"{v:.2f}", ha='center', va='center',
                                fontsize=6, color='black' if 0.3 < v < 0.8 else 'white')

    fig.colorbar(im, ax=axes, shrink=0.4, label='Recall (accuracy classe)')
    out_hm = OUT_DIR / 'heatmaps_lp.png'
    plt.savefig(out_hm, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {out_hm}")

    # ─── SORTIE 2 : Tableau récapitulatif ─────────────────────────────────────
    print("\n─── Tableau récapitulatif (meilleur par texture, sur test) ───")
    print(f"\n{'Texture':<18} {'Meilleure norm':<16} {'Meilleur block':>14} {'acc_test':>9} {'2e texture confondue':>22}")
    print("─" * 85)

    for t in TEXTURES:
        best_acc, best_norm, best_block = -1, None, None
        for norm in NORMS:
            for b in BLOCKS:
                v = results[norm][b].get(t, {}).get('test', -1)
                if v > best_acc:
                    best_acc, best_norm, best_block = v, norm, b

        # Texture la plus confondue : dans la matrice de confusion du meilleur (norm, block)
        cm = conf_mats[best_norm][best_block]
        row_idx_t = TEXTURES.index(t)
        row = cm[row_idx_t].copy()
        row[row_idx_t] = 0   # ignorer la diagonale
        confused_idx = np.argmax(row)
        confused_t   = TEXTURES[confused_idx]
        confused_name = TEXTURE_NAMES[confused_t] if row.sum() > 0 else "—"

        print(f"{TEXTURE_NAMES[t]:<18} {best_norm:<16} {best_block:>14} "
              f"{best_acc:>9.3f} {confused_name:>22}")

    # ─── SORTIE 3 (préparée pour affichage, sans commentaires dans les plots) ──
    # Sauvegarde résultats bruts
    import json
    def convert(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    results_serial = {}
    for norm in NORMS:
        results_serial[norm] = {}
        for b in BLOCKS:
            results_serial[norm][str(b)] = {
                str(t): results[norm][b][t] for t in TEXTURES
            }
    with open(OUT_DIR / 'lp_results.json', 'w') as f:
        json.dump(results_serial, f, indent=2, default=convert)
    print(f"\n  → {OUT_DIR / 'lp_results.json'}")

    # ─── Matrice de confusion du meilleur bloc global (moyenne test) ──────────
    # Trouver le (norm, block) avec la meilleure accuracy globale (macro) sur test
    best_global_acc, best_g_norm, best_g_block = -1, None, None
    for norm in NORMS:
        for b in BLOCKS:
            macro = np.mean([
                results[norm][b].get(t, {}).get('test', 0) for t in TEXTURES
            ])
            if macro > best_global_acc:
                best_global_acc, best_g_norm, best_g_block = macro, norm, b

    print(f"\n  Meilleur global (macro test) : {best_g_norm}, block {best_g_block}  → {best_global_acc:.3f}")

    cm = conf_mats[best_g_norm][best_g_block].astype(float)
    # Normaliser par ligne (recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im2 = ax2.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    tnames = [TEXTURE_NAMES[t] for t in TEXTURES]
    ax2.set_xticks(range(len(TEXTURES))); ax2.set_xticklabels(tnames, rotation=35, ha='right', fontsize=8)
    ax2.set_yticks(range(len(TEXTURES))); ax2.set_yticklabels(tnames, fontsize=8)
    ax2.set_xlabel('Prédit', fontsize=9)
    ax2.set_ylabel('Réel', fontsize=9)
    ax2.set_title(f'Confusion (test) — {best_g_norm}, block {best_g_block}', fontsize=10)
    for i in range(len(TEXTURES)):
        for j in range(len(TEXTURES)):
            ax2.text(j, i, f"{cm_norm[i,j]:.2f}", ha='center', va='center',
                     fontsize=7, color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.colorbar(im2, ax=ax2, label='Recall normalisé')
    plt.tight_layout()
    out_cm = OUT_DIR / 'confusion_best_global.png'
    plt.savefig(out_cm, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → {out_cm}")

    print("\n═══ Terminé ═══")

if __name__ == '__main__':
    main()
