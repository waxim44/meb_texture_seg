"""
train.py
══════════════════════════════════════════════════════════════════════════════
PHASE 3 — Tête de projection + loss SupCon + batching mutualisé par (image, t).

Utilisé par scripts/lora_supcon_phase3.py (sanity check overfit) et par la
future phase 4 (fold pilote).
══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SAM2 = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from transforms import transform_image, transform_coords, TRANSFORMS  # noqa: E402

IMG_SIZE = 1024
ORIG_H, ORIG_W = 768, 1280
PATCH_SZ = 128
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TRAIN_BLOCK = "block_9"  # bloc "champion" (stage 3, sous LoRA) utilisé pour SupCon
TRAIN_BLOCK_DIM = 384


# ─────────────────────────────────────────────────────────────────────────────
# Tête de projection
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP 2 couches (dim_features → 256 → 128), L2-norm en sortie."""

    def __init__(self, dim_in=TRAIN_BLOCK_DIM, dim_hidden=256, dim_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Loss SupCon (Khosla et al. 2020)
# ─────────────────────────────────────────────────────────────────────────────

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    features : (N, D) déjà L2-normalisées.
    labels   : (N,) entiers, positifs = même label (hors soi-même).
    """
    device = features.device
    N = features.shape[0]

    sim = torch.matmul(features, features.T) / temperature  # (N, N)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()  # stabilité numérique

    logits_mask = torch.ones((N, N), device=device) - torch.eye(N, device=device)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float().to(device) * logits_mask

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]

    loss = -mean_log_prob_pos.mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Préparation image / extraction features
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_array(arr: np.ndarray, device) -> torch.Tensor:
    img = Image.fromarray(arr).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device)


def register_single_block_hook(trunk, block_idx):
    captured = {}

    def _hook(m, inp, out):
        captured["feat"] = out  # NOT detached : garde le graphe pour le backward

    handle = trunk.blocks[block_idx].register_forward_hook(_hook)
    return captured, handle


def extract_patch_vec(feat_map, x_min, y_min, x_max, y_max, orig_H, orig_W):
    """feat_map: (1, H_feat, W_feat, C). Retourne (C,) — garde le graphe."""
    feat = feat_map[0]
    H_feat, W_feat, C = feat.shape
    scale_x = W_feat / orig_W
    scale_y = H_feat / orig_H
    fx1 = max(0, int(x_min * scale_x))
    fy1 = max(0, int(y_min * scale_y))
    fx2 = min(W_feat, max(fx1 + 1, int(x_max * scale_x)))
    fy2 = min(H_feat, max(fy1 + 1, int(y_max * scale_y)))
    region = feat[fy1:fy2, fx1:fx2, :]
    return region.mean(dim=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Batch : échantillonnage équilibré + mutualisation par (image, t)
# ─────────────────────────────────────────────────────────────────────────────

def sample_balanced_batch(patches_by_texture: dict, n_per_texture: int, rng, use_augmentation: bool):
    """
    patches_by_texture : {texture: [(image_path, x0,y0,x1,y1), ...]}
    Retourne une liste de (image_path, x0,y0,x1,y1, texture, t).
    """
    batch = []
    for texture, patches in patches_by_texture.items():
        chosen = rng.choices(patches, k=n_per_texture) if len(patches) < n_per_texture else \
            rng.sample(patches, n_per_texture)
        for (img_path, x0, y0, x1, y1) in chosen:
            t = rng.choice(TRANSFORMS) if use_augmentation else "identity"
            batch.append((img_path, x0, y0, x1, y1, texture, t))
    return batch


def sample_balanced_batch_bounded_images(
    patches_by_image: dict, n_per_texture: int, max_images: int, rng, use_augmentation: bool
):
    """
    patches_by_image : {image_path: {texture: [(x0,y0,x1,y1), ...]}}

    Borne le nombre d'images sources distinctes par batch à `max_images`
    (couverture-ensembliste gloutonne des textures présentes, puis complétée
    aléatoirement) — nécessaire car forward_batch garde tous les graphes de
    calcul en mémoire jusqu'au backward final : un batch touchant N images
    distinctes retient N forwards simultanément (coûteux en temps ET mémoire
    si N est grand, cf. rapport phase 4).

    Un seul transform t est tiré par image sélectionnée (partagé par tous les
    patchs qui en sont issus dans ce batch), donc nb forwards = nb images
    sélectionnées, quel que soit le nombre de patchs.
    """
    all_textures = set()
    for tex_map in patches_by_image.values():
        all_textures.update(tex_map.keys())

    images = list(patches_by_image.keys())
    remaining = set(all_textures)
    selected = []
    pool = list(images)
    rng.shuffle(pool)

    # Couverture gloutonne : priorité aux images couvrant le plus de textures manquantes
    while remaining and pool and len(selected) < max_images:
        best_img = max(pool, key=lambda im: len(patches_by_image[im].keys() & remaining))
        covered = patches_by_image[best_img].keys() & remaining
        if not covered:
            break
        selected.append(best_img)
        remaining -= covered
        pool.remove(best_img)

    # Complète jusqu'à max_images avec des images aléatoires pour la diversité
    while pool and len(selected) < max_images:
        img = pool.pop()
        selected.append(img)

    image_transforms = {
        img: (rng.choice(TRANSFORMS) if use_augmentation else "identity") for img in selected
    }

    texture_pool = defaultdict(list)  # texture -> [(image_path, x0,y0,x1,y1), ...] restreint à `selected`
    for img in selected:
        for texture, patches in patches_by_image[img].items():
            for p in patches:
                texture_pool[texture].append((img, *p))

    batch = []
    for texture in all_textures:
        candidates = texture_pool.get(texture, [])
        if not candidates:
            continue  # texture non couverte par les images sélectionnées (rare, cf. couverture)
        chosen = rng.choices(candidates, k=n_per_texture) if len(candidates) < n_per_texture else \
            rng.sample(candidates, n_per_texture)
        for (img_path, x0, y0, x1, y1) in chosen:
            t = image_transforms[img_path]
            batch.append((img_path, x0, y0, x1, y1, texture, t))
    return batch


def forward_batch(encoder, head, batch, device, block_idx=None, block_key=TRAIN_BLOCK):
    """
    Mutualise les forwards par (image_path, t) : un seul forward encodeur par
    couple distinct, extraction de tous les patchs concernés dans ce forward.
    Retourne (embeddings (N,128), labels_str list) dans l'ordre de `batch`.
    """
    if block_idx is None:
        block_idx = int(block_key.split("_")[1])

    groups = defaultdict(list)  # (image_path, t) -> [batch_idx, ...]
    for i, (img_path, *_rest) in enumerate(batch):
        t = _rest[-1]
        groups[(img_path, t)].append(i)

    embeddings = [None] * len(batch)
    labels = [None] * len(batch)
    image_cache = {}

    for (img_path, t), idxs in groups.items():
        if img_path not in image_cache:
            image_cache[img_path] = np.array(Image.open(img_path))
        img_arr = image_cache[img_path]
        H, W = img_arr.shape
        img_t = transform_image(img_arr, t)

        x = preprocess_array(img_t, device)
        captured, handle = register_single_block_hook(encoder.trunk, block_idx)
        encoder(x)
        handle.remove()
        feat_map = captured["feat"]

        for i in idxs:
            _img_path, x0, y0, x1, y1, texture, _t = batch[i]
            row0, col0 = int(y0), int(x0)
            row0p, col0p, Hp, Wp = transform_coords(row0, col0, t, H, W, S=PATCH_SZ)
            xp_min, yp_min, xp_max, yp_max = col0p, row0p, col0p + PATCH_SZ, row0p + PATCH_SZ
            vec = extract_patch_vec(feat_map, xp_min, yp_min, xp_max, yp_max, Hp, Wp)
            emb = head(vec)
            embeddings[i] = emb
            labels[i] = texture

    return torch.stack(embeddings, dim=0), labels
