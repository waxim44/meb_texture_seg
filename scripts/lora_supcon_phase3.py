"""
lora_supcon_phase3.py
══════════════════════════════════════════════════════════════════════════════
PHASE 3 — Sanity check AVANT tout entraînement réel.

Tests :
  1. OVERFIT VOLONTAIRE : sur un mini-sous-ensemble (2-3 images, ~50 patchs),
     SANS augmentation, le loss SupCon doit CHUTER fortement en <200 steps.
  2. CONTRÔLE NÉGATIF : avec LoRA gelé (aucun paramètre entraîné), le loss
     ne doit montrer aucune tendance de baisse.
  3. Mémoire GPU et temps/step affichés.
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SAM2 = _ROOT / "TextureSAM" / "sam2"
sys.path.insert(0, str(_SAM2))
sys.path.insert(0, str(_ROOT / "lora_supcon"))

from sam2.modeling.backbones.hieradet import Hiera  # noqa: E402
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck  # noqa: E402
from sam2.modeling.position_encoding import PositionEmbeddingSine  # noqa: E402
from lora import apply_lora, LORA_BLOCKS  # noqa: E402
from train import ProjectionHead, supcon_loss, sample_balanced_batch, forward_batch, TRAIN_BLOCK  # noqa: E402

CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
H5_PATH = _ROOT / "data" / "feature_database" / "database_meb_ouassim.h5"
IMG_DIR = _ROOT / "Image_Ouassim"
OUTDIR = _ROOT / "lora_supcon" / "phase_3"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
N_STEPS = 200
N_PER_TEXTURE = None  # défini dynamiquement selon les textures présentes dans le mini-set
LR = 1e-4
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.1

# Mini-sous-ensemble choisi pour la diversité de textures (~50 patchs, 2 images)
MINI_IMAGES = [
    "060722-Nabila-JP-Valves-WholeMount-SAureus-pat04-1-22.tif",
    "070525-JPB-MEB-EIHNValves-Ech5-ZigZag0005.tif",
]
MINI_CAP_PER_IMAGE = 25  # sous-échantillonnage pour rester proche de ~50 patchs

TARGET_TEXTURES = {
    "Totalement homogène", "Trou", "Granuleux", "Stratifié rectiligne",
    "Filaments", "Stratifié sinueux", "Faisceaux",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_image_encoder():
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


def load_encoder():
    encoder = build_image_encoder()
    sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    prefix = "image_encoder."
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    assert len(missing) == 0
    return encoder


def load_mini_set():
    f = h5py.File(str(H5_PATH), "r")
    names = f["metadata/image_names"][:]
    cats = f["metadata/category_names"][:]
    pos = f["metadata/positions"][:]

    by_texture = {}
    rng = random.Random(SEED)
    for img_name in MINI_IMAGES:
        idxs = [
            i for i in range(len(names))
            if names[i].decode() == img_name and cats[i].decode() in TARGET_TEXTURES
        ]
        rng.shuffle(idxs)
        idxs = idxs[:MINI_CAP_PER_IMAGE]
        for i in idxs:
            texture = cats[i].decode()
            x0, y0, x1, y1 = pos[i]
            img_path = str(IMG_DIR / img_name)
            by_texture.setdefault(texture, []).append((img_path, float(x0), float(y0), float(x1), float(y1)))

    n_total = sum(len(v) for v in by_texture.values())
    return by_texture, n_total


def build_model(freeze_lora: bool):
    encoder = load_encoder()
    lora_modules = apply_lora(encoder, LORA_BLOCKS)
    encoder = encoder.to(DEVICE)
    head = ProjectionHead().to(DEVICE)

    if freeze_lora:
        for m in lora_modules:
            m.lora_A.requires_grad = False
            m.lora_B.requires_grad = False
        for p in head.parameters():
            p.requires_grad = False
        params = []
    else:
        params = [p for m in lora_modules for p in (m.lora_A, m.lora_B)] + list(head.parameters())

    encoder.train()
    head.train()
    return encoder, head, lora_modules, params


def run_loop(by_texture, freeze_lora: bool, n_steps: int, lines: list, label: str):
    seed_everything(SEED)
    encoder, head, lora_modules, params = build_model(freeze_lora)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY) if params else None

    rng = random.Random(SEED + 1)
    n_per_texture = max(1, 50 // max(1, len(by_texture)))

    losses = []
    step_times = []
    peak_mem = 0.0

    for step in range(n_steps):
        t0 = time.time()
        batch = sample_balanced_batch(by_texture, n_per_texture, rng, use_augmentation=False)
        labels_texture = [b[5] for b in batch]
        label_ids = torch.tensor([sorted(set(labels_texture)).index(l) for l in labels_texture])

        if freeze_lora:
            with torch.no_grad():
                embeddings, _ = forward_batch(encoder, head, batch, DEVICE)
        else:
            optimizer.zero_grad()
            embeddings, _ = forward_batch(encoder, head, batch, DEVICE)

        loss = supcon_loss(embeddings, label_ids.to(DEVICE), temperature=TEMPERATURE)

        if not freeze_lora:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        step_times.append(time.time() - t0)
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)

        if step % 20 == 0 or step == n_steps - 1:
            lines.append(f"  [{label}] step {step:3d}  loss={loss.item():.4f}  batch_size={len(batch)}")

    return losses, step_times, peak_mem


def main():
    lines = ["═" * 70, "PHASE 3 — Rapport de validation", "═" * 70]

    by_texture, n_total = load_mini_set()
    lines.append(f"Mini-sous-ensemble : {len(MINI_IMAGES)} images, {n_total} patchs")
    for tex, patches in by_texture.items():
        lines.append(f"  {tex:25s} : {len(patches)} patchs")

    lines.append("")
    lines.append("── 1. OVERFIT VOLONTAIRE (LoRA entraîné, sans augmentation) ──")
    losses_train, times_train, mem_train = run_loop(by_texture, freeze_lora=False, n_steps=N_STEPS, lines=lines, label="train")

    lines.append("")
    lines.append("── 2. CONTRÔLE NÉGATIF (LoRA gelé) ──")
    losses_frozen, times_frozen, mem_frozen = run_loop(by_texture, freeze_lora=True, n_steps=N_STEPS, lines=lines, label="frozen")

    # Courbes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(losses_train)
    axes[0].set_title("Loss SupCon — LoRA entraîné")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[1].plot(losses_frozen, color="gray")
    axes[1].set_title("Loss SupCon — LoRA gelé (contrôle négatif)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")
    plt.tight_layout()
    fig_path = OUTDIR / "loss_curves.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()

    lines.append("")
    lines.append("── 3. MÉMOIRE / TEMPS ──")
    lines.append(f"Temps moyen/step (train)  : {np.mean(times_train)*1000:.1f} ms")
    lines.append(f"Temps moyen/step (frozen) : {np.mean(times_frozen)*1000:.1f} ms")
    lines.append(f"Mémoire GPU pic (train)   : {mem_train:.2f} GB")
    lines.append(f"Mémoire GPU pic (frozen)  : {mem_frozen:.2f} GB")

    # Verdicts
    early_train = np.mean(losses_train[:10])
    late_train = np.mean(losses_train[-10:])
    drop_ratio = (early_train - late_train) / early_train if early_train > 0 else 0
    ok_overfit = drop_ratio > 0.3  # chute forte

    early_frozen = np.mean(losses_frozen[:10])
    late_frozen = np.mean(losses_frozen[-10:])
    frozen_drop_ratio = abs(early_frozen - late_frozen) / early_frozen if early_frozen > 0 else 0
    ok_frozen = frozen_drop_ratio < 0.05  # reste ~constant (pas de tendance de baisse)

    lines.append("")
    lines.append("── VALIDATION P3 ──")
    lines.append(f"Loss train : early={early_train:.4f}  late={late_train:.4f}  chute={100*drop_ratio:.1f}%")
    lines.append(f"[{'x' if ok_overfit else ' '}] OVERFIT VOLONTAIRE (chute forte, >30% attendu)")
    lines.append(f"Loss frozen : early={early_frozen:.4f}  late={late_frozen:.4f}  variation={100*frozen_drop_ratio:.1f}%")
    lines.append(f"[{'x' if ok_frozen else ' '}] Contrôle négatif (loss stable, <5% variation)")
    lines.append(f"[x] Mémoire / temps par step affichés")
    go = ok_overfit and ok_frozen
    lines.append("")
    lines.append(f"VERDICT : {'GO' if go else 'NO-GO'}")

    report = "\n".join(lines)
    print(report)
    (OUTDIR / "report.txt").write_text(report)
    return go


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
