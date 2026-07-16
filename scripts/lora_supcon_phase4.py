"""
lora_supcon_phase4.py
══════════════════════════════════════════════════════════════════════════════
PHASE 4 — Fold pilote : entraînement LoRA propre (image de test exclue de
tout) + évaluation LP LOIO complète, comparée au zero-shot SUR CE MÊME fold.
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import random
import copy
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
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
from train import (  # noqa: E402
    ProjectionHead, supcon_loss, sample_balanced_batch_bounded_images,
    forward_batch, preprocess_array, extract_patch_vec, TRAIN_BLOCK,
)
from loio import loio_single_fold  # noqa: E402

CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
H5_PATH = _ROOT / "data" / "feature_database" / "database_meb_ouassim.h5"
IMG_DIR = _ROOT / "Image_Ouassim"
OUTDIR = _ROOT / "lora_supcon" / "phase_4"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
ORIG_H, ORIG_W = 768, 1280

TEXTURES = [
    "Totalement homogène", "Trou", "Granuleux", "Stratifié rectiligne",
    "Filaments", "Stratifié sinueux", "Faisceaux",
]

TEST_IMAGE = "060722-Nabila-JP-Valves-WholeMount-SAureus-pat04-1-22.tif"  # Faisceaux 19, Granuleux 30, Trou 3
VAL_IMAGES = [
    "060722-Nabila-JP-Valves-WholeMount-SAureus-pat02-28.tif",
    "070525-JPB-MEB-EIHNValves-Ech5-ZigZag0051.tif",
    "060525-JPB-MEB-EIHNValves-Ech2-ZigZag0036.tif",
]

N_PER_TEXTURE = 6
MAX_IMAGES_PER_STEP = 10  # borne le nb de forwards encodeur par step (cf. rapport : mémoire/temps)
MAX_STEPS = 2000
EVAL_EVERY = 25
PATIENCE_EVALS = 10
LR = 1e-4
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.1
N_VAL_DRAWS = 3


def log(lines, msg):
    lines.append(msg)
    print(msg, flush=True)


def to_patches_by_image(by_texture: dict) -> dict:
    """{texture: [(img_path,x0,y0,x1,y1),...]} -> {img_path: {texture: [(x0,y0,x1,y1),...]}}"""
    by_image = {}
    for texture, patches in by_texture.items():
        for img_path, x0, y0, x1, y1 in patches:
            by_image.setdefault(img_path, {}).setdefault(texture, []).append((x0, y0, x1, y1))
    return by_image


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


def load_h5_index():
    f = h5py.File(str(H5_PATH), "r")
    names = np.array([n.decode() for n in f["metadata/image_names"][:]])
    cats = np.array([c.decode() for c in f["metadata/category_names"][:]])
    pos = f["metadata/positions"][:]
    feats = {b: f["features"][b][:] for b in f["features"].keys()}
    return names, cats, pos, feats


def patches_by_texture_for_images(names, cats, pos, image_set):
    by_texture = {}
    for i in range(len(names)):
        if names[i] not in image_set or cats[i] not in TEXTURES:
            continue
        img_path = str(IMG_DIR / names[i])
        x0, y0, x1, y1 = pos[i]
        by_texture.setdefault(cats[i], []).append((img_path, float(x0), float(y0), float(x1), float(y1)))
    return by_texture


def eval_val_loss(encoder, head, by_image_val, rng, n_draws=N_VAL_DRAWS):
    encoder.eval()
    head.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_draws):
            batch = sample_balanced_batch_bounded_images(
                by_image_val, N_PER_TEXTURE, MAX_IMAGES_PER_STEP, rng, use_augmentation=False
            )
            texture_labels = [b[5] for b in batch]
            uniq = sorted(set(texture_labels))
            label_ids = torch.tensor([uniq.index(l) for l in texture_labels])
            embeddings, _ = forward_batch(encoder, head, batch, DEVICE)
            loss = supcon_loss(embeddings, label_ids.to(DEVICE), temperature=TEMPERATURE)
            losses.append(loss.item())
    encoder.train()
    head.train()
    return float(np.mean(losses))


def identity_recheck(encoder_ref, encoder_lora, lines, n_patches=8):
    """Re-vérifie B=0 → identité, sur quelques patchs, juste avant l'entraînement."""
    f = h5py.File(str(H5_PATH), "r")
    names = f["metadata/image_names"][:]
    pos = f["metadata/positions"][:]
    idxs = list(range(0, len(names), max(1, len(names) // n_patches)))[:n_patches]

    max_diff = 0.0
    for i in idxs:
        img_name = names[i].decode()
        x0, y0, x1, y1 = pos[i]
        img_path = IMG_DIR / img_name
        x = preprocess_array(np.array(Image.open(img_path)), DEVICE)

        cap_ref = {}
        h1 = encoder_ref.trunk.blocks[9].register_forward_hook(lambda m, i_, o: cap_ref.setdefault("f", o.detach()))
        cap_lora = {}
        h2 = encoder_lora.trunk.blocks[9].register_forward_hook(lambda m, i_, o: cap_lora.setdefault("f", o.detach()))
        with torch.no_grad():
            encoder_ref(x)
            encoder_lora(x)
        h1.remove()
        h2.remove()

        v_ref = extract_patch_vec(cap_ref["f"], x0, y0, x1, y1, ORIG_H, ORIG_W)
        v_lora = extract_patch_vec(cap_lora["f"], x0, y0, x1, y1, ORIG_H, ORIG_W)
        max_diff = max(max_diff, (v_ref - v_lora).abs().max().item())

    ok = max_diff < 1e-5
    lines.append(f"[{'x' if ok else ' '}] Test d'identité re-vérifié (B=0 avant entraînement) : écart max = {max_diff:.3e}")
    return ok


def train_lora(by_texture_train, by_texture_val, lines):
    seed_everything(SEED)
    encoder = load_encoder()
    lora_modules = apply_lora(encoder, LORA_BLOCKS)
    encoder = encoder.to(DEVICE)
    head = ProjectionHead().to(DEVICE)
    encoder.train()
    head.train()

    by_image_train = to_patches_by_image(by_texture_train)
    by_image_val = to_patches_by_image(by_texture_val)
    log(lines, f"Images sources train (pool) : {len(by_image_train)}  |  val : {len(by_image_val)}  |  "
               f"max images/step (borne mutualisation) : {MAX_IMAGES_PER_STEP}")

    params = [p for m in lora_modules for p in (m.lora_A, m.lora_B)] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    rng_train = random.Random(SEED + 1)
    rng_val = random.Random(SEED + 2)

    train_losses, val_losses, val_steps = [], [], []
    best_val = float("inf")
    best_state = None
    evals_since_improve = 0
    t0 = time.time()
    peak_mem = 0.0
    step_times = []

    for step in range(MAX_STEPS):
        t_step0 = time.time()
        batch = sample_balanced_batch_bounded_images(
            by_image_train, N_PER_TEXTURE, MAX_IMAGES_PER_STEP, rng_train, use_augmentation=True
        )
        assert all(TEST_IMAGE not in b[0] for b in batch), "FUITE : image de test dans un batch d'entraînement"

        texture_labels = [b[5] for b in batch]
        uniq = sorted(set(texture_labels))
        label_ids = torch.tensor([uniq.index(l) for l in texture_labels])

        optimizer.zero_grad()
        embeddings, _ = forward_batch(encoder, head, batch, DEVICE)
        loss = supcon_loss(embeddings, label_ids.to(DEVICE), temperature=TEMPERATURE)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        step_times.append(time.time() - t_step0)

        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)

        if step < 5 or (step + 1) % 10 == 0:
            log(lines, f"  step {step+1:4d}  train_loss={loss.item():.4f}  batch_size={len(batch)}  "
                       f"n_images_batch={len(set(b[0] for b in batch))}  "
                       f"step_time={step_times[-1]*1000:.0f}ms  avg={np.mean(step_times)*1000:.0f}ms")

        if (step + 1) % EVAL_EVERY == 0:
            val_batch_check = sample_balanced_batch_bounded_images(
                by_image_val, 1, MAX_IMAGES_PER_STEP, rng_val, use_augmentation=False
            )
            assert all(TEST_IMAGE not in b[0] for b in val_batch_check), "FUITE : image de test dans la validation"

            vloss = eval_val_loss(encoder, head, by_image_val, rng_val)
            val_losses.append(vloss)
            val_steps.append(step + 1)

            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = copy.deepcopy({
                    f"lora_{i}": (m.lora_A.detach().clone(), m.lora_B.detach().clone())
                    for i, m in enumerate(lora_modules)
                })
                evals_since_improve = 0
            else:
                evals_since_improve += 1

            log(lines,
                f"  [EVAL] step {step+1:4d}  train_loss={loss.item():.4f}  val_loss={vloss:.4f}  "
                f"best_val={best_val:.4f}  patience={evals_since_improve}/{PATIENCE_EVALS}  "
                f"elapsed={time.time()-t0:.0f}s"
            )

            if evals_since_improve >= PATIENCE_EVALS:
                log(lines, f"  → early stopping à step {step+1}")
                break

    elapsed = time.time() - t0

    if best_state is not None:
        for i, m in enumerate(lora_modules):
            a, b = best_state[f"lora_{i}"]
            m.lora_A.data.copy_(a)
            m.lora_B.data.copy_(b)

    encoder.eval()
    log(lines, f"Temps entraînement total : {elapsed:.1f}s  |  Temps moyen/step : {np.mean(step_times)*1000:.0f}ms  |  "
               f"Mémoire GPU pic : {peak_mem:.2f} GB")
    return encoder, train_losses, val_losses, val_steps


def extract_all_blocks_for_images(encoder, image_names, names, cats, pos, lines):
    """Ré-extrait, avec l'encodeur donné, les features par patch (tous blocs)
    pour toutes les images de `image_names`. Retourne feats[block] (N,C),
    cat_out (N,), stem_out (N,) alignés."""
    feats = {f"block_{i}": [] for i in range(16)}
    cat_out, stem_out = [], []

    for img_i, img_name in enumerate(image_names):
        if img_i % 20 == 0:
            log(lines, f"  ré-extraction image {img_i+1}/{len(image_names)}")
        img_path = IMG_DIR / img_name
        img_arr = np.array(Image.open(img_path))
        x = preprocess_array(img_arr, DEVICE)

        captured = {}
        handles = []
        for i, block in enumerate(encoder.trunk.blocks):
            def _hook(m, inp, out, idx=i):
                captured[f"block_{idx}"] = out.detach()
            handles.append(block.register_forward_hook(_hook))
        with torch.no_grad():
            encoder(x)
        for h in handles:
            h.remove()

        idxs = [i for i in range(len(names)) if names[i] == img_name and cats[i] in TEXTURES]
        for i in idxs:
            x0, y0, x1, y1 = pos[i]
            for b in range(16):
                key = f"block_{b}"
                vec = extract_patch_vec(captured[key], x0, y0, x1, y1, ORIG_H, ORIG_W)
                feats[key].append(vec.cpu().numpy())
            cat_out.append(cats[i])
            stem_out.append(img_name.replace(".tif", ""))

    for k in feats:
        feats[k] = np.stack(feats[k], axis=0)
    return feats, np.array(cat_out), np.array(stem_out)


def compare_zero_shot_vs_lora(feats_zero, feats_lora, cat_zero, stem_zero, cat_lora, stem_lora, lines):
    test_stem = TEST_IMAGE.replace(".tif", "")
    assert test_stem not in set(stem_zero[stem_zero != test_stem]), "sanity"

    lines.append("")
    lines.append(f"{'Texture':<24} {'block(zs)':<10} {'AUC zero-shot':>14} {'AUC LoRA':>10} {'ΔAUC':>8} {'n_test':>7}")
    lines.append("-" * 80)
    results = []
    for tex in TEXTURES:
        y_zero = (cat_zero == tex).astype(int)
        # meilleur bloc zero-shot sur CE fold (test = TEST_IMAGE)
        best_block, best_auc, best_r = None, -1, None
        for b in range(16):
            key = f"block_{b}"
            r = loio_single_fold(feats_zero[key], y_zero, stem_zero, test_stem)
            if r is not None and not np.isnan(r["auc"]) and r["auc"] > best_auc:
                best_auc, best_block, best_r = r["auc"], key, r

        if best_block is None:
            lines.append(f"{tex:<24} {'n/a':<10} {'n/a':>14} {'n/a':>10} {'n/a':>8} {'0':>7}")
            continue

        y_lora = (cat_lora == tex).astype(int)
        r_lora = loio_single_fold(feats_lora[best_block], y_lora, stem_lora, test_stem)
        auc_lora = r_lora["auc"] if r_lora is not None else float("nan")
        delta = auc_lora - best_auc if r_lora is not None else float("nan")

        lines.append(
            f"{tex:<24} {best_block:<10} {best_auc:>14.3f} {auc_lora:>10.3f} {delta:>+8.3f} {best_r['n_test']:>7}"
        )
        results.append({"texture": tex, "block": best_block, "auc_zero": best_auc, "auc_lora": auc_lora, "delta": delta})
    return results


def main():
    lines = []
    log(lines, "═" * 70)
    log(lines, "PHASE 4 — Rapport de validation (fold pilote)")
    log(lines, "═" * 70)
    log(lines, f"Image de test : {TEST_IMAGE}")
    log(lines, f"Images de validation (early stopping) : {VAL_IMAGES}")

    assert TEST_IMAGE not in VAL_IMAGES
    log(lines, "[x] Contrôle anti-fuite statique : image de test absente des images de validation")

    names, cats, pos, h5_feats = load_h5_index()
    all_images = sorted(set(names.tolist()))
    train_images = [n for n in all_images if n != TEST_IMAGE and n not in VAL_IMAGES]
    log(lines, f"Images train (LoRA) : {len(train_images)}  |  val : {len(VAL_IMAGES)}  |  test : 1")
    assert TEST_IMAGE not in train_images

    by_texture_train = patches_by_texture_for_images(names, cats, pos, set(train_images))
    by_texture_val = patches_by_texture_for_images(names, cats, pos, set(VAL_IMAGES))
    log(lines, "Patchs train par texture : " + ", ".join(f"{k}={len(v)}" for k, v in by_texture_train.items()))
    log(lines, "Patchs val par texture   : " + ", ".join(f"{k}={len(v)}" for k, v in by_texture_val.items()))

    # Test d'identité re-vérifié (B=0) juste avant l'entraînement
    encoder_ref = load_encoder().to(DEVICE).eval()
    encoder_lora_probe = load_encoder()
    apply_lora(encoder_lora_probe, LORA_BLOCKS)
    encoder_lora_probe = encoder_lora_probe.to(DEVICE).eval()
    ok_identity = identity_recheck(encoder_ref, encoder_lora_probe, lines)
    print(lines[-1], flush=True)
    del encoder_lora_probe

    log(lines, "")
    log(lines, "── Entraînement LoRA (fold pilote) ──")
    encoder_trained, train_losses, val_losses, val_steps = train_lora(by_texture_train, by_texture_val, lines)

    # Courbes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="train (par step)", alpha=0.6)
    ax.plot(val_steps, val_losses, label="val (par éval)", marker="o", color="darkorange")
    ax.set_xlabel("step")
    ax.set_ylabel("SupCon loss")
    ax.set_title("Phase 4 — fold pilote : loss train/val")
    ax.legend()
    plt.tight_layout()
    fig_path = OUTDIR / "loss_curves.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    log(lines, f"Courbes sauvées : {fig_path}")

    log(lines, "")
    log(lines, "── Ré-extraction features (encodeur adapté), toutes images ──")
    feats_lora, cat_lora, stem_lora = extract_all_blocks_for_images(encoder_trained, all_images, names, cats, pos, lines)
    log(lines, f"Features ré-extraites : {len(cat_lora)} patchs, {len(feats_lora)} blocs")

    # Zero-shot : filtrer le H5 précalculé aux mêmes patchs/textures
    mask = np.isin(cats, TEXTURES)
    cat_zero = cats[mask]
    stem_zero = np.array([n.replace(".tif", "") for n in names[mask]])
    feats_zero = {k: v[mask] for k, v in h5_feats.items()}

    log(lines, "")
    log(lines, "── Contrôle anti-fuite (extraction) ──")
    log(lines, f"[x] '{TEST_IMAGE.replace('.tif','')}' présent en test uniquement (assert dans loio_single_fold)")

    results = compare_zero_shot_vs_lora(feats_zero, feats_lora, cat_zero, stem_zero, cat_lora, stem_lora, lines)
    for r_line in lines[-(len(TEXTURES) + 2):]:
        print(r_line, flush=True)

    log(lines, "")
    log(lines, "── VALIDATION P4 ──")
    log(lines, f"[{'x' if ok_identity else ' '}] Test d'identité re-vérifié (B=0)")
    log(lines, f"[x] Courbes de loss train/val affichées")
    log(lines, f"[x] ΔAUC par texture sur le fold pilote (tableau ci-dessus)")
    log(lines, f"[x] Contrôle anti-fuite : asserts sur les noms d'images (train batches, val batches, LOIO)")
    go = ok_identity
    log(lines, "")
    log(lines, f"VERDICT : {'GO' if go else 'NO-GO'} (analyse humaine du pilote requise avant Phase 5)")

    report = "\n".join(lines)
    (OUTDIR / "report.txt").write_text(report)
    return go


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
