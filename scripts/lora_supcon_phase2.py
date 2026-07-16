"""
lora_supcon_phase2.py
══════════════════════════════════════════════════════════════════════════════
PHASE 2 — Validation de l'insertion LoRA dans l'encodeur.

Tests :
  1. TEST D'IDENTITÉ : avec B=0, features de patch (LoRA-wrapped) IDENTIQUES
     au zero-shot (écart max < 1e-5) sur ≥10 patchs, tous les blocs du
     trunk (0-15).
  2. GRADIENTS : après un backward factice, seuls lora_A et lora_B ont
     un .grad non-nul.
  3. % de paramètres entraînables (attendu < 1%).
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import h5py
import numpy as np
import torch
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SAM2 = _ROOT / "TextureSAM" / "sam2"
sys.path.insert(0, str(_SAM2))
sys.path.insert(0, str(_ROOT / "lora_supcon"))

from sam2.modeling.backbones.hieradet import Hiera  # noqa: E402
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck  # noqa: E402
from sam2.modeling.position_encoding import PositionEmbeddingSine  # noqa: E402
from lora import apply_lora, count_trainable_params, LORA_BLOCKS  # noqa: E402

CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
H5_PATH = _ROOT / "data" / "feature_database" / "database_meb_ouassim.h5"
IMG_DIR = _ROOT / "Image_Ouassim"
OUTDIR = _ROOT / "lora_supcon" / "phase_2"
OUTDIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 1024
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TEST_PATCHES = 15
TOL = 1e-5


def build_image_encoder() -> ImageEncoder:
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


def load_encoder() -> ImageEncoder:
    encoder = build_image_encoder()
    sd = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    prefix = "image_encoder."
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    assert len(missing) == 0, f"Missing keys: {missing}"
    return encoder.to(DEVICE).eval()


def preprocess(img_path) -> torch.Tensor:
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(DEVICE)


def register_block_hooks(trunk):
    captured = {}
    handles = []
    for i, block in enumerate(trunk.blocks):
        def _hook(m, inp, out, idx=i):
            captured[f"block_{idx}"] = out.detach()
        handles.append(block.register_forward_hook(_hook))
    return captured, handles


def extract_patch_vec(captured, key, x_min, y_min, x_max, y_max, orig_H, orig_W):
    feat = captured[key][0]  # (H_feat, W_feat, C)
    H_feat, W_feat, C = feat.shape
    scale_x = W_feat / orig_W
    scale_y = H_feat / orig_H
    fx1 = max(0, int(x_min * scale_x))
    fy1 = max(0, int(y_min * scale_y))
    fx2 = min(W_feat, max(fx1 + 1, int(x_max * scale_x)))
    fy2 = min(H_feat, max(fy1 + 1, int(y_max * scale_y)))
    region = feat[fy1:fy2, fx1:fx2, :]
    return region.mean(dim=(0, 1))


def get_test_patches():
    f = h5py.File(str(H5_PATH), "r")
    names = f["metadata/image_names"][:]
    pos = f["metadata/positions"][:]
    picked = []
    seen_images = []
    idx = 0
    step = max(1, len(names) // (N_TEST_PATCHES * 3))
    while len(picked) < N_TEST_PATCHES and idx < len(names):
        img_name = names[idx].decode()
        x0, y0, x1, y1 = pos[idx]
        picked.append((img_name, float(x0), float(y0), float(x1), float(y1)))
        seen_images.append(img_name)
        idx += step
    return picked


def identity_test(lines):
    lines.append("── 1. TEST D'IDENTITÉ (B=0) ──")

    encoder_ref = load_encoder()
    encoder_lora = load_encoder()
    lora_modules = apply_lora(encoder_lora, LORA_BLOCKS)
    encoder_lora.to(DEVICE)
    for m in lora_modules:
        assert torch.all(m.lora_B == 0), "lora_B doit être initialisé à zéro"
    encoder_lora.eval()

    patches = get_test_patches()
    lines.append(f"Patchs testés : {len(patches)}")

    max_diff_overall = 0.0
    per_patch_max = []

    for img_name, x0, y0, x1, y1 in patches:
        img_path = IMG_DIR / img_name
        x = preprocess(img_path)
        orig_H, orig_W = 768, 1280

        cap_ref, h_ref = register_block_hooks(encoder_ref.trunk)
        with torch.no_grad():
            encoder_ref(x)
        remove(h_ref)

        cap_lora, h_lora = register_block_hooks(encoder_lora.trunk)
        with torch.no_grad():
            encoder_lora(x)
        remove(h_lora)

        patch_max = 0.0
        for i in range(16):
            key = f"block_{i}"
            v_ref = extract_patch_vec(cap_ref, key, x0, y0, x1, y1, orig_H, orig_W)
            v_lora = extract_patch_vec(cap_lora, key, x0, y0, x1, y1, orig_H, orig_W)
            diff = (v_ref - v_lora).abs().max().item()
            patch_max = max(patch_max, diff)
        per_patch_max.append((img_name, patch_max))
        max_diff_overall = max(max_diff_overall, patch_max)

    lines.append(f"Écart max sur tous les blocs (0-15), tous les patchs : {max_diff_overall:.3e}")
    for img_name, d in per_patch_max:
        lines.append(f"  {img_name} : écart max = {d:.3e}")

    ok = max_diff_overall < TOL
    lines.append(f"[{'x' if ok else ' '}] TEST D'IDENTITÉ (écart < {TOL})")
    return ok, encoder_lora, lora_modules


def remove(handles):
    for h in handles:
        h.remove()


def gradient_test(lines, encoder_lora, lora_modules):
    lines.append("")
    lines.append("── 2. TEST GRADIENTS ──")
    encoder_lora.train()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, requires_grad=False)
    out = encoder_lora(x)
    loss = out["vision_features"].sum() if isinstance(out, dict) else out[0].sum()
    loss.backward()

    lora_A_ids = {id(m.lora_A) for m in lora_modules}
    lora_B_ids = {id(m.lora_B) for m in lora_modules}
    lora_param_ids = lora_A_ids | lora_B_ids

    n_B_with_grad = 0
    n_other_with_grad = 0
    for name, p in encoder_lora.named_parameters():
        is_lora = id(p) in lora_param_ids
        has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
        if is_lora:
            if id(p) in lora_B_ids and has_grad:
                n_B_with_grad += 1
        else:
            if has_grad:
                n_other_with_grad += 1
                lines.append(f"  ANOMALIE : {name} a un gradient non-nul (devrait être gelé)")

    lines.append(f"Params lora_B avec gradient non-nul : {n_B_with_grad}/{len(lora_B_ids)}")
    lines.append(
        "Note : lora_A a un gradient nul à l'init car B=0 (dL/dA passe par B "
        "dans la règle de chaîne) — c'est le comportement LoRA attendu, pas une anomalie."
    )
    lines.append(f"Params gelés (hors LoRA) avec gradient non-nul (attendu 0) : {n_other_with_grad}")
    ok = (n_B_with_grad == len(lora_B_ids)) and (n_other_with_grad == 0)
    lines.append(f"[{'x' if ok else ' '}] Seuls les params LoRA reçoivent un gradient (B non-nul, reste gelé)")
    encoder_lora.zero_grad()
    encoder_lora.eval()
    return ok


def param_count_test(lines, encoder_lora):
    lines.append("")
    lines.append("── 3. PARAMÈTRES ENTRAÎNABLES ──")
    trainable, total = count_trainable_params(encoder_lora)
    pct = 100 * trainable / total
    lines.append(f"Entraînables : {trainable:,} / {total:,} ({pct:.4f}%)")
    ok = pct < 1.0
    lines.append(f"[{'x' if ok else ' '}] < 1% de paramètres entraînables")
    return ok


def main():
    lines = ["═" * 70, "PHASE 2 — Rapport de validation", "═" * 70]

    ok_identity, encoder_lora, lora_modules = identity_test(lines)
    ok_grad = gradient_test(lines, encoder_lora, lora_modules)
    ok_params = param_count_test(lines, encoder_lora)

    lines.append("")
    lines.append("── VALIDATION P2 ──")
    lines.append(f"[{'x' if ok_identity else ' '}] TEST D'IDENTITÉ (B=0, écart < {TOL})")
    lines.append(f"[{'x' if ok_grad else ' '}] Gradients : seuls A, B non-nuls")
    lines.append(f"[{'x' if ok_params else ' '}] % params entraînables < 1%")
    go = ok_identity and ok_grad and ok_params
    lines.append("")
    lines.append(f"VERDICT : {'GO' if go else 'NO-GO'}")

    report = "\n".join(lines)
    print(report)
    (OUTDIR / "report.txt").write_text(report)
    return go


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
