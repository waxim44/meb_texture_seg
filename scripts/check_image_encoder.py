"""
Vérifie la composition de l'image encoder TextureSAM (version tiny).

Usage:
    python scripts/check_image_encoder.py
"""

import sys
from pathlib import Path

import torch

ROOT   = Path(__file__).resolve().parents[1]
SAM2   = ROOT / "TextureSAM" / "sam2"
sys.path.insert(0, str(SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Couleurs terminal ──────────────────────────────────────────────────────────
OK  = lambda s: f"\033[32m✅ {s}\033[0m"
ERR = lambda s: f"\033[31m❌ {s}\033[0m"
HDR = lambda s: f"\n\033[1;34m{'─'*60}\n  {s}\n{'─'*60}\033[0m"


# ── Construction du modèle (identique à feature_extractor.py) ─────────────────

def build_encoder():
    trunk = Hiera(
        embed_dim=96,
        num_heads=1,
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


# ── Checks ────────────────────────────────────────────────────────────────────

def check_version(encoder):
    """Vérifie qu'on utilise bien la version Tiny (et pas Large)."""
    print(HDR("1. Version du modèle — Tiny vs Large"))

    checks = [
        ("embed_dim initial",  encoder.trunk.patch_embed.proj.out_channels, 96,   144),
        ("num_heads initial",  encoder.trunk.blocks[0].attn.num_heads,      1,    2),
        ("total blocs trunk",  len(encoder.trunk.blocks),                   16,   48),
        ("scalp",              encoder.scalp,                               1,    1),
    ]

    for label, actual, expected_tiny, expected_large in checks:
        is_tiny = (actual == expected_tiny)
        status  = OK(f"{label} = {actual}  (tiny={expected_tiny}, large={expected_large})") \
                  if is_tiny else \
                  ERR(f"{label} = {actual}  attendu tiny={expected_tiny}")
        print(f"  {status}")


def check_trunk_stages(encoder):
    """Vérifie la structure des 4 stages du Hiera trunk."""
    print(HDR("2. Hiera Trunk — 4 stages"))

    stage_ends = encoder.trunk.stage_ends   # [0, 2, 13, 15]
    channel_list = encoder.trunk.channel_list  # dims par stage (ordre inverse)

    # Calcul des blocs par stage
    prev = -1
    stage_info = []
    for i, end in enumerate(stage_ends):
        n_blocks = end - prev
        dim = encoder.trunk.blocks[end].dim_out
        stage_info.append((i + 1, prev + 1, end, n_blocks, dim))
        prev = end

    expected_dims = [96, 192, 384, 768]
    all_ok = True
    for stage_num, start, end, n_blk, dim in stage_info:
        exp = expected_dims[stage_num - 1]
        ok_flag = (dim == exp)
        if not ok_flag:
            all_ok = False
        badge = OK("") if ok_flag else ERR("")
        print(f"  Stage {stage_num} : blocks.{start}–{end}  ({n_blk} blocs)  "
              f"dim_out={dim}  (attendu {exp})  {'✅' if ok_flag else '❌'}")

    print()
    q_pool_blocks = encoder.trunk.q_pool_blocks
    print(f"  Q-pool aux blocs : {q_pool_blocks}  "
          f"(downsample ×2 entre stages)")
    global_att = encoder.trunk.global_att_blocks
    print(f"  Global attention aux blocs : {global_att}")


def check_neck(encoder):
    """Vérifie la FPN neck : convolutions, d_model, fpn_top_down_levels."""
    print(HDR("3. FPN Neck — projections et fusion top-down"))

    expected_in = [768, 384, 192, 96]
    expected_out = 256

    all_ok = True
    for i, conv_seq in enumerate(encoder.neck.convs):
        conv   = conv_seq[0]
        in_ch  = conv.weight.shape[1]
        out_ch = conv.weight.shape[0]
        exp_in = expected_in[i]
        ok_flag = (in_ch == exp_in and out_ch == expected_out)
        if not ok_flag:
            all_ok = False
        stage = 4 - i
        print(f"  neck.convs.{i} → Stage {stage} :  "
              f"Conv2d({in_ch}, {out_ch}, 1×1)  "
              f"{'✅' if ok_flag else '❌'}")

    print()
    print(f"  fpn_top_down_levels : {encoder.neck.fpn_top_down_levels}"
          f"  (levels 0 et 1 = features brutes)")
    print(f"  fpn_interp_model    : {encoder.neck.fpn_interp_model}")
    print(f"  d_model             : {encoder.neck.d_model}")


def check_forward(encoder):
    """Forward pass sur une image fictive 1024×1024, vérifie les shapes de sortie."""
    print(HDR("4. Forward pass — shapes de sortie"))

    x = torch.zeros(1, 3, 1024, 1024)

    # Hooks sur neck.convs pour capturer les features par stage
    stage_shapes = {}
    handles = []
    conv_to_stage = {0: "stage_4", 1: "stage_3", 2: "stage_2", 3: "stage_1"}
    for idx, name in conv_to_stage.items():
        def _hook(m, inp, out, _n=name):
            stage_shapes[_n] = tuple(out.shape)
        handles.append(encoder.neck.convs[idx].register_forward_hook(_hook))

    with torch.no_grad():
        out = encoder(x)

    for h in handles:
        h.remove()

    # Shapes attendues pour 1024×1024 en entrée
    expected_shapes = {
        "stage_1": (1, 256, 256, 256),
        "stage_2": (1, 256, 128, 128),
        "stage_3": (1, 256,  64,  64),
        "stage_4": (1, 256,  32,  32),
    }

    print(f"  Entrée : {tuple(x.shape)}\n")
    for stage in ["stage_1", "stage_2", "stage_3", "stage_4"]:
        actual = stage_shapes[stage]
        exp    = expected_shapes[stage]
        ok_flag = (actual == exp)
        res = 1024 // actual[2]
        print(f"  {stage} : {actual}  "
              f"({res}px/vecteur)  {'✅' if ok_flag else '❌'}")

    print()
    print(f"  vision_features (après scalp=1) : {tuple(out['vision_features'].shape)}")
    print(f"  backbone_fpn    : {len(out['backbone_fpn'])} niveaux")
    for i, f in enumerate(out["backbone_fpn"]):
        print(f"    [{i}] {tuple(f.shape)}")


def check_params(encoder):
    """Compte le nombre de paramètres."""
    print(HDR("5. Nombre de paramètres"))

    total  = sum(p.numel() for p in encoder.parameters())
    trunk  = sum(p.numel() for p in encoder.trunk.parameters())
    neck   = sum(p.numel() for p in encoder.neck.parameters())

    print(f"  Trunk  (Hiera) : {trunk/1e6:.1f} M")
    print(f"  Neck   (FPN)   : {neck/1e6:.1f} M")
    print(f"  Total          : {total/1e6:.1f} M  "
          f"(Large ≈ 224 M → on est bien en Tiny)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*60}")
    print("  Image Encoder TextureSAM — vérification complète")
    print(f"{'═'*60}")
    print(f"  Fichier source : src/encoder/feature_extractor.py")
    print(f"  Architecture   : Hiera (trunk) + FPN (neck)")

    encoder = build_encoder()
    encoder.eval()

    check_version(encoder)
    check_trunk_stages(encoder)
    check_neck(encoder)
    check_forward(encoder)
    check_params(encoder)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
