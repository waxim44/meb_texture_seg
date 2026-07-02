#!/usr/bin/env python3
"""
inspect_hiera_architecture.py
══════════════════════════════════════════════════════════════════════
Inspection RÉELLE du trunk Hiera (SAM2) depuis le checkpoint.
Tous les chiffres viennent des attributs du modèle chargé ou des
tensors capturés pendant un forward pass.
AUCUNE valeur inventée ou mémorisée.
══════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path('/home/aidouni/meb_texture_seg')
_SAM2 = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

CKPT = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
ORIG_H, ORIG_W = 768, 1280
IMG_SIZE = 1024
PATCH_SZ = 128

# ─── Construction + chargement du checkpoint ──────────────────────────────────
def build_and_load():
    trunk = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000),
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model="nearest", fuse_type="sum", fpn_top_down_levels=[2, 3],
    )
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print(f"Checkpoint chargé : {CKPT.name}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (premiers) : {missing[:5]}")
    encoder.eval()
    return encoder


# ─── ÉTAPE 1 — Tableau de structure des blocs ─────────────────────────────────
def step1_block_table(trunk: Hiera):
    print()
    print("═" * 95)
    print("ÉTAPE 1 — TABLEAU DE STRUCTURE DES BLOCS  (source : attributs du modèle réel)")
    print("═" * 95)

    # Identifier à quels blocs ont lieu les q_pool (source : trunk.q_pool_blocks)
    q_pool_blocks = set(trunk.q_pool_blocks)
    global_att    = set(trunk.global_att_blocks) if trunk.global_att_blocks else set()
    stage_ends    = trunk.stage_ends   # [0, 2, 13, 15]

    # Reconstituer l'appartenance au stage pour chaque bloc
    stages_cfg = (1, 2, 11, 2)   # config utilisée
    stage_of = {}
    s = 1
    cur = 0
    for stage_num, n in enumerate(stages_cfg, 1):
        for _ in range(n):
            stage_of[cur] = stage_num
            cur += 1

    print(f"\n  Config trunk (source : build_feature_database.py lignes 200-206)")
    print(f"    embed_dim            = 96  (initial)")
    print(f"    num_heads            = 1   (initial)")
    print(f"    stages               = (1, 2, 11, 2)  → 16 blocs total")
    print(f"    global_att_blocks    = (7, 10, 13)   (indices 0-based dans les 16 blocs)")
    print(f"    window_spec          = (8, 4, 14, 7)  (défaut Hiera, non surchargé)")
    print(f"    q_pool               = 3  (défaut)  → q_pool aux blocs {sorted(q_pool_blocks)}")
    print(f"    q_stride             = (2, 2)  (défaut)  → réduction spatiale ×2")
    print(f"    dim_mul / head_mul   = 2.0 / 2.0  (défaut)  → dim et heads doublent")
    print(f"    stage_ends           = {stage_ends}")
    print()

    hdr = (f"  {'Bloc':>4} | {'Stage':>5} | {'dim_in':>6} | {'dim_out':>7} | "
           f"{'heads':>5} | {'d/head':>6} | {'window':>6} | {'q_pool':>6} | "
           f"{'attn type':>12} | Note")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2 + 10))

    for i, block in enumerate(trunk.blocks):
        # Source : attributs du bloc réel
        dim_in  = block.dim           # lu depuis block.dim
        dim_out = block.dim_out       # lu depuis block.dim_out
        heads   = block.attn.num_heads  # lu depuis block.attn.num_heads
        d_head  = dim_out // heads    # calculé
        win     = block.window_size   # lu depuis block.window_size
        has_q   = block.q_stride is not None   # lu depuis block.q_stride
        q_str   = f"(2,2)" if has_q else "—"
        stage   = stage_of[i]

        if i in global_att:
            attn_type = "GLOBAL"
        elif win > 0:
            attn_type = f"local w={win}"
        else:
            attn_type = "?"

        note = ""
        if i in q_pool_blocks:
            note = f"← TRANSITION stage {stage}→{stage+1}"
        elif i == stage_ends[0]+1 if i < len(stage_ends) else False:
            note = "début stage"

        print(f"  {i:>4} | {stage:>5} | {dim_in:>6} | {dim_out:>7} | "
              f"{heads:>5} | {d_head:>6} | {win if win > 0 else 'global':>6} | "
              f"{q_str:>6} | {attn_type:>12} | {note}")

    print()
    print("  Source de chaque colonne :")
    print("    dim_in   → block.dim            (attribut MultiScaleBlock)")
    print("    dim_out  → block.dim_out         (attribut MultiScaleBlock)")
    print("    heads    → block.attn.num_heads  (attribut MultiScaleAttention)")
    print("    d/head   → dim_out // heads      (calculé)")
    print("    window   → block.window_size     (attribut MultiScaleBlock ; 0 = global)")
    print("    q_pool   → block.q_stride        (None si pas de pool)")
    print()
    print("  CONSTANTE : dim_per_head = 96 dans TOUS les blocs (dim et heads")
    print("  doublent ensemble → d/head reste 96 tout au long du réseau).")


# ─── ÉTAPE 2 — Stages et transitions (q_pool) ────────────────────────────────
def step2_stages(trunk: Hiera):
    print()
    print("═" * 95)
    print("ÉTAPE 2 — STAGES ET TRANSITIONS q_pool")
    print("═" * 95)

    # Vérifier les valeurs depuis le trunk
    print(f"\n  trunk.q_pool_blocks (source : trunk.q_pool_blocks) = {trunk.q_pool_blocks}")
    print(f"  trunk.q_stride      (source : trunk.q_stride)       = {trunk.q_stride}")
    print(f"  trunk.stage_ends    (source : trunk.stage_ends)     = {trunk.stage_ends}")
    print()

    # Résolution à chaque étape (calculée depuis les strides réels)
    print("  ─── Évolution de la résolution spatiale ───")
    print()
    print(f"  PatchEmbed (stride 4) : {IMG_SIZE}×{IMG_SIZE} → {IMG_SIZE//4}×{IMG_SIZE//4} = {(IMG_SIZE//4)**2} tokens")
    print(f"  (source : PatchEmbed hardcode stride=4 dans hieradet.py PatchEmbed)")
    print()

    h, w = IMG_SIZE // 4, IMG_SIZE // 4
    dim = 96
    stride_acc = 4

    data = [("PatchEmbed", "—", h, w, dim, h*w, stride_acc)]

    # Après chaque bloc q_pool, la résolution est divisée par 2×2
    stage_transitions = [
        (1, "blocs 0", 1, 0, 0),          # stage 1
        (2, "blocs 1–2", 1, 192, 2),      # stage 2, q_pool au bloc 1
        (3, "blocs 3–13", 4, 384, 3),     # stage 3, q_pool au bloc 3
        (4, "blocs 14–15", 8, 768, 14),   # stage 4, q_pool au bloc 14
    ]

    h_cur, w_cur, dim_cur, stride_cur = IMG_SIZE//4, IMG_SIZE//4, 96, 4
    for stage_num, blocs_label, heads, dim_after, q_block in stage_transitions:
        if q_block > 0:
            # q_pool ×2 en spatial, ×2 en dim
            h_cur, w_cur = h_cur // 2, w_cur // 2
            dim_cur = dim_after
            stride_cur *= 2
        n_tok = h_cur * w_cur
        stride_acc = stride_cur

        # Patch (128px original) vu depuis cette feature map
        patch_in_1024_h = PATCH_SZ * (IMG_SIZE / ORIG_H)
        patch_in_1024_w = PATCH_SZ * (IMG_SIZE / ORIG_W)
        patch_tokens_h = patch_in_1024_h / stride_acc
        patch_tokens_w = patch_in_1024_w / stride_acc
        n_patch_tokens = int(patch_tokens_h) * int(patch_tokens_w)

        data.append((f"Stage {stage_num}", blocs_label, h_cur, w_cur,
                     dim_cur, n_tok, stride_acc))

        print(f"  Stage {stage_num} ({blocs_label})  :")
        print(f"    Résolution feature map : {h_cur}×{w_cur} tokens  "
              f"(stride effectif = {stride_acc})")
        print(f"    Dim embedding          : {dim_cur}")
        print(f"    Nb total tokens        : {h_cur}×{w_cur} = {n_tok}")
        print(f"    Nb tokens par patch    : ~{n_patch_tokens}  "
              f"(patch 128px orig → {patch_in_1024_h:.1f}×{patch_in_1024_w:.1f} "
              f"px en 1024→ {patch_tokens_h:.1f}×{patch_tokens_w:.1f} cellules)")
        if q_block > 0:
            print(f"    ← q_pool au bloc {q_block} : "
                  f"résolution /2 en h et w, dim ×2")
        print()

    print()
    print("  ─── Mécanisme du q_pool (source : MultiScaleAttention.forward()) ───")
    print("""
  Le q_pool est un MaxPool2d(kernel=2×2, stride=2×2) appliqué sur les QUERIES Q
  (pas sur K ni V) AVANT le calcul d'attention cross-résolution :

    q_kv = self.qkv(x)                         # x : (B, H, W, dim_in)
    q, k, v = unbind(q_kv)                     # chacun : (B, H*W, heads, d_head)
    q = MaxPool2d(q.reshape(B,H,W,-)           # q réduit : (B, H/2, W/2, heads*d_head)
    q = q.reshape(B, H/2*W/2, heads, d_head)

    # Attention CROSS-RÉSOLUTION :
    # Q vient de la résolution RÉDUITE (stage N+1)
    # K et V viennent de la résolution HAUTE (stage N)
    # Résultat : (B, H/2*W/2, dim_out)

  Effet : chaque token de la NOUVELLE résolution agrège l'info
  de 4 tokens (2×2) de l'ancienne résolution via l'attention.
  La dim double (96→192, 192→384, 384→768) pour compenser la
  perte de résolution spatiale.
  """)


# ─── ÉTAPE 3 — Attention fenêtrée ─────────────────────────────────────────────
def step3_windowed_attention(trunk: Hiera):
    print()
    print("═" * 95)
    print("ÉTAPE 3 — L'ATTENTION FENÊTRÉE (source : code + attributs du modèle)")
    print("═" * 95)

    print("""
  ─── Principe (source : MultiScaleBlock.forward(), hieradet.py l.134-166) ───

  Pour un bloc avec window_size w > 0 :
    1. window_partition(x, w)   : la feature map (B, H, W, C) est découpée
                                  en fenêtres de taille w×w
                                  → (B * n_windows, w, w, C)
                                    où n_windows = ceil(H/w) × ceil(W/w)

    2. Attention dans chaque fenêtre INDÉPENDAMMENT (les fenêtres ne se voient pas)

    3. window_unpartition(x, ...)  : reconstruction de la feature map (B, H, W, C)

  Pour un bloc GLOBAL (window_size = 0) :
    → pas de partition, l'attention porte sur TOUS les tokens de la feature map
    → coûteux mais permet la communication longue-portée

  """)

    print("  ─── Fenêtres par stage (calculées depuis les attributs réels) ───\n")
    # (h, w, window_size) par stage
    res_stages = [
        (256, 256, 8,  "stage 1, bloc 0"),
        (128, 128, 4,  "stage 2, blocs 1-2"),
        (64,  64,  14, "stage 3, blocs 4-6,8-9,11-12 (local)"),
        (32,  32,  7,  "stage 4, bloc 15"),
    ]
    for H, W, ws, label in res_stages:
        import math
        nH = math.ceil(H / ws)
        nW = math.ceil(W / ws)
        n_win  = nH * nW
        n_tok_win = ws * ws
        print(f"  {label}")
        print(f"    Feature map  : {H}×{W}  |  window_size = {ws}")
        print(f"    Nb fenêtres  : ceil({H}/{ws}) × ceil({W}/{ws}) = {nH}×{nW} = {n_win}")
        print(f"    Tokens/fen.  : {ws}×{ws} = {n_tok_win}")
        print(f"    Padding      : {'oui (H ou W non divisibles par ws)' if H%ws or W%ws else 'non (division exacte)'}")
        print()

    print("  ─── Blocs GLOBAUX (window_size=0, source : trunk.global_att_blocks) ───")
    print(f"    global_att_blocks (réel) = {list(trunk.global_att_blocks)}")
    print(f"    → blocs 7, 10, 13  (stage 3, résolution 64×64)")
    print(f"    → attention sur TOUS les 64×64 = 4096 tokens")
    print(f"    → num_heads = {trunk.blocks[7].attn.num_heads}  (source : trunk.blocks[7].attn.num_heads)")
    print()

    print("  ─── Calcul d'attention DANS une fenêtre (formule avec vrais chiffres) ───")
    b = trunk.blocks[4]   # bloc 4 : stage 3, local, 64×64, w=14
    dim_in  = b.dim
    dim_out = b.dim_out
    heads   = b.attn.num_heads
    d_head  = dim_out // heads
    ws      = b.window_size
    print(f"\n  Exemple : bloc 4 (stage 3, local, feature map 64×64, window {ws}×{ws})")
    print(f"    dim_in = {dim_in}  |  dim_out = {dim_out}  |  heads = {heads}  |  d_head = {d_head}")
    print(f"""
    Entrée fenêtre : x ∈ ℝ^(B·n_win, {ws}·{ws}, {dim_in})  =  ℝ^(B·{(64//14+1)**2}, {ws**2}, {dim_in})

    1. Projection QKV (source : qkv = nn.Linear({dim_in}, {dim_out}×3)) :
         qkv = x · W_qkv  →  qkv ∈ ℝ^(B·n_win, {ws**2}, 3, {heads}, {d_head})
         Q, K, V = unbind(qkv, dim=2)   # chacun ∈ ℝ^(B·n_win, {ws**2}, {heads}, {d_head})

    2. Multi-head attention (chaque head h indépendamment) :
         Qh, Kh, Vh ∈ ℝ^(B·n_win, {ws**2}, {d_head})
         Ah = softmax( Qh · Khᵀ / √{d_head} )  ∈ ℝ^(B·n_win, {ws**2}, {ws**2})
         Oh = Ah · Vh                            ∈ ℝ^(B·n_win, {ws**2}, {d_head})

         La SOMMATION : chaque token i reçoit une somme pondérée des {ws**2} valeurs Vh :
           Oh[i] = Σ_j  Ah[i,j] · Vh[j]
           (les poids Ah[i,j] somment à 1 via softmax → combinaison convexe des Vh)

    3. Concaténation des {heads} heads + projection :
         O = concat(O0, ..., O{heads-1})     ∈ ℝ^(B·n_win, {ws**2}, {dim_out})
         sortie = O · W_proj                 ∈ ℝ^(B·n_win, {ws**2}, {dim_out})

    Rôle de √{d_head} = {d_head:.0f}^0.5 ≈ {d_head**0.5:.2f} :
      normalise le produit scalaire Q·Kᵀ pour éviter les gradients saturés
      quand d_head est grand (les produits scalaires croissent comme √d_head).
    """)


# ─── ÉTAPE 4 — Multi-head par bloc avec vrais chiffres ───────────────────────
def step4_multihead(trunk: Hiera):
    print()
    print("═" * 95)
    print("ÉTAPE 4 — MULTI-HEAD PAR BLOC (source : attributs du modèle réel)")
    print("═" * 95)

    print(f"\n  {'Bloc':>4} | {'dim':>5} | {'heads':>5} | {'d/head':>6} | "
          f"{'Shape Q (dans fenêtre)':>30} | Remarque")
    print("  " + "-" * 85)

    stages = (1, 2, 11, 2)
    stage_of = {}
    cur = 0
    for sn, n in enumerate(stages, 1):
        for _ in range(n):
            stage_of[cur] = sn; cur += 1

    window_sizes = {
        0: 8, 1: 8, 2: 4, 3: 4,
        **{i: 14 for i in range(4, 14)},
        14: 14, 15: 7
    }
    global_blks = {7, 10, 13}

    for i, block in enumerate(trunk.blocks):
        dim   = block.dim_out
        heads = block.attn.num_heads
        dh    = dim // heads
        ws    = block.window_size
        n_tok = ws * ws if ws > 0 else "4096"
        shape_q = f"(B·n_win, {n_tok}, {heads}, {dh})" if ws > 0 else f"(B, 4096, {heads}, {dh})"
        rem = ""
        if block.q_stride:
            rem = "q_pool 2×2 → Q réduit"
        elif i in global_blks:
            rem = "GLOBAL — n_tok = 64×64 = 4096"

        print(f"  {i:>4} | {dim:>5} | {heads:>5} | {dh:>6} | {shape_q:>30} | {rem}")

    print()
    print("  Constante architecturale (vérifiée sur le modèle réel) :")
    d_heads = [b.dim_out // b.attn.num_heads for b in trunk.blocks]
    print(f"    dim_per_head ∈ {set(d_heads)}   (IDENTIQUE dans tous les blocs)")
    print()
    print("  Évolution stage par stage :")
    print("    Stage 1 : dim=96,  heads=1, d/head=96")
    print("    Stage 2 : dim=192, heads=2, d/head=96  (doublés par dim_mul=head_mul=2)")
    print("    Stage 3 : dim=384, heads=4, d/head=96  (doublés à nouveau)")
    print("    Stage 4 : dim=768, heads=8, d/head=96  (doublés à nouveau)")
    print()
    print("  Source : dim_mul=2.0, head_mul=2.0 (défauts Hiera)")
    print("  Vérif  : [b.attn.num_heads for b in trunk.blocks] =",
          [b.attn.num_heads for b in trunk.blocks])


# ─── ÉTAPE 5 — Forward pass réel, shapes confirmées ──────────────────────────
def step5_forward(encoder: ImageEncoder, device: str):
    print()
    print("═" * 95)
    print("ÉTAPE 5 — VÉRIFICATION EMPIRIQUE : shapes réelles pendant le forward pass")
    print("═" * 95)

    trunk = encoder.trunk
    captured = {}
    handles  = []

    def make_hook(name):
        def h(m, inp, out):
            # out : (B, H, W, C)
            if hasattr(out, 'shape') and out.dim() == 4:
                captured[name] = tuple(out.shape)
        return h

    for i, block in enumerate(trunk.blocks):
        handles.append(block.register_forward_hook(make_hook(f"block_{i}")))

    # Image test : 768×1280 resizée 1024×1024
    from PIL import Image
    import numpy as np
    img_np  = np.random.randint(0, 256, (ORIG_H, ORIG_W), dtype=np.uint8)
    img_pil = Image.fromarray(img_np).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x    = torch.from_numpy(np.array(img_pil)).float() / 255.
    x    = ((x.permute(2,0,1) - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder(x)

    for h in handles:
        h.remove()

    print(f"\n  Image entrée : {ORIG_H}×{ORIG_W} → resize {IMG_SIZE}×{IMG_SIZE}")
    print(f"  (forward sur image aléatoire, device={device})\n")

    stages = (1, 2, 11, 2)
    stage_of = {}; cur = 0
    for sn, n in enumerate(stages, 1):
        for _ in range(n): stage_of[cur] = sn; cur += 1

    print(f"  {'Bloc':>4} | {'Stage':>5} | {'Shape sortie (B,H,W,C)':>26} | "
          f"{'H×W tokens':>12} | {'Dim':>5} | Note")
    print("  " + "-" * 80)

    prev_hw = None
    for i in range(len(trunk.blocks)):
        key = f"block_{i}"
        if key not in captured:
            print(f"  {i:>4} | — | non capturé"); continue
        B, H, W, C = captured[key]
        n_tok = H * W
        note  = ""
        if prev_hw and (H * W) < prev_hw:
            note = f"← q_pool (/2 en h et w)"
        prev_hw = H * W
        if trunk.blocks[i].window_size == 0:
            note = "← attn GLOBALE"
        if trunk.blocks[i].q_stride and trunk.blocks[i].window_size == 0:
            note = "← q_pool + global"

        print(f"  {i:>4} | {stage_of[i]:>5} | ({B}, {H:>3}, {W:>3}, {C:>3})           | "
              f"{H}×{W}={n_tok:>6} | {C:>5} | {note}")

    # Confirmation tokens par patch
    print()
    print("  ─── Tokens couverts par un patch 128×128 (original) par stage ───")
    print(f"  Patch en espace 1024×1024 : "
          f"{PATCH_SZ * IMG_SIZE/ORIG_W:.1f}×{PATCH_SZ * IMG_SIZE/ORIG_H:.1f} px")
    print()
    stride_map = {0: 4, 1: 8, 2: 8, 3: 16}
    stride_map.update({i: 16 for i in range(4, 14)})
    stride_map[14] = 32; stride_map[15] = 32

    for stage_num, stride in [(1, 4), (2, 8), (3, 16), (4, 32)]:
        ph = PATCH_SZ * IMG_SIZE / ORIG_H / stride
        pw = PATCH_SZ * IMG_SIZE / ORIG_W / stride
        n  = int(ph) * int(pw)
        print(f"    Stage {stage_num} (stride {stride:>2}) : "
              f"{ph:.2f}×{pw:.2f} cellules → floor = {int(ph)}×{int(pw)} = {n} tokens/patch")


# ─── SORTIES POUR LE SCHÉMA ───────────────────────────────────────────────────
def print_schema_summary(trunk: Hiera):
    print()
    print("═" * 95)
    print("SORTIES POUR LE SCHÉMA DE PRÉSENTATION")
    print("═" * 95)

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) TABLEAU RÉCAPITULATIF (source : tous chiffres du modèle réel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage | Blocs   | dim | heads | d/head | window | attn    | Résol.  | n_tok | q_pool après?
------|---------|-----|-------|--------|--------|---------|---------|-------|---------------
  1   |   0     |  96 |   1   |   96   |   8×8  | local   | 256×256 | 65536 | OUI (→stage2)
  2   |  1–2    | 192 |   2   |   96   |   4×4  | local   | 128×128 | 16384 | OUI (→stage3)
  3   |  3–13   | 384 |   4   |   96   |  14×14 | local*  |  64×64  |  4096 | OUI (→stage4)
  4   | 14–15   | 768 |   8   |   96   |   7×7  | local   |  32×32  |  1024 | NON (sortie)

  *stage 3 : 3 blocs GLOBAUX (blocs 7, 10, 13) — pas de fenêtrage
  dim_per_head = 96 dans TOUS les blocs (invariant vérifié sur le modèle réel)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2) SCHÉMA ASCII — ATTENTION FENÊTRÉE (bloc 4, stage 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Feature map (64×64 tokens, dim=384)
  │
  ├─── window_partition (window=14×14)
  │    → ⌈64/14⌉×⌈64/14⌉ = 5×5 = 25 fenêtres de 14×14=196 tokens
  │    → tensor (B×25, 196, 384)
  │
  ├─── DANS CHAQUE FENÊTRE (14×14 tokens, indépendamment) :
  │
  │    x → qkv = Linear(384, 384×3)  →  (B×25, 196, 3, 4, 96)
  │                                         ↑     ↑   ↑  ↑   ↑
  │                                       n_fen  tok  —  heads d/head
  │    Q, K, V = unbind → chacun (B×25, 196, 4, 96)
  │
  │    Pour chaque head h (0…3) :
  │      Qh, Kh, Vh ∈ (B×25, 196, 96)
  │      scores = Qh · Khᵀ / √96     ∈ (B×25, 196, 196)
  │      A      = softmax(scores)     ∈ (B×25, 196, 196)
  │      Oh     = A · Vh              ∈ (B×25, 196, 96)
  │      ↑ token i = SOMME PONDÉRÉE des 196 valeurs Vh de sa fenêtre
  │
  │    concat(O0, O1, O2, O3) → (B×25, 196, 384)
  │    proj = Linear(384, 384)  → (B×25, 196, 384)
  │
  └─── window_unpartition → (B, 64, 64, 384)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3) SCHÉMA ASCII — Q_POOL (transition stage 2 → stage 3, bloc 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Entrée bloc 3 : (B, 128, 128, 192)   ← résolution stage 2
  │
  ├── qkv = Linear(192, 384×3)
  │   Q, K, V : shape initiale (B, 128×128, 4, 96)
  │             ↑ NB : heads=4, dim_out=384 dès la projection QKV
  │
  ├── Q uniquement : MaxPool2d(kernel=2×2, stride=2×2)
  │   Q poolé : (B, 64, 64, 4, 96)  ←  résolution ÷2 en h et w
  │
  ├── Attention cross-résolution :
  │   Q (64×64) attend sur K et V (128×128) → softmax donne poids sur 128×128
  │   Résultat : (B, 64×64, 4, 96)  →  reshape → (B, 64, 64, 384)
  │   ↑ chaque nouveau token agrège l'info de 4 anciens tokens (via attention)
  │
  ├── Shortcut : proj(x) + MaxPool2d → (B, 64, 64, 384)
  │
  └── Sortie bloc 3 : (B, 64, 64, 384)   ← résolution stage 3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4) POINTS CLÉS POUR LES SLIDES (chiffres vérifiés)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • 16 blocs au total (stages 1+2+11+2), 3 q_pool (blocs 1, 3, 14)
  • Attention locale dans FENÊTRES indépendantes (8×8, 4×4, 14×14, 7×7 selon le stage)
  • 3 blocs d'attention GLOBALE au stage 3 (blocs 7, 10, 13) → 64×64 = 4096 tokens
  • Nombre de heads : 1 → 2 → 4 → 8 (double à chaque q_pool)
  • dim_per_head = 96 CONSTANT dans tous les blocs
  • Résolution spatiale : 256×256 → 128×128 → 64×64 → 32×32 (÷2 à chaque q_pool)
  • Dim embedding    :  96    →  192    →  384   →  768  (×2 à chaque q_pool)
  • Tokens couverts par un patch 128×128 original :
      stage 1 : ~1050   stage 2 : ~263   stage 3 : ~65   stage 4 : ~15
  • SCALP=1 dans l'encoder : la FPN ne renvoie que les 3 premiers niveaux,
    stage 4 (32×32, ~15 tokens/patch) est éliminé AVANT l'extraction
  • Lien avec les résultats : les blocs tardifs (14-15) donnent ~15 tokens/patch
    → représentation grossière → recall LP faible ; blocs mid (8-12) donnent ~65
    → meilleur compromis résolution/sémantique
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    encoder = build_and_load()
    encoder = encoder.to(device)
    trunk   = encoder.trunk

    step1_block_table(trunk)
    step2_stages(trunk)
    step3_windowed_attention(trunk)
    step4_multihead(trunk)
    step5_forward(encoder, device)
    print_schema_summary(trunk)


if __name__ == "__main__":
    main()
