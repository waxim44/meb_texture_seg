"""
Inspection de l'architecture TextureSAM (Hiera Small + FPN Neck).
Lance directement sans Hydra — instancie les classes Python manuellement.

Usage:
    python scripts/inspect_stages.py
"""

import os
import sys
import zipfile
import tempfile

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_DIR = os.path.join(ROOT, "TextureSAM", "sam2")
sys.path.insert(0, SAM2_DIR)

CKPT_DIR = os.path.join(ROOT, "checkpoints", "sam2.1_hiera_small_1")
CKPT_PT  = os.path.join(ROOT, "checkpoints", "sam2.1_hiera_small_1.pt")

import torch
import torch.nn as nn

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine


# ── Helpers visuels ────────────────────────────────────────────────────────────

def hr(char="─", width=60):
    print(char * width)


def box_top(width=52):
    print("╔" + "═" * width + "╗")


def box_sep(width=52):
    print("╠" + "═" * width + "╣")


def box_bot(width=52):
    print("╚" + "═" * width + "╝")


def box_row(text, width=52):
    print("║" + text.center(width) + "║")


# ── Chargement checkpoint ──────────────────────────────────────────────────────

def _try_load_ckpt():
    """
    Essaie de charger le state_dict depuis un .pt ou depuis le répertoire
    archive extrait (zip PyTorch). Retourne le state_dict ou None.
    """
    # 1. Fichier .pt classique
    if os.path.isfile(CKPT_PT):
        try:
            sd = torch.load(CKPT_PT, map_location="cpu", weights_only=True)
            return sd.get("model", sd)
        except Exception:
            pass

    # 2. Répertoire = archive zip PyTorch extraite — on la re-zippe à la volée
    archive_dir = os.path.join(CKPT_DIR, "archive") if os.path.isdir(
        os.path.join(CKPT_DIR, "archive")) else CKPT_DIR
    if os.path.isdir(archive_dir):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp_path = tmp.name
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for dirpath, _, filenames in os.walk(archive_dir):
                    for fname in filenames:
                        full = os.path.join(dirpath, fname)
                        arcname = os.path.relpath(full, os.path.dirname(archive_dir))
                        zf.write(full, arcname)
            sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.unlink(tmp_path)
            return sd.get("model", sd)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return None


# ── Construction du modèle ─────────────────────────────────────────────────────

def build_image_encoder():
    """
    Instancie ImageEncoder (Hiera trunk + FPN neck) selon les paramètres
    de configs/sam2.1/sam2.1_hiera_s.yaml — sans Hydra.
    """
    trunk = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
        # window_spec et q_pool gardent leurs valeurs par défaut
        # window_spec=(8, 4, 14, 7), q_pool=3
    )

    pos_enc = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )

    neck = FpnNeck(
        position_encoding=pos_enc,
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1,
        stride=1,
        padding=0,
        fpn_interp_model="nearest",
        fuse_type="sum",
        fpn_top_down_levels=[2, 3],
    )

    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    return encoder


# ── Inspection blocs ───────────────────────────────────────────────────────────

def inspect_blocks(trunk):
    """
    Parcourt trunk.blocks et collecte les infos par bloc.
    Retourne une liste de dicts et la table des stages.
    """
    blocks_info = []
    stage_table = []  # [(stage_id, bloc_range, channels, transition)]

    current_stage = 1
    stage_start = 0
    prev_dim = None

    for idx, blk in enumerate(trunk.blocks):
        has_pool = blk.pool is not None
        dim_in = blk.dim
        dim_out = blk.dim_out
        win = blk.window_size

        if win == 0:
            attn_type = "global (win=0)"
        else:
            attn_type = f"window win={win}"

        blocks_info.append({
            "idx": idx,
            "has_pool": has_pool,
            "dim_in": dim_in,
            "dim_out": dim_out,
            "attn_type": attn_type,
        })

        # Début d'un nouveau stage = ce bloc a un pool
        if has_pool and idx > 0:
            # fermer le stage précédent
            stage_table.append({
                "stage": current_stage,
                "start": stage_start,
                "end": idx - 1,
                "channels": prev_dim,
                "transition": f"—",
            })
            current_stage += 1
            stage_start = idx
            # marquer la transition dans le dernier stage enregistré
            stage_table[-1]["transition_next"] = f"Pool @ bloc {idx}"

        prev_dim = dim_out

    # Fermer le dernier stage
    stage_table.append({
        "stage": current_stage,
        "start": stage_start,
        "end": len(trunk.blocks) - 1,
        "channels": prev_dim,
        "transition": "—",
    })

    # Propager les labels de transition
    for i, st in enumerate(stage_table):
        if "transition_next" in st:
            # La transition se produit au début du stage suivant
            if i + 1 < len(stage_table):
                stage_table[i + 1]["transition"] = st["transition_next"]

    return blocks_info, stage_table


# ── Affichage détaillé bloc par bloc ──────────────────────────────────────────

def print_blocks_detail(blocks_info):
    print()
    hr("═")
    print(" Détail bloc par bloc (trunk.blocks)")
    hr("═")
    header = f"  {'Bloc':>4}  {'dim_in':>7}  {'dim_out':>8}  {'Pool':>4}  {'Attention':<22}"
    print(header)
    hr()
    for b in blocks_info:
        pool_marker = "◉" if b["has_pool"] else "·"
        prefix = "▶ " if b["has_pool"] else "  "
        print(
            f"{prefix}{'bloc ' + str(b['idx']):>6}  "
            f"{b['dim_in']:>7}  {b['dim_out']:>8}  "
            f"{pool_marker:>4}  {b['attn_type']:<22}"
        )


# ── Tableau récapitulatif stages ───────────────────────────────────────────────

def print_stage_table(stage_table):
    print()
    W = 52
    box_top(W)
    box_row("Architecture Hiera Small  (stages=[1, 2, 11, 2])", W)
    print("╠" + "════════╦" + "══════════╦" + "══════════╦" + "═══════════════════╣")
    print("║ Stage  ║  Blocs   ║ Canaux   ║ Transition        ║")
    print("╠" + "════════╬" + "══════════╬" + "══════════╬" + "═══════════════════╣")
    for st in stage_table:
        s = st["stage"]
        blocs_range = (
            str(st["start"])
            if st["start"] == st["end"]
            else f"{st['start']}-{st['end']}"
        )
        ch = st["channels"]
        tr = st["transition"]
        print(
            f"║ {s:^6} ║ {blocs_range:^8} ║ {ch:^8} ║ {tr:<17} ║"
        )
    print("╚" + "════════╩" + "══════════╩" + "══════════╩" + "═══════════════════╝")


# ── Tableau FPN Neck ───────────────────────────────────────────────────────────

def print_fpn_table(neck):
    backbone_ch = neck.backbone_channel_list   # [768, 384, 192, 96]
    d_model = neck.d_model                      # 256
    n = len(backbone_ch)

    print()
    W = 52
    box_top(W)
    box_row("FPN Neck Sorties", W)
    print("╠" + "═════════╦" + "══════════════╦" + "═════════════════════════╣")
    print("║ Stage   ║  Dim entrée  ║  Dim sortie (après conv) ║")
    print("╠" + "═════════╬" + "══════════════╬" + "═════════════════════════╣")
    for i, ch in enumerate(backbone_ch):
        stage_id = n - i   # 4, 3, 2, 1
        print(f"║ {stage_id:^7} ║ {ch:^12} ║ {d_model:^25} ║")
    print("╚" + "═════════╩" + "══════════════╩" + "═════════════════════════╝")


# ── Forward pass avec hooks ────────────────────────────────────────────────────

def run_forward_pass(encoder, device):
    print()
    hr("═")
    print(" Forward pass — image 1×3×1024×1024")
    hr("═")

    hook_outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            hook_outputs[name] = out.shape
        return hook

    handles = []
    for i in range(len(encoder.neck.convs)):
        h = encoder.neck.convs[i].register_forward_hook(make_hook(f"neck.convs.{i}"))
        handles.append(h)

    encoder.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 1024, 1024, device=device)
        _ = encoder(x)

    for h in handles:
        h.remove()

    n = len(encoder.neck.convs)
    print("\n Shapes réelles des feature maps :")
    for i in range(n):
        key = f"neck.convs.{i}"
        stage_id = n - i  # convs.0 → stage 4, convs.1 → stage 3, etc.
        shape = hook_outputs.get(key, "N/A")
        print(f"  → Stage {stage_id} ({key}) : {shape}")

    return hook_outputs


# ── Résolution spatiale par stage ──────────────────────────────────────────────

def print_resolution_table(hook_outputs, neck, image_size=1024):
    n = len(neck.convs)
    print()
    hr("═")
    print(" Résolution par stage :")
    hr()
    for i in range(n):
        key = f"neck.convs.{i}"
        stage_id = n - i
        shape = hook_outputs.get(key)
        if shape is None:
            print(f"  → Stage {stage_id} : données absentes")
            continue
        # shape = (B, C, H, W)
        H, W = shape[2], shape[3]
        n_vecs = H * W
        coverage = image_size // H  # pixels couverts par vecteur
        print(
            f"  → Stage {stage_id} : H×W = {H}×{W}\n"
            f"      = {n_vecs} vecteurs\n"
            f"      = chaque vecteur couvre {coverage}×{coverage} pixels"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    hr("═")
    print(" Inspection TextureSAM — Hiera Small")
    hr("═")
    print(f"  Device : {device}")

    # Charge le modèle
    encoder = build_image_encoder()

    # Tente de charger le checkpoint
    state_dict = _try_load_ckpt()
    if state_dict is not None:
        # Filtrer uniquement les clés de l'image_encoder si le SD vient du modèle complet
        enc_prefix = "image_encoder."
        if any(k.startswith(enc_prefix) for k in state_dict):
            state_dict = {
                k[len(enc_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(enc_prefix)
            }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"  Checkpoint chargé (partiel) — {len(missing)} clés manquantes, "
                  f"{len(unexpected)} inattendues")
        else:
            print("  Checkpoint chargé avec succès.")
    else:
        print(
            "  checkpoint absent,\n"
            "  on utilise les poids aléatoires\n"
            "  pour l'inspection architecture"
        )

    encoder = encoder.to(device)

    # ── Inspection des blocs ───────────────────────────────────────────────────
    trunk = encoder.trunk
    blocks_info, stage_table = inspect_blocks(trunk)
    print_blocks_detail(blocks_info)

    # ── Tableau stages ─────────────────────────────────────────────────────────
    print_stage_table(stage_table)

    # ── Tableau FPN ────────────────────────────────────────────────────────────
    print_fpn_table(encoder.neck)

    # ── Forward pass ──────────────────────────────────────────────────────────
    try:
        hook_outputs = run_forward_pass(encoder, device)
        print_resolution_table(hook_outputs, encoder.neck)
    except Exception as e:
        print(f"\n  [ERREUR forward pass] {e}")

    print()
    hr("═")
    print(" Justification des 4 stages")
    hr("═")
    print("  Hiera Small empile 16 blocs (stages=[1, 2, 11, 2]).")
    print("  3 transitions MaxPool2d (q_pool=3) créent 4 résolutions")
    print("  distinctes : 256×256 → 128×128 → 64×64 → 32×32 avec")
    print("  des canaux doublés à chaque saut (96→192→384→768).")
    print("  Le FPN regroupe ces 4 niveaux en 256 canaux chacun,")
    print("  ce qui justifie d'explorer les 4 stages pour capturer")
    print("  texture fine (stage 1) jusqu'à sémantique globale (stage 4).")
    print()


if __name__ == "__main__":
    main()
