import sys
import os

# Ajouter le dossier sam2 au path pour que "from sam2.build_sam import ..." fonctionne
SAM2_DIR = os.path.join(os.path.dirname(__file__), "TextureSAM", "sam2")
sys.path.insert(0, SAM2_DIR)

import torch
from sam2.build_sam import build_sam2

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "sam2.1_hiera_small_1.pt")
CONFIG    = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
print(f"Checkpoint : {CHECKPOINT}")
print(f"Config : {CONFIG}")
print("=" * 80)

# Charger sans checkpoint si absent (architecture identique, poids aléatoires)
ckpt = CHECKPOINT if os.path.exists(CHECKPOINT) else None
if ckpt is None:
    print("⚠️  Checkpoint absent — architecture chargée avec poids aléatoires\n")
model = build_sam2(CONFIG, ckpt, device=DEVICE, mode="eval")

# ── 2. Architecture complète de l'image encoder ──────────────────────────────
print("\n=== image_encoder (repr complète) ===\n")
print(model.image_encoder)

# ── 3. Tous les sous-modules avec leur nom ───────────────────────────────────
print("\n=== named_modules de image_encoder ===\n")
for name, module in model.image_encoder.named_modules():
    print(f"{name}  →  {type(module).__name__}")

# ── 4. Modules contenant "stage", "block" ou "layer" dans leur nom ───────────
print("\n=== Modules avec 'stage' / 'block' / 'layer' dans le nom ===\n")
keywords = ("stage", "block", "layer")
matches = [
    (name, type(module).__name__)
    for name, module in model.image_encoder.named_modules()
    if any(k in name.lower() for k in keywords)
]
for name, cls in matches:
    print(f"{name}  →  {cls}")

# ── 5. Décompte ──────────────────────────────────────────────────────────────
print(f"\n=== Total modules trouvés : {len(matches)} ===")

stage_names = [name for name, _ in matches if "stage" in name.lower()]
block_names = [name for name, _ in matches if "block" in name.lower()]
layer_names  = [name for name, _ in matches if "layer"  in name.lower()]
print(f"  dont 'stage' : {len(stage_names)}")
print(f"  dont 'block' : {len(block_names)}")
print(f"  dont 'layer' : {len(layer_names)}")
