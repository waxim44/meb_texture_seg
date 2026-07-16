"""
lora_supcon_phase0.py
══════════════════════════════════════════════════════════════════════════════
PHASE 0 — Environnement et données (LoRA + SupCon sur TextureSAM).

Vérifie, avant tout code d'entraînement :
  1. H5 = database_meb_ouassim.h5, images = Image_Ouassim/ BRUTES
     (grayscale, mean ~85 — PAS PatchTagger RGB mean ~106).
  2. GPU disponible + mémoire.
  3. Checkpoint sam2.1_hiera_small_1.pt chargeable dans un Hiera réel.
  4. Inventaire des patchs par texture (7 classes cibles).

Produit un rapport texte dans lora_supcon/phase_0/report.txt avec les
cases de validation cochées.
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import glob
import os
import collections
from pathlib import Path

import numpy as np
import h5py
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SAM2 = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2) not in sys.path:
    sys.path.insert(0, str(_SAM2))

from sam2.modeling.backbones.hieradet import Hiera  # noqa: E402

H5_PATH = _ROOT / "data" / "feature_database" / "database_meb_ouassim.h5"
IMG_DIR = _ROOT / "Image_Ouassim"
CKPT_PATH = _ROOT / "checkpoints" / "sam2.1_hiera_small_1.pt"
OUTDIR = _ROOT / "lora_supcon" / "phase_0"
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COUNTS = {
    "Totalement homogène": 41,
    "Trou": 56,
    "Granuleux": 409,
    "Stratifié rectiligne": 64,
    "Filaments": 49,
    "Stratifié sinueux": 129,
    "Faisceaux": 68,
}


def check_data_source(lines):
    lines.append("── 1. Source des données (Ouassim vs PatchTagger) ──")
    lines.append(f"H5 utilisé   : {H5_PATH}")
    lines.append(f"Existe       : {H5_PATH.exists()}")
    lines.append(f"Images (dir) : {IMG_DIR}")

    files = sorted(glob.glob(str(IMG_DIR / "*.tif")))
    lines.append(f"Nb fichiers .tif : {len(files)}")

    im = Image.open(files[0])
    arr = np.array(im)
    lines.append(
        f"Exemple : {os.path.basename(files[0])}  mode={im.mode}  "
        f"shape={arr.shape}  dtype={arr.dtype}  mean={arr.mean():.2f}"
    )

    is_grayscale = im.mode == "L"
    is_ouassim_mean = 60 < arr.mean() < 110  # attendu ~85, PAS ~106 RGB
    ok = is_grayscale and is_ouassim_mean and H5_PATH.exists()
    lines.append(
        f"[{'x' if ok else ' '}] Ouassim confirmé "
        f"(grayscale={is_grayscale}, mean plausible={is_ouassim_mean})"
    )
    return ok


def check_gpu(lines):
    lines.append("")
    lines.append("── 2. GPU ──")
    avail = torch.cuda.is_available()
    lines.append(f"CUDA disponible : {avail}")
    if avail:
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        lines.append(f"Device : {name}")
        lines.append(f"Mémoire libre / totale : {free/1e9:.1f} / {total/1e9:.1f} GB")
    return avail


def check_checkpoint(lines):
    lines.append("")
    lines.append("── 3. Checkpoint sam2_hiera_s ──")
    lines.append(f"Chemin : {CKPT_PATH}")
    exists = CKPT_PATH.exists()
    lines.append(f"Existe : {exists}")
    if not exists:
        lines.append("[ ] checkpoint OK")
        return False

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    trunk = Hiera(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    trunk_state = {
        k[len("image_encoder.trunk."):]: v
        for k, v in state_dict.items()
        if k.startswith("image_encoder.trunk.")
    }
    missing, unexpected = trunk.load_state_dict(trunk_state, strict=False)
    loadable = len(trunk_state) > 0 and len(missing) == 0
    lines.append(f"Clés trunk trouvées dans le checkpoint : {len(trunk_state)}")
    lines.append(f"Clés manquantes après chargement : {len(missing)}")
    lines.append(f"Clés inattendues (ignorées) : {len(unexpected)}")
    lines.append(f"[{'x' if loadable else ' '}] checkpoint OK (Hiera chargé, strict sur trunk)")
    return loadable


def check_inventory(lines):
    lines.append("")
    lines.append("── 4. Inventaire des patchs par texture ──")
    f = h5py.File(str(H5_PATH), "r")
    names = f["metadata/image_names"][:]
    cats = f["metadata/category_names"][:]
    n_images = len(set(n.decode() for n in names))
    lines.append(f"Nb images (H5) : {n_images}")

    cnt = collections.Counter(c.decode() for c in cats)
    all_ok = True
    for tex, exp in EXPECTED_COUNTS.items():
        actual = cnt.get(tex, 0)
        ok = actual == exp
        all_ok &= ok
        lines.append(f"  {tex:25s} attendu={exp:4d}  trouvé={actual:4d}  {'OK' if ok else 'MISMATCH'}")
    lines.append(f"[{'x' if all_ok else ' '}] inventaire affiché et conforme")
    return all_ok


def main():
    lines = ["═" * 70, "PHASE 0 — Rapport de validation", "═" * 70]

    ok_data = check_data_source(lines)
    ok_gpu = check_gpu(lines)
    ok_ckpt = check_checkpoint(lines)
    ok_inv = check_inventory(lines)

    lines.append("")
    lines.append("── VALIDATION P0 ──")
    lines.append(f"[{'x' if ok_data else ' '}] Ouassim confirmé")
    lines.append(f"[{'x' if ok_ckpt else ' '}] checkpoint OK")
    lines.append(f"[{'x' if ok_inv else ' '}] inventaire affiché")
    lines.append(f"[{'x' if ok_gpu else ' '}] GPU disponible")
    go = ok_data and ok_ckpt and ok_inv and ok_gpu
    lines.append("")
    lines.append(f"VERDICT : {'GO' if go else 'NO-GO'}")

    report = "\n".join(lines)
    print(report)
    (OUTDIR / "report.txt").write_text(report)
    return go


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
