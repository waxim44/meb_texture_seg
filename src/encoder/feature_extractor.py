"""
TextureSAMExtractor — extrait les feature maps multi-échelles depuis
l'image encoder de TextureSAM (Hiera trunk + FPN neck).

Hooks sur neck.convs.0..3 → stages 4..1 (résolutions 32→256).
"""

import os
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── SAM2 path ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SAM2_DIR = _ROOT / "TextureSAM" / "sam2"
if str(_SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(_SAM2_DIR))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ── Constantes ─────────────────────────────────────────────────────────────────
# Normalisation SAM standard (ImageNet)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Mapping: indice conv FPN → id stage sémantique
# neck.convs.0 → stage_4 (32×32), .1 → stage_3 (64×64), etc.
_CONV_TO_STAGE = {0: "stage_4", 1: "stage_3", 2: "stage_2", 3: "stage_1"}


# ── Helpers checkpoint ─────────────────────────────────────────────────────────

def _resolve_ckpt(ckpt_cfg: str, root: Path):
    """
    Cherche le checkpoint à partir du chemin config (relatif à root).
    Accepte un .pt classique ou le répertoire archive extrait.
    Retourne le path résolu ou None.
    """
    candidates = [
        root / ckpt_cfg,
        root / (ckpt_cfg.rstrip(".pt")),  # sans extension
    ]
    for p in candidates:
        if p.is_file() or p.is_dir():
            return p
    return None


def _load_state_dict(ckpt_path: Path):
    """
    Charge le state_dict depuis un .pt ou un répertoire archive extrait.
    Retourne le dict ou None si échec.
    """
    # Cas 1 : fichier .pt standard
    if ckpt_path.is_file():
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            return sd.get("model", sd)
        except Exception:
            return None

    # Cas 2 : répertoire = zip PyTorch extrait — re-zipper à la volée
    archive_dir = ckpt_path / "archive" if (ckpt_path / "archive").is_dir() else ckpt_path
    if archive_dir.is_dir():
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp_path = tmp.name
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for fp in archive_dir.rglob("*"):
                    if fp.is_file():
                        info = zipfile.ZipInfo(str(fp.relative_to(archive_dir.parent)))
                        info.date_time = (1980, 1, 1, 0, 0, 0)  # ZIP ne supporte pas avant 1980
                        with open(fp, "rb") as fh:
                            zf.writestr(info, fh.read())
            sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.unlink(tmp_path)
            return sd.get("model", sd)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return None


# ── Construction de l'image encoder ───────────────────────────────────────────

def _build_image_encoder() -> ImageEncoder:
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


# ── Classe principale ──────────────────────────────────────────────────────────

class TextureSAMExtractor:
    """
    Extrait les feature maps multi-échelles de TextureSAM.

    Paramètres
    ----------
    cfg : OmegaConf DictConfig
        Doit contenir cfg.encoder.{checkpoint, stage, image_size, normalize}
    root_dir : str | Path, optional
        Répertoire racine du projet. Résout les chemins relatifs du cfg.
        Par défaut : racine déduite depuis l'emplacement de ce fichier.
    """

    def __init__(self, cfg, root_dir=None):
        self.cfg = cfg
        self.root = Path(root_dir) if root_dir else _ROOT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = getattr(cfg.encoder, "image_size", 1024)
        self.normalize = getattr(cfg.encoder, "normalize", True)

        self.features = {
            "stage_1": None,
            "stage_2": None,
            "stage_3": None,
            "stage_4": None,
        }
        self._hooks = []

        self._build_model()
        self._register_hooks()

    # ── Construction + chargement checkpoint ──────────────────────────────────

    def _build_model(self):
        self.encoder = _build_image_encoder()

        ckpt_path = _resolve_ckpt(self.cfg.encoder.checkpoint, self.root)
        loaded = False
        if ckpt_path is not None:
            sd = _load_state_dict(ckpt_path)
            if sd is not None:
                # Filtrer les clés image_encoder.* si SD complet
                prefix = "image_encoder."
                if any(k.startswith(prefix) for k in sd):
                    sd = {k[len(prefix):]: v for k, v in sd.items()
                          if k.startswith(prefix)}
                missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
                if not missing and not unexpected:
                    print("  [encoder] Checkpoint chargé.")
                else:
                    print(f"  [encoder] Checkpoint partiel "
                          f"({len(missing)} manquantes, {len(unexpected)} inattendues).")
                loaded = True

        if not loaded:
            print("  [encoder] WARNING — checkpoint absent ou illisible :")
            print("            on utilise les poids aléatoires pour l'inspection.")

        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _register_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        for conv_idx, stage_name in _CONV_TO_STAGE.items():
            def _hook(module, inp, out, _name=stage_name):
                # out : (B, 256, H, W)  →  stocké en (H, W, 256) numpy
                self.features[_name] = out.detach().cpu()

            h = self.encoder.neck.convs[conv_idx].register_forward_hook(_hook)
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Prétraitement image ────────────────────────────────────────────────────

    def _preprocess(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path)

        # Convertir en RGB (STMD est en niveaux de gris)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        x = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
        x = x.permute(2, 0, 1)                               # (3, H, W)

        if self.normalize:
            mean = _MEAN.to(x.device)
            std  = _STD.to(x.device)
            x = (x - mean) / std

        return x.unsqueeze(0).to(self.device)                # (1, 3, H, W)

    # ── Extract une image ──────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, image_path: str) -> dict:
        """
        Retourne dict {stage_1..4: np.ndarray (H, W, 256)}.
        """
        x = self._preprocess(image_path)
        _ = self.encoder(x)

        result = {}
        for stage_name, feat in self.features.items():
            if feat is not None:
                # (1, 256, H, W) → (H, W, 256)
                result[stage_name] = feat[0].permute(1, 2, 0).numpy()
            else:
                result[stage_name] = None
        return result

    # ── Extract un dossier complet ─────────────────────────────────────────────

    @torch.no_grad()
    def extract_dataset(self, image_dir: str, output_dir: str):
        """
        Parcourt image_dir, extrait les features de chaque image, sauvegarde en .npy.

        cfg.encoder.stage:
            0 → sauvegarde les 4 stages
            1..4 → sauvegarde uniquement ce stage
        """
        from tqdm import tqdm

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
        if not images:
            print(f"  [WARN] Aucune image trouvée dans {image_dir}")
            return

        target_stage = int(self.cfg.encoder.stage)
        stages_to_save = (
            list(_CONV_TO_STAGE.values()) if target_stage == 0
            else [f"stage_{target_stage}"]
        )

        for img_path in tqdm(images, desc=f"  extract {image_dir.name}", unit="img"):
            feats = self.extract(str(img_path))
            stem = img_path.stem
            for stage_name in stages_to_save:
                arr = feats.get(stage_name)
                if arr is not None:
                    stage_dir = output_dir / stage_name
                    stage_dir.mkdir(exist_ok=True)
                    np.save(stage_dir / f"{stem}.npy", arr)
