"""
Vérifie si le préprocess SAM (resize + normalisation ImageNet)
rend block_0 trivialement invariant aux transformations
brightness / contrast / gamma.
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
IMG_DIR  = ROOT / 'PatchTagger_Output' / 'full_images'
IMG_NAME = '310120-pat18-WholeMount-24.tif'
IMG_SIZE = 1024
MEAN     = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD      = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

img_path = IMG_DIR / IMG_NAME
if not img_path.exists():
    # fallback : première image disponible
    img_path = sorted(IMG_DIR.glob('*.tif'))[0]
    IMG_NAME = img_path.name

print(f'Image utilisée : {IMG_NAME}\n')

# ── Preprocess (identique à build_feature_database + LDA cell) ───────────────
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Resize 1024×1024 + normalisation ImageNet → (3, H, W)."""
    img_r = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img_r)).float() / 255.0   # (H, W, 3) ∈ [0,1]
    return (x.permute(2, 0, 1) - MEAN) / STD                # (3, H, W)

# ── Transformations ───────────────────────────────────────────────────────────
def apply_brightness(im, beta):
    return np.clip(im + beta, 0.0, 1.0)

def apply_contrast(im, alpha):
    return np.clip((im - 0.5) * alpha + 0.5, 0.0, 1.0)

def apply_gamma(im, gamma):
    return np.clip(im ** gamma, 0.0, 1.0)

# ── Charger image ─────────────────────────────────────────────────────────────
img_pil = Image.open(img_path).convert('RGB')
img_np  = np.array(img_pil).astype(np.float32) / 255.0

print(f'Image shape      : {img_np.shape}')
print(f'Image pixel mean : {img_np.mean():.4f}   std : {img_np.std():.4f}\n')

cases = {
    'original'        : img_np,
    'brightness +0.3' : apply_brightness(img_np,  0.3),
    'brightness -0.3' : apply_brightness(img_np, -0.3),
    'contrast 1.5'    : apply_contrast(img_np, 1.5),
    'contrast 0.5'    : apply_contrast(img_np, 0.5),
    'gamma 2.0'       : apply_gamma(img_np, 2.0),
    'gamma 0.5'       : apply_gamma(img_np, 0.5),
}

ref_tensor = preprocess(Image.fromarray((img_np * 255).astype(np.uint8)))

# ── Tableau de comparaison ────────────────────────────────────────────────────
print(f'{"Transformation":<18} │ {"mean tensor":>12} │ '
      f'{"std tensor":>10} │ {"diff vs original":>17}')
print('─' * 68)

for name, im in cases.items():
    t      = preprocess(Image.fromarray((im * 255).astype(np.uint8)))
    mean_t = float(t.mean())
    std_t  = float(t.std())
    diff   = float((t - ref_tensor).abs().mean())

    flag = ''
    if name != 'original' and diff < 0.01:
        flag = '  ← trivial (normalisé)'
    elif name != 'original' and diff > 0.05:
        flag = '  ← atteint le réseau'

    print(f'{name:<18} │ {mean_t:>12.4f} │ {std_t:>10.4f} │ '
          f'{diff:>17.4f}{flag}')

# ── Interprétation ────────────────────────────────────────────────────────────
print('\n─── Lecture ───')
print('diff vs original ≈ 0  → preprocess absorbe la transformation')
print('                        → invariance de block_0 serait triviale')
print('diff > 0 significatif → la transformation atteint le réseau')
print('                        → invariance vient vraiment de block_0')

print('\n─── Détail tenseur "original" (après preprocess) ───')
print(f'  mean  : {float(ref_tensor.mean()):+.4f}   (attendu ≈ 0 si ImageNet mean)')
print(f'  std   : {float(ref_tensor.std()):.4f}   (attendu ≈ 1 si ImageNet std)')
print(f'  min   : {float(ref_tensor.min()):+.4f}')
print(f'  max   : {float(ref_tensor.max()):+.4f}')

# ── Vérification analytique ───────────────────────────────────────────────────
print('\n─── Vérification analytique pour brightness +β ───')
beta  = 0.3
# preprocess(original) = (x - mean) / std
# preprocess(x + β)    = (x + β - mean) / std  = preprocess(x) + β/std
beta_shift = beta / STD.squeeze()
print(f'  β={beta}  →  shift attendu par canal :')
for c, (ch, s) in enumerate(zip(['R','G','B'], beta_shift.tolist())):
    print(f'    canal {ch} : β/std = {beta:.3f} / {STD.squeeze()[c]:.3f} = {s:+.4f}')
t_bright = preprocess(
    Image.fromarray(
        (apply_brightness(img_np, beta) * 255).astype(np.uint8)
    )
)
shift_obs = float((t_bright - ref_tensor).mean())
print(f'  shift observé (mean sur tous canaux/pixels) : {shift_obs:+.4f}')
print('  → brightness est un décalage additif constant : il ATTEINT le réseau.')
