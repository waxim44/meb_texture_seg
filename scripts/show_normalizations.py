#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

IMG_PATH = Path('/home/aidouni/meb_texture_seg/Image_Ouassim/060525-JPB-MEB-EIHNValves-Ech1-ZigZag20002.tif')
OUT      = Path('/home/aidouni/meb_texture_seg/output_ouassim/lp_norm_blocks')

img = np.array(Image.open(IMG_PATH)).astype(np.float32)
img_norm = img / 255.0

def save(arr, fname, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(arr, cmap='gray', vmin=arr.min(), vmax=arr.max())
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvé : {fname}")

# Originale
save(img_norm, 'transform_original.png', 'Originale')

# Gamma 0.7
g07 = np.power(img_norm, 0.7)
save(g07, 'transform_gamma_07.png', 'Gamma γ=0.7  (éclaircissement)')

# Gamma 1.5
g15 = np.power(img_norm, 1.5)
save(g15, 'transform_gamma_15.png', 'Gamma γ=1.5  (assombrissement)')

# Z-score par image
mean_ = img_norm.mean()
std_  = img_norm.std()
zsc   = (img_norm - mean_) / (std_ + 1e-8)
# pour affichage : ramener dans [0,1]
zsc_display = (zsc - zsc.min()) / (zsc.max() - zsc.min() + 1e-8)
save(zsc_display, 'transform_zscore.png', f'Z-score par image  (μ={mean_:.2f}, σ={std_:.2f})')
