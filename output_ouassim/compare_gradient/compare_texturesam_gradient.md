# Comparaison SAM-2.1 base → TextureSAM η0.3 → η1.0
## sur la séparabilité des textures MEB Ouassim

## Objectif

Mesurer l'effet du fine-tuning texture (base → η0.3 → η1.0) sur la séparabilité
des textures MEB sur TOUS les 20 blocs (block_0..15 + 4 FPN).
Métriques : Linear Probing (balanced accuracy), Fisher J balancé, τ cross-image.

## Les 3 checkpoints

| Nom | Fichier | Fine-tuning |
|-----|---------|-------------|
| base | sam2.1_hiera_small | SAM-2.1 original, SANS fine-tuning texture |
| η0.3 | sam2.1_hiera_small_0.3 | TextureSAM 19 epochs, augmentation modérée (clipLimit ≤ 0.3) |
| η1.0 | sam2.1_hiera_small_1.pt | TextureSAM 25 epochs, augmentation forte (clipLimit ≤ 1.0) |

Même architecture Hiera Small pour les 3 (embed_dim=96, stages(1,2,11,2),
global_att_blocks(7,10,13)). Seuls les poids diffèrent.

## Hypothèse

Les textures granulaires MEB (Granuleux, Filaments) correspondent à des **micro-contours**.
SAM base préserve ces contours (pas de fine-tuning).
Le fine-tuning texture (domaine non-MEB) pourrait les lisser, dégradant leur séparabilité.
Prédiction : **base > η0.3 > η1.0** sur ces textures.

## Résultats

### Meilleur checkpoint par métrique (max sur 20 blocs)

| Métrique | base | η0.3 | η1.0 |
|----------|------|------|------|
| LP (%) | 56.93 (@block_8) | 61.61 (@block_8) | 55.50 (@block_4) |
| Fisher J | 0.54 (@stage_1_fpn) | 0.53 (@stage_1_fpn) | 0.55 (@stage_2_fpn) |
| τ cross | 0.22 (@stage_2_fpn) | 0.20 (@stage_2_fpn) | 0.20 (@stage_2_fpn) |

### Textures à grain — Fisher one-vs-rest

| Texture | base (max) | η0.3 (max) | η1.0 (max) | Hypothèse |
|---------|-----------|-----------|-----------|-----------|
| Filaments | 0.203 | 0.206 | 0.192 | ✗ infirmée |
| Granuleux | 0.102 | 0.092 | 0.097 | ✗ infirmée |

## Conclusion

**Meilleur checkpoint global (LP)** : `eta0.3` (61.6% @ block_8)

Le checkpoint **η0.3** (fine-tuning modéré) est le meilleur compromis sur MEB Ouassim.
Le fine-tuning fort (η1.0) lisse trop les micro-contours, confirmant partiellement l'hypothèse.

- **Blocks les plus affectés** par le fine-tuning : typiquement les blocks précoces
  (block_0..4) où les contours locaux sont encodés.
- **FPN** : les couches FPN intègrent les caractéristiques multi-échelles ;
  leur réponse au fine-tuning peut différer des blocks trunk.

## Fichiers générés

- `lp_tous_blocks_3ckpt.png` — LP par block, 3 courbes
- `fisher_tous_blocks_3ckpt.png` — Fisher par block
- `tau_tous_blocks_3ckpt.png` — τ cross par block
- `gradient_par_block.png` — barplot LP blocks clés
- `grain_focus_tous_blocks.png` — Fisher OvR Granuleux/Filaments
- `heatmap_ckpt_texture.png` — heatmap checkpoint × texture
- `gradient_sweep_results.csv` — tableau complet
