# Invariance géométrique — block_0 TextureSAM — Résultats

## Configuration
- N_PER_CAT = 50 · SEED = 42
- Catégories : ['Totalement homogène', 'Faisceaux', 'Filaments', 'Stratifié rectiligne', 'Stratifié sinueux', 'Granuleux', 'Trou']
- Transformations : ['Rot 90°↻', 'Rot 180°', 'Rot 270°↻', 'Flip ↔', 'Flip ↕']

## Similarité par transformation (toutes catégories)

| Transformation | Similarité moy. | Std | Verdict |
|---|---|---|---|
| Rot 90°↻ | 0.8790 | 0.0468 | ⚠️ partiel |
| Rot 180° | 0.8797 | 0.0468 | ⚠️ partiel |
| Rot 270°↻ | 0.8791 | 0.0466 | ⚠️ partiel |
| Flip ↔ | 0.8815 | 0.0469 | ⚠️ partiel |
| Flip ↕ | 0.8805 | 0.0468 | ⚠️ partiel |

## Similarité par texture (moyenne sur toutes les transformations)

| Texture | Sim moy. | Sim min | Verdict | Note |
|---|---|---|---|---|
| Totalement homogène | 0.9042 | 0.9030 | ✅ | isotrope |
| Faisceaux | 0.8795 | 0.8785 | ⚠️ | orientée |
| Filaments | 0.8627 | 0.8616 | ⚠️ | orientée |
| Stratifié rectiligne | 0.7916 | 0.7905 | ⚠️ | orientée |
| Stratifié sinueux | 0.8903 | 0.8894 | ⚠️ | orientée |
| Granuleux | 0.9577 | 0.9564 | ✅ | isotrope |
| Trou | 0.8776 | 0.8767 | ⚠️ | isotrope |

## Synthèse

- **Texture la plus invariante** : Granuleux (0.9577)
- **Texture la moins invariante** : Stratifié rectiligne (0.7916)
- **Transformation la plus perturbatrice** : Rot 90°↻ (0.8790)

## Interprétation

Une similarité cosine < 0.9 entre features originales et features après
transformation indique que block_0 est SENSIBLE à cette transformation.
Les textures orientées (Faisceaux, Filaments, Stratifié) pourraient
légitimement être moins invariantes à la rotation : cela reflète une
propriété réelle de la texture (l'orientation fait partie de sa définition),
pas un défaut du réseau.

## Fichiers
- `geo_barplot_by_transform.png` : Similarité par transformation
- `geo_heatmap_cat_transform.png` : Heatmap texture × transformation
- `geo_diagnostic_patches.png` : Vignettes avec similarité par patch
