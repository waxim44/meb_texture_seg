# Texture par block — base Ouassim

## Objectif

Les metriques globales (Fisher moyen, LP global) melangent toutes les textures : une texture bien formee dans un block peut etre masquee par les autres mal formees. On decompose ici en analysant **chaque texture dans chaque block** via des metriques one-vs-rest.

Hypothese : differentes textures sont optimales dans differents blocks (ex. textures fines = blocks precoces, textures structurees = blocks tardifs).

## Demarche

### Pourquoi one-vs-rest

Pour chaque texture c, on la compare a **toutes les autres** (rest). Cela isole la question : "cette texture forme-t-elle un groupe compact et separe dans ce block ?" independamment des autres textures.

### Les 3 metriques

- **Fisher one-vs-rest** : `J_c = ||mu_c - mu_rest||² / (sigma_c² + sigma_rest²)`. Eleve si la texture c est compacte ET eloignee du reste.
- **Silhouette** : silhouette moyen des patches de c dans l'espace de features a 7 classes. Eleve si c est compacte et bien separee de ses voisins.
- **Recall one-vs-rest** : classifieur binaire (c vs reste), 5-fold stratifie par image. Mesure si les patches de c sont retrouvables sans connaitre les autres textures.

### Protocole
PCA-50d + L2-norm par block, SEED=42. Tous les 20 blocks (block_0..15 + stage_1/2/3/4_fpn). 7 textures valides.

## Resultats

### Meilleur block par texture (Fisher)

| Texture | Meilleur block | Fisher max | Sil max | Recall max |
|---------|----------------|-----------|---------|-----------|
| Totalement homogène | `stage_2_fpn` | 0.732 | 0.384 | 0.97 |
| Trou | `stage_1_fpn` | 0.708 | 0.396 | 0.72 |
| Stratifié rectiligne | `stage_2_fpn` | 0.259 | 0.143 | 0.47 |
| Filaments | `stage_1_fpn` | 0.197 | 0.070 | 0.72 |
| Faisceaux | `stage_4_fpn` | 0.156 | 0.060 | 0.58 |
| Granuleux | `stage_2_fpn` | 0.137 | -0.009 | 0.86 |
| Stratifié sinueux | `block_15` | 0.095 | 0.023 | 0.63 |

### Les blocks optimaux sont-ils differents par texture ?

Le Fisher est maximise par **4 blocks differents** : `block_15`, `stage_1_fpn`, `stage_2_fpn`, `stage_4_fpn`.

L'hypothese est **confirmee** : differentes textures trouvent leur meilleure representation dans des blocks differents. Il n'y a pas un seul block universellement optimal.

Implication : un systeme expert qui choisirait le block optimal par texture (oracle) ameliorerait la separation texturale vs l'approche "un seul block pour tout".

### Textures faciles vs dures

(seuil = mediane Fisher max (0.197))

**Textures representables** (Fisher max > mediane) : Totalement homogène, Trou, Stratifié rectiligne.
**Textures difficiles** (Fisher max <= mediane) : Filaments, Faisceaux, Granuleux, Stratifié sinueux.

- **Totalement homogène** : bien representee dans `stage_2_fpn` (Fisher=0.732). Texturallement distinctive dans l'espace de features.
- **Trou** : bien representee dans `stage_1_fpn` (Fisher=0.708). Texturallement distinctive dans l'espace de features.
- **Stratifié rectiligne** : representee dans `stage_2_fpn` (Fisher=0.259). Separation partielle, exploitable.
- **Filaments** : difficile meme dans son meilleur block `stage_1_fpn` (Fisher=0.197). Se confond avec d'autres textures.
- **Faisceaux** : difficile meme dans son meilleur block `stage_4_fpn` (Fisher=0.156). Se confond avec d'autres textures.
- **Granuleux** : difficile meme dans son meilleur block `stage_2_fpn` (Fisher=0.137). Se confond avec d'autres textures.
- **Stratifié sinueux** : difficile meme dans son meilleur block `block_15` (Fisher=0.095). Se confond avec d'autres textures.

## Conclusion

L'analyse par texture confirme que **les blocks optimaux varient selon la texture**. La moyenne globale cache cette heterogeneite.

Piste : une **representation par texture** (routing) — assigner dynamiquement chaque patch au block qui le separe le mieux de son contexte — serait superieure a un block unique. Cela necessite soit une meta-decision supervisee (quelle texture suis-je ?) soit un mecanisme d'attention appris.

Dans tous les cas, l'analyse one-vs-rest est plus informative que les metriques globales : elle revele quelles textures sont reellement separables et ou, independamment des autres.