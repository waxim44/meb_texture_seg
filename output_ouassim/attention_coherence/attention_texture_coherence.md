# Analyse de cohérence texturale dans l'attention globale de TextureSAM

## Objectif

Tester si l'attention globale de TextureSAM regroupe les patches de même texture :
un patch de catégorie *c* regarde-t-il préférentiellement les autres patches de la même catégorie *c* ?
La métrique centrale est le **ratio intra/inter** = attention moyenne vers patches de même texture
/ attention moyenne vers patches d'autres textures. Un ratio > 1 indique un regroupement.

## Pourquoi attention globale seulement

Dans Hiera Small (SAM2), les blocks à **fenêtre locale** (window_size > 0) ne peuvent voir
que leur voisinage immédiat. Un patch Granuleux ne voit pas les autres patches Granuleux
situés à l'autre bout de l'image. Seuls les **blocks 7, 10 et 13** ont `window_size = 0`
(attention globale) : chaque position peut attendre toutes les 4096 positions de la grille 64×64.

**Configuration** : 3 blocks × 4 heads = 12 configurations par texture.

## Démarche

1. **Extraction des poids Q, K** : hook sur la couche `qkv` (Linear) de chaque block global.
   Output shape : `(B, 4096, 1152)` → reshape → Q, K de forme `(1, 4, 4096, 96)`.

2. **Mapping patches → grille 64×64** : pour un patch annoté `(x_min, y_min, x_max, y_max)`
   en coordonnées image originale `(orig_H, orig_W)`, le centre est converti en position
   `(fy, fx)` dans la grille 64×64 avec `scale = 64 / orig_W` et `scale = 64 / orig_H`.

3. **Calcul du ratio intra/inter** : pour chaque query patch `q` de texture *c* :
   ```
   attn_row = softmax(Q[q] · Kᵀ / √96)   [4096 valeurs — softmax sur TOUT le contexte]
   intra = mean(attn_row[p] for p in patches_de_texture_c, p ≠ q)
   inter = mean(attn_row[p] for p in patches_d'autres_textures)
   ratio = intra / inter
   ```

## Résultats

**Images analysées** : 4 images Ouassim à 7 textures distinctes.

### Ratio meilleur par texture (max sur 12 configs)

| Texture | Meilleur ratio | (Block, Head) | Regroupée ? |
|---------|---------------|---------------|------------|
| Granuleux | 5515.781 | B7H1 | ✓ oui |
| Filaments | 1168.818 | B10H2 | ✓ oui |
| Trou | 122.232 | B10H2 | ✓ oui |
| Faisceaux | 11.108 | B7H1 | ✓ oui |
| Stratifié sinueux | 5.794 | B10H1 | ✓ oui |
| Totalement homogène | 0.000 | B7H0 | ✗ non |
| Stratifié rectiligne | 0.000 | B7H0 | ✗ non |

### Têtes les plus "texture-aware"

| Config | Ratio moyen (toutes textures) |
|--------|-------------------------------|
| B7H1 | 1023.001 ± 3151.349 |
| B7H3 | 382.721 ± 1115.044 |
| B10H2 | 382.157 ± 723.848 |
| B13H0 | 59.307 ± 136.169 |
| B13H3 | 10.778 ± 21.936 |
| B10H1 | 6.190 ± 4.853 |

**Textures faciles** (Homogène, Granuleux) : ratio moyen = **2.897**
**Textures difficiles** (Stratifié rectiligne) : ratio moyen = **1168.818**

## Conclusion

L'attention globale **regroupe 5/7 textures** (ratio > 1).
Le modèle encode partiellement la cohérence texturale même sans supervision explicite.

- **Tête la plus texture-aware** : `B7H1` (ratio moyen = 1023.001)
- **Textures regroupées** : Faisceaux, Filaments, Stratifié sinueux, Granuleux, Trou
- **Textures non regroupées** : Totalement homogène, Stratifié rectiligne

**Piste** : les têtes montrant ratio > 1 pourraient être exploitées comme signal de segmentation
faiblement supervisé, en propageant l'attention d'un patch annoté vers ses voisins texturaux.

**Lien avec les difficultés observées** : si Stratifié rectiligne a un faible ratio, l'attention
ne les distingue pas → cohérent avec le recall de 23% observé sur images Ouassim (block_4 LP).

## Fichiers générés

- `ratio_heatmap.png` — heatmap textures × 12 configs (vert > 1 = regroupe)
- `attention_maps_visuel.png` — cartes d'attention sur exemples
- `ratio_par_texture.png` — meilleur ratio par texture
- `heads_specialisees.png` — ratio moyen par (block, head)
