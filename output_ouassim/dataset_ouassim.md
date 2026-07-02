# Dataset Ouassim — Description

## Images

| Propriété        | Valeur                        |
|:-----------------|:------------------------------|
| Nombre d'images  | 60                            |
| Format           | TIFF grayscale                |
| Résolution       | 768 × 1280 px                 |
| Type             | uint8 (valeurs 0–255)         |
| Modalité         | Microscopie Électronique à Balayage (MEB) |
| Acquisition      | Valves cardiaques — ZigZag scan |

---

## Patches annotés

Annotés manuellement via PatchTagger. Chaque patch = 128 × 128 px extrait de l'image source.

| #  | Texture                  | Patches | Images sources |
|---:|:-------------------------|--------:|---------------:|
|  1 | Totalement homogène      |      41 |              9 |
|  2 | Plutôt homogène          |     848 |             36 |
|  3 | Faisceaux                |      68 |             10 |
|  4 | Filaments                |      49 |              8 |
|  5 | Stratifié rectiligne     |      64 |             10 |
|  6 | Stratifié sinueux        |     129 |             14 |
|  7 | Granuleux                |     409 |             21 |
|  8 | Sableux                  |     269 |             19 |
|  9 | Trou                     |      56 |             16 |
| 10 | Bactéries                |     135 |              8 |
| 11 | Cellule                  |      14 |              4 |
| 12 | Calcification            |      72 |              8 |
| 13 | Nd (non défini)          |    1429 |             48 |
|    | **Total**                | **3583**| **60**         |

---

## Sous-ensemble utilisé pour l'analyse LP

Textures retenues : **1, 3, 4, 5, 6, 7, 9** (exclut Nd, Plutôt homogène, Sableux, Bactéries, Cellule, Calcification)

| Texture               | Patches |
|:----------------------|--------:|
| Totalement homogène   |      41 |
| Faisceaux             |      68 |
| Filaments             |      49 |
| Stratifié rectiligne  |      64 |
| Stratifié sinueux     |     129 |
| Granuleux             |     409 |
| Trou                  |      56 |
| **Total**             | **816** |

---

## Remarques

- **Nd (1429 patches)** et **Plutôt homogène (848 patches)** représentent à eux deux 63 % du corpus total — exclus car trop hétérogènes sémantiquement.
- **Granuleux** domine le sous-ensemble utilisé (50 % des 816 patches).
- **Stratifié rectiligne** est concentré sur une seule image source à 75 % (`Ech5-ZigZag0005` : 48/64 patches).
- **Cellule (14 patches)** est trop peu représentée pour être incluse dans une évaluation fiable.
