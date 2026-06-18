# Analyse TCAV — Textures MEB (block_0)

## Objectif

Comprendre comment block_0 de TextureSAM encode chaque texture
en extrayant un **Concept Activation Vector (CAV)** par texture :
une direction dans l'espace des features (96d) qui représente
le concept de cette texture.

Référence : Kim et al. (2018), "Interpretability Beyond Feature
Attribution: Quantitative Testing with Concept Activation
Vectors (TCAV)", ICML.

## Méthode

### Étape 1 — Extraire les CAV

Pour chaque texture c :
- On prend tous les patches de la base HDF5
- On étiquette : 1 si texture = c, 0 sinon (one-vs-rest)
- On entraîne une régression logistique (sans PCA, sur 96d bruts)
- Le vecteur de poids appris = direction du concept "texture c"
- On le normalise → CAV(c)

Le CAV est la direction dans l'espace 96d qui sépare le mieux
cette texture des autres.

### Étape 2 — Similarité entre CAV

Pour chaque paire de textures :
- Similarité cosine entre leurs CAV (déjà normalisés → produit scalaire)
- CAV proches → concepts texturaux encodés de façon similaire
- Révèle quelles textures partagent un encodage commun

### Étape 3 — Cartes de sensibilité spatiale

Pour l'image cible :
- On extrait la feature map block_0 (H × W × 96) via forward hook
- On applique le même StandardScaler (fitté sur la base)
- Pour chaque position spatiale, on projette son vecteur sur CAV(c)
- Score élevé = cette zone ressemble au concept "texture c"
- On upsampe vers la taille originale (INTER_LINEAR)

### Étape 4 — Test de steering causal

Pour vérifier que le CAV contrôle vraiment la texture :
- On prend des patches d'autres textures (échantillon de 100)
- On les déplace : x_scaled + alpha × CAV(c)  (alpha ∈ [-3, 3])
- On reclasse ces patches déplacés avec un classifieur multi-classe
- Si la proportion classée "c" augmente avec alpha positif
  → le CAV contrôle causalement la perception de la texture

### Étape 5 — Dimensions dominantes

Pour chaque CAV :
- |poids| par dimension (0–95 de block_0)
- Top-5 dimensions avec le poids absolu le plus élevé
- Identifie les dimensions clés de chaque concept textural

## Fichiers générés

| Fichier | Description |
|---|---|
| `cav_similarity.png` | Matrice de similarité cosine entre CAV |
| `tcav_sensitivity_maps.png` | Cartes de sensibilité par texture sur l'image cible |
| `tcav_steering.png` | Test de steering causal (α de -3 à +3) |
| `cav_dimensions.png` | Poids absolus |CAV| par dimension (heatmap) |
| `cavs.pkl` | CAV sauvegardés + scaler + top dimensions |
| `README.md` | Ce fichier |

## Interprétation

- **Similarité CAV élevée** entre deux textures → block_0 les encode
  de façon proche (ex : Faisceaux/Filaments attendus proches)
- **Steering efficace** (proportion monte avec alpha) → le concept
  est une direction linéaire manipulable dans l'espace feature
- **Dimensions dominantes partagées** entre textures → encodage
  distribué et polysémantique (normal pour un réseau profond)
- **CAV négatifs** (similarité négative) → concepts "opposés"
  dans l'espace feature (ex : texture homogène vs texture complexe)


## Résultats

### Qualité des CAV (accuracy one-vs-rest)

- Totalement homogène : 100.0%
- Faisceaux : 99.1%
- Filaments : 99.8%
- Stratifié rectiligne : 100.0%
- Stratifié sinueux : 100.0%
- Granuleux : 100.0%
- Trou : 100.0%

### Top-5 dimensions par texture

- Totalement homogène : dims [46, 50, 85, 13, 40]
- Faisceaux : dims [46, 80, 54, 28, 79]
- Filaments : dims [80, 46, 65, 20, 21]
- Stratifié rectiligne : dims [85, 57, 69, 28, 86]
- Stratifié sinueux : dims [57, 50, 31, 4, 51]
- Granuleux : dims [57, 45, 74, 46, 28]
- Trou : dims [13, 54, 23, 46, 60]

### Image cible

`310120-pat18-WholeMount-24.tif`
