# GMM Segmentation Test — Méthode en pseudo-code langage naturel

## Objectif

Tester si les **modèles de mélange gaussien (GMM)** entraînés sur des patches
de textures MEB se **généralisent** à des images jamais vues lors de
l'entraînement, et en particulier mesurer l'effet du **desserrage des
gaussiennes** via la réduction de la dimension PCA et la régularisation de
covariance.

Chaque texture est représentée dans l'espace des features `block_0` de
TextureSAM (SAM2 Hiera Small, 96 dimensions) par une **gaussienne unique**
(unimodale, covariance full). On veut vérifier que cette gaussienne capture
bien la structure intrinsèque de la texture, pas juste les images
d'entraînement.

---

## Étape 1 — Choisir les images de test

*Pour chaque image présente dans la base de données HDF5 :*

On compte le nombre de catégories de texture distinctes (valides, i.e.
présentes dans la liste autorisée) et le nombre de patches annotés.

*Puis on trie les images par ordre décroissant de diversité :*
d'abord le nombre de catégories, puis le nombre de patches
(pour départager à diversité égale).

*On retient les 3 images en tête de ce classement.*

> Ces images sont mises de côté : elles ne serviront jamais à l'entraînement.

---

## Étape 2 — Tester une grille de configurations (desserrage des GMM)

*On exclut strictement tous les patches appartenant aux 3 images test.*

Pour chaque paire `(PCA_DIM, REG_COVAR)` de la grille
`{10, 20} × {1e-3, 1e-2, 1e-1}` :

1. On applique une **PCA à PCA_DIM dimensions** (ajustée sur le train, jamais
   sur le test).
2. On **L2-normalise** les vecteurs projetés (chaque vecteur a une norme 1).
3. *Pour chaque texture valide :*
   on ajuste une **gaussienne unique** (GaussianMixture, n_components=1,
   covariance_type='full', **reg_covar=REG_COVAR**) sur les patches de cette
   texture dans l'espace PCA-dim normalisé.
4. On calibre le seuil θ_c = percentile 5 des log-vraisemblances de ses
   propres patches d'entraînement (seuil PAR texture, valeur fixe p=5).
5. On prédit les patches test avec ce GMM et ce seuil.
6. On mesure le % d'inconnus et l'accuracy sur les patches reconnus.

On obtient 6 résultats (2 PCA_DIM × 3 REG_COVAR) que l'on compare.

---

## Étape 3 — Tableau comparatif de la grille

*Pour les 6 configurations, on affiche :*
- le pourcentage de patches "inconnus"
- l'accuracy équilibrée sur les patches reconnus

*On sélectionne la "meilleure" config :*
- en priorité : les configs dont l'accuracy dépasse 80%,
  parmi lesquelles on prend celle avec le moins d'inconnus
- sinon : celle qui maximise le produit accuracy × couverture
  (couverture = 100% - % inconnus), compromis entre les deux objectifs

*On sauve ce tableau en txt et csv.*

---

## Étape 4 — Prédire à l'échelle PATCH (pas pixel par pixel)

**Pourquoi à l'échelle patch ?**
Les GMM ont été entraînés sur des **vecteurs moyens par patch** (un seul
vecteur de 96d représente tout un patch). Si on tentait de prédire
pixel par pixel (en appliquant le modèle sur des features extraites à
une position unique dans la carte de features spatiale), on observerait
un **décalage de distribution** : les features spatiales locales n'ont
pas la même distribution statistique que les moyennes de patch. Les
vraisemblances seraient faussées et le seuil calibré ne serait plus valide.

*Pour chaque image test, on récupère ses patches depuis le HDF5 :*
positions (x1, y1, x2, y2) et features block_0 préextraites (96d).

*Pour chaque patch `x` :*

1. On applique la PCA apprise à l'étape 2 (sans re-ajuster), puis
   on L2-normalise.
2. *Pour chaque texture valide `c` :* on calcule `log p(x|c)` via le GMM.
3. On choisit `c* = argmax_c log p(x|c)` (texture la plus probable).
4. On compare `log p(x|c*)` au seuil `θ_{c*}(p)` :
   - Si la vraisemblance est suffisante → **label = c\***
   - Sinon → **label = "inconnu"** (le patch ne ressemble à aucune texture
     de façon convaincante)

**Ce que signifie "inconnu"** : le patch test n'appartient pas à la région
haute-vraisemblance d'aucun GMM. Cela peut signifier que la texture est
absente de l'ensemble d'entraînement, que l'image test a des conditions
d'acquisition différentes, ou que la zone est à la frontière entre deux
textures.

---

## Étape 5 — Heatmaps de la grille

*On trace deux heatmaps côte à côte (lignes = PCA_DIM, colonnes = REG_COVAR) :*
- une pour le % d'inconnus (rouge = beaucoup d'inconnus)
- une pour l'accuracy sur les patches reconnus (vert = haute précision)

La meilleure config est marquée d'une étoile.

---

## Étape 6 — Visualiser la meilleure config par image test

*Pour chaque image test, une figure avec 3 panneaux :*

- **Image originale** en niveaux de gris
- **Vrais labels** : patches colorés selon leur texture réelle, dessinés
  comme rectangles semi-transparents aux positions des patches
- **Prédiction GMM** (meilleure config) :
  - Patches reconnus → couleur de la texture prédite
  - Patches "inconnus" → gris foncé distinct

---

## Étape 7 — Matrice de confusion (meilleure config)

Sur les patches reconnus des 3 images test, matrice de confusion normalisée
par ligne : quelle texture vraie est prédite comme quelle autre.

---

---

## Pourquoi les gaussiennes sont-elles trop pointues ? — Explication du desserrage

### Problème : PCA-50d + covariance full = vraisemblance qui s'effondre

Imaginons une gaussienne en 50 dimensions avec covariance full. Sa densité
de probabilité est proportionnelle à :

    exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))

En haute dimension, même un léger déplacement de `x` par rapport au centre
`μ` donne une valeur exponentielle très négative. Concrètement : un patch
d'une image jamais vue, même s'il ressemble beaucoup à la texture apprise,
peut se retrouver légèrement décalé dans l'espace PCA (variation
d'éclairage, contraste, angle de vue). Sa log-vraisemblance tombe alors
bien en dessous du seuil calibré sur les images d'entraînement. Le GMM le
rejette comme "inconnu" alors qu'il aurait été reconnu visuellement.

Ce phénomène s'appelle la **concentration de la mesure** en haute dimension :
les données tendent à se concentrer sur une "coquille" autour du centre de
la gaussienne, et toute variation cross-image sort de cette coquille.

### Solution 1 : réduire la dimension PCA

*On garde moins de dimensions (10 ou 20 au lieu de 50) :*

Avec moins de dimensions, la gaussienne vit dans un espace de dimension
réduite. La région de haute vraisemblance est plus large relativement à
la dispersion des données : un patch qui varie légèrement de son centre
reste plus souvent dans cette région. La gaussienne "couvre" mieux les
variations cross-image.

Risque : on perd de l'information discriminante. Des textures proches
peuvent devenir indistinguables.

### Solution 2 : reg_covar — élargir la gaussienne par régularisation

*On ajoute reg_covar × Identité à la matrice de covariance estimée :*

    Σ_régularisée = Σ_estimée + reg_covar × I

Cela a deux effets :
1. **Stabilité numérique** : Σ est toujours inversible même avec peu de
   données (évite les gaussiennes "infiniment pointues" dans certaines directions).
2. **Élargissement** : on ajoute une composante isotrope à la gaussienne,
   ce qui l'élargit dans toutes les directions. Un patch légèrement décalé
   du centre a maintenant une vraisemblance moins catastrophique.

Concrètement : reg_covar = 1e-1 est 100× plus large que reg_covar = 1e-3.
La gaussienne tolère 100× plus de variation avant de rejeter un patch.

On **reste en vraisemblance absolue** (seuil par texture calibré sur les
vrais patches d'entraînement) — on desserre juste la gaussienne pour
qu'elle accepte plus de variation cross-image.

### Le compromis : desserrer trop peu vs trop

| Situation | % inconnus | Accuracy | Interprétation |
|-----------|-----------|----------|----------------|
| Trop pointu (dim élevée, reg faible) | très élevé (87-97%) | bonne sur reconnus | tout est rejeté, inutilisable |
| Trop lâche (dim faible, reg élevé) | très faible | potentiellement mauvaise | tout est accepté mais avec confusions |
| Équilibré | modéré | bonne | bon compromis couverture/précision |

La grille `PCA_DIM ∈ {10, 20} × REG_COVAR ∈ {1e-3, 1e-2, 1e-1}` permet
d'identifier la zone d'équilibre entre ces deux extrêmes.

---

## Interprétation des résultats

### % d'inconnus élevé

Un fort pourcentage de patches "inconnus" signifie que les images test
sont **difficiles à couvrir** par les GMM entraînés. Causes possibles :
- Variabilité intra-texture plus grande dans les images test
- Conditions d'acquisition différentes (zoom, contraste, échantillon)
- Textures qui n'existent pas (ou peu) dans le train

Un % d'inconnus élevé n'est pas forcément mauvais s'il est voulu : c'est
le système qui dit "je ne suis pas sûr".

### Accuracy élevée sur les patches reconnus

Si l'accuracy est haute, cela signifie que **lorsque le GMM ose classer,
il classe correctement**. C'est le signe que les gaussiennes capturent
bien les propriétés discriminantes de chaque texture et qu'elles
généralisent au-delà des images d'entraînement.

La combinaison idéale est : **peu d'inconnus ET haute accuracy** — ce qui
indiquerait une généralisation robuste. Dans la pratique, il existe un
compromis : un seuil bas (p=1%) laisse passer presque tout mais avec
potentiellement plus d'erreurs ; un seuil strict (p=25%) est très sélectif
mais les prédictions acceptées sont plus fiables.

### Lecture de la courbe seuil

La courbe montre le compromis couverture/précision en fonction du seuil :
- À gauche (p=1%) : presque tout est reconnu → couverture maximale
- À droite (p=25%) : seuls les patches très proches du centre de leur
  gaussienne sont reconnus → précision maximale

Le "meilleur" seuil est celui qui maximise l'accuracy sur les patches
reconnus, ce qui correspond au seuil recommandé pour le déploiement.

---

## Lien avec la théorie

Les GMM appris sur block_0 sont des **modèles génératifs** de textures. Ce
test évalue leur capacité à fonctionner comme **détecteurs de texture** :
"ce patch appartient-il à la distribution apprise pour cette texture ?"

Si la généralisation est bonne, cela confirme que block_0 de TextureSAM
encode des **propriétés intrinsèques des textures** (indépendantes des
images spécifiques d'entraînement), et pas seulement des artefacts
d'image ou des biais d'acquisition.
