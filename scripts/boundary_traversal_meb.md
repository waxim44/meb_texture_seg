# Traversée des frontières texturales — block_0 TextureSAM
## Méthode en pseudo-code langage naturel

---

## Objectif

Comprendre comment les features `block_0` de TextureSAM évoluent **en
traversant physiquement une frontière entre deux textures** dans l'image.
C'est une question critique pour la segmentation : est-ce que le modèle
encode une transition propre (A → A → B → B), une zone ambiguë au milieu
(A → A → ??? → B → B), ou un détour par une troisième région de l'espace
de features inattendue ?

---

## Pourquoi un forward pass sur l'image entière — et non une interpolation

### La tentante mais fausse interpolation

On pourrait calculer `v_A` et `v_B` (les features de deux patches
adjacents), puis simuler une trajectoire `v_A + t*(v_B - v_A)` pour
t ∈ [0, 1]. C'est rapide mais c'est une **ligne droite artificielle dans
l'espace de features** : elle ne passe pas nécessairement par la région
que le réseau "voit" réellement à la position physique de la frontière.

### Ce que fait réellement le réseau

Le réseau calcule ses features à partir des **pixels locaux et de leur
contexte voisin** (attention locale hiérarchique dans SAM2 Hiera). Ce
que le réseau "voit" à un pixel situé entre A et B dépend de l'image
réelle à cet endroit — pas d'une interpolation abstraite. La frontière
pourrait être nette, floue, ou passer par une zone de texture mixte.

Seule une véritable extraction de features à chaque position physique
révèle le comportement réel du réseau.

### Pourquoi un seul forward pass suffit

Le feature map `block_0` est une grille **spatiale** de taille 256×256×96
pour une image 1024×1024 : chaque cellule de la grille stocke les features
à une position spatiale précise. Un seul forward pass sur l'image entière
calcule **simultanément** les features à toutes les positions, y compris
les positions intermédiaires entre les patches A et B.

Il suffit donc de lire la valeur du feature map aux positions souhaitées
après le forward pass — sans relancer le réseau.

---

## Pseudo-code de la méthode

### Étape 1 — Recalcul des frontières depuis le HDF5

*Pour chaque image de la base de données :*
on considère toutes les paires de patches `(A, B)` de catégories
différentes. Deux patches sont **adjacents** si la distance entre leurs
bords est ≤ ADJ_TOL pixels — ce qui, pour des patches réguliers de 128 px,
correspond aux patches qui se touchent exactement.

On calcule l'**orientation de la frontière** : horizontale (H) si les
patches se touchent côte à côte, verticale (V) s'ils sont l'un au-dessus
de l'autre.

On attribue à chaque paire une **priorité** selon l'éloignement sémantique
des deux textures : Trou ↔ Granuleux (100), Granuleux ↔ Stratifié (90),
Faisceaux ↔ Trou (80), etc. On évite la paire Faisceaux ↔ Filaments
(priorité 0) car elle forme un continuum difficile à distinguer.

*On sélectionne les N_BOUNDARIES (défaut 5) frontières les plus prioritaires,
en prenant au maximum une frontière par paire de catégories.*

---

### Étape 2 — Entraînement PCA + GMM (tous les patches valides)

*Sur l'ensemble des patches valides de la base HDF5 :*

1. On réduit les 96 dimensions à PCA_DIM=10 dimensions (PCA ajustée sur
   tous les patches, pas uniquement l'entraînement — ici on analyse, on ne
   généralise pas).
2. On L2-normalise chaque vecteur (norme 1).
3. Pour chaque texture valide `c`, on entraîne une **gaussienne unique**
   (GaussianMixture, n_components=1, covariance_type='full',
   reg_covar=1e-1) — paramètres desserrés, issus de la grille
   `gmm_segmentation_test.py`.
4. On calibre le seuil θ_c = percentile 5 des log-vraisemblances sur les
   patches d'entraînement de `c`.
5. On calcule le **centroïde** de chaque catégorie : moyenne des features
   L2-normées en PCA-10d, puis re-normée.

---

### Étape 3 — Forward pass (un par image source)

*Pour chaque image contenant au moins une frontière sélectionnée :*

On fait **un seul** forward pass complet du réseau sur l'image originale
(redimensionnée à 1024×1024, normalisée ImageNet). Un hook sur `blocks[0]`
capture le feature map (H_feat × W_feat × 96), où H_feat = W_feat = 256
pour une image 1024×1024 (facteur ÷4 image→feature map).

**Facteur d'échelle** : pour une image originale de largeur `W` et hauteur
`H`, la position pixel `(x, y)` correspond à la position feature map
`(x * 256/W, y * 256/H)`.

---

### Étape 4 — Traversée et extraction des features

*Pour chaque frontière `(A, B)` :*

On calcule les centres des deux patches `(cx_A, cy_A)` et `(cx_B, cy_B)`
en pixels dans l'image originale.

On échantillonne N_STEPS=11 positions régulièrement espacées le long du
segment `center_A → center_B` : pour t ∈ {0, 0.1, …, 1.0},
`pt = center_A + t * (center_B - center_A)`.

À chaque position `pt`, on mappe `(px, py)` → `(px*sx, py*sy)` dans le
feature map (sx = W_feat/W_orig, sy = H_feat/H_orig). On extrait une
fenêtre de taille équivalente à un patch projeté (≈25×43 cellules pour
des images 1280×768). On fait la moyenne spatiale → vecteur 96d.
On L2-normalise le vecteur 96d (identique à la base HDF5), puis on
applique la PCA et on re-L2-normalise → vecteur de 10 dimensions.

---

### Étape 5 — 4 mesures sur la trajectoire

Pour chaque position t, on calcule :

**1. sim_A(t)** = similarité cosine entre `feat_t` et le centroïde de la
catégorie A. Doit décroître de t=0 à t=1 pour une transition propre.

**2. sim_B(t)** = similarité cosine entre `feat_t` et le centroïde de la
catégorie B. Doit croître de t=0 à t=1.

Le **croisement** `sim_A(t) = sim_B(t)` indique le point où le réseau
"bascule" d'une texture à l'autre. Un croisement proche de t=0.5 indique
une frontière propre ; un croisement absent ou très décalé indique une
transition abrupte ou une domination d'un côté.

**3. distance au segment feat [v_A, v_B]** :
On projette `feat_t` sur la droite passant par v_A et v_B dans l'espace
PCA-10d (v_A et v_B sont les features HDF5 des patches spécifiques). On
mesure la distance perpendiculaire (composante orthogonale à la droite).

- **Distance faible** : la trajectoire est proche de la ligne droite
  A→B. Block_0 interpole "proprement" — la zone de frontière est
  simplement un mélange des deux textures.
- **Distance forte** : la trajectoire fait un détour, passant par une
  région de l'espace de features éloignée de la ligne A-B. Cela peut
  signifier que le réseau "hallucine" une troisième texture à la frontière,
  ou que la zone de transition a des propriétés distinctes des deux
  côtés.

**4. prédiction GMM(t)** : on applique le classifieur GMM à `feat_t`.
La décision peut être : catégorie A, catégorie B, une autre catégorie
(texture hallucinée), ou **inconnu** (sous le seuil de toutes les GMM).
Un inconnu à t≈0.5 est le signe que la zone frontière est
**représentée dans un espace de features ambigu** pour le modèle.

---

## Figures générées

**Figure 1 — Courbes par frontière (`boundary_traversal_curves.png`)**

Pour chaque frontière : une colonne de deux panneaux.
- *Panneau du haut* : courbes `sim_A(t)` et `sim_B(t)` sur le même axe
  (croisement visible). En bas des courbes, une bande colorée indique la
  décision GMM à chaque pas : couleur de la catégorie prédite, ou gris
  foncé si "inconnu".
- *Panneau du bas* : courbe `distance_au_segment(t)`. Une zone pleine
  ombragée aide à visualiser l'amplitude du détour.

**Figure 2 — Trajectoires dans l'espace PCA-2 (`boundary_traversal_pca2.png`)**

Une PCA-2d est ajustée sur les features PCA-10d L2-normées de tous les
patches valides. On projette :
- les nuages de patches d'entraînement (fond coloré par catégorie)
- les 5 trajectoires de traversée (lignes colorées A→B)

Cette vue globale montre si les trajectoires traversent directement le
gradient entre les deux nuages, ou si elles font un détour visible.

**Figure 3 — Image source + segment diagnostique
(`boundary_traversal_diagnostic.png`)**

Pour les 2 premières frontières : l'image source originale avec le segment
de traversée superposé. Les positions t=0 et t=1 sont marquées par des
losanges, les positions intermédiaires par des cercles. La couleur de
chaque marqueur correspond à la décision GMM à cette position.

---

## Interprétation

### Transition propre ✅

- sim_A(t) décroît régulièrement, sim_B(t) croît régulièrement
- Croisement près de t=0.5
- Distance au segment faible tout au long du chemin
- GMM : prédiction A pour t<0.5, puis B pour t>0.5, éventuellement
  "inconnu" juste au niveau de la frontière (t≈0.5)

→ Block_0 encode bien les deux textures, et la frontière physique
correspond à une transition dans l'espace de features.

### Zone ambiguë à la frontière ⚠️

- Quelques positions "inconnu" près de t=0.5
- Distance au segment légèrement plus élevée à t=0.5
- Croisement visible mais pas parfaitement centré

→ La zone frontière est "floue" pour le modèle — la superposition des
deux textures crée une représentation intermédiaire que les GMM ne
reconnaissent pas clairement. Ce comportement est biologiquement attendu
aux interfaces texturales.

### Détour / 3e texture hallucinée ❌

- Distance au segment forte (pic à t=0.5)
- GMM prédit une catégorie TIERCE à t≈0.5 (ni A ni B)
- sim_A et sim_B restent simultanément faibles au milieu

→ Block_0 "voit" quelque chose d'inattendu à la frontière — une
propriété de l'image qui n'appartient ni à la texture A ni à la texture B.
Cela peut être un artefact d'acquisition (ligne de découpe entre zones),
une texture distincte visible à la frontière, ou une limitation du réseau.

### Collage (pas de transition)

- sim_A(t) reste haute jusqu'à t=0.9, puis chute brusquement
- GMM prédit A tout au long, puis B sur le dernier point
- Distance au segment faible mais croisement absent ou très tardif

→ Le réseau "colle" à la texture dominante et transite abruptement. Cela
peut refléter une asymétrie de la frontière (une texture empiète sur
l'autre), ou la prépondérance numérique d'une catégorie dans la région.

---

## Lien avec la segmentation

Si block_0 montre des transitions propres aux frontières, cela confirme
que le modèle peut naturellement segmenter les images en régions texturales
homogènes : les gradients de features sont alignés avec les gradients
physiques de l'image.

Si block_0 montre des détours ou des zones ambiguës, cela indique que la
segmentation sera difficile aux frontières : un classifieur de patches
"verra" un signal mixte dans cette zone, ce qui créera des erreurs de
classification systématiques aux bords des régions texturales.
