# Test d'invariance géométrique — block_0 TextureSAM
## Méthode en pseudo-code langage naturel

---

## Objectif

Mesurer si les features `block_0` de TextureSAM reconnaissent une texture
**quelle que soit son orientation** : rotation 90°, 180°, 270°, flip
horizontal ou vertical.

L'invariance à l'orientation est une **propriété définitionnelle de la
texture** : un matériau granuleux reste granuleux qu'on le regarde de face
ou de côté. Si block_0 est invariant, il encode bien la texture en tant que
telle, pas juste l'apparence d'une image spécifique.

---

## Pourquoi transformer l'IMAGE ENTIÈRE — pas le patch seul

C'est le point méthodologique le plus important de ce test.

**Raison 1 — Le contexte global**
SAM2 Hiera Small utilise de l'attention locale hiérarchique : les features
d'un token dépendent de ses voisins dans la fenêtre d'attention. Si on
extrait un patch isolé de 128×128 pixels et qu'on le passe seul dans le
réseau, ses features sont calculées sans voisinage réel. Le réseau "voit"
des bords artificiels partout autour du patch, ce qui fausse les features.
En revanche, les features dans la base HDF5 ont été extraites de l'image
entière : chaque patch bénéficiait du contexte de ses voisins réels.

**Raison 2 — L'upsampling altère la texture**
Un patch de 128×128 pixels devrait être redimensionné à 1024×1024 pour
entrer dans le réseau. Ce ×8 upsampling dénature complètement la texture
(artefacts d'interpolation, perte des fréquences hautes). La comparaison
avec les features HDF5 (extraites sans upsampling) serait impossible.

**La solution correcte** : on transforme l'image entière, on fait le
forward pass complet sur l'image transformée (1024×1024 après resize
standard), et on calcule la nouvelle position du patch dans l'image
transformée pour extraire ses features.

---

## Pseudo-code de chaque étape

### Étape 1 — Choisir les patches à analyser

*Pour chaque texture valide :*
on tire aléatoirement jusqu'à N=50 patches depuis la base HDF5.
On les groupe par image source pour n'ouvrir chaque image qu'une fois.

### Étape 2 — Features de référence

*Pour chaque patch sélectionné :*
on lit directement son vecteur block_0 depuis la base HDF5 (96 dimensions,
déjà la moyenne sur la région du patch dans l'image originale).
On le L2-normalise (norme 1) : c'est la feature de référence.

### Étape 3 — Pour chaque image, pour chaque transformation

*On ouvre l'image originale une fois.*
*Puis pour chacune des 5 transformations :*

**a) On transforme l'image entière** via numpy (rotation ou flip du tableau
pixels). Pour rot90 sens horaire : on fait un rot90 avec k=3. Pour flip
horizontal : on inverse les colonnes. L'image transformée peut avoir des
dimensions différentes de l'originale (pour rot90/270, largeur et hauteur
s'échangent).

**b) On passe l'image transformée dans le réseau** (resize → 1024×1024 →
normalisation ImageNet → forward pass). Un hook sur block_0 capture le
feature map spatial (H_feat × W_feat × 96).

**c) Pour chaque patch de cette image :**

On calcule la nouvelle position du patch dans l'image transformée. Pour
cela, on applique la même transformation géométrique aux 4 coins du
rectangle [x1,y1,x2,y2], puis on prend le min/max des nouvelles
coordonnées pour obtenir le nouveau rectangle.

On projette ce rectangle dans le feature map : position dans le feature
map = position dans l'image × (taille du feature map / taille de l'image).
On extrait la région correspondante et on la moyenne → vecteur 96d.
On le L2-normalise.

On calcule la **similarité cosine** entre la feature de référence (HDF5,
depuis l'image originale) et la feature transformée.

---

## Recalcul de la position après transformation

Pour une image de largeur W et hauteur H, et un point (x, y) (x = colonne,
y = ligne) :

| Transformation | Nouvelle position (x', y') | Nouvelles dims |
|---|---|---|
| Rotation 90° ↻ | (H−y, x) | largeur=H, hauteur=W |
| Rotation 180° | (W−x, H−y) | inchangées |
| Rotation 270° ↻ | (y, W−x) | largeur=H, hauteur=W |
| Flip horizontal | (W−x, y) | inchangées |
| Flip vertical | (x, H−y) | inchangées |

Pour un rectangle [x1,y1,x2,y2] : on transforme les 4 coins, puis on
prend x_min/x_max et y_min/y_max pour retrouver le nouveau rectangle.

**Attention pour rot90/270 :** largeur et hauteur s'échangent dans l'image
résultante — il faut utiliser les nouvelles dimensions pour le mapping
vers le feature map.

---

## Étape 4 — Agréger les résultats

*Pour chaque transformation :* on calcule la similarité cosine moyenne sur
tous les patches et toutes les catégories (± écart-type).

*Pour chaque paire (texture, transformation) :* on calcule la moyenne des
similarités cosines des N patches de cette texture.

On identifie :
- quelle transformation perturbe le plus (similarité globale la plus basse)
- quelle texture est la plus / moins invariante

---

## Figures générées

**Figure 1 — Barplot par transformation**
Un barplot (avec barre d'erreur = ±std) de la similarité cosine moyenne
pour chacune des 5 transformations. Lignes de référence à 1.0 (invariance
parfaite), 0.9 (seuil invariant), 0.75 (seuil sensible).

**Figure 2 — Heatmap texture × transformation**
Une grille 7 textures × 5 transformations, annotée avec la similarité
cosine moyenne. Rouge = sensible, vert = invariant.

**Figure 3 — Diagnostic vignettes**
Pour un patch exemple par catégorie : le patch original (recadré depuis
l'image), puis le patch recadré depuis l'image transformée, avec la
similarité cosine en titre (vert ≥ 0.9, orange ≥ 0.75, rouge < 0.75).
Permet de vérifier visuellement que la transformation est bien appliquée.

---

## Interprétation des résultats

### Similarité élevée (> 0.9) : block_0 est invariant ✅

Le réseau encode la même représentation interne pour la texture, quelle que
soit l'orientation de l'image. C'est la signature d'un encodeur qui capte
des propriétés intrinsèques (statistiques locales, motifs récurrents), pas
juste l'orientation absolue.

### Similarité intermédiaire (0.75–0.9) : partiellement invariant ⚠️

block_0 change partiellement sa représentation. Cela peut être un artefact
du preprocessing (l'image est redimensionnée à 1024×1024, ce qui peut
introduire une légère distorsion si elle n'est pas carrée), ou une vraie
sensibilité partielle du réseau à l'orientation.

### Similarité faible (< 0.75) : sensible ❌

block_0 encode l'orientation comme une propriété discriminante. Pour les
textures fortement orientées, c'est un résultat intéressant (voir section
suivante), pas nécessairement un échec.

### Cas particulier : les textures ORIENTÉES

Les textures **Faisceaux**, **Filaments**, **Stratifié rectiligne** et
**Stratifié sinueux** ont une orientation spatiale intrinsèque. Un patch
de fibres horizontales N'EST PAS le même que des fibres verticales : ce
sont potentiellement deux états biologiques différents.

Si block_0 est sensible à la rotation pour ces textures mais pas pour
**Granuleux** ou **Trou** (isotropes par nature), c'est cohérent avec
la physique du matériau.

Un résultat idéal serait :
- Textures isotropes (Granuleux, Trou) → invariantes à toutes les
  rotations et flips
- Textures orientées (Faisceaux, Filaments, Stratifiés) → sensibles aux
  rotations 90°/270° mais potentiellement invariantes au flip (selon la
  symétrie de la texture)
- Flip vertical ≠ flip horizontal pour les stratifiés (asymétrie
  naturelle haut/bas)

Ce pattern confirmerait que block_0 encode l'orientation comme information
pertinente pour les textures orientées — ce qui est biologiquement justifié.
