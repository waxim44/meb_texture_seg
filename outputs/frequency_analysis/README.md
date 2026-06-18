# Analyse fréquentielle — Textures MEB (block_0)

## Objectif

Tester si les features de block_0 encodent l'information de
**fréquence spatiale**, qui est la définition classique de la
texture (distribution spatiale répétée de motifs).

## Méthode

### Étape 1 — Profil fréquentiel par patch

Pour chaque patch :
- On calcule la FFT 2D (transformée de Fourier discrète)
- On obtient le spectre de puissance : énergie par fréquence
- On centre le spectre (fftshift) : basses fréquences au centre
- On moyenne l'énergie par anneaux concentriques :
  - Anneau central = basses fréquences (motifs grossiers)
  - Anneaux externes = hautes fréquences (motifs fins)
- On log-normalise le profil pour compresser la dynamique
- Résultat : vecteur de 20 valeurs (énergie par bande)

### Étape 2 — Signature fréquentielle par texture

Pour chaque texture :
- On moyenne les profils fréquentiels de ses patches (± std)
- Chaque texture a une signature caractéristique :
  - Granuleux → forte énergie haute fréquence
  - Trou → énergie concentrée basses fréquences
  - Filaments → pics à des fréquences intermédiaires

### Étape 3 — Corrélation features ↔ fréquences (CCA)

L'analyse de corrélation canonique (CCA) cherche les directions
dans l'espace block_0 (réduit à 20d par PCA) et dans
l'espace fréquentiel qui sont maximalement corrélées.

- Corrélation canonique élevée (> 0.5) → block_0 encode les fréquences
- On calcule 10 composantes canoniques

### Étape 4 — Prédictibilité (Ridge regression)

On entraîne une régression Ridge (5-fold CV) :
  block_0 (PCA-20d) → profil fréquentiel (20 bandes)

- R² élevé pour une bande → block_0 encode bien cette fréquence
- R² faible → block_0 ne contient pas cette information fréquentielle

## Interprétation

- **Signatures distinctes** entre textures → la fréquence spatiale
  discrimine les textures (cohérent avec la définition classique)
- **CCA élevée** → block_0 encode l'information fréquentielle
- **R² élevé** → on peut reconstruire le contenu fréquentiel
  depuis block_0, prouvant qu'il agit comme un analyseur
  fréquentiel appris (lien avec les filtres de Gabor classiques)
- **Lien Gabor** : les filtres de Gabor (utilisés dans compare_descriptors)
  sont des filtrages fréquentiels orientés ; si block_0 corrèle avec FFT,
  les deux approches capturent la même information fondamentale

## Fichiers générés

| Fichier | Description |
|---|---|
| `frequency_signatures.png` | Signature FFT par texture (profil moyen ± std) |
| `cca_freq_features.png` | Corrélations canoniques block_0 ↔ FFT |
| `frequency_predictability.png` | R² par bande de fréquence |
| `frequency_results.pkl` | Profils, résultats CCA et R² sauvegardés |
| `README.md` | Ce fichier |

## Résultats

- Patches analysés : 816
- Catégories : 7
- Corrélation canonique max   : 0.932
- Corrélation canonique moy.  : 0.444
- R² moyen (prédiction FFT)   : 0.095
- R² max   (bande 11)      : 0.183

### Signatures fréquentielles moyennes par texture

- Totalement homogène       : pic à la bande 0 (0.00 fréq. norm.)
- Faisceaux                 : pic à la bande 0 (0.00 fréq. norm.)
- Filaments                 : pic à la bande 0 (0.00 fréq. norm.)
- Stratifié rectiligne      : pic à la bande 0 (0.00 fréq. norm.)
- Stratifié sinueux         : pic à la bande 0 (0.00 fréq. norm.)
- Granuleux                 : pic à la bande 0 (0.00 fréq. norm.)
- Trou                      : pic à la bande 0 (0.00 fréq. norm.)
