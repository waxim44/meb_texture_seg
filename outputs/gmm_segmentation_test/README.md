# GMM Segmentation Test — Textures MEB

Test de généralisation des GMM de texture (block_0, TextureSAM SAM2 Hiera Small)
sur des images MEB JAMAIS VUES lors de l'entraînement.

## Images test sélectionnées (les 3 plus diverses)
- **070525-JPB-MEB-EIHNValves-Ech6-ZigZag0100.tif**
  - 5 catégories, 19 patches
  - Faisceaux, Filaments, Stratifié rectiligne, Stratifié sinueux, Granuleux
- **070525-JPB-MEB-EIHNValves-Ech5-ZigZag0051.tif**
  - 4 catégories, 27 patches
  - Faisceaux, Filaments, Stratifié rectiligne, Stratifié sinueux
- **070525-JPB-MEB-EIHNValves-Ech5-ZigZag0003.tif**
  - 4 catégories, 24 patches
  - Stratifié rectiligne, Stratifié sinueux, Granuleux, Trou

## Résultats (sur 70 patches test)

| Seuil | % Inconnus | Accuracy (reconnus) |
|-------|-----------|---------------------|
| p=1% | 87.1% | 100.0% |
| p=5% | 90.0% | 100.0% |
| p=10% | 91.4% | 100.0% |
| p=25% | 97.1% | 100.0% |

**Meilleur seuil** : p=1%
  - % inconnu : 87.1%
  - Accuracy équilibrée (patches reconnus) : 100.0%

## Fichiers générés

| Fichier | Description |
|---------|-------------|
| `gmm_test_<image>.png` | Comparaison vrais labels vs prédictions GMM par image |
| `gmm_threshold_curve.png` | Compromis % inconnu / accuracy selon le seuil |
| `gmm_confusion_matrix.png` | Matrice de confusion normalisée (meilleur seuil) |
| `gmm_results.pkl` | GMMs, PCA, seuils, métriques sauvegardées |
| `gmm_segmentation_test.md` | Documentation méthode en pseudo-code |
