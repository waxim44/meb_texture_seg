# Q1 — Intensité et outliers

**Configuration** : KEY=`stage_2_fpn` · SIL_THRESHOLD=0.0 · PCA=50d · N=816 patches

## Résultats par catégorie

| Cat | Nom | N_out | N_ok | μ_out | μ_ok | Δ | p-MW | sig |
|-----|-----|------:|-----:|------:|-----:|--:|-----:|-----|
| 1 | Totalement homogène | 3 | 38 | 62.6 | 107.6 | -45.0 | 3.75e-04 | *** |
| 3 | Faisceaux | 39 | 29 | 104.1 | 88.5 | +15.6 | 3.71e-03 | ** |
| 4 | Filaments | 15 | 34 | 117.3 | 120.0 | -2.7 | 6.41e-01 | ns |
| 5 | Stratifié rectiligne | 15 | 49 | 97.5 | 121.0 | -23.5 | 8.92e-03 | ** |
| 6 | Stratifié sinueux | 80 | 49 | 100.7 | 102.6 | -1.9 | 8.90e-01 | ns |
| 7 | Granuleux | 197 | 212 | 104.8 | 121.3 | -16.5 | 1.78e-17 | *** |
| 9 | Trou | 25 | 31 | 72.4 | 62.5 | +9.9 | 1.06e-01 | ns |

## Corrélation globale silhouette ↔ écart d'intensité

| Métrique | ρ / r | p-value |
|----------|------:|--------:|
| Spearman (luminosité) | -0.129 | 2.13e-04 |
| Spearman (contraste σ) | -0.204 | 4.05e-09 |
| Pearson (luminosité) | -0.169 | 1.29e-06 |

> Corrélation négative = outliers features = patches atypiques en intensité → Q1 confirmé.

## Verdict Q1 : **PARTIEL**

L'intensité explique partiellement les outliers.
Catégories sensibles (p<0.05) : Totalement homogène, Faisceaux, Stratifié rectiligne, Granuleux.
Catégories non sensibles : Filaments, Stratifié sinueux, Trou.
Pour les catégories sensibles, explorer Q2. Pour les autres, le problème est ailleurs.

- Catégories sensibles (p<0.05) : **Totalement homogène**, **Faisceaux**, **Stratifié rectiligne**, **Granuleux**
- Catégories non sensibles : Filaments, Stratifié sinueux, Trou

## Plots générés

| Fichier | Contenu |
|---------|---------|
| `intensite_outliers_vs_normaux.png` | Boxplot intensité outliers vs non-outliers par catégorie |
| `silhouette_vs_ecart_intensite.png` | Scatter silhouette ↔ écart d'intensité + tendance |
| `distribution_intensite_par_cat.png` | Distribution intensité avec outliers marqués |
| `heatmap_zscore_intensite.png` | Z-score d'intensité outliers vs non-outliers |
