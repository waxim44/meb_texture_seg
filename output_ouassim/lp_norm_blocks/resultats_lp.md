# Résultats Linear Probing — Texture × Normalisation × Block

**Setup :** split par image (seed 52, 70 % train / 30 % test)
**Blocks explorés :** 3–13 · **Normalisations :** baseline · gamma_0.7 · gamma_1.5 · zscore_image
**Critère :** meilleur acc_test avec gap train−test ≤ 0.25

---

| Texture            | N train | N test | Block | Transfo       | Acc train | Acc test  | Gap   |
|:-------------------|--------:|-------:|:-----:|:-------------:|----------:|----------:|:-----:|
| Homogène           |      22 |     19 |   6   | baseline      |      1.00 |  **0.95** |  0.05 |
| Faisceaux          |      45 |     23 |  12   | baseline      |      0.96 |  **0.74** |  0.22 |
| Filaments          |      30 |     19 |   6   | zscore_image  |      0.87 |  **0.68** |  0.18 |
| Strat. rectiligne  |      15 |     49 |   —   | —             |      0.00 |  **0.00** |   —   |
| Strat. sinueux     |      95 |     34 |   5   | baseline      |      0.86 |  **0.94** | −0.08 |
| Granuleux          |     276 |    133 |   4   | baseline      |      1.00 |  **0.97** |  0.03 |
| Trou               |      40 |     16 |   3   | zscore_image  |      0.93 |  **0.56** |  0.36 ⚠ |

---

**Notes**

- **Strat. rectiligne** — résultat non interprétable : 48/64 patches viennent d'une seule image tombée entièrement en test, laissant 15 patches d'entraînement.
- **Trou ⚠** — aucune combinaison ne respecte gap ≤ 0.25. Sur-apprentissage probable (16 patches test seulement).
- **Strat. sinueux** — gap négatif (−0.08) : texture cohérente, bien répartie entre les splits.
- **Baseline** domine 5/7 textures. zscore_image n'aide que sur Filaments et Trou.
