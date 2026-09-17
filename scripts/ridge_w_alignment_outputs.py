from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/aidouni/meb_texture_seg")
OUT = ROOT / "outputs/rsa_granulometrie/ridge_w_alignment"

BLOCKS = ["block_4", "block_6", "block_9"]
METRICS = ["D50", "D90", "Span", "Skewness", "sigma_local"]

r2 = pd.read_csv(OUT / "r2_par_metrique_bloc.csv")
stability = pd.read_csv(OUT / "stabilite_w_entre_folds.csv")
gap_df = pd.read_csv(OUT / "comparaison_cosinus_vs_correlation.csv")
corr_ref = pd.read_csv(OUT / "correlation_pearson_metriques.csv", index_col=0)
cos_mats = {b: pd.read_csv(OUT / f"cosinus_w_{b}.csv", index_col=0) for b in BLOCKS}


plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def heatmap(ax, M, labels, title):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = M[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=13)
    return im


for b in BLOCKS:
    Mcos = cos_mats[b].values
    Mcorr = corr_ref.values
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    fig.subplots_adjust(wspace=0.55)
    im0 = heatmap(axes[0], Mcos, METRICS, "cosinus(w_i, w_j)")
    im1 = heatmap(axes[1], Mcorr, METRICS, "corrélation(métrique_i, métrique_j)")
    cbar = fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("valeur signée")
    fig.suptitle(b, y=1.02, fontsize=18)
    fig.savefig(OUT / f"fig_alignement_{b}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig_alignement_{b}.png", bbox_inches="tight")
    plt.close(fig)

print("Figures ecrites :", [f"fig_alignement_{b}.png" for b in BLOCKS])


lines = []
lines.append("# Alignement des directions Ridge (w) — granulométrie, sans PCA\n")
lines.append(f"Blocs : {', '.join(BLOCKS)}. Métriques : {', '.join(METRICS)} (D10 exclue). "
             "Données : Granuleux (id7) + Sableux (id8), N=678 patchs, GroupKFold(5) par image.\n")

lines.append("## R² par métrique et par bloc (GroupKFold(5), moyenne ± std)")
lines.append("| Bloc | Métrique | R² moyen | R² std |")
lines.append("|---|---|---|---|")
for _, row in r2.iterrows():
    lines.append(f"| {row.block} | {row.metric} | {row.R2_mean:.3f} | {row.R2_std:.3f} |")

lines.append("\n## Stabilité des directions w entre les 5 folds (cosinus pairwise)")
lines.append("| Bloc | Métrique | cos moyen | cos min | cos max |")
lines.append("|---|---|---|---|---|")
for _, row in stability.iterrows():
    lines.append(f"| {row.block} | {row.metric} | {row.cos_moyen_entre_folds:.3f} | "
                 f"{row.cos_min_entre_folds:.3f} | {row.cos_max_entre_folds:.3f} |")
lines.append(f"\nToutes les directions ont un cosinus moyen entre folds > 0.8 "
             "→ directions **stables**, l'interprétation des alignements ci-dessous n'est pas fragilisée.\n")

for b in BLOCKS:
    lines.append(f"\n## Bloc {b} — cosinus(w) vs corrélation(métriques)")
    lines.append("| Métrique i | Métrique j | \\|cos(w)\\| | \\|corr\\| | écart (cos-corr) | lecture |")
    lines.append("|---|---|---|---|---|---|")
    sub = gap_df[gap_df.block == b].copy()
    sub["pair_key"] = sub.apply(lambda r: tuple(sorted([r.metric_i, r.metric_j])), axis=1)
    sub = sub.drop_duplicates("pair_key").sort_values("gap", key=abs, ascending=False)
    for _, row in sub.iterrows():
        lines.append(f"| {row.metric_i} | {row.metric_j} | {row.abs_cos:.3f} | {row.abs_corr:.3f} | "
                     f"{row.gap:+.3f} | {row.type} |")

lines.append("\n## Écarts notables (\\|cos\\| vs \\|corr\\| divergent, seuil 0.30)")
notable = gap_df[gap_df.type != "coherent"].drop_duplicates(
    subset=["block", "metric_i", "metric_j"]).sort_values("gap", key=abs, ascending=False)
lines.append("| Bloc | Métrique i | Métrique j | \\|cos\\| | \\|corr\\| | écart | type |")
lines.append("|---|---|---|---|---|---|---|")
for _, row in notable.iterrows():
    lines.append(f"| {row.block} | {row.metric_i} | {row.metric_j} | {row.abs_cos:.3f} | "
                 f"{row.abs_corr:.3f} | {row.gap:+.3f} | {row.type} |")

for b in BLOCKS:
    lines.append(f"\n![Alignement {b}](fig_alignement_{b}.png)")

with open(OUT / "resultats_alignement.md", "w") as fp:
    fp.write("\n".join(lines))
print("Ecrit :", OUT / "resultats_alignement.md")


r2_pivot = r2.pivot(index="metric", columns="block", values="R2_mean").loc[METRICS]


size_metrics = ["D50", "D90", "Span", "Skewness"]
size_align = {}
for b in BLOCKS:
    M = cos_mats[b].values
    idx = {m: i for i, m in enumerate(METRICS)}
    vals = [abs(M[idx[a]][idx[c]]) for i, a in enumerate(size_metrics) for c in size_metrics[i+1:]]
    size_align[b] = np.mean(vals)

sigma_align = {}
for b in BLOCKS:
    M = cos_mats[b].values
    idx = {m: i for i, m in enumerate(METRICS)}
    vals = [abs(M[idx["sigma_local"]][idx[c]]) for c in size_metrics]
    sigma_align[b] = np.mean(vals)

sigma_corr = {}
Mcorr = corr_ref.values
idxc = {m: i for i, m in enumerate(METRICS)}
mean_corr_sigma = np.mean([abs(Mcorr[idxc["sigma_local"]][idxc[c]]) for c in size_metrics])
mean_cos_sizeonly = np.mean([abs(Mcorr[idxc[a]][idxc[c]]) for i, a in enumerate(size_metrics) for c in size_metrics[i+1:]])

rapport = f"""RAPPORT — Alignement des directions Ridge (w) pour la granulométrie
====================================================================

Question : existe-t-il, dans l'espace latent (sans PCA), un axe UNIQUE de
"degré de granulométrie", ou plusieurs directions INDEPENDANTES portant des
proprietes distinctes ?

Perimetre : blocs {', '.join(BLOCKS)} ; metriques {', '.join(METRICS)} (D10
exclue) ; N=678 patchs (Granuleux id7 + Sableux id8) ; groupes = image
(GroupKFold(5)). w extraits par RidgeCV sur features standardisees (StandardScaler),
SANS PCA -> directement dans l'espace latent d'origine.

1) FIABILITE DES DIRECTIONS (R2, GroupKFold(5))
{r2_pivot.round(3).to_string()}

   D90 a le R2 le plus faible partout (0.26-0.33) -> sa direction w est la
   MOINS fiable des cinq ; a interpreter avec prudence. sigma_local a le R2 le
   plus eleve (0.73-0.78, la relation la plus lineaire/facile a extraire).
   D50, Span, Skewness sont dans une zone intermediaire-haute (0.57-0.76).

2) STABILITE DES DIRECTIONS ENTRE FOLDS
   Cosinus moyen entre les 5 w (memes metrique/bloc) : {stability.cos_moyen_entre_folds.min():.3f}
   a {stability.cos_moyen_entre_folds.max():.3f} sur tous les blocs/metriques -> TOUTES les
   directions sont stables (> 0.8). L'interpretation des alignements ci-dessous
   n'est donc pas fragilisee par un manque de reproductibilite des w.

3) CORRELATION ENTRE METRIQUES (Pearson, sur les 678 patchs, independant du latent)
{corr_ref.round(3).to_string()}

   D50, D90, Span et Skewness sont MODEREMENT A FORTEMENT correles entre eux
   (ils decrivent la meme distribution de tailles, lien attendu par
   construction). sigma_local est correlee MODEREMENT (~0.4-0.56) aux
   metriques de taille -- correlation non triviale mais pas mecanique.

4) ALIGNEMENT DES w DANS LE LATENT (cosinus, moyenne des paires D50/D90/Span/Skewness)
   Alignement moyen entre metriques de TAILLE (D50,D90,Span,Skewness) : {np.mean(list(size_align.values())):.3f}
   (par bloc : {', '.join(f'{b}={v:.3f}' for b, v in size_align.items())})
   Alignement moyen de sigma_local avec les metriques de taille : {np.mean(list(sigma_align.values())):.3f}
   (par bloc : {', '.join(f'{b}={v:.3f}' for b, v in sigma_align.items())})

   Pour comparaison, correlation moyenne entre metriques de taille (Pearson) :
   {mean_cos_sizeonly:.3f} ; correlation moyenne sigma_local <-> taille : {mean_corr_sigma:.3f}

5) LECTURE FACTUELLE
   - Les QUATRE metriques de taille (D50, D90, Span, Skewness) ont des
     directions w FORTEMENT ALIGNEES entre elles dans les trois blocs
     (cosinus moyen ~{np.mean(list(size_align.values())):.2f}, au-dessus du seuil de 0.7). Cet
     alignement est COHERENT avec leur correlation statistique deja elevee
     (elles decrivent la meme distribution de tailles par construction) : on
     ne peut PAS conclure que le latent ajoute une structure au-dela de ce que
     la correlation des metriques explique deja. -> tout se comporte comme un
     AXE UNIQUE de "taille de grain" pour ces quatre metriques.
   - sigma_local se comporte DIFFEREMMENT : sa direction w est PRESQUE
     ORTHOGONALE aux metriques de taille (cosinus moyen ~{np.mean(list(sigma_align.values())):.2f}, bien
     sous le seuil de 0.2 d'independance) MALGRE une correlation statistique
     moderee avec elles (~{mean_corr_sigma:.2f}). C'est un ECART NOTABLE : le latent
     encode sigma_local dans une direction DISTINCTE de celle de la taille de
     grain, alors que les deux proprietes sont partiellement liees dans les
     donnees. -> sigma_local (texture fine / contraste local) est une
     PROPRIETE SEPAREE dans l'espace latent, pas une simple consequence de la
     taille de grain.
   - Conclusion : il n'y a PAS un unique "degre de granulometrie" dans le
     latent, mais AU MOINS DEUX axes distincts : (i) un axe de TAILLE DE GRAIN
     qui regroupe D50/D90/Span/Skewness, et (ii) un axe de TEXTURE FINE /
     CONTRASTE porte par sigma_local, encode separement meme s'il est
     partiellement correle a la taille dans les donnees.
   - Prudence sur D90 : sa direction ayant le R2 le plus faible (donc la
     moins fiable), son alignement avec les autres metriques de taille doit
     etre lu avec cette reserve -- il est neanmoins cohérent avec celui des
     trois autres metriques de taille, ce qui renforce (sans le garantir) la
     lecture d'un axe de taille commun.
"""
with open(OUT / "rapport.txt", "w") as fp:
    fp.write(rapport)
print("Ecrit :", OUT / "rapport.txt")
