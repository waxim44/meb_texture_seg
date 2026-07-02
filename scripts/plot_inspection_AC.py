#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA = Path('/home/aidouni/meb_texture_seg/inspection_A_C/croisement_LP.csv')
OUT  = Path('/home/aidouni/meb_texture_seg/inspection_A_C')

df = pd.read_csv(DATA)

VERDICT_COLORS = {
    'RÉEL (A+LP)':    '#D62728',
    'RÉEL (C+LP)':    '#FF7F0E',
    'BÉNIN (A géom)': '#2CA02C',
    'BÉNIN (C intens)': '#1F77B4',
}

TEX_MARKERS = {
    'Tot.homogène': 'o',
    'Faisceaux':    's',
    'Filaments':    '^',
    'Strat.rect':   'D',
    'Strat.sin':    'P',
    'Granuleux':    'X',
    'Trou':         '*',
}

# ─── Figure 1 : scatter frac_A vs score_C_int ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

for _, row in df.iterrows():
    v    = row['verdict']
    tex  = row['texture_nom']
    size = 90 + 380 * (1 - row['loio_proba'])
    ax.scatter(row['frac_A'], row['score_C_int'],
               color=VERDICT_COLORS.get(v, 'gray'),
               marker=TEX_MARKERS.get(tex, 'o'),
               s=size, edgecolors='white', linewidths=0.7,
               alpha=0.88, zorder=3)

# Seuils de référence
ax.axvline(0.5, color='#888', lw=1.2, ls='--', alpha=0.5)
ax.axhline(0,   color='#888', lw=0.8, ls='-',  alpha=0.3)
ax.axhline(3.5, color='#888', lw=1,   ls=':',  alpha=0.45)
ax.axhline(-3.5,color='#888', lw=1,   ls=':',  alpha=0.45)

# Labels quadrants
kw = dict(fontsize=8.5, color='#666', style='italic', ha='left')
ax.text(0.02,  3.65, 'Zone C+ (intensité élevée)', **kw)
ax.text(0.02, -4.4,  'Zone C− (intensité basse)',  **kw)
ax.text(0.51, -4.4,  'Zone A\n(isolement géom.)', fontsize=8.5,
        color='#666', style='italic', ha='left')

ax.set_xlabel('frac_A  (fraction de blocs signalant l\'isolement géométrique)', fontsize=11)
ax.set_ylabel('score_C_int  (z-score intensité relative à l\'image)', fontsize=11)
ax.set_title('Carte de diagnostic — croisement outliers A × C × LP\n'
             'Taille ∝ 1 − P(LP correct)   |   Forme = texture   |   Couleur = verdict',
             fontsize=11, pad=10)
ax.set_xlim(-0.08, 1.15)
ax.set_ylim(-5.2, 5.0)
ax.grid(alpha=0.12)
ax.spines[['top', 'right']].set_visible(False)

verdict_handles = [mpatches.Patch(color=c, label=v)
                   for v, c in VERDICT_COLORS.items()]
tex_handles = [plt.scatter([], [], marker=m, color='#555', s=70, label=t)
               for t, m in TEX_MARKERS.items() if t in df['texture_nom'].values]

leg1 = ax.legend(handles=verdict_handles, title='Verdict', fontsize=9,
                 title_fontsize=9.5, loc='upper left', framealpha=0.92)
ax.add_artist(leg1)
ax.legend(handles=tex_handles, title='Texture', fontsize=9,
          title_fontsize=9.5, loc='upper right', framealpha=0.92)

plt.tight_layout()
plt.savefig(OUT / 'scatter_croisement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Scatter sauvé")

# ─── Figure 2 : barres empilées verdicts par texture ──────────────────────────
tex_order = df['texture_nom'].value_counts().index.tolist()
verdicts  = list(VERDICT_COLORS.keys())

counts = pd.crosstab(df['texture_nom'], df['verdict'])
for v in verdicts:
    if v not in counts.columns:
        counts[v] = 0
counts = counts[verdicts].reindex(tex_order).fillna(0)

fig, ax = plt.subplots(figsize=(10, 5.5))
bottom = np.zeros(len(tex_order))

for v in verdicts:
    vals = counts[v].values.astype(float)
    ax.bar(range(len(tex_order)), vals, bottom=bottom,
           color=VERDICT_COLORS[v], label=v, width=0.55,
           edgecolor='white', linewidth=0.6)
    for i, (val, bot) in enumerate(zip(vals, bottom)):
        if val > 0:
            ax.text(i, bot + val / 2, str(int(val)),
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')
    bottom += vals

ax.set_xticks(range(len(tex_order)))
ax.set_xticklabels(tex_order, fontsize=11)
ax.set_ylabel('Nombre de patches', fontsize=11)
ax.set_title('Répartition des verdicts par texture\n'
             '(patches croisés outliers A × C × LP)', fontsize=11)
ax.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.18)

plt.tight_layout()
plt.savefig(OUT / 'barres_verdicts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Barres sauvées")
