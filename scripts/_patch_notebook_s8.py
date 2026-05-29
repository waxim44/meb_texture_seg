"""
Patch notebooks/interactive_analysis.ipynb :
  - Cell 4 : ajoute 'features_fused' dans STATE
  - Cell 8 : ajoute hook encoder.neck pour features fusionnées
  - Ajoute Section 8 (3 nouvelles cellules)
"""
import json, re
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "interactive_analysis.ipynb"

nb = json.loads(NB_PATH.read_text())

# ─── helpers ──────────────────────────────────────────────────────────────────

def src(*lines):
    """Convertit des lignes en source notebook (liste avec \\n)."""
    out = []
    for l in lines:
        out.append(l + "\n")
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out

def code_cell(*lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src(*lines)}

def md_cell(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": src(*lines)}

# ─── PATCH Cell 4 : ajoute 'features_fused' dans STATE ───────────────────────

cell4 = nb["cells"][4]
raw4  = "".join(cell4["source"])
raw4  = raw4.replace(
    "    'attn_cache': {},\n}",
    "    'attn_cache'    : {},\n    'features_fused': {},\n}"
)
cell4["source"] = [l + "\n" for l in raw4.splitlines()]
cell4["source"][-1] = cell4["source"][-1].rstrip("\n")

# ─── PATCH Cell 8 : ajoute hook neck + stockage features_fused ───────────────

cell8 = nb["cells"][8]
raw8  = "".join(cell8["source"])

# 1. Ajouter fused_cache = {} juste après attn_cache = {}
raw8 = raw8.replace(
    "    attn_cache = {}\n    hooks = []",
    "    attn_cache  = {}\n    fused_cache = {}\n    hooks = []"
)

# 2. Ajouter le hook neck juste avant "# --- forward ---"
neck_hook_block = (
    "    # --- hook FPN neck (features fusionnees apres top-down) ---\n"
    "    def _nfhook(m, inp, out):\n"
    "        out_list, _ = out\n"
    "        # out_list[0]=Stage1(256x256) out_list[1]=Stage2(128x128)\n"
    "        # out_list[2]=Stage3(64x64) <- seul niveau avec fusion reelle\n"
    "        fused_cache['Stage 1 fus'] = out_list[0].detach().cpu()\n"
    "        fused_cache['Stage 2 fus'] = out_list[1].detach().cpu()\n"
    "        fused_cache['Stage 3 fus'] = out_list[2].detach().cpu()\n"
    "    hooks.append(encoder.neck.register_forward_hook(_nfhook))\n"
)
raw8 = raw8.replace(
    "    # --- forward ---",
    neck_hook_block + "    # --- forward ---"
)

# 3. Ajouter stockage features_fused apres le bloc post-process existant
store_fused_block = (
    "    FUSED_SIZES = {'Stage 1 fus': 256, 'Stage 2 fus': 128, 'Stage 3 fus': 64}\n"
    "    for fn, sz in FUSED_SIZES.items():\n"
    "        t = fused_cache[fn][0].permute(1, 2, 0).numpy()\n"
    "        STATE['features_fused'][fn] = t\n"
)
raw8 = raw8.replace(
    "    STATE['attn_cache'] = attn_cache",
    store_fused_block + "    STATE['attn_cache'] = attn_cache"
)

# 4. Ajouter print fused dans le bloc d'affichage final
raw8 = raw8.replace(
    "        print(f'  Attention capturee sur blocs: {list(attn_cache.keys())}')",
    "        print(f'  Attention capturee sur blocs: {list(attn_cache.keys())}')\n"
    "        print('  Features fusionnees :')\n"
    "        for fn in ['Stage 1 fus', 'Stage 2 fus', 'Stage 3 fus']:\n"
    "            print(f'    {fn}: {STATE[\"features_fused\"][fn].shape}')"
)

cell8["source"] = [l + "\n" for l in raw8.splitlines()]
cell8["source"][-1] = cell8["source"][-1].rstrip("\n")

# ─── NOUVELLES CELLULES — Section 8 ──────────────────────────────────────────

new_cells = []

# ── Markdown titre ────────────────────────────────────────────────────────────
new_cells.append(md_cell(
    "---",
    "## 8️⃣  Comparaison : features pures vs features fusionnées FPN",
    "",
    "**Architecture de la fusion (FPN top-down) :**",
    "```",
    "Stage 4 (32×32)  ──conv[0]──► lateral_4  ──────────────────────► out[3]  (pur)",
    "                                  │",
    "                              upsample ×2",
    "                                  │",
    "Stage 3 (64×64)  ──conv[1]──► lateral_3  ──+──► fused_3  ────────► out[2]  (fusionné ★)",
    "Stage 2 (128×128)──conv[2]──► lateral_2  ──────────────────────► out[1]  (pur)",
    "Stage 1 (256×256)──conv[3]──► lateral_1  ──────────────────────► out[0]  (pur)",
    "```",
    "`fpn_top_down_levels = [2, 3]` → **seul Stage 3 reçoit la fusion réelle**.  ",
    "Pour Stage 1 et Stage 2, *pur = fusionné* (tenseurs identiques).",
))

# ── PCA RGB comparaison ───────────────────────────────────────────────────────
new_cells.append(code_cell(
    "# ── PCA RGB : pur vs fusionné ───────────────────────────────────────────",
    "FUSED_NAMES = {'Stage 1': 'Stage 1 fus',",
    "               'Stage 2': 'Stage 2 fus',",
    "               'Stage 3': 'Stage 3 fus'}",
    "",
    "pca_cmp_toggle = widgets.ToggleButtons(",
    "    options=['Stage 1', 'Stage 2', 'Stage 3'],",
    "    description='Stage :', button_style='',",
    "    style={'button_width': '100px', 'description_width': 'initial'})",
    "out_pca_cmp = widgets.Output()",
    "",
    "def show_pca_cmp(change=None):",
    "    if not STATE['features'] or not STATE['features_fused']:",
    "        with out_pca_cmp:",
    "            print('Extraire les features d abord.')",
    "        return",
    "    sn     = pca_cmp_toggle.value",
    "    fn     = FUSED_NAMES[sn]",
    "    sz     = STAGE_SIZES[sn]",
    "    gt     = STATE['gt_maps'][sn]",
    "    feat_p = STATE['features'][sn]",
    "    feat_f = STATE['features_fused'][fn]",
    "    rgb_p, evr_p = compute_pca_rgb(feat_p)",
    "    rgb_f, evr_f = compute_pca_rgb(feat_f)",
    "    # Verifier si les tenseurs sont identiques",
    "    identical = np.allclose(feat_p, feat_f, atol=1e-5)",
    "    with out_pca_cmp:",
    "        clear_output(wait=True)",
    "        if identical:",
    "            print(f'INFO : {sn} pur == {sn} fus (pas de fusion top-down a ce niveau)')",
    "        fig, axes = plt.subplots(1, 4, figsize=(22, 5))",
    "        # Col 0 : image originale",
    "        axes[0].imshow(np.array(STATE['img_pil'].convert('RGB')))",
    "        axes[0].set_title('Image originale', fontsize=10)",
    "        axes[0].axis('off')",
    "        # Col 1 : PCA RGB pur",
    "        axes[1].imshow(rgb_p)",
    "        axes[1].set_title(",
    "            f'{sn} pur ({sz}x{sz})\\n'",
    "            f'EV: {evr_p.sum()*100:.1f}%  '",
    "            f'(PC1={evr_p[0]*100:.1f}% PC2={evr_p[1]*100:.1f}% PC3={evr_p[2]*100:.1f}%)',",
    "            fontsize=9)",
    "        axes[1].axis('off')",
    "        # Col 2 : PCA RGB fusionné",
    "        border_col = 'gold' if not identical else 'gray'",
    "        for sp in axes[2].spines.values():",
    "            sp.set_edgecolor(border_col); sp.set_linewidth(3)",
    "        axes[2].imshow(rgb_f)",
    "        fus_label = 'fusionné ★' if not identical else 'fusionné (= pur)'",
    "        axes[2].set_title(",
    "            f'{sn} {fus_label} ({sz}x{sz})\\n'",
    "            f'EV: {evr_f.sum()*100:.1f}%  '",
    "            f'(PC1={evr_f[0]*100:.1f}% PC2={evr_f[1]*100:.1f}% PC3={evr_f[2]*100:.1f}%)',",
    "            fontsize=9)",
    "        axes[2].axis('off')",
    "        # Col 3 : GT",
    "        cmap_gt = plt.colormaps['tab10']",
    "        classes = np.unique(gt)",
    "        gt_rgb  = np.zeros((*gt.shape, 3))",
    "        for k, cls in enumerate(classes):",
    "            gt_rgb[gt == cls] = mcolors.to_rgb(cmap_gt(k))",
    "        axes[3].imshow(gt_rgb)",
    "        axes[3].set_title(f'GT @ {sz}x{sz}', fontsize=10)",
    "        axes[3].axis('off')",
    "        plt.suptitle(",
    "            f'PCA RGB — {sn} pur vs fusionné' +",
    "            ('' if not identical else '  [identiques : pas de fusion à ce niveau]'),",
    "            fontsize=13)",
    "        plt.tight_layout()",
    "        plt.show()",
    "        # Afficher distance cosine entre les deux",
    "        if not identical:",
    "            X_p = l2_norm(feat_p.reshape(-1, 256))",
    "            X_f = l2_norm(feat_f.reshape(-1, 256))",
    "            cos_sim = np.einsum('nd,nd->n', X_p, X_f).mean()",
    "            print(f'Similarité cosine pixel-à-pixel (pur vs fus) : {cos_sim:.4f}')",
    "            print(f'  1.0 = identiques, 0.0 = orthogonaux')",
    "",
    "pca_cmp_toggle.observe(show_pca_cmp, names='value')",
    "display(pca_cmp_toggle, out_pca_cmp)",
    "show_pca_cmp()",
))

# ── t-SNE comparaison ─────────────────────────────────────────────────────────
new_cells.append(code_cell(
    "# ── t-SNE comparatif : pur vs fusionné ──────────────────────────────────",
    "tsne_cmp_toggle = widgets.ToggleButtons(",
    "    options=['Stage 1', 'Stage 2', 'Stage 3'],",
    "    description='Stage :', button_style='',",
    "    style={'button_width': '100px', 'description_width': 'initial'})",
    "tsne_cmp_btn = widgets.Button(description='Lancer t-SNE comparatif',",
    "                               button_style='warning', icon='play')",
    "out_tsne_cmp = widgets.Output()",
    "",
    "def on_tsne_cmp(b):",
    "    if not STATE['features'] or not STATE['features_fused']:",
    "        with out_tsne_cmp:",
    "            print('Extraire les features d abord.')",
    "        return",
    "    sn     = tsne_cmp_toggle.value",
    "    fn     = FUSED_NAMES[sn]",
    "    gt     = STATE['gt_maps'][sn]",
    "    feat_p = STATE['features'][sn]",
    "    feat_f = STATE['features_fused'][fn]",
    "    identical = np.allclose(feat_p, feat_f, atol=1e-5)",
    "    with out_tsne_cmp:",
    "        clear_output(wait=True)",
    "        if identical:",
    "            print(f'INFO : {sn} pur == {sn} fus — t-SNE identique (pas de fusion à ce niveau)')",
    "        fig, axes = plt.subplots(1, 2, figsize=(16, 7))",
    "        for col, (feat, label) in enumerate([",
    "                (feat_p, f'{sn} pur'),",
    "                (feat_f, f'{sn} fusionné{\" ★\" if not identical else \" (= pur)\"}'),",
    "        ]):",
    "            ax = axes[col]",
    "            X  = l2_norm(feat.reshape(-1, feat.shape[-1]))",
    "            y  = gt.flatten()",
    "            idx = stratified_subsample(X, y, 2000)",
    "            X_s, y_s = X[idx], y[idx]",
    "            print(f't-SNE {label} ({len(y_s)} pts)...', end=' ', flush=True)",
    "            t0 = time.time()",
    "            n_pc  = min(50, X_s.shape[0]-1, X_s.shape[1])",
    "            X_pca = PCA(n_components=n_pc, random_state=SEED).fit_transform(X_s)",
    "            X_2d  = TSNE(n_components=2, perplexity=30, max_iter=1000,",
    "                         random_state=SEED).fit_transform(X_pca)",
    "            print(f'{time.time()-t0:.1f}s')",
    "            classes = np.unique(y_s)",
    "            colors  = class_colors(classes)",
    "            for cls in classes:",
    "                mask = y_s == cls",
    "                ax.scatter(X_2d[mask,0], X_2d[mask,1],",
    "                           c=colors[int(cls)], s=6, alpha=0.7,",
    "                           label=f'cls {int(cls)}')",
    "            ax.set_title(f'{label}\\n{len(classes)} classes — {len(y_s)} pts',",
    "                          fontsize=11)",
    "            ax.legend(markerscale=2, fontsize=8)",
    "            ax.set_xticks([]); ax.set_yticks([])",
    "            # Calcul séparabilité inter-classe",
    "            centroids = {int(c): X_s[y_s==c].mean(axis=0) for c in classes}",
    "            cls_list  = list(centroids.keys())",
    "            dists = []",
    "            for ii in range(len(cls_list)):",
    "                for jj in range(ii+1, len(cls_list)):",
    "                    ci = centroids[cls_list[ii]]; ci /= np.linalg.norm(ci)+1e-8",
    "                    cj = centroids[cls_list[jj]]; cj /= np.linalg.norm(cj)+1e-8",
    "                    dists.append(1 - np.dot(ci, cj))",
    "            ax.set_xlabel(",
    "                f'Dist. cosine inter-classe : moy={np.mean(dists):.4f}  '",
    "                f'min={np.min(dists):.4f}  max={np.max(dists):.4f}',",
    "                fontsize=8)",
    "        plt.suptitle(",
    "            f't-SNE comparatif — {sn} pur vs fusionné',",
    "            fontsize=13)",
    "        plt.tight_layout()",
    "        plt.show()",
    "",
    "tsne_cmp_btn.on_click(on_tsne_cmp)",
    "display(widgets.HBox([tsne_cmp_toggle, tsne_cmp_btn]), out_tsne_cmp)",
))

# ─── Insérer les nouvelles cellules à la fin ──────────────────────────────────
nb["cells"].extend(new_cells)

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Notebook patché : {NB_PATH}")
print(f"Taille : {NB_PATH.stat().st_size / 1024:.1f} KB")
print(f"Nombre total de cellules : {len(nb['cells'])}")
