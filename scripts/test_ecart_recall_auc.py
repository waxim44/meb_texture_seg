#!/usr/bin/env python3
"""
Test — D'où vient l'écart entre l'ancien recall seul et les nouvelles métriques (AUC, etc) ?
Une SEULE implémentation LP LOIO (identique à test_multimetriques_loio.py), calculée
une fois pour toutes les textures × tous les blocs. On isole ensuite la cause de
l'écart : (A) choix du bloc (best-recall vs best-AUC), (B) position du seuil, (C) autre.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

H5   = Path("data/feature_database/database_meb_ouassim.h5")
OUT  = Path("test_ecart_recall_auc")
OUT.mkdir(exist_ok=True)

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1: "Tot.homogène", 3: "Faisceaux", 4: "Filaments", 5: "Strat.rect",
            6: "Strat.sin",   7: "Granuleux", 9: "Trou"}
PCA_DIM  = 50
C        = 1.0
SEED     = 42
SEUIL    = 0.5

# ─── Chargement + vérification images Ouassim ────────────────────────────────
print(f"Chargement H5 : {H5}")
assert H5.name == "database_meb_ouassim.h5", f"H5 inattendu : {H5}"

with h5py.File(H5, "r") as f:
    all_cat  = f["metadata"]["category_ids"][:]
    all_imgs = np.array([x.decode() for x in f["metadata"]["image_names"][:]])
    BLOCKS   = sorted(f["features"].keys(), key=lambda s: (len(s), s))
    feats    = {b: f["features"][b][:] for b in BLOCKS}

mask      = np.isin(all_cat, TEXTURES)
cat_ids   = all_cat[mask]
img_names = all_imgs[mask]
feats     = {b: feats[b][mask] for b in BLOCKS}
stems     = np.array([n.replace(".tif", "") for n in img_names])

print(f"  H5 confirmé Ouassim. {mask.sum()} patches | {len(BLOCKS)} blocs | "
      f"{len(set(stems))} images. Pas d'image brute chargée → pas de risque PatchTagger.\n")


def loio_pooled(X, y_bin, stems, verify_leak=False):
    """LOIO par image, PCA train-only, LR balanced → y_true/y_proba pooled sur tous les folds."""
    y_true_all, y_proba_all = [], []
    n_folds = 0
    checked = False
    for stem in sorted(set(stems)):
        te = stems == stem
        tr = ~te
        if y_bin[te].sum() == 0 or len(np.unique(y_bin[tr])) < 2:
            continue
        if verify_leak and not checked:
            inter = set(stems[te]) & set(stems[tr])
            print(f"    [anti-fuite] fold '{stem}' : test ∩ train = {inter} "
                  f"({'OK vide' if not inter else 'FUITE'})")
            checked = True
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]
        if X_tr.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]
        y_true_all.append(y_te)
        y_proba_all.append(proba_te)
        n_folds += 1
    if n_folds == 0:
        return None
    return np.concatenate(y_true_all), np.concatenate(y_proba_all), n_folds


def metrics_at_threshold(y_true, y_proba, seuil=0.5):
    y_pred = (y_proba >= seuil).astype(int)
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ─── Passe unique : texture × bloc → pooled (y_true, y_proba) + métriques@0.5 ─
print("Calcul LP LOIO — une seule passe, toutes textures × tous blocs ...")
pooled = {}     # pooled[t][b] = (y_true, y_proba, n_folds)
table = {}      # table[t][b] = dict(auc, recall, precision, f1, n_folds)
first = True

for t in TEXTURES:
    y_bin = (cat_ids == t).astype(int)
    pooled[t] = {}
    table[t] = {}
    for b in BLOCKS:
        r = loio_pooled(feats[b], y_bin, stems, verify_leak=first)
        if r is None:
            table[t][b] = None
            continue
        first = False
        y_true, y_proba, n_folds = r
        pooled[t][b] = (y_true, y_proba, n_folds)
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
        m = metrics_at_threshold(y_true, y_proba, SEUIL)
        table[t][b] = {"auc": auc, "n_folds": n_folds, **m}
    print(f"  t{t} {TNAMES[t]:<15} ok")

print()

# ═══ 1) MÊME BLOC → recall identique ? (contrôle de cohérence / déterminisme) ═
print("=" * 90)
print("1) CONTRÔLE DE COHÉRENCE — recall recalculé deux fois indépendamment, au même bloc")
print("   (même code, même protocole → doit être STRICTEMENT identique)")
print("=" * 90)
check_textures = [5, 3, 7]  # Strat.rect, Faisceaux, Granuleux
for t in check_textures:
    y_bin = (cat_ids == t).astype(int)
    best_auc_block = max(BLOCKS, key=lambda b: (table[t][b]["auc"] if table[t][b] else -1))
    # recalcul indépendant (2e appel, même bloc, même texture)
    r2 = loio_pooled(feats[best_auc_block], y_bin, stems, verify_leak=False)
    y_true2, y_proba2, _ = r2
    recall_2nd = metrics_at_threshold(y_true2, y_proba2, SEUIL)["recall"]
    recall_1st = table[t][best_auc_block]["recall"]
    diff = abs(recall_1st - recall_2nd)
    print(f"  {TNAMES[t]:<15} bloc={best_auc_block:<14} recall(1er calcul)={recall_1st:.6f}  "
          f"recall(2e calcul)={recall_2nd:.6f}  diff={diff:.2e} "
          f"{'✓ identique' if diff < 1e-9 else '✗ DIFFÉRENT — protocole caché à vérifier !'}")
print()

# ═══ 2) CHOIX DU BLOC : bloc-max-recall vs bloc-max-AUC ═══════════════════════
print("=" * 100)
print("2) CHOIX DU BLOC — meilleur bloc par RECALL vs meilleur bloc par AUC")
print("=" * 100)
print(f"{'Texture':<16} {'bloc max-RECALL':<16} {'recall':>8} {'AUC':>7}   "
      f"{'bloc max-AUC':<16} {'recall':>8} {'AUC':>7}   {'même bloc?':>10}")
print("-" * 100)
comparison2 = {}
for t in TEXTURES:
    valid_blocks = [b for b in BLOCKS if table[t][b] is not None]
    best_recall_b = max(valid_blocks, key=lambda b: table[t][b]["recall"])
    best_auc_b    = max(valid_blocks, key=lambda b: table[t][b]["auc"])
    dr, da = table[t][best_recall_b], table[t][best_auc_b]
    same = (best_recall_b == best_auc_b)
    comparison2[t] = dict(best_recall_b=best_recall_b, best_auc_b=best_auc_b, same=same,
                          recall_at_recall_b=dr["recall"], auc_at_recall_b=dr["auc"],
                          recall_at_auc_b=da["recall"], auc_at_auc_b=da["auc"])
    print(f"{TNAMES[t]:<16} {best_recall_b:<16} {dr['recall']:>8.3f} {dr['auc']:>7.3f}   "
          f"{best_auc_b:<16} {da['recall']:>8.3f} {da['auc']:>7.3f}   "
          f"{'  oui' if same else '  NON':>10}")
print()

# ═══ 3) EFFET DU SEUIL — recall & précision vs seuil, au bloc max-AUC ═════════
print("=" * 90)
print("3) EFFET DU SEUIL — recall & précision en fonction du seuil (bloc max-AUC)")
print("=" * 90)
curve_textures = [5, 3, 7]  # Strat.rect (test seuil), Faisceaux (test seuil), Granuleux (contraste solide)
thresholds = np.arange(0.1, 0.95, 0.05)

fig, axes = plt.subplots(1, len(curve_textures), figsize=(15, 4.5), sharey=True)
for ax, t in zip(axes, curve_textures):
    best_auc_b = comparison2[t]["best_auc_b"]
    y_true, y_proba, n_folds = pooled[t][best_auc_b]
    recalls, precisions = [], []
    for s in thresholds:
        m = metrics_at_threshold(y_true, y_proba, s)
        recalls.append(m["recall"])
        precisions.append(m["precision"])
    ax.plot(thresholds, recalls, marker="o", ms=3, label="Recall", color="#2980b9")
    ax.plot(thresholds, precisions, marker="s", ms=3, label="Précision", color="#e67e22")
    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    auc_val = table[t][best_auc_b]["auc"]
    ax.set_title(f"{TNAMES[t]}\n(bloc {best_auc_b}, AUC={auc_val:.3f})", fontsize=10)
    ax.set_xlabel("Seuil de décision")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Score")
plt.suptitle("Recall & Précision vs Seuil — à séparabilité (AUC) fixée, le choix du seuil "
             "change le recall", fontsize=11)
plt.tight_layout()
out = OUT / "recall_precision_vs_seuil.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Courbes sauvées : {out}\n")

# ═══ CONCLUSION FACTUELLE PAR TEXTURE ══════════════════════════════════════════
OLD_RECALL_AT_OLD_BEST = {1: 1.00, 7: 0.86, 4: 0.84, 3: 0.79, 9: 0.74, 6: 0.52, 5: 0.50}

print("=" * 100)
print("CONCLUSION — cause de l'écart entre l'ancien recall (best-bloc-recall) et le "
      "nouveau tableau (best-bloc-AUC)")
print("=" * 100)
for t in TEXTURES:
    c = comparison2[t]
    old = OLD_RECALL_AT_OLD_BEST.get(t, float("nan"))
    new_recall_at_auc_block = c["recall_at_auc_b"]
    ecart = new_recall_at_auc_block - old
    if c["same"]:
        cause = "(C) aucun changement de bloc — écart résiduel (bruit/arrondi) à investiguer"
    elif abs(ecart) < 0.02:
        cause = "(A) choix du bloc explique tout — recall quasi identique une fois le bloc fixé"
    else:
        cause = "(A)+(B) changement de bloc ET le seuil 0.5 masque une partie du recall potentiel"
    print(f"  {TNAMES[t]:<16} ancien_recall={old:.2f}  "
          f"nouveau_recall(bloc max-AUC)={new_recall_at_auc_block:.3f}  écart={ecart:+.3f}  "
          f"bloc changé={'non' if c['same'] else 'oui'} → {cause}")

print("\nRappel factuel : aucune des deux mesures n'est fausse. Le recall répond à "
      "'combien je rate, à CE seuil, sur CE bloc'. L'AUC répond à 'les deux classes "
      "sont-elles séparables, indépendamment du seuil'. Elles peuvent légitimement "
      "diverger sans contradiction.")

# ─── Sauvegarde rapport texte ──────────────────────────────────────────────────
report = OUT / "rapport_ecart.txt"
with open(report, "w") as f:
    f.write("Test — origine de l'écart ancien recall vs nouvelles métriques (AUC etc.)\n")
    f.write(f"H5 : {H5}\n\n")
    f.write("1) Contrôle de cohérence (même bloc, recalcul indépendant) :\n")
    for t in check_textures:
        y_bin = (cat_ids == t).astype(int)
        best_auc_block = comparison2[t]["best_auc_b"]
        f.write(f"  {TNAMES[t]}: bloc={best_auc_block} — recall reproductible (voir stdout)\n")
    f.write("\n2) Bloc max-recall vs bloc max-AUC :\n")
    for t in TEXTURES:
        c = comparison2[t]
        f.write(f"  {TNAMES[t]:<16} max-recall→{c['best_recall_b']} (recall={c['recall_at_recall_b']:.3f}, "
                f"AUC={c['auc_at_recall_b']:.3f})  |  max-AUC→{c['best_auc_b']} "
                f"(recall={c['recall_at_auc_b']:.3f}, AUC={c['auc_at_auc_b']:.3f})  "
                f"même bloc={c['same']}\n")
    f.write("\n3) Voir recall_precision_vs_seuil.png pour l'effet du seuil.\n")
    f.write("\nConclusion : voir stdout du script pour le détail par texture.\n")

print(f"\nRapport texte sauvé : {report}")
print(f"Toutes les sorties dans : {OUT}/")
