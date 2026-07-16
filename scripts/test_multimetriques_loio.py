#!/usr/bin/env python3
"""
Test — Séparabilité des textures : AUC + recall + précision + F1, par texture × bloc.
Protocole LP LOIO strict (anti-fuite), identique aux tests validés (best_block_loio.py) :
LOIO par image, PCA(50) fittée train-only, LogisticRegression balanced.
Métriques calculées "pooled" (accumulées sur tous les folds), pas moyennées par fold.
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

H5       = Path("data/feature_database/database_meb_ouassim.h5")
OUT      = Path("test_multimetriques")
OUT.mkdir(exist_ok=True)

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1: "Tot.homogène", 3: "Faisceaux", 4: "Filaments", 5: "Strat.rect",
            6: "Strat.sin",   7: "Granuleux", 9: "Trou"}
PCA_DIM  = 50
C        = 1.0
SEED     = 42
SEUIL    = 0.5

OLD_RECALL = {1: 1.00, 7: 0.86, 4: 0.84, 3: 0.79, 9: 0.74, 6: 0.52, 5: 0.50}

# ─── Chargement + vérification images Ouassim ────────────────────────────────
print(f"Chargement H5 : {H5}")
assert H5.name == "database_meb_ouassim.h5", f"H5 inattendu : {H5}"

with h5py.File(H5, "r") as f:
    all_cat  = f["metadata"]["category_ids"][:]
    all_imgs = np.array([x.decode() for x in f["metadata"]["image_names"][:]])
    BLOCKS   = sorted(f["features"].keys(), key=lambda s: (len(s), s))
    feats    = {b: f["features"][b][:] for b in BLOCKS}

print(f"  H5 confirmé : {H5} (features précalculées, pas de chargement d'images brutes "
      f"→ pas de risque PatchTagger ici).")

mask      = np.isin(all_cat, TEXTURES)
cat_ids   = all_cat[mask]
img_names = all_imgs[mask]
feats     = {b: feats[b][mask] for b in BLOCKS}
stems     = np.array([n.replace(".tif", "") for n in img_names])

print(f"  {mask.sum()} patches | {len(BLOCKS)} blocs | {len(set(stems))} images")

# ─── LP LOIO multi-métriques ──────────────────────────────────────────────────
def loio_metrics(X, y_bin, stems, verify_fold_leak=False):
    """LOIO par image, PCA train-only, LR balanced → métriques pooled sur tous les folds."""
    y_true_all, y_proba_all, y_pred_all = [], [], []
    n_folds = 0
    checked_leak = False
    for stem in sorted(set(stems)):
        te = stems == stem
        tr = ~te
        if y_bin[te].sum() == 0:
            continue
        if len(np.unique(y_bin[tr])) < 2:
            continue

        if verify_fold_leak and not checked_leak:
            test_imgs = set(stems[te])
            train_imgs = set(stems[tr])
            inter = test_imgs & train_imgs
            print(f"    [anti-fuite] fold '{stem}' : images test ∩ train = {inter} "
                  f"({'OK vide' if not inter else 'FUITE DÉTECTÉE'})")
            checked_leak = True

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]
        if X_tr.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(C=C, class_weight="balanced",
                                  max_iter=1000, random_state=SEED)
        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]
        pred_te = (proba_te >= SEUIL).astype(int)

        y_true_all.append(y_te)
        y_proba_all.append(proba_te)
        y_pred_all.append(pred_te)
        n_folds += 1

    if n_folds == 0:
        return None

    y_true_all  = np.concatenate(y_true_all)
    y_proba_all = np.concatenate(y_proba_all)
    y_pred_all  = np.concatenate(y_pred_all)

    if len(np.unique(y_true_all)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true_all, y_proba_all)

    return {
        "auc":       auc,
        "recall":    recall_score(y_true_all, y_pred_all, zero_division=0),
        "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
        "f1":        f1_score(y_true_all, y_pred_all, zero_division=0),
        "n_folds":   n_folds,
    }


print("\nCalcul LP LOIO (tous blocs × toutes textures, 4 métriques) ...")
results = {}  # results[texture][bloc] = dict(auc, recall, precision, f1, n_folds)
first_check_done = False

for t in TEXTURES:
    y_bin = (cat_ids == t).astype(int)
    results[t] = {}
    for b in BLOCKS:
        verify = not first_check_done
        r = loio_metrics(feats[b], y_bin, stems, verify_fold_leak=verify)
        if verify and r is not None:
            first_check_done = True
        results[t][b] = r
    best = max(BLOCKS, key=lambda b: (results[t][b]["auc"] if results[t][b] else -1))
    d = results[t][best]
    print(f"  t{t} {TNAMES[t]:<15} → best(AUC)={best:<14} "
          f"AUC={d['auc']:.3f} recall={d['recall']:.3f} "
          f"prec={d['precision']:.3f} f1={d['f1']:.3f} ({d['n_folds']} folds)")

# ─── Alerte fuite potentielle ─────────────────────────────────────────────────
all_aucs = [results[t][b]["auc"] for t in TEXTURES for b in BLOCKS if results[t][b]]
if np.mean([a >= 0.98 for a in all_aucs]) > 0.5:
    print("\n⚠ ALERTE : une majorité d'AUC ≥ 0.98 — possible fuite de données, à vérifier !")
else:
    print(f"\nAUC max observée = {max(all_aucs):.3f}, AUC médiane = {np.median(all_aucs):.3f} "
          f"(pas de signal de fuite généralisée).")

# ─── Heatmaps (texture × bloc), une par métrique ──────────────────────────────
def build_matrix(metric):
    M = np.full((len(TEXTURES), len(BLOCKS)), np.nan)
    for i, t in enumerate(TEXTURES):
        for j, b in enumerate(BLOCKS):
            d = results[t][b]
            if d is not None:
                M[i, j] = d[metric]
    return M

metric_labels = {"auc": "AUC", "recall": "Recall", "precision": "Précision", "f1": "F1"}
tex_labels = [TNAMES[t] for t in TEXTURES]

for metric, label in metric_labels.items():
    M = build_matrix(metric)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(BLOCKS)))
    ax.set_xticklabels(BLOCKS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(TEXTURES)))
    ax.set_yticklabels(tex_labels, fontsize=10)
    for i in range(len(TEXTURES)):
        for j in range(len(BLOCKS)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < v < 0.85 else "white")
    ax.set_title(f"{label} — LP LOIO par texture × bloc (H5 Ouassim)", fontsize=12)
    fig.colorbar(im, ax=ax, label=label, fraction=0.02, pad=0.01)
    plt.tight_layout()
    out = OUT / f"heatmap_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap sauvée : {out}")

# ─── Tableau au meilleur bloc (choisi par AUC) ────────────────────────────────
best_per_tex = {}
for t in TEXTURES:
    best_b = max(BLOCKS, key=lambda b: (results[t][b]["auc"] if results[t][b] else -1))
    d = results[t][best_b]
    best_per_tex[t] = {"bloc": best_b, **d}

print("\n" + "=" * 90)
print(f"{'Texture':<16} {'Meilleur bloc':<14} {'AUC':>7} {'Recall':>8} {'Précision':>10} {'F1':>7} {'n_folds':>8}")
print("-" * 90)
for t in sorted(best_per_tex, key=lambda t: -best_per_tex[t]["auc"]):
    d = best_per_tex[t]
    print(f"{TNAMES[t]:<16} {d['bloc']:<14} {d['auc']:>7.3f} {d['recall']:>8.3f} "
          f"{d['precision']:>10.3f} {d['f1']:>7.3f} {d['n_folds']:>8}")
print("=" * 90)

# ─── Diagnostic automatique ────────────────────────────────────────────────────
print("\nDiagnostic automatique (au meilleur bloc, choisi par AUC) :")
diagnostics = {}
for t in TEXTURES:
    d = best_per_tex[t]
    auc, prec, rec = d["auc"], d["precision"], d["recall"]
    if auc >= 0.80 and prec >= 0.70 and rec >= 0.70:
        diag = "séparation solide"
    elif rec >= 0.80 and prec < 0.50:
        diag = "biais recall possible"
    elif auc < 0.65:
        diag = "non séparable"
    else:
        diag = "intermédiaire"
    diagnostics[t] = diag
    print(f"  {TNAMES[t]:<16} (bloc {d['bloc']}, AUC={auc:.3f}, prec={prec:.3f}, rec={rec:.3f}) → {diag}")

# ─── Comparaison à l'ancien recall seul ───────────────────────────────────────
print("\nComparaison au recall seul (ancien test) :")
print(f"{'Texture':<16} {'Ancien recall':>14} {'Nouveau recall':>15} {'Écart':>8} {'AUC':>7} {'Précision':>10}")
for t in TEXTURES:
    d = best_per_tex[t]
    old = OLD_RECALL.get(t, float("nan"))
    new = d["recall"]
    ecart = new - old
    flag = "  ⚠ écart notable" if abs(ecart) > 0.10 else ""
    print(f"{TNAMES[t]:<16} {old:>14.2f} {new:>15.3f} {ecart:>+8.3f} {d['auc']:>7.3f} {d['precision']:>10.3f}{flag}")

# ─── Sauvegarde texte du rapport ───────────────────────────────────────────────
report_path = OUT / "rapport.txt"
with open(report_path, "w") as f:
    f.write("Test multi-métriques — séparabilité des textures (LP LOIO)\n")
    f.write(f"H5 : {H5}\n")
    f.write(f"Patches : {mask.sum()} | Blocs : {len(BLOCKS)} | Images : {len(set(stems))}\n\n")
    f.write(f"{'Texture':<16} {'Meilleur bloc':<14} {'AUC':>7} {'Recall':>8} {'Précision':>10} {'F1':>7} {'n_folds':>8}  Diagnostic\n")
    for t in sorted(best_per_tex, key=lambda t: -best_per_tex[t]["auc"]):
        d = best_per_tex[t]
        f.write(f"{TNAMES[t]:<16} {d['bloc']:<14} {d['auc']:>7.3f} {d['recall']:>8.3f} "
                f"{d['precision']:>10.3f} {d['f1']:>7.3f} {d['n_folds']:>8}  {diagnostics[t]}\n")
    f.write("\nComparaison ancien recall seul vs nouveau :\n")
    for t in TEXTURES:
        d = best_per_tex[t]
        old = OLD_RECALL.get(t, float("nan"))
        f.write(f"{TNAMES[t]:<16} ancien={old:.2f} nouveau={d['recall']:.3f} "
                f"écart={d['recall']-old:+.3f} AUC={d['auc']:.3f} précision={d['precision']:.3f}\n")

print(f"\nRapport texte sauvé : {report_path}")
print(f"\nToutes les sorties dans : {OUT}/")
