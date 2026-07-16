#!/usr/bin/env python3
"""
Test — SVM (linéaire + RBF) vs LP : la frontière non-linéaire débloque-t-elle
les textures à AUC moyenne (Faisceaux, Strat.sin) ?
Protocole LP LOIO identique (PCA train-only, class_weight='balanced'),
mêmes folds pour les 3 classifieurs, métriques poolées.
SVM-RBF : nested CV (GroupKFold par image, à l'intérieur du train du fold)
pour sélectionner (C, gamma) sans jamais toucher l'image test.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

H5   = Path("data/feature_database/database_meb_ouassim.h5")
OUT  = Path("test_svm_vs_lp")
OUT.mkdir(exist_ok=True)

TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1: "Tot.homogène", 3: "Faisceaux", 4: "Filaments", 5: "Strat.rect",
            6: "Strat.sin",   7: "Granuleux", 9: "Trou"}
PCA_DIM  = 50
SEED     = 42
SEUIL    = 0.5
FOCUS_TEXTURES = [3, 6]  # Faisceaux, Strat.sin

RBF_C_GRID     = [0.1, 1, 10, 100]
RBF_GAMMA_GRID = ["scale", 0.001, 0.01, 0.1]
INNER_SPLITS   = 3

# ─── Chargement + vérification images Ouassim ────────────────────────────────
print(f"Chargement H5 : {H5}")
assert H5.name == "database_meb_ouassim.h5", f"H5 inattendu : {H5}"

with h5py.File(H5, "r") as f:
    all_cat  = f["metadata"]["category_ids"][:]
    all_imgs = np.array([x.decode() for x in f["metadata"]["image_names"][:]])
    BLOCKS   = sorted(f["features"].keys(), key=lambda s: (len(s), s))
    feats    = {b: f["features"][b][:] for b in BLOCKS}

print("  H5 confirmé Ouassim. Features précalculées (moyennes par patch, protocole "
      "standard) — aucune image brute chargée, pas de risque PatchTagger.")

mask      = np.isin(all_cat, TEXTURES)
cat_ids   = all_cat[mask]
img_names = all_imgs[mask]
feats     = {b: feats[b][mask] for b in BLOCKS}
stems     = np.array([n.replace(".tif", "") for n in img_names])
print(f"  {mask.sum()} patches | {len(BLOCKS)} blocs | {len(set(stems))} images\n")


def nested_rbf_fit(Xtr, ytr, groups_tr):
    """Nested CV (GroupKFold par image) pour choisir (C, gamma) du SVM-RBF,
    sans jamais toucher les données du fold test. Retourne le modèle final
    refit sur tout le train, les hyperparams choisis, et l'AUC interne."""
    unique_groups = np.unique(groups_tr)
    n_inner = min(INNER_SPLITS, len(unique_groups))
    if n_inner < 2:
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, ytr)
        return clf, {"C": 1.0, "gamma": "scale"}, None

    gkf = GroupKFold(n_splits=n_inner)
    splits = list(gkf.split(Xtr, ytr, groups=groups_tr))
    grid_scores = {}
    for cC in RBF_C_GRID:
        for cG in RBF_GAMMA_GRID:
            aucs = []
            for tr_idx, va_idx in splits:
                y_tr_in, y_va_in = ytr[tr_idx], ytr[va_idx]
                if len(np.unique(y_tr_in)) < 2 or len(np.unique(y_va_in)) < 2:
                    continue
                clf = SVC(kernel="rbf", C=cC, gamma=cG, class_weight="balanced", random_state=SEED)
                clf.fit(Xtr[tr_idx], y_tr_in)
                score = clf.decision_function(Xtr[va_idx])
                aucs.append(roc_auc_score(y_va_in, score))
            grid_scores[(cC, cG)] = np.mean(aucs) if aucs else -1.0

    best_params = max(grid_scores, key=grid_scores.get)
    best_C, best_gamma = best_params
    final = SVC(kernel="rbf", C=best_C, gamma=best_gamma, class_weight="balanced", random_state=SEED)
    final.fit(Xtr, ytr)
    return final, {"C": best_C, "gamma": best_gamma}, grid_scores[best_params]


def loio_three_classifiers(X, y_bin, stems, want_rbf_params=False, want_train_auc=False):
    """LOIO par image. À chaque fold : PCA train-only, puis LP / SVM-lin / SVM-RBF
    (nested) sur les MÊMES données transformées. Retourne métriques poolées
    pour chaque classifieur (+ diagnostics optionnels)."""
    scores = {"LP": [], "SVM-lin": [], "SVM-RBF": []}
    y_true_all = []
    rbf_params_log = []
    train_auc_pairs = []  # (train_auc, test_auc_du_fold) pour RBF si demandé

    for stem in sorted(set(stems)):
        te = stems == stem
        tr = ~te
        if y_bin[te].sum() == 0 or len(np.unique(y_bin[tr])) < 2:
            continue

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_bin[tr], y_bin[te]
        stems_tr = stems[tr]

        if X_tr.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        # 1) LP
        lp = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
        lp.fit(X_tr, y_tr)
        scores["LP"].append(lp.predict_proba(X_te)[:, 1])

        # 2) SVM linéaire (contrôle)
        svl = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=SEED)
        svl.fit(X_tr, y_tr)
        scores["SVM-lin"].append(svl.decision_function(X_te))

        # 3) SVM-RBF (nested CV pour C, gamma)
        svr, best_params, inner_auc = nested_rbf_fit(X_tr, y_tr, stems_tr)
        te_score = svr.decision_function(X_te)
        scores["SVM-RBF"].append(te_score)
        if want_rbf_params:
            rbf_params_log.append(best_params)
        if want_train_auc:
            tr_score = svr.decision_function(X_tr)
            if len(np.unique(y_tr)) > 1 and len(np.unique(y_te)) > 1:
                train_auc_pairs.append((roc_auc_score(y_tr, tr_score),
                                         roc_auc_score(y_te, te_score)))

        y_true_all.append(y_te)

    if not y_true_all:
        return None

    y_true_all = np.concatenate(y_true_all)
    out = {}
    for clf_name in ["LP", "SVM-lin", "SVM-RBF"]:
        s = np.concatenate(scores[clf_name])
        if clf_name == "LP":
            pred = (s >= SEUIL).astype(int)
        else:
            pred = (s >= 0).astype(int)
        auc = roc_auc_score(y_true_all, s) if len(np.unique(y_true_all)) > 1 else float("nan")
        out[clf_name] = {
            "auc": auc,
            "recall": recall_score(y_true_all, pred, zero_division=0),
            "precision": precision_score(y_true_all, pred, zero_division=0),
            "f1": f1_score(y_true_all, pred, zero_division=0),
            "n_folds": len(scores[clf_name]),
        }
    if want_rbf_params:
        out["_rbf_params_log"] = rbf_params_log
    if want_train_auc:
        out["_rbf_train_test_auc"] = train_auc_pairs
    return out


# ─── Passe principale : tous les blocs × toutes les textures × 3 classifieurs ─
print("Calcul LOIO (LP + SVM-lin + SVM-RBF nested), tous blocs × toutes textures ...")
print("(Ceci est un calcul long : nested CV du RBF à chaque fold.)\n")

results = {}  # results[t][b] = {"LP":..., "SVM-lin":..., "SVM-RBF":...}
for t in TEXTURES:
    y_bin = (cat_ids == t).astype(int)
    results[t] = {}
    want_diag = t in FOCUS_TEXTURES
    for b in BLOCKS:
        r = loio_three_classifiers(feats[b], y_bin, stems,
                                    want_rbf_params=want_diag, want_train_auc=want_diag)
        results[t][b] = r
    best_lp_b = max(BLOCKS, key=lambda b: results[t][b]["LP"]["auc"] if results[t][b] else -1)
    d = results[t][best_lp_b]["LP"]
    print(f"  t{t} {TNAMES[t]:<15} → LP best={best_lp_b:<14} AUC={d['auc']:.3f}")

# ─── 1) Tableau au meilleur bloc, par classifieur ─────────────────────────────
print("\n" + "=" * 110)
print("1) MEILLEUR BLOC PAR CLASSIFIEUR (choisi par AUC, séparément pour chacun)")
print("=" * 110)
print(f"{'Texture':<14} {'LP bloc':<12}{'AUC':>6}   {'SVMlin bloc':<12}{'AUC':>6}   "
      f"{'SVMrbf bloc':<12}{'AUC':>6}   {'ΔAUC(RBF-LP)':>13}")
print("-" * 110)

summary_rows = {}
for t in TEXTURES:
    row = {}
    for clf_name in ["LP", "SVM-lin", "SVM-RBF"]:
        best_b = max(BLOCKS, key=lambda b: results[t][b][clf_name]["auc"] if results[t][b] else -1)
        row[clf_name] = (best_b, results[t][best_b][clf_name]["auc"], results[t][best_b])
    delta = row["SVM-RBF"][1] - row["LP"][1]
    summary_rows[t] = (row, delta)
    print(f"{TNAMES[t]:<14} {row['LP'][0]:<12}{row['LP'][1]:>6.3f}   "
          f"{row['SVM-lin'][0]:<12}{row['SVM-lin'][1]:>6.3f}   "
          f"{row['SVM-RBF'][0]:<12}{row['SVM-RBF'][1]:>6.3f}   {delta:>+13.3f}")
print("=" * 110)

# ─── 2) Focus Faisceaux / Strat.sin : 4 métriques × 3 classifieurs ────────────
print("\n" + "=" * 100)
print("2) FOCUS — Faisceaux et Strat.sin : métriques complètes × 3 classifieurs (à leur meilleur bloc)")
print("=" * 100)
for t in FOCUS_TEXTURES:
    print(f"\n  {TNAMES[t]} :")
    print(f"  {'Classifieur':<10}{'Bloc':<14}{'AUC':>7}{'Recall':>9}{'Précision':>11}{'F1':>7}")
    for clf_name in ["LP", "SVM-lin", "SVM-RBF"]:
        best_b, auc, d = summary_rows[t][0][clf_name]
        m = d[clf_name]
        print(f"  {clf_name:<10}{best_b:<14}{m['auc']:>7.3f}{m['recall']:>9.3f}"
              f"{m['precision']:>11.3f}{m['f1']:>7.3f}")

# ─── 3) Graphe barres AUC (3 classifieurs par texture) ────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(TEXTURES))
width = 0.26
colors = {"LP": "#3498db", "SVM-lin": "#95a5a6", "SVM-RBF": "#e74c3c"}
for i, clf_name in enumerate(["LP", "SVM-lin", "SVM-RBF"]):
    vals = [summary_rows[t][0][clf_name][1] for t in TEXTURES]
    ax.bar(x + (i - 1) * width, vals, width, label=clf_name, color=colors[clf_name])
ax.set_xticks(x)
ax.set_xticklabels([TNAMES[t] for t in TEXTURES], fontsize=10)
ax.set_ylabel("AUC (meilleur bloc par classifieur)")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
ax.legend()
ax.set_title("LP vs SVM-linéaire vs SVM-RBF — AUC par texture (LP LOIO)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
out_fig = OUT / "auc_lp_vs_svm.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGraphe sauvé : {out_fig}")

# ─── 4) Contrôle surapprentissage RBF (2 textures focus) ──────────────────────
print("\n" + "=" * 90)
print("4) CONTRÔLE SURAPPRENTISSAGE RBF — AUC train (optimiste, intra-fold) vs AUC test")
print("=" * 90)
for t in FOCUS_TEXTURES:
    best_b = summary_rows[t][0]["SVM-RBF"][0]
    pairs = results[t][best_b].get("_rbf_train_test_auc", [])
    if pairs:
        train_aucs = [p[0] for p in pairs]
        test_aucs  = [p[1] for p in pairs]
        print(f"  {TNAMES[t]:<15} (bloc {best_b}) : "
              f"AUC train moyen={np.mean(train_aucs):.3f}  "
              f"AUC test moyen (par fold, non poolé)={np.mean(test_aucs):.3f}  "
              f"écart={np.mean(train_aucs)-np.mean(test_aucs):+.3f}")
        if np.mean(train_aucs) > 0.97 and (np.mean(train_aucs) - np.mean(test_aucs)) > 0.15:
            print(f"    ⚠ écart important → signe de surapprentissage du RBF sur ce bloc/texture")
    else:
        print(f"  {TNAMES[t]:<15} : pas de paires valides (folds trop déséquilibrés)")

# ─── 5) Stabilité des hyperparamètres RBF (2 textures focus) ──────────────────
print("\n" + "=" * 90)
print("5) STABILITÉ DES HYPERPARAMÈTRES RBF (C, gamma) choisis par fold")
print("=" * 90)
for t in FOCUS_TEXTURES:
    best_b = summary_rows[t][0]["SVM-RBF"][0]
    params_log = results[t][best_b].get("_rbf_params_log", [])
    if params_log:
        counts = Counter((p["C"], p["gamma"]) for p in params_log)
        n_unique = len(counts)
        print(f"  {TNAMES[t]:<15} (bloc {best_b}, {len(params_log)} folds) : "
              f"{n_unique} combinaisons (C,gamma) différentes choisies")
        for (cC, cG), n in counts.most_common():
            print(f"      C={cC:<6} gamma={cG:<8} → {n} fold(s)")
        if n_unique / max(len(params_log), 1) > 0.5:
            print(f"    ⚠ instabilité élevée : hyperparamètres très variables d'un fold à l'autre")
    else:
        print(f"  {TNAMES[t]:<15} : log vide")

# ─── 6) Contrôle anti-fuite (rappel explicite) ────────────────────────────────
print("\n" + "=" * 90)
print("6) CONTRÔLE ANTI-FUITE")
print("=" * 90)
example_t = FOCUS_TEXTURES[0]
example_stem = sorted(set(stems))[0]
te_mask = stems == example_stem
tr_mask = ~te_mask
inter = set(stems[te_mask]) & set(stems[tr_mask])
print(f"  Split outer (image '{example_stem}') : test ∩ train = {inter} "
      f"({'OK vide' if not inter else 'FUITE'})")
print("  Nested CV du RBF : GroupKFold sur les noms d'image du TRAIN du fold "
      "uniquement (l'image test du fold externe n'entre jamais dans le grid search).")
print("  PCA fittée sur le train externe uniquement (même transform réutilisé pour "
      "la nested CV interne — simplification documentée : aucune image de test "
      "externe ne contribue jamais à la PCA).")

# ─── VERDICT global ────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("VERDICT — la non-linéarité débloque-t-elle Faisceaux / Strat.sin, ou le plafond "
      "est-il dans l'encodeur/les données ?")
print("=" * 100)
for t in TEXTURES:
    row, delta = summary_rows[t]
    lp_auc, svl_auc, svr_auc = row["LP"][1], row["SVM-lin"][1], row["SVM-RBF"][1]
    lin_consistent = abs(lp_auc - svl_auc) < 0.05
    if delta > 0.05:
        verdict = "GAIN significatif du RBF → la non-linéarité débloque de l'info"
    elif delta > 0.02:
        verdict = "gain marginal du RBF"
    else:
        verdict = "pas de gain → plafond confirmé côté encodeur/données, pas le classifieur"
    print(f"  {TNAMES[t]:<15} LP={lp_auc:.3f} SVMlin={svl_auc:.3f} SVMrbf={svr_auc:.3f}  "
          f"ΔAUC(RBF-LP)={delta:+.3f}  "
          f"[LP≈SVMlin: {'oui' if lin_consistent else 'NON, à vérifier'}]  → {verdict}")

# ─── Sauvegarde rapport texte ──────────────────────────────────────────────────
report = OUT / "rapport_svm_vs_lp.txt"
with open(report, "w") as f:
    f.write("Test — SVM (linéaire + RBF) vs LP : la non-linéarité débloque-t-elle les textures ?\n")
    f.write(f"H5 : {H5}\n\n")
    f.write("1) Meilleur bloc par classifieur :\n")
    for t in TEXTURES:
        row, delta = summary_rows[t]
        f.write(f"  {TNAMES[t]:<16} LP:{row['LP'][0]}({row['LP'][1]:.3f})  "
                f"SVMlin:{row['SVM-lin'][0]}({row['SVM-lin'][1]:.3f})  "
                f"SVMrbf:{row['SVM-RBF'][0]}({row['SVM-RBF'][1]:.3f})  ΔAUC={delta:+.3f}\n")
    f.write("\nVoir stdout du script pour le détail des sections 2-6 (focus, "
            "surapprentissage, stabilité hyperparamètres, anti-fuite).\n")

print(f"\nRapport texte sauvé : {report}")
print(f"Toutes les sorties dans : {OUT}/")
