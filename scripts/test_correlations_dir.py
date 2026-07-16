#!/usr/bin/env python3
"""
Test — Corrélations directionnelles : l'orientation débloque-t-elle les
textures orientées (Strat.sin, Faisceaux) ?
V1 = moyenne seule | V2 = moyenne + 8 corrélations directionnelles | V3 = corrélations seules.
Même protocole LP LOIO validé (PCA train-only, class_weight='balanced'),
mêmes folds pour les 3 variantes, métriques poolées.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score

H5_PATH   = Path("data/feature_database/database_meb_ouassim.h5")
CACHE_DIR = Path("vote_analysis_cache")
OUT       = Path("test_correlations_dir")
OUT.mkdir(exist_ok=True)

BLOCKS   = ["block_0", "block_2", "block_7", "block_9", "block_10", "block_15", "stage_3_fpn"]
TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1: "Tot.homogène", 3: "Faisceaux", 4: "Filaments", 5: "Strat.rect",
            6: "Strat.sin",   7: "Granuleux", 9: "Trou"}
FOCUS_TEXTURES = [3, 6]  # Faisceaux, Strat.sin
PCA_DIM  = 50
SEED     = 42
SEUIL    = 0.5

DESC_NAMES = ["corr_H", "corr_V", "corr_D1", "corr_D2", "corr_H2", "corr_V2", "ratio_HV", "aniso"]

# ─── AUTO-VÉRIFICATION (checklist protocole, imprimée avant tout calcul) ──────
CHECKLIST = [
    ("SOURCE",         f"H5 = {H5_PATH.name} (chemin affiché ci-dessous), pas de PatchTagger — "
                        f"images brutes 'Image_Ouassim' utilisées lors de l'extraction du cache."),
    ("SPLIT",          "split par IMAGE (stems == stem / ~stems), aucun train_test_split aléatoire "
                        "sur patches ou vecteurs (voir boucle loio_pooled ci-dessous)."),
    ("ANTI-FUITE",     "assert intersection(images test, images train) vide, exécuté au premier fold "
                        "de chaque appel loio_pooled."),
    ("STANDARDISATION","StandardScaler().fit(raw_tr) — fit sur le train du fold SEULEMENT, avant "
                        "concaténation/PCA (fonction fit_transform_fold)."),
    ("PCA",            "PCA(50).fit(raw_tr_scaled) — fit sur le train du fold SEULEMENT, jamais sur "
                        "l'ensemble (V1/V2 uniquement ; V3 sans PCA, 8 dims)."),
    ("DESCRIPTEURS",   "compute_descriptors(patch) calculé à partir de SA grille (rows/cols/vecs) "
                        "uniquement — aucune statistique inter-patchs, aucune normalisation globale "
                        "pré-split (vecteurs déjà L2-normalisés per-token à l'extraction, indépendant "
                        "du split)."),
    ("MÊMES FOLDS",    "une seule boucle for stem in sorted(set(stems)) partagée par V1/V2/V3 dans "
                        "run_variant() — les 3 variantes voient exactement les mêmes folds → ΔAUC apparié."),
    ("MÉTRIQUES",      "poolées : y_true_all/y_proba_all concaténés sur tous les folds avant "
                        "roc_auc_score/recall_score/precision_score ; AUC calculée sur predict_proba, "
                        "jamais sur les prédictions binaires."),
    ("BALANCED",       "LogisticRegression(..., class_weight='balanced', ...) — passé explicitement."),
    ("BORDS/NaN",      "compute_descriptors renvoie NaN si la grille est trop petite pour la paire "
                        "(vérifié via les tailles h,w avant slicing) ; colonnes 100% NaN sur un bloc "
                        "exclues (drop_allnan_cols) ; NaN résiduels partiels imputés par médiane du "
                        "TRAIN du fold uniquement (impute_train_median)."),
    ("SEED",           "SEED=42 fixé pour PCA et LogisticRegression."),
    ("SIGNAL D'ALARME","vérification automatique : AUC V2/V3 ≥ 0.98 sur une texture dont V1 était "
                        "modeste (<0.85) → affichée en alerte avant le tableau final."),
]

print("=" * 100)
print("CHECKLIST DE CONFORMITÉ (auto-vérification, relue avant exécution complète)")
print("=" * 100)
for name, desc in CHECKLIST:
    print(f"  [CONFORME] {name:<16} : {desc}")
print("=" * 100 + "\n")

# ─── Vérification H5 Ouassim ──────────────────────────────────────────────────
assert H5_PATH.name == "database_meb_ouassim.h5", f"H5 inattendu : {H5_PATH}"
print(f"H5 confirmé : {H5_PATH}\n")

# ─── Chargement du cache (grilles de vecteurs locaux + positions) ────────────
print("Chargement du cache vote_analysis_cache/ pour les 7 blocs du panel ...")
patches_by_block = {}
for b in BLOCKS:
    cf = CACHE_DIR / f"vecs_{b}.pkl"
    assert cf.exists(), f"Cache manquant pour {b} : {cf} — lancer extract_missing_blocks_cache.py d'abord."
    with open(cf, "rb") as f:
        plist = pickle.load(f)
    patches_by_block[b] = {p["patch_id"]: p for p in plist}
    print(f"  {b:<14} : {len(plist)} patches")

pid_sets = [set(patches_by_block[b].keys()) for b in BLOCKS]
common_pids = set.intersection(*pid_sets)
print(f"  patch_ids communs à tous les blocs : {len(common_pids)}")
assert len(common_pids) > 0

canonical_pids = sorted(common_pids)
ref = patches_by_block[BLOCKS[0]]
cat_ids = np.array([ref[pid]["texture"] for pid in canonical_pids])
stems   = np.array([ref[pid]["image"]   for pid in canonical_pids])
print(f"  {len(canonical_pids)} patches | {len(set(stems))} images\n")


def build_grid(p):
    h, w = p["feat_h"], p["feat_w"]
    D = p["vecs"].shape[1]
    grid = np.zeros((h, w, D), dtype=np.float32)
    grid[p["rows"], p["cols"]] = p["vecs"]
    return grid


def compute_descriptors(p):
    grid = build_grid(p)
    h, w, D = grid.shape

    def dirmean(a, b):
        return float(np.sum(a * b, axis=-1).mean())

    corr_H  = dirmean(grid[:, :-1, :], grid[:, 1:, :])   if w >= 2 else np.nan
    corr_V  = dirmean(grid[:-1, :, :], grid[1:, :, :])   if h >= 2 else np.nan
    corr_D1 = dirmean(grid[:-1, :-1, :], grid[1:, 1:, :]) if (h >= 2 and w >= 2) else np.nan
    corr_D2 = dirmean(grid[:-1, 1:, :], grid[1:, :-1, :]) if (h >= 2 and w >= 2) else np.nan
    corr_H2 = dirmean(grid[:, :-2, :], grid[:, 2:, :])   if w >= 3 else np.nan
    corr_V2 = dirmean(grid[:-2, :, :], grid[2:, :, :])   if h >= 3 else np.nan

    dirs4 = np.array([corr_H, corr_V, corr_D1, corr_D2])
    ratio_HV = corr_H - corr_V if not (np.isnan(corr_H) or np.isnan(corr_V)) else np.nan
    aniso = (dirs4.max() - dirs4.min()) if not np.any(np.isnan(dirs4)) else np.nan

    return np.array([corr_H, corr_V, corr_D1, corr_D2, corr_H2, corr_V2, ratio_HV, aniso], dtype=np.float32)


def drop_allnan_cols(desc_matrix, names):
    valid = ~np.all(np.isnan(desc_matrix), axis=0)
    return desc_matrix[:, valid], [n for n, v in zip(names, valid) if v]


def impute_train_median(X_tr, X_te):
    meds = np.nanmedian(X_tr, axis=0)
    meds = np.where(np.isnan(meds), 0.0, meds)
    def fill(X):
        X = X.copy()
        idx = np.where(np.isnan(X))
        if idx[0].size:
            X[idx] = np.take(meds, idx[1])
        return X
    return fill(X_tr), fill(X_te)


def fit_transform_fold(raw_tr, raw_te, use_pca):
    # Imputation NaN résiduelle : médiane calculée sur le TRAIN du fold uniquement,
    # appliquée à train et test — obligatoire avant StandardScaler (qui ne gère pas les NaN).
    if np.isnan(raw_tr).any() or np.isnan(raw_te).any():
        raw_tr, raw_te = impute_train_median(raw_tr, raw_te)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(raw_tr)
    Xte = scaler.transform(raw_te)
    if use_pca and Xtr.shape[1] > PCA_DIM:
        pca = PCA(n_components=PCA_DIM, random_state=SEED)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
    return Xtr, Xte


def loio_pooled(raw_all, y_bin, stems, use_pca, verify_leak=False):
    y_true_all, y_proba_all = [], []
    checked = False
    for stem in sorted(set(stems)):
        te = stems == stem
        tr = ~te
        if y_bin[te].sum() == 0 or len(np.unique(y_bin[tr])) < 2:
            continue
        if verify_leak and not checked:
            inter = set(stems[te]) & set(stems[tr])
            assert not inter, f"FUITE : {inter}"
            print(f"    [anti-fuite] fold '{stem}' : test ∩ train = {inter} (OK vide)")
            checked = True
        raw_tr, raw_te = raw_all[tr].copy(), raw_all[te].copy()
        y_tr, y_te = y_bin[tr], y_bin[te]
        Xtr, Xte = fit_transform_fold(raw_tr, raw_te, use_pca)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
        clf.fit(Xtr, y_tr)
        y_true_all.append(y_te)
        y_proba_all.append(clf.predict_proba(Xte)[:, 1])
    if not y_true_all:
        return None
    y_true_all = np.concatenate(y_true_all)
    y_proba_all = np.concatenate(y_proba_all)
    y_pred_all = (y_proba_all >= SEUIL).astype(int)
    auc = roc_auc_score(y_true_all, y_proba_all) if len(np.unique(y_true_all)) > 1 else float("nan")
    return {
        "auc": auc,
        "recall": recall_score(y_true_all, y_pred_all, zero_division=0),
        "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
    }


# ─── Passe principale : pour chaque bloc, préparer les features des 3 variantes ─
print("Calcul LP LOIO — V1 (moyenne) / V2 (moyenne+corr) / V3 (corr seules), "
      "tous blocs × toutes textures ...\n")

results = {}       # results[b][t][variant] = {auc, recall, precision}
raw_desc_stats = {}  # raw_desc_stats[b][t] = (mean±std par classe, noms colonnes valides)
first_leak_check = True

for b in BLOCKS:
    patches = [patches_by_block[b][pid] for pid in canonical_pids]
    mean_vecs = np.stack([p["vecs"].mean(axis=0) for p in patches])
    desc_raw = np.stack([compute_descriptors(p) for p in patches])
    desc_valid, desc_names_valid = drop_allnan_cols(desc_raw, DESC_NAMES)
    nan_frac = np.isnan(desc_valid).mean(axis=0) if desc_valid.size else np.array([])

    dropped = [n for n in DESC_NAMES if n not in desc_names_valid]
    print(f"  {b:<14} : mean_dim={mean_vecs.shape[1]:<4} descripteurs valides={desc_names_valid} "
          f"{'(dropped: ' + str(dropped) + ')' if dropped else ''}")
    if desc_valid.size:
        for n, f in zip(desc_names_valid, nan_frac):
            if f > 0:
                print(f"      NaN partiel dans '{n}' : {f*100:.1f}% des patches (imputé par médiane train/fold)")

    results[b] = {}
    raw_desc_stats[b] = {}
    for t in TEXTURES:
        y_bin = (cat_ids == t).astype(int)
        verify = first_leak_check
        variants = {
            "V1": (mean_vecs, True),
            "V2": (np.concatenate([mean_vecs, desc_valid], axis=1) if desc_valid.size else mean_vecs, True),
            "V3": (desc_valid, False),
        }
        results[b][t] = {}
        for vname, (raw_all, use_pca) in variants.items():
            if vname == "V3" and raw_all.size == 0:
                results[b][t][vname] = None
                continue
            # imputation NaN résiduels : on impute par fold à l'intérieur de loio; ici on ne modifie
            # rien de global, l'imputation train-only se fait dans fit_transform_fold via un wrapper
            r = loio_pooled(raw_all, y_bin, stems, use_pca, verify_leak=(verify and vname == "V1"))
            results[b][t][vname] = r
        if verify:
            first_leak_check = False

        # stats brutes des descripteurs (moyenne ± std) par classe, pour sanity-check (sortie 4)
        if desc_valid.size:
            pos_desc = desc_valid[y_bin == 1]
            neg_desc = desc_valid[y_bin == 0]
            raw_desc_stats[b][t] = {
                "names": desc_names_valid,
                "pos_mean": np.nanmean(pos_desc, axis=0), "pos_std": np.nanstd(pos_desc, axis=0),
                "neg_mean": np.nanmean(neg_desc, axis=0), "neg_std": np.nanstd(neg_desc, axis=0),
            }
    print(f"    ... {b} terminé")

print()

# ─── 1) TABLEAU PRINCIPAL — ΔAUC(V2-V1) par texture × bloc ────────────────────
delta_matrix = np.full((len(TEXTURES), len(BLOCKS)), np.nan)
v1_matrix = np.full((len(TEXTURES), len(BLOCKS)), np.nan)
v2_matrix = np.full((len(TEXTURES), len(BLOCKS)), np.nan)
v3_matrix = np.full((len(TEXTURES), len(BLOCKS)), np.nan)

alerts = []
for i, t in enumerate(TEXTURES):
    for j, b in enumerate(BLOCKS):
        r1 = results[b][t]["V1"]
        r2 = results[b][t]["V2"]
        r3 = results[b][t]["V3"]
        if r1:
            v1_matrix[i, j] = r1["auc"]
        if r2:
            v2_matrix[i, j] = r2["auc"]
            if r1:
                delta_matrix[i, j] = r2["auc"] - r1["auc"]
        if r3:
            v3_matrix[i, j] = r3["auc"]
        # signal d'alarme
        if r1 and r1["auc"] < 0.85:
            for rX, name in [(r2, "V2"), (r3, "V3")]:
                if rX and not np.isnan(rX["auc"]) and rX["auc"] >= 0.98:
                    alerts.append((TNAMES[t], b, name, rX["auc"], r1["auc"]))

print("=" * 100)
print("1) TABLEAU PRINCIPAL — ΔAUC(V2 − V1) par texture × bloc (folds appariés)")
print("=" * 100)
header = f"{'Texture':<14}" + "".join(f"{b:>13}" for b in BLOCKS)
print(header)
for i, t in enumerate(TEXTURES):
    row = f"{TNAMES[t]:<14}" + "".join(f"{delta_matrix[i,j]:>+13.3f}" for j in range(len(BLOCKS)))
    print(row)
print()

if alerts:
    print("⚠ SIGNAUX D'ALARME (V2/V3 AUC ≥ 0.98 alors que V1 < 0.85 — suspecter une fuite) :")
    for tname, b, vname, auc_high, auc_v1 in alerts:
        print(f"   {tname} @ {b} : {vname} AUC={auc_high:.3f} vs V1 AUC={auc_v1:.3f}")
else:
    print("Pas de signal d'alarme (aucune AUC V2/V3 suspecte par rapport à V1).")

# heatmap ΔAUC
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(delta_matrix, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
ax.set_xticks(range(len(BLOCKS))); ax.set_xticklabels(BLOCKS, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(TEXTURES))); ax.set_yticklabels([TNAMES[t] for t in TEXTURES], fontsize=10)
for i in range(len(TEXTURES)):
    for j in range(len(BLOCKS)):
        v = delta_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.1 else "white")
ax.set_title("ΔAUC (V2 moyenne+corrélations − V1 moyenne seule)")
fig.colorbar(im, ax=ax, label="ΔAUC", fraction=0.025, pad=0.01)
plt.tight_layout()
out_fig = OUT / "heatmap_delta_auc.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Heatmap sauvée : {out_fig}\n")

# ─── 2) FOCUS Faisceaux / Strat.sin — V1/V2/V3 aux 7 blocs, 3 métriques ───────
print("=" * 100)
print("2) FOCUS — Faisceaux et Strat.sin : V1/V2/V3 aux 7 blocs (AUC, recall, précision)")
print("=" * 100)
for t in FOCUS_TEXTURES:
    print(f"\n  {TNAMES[t]} :")
    print(f"  {'Bloc':<14}{'V1 AUC':>8}{'V2 AUC':>8}{'V3 AUC':>8}   "
          f"{'V1 rec':>7}{'V2 rec':>7}{'V3 rec':>7}   {'V1 pre':>7}{'V2 pre':>7}{'V3 pre':>7}")
    for b in BLOCKS:
        r1, r2, r3 = results[b][t]["V1"], results[b][t]["V2"], results[b][t]["V3"]
        def g(r, k):
            return r[k] if r else float("nan")
        print(f"  {b:<14}{g(r1,'auc'):>8.3f}{g(r2,'auc'):>8.3f}{g(r3,'auc'):>8.3f}   "
              f"{g(r1,'recall'):>7.3f}{g(r2,'recall'):>7.3f}{g(r3,'recall'):>7.3f}   "
              f"{g(r1,'precision'):>7.3f}{g(r2,'precision'):>7.3f}{g(r3,'precision'):>7.3f}")

# ─── 3) TABLEAU V3 — corrélations seules, AUC par texture × bloc ──────────────
print("\n" + "=" * 100)
print("3) TABLEAU V3 — corrélations SEULES : AUC par texture × bloc "
      "(>0.65 quelque part = signal d'orientation propre)")
print("=" * 100)
print(header)
for i, t in enumerate(TEXTURES):
    row = f"{TNAMES[t]:<14}" + "".join(f"{v3_matrix[i,j]:>13.3f}" for j in range(len(BLOCKS)))
    print(row)
max_v3 = np.nanmax(v3_matrix)
print(f"\nMax AUC V3 (corrélations seules) sur tout le panel : {max_v3:.3f} — "
      f"{'signal d’orientation propre détecté quelque part' if max_v3 > 0.65 else 'pas de signal d’orientation net'}")

# ─── 4) Valeurs brutes des 8 descripteurs (moyenne ± std par classe) ──────────
print("\n" + "=" * 100)
print("4) SANITY-CHECK — valeurs brutes des descripteurs directionnels (moyenne ± std), "
      "texture C vs reste, par bloc")
print("=" * 100)
for t in TEXTURES:
    print(f"\n  {TNAMES[t]} :")
    for b in BLOCKS:
        st = raw_desc_stats[b].get(t)
        if not st:
            continue
        line = "    " + f"{b:<14}"
        for name, pm, ps, nm, ns in zip(st["names"], st["pos_mean"], st["pos_std"], st["neg_mean"], st["neg_std"]):
            if name == "ratio_HV":
                line += f"  {name}: C={pm:+.3f}±{ps:.3f} rest={nm:+.3f}±{ns:.3f}"
        print(line)
print("\nAttendu : Strat.rect/Strat.sin/Faisceaux → |ratio_HV| élevé pour la texture C (anisotropie) ;"
      " Trou/Tot.homogène → ratio_HV ≈ 0 (isotrope). Vérifier ci-dessus.")

# ─── 5) VERDICT factuel par texture ────────────────────────────────────────────
print("\n" + "=" * 100)
print("5) VERDICT — l'orientation est-elle l'info manquante ?")
print("=" * 100)
verdicts = {}
for i, t in enumerate(TEXTURES):
    best_j = np.nanargmax(v2_matrix[i]) if not np.all(np.isnan(v2_matrix[i])) else None
    v1_best = np.nanmax(v1_matrix[i]) if not np.all(np.isnan(v1_matrix[i])) else float("nan")
    v2_best = np.nanmax(v2_matrix[i]) if not np.all(np.isnan(v2_matrix[i])) else float("nan")
    v3_best = np.nanmax(v3_matrix[i]) if not np.all(np.isnan(v3_matrix[i])) else float("nan")
    delta_best = v2_best - v1_best
    if delta_best > 0.03:
        verdict = "orientation = info manquante (V2 > V1)"
    elif v3_best > 0.65:
        verdict = "signal réel mais redondant avec la moyenne (V2≈V1, V3>hasard)"
    else:
        verdict = "pas de signal d'orientation (V3≈hasard, V2≈V1)"
    verdicts[t] = verdict
    print(f"  {TNAMES[t]:<15} V1_best={v1_best:.3f}  V2_best={v2_best:.3f}  V3_best={v3_best:.3f}  "
          f"ΔAUC_best={delta_best:+.3f}  → {verdict}")

# ─── Sauvegarde rapport texte ──────────────────────────────────────────────────
report = OUT / "rapport_correlations_dir.txt"
with open(report, "w") as f:
    f.write("Test — corrélations directionnelles : l'orientation débloque-t-elle les textures ?\n")
    f.write(f"H5 : {H5_PATH}\n\n")
    f.write("Verdict par texture :\n")
    for t in TEXTURES:
        f.write(f"  {TNAMES[t]:<16} {verdicts[t]}\n")
    f.write("\nVoir stdout du script pour le détail (tableaux 1-4, checklist, alertes).\n")

print(f"\nRapport texte sauvé : {report}")
print(f"Toutes les sorties dans : {OUT}/")
