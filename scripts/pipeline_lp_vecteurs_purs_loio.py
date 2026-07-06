#!/usr/bin/env python3
"""
TEST — LP sur VECTEURS LOCAUX PURS (sans moyennage), LOIO par image

Protocole :
  - Extraction vecteurs locaux [n_pos, D] par patch, par bloc (image entière)
  - LP LOIO par image (anti-fuite), one-vs-rest + multiclasse
  - Vote dur / vote souple par patch
  - Diagnostics : patches unanimes mal classés, patches divisés + carte spatiale

Blocs : block_0 (early), block_7 (intermédiaire), stage_3_fpn (fusionné)
Config : PCA(50), LR(C=1, balanced), SEED=42, MAX_VEC_PATCH=64
Output : test_vecteurs_purs/
"""

import os
import sys
import logging
import tempfile
import zipfile
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

# ─── Paths ───────────────────────────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent
sys.path.insert(0, str(_ROOT / "TextureSAM" / "sam2"))

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ─── Config ──────────────────────────────────────────────────────────────────
H5_PATH   = _ROOT / "data/feature_database/database_meb_ouassim.h5"
IMG_DIR   = _ROOT / "Image_Ouassim"
CKPT_PATH = _ROOT / "checkpoints/sam2.1_hiera_small_1.pt"
CKPT_DIR  = _ROOT / "checkpoints/sam2.1_hiera_small_1"
OUT_DIR   = _ROOT / "test_vecteurs_purs"

TEXTURES     = [1, 3, 4, 5, 6, 7, 9]
TNAMES       = {1: "Tot.homogène", 3: "Faisceaux", 4: "Filaments",
                5: "Strat.rect",  6: "Strat.sin", 7: "Granuleux", 9: "Trou"}
# Référence : meilleur recall par moyenne (best_block_loio.py)
REF_RECALL   = {1: 1.00, 3: 0.79, 4: 0.84, 5: 0.50, 6: 0.52, 7: 0.86, 9: 0.74}
BLOCS_STUDY  = ["block_0", "block_7", "stage_3_fpn"]
PCA_DIM      = 50
LP_C         = 1.0
MAX_VEC_PATCH = 64
SEED         = 42
IMG_SIZE     = 1024

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Colors for the 7 textures + unknown (-1)
TEX_COLORS = {
    1:  "#2ecc71", 3:  "#3498db", 4:  "#e74c3c",
    5:  "#9b59b6", 6:  "#f39c12", 7:  "#1abc9c",
    9:  "#e67e22", -1: "#cccccc",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Model ───────────────────────────────────────────────────────────────────

def _build_encoder() -> ImageEncoder:
    trunk = Hiera(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000
        ),
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0,
        fpn_interp_model="nearest", fuse_type="sum",
        fpn_top_down_levels=[2, 3],
    )
    return ImageEncoder(trunk=trunk, neck=neck, scalp=1)


def _load_model(device: str) -> ImageEncoder:
    enc = _build_encoder()
    tmp = None
    if CKPT_PATH.is_file():
        sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    elif CKPT_DIR.is_dir():
        arch = CKPT_DIR / "archive" if (CKPT_DIR / "archive").is_dir() else CKPT_DIR
        tf = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tf.close(); tmp = tf.name
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for fp in sorted(arch.rglob("*")):
                if fp.is_file():
                    info = zipfile.ZipInfo(str(fp.relative_to(arch.parent)))
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    with open(fp, "rb") as fh:
                        zf.writestr(info, fh.read())
        sd = torch.load(tmp, map_location="cpu", weights_only=False)
    else:
        log.warning("Checkpoint introuvable — poids aléatoires")
        return enc.to(device).eval()
    if tmp:
        os.unlink(tmp)
    sd = sd.get("model", sd)
    prefix = "image_encoder."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    m, u = enc.load_state_dict(sd, strict=False)
    if m or u:
        log.warning("Checkpoint partiel (%d manquants, %d inattendus)", len(m), len(u))
    else:
        log.info("Checkpoint chargé OK")
    return enc.to(device).eval()


def _register_hooks(enc: ImageEncoder, blocs: list[str]):
    """Pose des hooks uniquement sur les blocs demandés."""
    captured = {}
    handles  = []
    for i, block in enumerate(enc.trunk.blocks):
        key = f"block_{i}"
        if key in blocs:
            def _bh(m, inp, out, k=key):
                captured[k] = out.detach()
            handles.append(block.register_forward_hook(_bh))
    # FPN convs : conv[0]=stage_4, [1]=stage_3, [2]=stage_2, [3]=stage_1
    fpn_map = {0: "stage_4_fpn", 1: "stage_3_fpn", 2: "stage_2_fpn", 3: "stage_1_fpn"}
    for ci, key in fpn_map.items():
        if key in blocs:
            def _fh(m, inp, out, k=key):
                captured[k] = out.detach().permute(0, 2, 3, 1)
            handles.append(enc.neck.convs[ci].register_forward_hook(_fh))
    return captured, handles


# ─── Extraction locale ───────────────────────────────────────────────────────

def _preprocess(img_path: Path, device: str):
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - _MEAN) / _STD
    return x.unsqueeze(0).to(device), orig_h, orig_w


def _patch_region(feat_hw, orig_h, orig_w, x_min, y_min, x_max, y_max):
    """Coordonnées de la région du patch dans la feature map."""
    H_f, W_f = feat_hw
    sx = W_f / orig_w
    sy = H_f / orig_h
    fx1 = max(0, int(x_min * sx))
    fy1 = max(0, int(y_min * sy))
    fx2 = min(W_f, max(fx1 + 1, int(x_max * sx)))
    fy2 = min(H_f, max(fy1 + 1, int(y_max * sy)))
    return fy1, fy2, fx1, fx2


def _extract_local_vecs(feat_map, fy1, fy2, fx1, fx2, pid, seed_offset):
    """
    Extrait les vecteurs locaux d'une région de feature map.
    Retourne vecs (n, C), rows (n,), cols (n,) relatifs à la région (0-based).
    """
    region = feat_map[fy1:fy2, fx1:fx2, :]  # (h, w, C)
    h, w, C = region.shape
    vecs = region.reshape(-1, C).cpu().numpy().astype(np.float32)
    # Positions relatives à la région du patch
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows = rr.ravel()
    cols = cc.ravel()
    # L2-norm
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    # Sous-échantillonnage déterministe par patch
    if len(vecs) > MAX_VEC_PATCH:
        rng = np.random.default_rng(SEED + seed_offset)
        idx = rng.choice(len(vecs), MAX_VEC_PATCH, replace=False)
        vecs = vecs[idx]
        rows = rows[idx]
        cols = cols[idx]
    return vecs, rows, cols, h, w


# ─── Étape 0 : Extraction ────────────────────────────────────────────────────

def extract_all_local_vectors():
    """
    Pour chaque image, forward pass → extraction vecteurs locaux par patch.
    Retourne local_data[bloc] = dict of arrays, + patch_meta.
    """
    log.info("═" * 60)
    log.info("ÉTAPE 0 — Extraction vecteurs locaux")
    log.info("═" * 60)

    # Chargement métadonnées H5
    with h5py.File(H5_PATH, "r") as f:
        all_cats  = f["metadata/category_ids"][:]
        all_imgs  = np.array([x.decode() for x in f["metadata/image_names"][:]])
        all_pos   = f["metadata/positions"][:]  # [x_min, y_min, x_max, y_max]

    mask = np.isin(all_cats, TEXTURES)
    cats  = all_cats[mask]
    imgs  = all_imgs[mask]
    pos   = all_pos[mask]
    stems = np.array([n.replace(".tif", "") for n in imgs])
    N = int(mask.sum())
    log.info("%d patches × %d textures × %d images", N, len(TEXTURES),
             len(np.unique(stems)))

    # Patch metadata
    patch_meta = {
        "tex":   cats,
        "img":   stems,
        "xmin":  pos[:, 0].astype(int),
        "ymin":  pos[:, 1].astype(int),
        "xmax":  pos[:, 2].astype(int),
        "ymax":  pos[:, 3].astype(int),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device : %s", device)
    enc = _load_model(device)
    captured, handles = _register_hooks(enc, BLOCS_STUDY)

    # Accumulateurs — métadonnées séparées par bloc (n_tokens/patch varie selon résolution)
    local_vecs = {b: [] for b in BLOCS_STUDY}
    local_rows = {b: [] for b in BLOCS_STUDY}
    local_cols = {b: [] for b in BLOCS_STUDY}
    local_tex  = {b: [] for b in BLOCS_STUDY}
    local_pid  = {b: [] for b in BLOCS_STUDY}
    local_img  = {b: [] for b in BLOCS_STUDY}
    feat_h_per_patch = {b: [] for b in BLOCS_STUDY}
    feat_w_per_patch = {b: [] for b in BLOCS_STUDY}

    # Regrouper patches par image
    by_img = defaultdict(list)
    for pid in range(N):
        by_img[stems[pid]].append(pid)

    unique_imgs = sorted(by_img.keys())
    n_imgs = len(unique_imgs)

    for i_img, stem in enumerate(unique_imgs):
        img_name = stem + ".tif"
        img_path = IMG_DIR / img_name
        if not img_path.exists():
            log.warning("[%d/%d] Image introuvable : %s", i_img+1, n_imgs, img_name)
            continue

        try:
            tensor, orig_h, orig_w = _preprocess(img_path, device)
        except Exception as e:
            log.error("[%d/%d] Prétraitement %s : %s", i_img+1, n_imgs, img_name, e)
            continue

        captured.clear()
        with torch.no_grad():
            enc(tensor)

        pids_img = by_img[stem]
        n_ok = 0

        for pid in pids_img:
            x_min = patch_meta["xmin"][pid]
            y_min = patch_meta["ymin"][pid]
            x_max = patch_meta["xmax"][pid]
            y_max = patch_meta["ymax"][pid]

            # Vérifier que tous les blocs ont leurs features
            if not all(b in captured for b in BLOCS_STUDY):
                log.warning("Bloc manquant pour image %s", stem)
                continue

            tmp_vecs = {}
            for b in BLOCS_STUDY:
                feat = captured[b][0]  # (H_f, W_f, C)
                H_f, W_f, _ = feat.shape
                fy1, fy2, fx1, fx2 = _patch_region(
                    (H_f, W_f), orig_h, orig_w, x_min, y_min, x_max, y_max
                )
                v, r, c, fh, fw = _extract_local_vecs(feat, fy1, fy2, fx1, fx2,
                                                        pid, seed_offset=pid)
                tmp_vecs[b] = (v, r, c, fh, fw)

            for b in BLOCS_STUDY:
                v, r, c, fh, fw = tmp_vecs[b]
                n_v = len(v)
                local_vecs[b].append(v)
                local_rows[b].append(r)
                local_cols[b].append(c)
                feat_h_per_patch[b].append(fh)
                feat_w_per_patch[b].append(fw)
                local_tex[b].extend([cats[pid]] * n_v)
                local_pid[b].extend([pid] * n_v)
                local_img[b].extend([stem] * n_v)
            n_ok += 1

        log.info("[%d/%d] %s : %d patches", i_img+1, n_imgs, stem, n_ok)

    for h in handles:
        h.remove()

    data = {}
    for b in BLOCS_STUDY:
        data[b] = {
            "vecs": np.concatenate(local_vecs[b], axis=0),
            "rows": np.concatenate(local_rows[b], axis=0).astype(np.int16),
            "cols": np.concatenate(local_cols[b], axis=0).astype(np.int16),
            "tex":  np.array(local_tex[b], dtype=np.int16),
            "pid":  np.array(local_pid[b], dtype=np.int32),
            "img":  np.array(local_img[b], dtype=object),
            "feat_h": np.array(feat_h_per_patch[b], dtype=np.int16),  # per patch
            "feat_w": np.array(feat_w_per_patch[b], dtype=np.int16),
        }
        D = data[b]["vecs"].shape[1]
        N_v = data[b]["vecs"].shape[0]
        log.info("  %s : %d vecteurs, dim=%d", b, N_v, D)

    return data, patch_meta


# ─── Étape 1 : LP LOIO ───────────────────────────────────────────────────────

def run_loio(data: dict, patch_meta: dict):
    """
    LP LOIO par image, one-vs-rest (recall) + multiclasse (diagnostics).
    Retourne results (tableau) et all_diag (diagnostics par bloc).
    """
    log.info("═" * 60)
    log.info("ÉTAPE 1 — LP LOIO par image (anti-fuite)")
    log.info("═" * 60)

    # results[bloc][tex] = {"vec": [...], "dur": [...], "spl": [...]}
    results  = {b: {t: {"vec": [], "dur": [], "spl": []} for t in TEXTURES}
                for b in BLOCS_STUDY}
    all_diag = {b: [] for b in BLOCS_STUDY}  # diagnostics multiclasse

    for b in BLOCS_STUDY:
        log.info("── Bloc : %s ──", b)
        bd = data[b]
        vecs = bd["vecs"]
        tex  = bd["tex"]
        pids = bd["pid"]
        imgs = bd["img"]
        rows = bd["rows"]
        cols = bd["cols"]

        unique_imgs = sorted(set(imgs))
        anti_fuite_shown = False

        for stem in unique_imgs:
            te = imgs == stem
            tr = ~te

            # ── Contrôle anti-fuite ──────────────────────────────────────────
            test_pids_set  = set(int(p) for p in pids[te])
            train_pids_set = set(int(p) for p in pids[tr])
            overlap = test_pids_set & train_pids_set
            if overlap:
                log.error("FUITE DÉTECTÉE! patch_ids partagés : %s → STOP", overlap)
                sys.exit(1)
            if not anti_fuite_shown:
                log.info("  Anti-fuite OK pour image '%s' : patch_id(test)∩patch_id(train)=∅", stem)
                anti_fuite_shown = True

            X_tr_raw = vecs[tr]
            X_te_raw = vecs[te]
            y_tr_tex = tex[tr]
            y_te_tex = tex[te]

            if X_tr_raw.shape[1] > PCA_DIM:
                pca    = PCA(n_components=PCA_DIM, random_state=SEED)
                X_tr   = pca.fit_transform(X_tr_raw)
                X_te   = pca.transform(X_te_raw)
            else:
                X_tr, X_te = X_tr_raw.copy(), X_te_raw.copy()

            # ── One-vs-rest par texture ───────────────────────────────────────
            for C in TEXTURES:
                y_tr = (y_tr_tex == C).astype(int)
                y_te = (y_te_tex == C).astype(int)

                if y_te.sum() == 0:
                    continue
                if len(np.unique(y_tr)) < 2:
                    continue

                clf = LogisticRegression(
                    C=LP_C, class_weight="balanced",
                    max_iter=500, random_state=SEED, solver="lbfgs",
                )
                clf.fit(X_tr, y_tr)
                proba_v = clf.predict_proba(X_te)[:, 1]
                pred_v  = (proba_v >= 0.5).astype(int)

                # Recall vecteur
                rc_vec = recall_score(y_te, pred_v, zero_division=0.0)
                results[b][C]["vec"].append(rc_vec)

                # Patch-level vote
                te_pids = pids[te]
                unique_te_pids = np.unique(te_pids)
                y_patch_true, y_dur, y_spl = [], [], []

                for pid in unique_te_pids:
                    idx = te_pids == pid
                    if idx.sum() == 0:
                        continue
                    true_lbl = int(y_te[idx][0])
                    p_proba  = proba_v[idx]
                    p_pred   = pred_v[idx]
                    y_patch_true.append(true_lbl)
                    y_dur.append(int(p_pred.mean() >= 0.5))
                    y_spl.append(int(p_proba.mean() >= 0.5))

                y_patch_true = np.array(y_patch_true)
                if y_patch_true.sum() > 0:
                    results[b][C]["dur"].append(
                        recall_score(y_patch_true, y_dur, zero_division=0.0))
                    results[b][C]["spl"].append(
                        recall_score(y_patch_true, y_spl, zero_division=0.0))

            # ── LP multiclasse (diagnostics) ──────────────────────────────────
            tex_in_te = np.unique(y_te_tex)
            tex_in_tr = np.unique(y_tr_tex)
            if len(tex_in_tr) < 2:
                continue

            tex_to_i = {t: i for i, t in enumerate(TEXTURES)}
            y_tr_mc  = np.array([tex_to_i[t] for t in y_tr_tex])
            y_te_mc  = np.array([tex_to_i[t] for t in y_te_tex])

            clf_mc = LogisticRegression(
                C=LP_C, class_weight="balanced", max_iter=500,
                random_state=SEED, solver="lbfgs",
            )
            try:
                clf_mc.fit(X_tr, y_tr_mc)
            except Exception as e:
                log.warning("  LP multiclasse échoué (%s, %s) : %s", b, stem, e)
                continue

            proba_mc = clf_mc.predict_proba(X_te)  # (N_te, n_tex)

            # Classes connues du classifier (peut être < 7 si classe absente du train)
            known_classes_idx = clf_mc.classes_  # indices dans TEXTURES
            full_proba = np.zeros((len(proba_mc), len(TEXTURES)))
            full_proba[:, known_classes_idx] = proba_mc

            te_pids = pids[te]
            te_rows = rows[te]
            te_cols = cols[te]

            for pid in np.unique(te_pids):
                idx = te_pids == pid
                p_proba_mc = full_proba[idx]  # (n_v, 7)
                p_rows     = te_rows[idx]
                p_cols     = te_cols[idx]

                mean_proba      = p_proba_mc.mean(axis=0)
                voted_idx       = int(np.argmax(mean_proba))
                voted_class     = TEXTURES[voted_idx]

                # taux_accord : fraction votant la classe majoritaire (par vecteur)
                vec_voted_idx   = np.argmax(p_proba_mc, axis=1)
                counts          = np.bincount(vec_voted_idx, minlength=len(TEXTURES))
                majority_count  = int(counts.max())
                taux_accord     = majority_count / len(vec_voted_idx)
                majority_tex    = TEXTURES[int(np.argmax(counts))]

                true_tex = int(tex[te][idx][0])

                all_diag[b].append({
                    "patch_id":      int(pid),
                    "texture_vraie": true_tex,
                    "texture_votee": voted_class,
                    "taux_accord":   taux_accord,
                    "image":         stem,
                    "per_vec_voted": [TEXTURES[vi] for vi in vec_voted_idx.tolist()],
                    "per_vec_rows":  p_rows.tolist(),
                    "per_vec_cols":  p_cols.tolist(),
                })

        # Résumé par bloc
        log.info("  Résultats %s :", b)
        for C in TEXTURES:
            r_v = results[b][C]["vec"]
            r_d = results[b][C]["dur"]
            r_s = results[b][C]["spl"]
            mv  = np.mean(r_v) if r_v else float("nan")
            ms  = np.mean(r_s) if r_s else float("nan")
            log.info("    t%d %-14s  vec=%.3f  dur=%.3f  spl=%.3f  (ref=%.2f)",
                     C, TNAMES[C],
                     mv, np.mean(r_d) if r_d else float("nan"), ms,
                     REF_RECALL[C])

    return results, all_diag


# ─── Sorties quantitatives ───────────────────────────────────────────────────

def output_1_table(results: dict):
    """Tableau par texture × bloc."""
    log.info("Sortie 1 — Tableau quantitatif")
    header = (f"{'Texture':<16} {'Bloc':<14} {'Réf(moy)':<10} "
              f"{'recall_vec':<12} {'vote_dur':<10} {'vote_souple':<12} {'Δ(spl−réf)':<10}")
    lines = [header, "─" * len(header)]

    rows_data = []
    for C in TEXTURES:
        for b in BLOCS_STUDY:
            r_v  = results[b][C]["vec"]
            r_d  = results[b][C]["dur"]
            r_s  = results[b][C]["spl"]
            mv   = np.mean(r_v) if r_v else float("nan")
            sv   = np.std(r_v)  if r_v else float("nan")
            md   = np.mean(r_d) if r_d else float("nan")
            ms   = np.mean(r_s) if r_s else float("nan")
            ss   = np.std(r_s)  if r_s else float("nan")
            ref  = REF_RECALL[C]
            delta = ms - ref if not np.isnan(ms) else float("nan")
            line = (f"{TNAMES[C]:<16} {b:<14} {ref:<10.2f} "
                    f"{mv:.3f}±{sv:.3f}  {md:.3f}      {ms:.3f}±{ss:.3f}  {delta:+.3f}")
            lines.append(line)
            rows_data.append({
                "texture": TNAMES[C], "tex_id": C, "bloc": b,
                "ref_moy": ref, "recall_vec": mv, "std_vec": sv,
                "vote_dur": md, "vote_souple": ms, "std_souple": ss,
                "delta_spl_ref": delta,
            })

    table_txt = "\n".join(lines)
    print("\n" + table_txt)
    (OUT_DIR / "tableau_vecteurs_purs.txt").write_text(table_txt)

    # CSV
    import csv
    csv_path = OUT_DIR / "tableau_vecteurs_purs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_data[0].keys())
        w.writeheader()
        w.writerows(rows_data)
    log.info("  → %s", csv_path)
    return rows_data


def output_2_graph(results: dict, rows_data: list):
    """Graphe par texture : recall selon le bloc, moyenne vs vote souple."""
    log.info("Sortie 2 — Graphe comparatif")

    order = sorted(TEXTURES, key=lambda t: -REF_RECALL[t])
    n_tex = len(order)
    x = np.arange(n_tex)
    width = 0.22

    fig, ax = plt.subplots(figsize=(13, 6))
    colors_ref    = "#95a5a6"
    colors_blocks = {"block_0": "#e74c3c", "block_7": "#3498db", "stage_3_fpn": "#2ecc71"}
    offsets = {"block_0": -1, "block_7": 0, "stage_3_fpn": 1}

    # Référence (moyenne agrégée)
    ref_vals = [REF_RECALL[t] for t in order]
    ax.bar(x - 1.5 * width, ref_vals, width=width, color=colors_ref,
           alpha=0.7, label="Réf. (moyenne agrégée)", edgecolor="white")

    for b in BLOCS_STUDY:
        row_map = {r["tex_id"]: r for r in rows_data if r["bloc"] == b}
        spl_means = [row_map[t]["vote_souple"] if t in row_map else float("nan") for t in order]
        spl_stds  = [row_map[t]["std_souple"]  if t in row_map else 0.0         for t in order]
        off = offsets[b]
        ax.bar(x + off * width, spl_means, width=width, color=colors_blocks[b],
               alpha=0.85, label=f"{b} (vote souple)", edgecolor="white")
        ax.errorbar(x + off * width, spl_means, yerr=spl_stds,
                    fmt="none", capsize=3, ecolor="black", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([TNAMES[t] for t in order], fontsize=10, rotation=20, ha="right")
    ax.set_ylabel("Recall LOIO (one-vs-rest)", fontsize=11)
    ax.set_ylim(0, 1.22)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("LP — Vecteurs locaux purs vs Moyenne agrégée\n"
                 "Vote souple (moyenne des probas) par bloc, LOIO par image",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    out = OUT_DIR / "graphe_vote_souple_vs_moy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  → %s", out)


# ─── Sorties diagnostiques ────────────────────────────────────────────────────

def _load_patch_img(patch_meta: dict, pid: int):
    """Charge le crop du patch original depuis l'image source."""
    from PIL import Image as PILImage
    stem = patch_meta["img"][pid]
    img_path = IMG_DIR / (stem + ".tif")
    if not img_path.exists():
        return None
    img = PILImage.open(img_path).convert("L")
    x1 = patch_meta["xmin"][pid]; y1 = patch_meta["ymin"][pid]
    x2 = patch_meta["xmax"][pid]; y2 = patch_meta["ymax"][pid]
    crop = np.array(img)[y1:y2, x1:x2]
    return crop


def output_3_outliers(all_diag: dict, patch_meta: dict):
    """Patches unanimes mais mal classés (taux_accord ≥ 0.8 et votée ≠ vraie)."""
    log.info("Sortie 3 — Patches unanimes mais mal classés")

    out_csv = OUT_DIR / "outliers_unanimes.csv"
    out_vis = OUT_DIR / "outliers_unanimes_visuel"
    out_vis.mkdir(exist_ok=True)

    import csv
    csv_rows = []

    for b in BLOCS_STUDY:
        diag = all_diag[b]
        outliers = [d for d in diag
                    if d["taux_accord"] >= 0.8 and d["texture_votee"] != d["texture_vraie"]]
        outliers.sort(key=lambda d: -d["taux_accord"])
        log.info("  %s : %d patches unanimes mal classés", b, len(outliers))

        # Déduplique par patch_id (même patch peut apparaître dans plusieurs folds ?)
        seen = set()
        unique_outliers = []
        for d in outliers:
            if d["patch_id"] not in seen:
                seen.add(d["patch_id"])
                unique_outliers.append(d)

        for rank, d in enumerate(unique_outliers[:20]):  # top 20 par bloc
            pid = d["patch_id"]
            crop = _load_patch_img(patch_meta, pid)

            fig, axes = plt.subplots(1, 2 if crop is not None else 1,
                                     figsize=(7 if crop is not None else 3.5, 3.5))
            if crop is not None and not isinstance(axes, np.ndarray):
                axes = [axes, None]
            elif crop is None:
                axes = [axes]

            if crop is not None:
                axes[0].imshow(crop, cmap="gray")
                axes[0].set_title(f"Patch #{pid}", fontsize=9)
                axes[0].axis("off")

            info_ax = axes[1] if len(axes) > 1 and axes[1] is not None else axes[0]
            info_ax.axis("off")
            info_ax.text(
                0.5, 0.5,
                f"Annoté : {TNAMES[d['texture_vraie']]} (t{d['texture_vraie']})\n"
                f"Voté   : {TNAMES[d['texture_votee']]} (t{d['texture_votee']})\n"
                f"Accord : {d['taux_accord']:.2f}\n"
                f"Image  : {d['image'][:30]}\n"
                f"Bloc   : {b}",
                transform=info_ax.transAxes,
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#f39c12", lw=1.2),
            )
            plt.tight_layout()
            fname = out_vis / f"{b}_rank{rank:02d}_pid{pid}.png"
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close()

            csv_rows.append({
                "bloc": b, "rank": rank,
                "patch_id": pid,
                "texture_vraie": TNAMES[d["texture_vraie"]],
                "texture_votee": TNAMES[d["texture_votee"]],
                "taux_accord": round(d["taux_accord"], 4),
                "image": d["image"],
            })

    with open(out_csv, "w", newline="") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
    log.info("  → %s  (%d lignes)", out_csv, len(csv_rows))
    log.info("  → %s  (visuels)", out_vis)


def output_4_divises_spatial(all_diag: dict, patch_meta: dict):
    """Patches divisés (taux_accord ≤ 0.6) + carte spatiale des votes."""
    log.info("Sortie 4 — Patches divisés + cartes spatiales")

    out_vis = OUT_DIR / "patches_divises_spatial"
    out_vis.mkdir(exist_ok=True)

    tex_list = TEXTURES
    tex_to_color = {t: TEX_COLORS[t] for t in tex_list}
    tex_to_color[-1] = TEX_COLORS[-1]

    for b in BLOCS_STUDY:
        diag = all_diag[b]
        divises = [d for d in diag if d["taux_accord"] <= 0.6]
        divises.sort(key=lambda d: d["taux_accord"])  # les plus divisés en premier
        log.info("  %s : %d patches divisés", b, len(divises))

        seen = set()
        unique_div = []
        for d in divises:
            if d["patch_id"] not in seen:
                seen.add(d["patch_id"])
                unique_div.append(d)

        for rank, d in enumerate(unique_div[:16]):
            pid = d["patch_id"]
            crop = _load_patch_img(patch_meta, pid)

            # Reconstituer la carte spatiale des votes
            per_rows = d["per_vec_rows"]
            per_cols = d["per_vec_cols"]
            per_voted = d["per_vec_voted"]
            if not per_rows:
                continue

            max_r = max(per_rows) + 1
            max_c = max(per_cols) + 1
            vote_map = np.full((max_r, max_c), -1, dtype=np.int16)
            for r, c, tv in zip(per_rows, per_cols, per_voted):
                vote_map[r, c] = tv

            # Colormap RGB
            color_map_rgb = np.ones((max_r, max_c, 3))
            for r in range(max_r):
                for c in range(max_c):
                    tv = vote_map[r, c]
                    hex_c = TEX_COLORS.get(tv, "#cccccc")
                    rgb = tuple(int(hex_c.lstrip("#")[i:i+2], 16) / 255
                                for i in (0, 2, 4))
                    color_map_rgb[r, c] = rgb

            ncols = 3 if crop is not None else 2
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            ax_idx = 0

            if crop is not None:
                axes[ax_idx].imshow(crop, cmap="gray")
                axes[ax_idx].set_title(f"Crop patch #{pid}", fontsize=9)
                axes[ax_idx].axis("off")
                ax_idx += 1

            axes[ax_idx].imshow(color_map_rgb, aspect="auto", interpolation="nearest")
            axes[ax_idx].set_title(
                f"Votes spatiaux ({b})\naccord={d['taux_accord']:.2f}", fontsize=9)
            axes[ax_idx].set_xlabel("col (feat map)")
            axes[ax_idx].set_ylabel("row (feat map)")
            ax_idx += 1

            # Légende
            patches = [mpatches.Patch(color=TEX_COLORS[t], label=f"t{t} {TNAMES[t]}")
                       for t in TEXTURES]
            patches.append(mpatches.Patch(color=TEX_COLORS[-1], label="non-échantillonné"))
            axes[ax_idx - 1].legend(handles=patches, loc="upper right",
                                    fontsize=7, framealpha=0.85)

            axes[ax_idx].axis("off")
            axes[ax_idx].text(
                0.5, 0.6,
                f"Annoté : {TNAMES[d['texture_vraie']]} (t{d['texture_vraie']})\n"
                f"Voté   : {TNAMES[d['texture_votee']]} (t{d['texture_votee']})\n"
                f"Accord : {d['taux_accord']:.2f}\n"
                f"# vecteurs : {len(per_rows)}\n"
                f"Image : {d['image'][:25]}",
                transform=axes[ax_idx].transAxes,
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#2ecc71", lw=1.2),
            )
            plt.suptitle(
                f"Patch divisé — {b}  (taux_accord={d['taux_accord']:.2f}  "
                f"vraitex={TNAMES[d['texture_vraie']]})",
                fontsize=10,
            )
            plt.tight_layout()
            fname = out_vis / f"{b}_rank{rank:02d}_pid{pid}.png"
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close()

    log.info("  → %s", out_vis)


def output_5_histogram(all_diag: dict):
    """Histogramme du taux_accord par texture, par bloc."""
    log.info("Sortie 5 — Histogramme taux_accord par texture")

    for b in BLOCS_STUDY:
        diag = all_diag[b]
        if not diag:
            continue

        fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
        axes = axes.ravel()

        order = sorted(TEXTURES, key=lambda t: -REF_RECALL[t])
        for i, t in enumerate(order):
            pds = [d["taux_accord"] for d in diag if d["texture_vraie"] == t]
            ax = axes[i]
            if pds:
                ax.hist(pds, bins=20, range=(0, 1),
                        color=TEX_COLORS[t], edgecolor="white", alpha=0.85)
                ax.axvline(np.mean(pds), color="black", lw=1.5, ls="--",
                           label=f"moy={np.mean(pds):.2f}")
                ax.legend(fontsize=8)
            ax.set_title(f"t{t} {TNAMES[t]}", fontsize=9)
            ax.set_xlabel("taux_accord", fontsize=8)
            ax.set_xlim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)

        axes[-1].axis("off")
        fig.suptitle(f"Distribution taux_accord (cohérence interne) — {b}\n"
                     "1.0 = patches unanimes, 0.5 = patches divisés",
                     fontsize=11)
        plt.tight_layout()
        out = OUT_DIR / f"hist_taux_accord_{b}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  → %s", out)


def output_6_antifuite_report(data: dict):
    """Rapport anti-fuite : confirme patch_id(test)∩patch_id(train)=∅."""
    log.info("Sortie 6 — Rapport anti-fuite")
    b = BLOCS_STUDY[0]
    bd = data[b]
    pids = bd["pid"]
    imgs = bd["img"]

    unique_imgs = sorted(set(imgs))
    any_leak = False

    lines = ["RAPPORT ANTI-FUITE\n" + "─" * 50]
    for stem in unique_imgs[:5]:  # vérifier les 5 premiers folds
        te = imgs == stem
        tr = ~te
        test_pids  = set(int(p) for p in pids[te])
        train_pids = set(int(p) for p in pids[tr])
        overlap = test_pids & train_pids
        status = "OK (∅)" if not overlap else f"FUITE! {len(overlap)} patches partagés"
        lines.append(f"  Fold '{stem}' : test={len(test_pids)} patches, "
                     f"train={len(train_pids)} patches → {status}")
        if overlap:
            any_leak = True

    lines.append("\n" + ("⚠ FUITE DÉTECTÉE" if any_leak else "✓ Aucune fuite détectée"))
    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "antifuite_report.txt").write_text(report)
    log.info("  → %s", OUT_DIR / "antifuite_report.txt")


# ─── Verdict ─────────────────────────────────────────────────────────────────

def print_verdict(rows_data: list):
    print("\n" + "═" * 65)
    print("VERDICT — vote souple vs moyenne agrégée (par texture, meilleur bloc)")
    print("═" * 65)
    # Meilleur bloc par texture (vote souple)
    by_tex = defaultdict(list)
    for r in rows_data:
        by_tex[r["tex_id"]].append(r)
    for C in sorted(TEXTURES, key=lambda t: -REF_RECALL[t]):
        rows = by_tex[C]
        best = max(rows, key=lambda r: r["vote_souple"] if not np.isnan(r["vote_souple"]) else -1)
        ref = REF_RECALL[C]
        ms  = best["vote_souple"]
        d   = ms - ref if not np.isnan(ms) else float("nan")
        verdict = ("vote_souple > moy ✓ (moins d'agrégation = meilleur)"
                   if d > 0.02 else
                   "vote_souple ≈ moy ~ (agrégation n'est pas le levier)"
                   if d > -0.02 else
                   "vote_souple < moy ✗ (vecteurs bruts trop bruités)")
        print(f"  t{C} {TNAMES[C]:<16} ref={ref:.2f}  spl={ms:.3f}  Δ={d:+.3f}  → {verdict}")
    print("═" * 65)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output : %s", OUT_DIR)

    data, patch_meta = extract_all_local_vectors()
    results, all_diag = run_loio(data, patch_meta)

    rows_data = output_1_table(results)
    output_2_graph(results, rows_data)
    output_3_outliers(all_diag, patch_meta)
    output_4_divises_spatial(all_diag, patch_meta)
    output_5_histogram(all_diag)
    output_6_antifuite_report(data)
    print_verdict(rows_data)

    log.info("\nTous les outputs dans : %s", OUT_DIR)


if __name__ == "__main__":
    main()
