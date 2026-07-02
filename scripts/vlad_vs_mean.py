"""
VLAD vs Mean Pooling — Test tranchant
Block b7, K=8, PCA=50, LOIO one-vs-rest par texture.
Vocab global = borne OPTIMISTE (à noter dans le verdict).
"""

import sys, re, struct
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT       = Path('/home/aidouni/meb_texture_seg')
SAM2DIR    = ROOT / 'TextureSAM/sam2'
CKPT       = ROOT / 'checkpoints/sam2.1_hiera_small_1.pt'
PATCH_ROOT = ROOT / 'PatchTagger_Output/patches'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUT_DIR    = ROOT / 'output_ouassim/vlad_vs_mean'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SAM2DIR))
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

BLOCK    = 7
D        = 384
K        = 8
PCA_DIM  = 50
TEXTURES = [1, 3, 4, 5, 6, 7, 9]
TNAMES   = {1: "Tot.homogène", 3: "Faisceaux", 4: "Filaments",
            5: "Strat.rect", 6: "Strat.sin", 7: "Granuleux", 9: "Trou"}
TUNIFORM = {1, 9}
SEED     = 42
EPS      = 1e-8
C_GRID   = [0.01, 0.1, 1.0, 10.0]

ORIG_H, ORIG_W = 768, 1280
IMG_SIZE = 1024
PATCH_PX = 128
fH       = IMG_SIZE / ORIG_H   # 1.333
fW       = IMG_SIZE / ORIG_W   # 0.800
STRIDE   = 16                  # effective stride at b7 (stage 3)

MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─── I/O ──────────────────────────────────────────────────────────────────────

def read_tiff_gray(path):
    with open(path, 'rb') as f:
        data = f.read()
    bo  = '<' if data[:2] == b'II' else '>'
    ifd = struct.unpack(bo + 'I', data[4:8])[0]
    pos = ifd
    n   = struct.unpack(bo + 'H', data[pos:pos + 2])[0]
    pos += 2
    tags = {}
    for _ in range(n):
        e = data[pos:pos + 12]; pos += 12
        tag, dtype, _ = struct.unpack(bo + 'HHI', e[:8]); v = e[8:12]
        if dtype == 3:
            v = struct.unpack(bo + 'H', v[:2])[0]
        elif dtype == 4:
            v = struct.unpack(bo + 'I', v)[0]
        tags[tag] = v
    w, h = tags[256], tags[257]
    with open(path, 'rb') as f:
        f.seek(tags[273])
        raw = np.frombuffer(f.read(h * w), dtype=np.uint8).reshape(h, w)
    return raw


def parse_patches():
    pat = re.compile(r'^(.+)_\((\d+)_(\d+)\)$')
    result = []
    for t in TEXTURES:
        for f in sorted((PATCH_ROOT / str(t)).iterdir()):
            if f.suffix != '.tif' or '_cp_masks_' in f.name:
                continue
            m = pat.match(f.stem)
            if m:
                result.append({'texture': t, 'stem': m.group(1),
                               'row': int(m.group(2)), 'col': int(m.group(3))})
    return result


# ─── Modèle ───────────────────────────────────────────────────────────────────

def build_model():
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
                  global_att_blocks=(7, 10, 13),
                  window_pos_embed_bkg_spatial_size=(7, 7))
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000),
        d_model=256, backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0, fpn_interp_model='nearest',
        fuse_type='sum', fpn_top_down_levels=[2, 3])
    enc = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    sd  = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
    sd_enc = {k[len('image_encoder.'):]: v
              for k, v in sd.items() if k.startswith('image_encoder.')}
    enc.load_state_dict(sd_enc, strict=False)
    return enc.cuda().eval()


def preprocess(stem):
    img = read_tiff_gray(IMG_DIR / (stem + '.tif'))
    pil = Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x   = np.array(pil, dtype=np.float32) / 255.0
    t   = torch.from_numpy(np.stack([x, x, x], axis=0)).float()
    t   = (t - MEAN_T) / STD_T
    return t.unsqueeze(0).cuda()


def get_local_vecs(feat, row, col):
    """feat: (H_f, W_f, C) tensor — retourne (n_pos, C) float32."""
    H_f, W_f, C = feat.shape
    y0 = int(row       * PATCH_PX * fH / STRIDE)
    y1 = min(H_f, max(y0 + 1, int((row + 1) * PATCH_PX * fH / STRIDE)))
    x0 = int(col       * PATCH_PX * fW / STRIDE)
    x1 = min(W_f, max(x0 + 1, int((col + 1) * PATCH_PX * fW / STRIDE)))
    return feat[y0:y1, x0:x1, :].numpy().reshape(-1, C).astype(np.float32)


# ─── VLAD ─────────────────────────────────────────────────────────────────────

def vlad_encode(local_vecs, centroids):
    """local_vecs: (n, D), centroids: (K, D) → (K*D,) power+L2 normé."""
    K_c, Dc = centroids.shape
    diffs   = local_vecs[:, None, :] - centroids[None, :, :]  # (n, K, D)
    assigns = (diffs ** 2).sum(axis=2).argmin(axis=1)          # (n,)
    vlad    = np.zeros((K_c, Dc), dtype=np.float32)
    for k in range(K_c):
        mask = assigns == k
        if mask.sum() > 0:
            vlad[k] = (local_vecs[mask] - centroids[k]).sum(axis=0)
    vlad = vlad.flatten()
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))   # power-norm
    norm = np.linalg.norm(vlad)
    if norm > EPS:
        vlad /= norm                                 # L2-norm
    return vlad


# ─── Inner CV ─────────────────────────────────────────────────────────────────

def select_C(X_tr, y_tr):
    """Grid-search C on training data, balanced accuracy metric."""
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    if n_pos < 4 or n_neg < 4:
        return 1.0
    n_splits = min(3, min(n_pos, n_neg))
    if n_splits < 2:
        return 1.0
    skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    best_c, best_score = 1.0, -1.0
    for c in C_GRID:
        scores = []
        for tr_idx, val_idx in skf.split(X_tr, y_tr):
            try:
                clf = LogisticRegression(C=c, class_weight='balanced',
                                         max_iter=500, solver='lbfgs',
                                         random_state=SEED)
                clf.fit(X_tr[tr_idx], y_tr[tr_idx])
                scores.append(balanced_accuracy_score(
                    y_tr[val_idx], clf.predict(X_tr[val_idx])))
            except Exception:
                scores.append(0.0)
        ms = float(np.mean(scores))
        if ms > best_score:
            best_score, best_c = ms, c
    return best_c


# ─── LOIO ─────────────────────────────────────────────────────────────────────

def loio_one_vs_rest(patches, repr_key, texture_c):
    """
    Leave-One-Image-Out one-vs-rest pour texture_c.
    Retourne: (array de recalls, n_folds évalués)
    """
    stems  = sorted(set(p['stem'] for p in patches))
    recalls = []

    for stem_test in stems:
        test_p  = [p for p in patches if p['stem'] == stem_test]
        train_p = [p for p in patches if p['stem'] != stem_test]

        if not train_p:
            continue
        n_pos_test = sum(1 for p in test_p if p['texture'] == texture_c)
        if n_pos_test == 0:
            continue                  # image test sans patch de classe c → skip

        X_train = np.array([p[repr_key] for p in train_p])
        y_train = np.array([1 if p['texture'] == texture_c else 0
                            for p in train_p], dtype=np.int32)
        X_test  = np.array([p[repr_key] for p in test_p])
        y_test  = np.array([1 if p['texture'] == texture_c else 0
                            for p in test_p], dtype=np.int32)

        if len(np.unique(y_train)) < 2:
            recalls.append(0.0)
            continue

        # PCA fit sur train uniquement
        dim_in = X_train.shape[1]
        if dim_in > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_train)
            X_te = pca.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        best_c = select_C(X_tr, y_train)

        clf = LogisticRegression(C=best_c, class_weight='balanced',
                                 max_iter=1000, solver='lbfgs', random_state=SEED)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    return np.array(recalls, dtype=np.float64), len(recalls)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═══ VLAD vs Mean Pooling — b7 — K=8 — LOIO one-vs-rest ═══\n")

    patches_meta = parse_patches()
    print(f"Patches : {len(patches_meta)}")
    for t in TEXTURES:
        n = sum(1 for p in patches_meta if p['texture'] == t)
        print(f"  t{t} {TNAMES[t]:<15}: {n}")

    # ── Étape 1 : extraction vecteurs locaux ──────────────────────────────────
    print(f"\nÉtape 1 — Extraction à b{BLOCK} (D={D})...")
    enc    = build_model()
    cap    = {}
    handle = enc.trunk.blocks[BLOCK].register_forward_hook(
        lambda m, i, o: cap.__setitem__(BLOCK, o.detach().cpu()))

    by_img = defaultdict(list)
    for p in patches_meta:
        by_img[p['stem']].append(p)

    patches        = []
    all_local_list = []

    for img_stem, img_patches in sorted(by_img.items()):
        cap.clear()
        with torch.no_grad():
            enc(preprocess(img_stem))
        feat = cap[BLOCK][0]   # (H_f, W_f, D) — Hiera output format

        for p in img_patches:
            lvecs    = get_local_vecs(feat, p['row'], p['col'])  # (n_pos, D)
            mean_vec = lvecs.mean(axis=0)                         # (D,)
            patches.append({'texture': p['texture'], 'stem': p['stem'],
                            'local_vecs': lvecs, 'mean': mean_vec})
            all_local_list.append(lvecs)

    handle.remove()
    all_local = np.concatenate(all_local_list, axis=0)
    print(f"  {len(patches)} patches extraits, "
          f"{all_local.shape[0]} vecteurs locaux empilés — shape {all_local.shape}")

    # ── Étape 2 : sanité K-means ──────────────────────────────────────────────
    print(f"\nÉtape 2 — K-means K={K} + test de sanité silhouette...")
    km         = KMeans(n_clusters=K, random_state=SEED, n_init=10, max_iter=300)
    km.fit(all_local)
    centroids  = km.cluster_centers_          # (K, D)
    sizes      = np.bincount(km.labels_)

    rng     = np.random.default_rng(SEED)
    sil_n   = min(8000, all_local.shape[0])
    sil_idx = rng.choice(all_local.shape[0], sil_n, replace=False)
    X_sub   = all_local[sil_idx]
    L_sub   = km.labels_[sil_idx]

    try:
        sil = float(silhouette_score(X_sub, L_sub, sample_size=min(5000, sil_n),
                                     random_state=SEED))
    except Exception:
        sil = float(silhouette_score(X_sub, L_sub))

    print(f"  Inertie K-means : {km.inertia_:.2f}")
    print(f"  Silhouette      : {sil:.4f}")
    print(f"  Tailles clusters: {sorted(sizes.tolist())}")
    if sil > 0.15:
        print("  → CLUSTERS NETS ✓ — VLAD a une bonne base")
    else:
        print(f"  → [DRAPEAU] silhouette={sil:.4f} ≤ 0.15 : clusters FLOUS")
        print("     VLAD pourrait sous-performer, on continue quand même")

    # ── Étape 4 : encodage VLAD ───────────────────────────────────────────────
    print(f"\nÉtape 4 — Encodage VLAD (K={K}×D={D}={K*D} dim)...")
    for p in patches:
        p['vlad'] = vlad_encode(p['local_vecs'], centroids)

    vlad_norms = [np.linalg.norm(p['vlad']) for p in patches[:10]]
    print(f"  Norme VLAD premiers patches : {[f'{n:.4f}' for n in vlad_norms]}")
    print(f"  (attendu ≈ 1.0 après power+L2 norm)")
    print(f"\n  PCA appliquée : VLAD {K*D}→{PCA_DIM}  |  Mean {D}→{PCA_DIM}")

    # ── Étape 5 : LOIO ────────────────────────────────────────────────────────
    print(f"\nÉtape 5 — LOIO one-vs-rest (PCA={PCA_DIM}, inner CV C∈{C_GRID})...")
    print(f"  (vocab global assumé : borne OPTIMISTE)\n")

    results = {}
    for texture_c in TEXTURES:
        n_pat = sum(1 for p in patches if p['texture'] == texture_c)
        n_img = len(set(p['stem'] for p in patches if p['texture'] == texture_c))
        print(f"  t{texture_c} {TNAMES[texture_c]:<15} (N={n_pat}, {n_img} images)...")

        r_vlad, nf_v = loio_one_vs_rest(patches, 'vlad', texture_c)
        r_mean, nf_m = loio_one_vs_rest(patches, 'mean', texture_c)

        results[texture_c] = {'vlad': r_vlad, 'mean': r_mean,
                              'n_patches': n_pat, 'n_images': n_img,
                              'n_folds': nf_v}
        gain = r_vlad.mean() - r_mean.mean() if (len(r_vlad) and len(r_mean)) else float('nan')
        print(f"    VLAD : {r_vlad.mean():.3f} ± {r_vlad.std():.3f}  ({nf_v} folds)")
        print(f"    Mean : {r_mean.mean():.3f} ± {r_mean.std():.3f}  ({nf_m} folds)")
        print(f"    Gain : {gain:+.3f}\n")

    # ── Étape 6 : tableau + verdict ───────────────────────────────────────────
    print("═"*80)
    print("TABLEAU FINAL")
    print("═"*80)
    header = (f"{'Texture':<18} {'N':>4}  {'Recall_Mean':^13}  {'Recall_VLAD':^13}  "
              f"{'Gain':>7}  {'Tag':>4}  Note")
    print(header)
    print("─"*80)

    gains_s, gains_u = [], []
    gains_by_t = {}

    for t in TEXTURES:
        r    = results[t]
        rv   = r['vlad'].mean() if len(r['vlad']) > 0 else float('nan')
        rm   = r['mean'].mean() if len(r['mean']) > 0 else float('nan')
        sv   = r['vlad'].std()  if len(r['vlad']) > 0 else float('nan')
        sm   = r['mean'].std()  if len(r['mean']) > 0 else float('nan')
        gain = rv - rm
        gains_by_t[t] = gain
        tag  = '[U]' if t in TUNIFORM else '[S]'
        note = ''
        if t == 5:
            note = '⚠ 1 img dominante'
        elif r['n_patches'] < 20:
            note = '⚠ N faible'
        print(f"{TNAMES[t]:<18} {r['n_patches']:>4}  "
              f"{rm:>6.3f}±{sm:.3f}  "
              f"{rv:>6.3f}±{sv:.3f}  "
              f"{gain:>+7.3f}  {tag:>4}  {note}")
        if not np.isnan(gain):
            (gains_s if t not in TUNIFORM else gains_u).append(gain)

    print("─"*80)
    gs = float(np.mean(gains_s)) if gains_s else float('nan')
    gu = float(np.mean(gains_u)) if gains_u else float('nan')
    print(f"{'Gain moyen [S] (structurées)':<40} {gs:>+7.3f}")
    print(f"{'Gain moyen [U] (uniformes)':<40} {gu:>+7.3f}")

    # ── Graphique gain ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(TEXTURES))
    colors = ['#2196F3' if t in TUNIFORM else '#E53935' for t in TEXTURES]
    bars = ax.bar(x, [gains_by_t[t] for t in TEXTURES], color=colors, alpha=0.8, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.05, color='gray', lw=0.8, ls='--', label='seuil +5%')
    ax.set_xticks(x)
    ax.set_xticklabels([TNAMES[t] for t in TEXTURES], rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('Gain recall VLAD − Mean', fontsize=9)
    ax.set_title(f'VLAD vs Mean — b{BLOCK} — K={K} — LOIO recall\n'
                 f'Bleu=[U] Rouge=[S]   Silhouette K-means={sil:.3f}',
                 fontsize=9)
    for bar, g in zip(bars, [gains_by_t[t] for t in TEXTURES]):
        ax.text(bar.get_x() + bar.get_width()/2,
                g + (0.005 if g >= 0 else -0.015),
                f'{g:+.3f}', ha='center', va='bottom', fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2196F3', label='[U] uniforme'),
                        Patch(color='#E53935', label='[S] structurée')],
              fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'gain_vlad_vs_mean.png', dpi=140, bbox_inches='tight')
    plt.close()

    print(f"\n  → {OUT_DIR/'gain_vlad_vs_mean.png'}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "═"*80)
    print("VERDICT")
    print("═"*80)

    n_win = sum(1 for t in TEXTURES if not np.isnan(gains_by_t[t]) and gains_by_t[t] > 0)
    print(f"\n1. VLAD > Mean sur {n_win}/{len(TEXTURES)} textures")
    for t in TEXTURES:
        g   = gains_by_t[t]
        tag = '[U]' if t in TUNIFORM else '[S]'
        sym = '↑' if g > 0.02 else ('↓' if g < -0.02 else '=')
        print(f"   {sym} {tag} {TNAMES[t]:<15} {g:+.3f}")

    print(f"\n2. Signature [S]>[U] ?")
    print(f"   Gain moyen [S] = {gs:+.3f}")
    print(f"   Gain moyen [U] = {gu:+.3f}")
    if not np.isnan(gs) and not np.isnan(gu):
        if gs > gu + 0.05:
            print("   → OUI — gain concentré sur [S] — HYPOTHÈSE CONFIRMÉE")
            print("     Le pooling détruisait de l'info distributionnelle structurelle")
        elif abs(gs - gu) < 0.05:
            print("   → NON — gain diffus ou nul — info distributionnelle non discriminante")
        else:
            print(f"   → MIXTE — pas de signature claire [S]>[U]")

    print(f"\n3. Vocab global (borne OPTIMISTE)")
    if gs > 0.05:
        print("   VLAD gagne → résultat PROVISOIRE")
        print("   À RECONFIRMER en vocab-par-fold (sans fuite vocab) avant conclusion ferme")
    else:
        print("   VLAD n'améliore pas sur la borne haute → résultat DÉFINITIF")
        print("   Pas besoin de reconfirmer — info distributionnelle inutilisable ici")

    print(f"\n4. Silhouette K-means = {sil:.4f} ({'net' if sil>0.15 else 'FLOU [DRAPEAU]'})")
    if sil <= 0.15:
        print(f"   Clusters flous dans l'espace b{BLOCK} — cohérent avec gain VLAD faible")
        print("   Interprétation : l'espace b7 ne se prête pas bien à une quantification dure")

    print(f"\n5. Non-évaluables")
    print(f"   t5 Strat.rect : 1 image dominante (75%) → résultat non fiable en LOIO")

    print("\n═══ Terminé ═══")


if __name__ == '__main__':
    main()
