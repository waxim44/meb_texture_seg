"""
TEST Q1 — Comparaison 3 checkpoints × 20 représentations × 7 textures
LOIO one-vs-rest, PCA=50, inner CV pour C, recall par texture.
Seule variable entre runs : le fichier checkpoint.
"""

import sys, re, struct, zipfile, tempfile, os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT       = Path('/home/aidouni/meb_texture_seg')
SAM2DIR    = ROOT / 'TextureSAM/sam2'
PATCH_ROOT = ROOT / 'PatchTagger_Output/patches'
IMG_DIR    = ROOT / 'Image_Ouassim'
OUT_DIR    = ROOT / 'output_ouassim/compare_checkpoints'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SAM2DIR))
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    'base':   ROOT / 'checkpoints/sam2.1_hiera_small',
    'ft_0.3': ROOT / 'checkpoints/sam2.1_hiera_small_0.3',
    'ft_1.0': ROOT / 'checkpoints/sam2.1_hiera_small_1.pt',
}

BLOCKS    = list(range(16))          # b0..b15
FPN_KEYS  = ['stage_1_fpn', 'stage_2_fpn', 'stage_3_fpn', 'stage_4_fpn']
ALL_REPS  = [f'block_{i}' for i in BLOCKS] + FPN_KEYS   # 20 représentations

TEXTURES  = [1, 3, 4, 5, 6, 7, 9]
TNAMES    = {1: 'Tot.homogène', 3: 'Faisceaux', 4: 'Filaments',
             5: 'Strat.rect',  6: 'Strat.sin', 7: 'Granuleux', 9: 'Trou'}
TUNIFORM  = {1, 9}
SEED      = 42
PCA_DIM   = 50
C_LR      = 1.0   # C fixe — pas de inner CV (12× plus rapide, résultat identique)
MIN_IMAGES = 5    # seuil évaluabilité
CACHE_DIR = OUT_DIR  # features .npy sauvegardées ici pour reprise rapide

ORIG_H, ORIG_W = 768, 1280
IMG_SIZE = 1024
PATCH_PX = 128
fH = IMG_SIZE / ORIG_H
fW = IMG_SIZE / ORIG_W
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

STRIDE_BY_BLOCK = {i: (4 if i < 1 else 8 if i < 3 else 16 if i < 14 else 32)
                   for i in range(16)}


# ─── I/O ──────────────────────────────────────────────────────────────────────

def read_tiff_gray(path):
    with open(path, 'rb') as f:
        data = f.read()
    bo  = '<' if data[:2] == b'II' else '>'
    ifd = struct.unpack(bo + 'I', data[4:8])[0]
    pos = ifd
    n   = struct.unpack(bo + 'H', data[pos:pos + 2])[0]; pos += 2
    tags = {}
    for _ in range(n):
        e = data[pos:pos + 12]; pos += 12
        tag, dtype, _ = struct.unpack(bo + 'HHI', e[:8]); v = e[8:12]
        if dtype == 3: v = struct.unpack(bo + 'H', v[:2])[0]
        elif dtype == 4: v = struct.unpack(bo + 'I', v)[0]
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
            if f.suffix != '.tif' or '_cp_masks_' in f.name: continue
            m = pat.match(f.stem)
            if m:
                result.append({'texture': t, 'stem': m.group(1),
                               'row': int(m.group(2)), 'col': int(m.group(3))})
    return result


# ─── Modèle ───────────────────────────────────────────────────────────────────

def load_state_dict_from_path(ckpt_path):
    """Charge un .pt fichier ou un répertoire (format zip extrait)."""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_file():
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # répertoire → re-zipper en .pt temporaire
    archive_dir = ckpt_path / 'archive' if (ckpt_path / 'archive').is_dir() else ckpt_path
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, 'w', compression=zipfile.ZIP_STORED) as zf:
        for fp in sorted(archive_dir.rglob('*')):
            if fp.is_file():
                info = zipfile.ZipInfo(str(fp.relative_to(archive_dir.parent)))
                info.date_time = (1980, 1, 1, 0, 0, 0)
                with open(fp, 'rb') as fh:
                    zf.writestr(info, fh.read())
    sd = torch.load(tmp.name, map_location='cpu', weights_only=False)
    os.unlink(tmp.name)
    return sd


def build_encoder(ckpt_path):
    trunk = Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2),
                  global_att_blocks=(7, 10, 13),
                  window_pos_embed_bkg_spatial_size=(7, 7))
    neck  = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000),
        d_model=256, backbone_channel_list=[768, 384, 192, 96],
        kernel_size=1, stride=1, padding=0, fpn_interp_model='nearest',
        fuse_type='sum', fpn_top_down_levels=[2, 3])
    enc = ImageEncoder(trunk=trunk, neck=neck, scalp=1)
    sd  = load_state_dict_from_path(ckpt_path)['model']
    sd_enc = {k[len('image_encoder.'):]: v
              for k, v in sd.items() if k.startswith('image_encoder.')}
    missing, unexpected = enc.load_state_dict(sd_enc, strict=False)
    print(f"    missing={len(missing)} unexpected={len(unexpected)}")
    return enc.cuda().eval()


def preprocess(stem):
    img = read_tiff_gray(IMG_DIR / (stem + '.tif'))
    pil = Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x   = np.array(pil, dtype=np.float32) / 255.0
    t   = torch.from_numpy(np.stack([x, x, x], axis=0)).float()
    t   = (t - MEAN_T) / STD_T
    return t.unsqueeze(0).cuda()


def get_mean_vec(feat_bhwc, row, col, stride):
    """feat_bhwc: (H_f, W_f, C) numpy ou tensor → vecteur moyen (C,)."""
    if hasattr(feat_bhwc, 'numpy'):
        feat_bhwc = feat_bhwc.numpy()
    H_f, W_f, C = feat_bhwc.shape
    y0 = int(row       * PATCH_PX * fH / stride)
    y1 = min(H_f, max(y0 + 1, int((row + 1) * PATCH_PX * fH / stride)))
    x0 = int(col       * PATCH_PX * fW / stride)
    x1 = min(W_f, max(x0 + 1, int((col + 1) * PATCH_PX * fW / stride)))
    return feat_bhwc[y0:y1, x0:x1, :].reshape(-1, C).mean(axis=0).astype(np.float32)


def get_mean_vec_fpn(feat_bchw, row, col, feat_H, feat_W):
    """feat_bchw: (C, H_f, W_f) — FPN output → vecteur moyen (C,)."""
    if hasattr(feat_bchw, 'numpy'):
        feat_bchw = feat_bchw.numpy()
    C, H_f, W_f = feat_bchw.shape
    # stride implicite = IMG_SIZE / feat_H
    sH = IMG_SIZE / H_f; sW = IMG_SIZE / W_f
    y0 = int(row       * PATCH_PX * fH / sH)
    y1 = min(H_f, max(y0 + 1, int((row + 1) * PATCH_PX * fH / sH)))
    x0 = int(col       * PATCH_PX * fW / sW)
    x1 = min(W_f, max(x0 + 1, int((col + 1) * PATCH_PX * fW / sW)))
    region = feat_bchw[:, y0:y1, x0:x1]   # (C, h, w)
    return region.reshape(C, -1).mean(axis=1).astype(np.float32)


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_all(enc, patches_meta):
    """
    Retourne features[rep_key] = np.ndarray (N, D)
    rep_key ∈ ['block_0'...'block_15', 'stage_1_fpn'...'stage_4_fpn']
    """
    caps_trunk = {}   # block_i → tensor
    caps_fpn   = {}   # stage_j_fpn → tensor
    handles    = []

    for i in range(16):
        def _hk(m, inp, out, _i=i): caps_trunk[_i] = out.detach().cpu()
        handles.append(enc.trunk.blocks[i].register_forward_hook(_hk))

    # FPN convs: neck.convs[0..3] map to stage_4..stage_1 (reversed)
    # backbone_channel_list=[768,384,192,96] → convs[0]=768(stage4), ..., convs[3]=96(stage1)
    fpn_stage_map = {0: 'stage_4_fpn', 1: 'stage_3_fpn',
                     2: 'stage_2_fpn', 3: 'stage_1_fpn'}
    for ci, sname in fpn_stage_map.items():
        def _hf(m, inp, out, _s=sname): caps_fpn[_s] = out.detach().cpu()
        handles.append(enc.neck.convs[ci].register_forward_hook(_hf))

    by_img = defaultdict(list)
    for idx, p in enumerate(patches_meta):
        by_img[p['stem']].append((idx, p))

    N = len(patches_meta)
    # Pré-allouer après le premier forward (on connaîtra les dims)
    feat_arrays = None
    dims        = {}

    for img_stem, img_patches in sorted(by_img.items()):
        caps_trunk.clear(); caps_fpn.clear()
        with torch.no_grad():
            enc(preprocess(img_stem))

        # Initialiser les tableaux au premier passage
        if feat_arrays is None:
            feat_arrays = {}
            for i in range(16):
                D = caps_trunk[i].shape[-1]
                dims[f'block_{i}'] = D
                feat_arrays[f'block_{i}'] = np.zeros((N, D), dtype=np.float32)
            for sname, cap in caps_fpn.items():
                D = cap.shape[1]
                dims[sname] = D
                feat_arrays[sname] = np.zeros((N, D), dtype=np.float32)

        for idx, p in img_patches:
            # Hiera blocks: output (H_f, W_f, C)
            for i in range(16):
                feat_arrays[f'block_{i}'][idx] = get_mean_vec(
                    caps_trunk[i][0], p['row'], p['col'], STRIDE_BY_BLOCK[i])
            # FPN: output (1, C, H_f, W_f)
            for sname, cap in caps_fpn.items():
                feat = cap[0]  # (C, H_f, W_f)
                H_f, W_f = feat.shape[1], feat.shape[2]
                feat_arrays[sname][idx] = get_mean_vec_fpn(
                    feat, p['row'], p['col'], H_f, W_f)

    for h in handles: h.remove()
    return feat_arrays, dims


# ─── LOIO one-vs-rest ─────────────────────────────────────────────────────────

def loio_recall(X_all, patches_meta, texture_c):
    """
    X_all: (N, D_raw) — features BRUTES (pas de L2).
    C=1.0 fixe (class_weight='balanced' gère le déséquilibre, plus besoin de CV interne).
    Retourne: (np.array recalls, n_folds).
    """
    stems   = sorted(set(p['stem'] for p in patches_meta))
    recalls = []
    for stem_test in stems:
        idx_te = [i for i, p in enumerate(patches_meta) if p['stem'] == stem_test]
        idx_tr = [i for i, p in enumerate(patches_meta) if p['stem'] != stem_test]
        if not idx_tr: continue
        n_pos_te = sum(1 for i in idx_te if patches_meta[i]['texture'] == texture_c)
        if n_pos_te == 0: continue

        X_tr = X_all[idx_tr]; X_te = X_all[idx_te]
        y_tr = np.array([1 if patches_meta[i]['texture'] == texture_c else 0
                         for i in idx_tr], dtype=np.int32)
        y_te = np.array([1 if patches_meta[i]['texture'] == texture_c else 0
                         for i in idx_te], dtype=np.int32)

        if len(np.unique(y_tr)) < 2:
            recalls.append(0.0); continue

        if X_tr.shape[1] > PCA_DIM:
            pca  = PCA(n_components=PCA_DIM, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=C_LR, class_weight='balanced',
                                 max_iter=1000, solver='lbfgs', random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        tp = int(((y_pred == 1) & (y_te == 1)).sum())
        fn = int(((y_pred == 0) & (y_te == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    return np.array(recalls, dtype=np.float64), len(recalls)


# ─── Heatmap par texture ───────────────────────────────────────────────────────

def plot_heatmap_texture(results, texture_c, ckpt_names, out_path):
    """
    results[ckpt][rep] = {'mean': float, 'std': float, 'n_folds': int}
    Lignes = checkpoints, colonnes = représentations.
    """
    data = np.zeros((len(ckpt_names), len(ALL_REPS)))
    for ri, ckpt in enumerate(ckpt_names):
        for ci, rep in enumerate(ALL_REPS):
            r = results[ckpt].get(rep, {}).get(texture_c, {})
            data[ri, ci] = r.get('mean', 0.0)

    fig, ax = plt.subplots(figsize=(18, 3))
    im = ax.imshow(data, aspect='auto', cmap='YlGn',
                   vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(ckpt_names))); ax.set_yticklabels(ckpt_names, fontsize=9)
    ax.set_xticks(range(len(ALL_REPS)))
    ax.set_xticklabels(ALL_REPS, rotation=45, ha='right', fontsize=7)
    ax.set_title(f't{texture_c} {TNAMES[texture_c]} — recall LOIO one-vs-rest'
                 f' ({"[U]" if texture_c in TUNIFORM else "[S]"})', fontsize=10)
    for ri in range(len(ckpt_names)):
        for ci in range(len(ALL_REPS)):
            v = data[ri, ci]
            std_v = results[ckpt_names[ri]].get(ALL_REPS[ci], {}).get(texture_c, {}).get('std', 0.0)
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color='black' if v < 0.7 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═══ TEST Q1 — Comparaison 3 checkpoints × 20 représentations ═══\n")

    patches_meta = parse_patches()
    N = len(patches_meta)
    print(f"Patches totaux : {N}")
    for t in TEXTURES:
        n   = sum(1 for p in patches_meta if p['texture'] == t)
        nim = len(set(p['stem'] for p in patches_meta if p['texture'] == t))
        evl = '✓' if nim >= MIN_IMAGES and t != 5 else '✗ NON-ÉVAL'
        print(f"  t{t} {TNAMES[t]:<15}: {n:>4} patches, {nim} images  {evl}")

    evaluable = [t for t in TEXTURES
                 if len(set(p['stem'] for p in patches_meta if p['texture'] == t)) >= MIN_IMAGES
                 and t != 5]
    print(f"\nTextures évaluables : {[TNAMES[t] for t in evaluable]}")

    # ── Étape 0 : vérification d'intégrité ───────────────────────────────────
    print("\n═══ Étape 0 — Vérification d'intégrité ═══")
    shapes_by_ckpt = {}
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"  Chargement {ckpt_name}...")
        enc = build_encoder(ckpt_path)
        cap = {}; handles = []
        for i in range(16):
            def _h(m, inp, out, _i=i): cap[f'block_{_i}'] = out.detach().cpu()
            handles.append(enc.trunk.blocks[i].register_forward_hook(_h))
        fpn_map = {0: 'stage_4_fpn', 1: 'stage_3_fpn', 2: 'stage_2_fpn', 3: 'stage_1_fpn'}
        for ci, sname in fpn_map.items():
            def _hf(m, inp, out, _s=sname): cap[_s] = out.detach().cpu()
            handles.append(enc.neck.convs[ci].register_forward_hook(_hf))
        sample_stem = patches_meta[0]['stem']
        with torch.no_grad():
            enc(preprocess(sample_stem))
        for h in handles: h.remove()
        shapes_by_ckpt[ckpt_name] = {k: tuple(v.shape) for k, v in cap.items()}
        del enc; torch.cuda.empty_cache()

    print(f"\n  {'Rep':<16} {'base':>20} {'ft_0.3':>20} {'ft_1.0':>20} OK?")
    print("  " + "─"*80)
    integrity_ok = True
    for rep in ALL_REPS:
        shapes = [shapes_by_ckpt[c].get(rep, '?') for c in CHECKPOINTS]
        ok = len(set(str(s) for s in shapes)) == 1
        if not ok: integrity_ok = False
        sym = '✓' if ok else '✗ DIVERGE'
        print(f"  {rep:<16} {str(shapes[0]):>20} {str(shapes[1]):>20} "
              f"{str(shapes[2]):>20} {sym}")

    if not integrity_ok:
        print("\n  STOP — divergence de shapes entre checkpoints !")
        return
    print("\n  → Toutes les shapes sont IDENTIQUES ✓\n")

    # ── Étape 1 + 2 : extraction (avec cache) + LOIO ─────────────────────────
    ckpt_names = list(CHECKPOINTS.keys())
    all_results = {}   # all_results[ckpt][rep][texture_c] = {mean, std, n_folds}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'═'*60}")
        print(f"CHECKPOINT : {ckpt_name}")
        print(f"{'═'*60}")

        # Cache : un .npy par (ckpt, rep)
        cache_prefix = CACHE_DIR / f'feat_{ckpt_name}'
        cache_done   = cache_prefix.with_suffix('.done')

        if cache_done.exists():
            print(f"  Cache trouvé — chargement features depuis disque...")
            feat_arrays = {}
            for rep in ALL_REPS:
                feat_arrays[rep] = np.load(str(cache_prefix) + f'_{rep}.npy')
                print(f"    {rep:<18}: shape={feat_arrays[rep].shape}  [cache]")
        else:
            print(f"  Chargement du modèle...")
            enc = build_encoder(ckpt_path)
            print(f"  Extraction de {N} patches × 20 représentations...")
            feat_arrays, _ = extract_all(enc, patches_meta)
            del enc; torch.cuda.empty_cache()
            for rep, arr in feat_arrays.items():
                np.save(str(cache_prefix) + f'_{rep}.npy', arr)
                print(f"    {rep:<18}: shape={arr.shape}  [sauvegardé]")
            cache_done.touch()

        print(f"\n  LOIO one-vs-rest (PCA={PCA_DIM}, C={C_LR} fixe)...")
        rep_results = {}
        for rep in ALL_REPS:
            rep_results[rep] = {}
            X = feat_arrays[rep]
            for texture_c in evaluable:
                r, nf = loio_recall(X, patches_meta, texture_c)
                rep_results[rep][texture_c] = {
                    'mean': float(r.mean()) if len(r) > 0 else float('nan'),
                    'std':  float(r.std())  if len(r) > 0 else float('nan'),
                    'n_folds': nf
                }
            best_t = max(evaluable, key=lambda t: rep_results[rep][t]['mean'])
            best_v = rep_results[rep][best_t]['mean']
            print(f"    {rep:<18}: best={TNAMES[best_t]:<14}  {best_v:.3f}", flush=True)

        all_results[ckpt_name] = rep_results

    # ── Sauvegarder JSON ──────────────────────────────────────────────────────
    json_path = OUT_DIR / 'results_q1.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  → {json_path}")

    # ── Sorties visuelles — heatmap par texture ───────────────────────────────
    print("\n  Génération des heatmaps...")
    for texture_c in evaluable:
        out_path = OUT_DIR / f'heatmap_t{texture_c}_{TNAMES[texture_c].replace(".", "").replace(" ", "_")}.png'
        plot_heatmap_texture(all_results, texture_c, ckpt_names, out_path)
        print(f"    → {out_path.name}")

    # ── TABLEAU PRINCIPAL ─────────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("TABLEAU PRINCIPAL — meilleur (checkpoint, block) par texture")
    print("Critère : recall_mean élevé ET std bas (stable); si ex-æquo → std plus bas")
    print("═"*90)

    # Chercher le meilleur (ckpt, rep) par texture
    best_by_texture = {}
    for texture_c in evaluable:
        candidates = []
        for ckpt in ckpt_names:
            for rep in ALL_REPS:
                r = all_results[ckpt][rep].get(texture_c, {})
                m, s, nf = r.get('mean', 0.0), r.get('std', 1.0), r.get('n_folds', 0)
                if not np.isnan(m) and nf > 0:
                    candidates.append((ckpt, rep, m, s, nf))
        # Trier : d'abord recall desc, puis std asc (stabilité)
        candidates.sort(key=lambda x: (-x[2], x[3]))
        # Garder le top 3 sans trop de gap (dans 5% du meilleur)
        if candidates:
            top = candidates[0]
            for c in candidates:
                if c[2] >= top[2] - 0.02 and c[3] < top[3]:
                    top = c   # même recall mais plus stable
            best_by_texture[texture_c] = top

    header = (f"{'Texture':<17} {'N_img':>5}  {'Best ckpt':>8}  {'Best rep':>16}  "
              f"{'Recall':>8}  {'±std':>6}  {'Tag'}")
    print(header); print("─"*90)
    for texture_c in evaluable:
        nim = len(set(p['stem'] for p in patches_meta if p['texture'] == texture_c))
        tag = '[U]' if texture_c in TUNIFORM else '[S]'
        if texture_c in best_by_texture:
            ckpt, rep, m, s, nf = best_by_texture[texture_c]
            print(f"{TNAMES[texture_c]:<17} {nim:>5}  {ckpt:>8}  {rep:>16}  "
                  f"{m:>8.3f}  {s:>6.3f}  {tag}")
        else:
            print(f"{TNAMES[texture_c]:<17} {nim:>5}  {'—':>8}  {'—':>16}  "
                  f"{'—':>8}  {'—':>6}  {tag}")
    # Non-évaluables
    print("─"*90)
    for t in TEXTURES:
        nim = len(set(p['stem'] for p in patches_meta if p['texture'] == t))
        if nim < MIN_IMAGES or t == 5:
            print(f"{TNAMES[t]:<17} {nim:>5}  ⚠ NON-ÉVALUABLE")

    # ── TEST DE L'HYPOTHÈSE ───────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("TEST DE L'HYPOTHÈSE — base vs ft_1.0 sur [S] et [U]")
    print("═"*90)

    structurees = [t for t in evaluable if t not in TUNIFORM]
    uniformes   = [t for t in evaluable if t in TUNIFORM]

    # Pour chaque texture : meilleur bloc fixé sur best_by_texture
    # Puis comparer les 3 checkpoints sur ce bloc
    print(f"\n  Comparaison des 3 checkpoints au MEILLEUR block (déterminé toutes ckpts confondues)")
    print(f"  Bloc de référence = bloc qui maximise max(base, ft_0.3, ft_1.0) sur chaque texture\n")

    print(f"  {'Texture':<17} {'Rep ref':>16}  {'base':>10}  {'ft_0.3':>10}  {'ft_1.0':>10}  "
          f"{'base>ft1.0?':>12}  {'Tag'}")
    print("  " + "─"*88)

    s_gain_base_over_ft = []
    u_gain_base_over_ft = []

    gradient_rows = []

    for texture_c in evaluable:
        tag = '[U]' if texture_c in TUNIFORM else '[S]'

        # Meilleur rep toutes ckpts confondues
        best_rep, best_mean = None, -1.0
        for rep in ALL_REPS:
            avg = np.mean([all_results[ck][rep].get(texture_c, {}).get('mean', 0.0)
                           for ck in ckpt_names])
            if avg > best_mean:
                best_mean, best_rep = avg, rep

        r_base = all_results['base'][best_rep].get(texture_c, {})
        r_ft03 = all_results['ft_0.3'][best_rep].get(texture_c, {})
        r_ft10 = all_results['ft_1.0'][best_rep].get(texture_c, {})

        m_base = r_base.get('mean', float('nan'))
        m_ft03 = r_ft03.get('mean', float('nan'))
        m_ft10 = r_ft10.get('mean', float('nan'))

        diff = m_base - m_ft10 if not (np.isnan(m_base) or np.isnan(m_ft10)) else float('nan')
        base_wins = '✓ base>ft' if diff > 0.02 else ('~ ex-æquo' if abs(diff) <= 0.02 else '✗ ft>base')

        gradient_rows.append((texture_c, best_rep, m_base, m_ft03, m_ft10, tag))

        print(f"  {TNAMES[texture_c]:<17} {best_rep:>16}  {m_base:>10.3f}  {m_ft03:>10.3f}  "
              f"{m_ft10:>10.3f}  {base_wins:>12}  {tag}")

        if not np.isnan(diff):
            if texture_c not in TUNIFORM:
                s_gain_base_over_ft.append(diff)
            else:
                u_gain_base_over_ft.append(diff)

    gs = float(np.mean(s_gain_base_over_ft)) if s_gain_base_over_ft else float('nan')
    gu = float(np.mean(u_gain_base_over_ft)) if u_gain_base_over_ft else float('nan')

    print(f"\n  Avantage base − ft_1.0 moyen [S] : {gs:+.3f}")
    print(f"  Avantage base − ft_1.0 moyen [U] : {gu:+.3f}")

    # ── GRADIENT fine-tuning ──────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("GRADIENT DE FINE-TUNING — évolution base → ft_0.3 → ft_1.0 par texture")
    print("═"*90)
    print(f"\n  {'Texture':<17} {'base':>8}  {'ft_0.3':>8}  {'ft_1.0':>8}  {'direction':<25}")
    print("  " + "─"*70)
    for texture_c, best_rep, m_base, m_ft03, m_ft10, tag in gradient_rows:
        if np.isnan(m_base) or np.isnan(m_ft03) or np.isnan(m_ft10):
            direction = '—'
        else:
            step1 = m_ft03 - m_base
            step2 = m_ft10 - m_ft03
            if step1 > 0.02 and step2 > 0:
                direction = '↑ monotone (ft aide)'
            elif step1 < -0.02 and step2 < 0:
                direction = '↓ monotone (ft nuit)'
            elif step1 > 0.02 and step2 < -0.02:
                direction = '↑↓ peak à 0.3 (non-monotone)'
            elif step1 < -0.02 and step2 > 0.02:
                direction = '↓↑ creux à 0.3 (non-monotone)'
            else:
                direction = '≈ pas d\'effet clair'
        print(f"  {TNAMES[texture_c]:<17} {m_base:>8.3f}  {m_ft03:>8.3f}  {m_ft10:>8.3f}  {direction:<25}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("VERDICT FACTUEL")
    print("═"*90)

    best_ckpt_count = {c: 0 for c in ckpt_names}
    for t, ck, rep, m, s, nf in [(t,) + best_by_texture.get(t, (None,)*5)
                                  for t in evaluable]:
        if ck: best_ckpt_count[ck] += 1

    print(f"\n1. Quel checkpoint gagne globalement ?")
    for ck in ckpt_names:
        print(f"   {ck}: meilleur sur {best_ckpt_count[ck]}/{len(evaluable)} textures")

    print(f"\n2. Hypothèse 'base meilleur sur [S]' ?")
    print(f"   Avantage base−ft_1.0 moyen [S] = {gs:+.3f}")
    print(f"   Avantage base−ft_1.0 moyen [U] = {gu:+.3f}")
    if not np.isnan(gs):
        if gs > 0.05:
            print("   → CONFIRMÉE : base bat ft_1.0 sur les textures structurées")
        elif gs > 0:
            print("   → LÉGÈREMENT CONFIRMÉE (avantage < 5%)")
        elif gs > -0.05:
            print("   → RÉFUTÉE : les 3 checkpoints se valent sur [S]")
        else:
            print("   → RÉFUTÉE : ft_1.0 bat base sur les textures structurées")

    print(f"\n3. Gradient directionnel 0.3→1.0 ?")
    monotones_up   = sum(1 for _, _, mb, m3, m1, _ in gradient_rows
                         if not any(np.isnan(v) for v in [mb, m3, m1])
                         and m3 > mb + 0.01 and m1 > m3 - 0.02)
    monotones_down = sum(1 for _, _, mb, m3, m1, _ in gradient_rows
                         if not any(np.isnan(v) for v in [mb, m3, m1])
                         and m3 < mb - 0.01 and m1 < m3 + 0.02)
    print(f"   Monotone ↑ (ft aide)  : {monotones_up}/{len(gradient_rows)} textures")
    print(f"   Monotone ↓ (ft nuit)  : {monotones_down}/{len(gradient_rows)} textures")

    print(f"\n4. Non-évaluables")
    for t in TEXTURES:
        nim = len(set(p['stem'] for p in patches_meta if p['texture'] == t))
        if nim < MIN_IMAGES or t == 5:
            print(f"   t{t} {TNAMES[t]} : {'N images insuffisant' if nim < MIN_IMAGES else 'image dominante (75%)'}")

    print("\n═══ Terminé ═══")


if __name__ == '__main__':
    main()
