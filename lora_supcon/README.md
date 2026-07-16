# LoRA + SupCon sur TextureSAM — état d'avancement

But : adapter légèrement l'espace des features de TextureSAM (LoRA, encodeur
gelé) pour que les patchs de MÊME texture se rapprochent malgré la variance
inter-images. Juge final : ΔAUC (LoRA − zero-shot), LP LOIO poolé, mêmes
folds. Protocole en 5 phases avec validation GO/NO-GO obligatoire à chaque
étape — **ne jamais sauter une phase ni enchaîner sur la phase 5 sans accord
humain explicite après lecture du pilote (phase 4)**.

Ce fichier est le point d'entrée pour reprendre le travail (nouvelle session
Claude ou humain) : lire ceci avant de relancer quoi que ce soit.

## Environnement

- Interpréteur : **`python3.12`** (install user `~/.local`, PAS `python3`
  système ni les envs conda `base`/`PatchTagger`/`HeartINR` qui n'ont pas
  torch). `python3.12 -c "import torch; print(torch.cuda.is_available())"`
  doit renvoyer `True`.
- GPU : RTX 4090, 24GB. Vérifier `nvidia-smi` avant de lancer un entraînement
  (aucun autre process compute ne doit tourner).
- Données : H5 = `data/feature_database/database_meb_ouassim.h5` (features
  précalculées zero-shot), images brutes = `Image_Ouassim/*.tif` (grayscale,
  768×1280, mean≈85). **Jamais** PatchTagger (RGB, mean≈106) — cf. mémoire
  `feedback_h5_ouassim_not_patchtagger`.
- Checkpoint : `checkpoints/sam2.1_hiera_small_1.pt` (Hiera small, stages
  `(1,2,11,2)`, `global_att_blocks=(7,10,13)`, dims 96→192→384→768).

## Structure du code

```
lora_supcon/
  transforms.py   — transform_image / transform_coords (flipH/V, rot90/180/270)
  lora.py         — LoRALinear + apply_lora (gèle TOUT l'encodeur, insère sur
                     qkv des blocs 4-13 = stage 3 ; r=8, alpha=16, dropout=0.1)
  train.py        — ProjectionHead (384→256→128, L2), supcon_loss,
                     sample_balanced_batch_bounded_images, forward_batch
  loio.py         — loio_single_fold (PCA-50 train-only, LR balanced, AUC),
                     même protocole que scripts/test_multimetriques_loio.py
  phase_N/        — report.txt (+ figures) produits par chaque script de phase

scripts/
  lora_supcon_phase0.py  — env/données (GO)
  lora_supcon_phase1.py  — transforms + coord mapping (GO)
  lora_supcon_phase2.py  — insertion LoRA + test d'identité (GO)
  lora_supcon_phase3.py  — SupCon + sanity check overfit (GO)
  lora_supcon_phase4.py  — fold pilote complet (GO pipeline, résultat mitigé)
```

Lancer : `python3.12 scripts/lora_supcon_phaseN.py` (chaque script écrit son
rapport dans `lora_supcon/phase_N/report.txt`).

## Où on en est (phases 0-4 : GO pipeline ; décision humaine en attente)

**Phase 0-3 : GO, rien à revoir.**
- P0 : Ouassim confirmé, checkpoint chargeable, inventaire patchs conforme
  (41/68/49/64/129/409/56 pour les 7 textures).
- P1 : formules de transformation coord validées pixel-perfect (180/180 tests,
  21600 vérifications de bornes).
- P2 : test d'identité (B=0) exact (écart 0.000e+00), 0.357% params
  entraînables. **Bug corrigé** : `apply_lora` gelait initialement seulement
  le trunk, pas le neck (FpnNeck) → fuite de gradient. Fixé : `apply_lora`
  gèle maintenant tout l'encodeur avant d'insérer les adaptateurs.
- P3 : overfit volontaire sur mini-set (2 images, 50 patchs, sans aug) : loss
  3.19→2.20 (-31%), convergence propre en <100 steps. Contrôle négatif (LoRA
  gelé) : loss stable (bruit ±0.5%, aucune tendance). **Bug corrigé** : le
  mini-set initial incluait des classes hors-cible (Nd, Bactéries, Cellule) —
  filtré aux 7 textures cibles.

**Phase 4 : pipeline GO, résultat scientifique MITIGÉ — décision humaine
requise avant phase 5.**

Bug de performance majeur trouvé et corrigé pendant cette phase :
- `sample_balanced_batch` (naïf) tirait des patchs sur tout le pool train
  (56 images) → un batch touchait ~30-40 images distinctes. `forward_batch`
  fait un forward encodeur par groupe `(image, t)` mais ne backprop qu'une
  seule fois à la fin (loss SupCon sur tout le batch) → tous ces graphes de
  calcul restaient en mémoire simultanément avant le backward. Résultat :
  ~18-20GB de pic mémoire, ~1-4s/step (un run à 2000 steps aurait pris
  >1h, voire OOM).
- **Fix** : `sample_balanced_batch_bounded_images` (dans `train.py`) borne le
  nombre d'images sources distinctes par step à `MAX_IMAGES_PER_STEP=10` via
  une couverture gloutonne des textures (seulement ~2 images suffisent à
  couvrir les 7 textures dans ce dataset) + complément aléatoire pour la
  diversité, avec **un seul transform partagé par image sélectionnée** (donc
  nb forwards/step = nb images sélectionnées, indépendant du nb de patchs).
- Après fix : ~310ms/step, 5.7GB de pic mémoire (~10-15x plus rapide).
  **Toujours utiliser `sample_balanced_batch_bounded_images`, jamais
  `sample_balanced_batch` (naïve, gardée dans train.py mais dépréciée pour
  l'entraînement réel — trop lente/mémoire au-delà d'un mini-set de 2-3
  images).**

Setup du fold pilote :
- `TEST_IMAGE = "060722-Nabila-JP-Valves-WholeMount-SAureus-pat04-1-22.tif"`
  (Faisceaux 19, Granuleux 30, Trou 3 — choisie pour contenir Faisceaux, une
  des textures à "débloquer").
- `VAL_IMAGES` = 3 images dédiées à l'early stopping (patience=10 évals,
  éval toutes les 25 steps), jamais utilisées pour le gradient LoRA.
- `TRAIN` = les 56 autres images.
- Anti-fuite vérifié par assert à chaque step (train ET val ET LOIO) : l'image
  de test n'apparaît jamais.

Résultat (early stop à step 350, meilleur checkpoint restauré = step 100,
val_loss=2.717 — au-delà le train continue de baisser mais le val remonte,
overfitting net et bien visible sur `phase_4/loss_curves.png`) :

| Texture | bloc (meilleur zero-shot, ce fold) | AUC zero-shot | AUC LoRA | Δ |
|---|---|---|---|---|
| Granuleux | block_9 | 0.661 | 0.745 | **+0.085** |
| Trou | block_0 | 0.993 | 0.986 | -0.007 (plafond, bruit) |
| Faisceaux | block_15 | 0.687 | 0.617 | **-0.070** |
| Tot.homogène / Strat.rect / Filaments / Strat.sin | — | — | — | n/a (absentes de cette image de test) |

**Point non résolu / prochaine étape** : Granuleux s'améliore nettement, mais
Faisceaux — la texture même que ce fold devait tenter de débloquer — se
dégrade. Est-ce du bruit de fold unique (n=52 patchs test, 3 textures
seulement présentes) ou un effet systématique ? Pas encore déterminé.

Options envisagées avec l'utilisateur (discussion interrompue, pas encore
tranchée) :
1. Relancer 2-3 folds pilotes supplémentaires (image contenant Strat.sin,
   une autre avec Faisceaux) — peu coûteux maintenant (~2 min/fold après le
   fix de batching) et dirait si la baisse Faisceaux est un artefact de fold
   ou systématique.
2. Retuner avant d'aller plus loin (bloc "champion" utilisé pour le signal
   SupCon — actuellement `TRAIN_BLOCK="block_9"` dans `train.py` —, lr,
   température, plus d'images de validation).
3. Passer direct à la phase 5 (10 folds) et accepter le signal mitigé.
4. S'arrêter là.

**Phase 5 : NON DÉMARRÉE.** Ne pas lancer sans trancher le point ci-dessus.

## Hyperparamètres actuels (scripts/lora_supcon_phase4.py)

```
LoRA        : r=8, alpha=16, dropout=0.1, blocs 4-13 (stage 3)
Optim       : AdamW, lr=1e-4, weight_decay=0.01
SupCon      : température=0.1
Batch       : N_PER_TEXTURE=6 (7 textures × 6 = 42 patchs/step)
              MAX_IMAGES_PER_STEP=10
Entraînement: MAX_STEPS=2000, EVAL_EVERY=25, PATIENCE_EVALS=10
Bloc entraîné (signal SupCon) : TRAIN_BLOCK="block_9" (train.py)
              — à l'éval, le meilleur bloc est re-choisi par texture parmi
              les 16 (pas figé à block_9).
```

## Décisions de design à respecter (issues du prompt original, ne pas dévier)

- Images ENTIÈRES uniquement (jamais de crops isolés).
- Augmentations : flips H/V + rotations 90/180/270 SEULEMENT, sur l'image
  entière. Pas de crop-resize, pas d'augmentation d'intensité.
- Coordonnées des patchs transformées par formules fermées (`transforms.py`,
  validées pixel-perfect en phase 1).
- LoRA uniquement sur qkv des blocs du stage 3 (4-13).
- SupCon : positifs = même texture (y compris versions augmentées).
- Anti-fuite stricte : LoRA entraîné par fold, l'image de test n'influence
  jamais A/B/tête/early-stopping (assertions systématiques dans le code).
- SEED fixé partout.
