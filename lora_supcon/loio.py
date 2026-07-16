"""
loio.py — LP LOIO (Linear Probing, Leave-One-Image-Out), protocole strict
identique à scripts/test_multimetriques_loio.py : PCA(50) train-only,
LogisticRegression balanced, AUC pooled (ou single-fold pour le pilote).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PCA_DIM = 50
C = 1.0
SEED = 42


def loio_single_fold(X, y_bin, stems, test_stem):
    """LOIO à un seul fold fixé (test_stem). Retourne None si le fold est
    invalide (pas de positif en test, ou classe unique en train)."""
    te = stems == test_stem
    tr = ~te
    assert test_stem not in set(stems[tr]), "FUITE : image de test présente dans le train du LP"

    if y_bin[te].sum() == 0 or len(np.unique(y_bin[tr])) < 2:
        return None

    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_bin[tr], y_bin[te]
    if X_tr.shape[1] > PCA_DIM:
        pca = PCA(n_components=PCA_DIM, random_state=SEED)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    proba_te = clf.predict_proba(X_te)[:, 1]

    if len(np.unique(y_te)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_te, proba_te)
    return {"auc": auc, "n_test": int(te.sum()), "n_pos_test": int(y_te.sum())}
