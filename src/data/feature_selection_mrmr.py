"""
mrmr.py
-------
Supervised feature selection using Minimum Redundancy Maximum Relevance (mRMR).

Fitted on training data only — test data is transformed using the features
selected from training, never used to influence selection.
"""

import numpy as np
import pandas as pd
from mrmr import mrmr_classif


def select_features_mrmr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    K: int = 50,
    modality_name: str = ""
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Apply mRMR feature selection fitted on training data only.

    mRMR selects K features that maximise mutual information with the
    survival label (LTS/non-LTS) while minimising redundancy among
    selected features. This is fundamentally different from variance
    filtering, which is blind to the survival label entirely.

    Parameters
    ----------
    X_train : pd.DataFrame, shape (n_train, n_features)
        Training omics matrix. Rows = patients, columns = features.
    y_train : pd.Series, shape (n_train,)
        Binary LTS label for training patients (1=LTS, 0=non-LTS).
    X_test : pd.DataFrame, shape (n_test, n_features)
        Test omics matrix. Same columns as X_train.
        Transformed using features selected from training — never fitted.
    K : int
        Number of features to select. Default 50.
    modality_name : str
        Name for logging (e.g. 'CNA', 'mRNA', 'Methylation').

    Returns
    -------
    X_train_sel : np.ndarray, shape (n_train, K)
        Training data reduced to K selected features.
    X_test_sel : np.ndarray, shape (n_test, K)
        Test data reduced to the same K features.
    selected_features : list
        Column names of the K selected features.
    """
    # Ensure clean indices to avoid alignment issues
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = pd.Series(y_train.values if hasattr(y_train, 'values') else y_train)

    print(f"  [{modality_name}] mRMR: {X_train.shape[1]:,} features → selecting {K}")

    # mRMR is fit on training data only
    # Returns an ordered list of K feature names (most relevant first)
    selected_features = mrmr_classif(X=X_train, y=y_train, K=K)

    X_train_sel = X_train[selected_features].values.astype(np.float32)
    X_test_sel  = X_test[selected_features].values.astype(np.float32)

    print(f"  [{modality_name}] Done → shape {X_train_sel.shape}")

    return X_train_sel, X_test_sel, selected_features


def run_mrmr_all_modalities(
    X_cna_train:   pd.DataFrame,
    X_mrna_train:  pd.DataFrame,
    X_meth_train:  pd.DataFrame,
    y_train:       pd.Series,
    X_cna_test:    pd.DataFrame,
    X_mrna_test:   pd.DataFrame,
    X_meth_test:   pd.DataFrame,
    K: int = 50,
    # Optional 4th modality — mutation binary matrix
    X_mut_train:   pd.DataFrame = None,
    X_mut_test:    pd.DataFrame = None,
) -> dict:
    """
    Run mRMR independently on each omics modality.
    If X_mut_train / X_mut_test are provided, mutation is included as a
    4th modality with its own K selected features.

    Selection is always fit on training data only.

    Returns
    -------
    dict with keys:
        'cna_train',  'cna_test',  'cna_features'
        'mrna_train', 'mrna_test', 'mrna_features'
        'meth_train', 'meth_test', 'meth_features'
        'mut_train',  'mut_test',  'mut_features'   ← only if mutation provided
    """
    print("\n── mRMR Feature Selection ──────────────────────────────────")

    cna_tr, cna_te, cna_feats = select_features_mrmr(
        X_cna_train, y_train, X_cna_test, K=K, modality_name="CNA"
    )
    mrna_tr, mrna_te, mrna_feats = select_features_mrmr(
        X_mrna_train, y_train, X_mrna_test, K=K, modality_name="mRNA"
    )
    meth_tr, meth_te, meth_feats = select_features_mrmr(
        X_meth_train, y_train, X_meth_test, K=K, modality_name="Methylation"
    )

    n_modalities = 3
    result = {
        "cna_train":    cna_tr,   "cna_test":    cna_te,   "cna_features":  cna_feats,
        "mrna_train":   mrna_tr,  "mrna_test":   mrna_te,  "mrna_features": mrna_feats,
        "meth_train":   meth_tr,  "meth_test":   meth_te,  "meth_features": meth_feats,
    }

    if X_mut_train is not None and X_mut_test is not None:
        mut_tr, mut_te, mut_feats = select_features_mrmr(
            X_mut_train, y_train, X_mut_test, K=K, modality_name="Mutation"
        )
        result["mut_train"]    = mut_tr
        result["mut_test"]     = mut_te
        result["mut_features"] = mut_feats
        n_modalities = 4

    print(f"\n  Total features after mRMR: {K * n_modalities} "
          f"({K} per modality × {n_modalities} modalities)")
    print("────────────────────────────────────────────────────────────\n")

    return result