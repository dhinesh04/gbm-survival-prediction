"""
config.py
---------
Single source of truth for every hyperparameter used across the pipeline.

Import in any module with:
    from config import *          # bring everything into namespace
    from config import LR, EPOCHS # or import selectively

Sections
--------
  1. Reproducibility
  2. Data paths
  3. LTS threshold experiment
  4. Train / test split
  5. mRMR feature selection
  6. SNF (Patient Similarity Network)
  7. PSN diagnostics
  8. Survival-aware PSN blend
  9. GCN architecture
 10. GCN training
 11. Joint loss weights
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42          # used everywhere: splits, CV, torch seed, np seed


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_DIR  = "cbioportal_data"  # folder with cBioPortal raw txt files
# Processed data dir is built dynamically per threshold in run_experiments.py
# Default for running main.py / data_processing.py standalone:
DEFAULT_DATA_DIR  = "data"
DEFAULT_PLOTS_DIR = "plots"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LTS THRESHOLD EXPERIMENT
#    data_processing.py will be run once per threshold.
#    Each run saves its outputs to  data_{threshold}m/
#    Plots are saved to            plots_{threshold}m/
# ─────────────────────────────────────────────────────────────────────────────
LTS_THRESHOLDS   = [12, 18, 24]  # months — one full experiment per value
DEFAULT_THRESHOLD = 24           # used when running main.py / data_processing.py standalone


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.30        # fraction of patients held out as test set


# ─────────────────────────────────────────────────────────────────────────────
# 5. mRMR FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
K_MRMR         = 50          # features selected per omics modality


# ─────────────────────────────────────────────────────────────────────────────
# 6. SNF — PATIENT SIMILARITY NETWORK
# ─────────────────────────────────────────────────────────────────────────────
K_SNF          = 20          # number of nearest neighbours in SNF affinity
MU_SNF         = 0.5         # SNF scaling parameter (controls kernel width)
N_ITER_SNF     = 20          # SNF message-passing iterations


# ─────────────────────────────────────────────────────────────────────────────
# 7. PSN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
N_PERMUTATIONS = 1000        # permutations for label-smoothness z-score null


# ─────────────────────────────────────────────────────────────────────────────
# 8. SURVIVAL-AWARE PSN BLEND
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_SURVIVAL = 0.2         # blend weight: 0 = omics-only, 1 = survival-only
                              # recommended range: 0.1 – 0.3


# ─────────────────────────────────────────────────────────────────────────────
# 9. GCN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIM     = 64          # hidden units in each GCN layer
DROPOUT        = 0.5         # dropout rate applied after each GCN layer


# ─────────────────────────────────────────────────────────────────────────────
# 10. GCN TRAINING
# ─────────────────────────────────────────────────────────────────────────────
LR             = 0.001       # Adam learning rate
WEIGHT_DECAY   = 0.01        # L2 regularisation coefficient
EPOCHS         = 500         # maximum training epochs per fold
PATIENCE       = 30          # early stopping patience (epochs without improvement)
MIN_EPOCHS     = 80          # minimum epochs before early stopping kicks in
                              # (Cox head needs more steps to learn risk ranking)
N_FOLDS        = 5           # stratified k-fold cross-validation splits
ADJ_THRESHOLD  = 0.0         # adjacency edge threshold (edges below are zeroed)
K_TEST         = 10          # k-NN links used to attach test nodes to the PSN


# ─────────────────────────────────────────────────────────────────────────────
# 11. JOINT LOSS WEIGHTS
#     L = ALPHA_BIN * L_binary + ALPHA_COX * L_cox
#     Cox head is the primary objective — binary head is auxiliary.
#     Higher ALPHA_COX prevents binary head from saturating the backbone
#     before the Cox head has had sufficient gradient steps.
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_BIN      = 0.5         # weight on binary cross-entropy loss
ALPHA_COX      = 2.0         # weight on Cox partial likelihood loss