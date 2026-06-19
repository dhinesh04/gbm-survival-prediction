"""
config.py
---------
Single source of truth for every hyperparameter used across the pipeline.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_DIR      = "cbioportal_data"
DEFAULT_DATA_DIR  = "data"
DEFAULT_PLOTS_DIR = "plots"

# ─────────────────────────────────────────────────────────────────────────────
# 3. LTS THRESHOLD EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
# Retained only because data_processing.py / older modules may still
# reference it. gcn_train.py no longer uses any threshold anywhere —
# CV stratification is done on event status (deceased vs censored).
LTS_THRESHOLDS    = [12, 18, 24]
DEFAULT_THRESHOLD = 24

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 5. mRMR FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
K_MRMR         = 100

# ─────────────────────────────────────────────────────────────────────────────
# 6. SNF — PATIENT SIMILARITY NETWORK
# ─────────────────────────────────────────────────────────────────────────────
K_SNF          = 20
MU_SNF         = 0.5
N_ITER_SNF     = 20

# ─────────────────────────────────────────────────────────────────────────────
# 7. PSN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
N_PERMUTATIONS = 1000

# ─────────────────────────────────────────────────────────────────────────────
# 8. SURVIVAL-AWARE PSN BLEND
# ─────────────────────────────────────────────────────────────────────────────
ALPHA_SURVIVAL = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# 9. GCN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIM     = 64          # hidden units in each GCN layer
DROPOUT        = 0.5         # dropout rate applied after each GCN layer

# Per-modality encoder output dimension.
# Each omics/clinical block is independently projected from its raw dimension
# into ENC_DIM before concatenation and graph convolution.  This lets each
# modality learn its own representation space, which is important because
# CNA (discrete copy ratios), mRNA (log-expression), and methylation
# (β-values in [0,1]) have fundamentally different statistical distributions.
# With 4 modalities and ENC_DIM=32: GCN input = 32 × 4 = 128 dims.
# Set to None to disable modality encoders (plain concatenation, ablation A7).
ENC_DIM        = 32

# ─────────────────────────────────────────────────────────────────────────────
# 10. GCN TRAINING
# ─────────────────────────────────────────────────────────────────────────────
LR             = 0.001
WEIGHT_DECAY   = 0.01
EPOCHS         = 500
PATIENCE       = 30
MIN_EPOCHS     = 80
N_FOLDS        = 5
ADJ_THRESHOLD  = 0.0
K_TEST         = 10

# ─────────────────────────────────────────────────────────────────────────────
# 11. JOINT LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# ALPHA_AFT / ALPHA_COX below are the legacy fixed defaults. They are kept
# for backward compatibility with modules not yet updated to the grid search
# (ablation_studies.py, baseline_comparison.py — pending update).
#
# gcn_train.py IGNORES these two and instead performs a nested 5-fold CV
# grid search over ALPHA_AFT_GRID × ALPHA_COX_GRID, selecting the winning
# pair via SELECTION_METRIC, then uses that pair for the final retrain.
ALPHA_AFT      = 0.5
ALPHA_COX      = 2.0

# Candidate values searched via nested 5-fold CV in gcn_train.py.
# 4 x 4 = 16 combinations x 5 folds = 80 fold-trainings, plus 1 final retrain.
# Shrink these lists first if a faster turnaround is needed.
ALPHA_AFT_GRID = [0.25, 0.5, 1.0, 2.0]
ALPHA_COX_GRID = [0.5, 1.0, 2.0, 4.0]

# Which CV metric selects the winning (alpha_aft, alpha_cox) pair.
#   "mae"    — lowest mean CV validation MAE (months). The AFT head's own
#              native metric — recommended default.
#   "cindex" — highest mean CV validation Cox C-index.
SELECTION_METRIC = "mae"