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
LTS_THRESHOLDS    = [12, 18, 24]
DEFAULT_THRESHOLD = 24

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 5. mRMR FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
K_MRMR         = 50

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
ALPHA_BIN      = 0.5
ALPHA_COX      = 2.0