"""Single source of truth for hyperparameters used across the pipeline."""

RANDOM_STATE = 42

# data paths
RAW_DATA_DIR      = "cbioportal_data"
DEFAULT_DATA_DIR  = "data"
DEFAULT_PLOTS_DIR = "plots"

# LTS threshold experiment - kept for older modules; gcn_train.py stratifies
# CV on event status (deceased vs censored) instead
LTS_THRESHOLDS    = [12, 18, 24]
DEFAULT_THRESHOLD = 24

TEST_SIZE = 0.30

# mRMR feature selection
K_MRMR = 50

# SNF - patient similarity network
K_SNF      = 20
MU_SNF     = 0.5
N_ITER_SNF = 20

# PSN diagnostics
N_PERMUTATIONS = 1000

# survival-aware PSN blend
ALPHA_SURVIVAL = 0

# GCN architecture
HIDDEN_DIM = 64   # hidden units per GCN layer
DROPOUT    = 0.5  # dropout after each GCN layer

# per-modality encoder output dim; each omics/clinical block is projected to
# ENC_DIM before concat + graph conv, since CNA/mRNA/methylation live on very
# different scales. With 4 modalities: GCN input = ENC_DIM * 4.
# Set to None to disable encoders (plain concat, ablation A7).
ENC_DIM = 32

# modality fusion, used when all 4 modalities are present:
#   "none"            - plain concat, no learned fusion
#   "attention"        - ModalityAttentionFusion, learned-query pooling attention (not self-attention)
#   "omics_self_attn"  - real self-attention over CNA/mRNA/Meth only, clinical concatenated in after (mirrors MoACG, Qin et al. 2026)
FUSION_TYPE    = "omics_self_attn"
N_FUSION_HEADS = 4  # only used when FUSION_TYPE == "attention"

# GCN training
LR            = 0.001
WEIGHT_DECAY  = 0.01
EPOCHS        = 500
PATIENCE      = 30
MIN_EPOCHS    = 80
N_FOLDS       = 5
ADJ_THRESHOLD = 0.0
K_TEST        = 10

# joint loss weights - legacy fixed defaults, still used by ablation_studies.py
# and baseline_comparison.py. gcn_train.py ignores these and grid-searches
# ALPHA_AFT_GRID x ALPHA_COX_GRID via nested 5-fold CV instead.
ALPHA_AFT = 0.5
ALPHA_COX = 2.0

# grid searched in gcn_train.py: 4x4x5 folds + 1 final retrain
ALPHA_AFT_GRID = [0.25, 0.5, 1.0, 2.0]
ALPHA_COX_GRID = [0.5, 1.0, 2.0, 4.0]

# metric used to pick the winning (alpha_aft, alpha_cox) pair: "mae" (AFT's
# native metric, default) or "cindex"
SELECTION_METRIC = "mae"
