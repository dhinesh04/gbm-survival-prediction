# #!/bin/bash
# # run.sh
# # ------------------
# # Runs the full GBM pipeline for LTS thresholds: 12, 18, and 24 months.
# #
# # Data processing runs ONCE — labels.csv contains OS_MONTHS and OS_STATUS
# # only. The LTS binary label is derived inline in main.py per threshold,
# # so the same processed data works for all threshold experiments.
# #
# # Every invocation creates a self-contained, timestamped run folder:
# #
# #   src/outputs/runs/<TIMESTAMP>/
# #     logs/   — experiment log
# #     data/   — processed CSVs (single folder, shared across thresholds)
# #     plots/  — figures per threshold  (plots_12m/, plots_18m/, ...)
# #
# # Usage
# # -----
# #   bash run.sh                          # thresholds 12 18 24
# #   bash run.sh 12 18                    # custom subset
# #   bash run.sh --use_mutations          # include mutation modality
# #   bash run.sh 12 --use_mutations       # both

# export PYTHONPATH="$(pwd)"
# set -e

# # ── Create a unique run folder ────────────────────────────────────────────────
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# RUN_DIR="src/outputs/runs/${TIMESTAMP}"

# mkdir -p "${RUN_DIR}/logs"
# mkdir -p "${RUN_DIR}/data"
# mkdir -p "${RUN_DIR}/plots"

# LOG_FILE="${RUN_DIR}/logs/experiment.log"

# exec > >(tee -a "$LOG_FILE") 2>&1

# # ── Parse arguments ───────────────────────────────────────────────────────────
# USE_MUTATIONS=false
# THRESHOLD_ARGS=()
# for ARG in "$@"; do
#     if [ "$ARG" = "--use_mutations" ]; then
#         USE_MUTATIONS=true
#     else
#         THRESHOLD_ARGS+=("$ARG")
#     fi
# done
# THRESHOLDS="${THRESHOLD_ARGS[*]:-12 18 24}"
# MUTATION_FLAG=""
# if [ "$USE_MUTATIONS" = true ]; then
#     MUTATION_FLAG="--use_mutations"
# fi

# EXPERIMENT_START=$SECONDS

# echo "============================================================"
# echo "  GBM Experiment Run"
# echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
# echo "  Run folder: ${RUN_DIR}/"
# echo "  Log file  : ${LOG_FILE}"
# echo "  Thresholds: ${THRESHOLDS}"
# echo "  Mutations : ${USE_MUTATIONS}"
# echo "============================================================"

# # ── Step 0: Data Processing (runs ONCE, shared across all thresholds) ─────────
# DATA_DIR="${RUN_DIR}/data"

# echo ""
# echo "--- Step 0: Data Processing (runs once) ---"
# python src/data/data_processing.py \
#     --output_dir "$DATA_DIR" \
#     $MUTATION_FLAG

# # ── Per-threshold loop (GCN training + analysis only) ────────────────────────
# for T in $THRESHOLDS; do
#     THRESHOLD_START=$SECONDS

#     PLOTS_DIR="${RUN_DIR}/plots/plots_${T}m"
#     mkdir -p "$PLOTS_DIR"

#     echo ""
#     echo "============================================================"
#     echo "  LTS Threshold = ${T} months"
#     echo "  Started: $(date '+%H:%M:%S')"
#     echo "============================================================"

#     echo ""
#     echo "--- Main Pipeline + GCN (threshold=${T}m) ---"
#     python main.py \
#         --threshold  "$T" \
#         --data_dir   "$DATA_DIR" \
#         --plots_dir  "$PLOTS_DIR" \
#         $MUTATION_FLAG

#     THRESHOLD_ELAPSED=$(( SECONDS - THRESHOLD_START ))
#     echo ""
#     echo "  ✓ Threshold ${T}m done in $(( THRESHOLD_ELAPSED / 60 ))m $(( THRESHOLD_ELAPSED % 60 ))s"
#     echo "    plots -> ${PLOTS_DIR}/"
# done

# # ── Summary ───────────────────────────────────────────────────────────────────
# TOTAL_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

# echo ""
# echo "============================================================"
# echo "  All experiments complete!"
# echo "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
# echo "  Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
# echo ""
# echo "  Run folder: ${RUN_DIR}/"
# echo "  Output layout:"
# echo "    data ->  ${DATA_DIR}/  (shared)"
# for T in $THRESHOLDS; do
#     echo "    ${T}m  ->  ${RUN_DIR}/plots/plots_${T}m/"
# done
# echo "    log  ->  ${LOG_FILE}"
# echo "============================================================"


#!/bin/bash
# run.sh
# ------------------
# Runs the full GBM pipeline once.
#
# Data processing runs once. The GCN is trained once with AFT + Cox heads.
# Threshold sensitivity (AUC at 12m / 18m / 24m) is evaluated post-hoc
# from the same predicted survival months — no retraining per threshold.
#
# Output layout:
#   src/outputs/runs/<TIMESTAMP>/
#     logs/   — experiment log
#     data/   — processed CSVs
#     plots/  — all figures
#
# Usage
# -----
#   bash run.sh
#   bash run.sh --use_mutations
#!/bin/bash
# run.sh
# ------------------
# Runs the full GBM pipeline once.
#
# Data processing runs once. The GCN is trained once with AFT + Cox heads.
# Threshold sensitivity (AUC at 12m / 18m / 24m) is evaluated post-hoc
# from the same predicted survival months — no retraining per threshold.
#
# Output layout:
#   src/outputs/runs/<TIMESTAMP>/
#     logs/   — experiment log
#     data/   — processed CSVs
#     plots/  — all figures
#
# Usage
# -----
#   bash run.sh
#   bash run.sh --use_mutations

export PYTHONPATH="$(pwd)"
set -e

# ── Create timestamped run folder ─────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="src/outputs/runs/${TIMESTAMP}"

mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/data"
mkdir -p "${RUN_DIR}/plots"

LOG_FILE="${RUN_DIR}/logs/experiment.log"

exec > >(tee -a "$LOG_FILE") 2>&1

# ── Parse arguments ───────────────────────────────────────────────────────────
USE_MUTATIONS=false
for ARG in "$@"; do
    if [ "$ARG" = "--use_mutations" ]; then
        USE_MUTATIONS=true
    fi
done
MUTATION_FLAG=""
if [ "$USE_MUTATIONS" = true ]; then
    MUTATION_FLAG="--use_mutations"
fi

EXPERIMENT_START=$SECONDS

echo "============================================================"
echo "  GBM Experiment Run"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Run folder: ${RUN_DIR}/"
echo "  Log file  : ${LOG_FILE}"
echo "  Mutations : ${USE_MUTATIONS}"
echo "============================================================"

# ── Step 1: Data Processing (runs once) ──────────────────────────────────────
DATA_DIR="${RUN_DIR}/data"
PLOTS_DIR="${RUN_DIR}/plots"

echo ""
echo "--- Step 1/2: Data Processing ---"
python src/data/data_processing.py \
    --output_dir "$DATA_DIR" \
    $MUTATION_FLAG

# ── Step 2: Full pipeline + GCN (runs once) ───────────────────────────────────
echo ""
echo "--- Step 2/2: Main Pipeline + GCN ---"
python main.py \
    --data_dir  "$DATA_DIR" \
    --plots_dir "$PLOTS_DIR" \
    $MUTATION_FLAG

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

echo ""
echo "============================================================"
echo "  All done!"
echo "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo ""
echo "  Run folder: ${RUN_DIR}/"
echo "    data  ->  ${DATA_DIR}/"
echo "    plots ->  ${PLOTS_DIR}/"
echo "    log   ->  ${LOG_FILE}"
echo "============================================================"