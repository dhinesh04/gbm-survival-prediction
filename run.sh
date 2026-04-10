#!/bin/bash
# run.sh
# ------------------
# Runs the full GBM pipeline for LTS thresholds: 12, 18, and 24 months.
# Every invocation creates a self-contained, timestamped run folder:
#
#   src/outputs/runs/<TIMESTAMP>/
#     logs/   — experiment log
#     data/   — processed CSVs per threshold  (data_12m/, data_18m/, ...)
#     plots/  — figures per threshold          (plots_12m/, plots_18m/, ...)
#
# Nothing outside that folder is written, so re-runs never overwrite old results.
#
# Usage
# -----
#   bash run.sh                          # thresholds 12 18 24
#   bash run.sh 12 18                    # custom subset
#   bash run.sh --use_mutations          # include mutation modality
#   bash run.sh 12 --use_mutations       # both

export PYTHONPATH="$(pwd)"
set -e  # exit immediately if any command fails

# ── Create a unique run folder ────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="src/outputs/runs/${TIMESTAMP}"

mkdir -p "${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}/data"
mkdir -p "${RUN_DIR}/plots"

LOG_FILE="${RUN_DIR}/logs/experiment.log"

# Redirect ALL output through tee → terminal + log file simultaneously
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Parse arguments ───────────────────────────────────────────────────────────
USE_MUTATIONS=false
THRESHOLD_ARGS=()
for ARG in "$@"; do
    if [ "$ARG" = "--use_mutations" ]; then
        USE_MUTATIONS=true
    else
        THRESHOLD_ARGS+=("$ARG")
    fi
done
THRESHOLDS="${THRESHOLD_ARGS[*]:-12 18 24}"
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
echo "  Thresholds: ${THRESHOLDS}"
echo "  Mutations : ${USE_MUTATIONS}"
echo "============================================================"

# ── Per-threshold loop ────────────────────────────────────────────────────────
for T in $THRESHOLDS; do
    THRESHOLD_START=$SECONDS

    DATA_DIR="${RUN_DIR}/data/data_${T}m"
    PLOTS_DIR="${RUN_DIR}/plots/plots_${T}m"

    mkdir -p "$DATA_DIR"
    mkdir -p "$PLOTS_DIR"

    echo ""
    echo "============================================================"
    echo "  LTS Threshold = ${T} months"
    echo "  Started: $(date '+%H:%M:%S')"
    echo "============================================================"

    echo ""
    echo "--- Step 1/2: Data Processing (threshold=${T}m) ---"
    python src/data/data_processing.py \
        --threshold  "$T" \
        --output_dir "$DATA_DIR" \
        $MUTATION_FLAG

    echo ""
    echo "--- Step 2/2: Main Pipeline + GCN (threshold=${T}m) ---"
    python main.py \
        --threshold  "$T" \
        --data_dir   "$DATA_DIR" \
        --plots_dir  "$PLOTS_DIR" \
        $MUTATION_FLAG

    THRESHOLD_ELAPSED=$(( SECONDS - THRESHOLD_START ))
    echo ""
    echo "  ✓ Threshold ${T}m done in $(( THRESHOLD_ELAPSED / 60 ))m $(( THRESHOLD_ELAPSED % 60 ))s"
    echo "    data  -> ${DATA_DIR}/"
    echo "    plots -> ${PLOTS_DIR}/"
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo ""
echo "  Run folder: ${RUN_DIR}/"
echo "  Output layout:"
for T in $THRESHOLDS; do
    echo "    ${T}m  ->  ${RUN_DIR}/data/data_${T}m/"
    echo "          ->  ${RUN_DIR}/plots/plots_${T}m/"
done
echo "    log  ->  ${LOG_FILE}"
echo "============================================================"