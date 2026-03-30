#!/bin/bash
# run.sh
# ------------------
# Runs the full GBM pipeline for LTS thresholds: 12, 18, and 24 months.
# Each threshold gets its own data and plots directory.
# All output is logged to logs/experiment_<timestamp>.log
#

set -e  # exit immediately if any command fails

# ── Log setup ────────────────────────────────────────────────────────────────
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

# Redirect ALL output (stdout + stderr) through tee so it prints to terminal
# AND writes to the log file simultaneously
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Start ─────────────────────────────────────────────────────────────────────
USE_MUTATIONS=false
THRESHOLD_ARGS=()
for ARG in "$@"; do
    if [ "$ARG" = "--use_mutations" ]; then
        USE_MUTATIONS=true
    else
        THRESHOLD_ARGS+=("$ARG")
    fi
done
THRESHOLDS=${THRESHOLD_ARGS[@]:-12 18 24}
MUTATION_FLAG=""
if [ "$USE_MUTATIONS" = true ]; then
    MUTATION_FLAG="--use_mutations"
fi
EXPERIMENT_START=$SECONDS

echo "============================================================"
echo "  GBM Experiment Run"
echo "  Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log file : $LOG_FILE"
echo "  Thresholds: $THRESHOLDS"
echo "  Mutation mode: $USE_MUTATIONS"
echo "============================================================"

# ── Per-threshold loop ────────────────────────────────────────────────────────
for T in $THRESHOLDS; do
    THRESHOLD_START=$SECONDS

    echo ""
    echo "============================================================"
    echo "  LTS Threshold = ${T} months"
    echo "  Started: $(date '+%H:%M:%S')"
    echo "============================================================"

    echo ""
    echo "--- Step 1/2: Data Processing (threshold=${T}m) ---"
    python data_processing.py --threshold $T --output_dir data_${T}m $MUTATION_FLAG

    echo ""
    echo "--- Step 2/2: Main Pipeline + GCN (threshold=${T}m) ---"
    python main.py --threshold $T --data_dir data_${T}m --plots_dir plots_${T}m $MUTATION_FLAG

    THRESHOLD_ELAPSED=$(( SECONDS - THRESHOLD_START ))
    echo ""
    echo "  ✓ Threshold ${T}m done in $(( THRESHOLD_ELAPSED / 60 ))m $(( THRESHOLD_ELAPSED % 60 ))s"
    echo "    data  -> data_${T}m/"
    echo "    plots -> plots_${T}m/"
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - EXPERIMENT_START ))

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo ""
echo "  Output directories:"
for T in $THRESHOLDS; do
    echo "    ${T}m  ->  data_${T}m/  |  plots_${T}m/"
done
echo ""
echo "  Full log saved to: $LOG_FILE"
echo "============================================================"