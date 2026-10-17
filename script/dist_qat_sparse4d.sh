#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PHASE=$4
PORT=${PORT:-29111}

EXTRA_ARGS="${@:5}"

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$PROJ_DIR":$PYTHONPATH

if [ "$PHASE" = "calib" ]; then
    echo "============================================"
    echo "[Phase] Single-GPU calib + export"
    echo "============================================"

    PYTHONPATH="$PYTHONPATH" python3 "$PROJ_DIR/tools/qat_workflow.py" \
        "$CONFIG" "$CHECKPOINT" \
        --phase calib \
        --launcher none \
        $EXTRA_ARGS

elif [ "$PHASE" = "train" ]; then
    echo "============================================"
    echo "[Phase] Multi-GPU QAT training ($GPUS GPUs)"
    echo "============================================"

    PYTHONPATH="$PYTHONPATH" python3 -m torch.distributed.launch \
        --nproc_per_node="$GPUS" \
        --master_port="$PORT" \
        "$PROJ_DIR/tools/qat_workflow.py" \
        "$CONFIG" "$CHECKPOINT" \
        --phase train \
        --launcher pytorch \
        $EXTRA_ARGS

elif [ "$PHASE" = "all" ]; then
    echo "============================================"
    echo "[Phase 1] Single-GPU calib + export"
    echo "============================================"

    PYTHONPATH="$PYTHONPATH" python3 "$PROJ_DIR/tools/qat_workflow.py" \
        "$CONFIG" "$CHECKPOINT" \
        --phase calib \
        --launcher none \
        $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "[Phase 1] FAILED"
        exit 1
    fi

    echo ""
    echo "============================================"
    echo "[Phase 2] Multi-GPU QAT training ($GPUS GPUs)"
    echo "============================================"

    PYTHONPATH="$PYTHONPATH" python3 -m torch.distributed.launch \
        --nproc_per_node="$GPUS" \
        --master_port="$PORT" \
        "$PROJ_DIR/tools/qat_workflow.py" \
        "$CONFIG" "$CHECKPOINT" \
        --phase train \
        --launcher pytorch \
        $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "[Phase 2] FAILED"
        exit 1
    fi

else
    echo "Usage: $0 <CONFIG> <CHECKPOINT> <GPUS> <PHASE: calib|train|all> [EXTRA_ARGS...]"
    exit 1
fi
