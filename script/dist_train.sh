#!/bin/bash
set -e

# ==============================
# Sparse4D 多 GPU 分布式训练脚本
# ==============================

# 1. 路径配置
PROJECT_DIR="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws"
TRAIN_SCRIPT="$PROJECT_DIR/script/train.py"
CONFIG="$PROJECT_DIR/dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"

# 2. 训练配置
SESSION_NAME="sparse4d_0925_2wframe"
GPU_IDS="4,5,6,7"
NUM_GPUS=4
MASTER_PORT=28670

# 3. 日志配置
LOG_DIR="$PROJECT_DIR/train_log"
LOG_FILE="$LOG_DIR/0925use_deformable_func_16261frame_r50_bs88.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# 4. 检查 tmux 会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux 会话已存在: $SESSION_NAME"
    echo "进入会话: tmux attach -t $SESSION_NAME"
    exit 1
fi

# 5. 设置环境变量
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH="$PROJECT_DIR/..:$PYTHONPATH"

# 6. 构建训练命令
TRAIN_CMD="python3 -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    $CONFIG \
    --no-validate \
    --launcher pytorch \
    > $LOG_FILE 2>&1"

# 7. 创建 tmux 会话并启动训练
tmux new-session -d -s "$SESSION_NAME" \
    "cd '$PROJECT_DIR' && $TRAIN_CMD"

echo "======================================"
echo "Sparse4D 训练已启动"
echo "tmux 会话: $SESSION_NAME"
echo "GPU: $GPU_IDS"
echo "GPU 数量: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "日志: $LOG_FILE"
echo "======================================"
echo "查看日志: tail -f $LOG_FILE"
echo "进入会话: tmux attach -t $SESSION_NAME"
