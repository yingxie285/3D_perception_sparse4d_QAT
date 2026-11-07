run_sparse4d_quant_mode() (
    set -euo pipefail

    cd /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws

    export HORIZON_QAT=1
    export CUDA_VISIBLE_DEVICES=7
    export PYTHONPATH="$PWD/..:${PYTHONPATH:-}"
    export SPARSE4D_PERF_MODE="$1"
    export QAT_OUTPUT_PREFIX="$2"

    mkdir -p "$PWD/Sparse4D_QAT/$QAT_OUTPUT_PREFIX"

    python3 script/qat_workflow_sparse4d.py \
      --config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
      --base-checkpoint ckpt/iter_36800.pth \
      --phase calib \
      --march nash-m \
      --device 0 \
      --launcher none \
      --no-validate \
      --skip-onnx \
      --opt 2 \
      --output-dir "$PWD/Sparse4D_QAT/$QAT_OUTPUT_PREFIX/qat_export" \
      2>&1 | tee "$PWD/Sparse4D_QAT/$QAT_OUTPUT_PREFIX/calib_compile.log"

    python3 script/test_calib.py \
      --config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
      --base-checkpoint ckpt/iter_36800.pth \
      --checkpoint "$PWD/Sparse4D_QAT/$QAT_OUTPUT_PREFIX.ckpt" \
      --march nash-m \
      --launcher none \
      --eval bbox NDS mAP \
      2>&1 | tee "$PWD/Sparse4D_QAT/$QAT_OUTPUT_PREFIX/eval_calib.log"
)

run_sparse4d_quant_mode linear8 perf_r4_linear8