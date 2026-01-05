#!/bin/bash

# Dual-Stream ProbeFormer Ablation Study Script
# Compares: Baseline vs Dual-Stream vs Diversity Loss vs Full (Dual-Stream + Diversity)

# Activate the environment
source activate AIDE
export CUDA_VISIBLE_DEVICES=0

# Define paths
TRAIN_DATA="/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small"
VAL_DATA="/sda/home/temp/weiwenfei/Datasets/progan_val"
TEST_DATA="/sda/home/temp/weiwenfei/Datasets/CnnDetTest"

# Pre-trained Stage1 model path
INTERMEDIATE_MODEL="/sda/home/temp/weiwenfei/forgenSvd/check_points/svd_ablation_with_svd/train_stage_1/model/intermediate_model_best.pth"

# Common training parameters
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
EPOCHS=5
LR=2e-6
SEED=3407
NUM_WORKERS=8
DIVERSITY_WEIGHT=0.5
NUM_PROBES=4

echo "=========================================="
echo "Dual-Stream ProbeFormer Ablation Study"
echo "=========================================="
echo "Train data: ${TRAIN_DATA}"
echo "Val data: ${VAL_DATA}"
echo "Test data: ${TEST_DATA}"
echo "=========================================="


# ==========================================
# Experiment 1: Baseline (Original ProbeFormer)
# CLS only, no dual-stream, no diversity loss
# ==========================================
# echo ""
# echo "=========================================="
# echo "Experiment 1: Baseline (CLS only)"
# echo "Original ProbeFormer without Dual-Stream or Diversity Loss"
# echo "=========================================="

# EXP_NAME="baseline_cls_only"

# python train.py \
#     --experiment_name ${EXP_NAME} \
#     --train_data_root ${TRAIN_DATA} \
#     --val_data_root ${VAL_DATA} \
#     --train_classes car cat chair horse \
#     --val_classes car cat chair horse \
#     --training_stage 2 \
#     --stage2_batch_size ${BATCH_SIZE} \
#     --stage2_epochs ${EPOCHS} \
#     --stage2_learning_rate ${LR} \
#     --stage2_lr_decay_step 2 \
#     --stage2_lr_decay_factor 0.7 \
#     --FAFormer_layers 2 \
#     --FAFormer_head 2 \
#     --FAFormer_reduction_factor 1 \
#     --use_probeformer \
#     --num_probes ${NUM_PROBES} \
#     --use_dual_stream False \
#     --diversity_weight 0.0 \
#     --intermediate_model_path ${INTERMEDIATE_MODEL} \
#     --num_workers ${NUM_WORKERS} \
#     --seed ${SEED}

# echo ""
# echo "Evaluating Baseline model..."
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root ${TEST_DATA} \
#     --eval_stage 2 \
#     --batch_size ${EVAL_BATCH_SIZE} \
#     --use_probeformer \
#     --num_probes ${NUM_PROBES} \
#     --num_workers ${NUM_WORKERS} \
#     --seed ${SEED}


# ==========================================
# Experiment 2: Dual-Stream Only
# CLS + AP, no diversity loss
# ==========================================
# echo ""
# echo "=========================================="
# echo "Experiment 2: Dual-Stream Only"
# echo "CLS + AP features, no Diversity Loss"
# echo "=========================================="

# EXP_NAME="dualstream_no_div"

# python train.py \
#     --experiment_name ${EXP_NAME} \
#     --train_data_root ${TRAIN_DATA} \
#     --val_data_root ${VAL_DATA} \
#     --train_classes car cat chair horse \
#     --val_classes car cat chair horse \
#     --training_stage 2 \
#     --stage2_batch_size ${BATCH_SIZE} \
#     --stage2_epochs ${EPOCHS} \
#     --stage2_learning_rate ${LR} \
#     --stage2_lr_decay_step 2 \
#     --stage2_lr_decay_factor 0.7 \
#     --FAFormer_layers 2 \
#     --FAFormer_head 2 \
#     --FAFormer_reduction_factor 1 \
#     --use_probeformer \
#     --num_probes ${NUM_PROBES} \
#     --use_dual_stream \
#     --diversity_weight 0.0 \
#     --intermediate_model_path ${INTERMEDIATE_MODEL} \
#     --num_workers ${NUM_WORKERS} \
#     --seed ${SEED}

# echo ""
# echo "Evaluating Dual-Stream Only model..."
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root ${TEST_DATA} \
#     --eval_stage 2 \
#     --batch_size ${EVAL_BATCH_SIZE} \
#     --use_probeformer \
#     --num_probes ${NUM_PROBES} \
#     --use_dual_stream \
#     --num_workers ${NUM_WORKERS} \
#     --seed ${SEED}


# ==========================================
# Experiment 3: Diversity Loss Only
# CLS only, with diversity loss
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 3: Diversity Loss Only"
echo "CLS only, with Diversity Loss (orthogonal probes)"
echo "=========================================="

EXP_NAME="diversity_loss_only"

python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root ${TRAIN_DATA} \
    --val_data_root ${VAL_DATA} \
    --train_classes car cat chair horse \
    --val_classes car cat chair horse \
    --training_stage 2 \
    --stage2_batch_size ${BATCH_SIZE} \
    --stage2_epochs ${EPOCHS} \
    --stage2_learning_rate ${LR} \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_head 2 \
    --FAFormer_reduction_factor 1 \
    --use_probeformer \
    --num_probes ${NUM_PROBES} \
    --diversity_weight ${DIVERSITY_WEIGHT} \
    --intermediate_model_path ${INTERMEDIATE_MODEL} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo ""
echo "Evaluating Diversity Loss Only model..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${TEST_DATA} \
    --eval_stage 2 \
    --batch_size ${EVAL_BATCH_SIZE} \
    --use_probeformer \
    --num_probes ${NUM_PROBES} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}


# ==========================================
# Experiment 4: Full Dual-Stream (Dual-Stream + Diversity)
# CLS + AP, with diversity loss
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 4: Full Dual-Stream"
echo "CLS + AP features, with Diversity Loss"
echo "=========================================="

EXP_NAME="full_dualstream"

python train.py \
    --experiment_name ${EXP_NAME} \
    --train_data_root ${TRAIN_DATA} \
    --val_data_root ${VAL_DATA} \
    --train_classes car cat chair horse \
    --val_classes car cat chair horse \
    --training_stage 2 \
    --stage2_batch_size ${BATCH_SIZE} \
    --stage2_epochs ${EPOCHS} \
    --stage2_learning_rate ${LR} \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_head 2 \
    --FAFormer_reduction_factor 1 \
    --use_probeformer \
    --num_probes ${NUM_PROBES} \
    --use_dual_stream \
    --diversity_weight ${DIVERSITY_WEIGHT} \
    --intermediate_model_path ${INTERMEDIATE_MODEL} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo ""
echo "Evaluating Full Dual-Stream model..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${TEST_DATA} \
    --eval_stage 2 \
    --batch_size ${EVAL_BATCH_SIZE} \
    --use_probeformer \
    --num_probes ${NUM_PROBES} \
    --use_dual_stream \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}


echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Summary of experiments:"
echo "1. baseline_cls_only - Original ProbeFormer (CLS only, no diversity)"
echo "2. dualstream_no_div - Dual-Stream (CLS+AP) without Diversity Loss"
echo "3. diversity_loss_only - CLS only with Diversity Loss"
echo "4. full_dualstream - Dual-Stream (CLS+AP) with Diversity Loss"
echo "=========================================="
echo "Expected conclusion:"
echo "- Dual-Stream helps by providing more texture info"
echo "- Diversity Loss helps probes specialize"
echo "- Best performance: full_dualstream"
echo "=========================================="
