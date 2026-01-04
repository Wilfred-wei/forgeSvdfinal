#!/bin/bash

# SVD Ablation Study Script
# This script compares models with and without SVD orthogonal constraint

# Activate the environment
source activate AIDE
export CUDA_VISIBLE_DEVICES=1
# Define paths
TRAIN_DATA="/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small"
VAL_DATA="/sda/home/temp/weiwenfei/Datasets/progan_val"
TEST_DATA="/sda/home/temp/weiwenfei/Datasets/CnnDetTest"

# Common training parameters
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
EPOCHS=10
LR=0.00005
WSGM_COUNT=12
WSGM_REDUCTION=4
SEED=3407

echo "=========================================="
echo "SVD Ablation Study"
echo "=========================================="
echo "Train data: ${TRAIN_DATA}"
echo "Val data: ${VAL_DATA}"
echo "Test data: ${TEST_DATA}"
echo "=========================================="

# # ==========================================
# # Experiment 1: With SVD Orthogonal Constraint
# # ==========================================
# echo ""
# echo "=========================================="
# echo "Experiment 1: WITH SVD Orthogonal Constraint"
# echo "svd_rank=64, orth_lambda=0.1"
# echo "=========================================="

# EXP_NAME="svd_ablation_with_svd"

# python train.py \
#     --experiment_name ${EXP_NAME} \
#     --train_data_root ${TRAIN_DATA} \
#     --val_data_root ${VAL_DATA} \
#     --train_classes car cat chair horse \
#     --val_classes car cat chair horse \
#     --training_stage 1 \
#     --stage1_batch_size ${BATCH_SIZE} \
#     --stage1_epochs ${EPOCHS} \
#     --stage1_learning_rate ${LR} \
#     --stage1_lr_decay_step 2 \
#     --stage1_lr_decay_factor 0.7 \
#     --WSGM_count ${WSGM_COUNT} \
#     --WSGM_reduction_factor ${WSGM_REDUCTION} \
#     --svd_rank 64 \
#     --orth_lambda 0.1 \
#     --num_workers 4 \
#     --seed ${SEED}

# echo ""
# echo "Evaluating with SVD model..."
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root ${TEST_DATA} \
#     --eval_stage 1 \
#     --batch_size ${EVAL_BATCH_SIZE} \
#     --WSGM_count ${WSGM_COUNT} \
#     --WSGM_reduction_factor ${WSGM_REDUCTION} \
#     --svd_rank 64 \
#     --orth_lambda 0.1 \
#     --num_workers 4 \
#     --seed ${SEED}

# # ==========================================
# # Experiment 2: WITHOUT SVD Orthogonal Constraint (Baseline)
# # ==========================================
# echo ""
# echo "=========================================="
# echo "Experiment 2: WITHOUT SVD (Baseline)"
# echo "orth_lambda=0.0"
# echo "=========================================="

# EXP_NAME="svd_ablation_without_svd"

# python train.py \
#     --experiment_name ${EXP_NAME} \
#     --train_data_root ${TRAIN_DATA} \
#     --val_data_root ${VAL_DATA} \
#     --train_classes car cat chair horse \
#     --val_classes car cat chair horse \
#     --training_stage 1 \
#     --stage1_batch_size ${BATCH_SIZE} \
#     --stage1_epochs ${EPOCHS} \
#     --stage1_learning_rate ${LR} \
#     --stage1_lr_decay_step 2 \
#     --stage1_lr_decay_factor 0.7 \
#     --WSGM_count ${WSGM_COUNT} \
#     --WSGM_reduction_factor ${WSGM_REDUCTION} \
#     --svd_rank 64 \
#     --orth_lambda 0.0 \
#     --num_workers 4 \
#     --seed ${SEED}

# echo ""
# echo "Evaluating without SVD model..."
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root ${TEST_DATA} \
#     --eval_stage 1 \
#     --batch_size ${EVAL_BATCH_SIZE} \
#     --WSGM_count ${WSGM_COUNT} \
#     --WSGM_reduction_factor ${WSGM_REDUCTION} \
#     --svd_rank 64 \
#     --orth_lambda 0.0 \
#     --num_workers 4 \
#     --seed ${SEED}

# ==========================================
# Experiment 3: Different SVD ranks comparison
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 3: Testing different SVD ranks"
echo "=========================================="

for RANK in 32 64 128; do
    echo ""
    echo "Testing svd_rank=${RANK}..."
    EXP_NAME="svd_rank_${RANK}"

    python train.py \
        --experiment_name ${EXP_NAME} \
        --train_data_root ${TRAIN_DATA} \
        --val_data_root ${VAL_DATA} \
        --train_classes car cat chair horse \
        --val_classes car cat chair horse \
        --training_stage 1 \
        --stage1_batch_size ${BATCH_SIZE} \
        --stage1_epochs ${EPOCHS} \
        --stage1_learning_rate ${LR} \
        --stage1_lr_decay_step 2 \
        --stage1_lr_decay_factor 0.7 \
        --WSGM_count ${WSGM_COUNT} \
        --WSGM_reduction_factor ${WSGM_REDUCTION} \
        --svd_rank ${RANK} \
        --orth_lambda 0.1 \
        --num_workers 4 \
        --seed ${SEED}

    python evaluate.py \
        --experiment_name ${EXP_NAME} \
        --eval_data_root ${TEST_DATA} \
        --eval_stage 1 \
        --batch_size ${EVAL_BATCH_SIZE} \
        --WSGM_count ${WSGM_COUNT} \
        --WSGM_reduction_factor ${WSGM_REDUCTION} \
        --svd_rank ${RANK} \
        --orth_lambda 0.1 \
        --num_workers 4 \
        --seed ${SEED}
done

# ==========================================
# Experiment 4: Different orth_lambda values
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 4: Testing different orth_lambda values"
echo "=========================================="

for LAMBDA in 0.01 0.05 0.1 0.5; do
    echo ""
    echo "Testing orth_lambda=${LAMBDA}..."
    EXP_NAME="orth_lambda_${LAMBDA}"

    python train.py \
        --experiment_name ${EXP_NAME} \
        --train_data_root ${TRAIN_DATA} \
        --val_data_root ${VAL_DATA} \
        --train_classes car cat chair horse \
        --val_classes car cat chair horse \
        --training_stage 1 \
        --stage1_batch_size ${BATCH_SIZE} \
        --stage1_epochs ${EPOCHS} \
        --stage1_learning_rate ${LR} \
        --stage1_lr_decay_step 2 \
        --stage1_lr_decay_factor 0.7 \
        --WSGM_count ${WSGM_COUNT} \
        --WSGM_reduction_factor ${WSGM_REDUCTION} \
        --svd_rank 64 \
        --orth_lambda ${LAMBDA} \
        --num_workers 4 \
        --seed ${SEED}

    python evaluate.py \
        --experiment_name ${EXP_NAME} \
        --eval_data_root ${TEST_DATA} \
        --eval_stage 1 \
        --batch_size ${EVAL_BATCH_SIZE} \
        --WSGM_count ${WSGM_COUNT} \
        --WSGM_reduction_factor ${WSGM_REDUCTION} \
        --svd_rank 64 \
        --orth_lambda ${LAMBDA} \
        --num_workers 4 \
        --seed ${SEED}
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Summary of experiments:"
echo "1. svd_ablation_with_svd - SVD enabled (rank=64, lambda=0.1)"
echo "2. svd_ablation_without_svd - Baseline (lambda=0.0)"
echo "3. svd_rank_{32,64,128} - Different SVD ranks"
echo "4. orth_lambda_{0.01,0.05,0.1,0.5} - Different lambda values"
echo "=========================================="
