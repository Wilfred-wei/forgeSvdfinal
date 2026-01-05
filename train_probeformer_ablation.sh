#!/bin/bash

# ProbeFormer vs FAFormer Ablation Study Script
# This script compares ProbeFormer (multi-probe mechanism) with FAFormer (single CLS token)

# Activate the environment
source activate AIDE
export CUDA_VISIBLE_DEVICES=0

# Define paths
TRAIN_DATA="/sda/home/temp/weiwenfei/Datasets/CNNSpot_Split/train_small"
VAL_DATA="/sda/home/temp/weiwenfei/Datasets/progan_val"
TEST_DATA="/sda/home/temp/weiwenfei/Datasets/CnnDetTest"

# Pre-trained Stage1 model path (global parameter)
INTERMEDIATE_MODEL="/sda/home/temp/weiwenfei/forgenSvd/check_points/svd_ablation_with_svd/train_stage_1/model/intermediate_model_best.pth"

# Common training parameters
BATCH_SIZE=16
EVAL_BATCH_SIZE=64
EPOCHS=5
LR=2e-6
SEED=3407
NUM_WORKERS=8

echo "=========================================="
echo "ProbeFormer vs FAFormer Ablation Study"
echo "=========================================="
echo "Train data: ${TRAIN_DATA}"
echo "Val data: ${VAL_DATA}"
echo "Test data: ${TEST_DATA}"
echo "=========================================="

# ==========================================
# Experiment 1: FAFormer (Baseline)
# ==========================================
# echo ""
# echo "=========================================="
# echo "Experiment 1: FAFormer (Baseline)"
# echo "Using single CLS token mechanism"
# echo "=========================================="

# EXP_NAME="faformer_baseline"

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
#     --intermediate_model_path ${INTERMEDIATE_MODEL} \
#     --num_workers ${NUM_WORKERS} \
#     --seed ${SEED}

# echo ""
# echo "Evaluating FAFormer model..."
# python evaluate.py \
#     --experiment_name ${EXP_NAME} \
#     --eval_data_root ${TEST_DATA} \
#     --eval_stage 2 \
#     --batch_size ${EVAL_BATCH_SIZE} \
#     --num_workers ${NUM_WORKERS} \
#     --seed ${SEED}


# ==========================================
# Experiment 2: ProbeFormer (5 probes)
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 2: ProbeFormer (5 probes)"
echo "Using multi-probe mechanism with 5 probes"
echo "=========================================="

EXP_NAME="probeformer_5probes"

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
    --num_probes 5 \
    --intermediate_model_path ${INTERMEDIATE_MODEL} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo ""
echo "Evaluating ProbeFormer (5 probes) model..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${TEST_DATA} \
    --eval_stage 2 \
    --batch_size ${EVAL_BATCH_SIZE} \
    --use_probeformer \
    --num_probes 5 \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}


# ==========================================
# Experiment 3: ProbeFormer (3 probes)
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 3: ProbeFormer (3 probes)"
echo "Using multi-probe mechanism with 3 probes"
echo "=========================================="

EXP_NAME="probeformer_3probes"

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
    --num_probes 3 \
    --intermediate_model_path ${INTERMEDIATE_MODEL} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo ""
echo "Evaluating ProbeFormer (3 probes) model..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${TEST_DATA} \
    --eval_stage 2 \
    --batch_size ${EVAL_BATCH_SIZE} \
    --use_probeformer \
    --num_probes 3 \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}


# ==========================================
# Experiment 4: ProbeFormer (2 probes)
# ==========================================
echo ""
echo "=========================================="
echo "Experiment 4: ProbeFormer (2 probes)"
echo "Using multi-probe mechanism with 2 probes"
echo "=========================================="

EXP_NAME="probeformer_2probes"

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
    --num_probes 2 \
    --intermediate_model_path ${INTERMEDIATE_MODEL} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

echo ""
echo "Evaluating ProbeFormer (2 probes) model..."
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ${TEST_DATA} \
    --eval_stage 2 \
    --batch_size ${EVAL_BATCH_SIZE} \
    --use_probeformer \
    --num_probes 2 \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}


echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Summary of experiments:"
echo "1. faformer_baseline - FAFormer with single CLS token"
echo "2. probeformer_8probes - ProbeFormer with 5 probes"
echo "3. probeformer_3probes - ProbeFormer with 3 probes"
echo "4. probeformer_2probes - ProbeFormer with 2 probes"
echo "=========================================="
