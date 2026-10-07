#!/usr/bin/env bash
set -e

NAME=$1
EPOCHS=$2
LR=$3
MIN_LR=$4
WEIGHT_DECAY=$5
AUTO_CW=$6

if [ -z "$NAME" ]; then
  echo "Usage: bash run_candidate.sh <name> <epochs> <lr> <min_lr> <weight_decay>"
  echo "Example: bash run_candidate.sh lr3e6_ep5 5 3e-6 1e-6 0.05"
  exit 1
fi

if [ -z "$EPOCHS" ]; then EPOCHS=5; fi
if [ -z "$LR" ]; then LR=3e-6; fi
if [ -z "$MIN_LR" ]; then MIN_LR=1e-6; fi
if [ -z "$WEIGHT_DECAY" ]; then WEIGHT_DECAY=0.05; fi
if [ -z "$AUTO_CW" ]; then AUTO_CW=1; fi

ROOT=/root/autodl-tmp/final_tune_lab
CODE=$ROOT/code/train_l4s_qz_tune
BEST=$ROOT/best_pool/current_best.pth
DATA=/root/autodl-tmp/datasetss/landslide4Sense_trainval
LOVEDA=/root/autodl-tmp/checkpoints/train_la_qz/loveda_best.pth

TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR=$ROOT/runs/${TIME}_${NAME}

SAVE_DIR=$RUN_DIR/checkpoints
LOG_DIR=$RUN_DIR/logs

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

echo "===================================================================================================="
echo "Run name:      $NAME"
echo "Run dir:       $RUN_DIR"
echo "Resume ckpt:   $BEST"
echo "Epochs:        $EPOCHS"
echo "LR:            $LR"
echo "Min LR:        $MIN_LR"
echo "Weight decay:  $WEIGHT_DECAY"
echo "Auto class weights: $AUTO_CW"
echo "===================================================================================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $CODE/train.py \
  --data_root $DATA \
  --loveda_ckpt $LOVEDA \
  --resume_ckpt $BEST \
  --save_dir $SAVE_DIR \
  --log_dir $LOG_DIR \
  --img_size 128 \
  --batch_size 64 \
  --epochs $EPOCHS \
  --lr $LR \
  --min_lr $MIN_LR \
  --weight_decay $WEIGHT_DECAY \
  --num_workers 8 \
  --amp 1 \
  --channels_last 1 \
  --tta 1 \
  --use_loveda_pretrain 1 \
  --auto_class_weights $AUTO_CW

python /root/autodl-tmp/train_l4s_qz/test.py \
  --data_root $DATA \
  --ckpt $SAVE_DIR/last.pth \
  --out_csv $LOG_DIR/test_metrics_last.csv \
  --img_size 128 \
  --batch_size 64 \
  --num_workers 8 \
  --amp 1 \
  --channels_last 1 \
  --tta 1

python $ROOT/tools/promote_if_better.py \
  --candidate_ckpt $SAVE_DIR/last.pth \
  --metrics_csv $LOG_DIR/test_metrics_last.csv \
  --run_name ${TIME}_${NAME} \
  --best_dir $ROOT/best_pool \
  --metric_key Landslide_IoU \
  --min_delta 0.0001

echo "Finished run: $RUN_DIR"
