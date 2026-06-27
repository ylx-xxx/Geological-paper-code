#!/usr/bin/env bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/root/autodl-tmp/q2_extra_ablation
CODE=$BASE/code/train_l4s_qz_ablation
RUNS=$BASE/runs

DATA_ROOT=/root/autodl-tmp/datasetss/landslide4Sense
LOVEDA=/root/autodl-tmp/checkpoints/train_la_qz/loveda_best.pth
FINAL_CKPT=/root/autodl-tmp/final_tune_lab/best_pool/current_best.pth

EPOCHS=${EPOCHS:-60}
BS=${BS:-64}
NW=${NW:-8}

MODE=${1:-all}

train_and_test () {
  NAME=$1
  CHANNEL=$2
  PATCH=$3
  LOSS=$4
  SEED=$5

  SAVE_DIR=$RUNS/$NAME/checkpoints
  LOG_DIR=$RUNS/$NAME/logs

  mkdir -p $SAVE_DIR $LOG_DIR

  echo "===================================================================================================="
  echo "[TRAIN] $NAME | channel=$CHANNEL | patch=$PATCH | loss=$LOSS | seed=$SEED"
  echo "===================================================================================================="

  python $CODE/train_ablation.py \
    --data_root $DATA_ROOT \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR \
    --loveda_ckpt $LOVEDA \
    --channel_mode $CHANNEL \
    --patch_init $PATCH \
    --loss_mode $LOSS \
    --img_size 128 \
    --batch_size $BS \
    --epochs $EPOCHS \
    --lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.05 \
    --num_workers $NW \
    --amp 1 \
    --channels_last 1 \
    --grad_clip 1.0 \
    --val_tta 1 \
    --seed $SEED \
    --use_loveda_pretrain 1

  echo "===================================================================================================="
  echo "[TEST] $NAME"
  echo "===================================================================================================="

  python $CODE/test_ablation.py \
    --data_root $DATA_ROOT \
    --ckpt $SAVE_DIR/best.pth \
    --out_csv $LOG_DIR/test_metrics.csv \
    --channel_mode $CHANNEL \
    --img_size 128 \
    --batch_size $BS \
    --num_workers $NW \
    --amp 1 \
    --channels_last 1 \
    --tta 1
}


test_final_tta () {
  NAME=$1
  TTA_FLAG=$2

  LOG_DIR=$RUNS/$NAME/logs
  mkdir -p $LOG_DIR

  echo "===================================================================================================="
  echo "[TTA TEST] $NAME | tta=$TTA_FLAG"
  echo "===================================================================================================="

  python $CODE/test_ablation.py \
    --data_root $DATA_ROOT \
    --ckpt $FINAL_CKPT \
    --out_csv $LOG_DIR/test_metrics.csv \
    --channel_mode full14 \
    --img_size 128 \
    --batch_size $BS \
    --num_workers $NW \
    --amp 1 \
    --channels_last 1 \
    --tta $TTA_FLAG
}


if [ "$MODE" = "smoke" ]; then
  EPOCHS=1
  train_and_test smoke_rgb rgb mean ce_dice_cw 42
  exit 0
fi


if [ "$MODE" = "input" ] || [ "$MODE" = "all" ]; then
  train_and_test input_rgb rgb mean ce_dice_cw 42
  train_and_test input_ms12 ms12 mean ce_dice_cw 42
  train_and_test input_rgb_topo rgb_topo mean ce_dice_cw 42
  train_and_test input_full14 full14 mean ce_dice_cw 42
fi


if [ "$MODE" = "patch" ] || [ "$MODE" = "all" ]; then
  train_and_test patch_random_extra full14 random ce_dice_cw 42
  # patch_mean_extra is equivalent to input_full14 when EPOCHS/seed/config are unchanged.
  # If you want a separate repeated run, uncomment the following line.
  # train_and_test patch_mean_extra full14 mean ce_dice_cw 42
fi


if [ "$MODE" = "loss" ] || [ "$MODE" = "all" ]; then
  train_and_test loss_ce full14 mean ce 42
  train_and_test loss_ce_dice full14 mean ce_dice 42
  # CE + Dice + class weight is represented by input_full14.
fi


if [ "$MODE" = "tta" ] || [ "$MODE" = "all" ]; then
  test_final_tta tta_off 0
  test_final_tta tta_on 1
fi


if [ "$MODE" = "seed" ] || [ "$MODE" = "all" ]; then
  train_and_test seed_2026 full14 mean ce_dice_cw 2026
  train_and_test seed_2027 full14 mean ce_dice_cw 2027
  train_and_test seed_2028 full14 mean ce_dice_cw 2028
fi

echo "===================================================================================================="
echo "All requested extra ablations finished."
echo "===================================================================================================="
