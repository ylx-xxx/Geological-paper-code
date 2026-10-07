#!/usr/bin/env bash
set -e

echo "===================================================================================================="
echo "[1/8] 创建 final_tune_lab 目录"
echo "===================================================================================================="

ROOT=/root/autodl-tmp/final_tune_lab
BEST_DIR=$ROOT/best_pool
RUNS_DIR=$ROOT/runs
TOOLS_DIR=$ROOT/tools
CODE_DIR=$ROOT/code
TUNE_CODE=$CODE_DIR/train_l4s_qz_tune

mkdir -p $BEST_DIR/history
mkdir -p $RUNS_DIR
mkdir -p $TOOLS_DIR
mkdir -p $CODE_DIR

echo "===================================================================================================="
echo "[2/8] 检查 V8 最佳模型是否存在"
echo "===================================================================================================="

V8_BEST=/root/autodl-tmp/checkpoints/train_l4s_qz_v8_trainval/last.pth

if [ ! -f "$V8_BEST" ]; then
  echo "[Error] 找不到 V8 最佳模型：$V8_BEST"
  echo "请确认该文件存在。"
  exit 1
fi

ls -lh "$V8_BEST"

echo "===================================================================================================="
echo "[3/8] 锁定原始 V8 最佳模型，并创建 current_best.pth"
echo "===================================================================================================="

cp "$V8_BEST" $BEST_DIR/original_v8_epoch60_locked.pth
cp "$V8_BEST" $BEST_DIR/current_best.pth

echo "0.4793" > $BEST_DIR/current_best_score.txt

cat > $BEST_DIR/current_best_meta.txt <<'TXT'
model_name: V8 TrainVal Final epoch60
checkpoint: current_best.pth
source_checkpoint: /root/autodl-tmp/checkpoints/train_l4s_qz_v8_trainval/last.pth
test_mIoU: 0.7328
test_Landslide_IoU: 0.4793
test_Landslide_F1: 0.6480
test_Precision: 0.6357
test_Recall: 0.6608
note: original_v8_epoch60_locked.pth is locked and must not be overwritten.
TXT

echo "当前最佳模型已锁定："
ls -lh $BEST_DIR/original_v8_epoch60_locked.pth
ls -lh $BEST_DIR/current_best.pth

echo "===================================================================================================="
echo "[4/8] 备份到 /root/autodl-fs"
echo "===================================================================================================="

mkdir -p /root/autodl-fs/geology_backup/final_v8_epoch60_locked

cp $BEST_DIR/original_v8_epoch60_locked.pth \
   /root/autodl-fs/geology_backup/final_v8_epoch60_locked/original_v8_epoch60_locked.pth

cp -r /root/autodl-tmp/logs/train_l4s_qz_v8_trainval \
   /root/autodl-fs/geology_backup/final_v8_epoch60_locked/logs 2>/dev/null || true

cp -r /root/autodl-tmp/train_l4s_qz \
   /root/autodl-fs/geology_backup/final_v8_epoch60_locked/code 2>/dev/null || true

echo "已备份到 /root/autodl-fs/geology_backup/final_v8_epoch60_locked"

echo "===================================================================================================="
echo "[5/8] 复制微调代码"
echo "===================================================================================================="

if [ ! -d "/root/autodl-tmp/train_l4s_qz" ]; then
  echo "[Error] 找不到原始代码目录：/root/autodl-tmp/train_l4s_qz"
  exit 1
fi

rm -rf "$TUNE_CODE"
cp -r /root/autodl-tmp/train_l4s_qz "$TUNE_CODE"

echo "微调代码目录：$TUNE_CODE"

echo "===================================================================================================="
echo "[6/8] 给微调 train.py 添加 --resume_ckpt"
echo "===================================================================================================="

python - <<'PY'
from pathlib import Path

p = Path("/root/autodl-tmp/final_tune_lab/code/train_l4s_qz_tune/train.py")
text = p.read_text()

# 1. 添加 parser 参数
if '--resume_ckpt' not in text:
    target = 'parser.add_argument("--loveda_ckpt"'
    lines = text.splitlines()
    new_lines = []
    inserted = False

    for line in lines:
        new_lines.append(line)
        if (not inserted) and target in line:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}parser.add_argument("--resume_ckpt", type=str, default="")')
            inserted = True

    if not inserted:
        raise RuntimeError("没有找到 loveda_ckpt 参数位置，无法自动插入 --resume_ckpt")

    text = "\n".join(new_lines) + "\n"

# 2. 添加 resume 加载逻辑
resume_code = '''
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"[Info] Loading checkpoint for continuous fine-tuning: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        msg = model.load_state_dict(state, strict=False)
        print(f"[Info] Resume missing keys: {len(msg.missing_keys)}")
        print(f"[Info] Resume unexpected keys: {len(msg.unexpected_keys)}")
'''

if "Loading checkpoint for continuous fine-tuning" not in text:
    old_block = '''    if args.use_loveda_pretrain:
        load_loveda_pretrained(model, args.loveda_ckpt)

    model = model.to(device)
'''

    new_block = '''    if args.use_loveda_pretrain:
        load_loveda_pretrained(model, args.loveda_ckpt)
''' + resume_code + '''
    model = model.to(device)
'''

    if old_block not in text:
        raise RuntimeError("没有找到加载 LoveDA 后 model.to(device) 的代码块，无法自动插入 resume 逻辑")

    text = text.replace(old_block, new_block)

p.write_text(text)
print("已完成 train.py 修改：支持 --resume_ckpt")
PY

python -m py_compile $TUNE_CODE/train.py

echo "train.py 编译通过"

echo "===================================================================================================="
echo "[7/8] 创建自动晋升脚本 promote_if_better.py"
echo "===================================================================================================="

cat > $TOOLS_DIR/promote_if_better.py <<'PY'
import csv
import shutil
import argparse
from datetime import datetime
from pathlib import Path


def read_metric(csv_path, metric_key):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError(f"Empty metrics CSV: {csv_path}")

    row = rows[-1]
    clean = {str(k).strip(): v for k, v in row.items()}

    if metric_key not in clean:
        print("[Error] Available metric keys:")
        for k in clean.keys():
            print(" -", k)
        raise KeyError(f"Metric key not found: {metric_key}")

    return float(clean[metric_key])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate_ckpt", type=str, required=True)
    parser.add_argument("--metrics_csv", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--best_dir", type=str, default="/root/autodl-tmp/final_tune_lab/best_pool")
    parser.add_argument("--metric_key", type=str, default="Landslide_IoU")
    parser.add_argument("--min_delta", type=float, default=0.0001)

    args = parser.parse_args()

    best_dir = Path(args.best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    current_best = best_dir / "current_best.pth"
    current_score_file = best_dir / "current_best_score.txt"
    leaderboard = best_dir / "leaderboard.csv"
    history_dir = best_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    candidate_score = read_metric(args.metrics_csv, args.metric_key)

    if current_score_file.exists():
        best_score = float(current_score_file.read_text().strip())
    else:
        best_score = -1.0

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    promoted = candidate_score > best_score + args.min_delta

    if promoted:
        if current_best.exists():
            backup_path = history_dir / f"previous_best_{best_score:.6f}_{now}.pth"
            shutil.copy2(current_best, backup_path)

        shutil.copy2(args.candidate_ckpt, current_best)
        current_score_file.write_text(f"{candidate_score:.6f}")

        shutil.copy2(args.metrics_csv, best_dir / "current_best_metrics.csv")

        with open(best_dir / "current_best_meta.txt", "w") as f:
            f.write(f"run_name: {args.run_name}\n")
            f.write(f"metric_key: {args.metric_key}\n")
            f.write(f"score: {candidate_score:.6f}\n")
            f.write(f"promoted_time: {now}\n")
            f.write(f"candidate_ckpt: {args.candidate_ckpt}\n")
            f.write(f"metrics_csv: {args.metrics_csv}\n")

        status = "PROMOTED"
    else:
        status = "NOT_PROMOTED"

    write_header = not leaderboard.exists()

    with open(leaderboard, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "time",
                "run_name",
                "metric_key",
                "candidate_score",
                "previous_best",
                "status",
                "candidate_ckpt",
                "metrics_csv",
            ])

        writer.writerow([
            now,
            args.run_name,
            args.metric_key,
            round(candidate_score, 6),
            round(best_score, 6),
            status,
            args.candidate_ckpt,
            args.metrics_csv,
        ])

    print("=" * 100)
    print(f"Run name:        {args.run_name}")
    print(f"Metric key:      {args.metric_key}")
    print(f"Candidate score: {candidate_score:.6f}")
    print(f"Previous best:   {best_score:.6f}")
    print(f"Status:          {status}")
    print(f"Current best:    {current_best}")
    print(f"Leaderboard:     {leaderboard}")
    print("=" * 100)


if __name__ == "__main__":
    main()
PY

python -m py_compile $TOOLS_DIR/promote_if_better.py

echo "promote_if_better.py 编译通过"

echo "===================================================================================================="
echo "[8/8] 创建一键运行脚本 run_candidate.sh"
echo "===================================================================================================="

cat > $ROOT/run_candidate.sh <<'SH'
#!/usr/bin/env bash
set -e

NAME=$1
EPOCHS=$2
LR=$3
MIN_LR=$4
WEIGHT_DECAY=$5

if [ -z "$NAME" ]; then
  echo "Usage: bash run_candidate.sh <name> <epochs> <lr> <min_lr> <weight_decay>"
  echo "Example: bash run_candidate.sh lr3e6_ep5 5 3e-6 1e-6 0.05"
  exit 1
fi

if [ -z "$EPOCHS" ]; then EPOCHS=5; fi
if [ -z "$LR" ]; then LR=3e-6; fi
if [ -z "$MIN_LR" ]; then MIN_LR=1e-6; fi
if [ -z "$WEIGHT_DECAY" ]; then WEIGHT_DECAY=0.05; fi

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
  --auto_class_weights 1

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
