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
