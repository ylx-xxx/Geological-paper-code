import csv
import argparse
import torch
from pathlib import Path


def load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt
    return ckpt, {"model": ckpt}


def average_checkpoints(ckpt_paths, out_path):
    assert len(ckpt_paths) > 0

    avg_state = None
    used = []

    for path in ckpt_paths:
        path = Path(path)
        if not path.exists():
            print(f"[Skip] not found: {path}")
            continue

        state, _ = load_state(path)

        if avg_state is None:
            avg_state = {}
            for k, v in state.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    avg_state[k] = v.clone().float()
                else:
                    avg_state[k] = v.clone() if torch.is_tensor(v) else v
            used.append(str(path))
        else:
            for k, v in state.items():
                if k not in avg_state:
                    continue
                if torch.is_tensor(v) and v.dtype.is_floating_point and avg_state[k].shape == v.shape:
                    avg_state[k] += v.float()
            used.append(str(path))

    if avg_state is None or len(used) == 0:
        raise RuntimeError("No valid checkpoints were loaded.")

    n = len(used)

    for k, v in avg_state.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            avg_state[k] = v / n

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": avg_state,
            "soup_used_checkpoints": used,
            "num_checkpoints": n,
        },
        out_path,
    )

    print("=" * 100)
    print(f"Saved soup checkpoint: {out_path}")
    print(f"Used checkpoints: {n}")
    for p in used:
        print(" -", p)
    print("=" * 100)


def get_top_checkpoints_from_leaderboard(leaderboard, min_score, topk):
    leaderboard = Path(leaderboard)

    if not leaderboard.exists():
        raise FileNotFoundError(f"leaderboard not found: {leaderboard}")

    rows = []

    with open(leaderboard, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row["candidate_score"])
            except Exception:
                continue

            ckpt = row.get("candidate_ckpt", "")

            if score >= min_score and ckpt:
                rows.append((score, ckpt, row.get("run_name", "")))

    rows = sorted(rows, key=lambda x: x[0], reverse=True)

    if topk > 0:
        rows = rows[:topk]

    print("=" * 100)
    print("Selected candidate checkpoints from leaderboard:")
    for score, ckpt, name in rows:
        print(f"{score:.6f} | {name} | {ckpt}")
    print("=" * 100)

    return [ckpt for score, ckpt, name in rows]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--current_best", type=str, default="/root/autodl-tmp/final_tune_lab/best_pool/current_best.pth")
    parser.add_argument("--leaderboard", type=str, default="/root/autodl-tmp/final_tune_lab/best_pool/leaderboard.csv")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--min_score", type=float, default=0.4700)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--include_current_best", type=int, default=1)

    args = parser.parse_args()

    ckpts = []

    if args.include_current_best:
        ckpts.append(args.current_best)

    ckpts += get_top_checkpoints_from_leaderboard(
        args.leaderboard,
        min_score=args.min_score,
        topk=args.topk,
    )

    # 去重但保留顺序
    seen = set()
    final_ckpts = []
    for p in ckpts:
        if p not in seen:
            final_ckpts.append(p)
            seen.add(p)

    average_checkpoints(final_ckpts, args.out)


if __name__ == "__main__":
    main()
