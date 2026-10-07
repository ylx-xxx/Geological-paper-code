"""Run every planned comparison, freeze all weights, then evaluate the benchmark."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import pairs, sha256, write_json


def planned_jobs():
    seeds = [2026, 2027, 2028]
    arms = [
        ("rgb_loveda", "rgb", "loveda", "mean"),
        ("full14_loveda_mean", "full14", "loveda", "mean"),
        ("full14_loveda_random", "full14", "loveda", "random"),
        ("full14_scratch", "full14", "random", "mean"),
    ]
    jobs = [
        dict(
            name=f"{name}_s{seed}",
            channels=channels,
            initialization=initialization,
            patch_extra=extra,
            seed=seed,
            protocol="train_only",
        )
        for seed in seeds
        for name, channels, initialization, extra in arms
    ]
    jobs += [
        dict(
            name=f"trainval_{extra}_s{seed}",
            channels="full14",
            initialization="loveda",
            patch_extra=extra,
            seed=seed,
            protocol="trainval_fixed",
        )
        for seed in seeds
        for extra in ["mean", "random"]
    ]
    return jobs


def terminate_process_group(process):
    # This orchestrator targets the Linux GPU environment used for the study.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--loveda-checkpoint", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--preserve-checkpoint",
        help="Optional original model whose hash is checked before and after",
    )
    parser.add_argument("--training-budget-seconds", type=int, default=14400)
    parser.add_argument("--prior-training-seconds", type=float, default=0)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError(
            "Run this orchestration command in the Linux GPU environment"
        )
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError(
            "Use a fresh suite directory; interrupted suites are retained"
        )
    if "checkpoints" in root.parts:
        raise ValueError(
            "Write new experiments outside original checkpoint directories"
        )
    source = Path(args.loveda_checkpoint).resolve()
    if sha256(source) != args.expected_source_sha256:
        raise ValueError("LoveDA source checkpoint differs from the frozen identity")
    data = Path(args.data_root).resolve()
    if len(pairs(data, "train")) != 3799 or len(pairs(data, "val")) != 245:
        raise ValueError(
            "Expected the native Landslide4Sense training and validation splits"
        )
    if not 0 <= args.prior_training_seconds < args.training_budget_seconds:
        raise ValueError("Invalid cumulative training budget")
    code = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True)
    jobs = planned_jobs()
    original_hash = (
        sha256(args.preserve_checkpoint) if args.preserve_checkpoint else None
    )
    plan = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [2026, 2027, 2028],
        "epochs": 60,
        "jobs": jobs,
        "total_training_budget_seconds": args.training_budget_seconds,
        "prior_training_seconds": args.prior_training_seconds,
        "no_model_evaluation_until_all_training_complete": True,
        "selection": "TrainData-only: validation foreground IoU; TrainVal: fixed epoch 60",
        "tta_views": 3,
        "prediction": "argmax of mean logits; no threshold tuning",
        "test_role": "previously exposed benchmark; no test-based promotion",
        "source_sha256": args.expected_source_sha256,
        "original_v8_sha256": original_hash,
        "code_sha256": {
            str(path.relative_to(code)): sha256(path)
            for folder in [code / "reproducibility", code / "train_l4s_qz"]
            for path in sorted(folder.glob("*.py"))
        },
        "orchestrator": "portable wrapper; see executed snapshot for the original dated run",
    }
    write_json(root / "matched_plan.json", plan)
    records, used = [], args.prior_training_seconds
    started = time.monotonic()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")

    def run(name, argv, timeout, training=False):
        nonlocal used
        begin = time.monotonic()
        write_json(
            root / "matched_status.json",
            {
                "phase": "training" if training else "evaluation",
                "job": name,
                "completed_jobs": len(records),
                "training_seconds_used": used,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        with (root / f"{name}.log").open("x") as stream:
            process = subprocess.Popen(
                [sys.executable, *argv],
                cwd=code,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_process_group(process)
                rc = 124
        elapsed = time.monotonic() - begin
        if training:
            used += elapsed
        records.append(
            dict(job=name, returncode=rc, seconds=elapsed, training=training)
        )
        write_json(root / "matched_jobs.json", records)
        if rc:
            raise RuntimeError(f"{name} failed with exit code {rc}")

    try:
        run("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], 180)
        for job in jobs:
            allowance = min(1200, int(args.training_budget_seconds - used) - 90)
            if allowance < 180:
                raise RuntimeError(
                    "Cumulative training budget exhausted; evaluation remains unopened"
                )
            run(
                job["name"],
                [
                    "-m",
                    "reproducibility.train",
                    "--data-root",
                    str(data),
                    "--out",
                    str(root / "matched_runs" / job["name"]),
                    "--loveda-checkpoint",
                    str(source),
                    "--channels",
                    job["channels"],
                    "--initialization",
                    job["initialization"],
                    "--patch-extra",
                    job["patch_extra"],
                    "--protocol",
                    job["protocol"],
                    "--seed",
                    str(job["seed"]),
                    "--epochs",
                    "60",
                    "--max-seconds",
                    str(allowance),
                ],
                allowance + 60,
                training=True,
            )
            summary = json.loads(
                (root / "matched_runs" / job["name"] / "summary.json").read_text()
            )
            if summary["status"] != "completed" or summary["completed_epochs"] != 60:
                raise RuntimeError("Incomplete training; do not evaluate the suite")
        frozen = []
        for job in jobs:
            name = (
                "fixed_final.pt" if job["protocol"] == "trainval_fixed" else "best.pt"
            )
            checkpoint = root / "matched_runs" / job["name"] / name
            frozen.append(
                job
                | {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
        write_json(
            root / "matched_checkpoints_frozen.json",
            {
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "models": frozen,
            },
        )
        for spec in frozen:
            run(
                "test_" + spec["name"],
                [
                    "-m",
                    "reproducibility.evaluate",
                    "--data-root",
                    str(data),
                    "--checkpoint",
                    spec["checkpoint"],
                    "--expected-sha256",
                    spec["checkpoint_sha256"],
                    "--channels",
                    spec["channels"],
                    "--out",
                    str(root / "matched_test" / spec["name"]),
                ],
                240,
            )
        if original_hash and sha256(args.preserve_checkpoint) != original_hash:
            raise ValueError("The protected checkpoint changed")
        write_json(
            root / "matched_complete.json",
            {
                "status": "complete",
                "jobs": len(jobs),
                "training_seconds_including_prior": used,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        write_json(
            root / "matched_status.json",
            {"phase": "complete", "training_seconds_used": used},
        )
    except Exception as exc:
        write_json(
            root / "matched_status.json",
            {
                "phase": "failed",
                "error": str(exc),
                "training_seconds_used": used,
            },
        )
        raise


if __name__ == "__main__":
    main()
