import argparse
from pathlib import Path
import torch


def load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    base = load_state(args.base)
    cand = load_state(args.candidate)

    out_state = {}

    for k, v in base.items():
        if (
            k in cand
            and torch.is_tensor(v)
            and torch.is_tensor(cand[k])
            and v.shape == cand[k].shape
            and v.dtype.is_floating_point
            and cand[k].dtype.is_floating_point
        ):
            out_state[k] = args.alpha * v.float() + (1.0 - args.alpha) * cand[k].float()
        else:
            out_state[k] = v.clone() if torch.is_tensor(v) else v

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": out_state,
            "base": args.base,
            "candidate": args.candidate,
            "alpha": args.alpha,
        },
        out_path,
    )

    print("=" * 100)
    print(f"Saved weighted soup: {out_path}")
    print(f"base weight:      {args.alpha}")
    print(f"candidate weight: {1.0 - args.alpha}")
    print("=" * 100)


if __name__ == "__main__":
    main()
