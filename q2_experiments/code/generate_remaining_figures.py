import sys
import inspect
import importlib.util
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/root/autodl-tmp/q2_experiments")
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"

DATA_ROOT = Path("/root/autodl-tmp/datasetss/landslide4Sense")
DATA_ROOT_TRAINVAL = Path("/root/autodl-tmp/datasetss/landslide4Sense_trainval")
FINAL_CKPT = Path("/root/autodl-tmp/final_tune_lab/best_pool/current_best.pth")
TRAIN_CODE_DIR = Path("/root/autodl-tmp/train_l4s_qz")

for d in [
    FIG_DIR / "dataset",
    FIG_DIR / "visualizations" / "final_v8",
    FIG_DIR / "visualizations" / "final_v8" / "individual_cases",
    FIG_DIR / "area_group",
    FIG_DIR / "sample_distribution",
]:
    d.mkdir(parents=True, exist_ok=True)


def read_h5(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        for key in ["img", "mask", "test", "data", "image", "label"]:
            if key in f:
                arr = f[key][:]
                return np.asarray(arr)

        if len(keys) == 1:
            return np.asarray(f[keys[0]][:])

        # 兜底：选第一个二维或三维数组
        for key in keys:
            try:
                arr = np.asarray(f[key][:])
                if arr.ndim in [2, 3]:
                    return arr
            except Exception:
                pass

        raise RuntimeError(f"No valid array key found in {path}. Keys={keys}")


def find_mask_dir(data_root, split):
    split_dir = Path(data_root) / split

    candidates = [
        split_dir / "mask",
        split_dir / "masks",
        split_dir / "test",
        split_dir / "label",
        split_dir / "labels",
    ]

    for c in candidates:
        if c.exists() and len(list(c.glob("*.h5"))) > 0:
            return c

    # 兜底搜索
    h5_dirs = []
    for p in split_dir.rglob("*.h5"):
        if "img" not in str(p.parent).lower() and "image" not in str(p.parent).lower():
            h5_dirs.append(p.parent)

    if h5_dirs:
        return sorted(set(h5_dirs), key=lambda x: str(x))[0]

    raise FileNotFoundError(f"Cannot find mask dir for {split} under {split_dir}")


def find_img_dir(data_root, split):
    split_dir = Path(data_root) / split

    candidates = [
        split_dir / "img",
        split_dir / "image",
        split_dir / "images",
    ]

    for c in candidates:
        if c.exists() and len(list(c.glob("*.h5"))) > 0:
            return c

    for p in split_dir.rglob("*.h5"):
        if "img" in str(p.parent).lower() or "image" in str(p.parent).lower():
            return p.parent

    raise FileNotFoundError(f"Cannot find image dir for {split} under {split_dir}")


def find_mask_for_img(mask_dir, img_path):
    stem = img_path.stem
    idx = stem.split("_")[-1]

    candidates = [
        mask_dir / f"mask_{idx}.h5",
        mask_dir / f"label_{idx}.h5",
        mask_dir / f"{stem}.h5",
        mask_dir / img_path.name,
    ]

    for p in candidates:
        if p.exists():
            return p

    # 兜底：按排序位置找同序号
    masks = sorted(mask_dir.glob("*.h5"))
    imgs = sorted(img_path.parent.glob("*.h5"))
    try:
        pos = imgs.index(img_path)
        if pos < len(masks):
            return masks[pos]
    except Exception:
        pass

    raise FileNotFoundError(f"Mask not found for {img_path}")


# ==========================================================
# 1. Dataset distribution
# ==========================================================

def generate_dataset_distribution():
    out_dir = FIG_DIR / "dataset"
    rows = []
    sample_rows = []

    for split in ["TrainData", "ValidData", "TestData"]:
        try:
            mask_dir = find_mask_dir(DATA_ROOT, split)
        except Exception as e:
            print(f"[Warning] Skip split {split}: {e}")
            continue

        masks = sorted(mask_dir.glob("*.h5"))

        non_count = 0
        land_count = 0

        for p in masks:
            try:
                m = read_h5(p).astype(np.uint8)
                m = np.squeeze(m)

                non = int((m == 0).sum())
                land = int((m == 1).sum())

                non_count += non
                land_count += land

                sample_rows.append({
                    "Split": split,
                    "Mask": p.name,
                    "NonLandslide_Pixels": non,
                    "Landslide_Pixels": land,
                    "Total_Pixels": non + land,
                    "Positive_Ratio_%": land / max(non + land, 1) * 100
                })

            except Exception as e:
                print(f"[Warning] Failed reading mask {p}: {e}")

        total = non_count + land_count

        rows.append({
            "Split": split,
            "Num_Masks": len(masks),
            "NonLandslide_Pixels": non_count,
            "Landslide_Pixels": land_count,
            "Total_Pixels": total,
            "Landslide_Ratio_%": land_count / max(total, 1) * 100
        })

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(sample_rows)

    df.to_csv(out_dir / "dataset_pixel_distribution.csv", index=False)
    sdf.to_csv(out_dir / "sample_positive_ratio_distribution.csv", index=False)

    if len(df) == 0:
        raise RuntimeError("No dataset distribution rows generated.")

    # 1. Split landslide ratio
    plt.figure(figsize=(7, 5))
    x = np.arange(len(df))
    vals = df["Landslide_Ratio_%"].astype(float).values
    bars = plt.bar(x, vals)
    plt.xticks(x, df["Split"].values)
    plt.ylabel("Landslide pixel ratio (%)")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(out_dir / "landslide_pixel_ratio_by_split.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Overall pie
    total_non = df["NonLandslide_Pixels"].sum()
    total_land = df["Landslide_Pixels"].sum()

    plt.figure(figsize=(5.5, 5.5))
    plt.pie(
        [total_non, total_land],
        labels=["Non-landslide", "Landslide"],
        autopct="%1.2f%%",
        startangle=90
    )
    plt.title("Overall pixel-level class distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "overall_class_distribution_pie.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Sample positive ratio histogram
    if len(sdf) > 0:
        plt.figure(figsize=(7, 5))

        for split in sorted(sdf["Split"].unique()):
            tmp = sdf[sdf["Split"] == split]
            plt.hist(tmp["Positive_Ratio_%"], bins=30, alpha=0.45, label=split)

        plt.xlabel("Landslide pixel ratio per sample (%)")
        plt.ylabel("Number of samples")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "sample_positive_ratio_histogram.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("=" * 100)
    print("[OK] Dataset distribution generated")
    print(df.to_string(index=False))
    print(f"Saved to: {out_dir}")
    print("=" * 100)


# ==========================================================
# 2. Dynamic model loading
# ==========================================================

def import_module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_state_dict_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def try_build_model_from_class(cls):
    kwargs_list = [
        {"num_classes": 2, "img_size": 128},
        {"num_classes": 2},
        {"n_classes": 2},
        {"in_chans": 14, "num_classes": 2, "img_size": 128},
        {"in_channels": 14, "num_classes": 2, "img_size": 128},
        {"in_ch": 14, "num_classes": 2, "img_size": 128},
        {"pretrain_path": "", "num_classes": 2},
        {},
    ]

    for kwargs in kwargs_list:
        try:
            model = cls(**kwargs)
            return model, kwargs
        except Exception:
            continue

    return None, None


def build_final_model(device):
    sys.path.insert(0, str(TRAIN_CODE_DIR))

    module_paths = [
        TRAIN_CODE_DIR / "model.py",
        TRAIN_CODE_DIR / "train.py",
        TRAIN_CODE_DIR / "test.py",
    ]

    modules = []

    for p in module_paths:
        if p.exists():
            try:
                mod = import_module_from_file(p, p.stem + "_dynamic_import")
                modules.append(mod)
                print(f"[Info] Imported module: {p}")
            except Exception as e:
                print(f"[Warning] Failed importing {p}: {e}")

    if not modules:
        raise RuntimeError("No train_l4s_qz module could be imported.")

    state = load_state_dict_from_ckpt(FINAL_CKPT)

    candidates = []

    preferred_names = [
        "SwinUPerNet",
        "L4SSwinUPerNet",
        "L4SNet",
        "TGRSNet",
        "SegModel",
        "Net",
    ]

    for mod in modules:
        for name in preferred_names:
            if hasattr(mod, name):
                obj = getattr(mod, name)
                if inspect.isclass(obj) and issubclass(obj, nn.Module):
                    candidates.append((name, obj))

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, nn.Module):
                continue

            lname = name.lower()
            if any(k in lname for k in ["swin", "uper", "l4s", "tgrs"]) and (name, obj) not in candidates:
                candidates.append((name, obj))

    tried = []

    dummy = torch.randn(1, 14, 128, 128)

    best_model = None
    best_score = 10 ** 9
    best_info = None

    for name, cls in candidates:
        model, kwargs = try_build_model_from_class(cls)

        if model is None:
            continue

        try:
            msg = model.load_state_dict(state, strict=False)
            missing = len(msg.missing_keys)
            unexpected = len(msg.unexpected_keys)

            model.eval()
            with torch.no_grad():
                out = model(dummy)

            if isinstance(out, dict):
                if "out" in out:
                    out = out["out"]
                else:
                    out = list(out.values())[0]

            if not torch.is_tensor(out):
                continue

            if out.ndim != 4:
                continue

            if out.shape[1] != 2:
                continue

            score = missing + unexpected

            tried.append((name, kwargs, missing, unexpected, tuple(out.shape), score))

            if score < best_score:
                best_score = score
                best_model = model
                best_info = (name, kwargs, missing, unexpected, tuple(out.shape))

        except Exception as e:
            tried.append((name, kwargs, "failed", str(e), "", 10 ** 9))

    diag_path = FIG_DIR / "visualizations" / "final_v8" / "model_loading_diagnostics.txt"

    with open(diag_path, "w", encoding="utf-8") as f:
        for item in tried:
            f.write(str(item) + "\n")

    if best_model is None:
        raise RuntimeError(f"Could not build a compatible model. Diagnostics saved to {diag_path}")

    name, kwargs, missing, unexpected, out_shape = best_info
    print("=" * 100)
    print(f"[OK] Selected model class: {name}")
    print(f"[OK] Constructor kwargs: {kwargs}")
    print(f"[OK] Missing keys: {missing}")
    print(f"[OK] Unexpected keys: {unexpected}")
    print(f"[OK] Dummy output shape: {out_shape}")
    print("=" * 100)

    best_model = best_model.to(device)
    best_model.eval()

    return best_model


# ==========================================================
# 3. Visualization utilities
# ==========================================================

def percentile_stretch(x):
    x = x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    for c in range(x.shape[-1]):
        p2, p98 = np.percentile(x[..., c], 2), np.percentile(x[..., c], 98)
        out[..., c] = np.clip((x[..., c] - p2) / (p98 - p2 + 1e-6), 0, 1)

    return out


def make_rgb(img):
    if img.ndim == 2:
        return np.repeat(percentile_stretch(img[..., None]), 3, axis=-1)

    if img.shape[-1] >= 4:
        rgb = np.stack([img[..., 3], img[..., 2], img[..., 1]], axis=-1)
    elif img.shape[-1] >= 3:
        rgb = img[..., :3]
    else:
        rgb = np.repeat(img[..., :1], 3, axis=-1)

    return percentile_stretch(rgb)


def make_false_color(img):
    if img.ndim == 2:
        return make_rgb(img)

    if img.shape[-1] >= 8:
        fc = np.stack([img[..., 7], img[..., 3], img[..., 2]], axis=-1)
    else:
        fc = make_rgb(img)

    return percentile_stretch(fc)


def overlay_mask(rgb, mask, color=(1, 0, 0), alpha=0.35):
    out = rgb.copy()
    color_arr = np.zeros_like(rgb)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]

    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color_arr[m]

    return np.clip(out, 0, 1)


def make_error_map(gt, pred):
    err = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)

    tp = (gt == 1) & (pred == 1)
    fp = (gt == 0) & (pred == 1)
    fn = (gt == 1) & (pred == 0)

    err[tp] = [0.0, 1.0, 0.0]    # green
    err[fp] = [1.0, 0.0, 0.0]    # red
    err[fn] = [0.0, 0.2, 1.0]    # blue

    return err


def sample_metrics(gt, pred):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)

    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())

    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    pos_ratio = float((gt == 1).mean() * 100)

    if pos_ratio < 1:
        area_group = "0-1%"
    elif pos_ratio < 3:
        area_group = "1-3%"
    elif pos_ratio < 5:
        area_group = "3-5%"
    else:
        area_group = ">5%"

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Positive_Ratio_%": pos_ratio,
        "Area_Group": area_group,
    }


@torch.no_grad()
def predict(model, img, device):
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    logits = model(x)

    if isinstance(logits, dict):
        if "out" in logits:
            logits = logits["out"]
        else:
            logits = list(logits.values())[0]

    # TTA horizontal flip
    xf = torch.flip(x, dims=[3])
    logits_f = model(xf)

    if isinstance(logits_f, dict):
        if "out" in logits_f:
            logits_f = logits_f["out"]
        else:
            logits_f = list(logits_f.values())[0]

    logits_f = torch.flip(logits_f, dims=[3])
    logits = 0.5 * (logits + logits_f)

    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return pred


def plot_single_case(name, img, gt, pred, m, out_path):
    rgb = make_rgb(img)
    fc = make_false_color(img)
    gt_overlay = overlay_mask(rgb, gt, color=(0, 1, 0), alpha=0.35)
    pred_overlay = overlay_mask(rgb, pred, color=(1, 0, 0), alpha=0.35)
    err = make_error_map(gt, pred)

    fig, axes = plt.subplots(1, 6, figsize=(21, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    axes[1].imshow(fc)
    axes[1].set_title("False color")

    axes[2].imshow(gt_overlay)
    axes[2].set_title("GT overlay")

    axes[3].imshow(pred_overlay)
    axes[3].set_title("Prediction overlay")

    axes[4].imshow(err)
    axes[4].set_title("Error map\nGreen=TP Red=FP Blue=FN")

    axes[5].imshow(rgb)
    try:
        axes[5].contour(gt, levels=[0.5], linewidths=0.8)
        axes[5].contour(pred, levels=[0.5], linewidths=0.8)
    except Exception:
        pass
    axes[5].set_title(f"Boundary\nIoU={m['IoU']:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"{name} | IoU={m['IoU']:.3f}, F1={m['F1']:.3f}, Pos={m['Positive_Ratio_%']:.2f}%",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_montage(items, out_path, title):
    if len(items) == 0:
        return

    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))

    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, item in enumerate(items):
        name, img, gt, pred, m = item

        rgb = make_rgb(img)
        pred_overlay = overlay_mask(rgb, pred, color=(1, 0, 0), alpha=0.35)
        err = make_error_map(gt, pred)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Input")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground truth")

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title("Prediction")

        axes[i, 3].imshow(err)
        axes[i, 3].set_title(f"Error map | IoU={m['IoU']:.3f}")

        for j in range(4):
            axes[i, j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_final_visualizations():
    out_dir = FIG_DIR / "visualizations" / "final_v8"
    indiv_dir = out_dir / "individual_cases"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    if not FINAL_CKPT.exists():
        raise FileNotFoundError(f"Final checkpoint not found: {FINAL_CKPT}")

    img_dir = find_img_dir(DATA_ROOT_TRAINVAL, "TestData")
    mask_dir = find_mask_dir(DATA_ROOT_TRAINVAL, "TestData")

    img_paths = sorted(img_dir.glob("*.h5"))

    if len(img_paths) == 0:
        raise RuntimeError(f"No test images found in {img_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_final_model(device)

    rows = []
    cache = []

    for img_path in img_paths:
        try:
            mask_path = find_mask_for_img(mask_dir, img_path)

            img = read_h5(img_path).astype(np.float32)
            gt = read_h5(mask_path).astype(np.uint8)
            gt = np.squeeze(gt)

            if img.ndim != 3:
                print(f"[Warning] skip non-3D image: {img_path}, shape={img.shape}")
                continue

            pred = predict(model, img, device)

            m = sample_metrics(gt, pred)
            m["Sample"] = img_path.name

            rows.append(m)
            cache.append((img_path.name, img, gt, pred, m))

        except Exception as e:
            print(f"[Warning] failed sample {img_path.name}: {e}")

    if len(cache) == 0:
        raise RuntimeError("No valid visualization samples were generated.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sample_metrics.csv", index=False)

    group_df = df.groupby("Area_Group").agg({
        "IoU": "mean",
        "Precision": "mean",
        "Recall": "mean",
        "F1": "mean",
        "Sample": "count",
        "Positive_Ratio_%": "mean",
    }).reset_index().rename(columns={"Sample": "Num_Samples"})

    order = ["0-1%", "1-3%", "3-5%", ">5%"]
    group_df["Area_Group"] = pd.Categorical(group_df["Area_Group"], categories=order, ordered=True)
    group_df = group_df.sort_values("Area_Group")
    group_df.to_csv(out_dir / "area_group_metrics.csv", index=False)

    cache_sorted = sorted(cache, key=lambda x: x[4]["IoU"])

    worst = cache_sorted[:min(6, len(cache_sorted))]
    best = cache_sorted[-min(6, len(cache_sorted)):]

    mixed = worst[:3] + best[-3:]

    for name, img, gt, pred, m in mixed:
        safe = name.replace(".h5", "")
        plot_single_case(name, img, gt, pred, m, indiv_dir / f"vis_{safe}.png")

    plot_montage(worst, out_dir / "montage_failure_cases.png", "Failure cases with low sample-level IoU")
    plot_montage(best, out_dir / "montage_success_cases.png", "Successful cases with high sample-level IoU")
    plot_montage(mixed, out_dir / "montage_mixed_cases.png", "Mixed qualitative prediction and error maps")

    # per-sample IoU histogram
    sample_out = FIG_DIR / "sample_distribution"
    sample_out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(df["IoU"], bins=30)
    plt.xlabel("Sample-level Landslide IoU")
    plt.ylabel("Number of samples")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(sample_out / "per_sample_iou_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["Positive_Ratio_%"], df["IoU"], s=15, alpha=0.7)
    plt.xlabel("Landslide pixel ratio per sample (%)")
    plt.ylabel("Sample-level Landslide IoU")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(sample_out / "positive_ratio_vs_iou_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("=" * 100)
    print("[OK] Final visualizations generated")
    print(f"Saved to: {out_dir}")
    print(group_df.to_string(index=False))
    print("=" * 100)


# ==========================================================
# 4. Area-group figures
# ==========================================================

def generate_area_group_figures():
    csv_path = FIG_DIR / "visualizations" / "final_v8" / "area_group_metrics.csv"
    out_dir = FIG_DIR / "area_group"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing area group CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    order = ["0-1%", "1-3%", "3-5%", ">5%"]
    df["Area_Group"] = pd.Categorical(df["Area_Group"], categories=order, ordered=True)
    df = df.sort_values("Area_Group")

    metrics = ["IoU", "Precision", "Recall", "F1"]
    available = [m for m in metrics if m in df.columns]

    x = np.arange(len(df))
    width = 0.18

    plt.figure(figsize=(9, 5))

    for i, m in enumerate(available):
        plt.bar(
            x + (i - len(available) / 2) * width + width / 2,
            df[m].astype(float),
            width,
            label=m
        )

    plt.xticks(x, df["Area_Group"].astype(str))
    plt.xlabel("Landslide pixel ratio group")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "area_group_metrics_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    if "Num_Samples" in df.columns:
        plt.figure(figsize=(7, 5))
        bars = plt.bar(np.arange(len(df)), df["Num_Samples"].astype(float))
        plt.xticks(np.arange(len(df)), df["Area_Group"].astype(str))
        plt.xlabel("Landslide pixel ratio group")
        plt.ylabel("Number of samples")
        plt.grid(axis="y", alpha=0.3)

        for b, v in zip(bars, df["Num_Samples"].astype(float)):
            plt.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{int(v)}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        plt.savefig(out_dir / "area_group_sample_counts.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("=" * 100)
    print("[OK] Area-group figures generated")
    print(f"Saved to: {out_dir}")
    print("=" * 100)


def main():
    failed = []

    steps = [
        ("Dataset distribution", generate_dataset_distribution),
        ("Final visualizations", generate_final_visualizations),
        ("Area-group figures", generate_area_group_figures),
    ]

    for name, fn in steps:
        print("\n" + "=" * 100)
        print(f"[Running] {name}")
        print("=" * 100)

        try:
            fn()
        except Exception as e:
            failed.append(name)
            print(f"[Failed] {name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 100)
    print("Remaining figure generation finished.")

    if failed:
        print("Failed steps:")
        for x in failed:
            print(" -", x)
    else:
        print("No failed steps.")

    print("=" * 100)


if __name__ == "__main__":
    main()
