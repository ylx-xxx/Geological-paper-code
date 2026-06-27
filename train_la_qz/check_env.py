# train_la_qz/check_env.py
import os
import sys
import platform
import subprocess
from pathlib import Path

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[FAILED] {e}"

def count_files(path, suffix=None):
    p = Path(path)
    if not p.exists():
        return "路径不存在"
    if suffix is None:
        return len([x for x in p.iterdir() if x.is_file()])
    return len(list(p.glob(f"*{suffix}")))

def check_dir(path):
    p = Path(path)
    return "存在" if p.exists() else "不存在"

def main():
    print("=" * 80)
    print("服务器训练环境检查")
    print("=" * 80)

    print("\n[1] 基础环境")
    print("Python:", sys.version.replace("\n", " "))
    print("Python路径:", sys.executable)
    print("系统:", platform.platform())
    print("当前工作目录:", os.getcwd())

    print("\n[2] GPU / CUDA")
    print("nvidia-smi:")
    print(run_cmd("nvidia-smi"))

    print("\n[3] PyTorch / CUDA / timm")
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
        print("CUDA设备数量:", torch.cuda.device_count())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {prop.name}")
                print(f"  显存: {prop.total_memory / 1024**3:.2f} GB")
                print(f"  compute capability: {prop.major}.{prop.minor}")
    except Exception as e:
        print("torch导入失败:", repr(e))

    try:
        import timm
        print("timm:", timm.__version__)
    except Exception as e:
        print("timm导入失败:", repr(e))

    try:
        import cv2
        print("opencv-python:", cv2.__version__)
    except Exception as e:
        print("cv2导入失败:", repr(e))

    try:
        import PIL
        print("Pillow:", PIL.__version__)
    except Exception as e:
        print("Pillow导入失败:", repr(e))

    try:
        import numpy as np
        print("numpy:", np.__version__)
    except Exception as e:
        print("numpy导入失败:", repr(e))

    print("\n[4] 项目目录检查")
    root = Path.cwd()

    candidate_roots = [
        root,
        root.parent,
        Path("/root/autodl-tmp"),
        Path("/root/autodl-tmp/地质论文"),
        Path("/root/autodl-tmp/geology"),
        Path("/root/autodl-tmp/project"),
    ]

    dataset_root = None
    for base in candidate_roots:
        p = base / "datasetss"
        if p.exists():
            dataset_root = p
            break

    if dataset_root is None:
        print("未自动找到 datasetss 文件夹")
        print("请确认你的目录是否类似：/root/autodl-tmp/xxx/datasetss")
    else:
        print("找到 datasetss:", dataset_root)

    pre_model_candidates = [
        root / "pre_model",
        root.parent / "pre_model",
        Path("/root/autodl-tmp/pre_model"),
        Path("/root/autodl-tmp/地质论文/pre_model"),
    ]

    print("\n[5] pre_model 检查")
    found_pre = False
    for p in pre_model_candidates:
        if p.exists():
            found_pre = True
            print("找到 pre_model:", p)
            for f in p.iterdir():
                print("  -", f.name)
    if not found_pre:
        print("未找到 pre_model 文件夹")
        print("说明：没有本地 Swin 预训练权重也可以训练，后续代码可设置 pretrained=False 或自动下载。")

    print("\n[6] LoveDA 数据检查")
    if dataset_root:
        loveda = dataset_root / "loveda"
        paths = {
            "Train Rural images": loveda / "Train" / "Train" / "Rural" / "images_png",
            "Train Rural masks": loveda / "Train" / "Train" / "Rural" / "masks_png",
            "Train Urban images": loveda / "Train" / "Train" / "Urban" / "images_png",
            "Train Urban masks": loveda / "Train" / "Train" / "Urban" / "masks_png",
            "Val Rural images": loveda / "Val" / "Val" / "Rural" / "images_png",
            "Val Rural masks": loveda / "Val" / "Val" / "Rural" / "masks_png",
            "Val Urban images": loveda / "Val" / "Val" / "Urban" / "images_png",
            "Val Urban masks": loveda / "Val" / "Val" / "Urban" / "masks_png",
        }

        for name, path in paths.items():
            print(f"{name}: {check_dir(path)} | png数量: {count_files(path, '.png')}")

    print("\n[7] Landslide4Sense 数据检查")
    if dataset_root:
        l4s = dataset_root / "landslide4Sense"
        paths = {
            "Train img": l4s / "TrainData" / "img",
            "Train mask": l4s / "TrainData" / "mask",
            "Valid img": l4s / "ValidData" / "img",
            "Valid mask": l4s / "ValidData" / "mask",
            "Test img": l4s / "TestData" / "img",
            "Test mask/test": l4s / "TestData" / "test",
        }

        for name, path in paths.items():
            print(f"{name}: {check_dir(path)} | h5数量: {count_files(path, '.h5')}")

    print("\n[8] H5读取测试")
    try:
        import h5py
        print("h5py:", h5py.__version__)

        if dataset_root:
            sample_img = dataset_root / "landslide4Sense" / "TrainData" / "img" / "image_1.h5"
            sample_mask = dataset_root / "landslide4Sense" / "TrainData" / "mask" / "mask_1.h5"

            if sample_img.exists():
                with h5py.File(sample_img, "r") as f:
                    print("image_1.h5 keys:", list(f.keys()))
                    for k in f.keys():
                        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

            if sample_mask.exists():
                with h5py.File(sample_mask, "r") as f:
                    print("mask_1.h5 keys:", list(f.keys()))
                    for k in f.keys():
                        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")

    except Exception as e:
        print("h5py读取失败:", repr(e))

    print("\n[9] 推荐训练参数初判")
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if mem >= 30:
                print("推荐 batch_size: 48 或 64")
                print("推荐 num_workers: 12~16")
            elif mem >= 24:
                print("推荐 batch_size: 32 或 48")
                print("推荐 num_workers: 8~12")
            elif mem >= 16:
                print("推荐 batch_size: 16 或 24")
                print("推荐 num_workers: 8")
            else:
                print("推荐 batch_size: 8 或 12")
                print("推荐 num_workers: 4~8")

            print("推荐开启: AMP=True, channels_last=True, cudnn.benchmark=True")
    except Exception:
        pass

    print("\n检查完成。请把终端完整输出复制给我。")
    print("=" * 80)

if __name__ == "__main__":
    main()