"""使用本地 RN50 权重和临时合成数据验证完整流程，不代表模型效果。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


def main():
    """创建临时数据并运行真实训练、恢复和两种评估；退出时清理临时检查点。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model_dir", default=str(ROOT / "models"))
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "clip_cn_rn50.pt").is_file():
        raise FileNotFoundError("本测试需要本地 clip_cn_rn50.pt，不会下载权重")

    with tempfile.TemporaryDirectory(prefix="cpkg-smoke-") as directory:
        root = Path(directory)
        poems = [
            {"1": {"诗句": "明月照山河", "译文": "明亮的月光照耀山河"}},
            {"2": {"诗句": "春风入故园", "译文": "春天的风吹入故乡的园子"}},
        ]
        knowledge = {
            "明月": [{"Meaning": "明亮的月亮", "tag": "moon"}],
            "春风": [{"Meaning": "春天的风", "tag": "spring"}],
        }
        retrieval = {
            "明月照山河": {"image": "moon.png"},
            "春风入故园": {"image": "spring.png"},
            "清风过远山": {"image": "not_prepared.png"},
        }
        for name, data in (("train.json", poems), ("kg.json", knowledge), ("retrieval.json", retrieval)):
            (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        Image.new("RGB", (64, 64), (40, 70, 130)).save(root / "moon.png")
        Image.new("RGB", (64, 64), (90, 160, 50)).save(root / "spring.png")
        common = [
            sys.executable, str(ROOT / "main.py"), "--device", args.device,
            "--model_dir", str(model_dir), "--seq_length", "12",
        ]
        train = [
            "--mode", "train", "--train_path", str(root / "train.json"),
            "--val_path", str(root / "train.json"), "--kg_node", str(root / "kg.json"),
            "--image_root", str(root), "--output_model_path", str(root / "checkpoint"),
            "--eval_interval", "1",
        ]
        subprocess.run(common + train + ["--batch_size", "2", "--epochs_num", "1"], cwd=ROOT, check=True)
        first = root / "checkpoint-1"
        subprocess.run(
            common + train + ["--batch_size", "4", "--epochs_num", "2", "--resume", str(first)],
            cwd=ROOT, check=True,
        )
        resumed = root / "checkpoint-2"
        checkpoint = torch.load(resumed, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 2, "恢复训练后轮数错误"
        assert checkpoint["optimizer_state_dict"]["state"], "检查点缺少优化器状态"
        assert checkpoint["bad_epochs"] == 0, "改变批大小后应重新建立验证损失基准"
        del checkpoint
        evaluation = [
            "--retrieval_path", str(root / "retrieval.json"), "--retrieval_images", str(root),
            "--batch_size", "2", "--max_samples", "2",
        ]
        subprocess.run(
            common + evaluation + ["--mode", "evaluate", "--checkpoint", str(resumed)],
            cwd=ROOT, check=True,
        )
        subprocess.run(common + evaluation + ["--mode", "baseline"], cwd=ROOT, check=True)
        print("真实权重的训练、恢复、CPKG 评估和基线评估全部通过；合成数据不用于效果评价。")


if __name__ == "__main__":
    main()
