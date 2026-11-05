"""验证训练统计、检索指标和命令行入口的实际行为。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from main import evaluate, get_args, get_metric, load_checkpoint, restore_optimizer, run_epoch, save_checkpoint
from test_model import make_model
from uer.utils.optimizers import BertAdam


class TrainingTests(unittest.TestCase):
    def test_cli_selects_real_training_options(self):
        args = get_args(["--mode", "train", "--vi_model", "RN50", "--max_samples", "2"])
        self.assertEqual(args.mode, "train")
        self.assertEqual(args.max_samples, 2)
        self.assertIsNone(args.resume)

    def test_retrieval_metric_uses_diagonal_relevance(self):
        scores = torch.tensor([[9., 1., 2.], [8., 7., 1.], [9., 8., 7.]])
        mean_ap, recall = get_metric(scores, K=2)
        self.assertAlmostEqual(mean_ap, 0.5)
        self.assertAlmostEqual(recall, 2 / 3)

    def test_top_k_larger_than_dataset_is_supported(self):
        mean_ap, recall = get_metric(torch.eye(2), K=20)
        self.assertEqual((mean_ap, recall), (1.0, 1.0))

    def test_non_square_retrieval_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            get_metric(torch.ones(2, 3), K=1)

    def test_epoch_statistics_do_not_leak_between_calls(self):
        model = make_model()
        src = torch.tensor([[2, 4, 5, 3, 0], [2, 6, 7, 3, 0]])
        spans = torch.tensor([[1, 3], [1, 3]])
        features = torch.eye(2, 8)
        loader = DataLoader(TensorDataset(src, spans, features, features, features), batch_size=2)
        with torch.no_grad():
            expected = [loss.item() for loss in model(src, spans, features, features, features)]
        first = run_epoch(model, loader, torch.device("cpu"))
        second = run_epoch(model, loader, torch.device("cpu"))
        for actual, repeated, loss in zip(first, second, expected):
            self.assertAlmostEqual(actual, loss, places=5)
            self.assertAlmostEqual(actual, repeated, places=5)

    def test_train_epoch_updates_parameters(self):
        model = make_model()
        src = torch.tensor([[2, 4, 5, 3], [2, 6, 7, 3]])
        spans = torch.tensor([[1, 3], [1, 3]])
        features = torch.eye(2, 8)
        loader = DataLoader(TensorDataset(src, spans, features, features, features), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.sim_linear_1.weight.detach().clone()
        losses = run_epoch(model, loader, torch.device("cpu"), optimizer=optimizer)
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses))
        self.assertFalse(torch.equal(before, model.sim_linear_1.weight))

    def test_resume_after_batch_size_change_keeps_learning_rate_positive(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        original = BertAdam([parameter], lr=2e-5, t_total=100)
        original.state[parameter] = {
            "step": 40, "next_m": torch.zeros(1), "next_v": torch.zeros(1),
        }
        optimizer = BertAdam([parameter], lr=2e-5, t_total=10)
        restore_optimizer(optimizer, {"optimizer_state_dict": original.state_dict()}, remaining_steps=10)
        self.assertEqual(optimizer.state[parameter]["step"], 40)
        self.assertEqual(optimizer.param_groups[0]["t_total"], 50)
        self.assertGreater(optimizer.get_lr()[0], 0)

    def test_checkpoint_round_trip_preserves_predictions_and_progress(self):
        model = make_model()
        optimizer = BertAdam(model.parameters(), lr=2e-5, t_total=10)
        args = get_args(["--mode", "train"])
        with tempfile.TemporaryDirectory() as directory:
            args.output_model_path = str(Path(directory) / "nested" / "checkpoint")
            path = save_checkpoint(model, optimizer, args, epoch=3, best_loss=0.5, bad_epochs=1)
            restored = make_model()
            with torch.no_grad():
                restored.sim_linear_1.weight.zero_()
            checkpoint = load_checkpoint(restored, path, args)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["bad_epochs"], 1)
            tokens = torch.tensor([[2, 4, 5, 3]])
            torch.testing.assert_close(model.encode_text(tokens), restored.encode_text(tokens))

    def test_checkpoint_rejects_mismatched_sequence_length(self):
        model = make_model()
        args = get_args(["--mode", "train"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            torch.save({"model_state_dict": model.state_dict(), "seq_length": 8}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(model, path, args)

    def test_training_resets_early_stopping_when_validation_batch_size_changes_or_is_unknown(self):
        model = make_model()
        args = get_args(["--mode", "train", "--batch_size", "4"])
        for saved_batch_size in (2, None):
            with self.subTest(saved_batch_size=saved_batch_size), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint"
                saved_args = {} if saved_batch_size is None else {"batch_size": saved_batch_size}
                torch.save({
                    "model_state_dict": model.state_dict(), "epoch": 3,
                    "args": saved_args, "best_loss": 0.5, "bad_epochs": 4,
                }, path)
                restored = load_checkpoint(model, path, args)
                self.assertEqual(restored["epoch"], 3)
                self.assertEqual(restored["best_loss"], float("inf"))
                self.assertEqual(restored["bad_epochs"], 0)
                original = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(original["bad_epochs"], 4)

    def test_checkpoint_keeps_early_stopping_with_the_same_validation_batch_size(self):
        model = make_model()
        args = get_args(["--mode", "train", "--batch_size", "4"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": 3,
                "args": {"batch_size": 4}, "best_loss": 0.5, "bad_epochs": 4,
            }, path)
            restored = load_checkpoint(model, path, args)
            self.assertEqual(restored["best_loss"], 0.5)
            self.assertEqual(restored["bad_epochs"], 4)

    def test_evaluation_applies_the_sample_limit_before_checking_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "moon.png")
            path = root / "retrieval.json"
            path.write_text(json.dumps({
                "明月照山河": {"image": "moon.png"},
                "春风入故园": {"image": "missing.png"},
            }, ensure_ascii=False), encoding="utf-8")
            args = get_args([
                "--mode", "baseline", "--max_samples", "1",
                "--retrieval_path", str(path), "--retrieval_images", str(root),
            ])
            # 用固定特征替代外部预训练模型，仍运行真实的图片读取和指标计算。
            model = SimpleNamespace(
                encode_image=lambda images: torch.ones(len(images), 8),
                encode_text=lambda tokens: torch.ones(len(tokens), 8),
            )
            output = io.StringIO()
            with patch("main.get_modal", return_value=(model, lambda image: torch.zeros(3, 8, 8), None)):
                with redirect_stdout(output):
                    evaluate(args, torch.device("cpu"))
            self.assertIn("mAP@1=1.000000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
