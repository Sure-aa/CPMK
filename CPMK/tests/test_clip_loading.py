"""验证下载缓存完整性和两种 CLIP 检查点键名，不访问网络或加载大模型。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from Chinese_CLIP_master.cn_clip.clip import utils
from Chinese_CLIP_master.cn_clip.clip.model import BERT


class DownloadResponse(io.BytesIO):
    """提供内存字节流和响应长度，代替外部下载连接。"""

    def __init__(self, data, total=None):
        super().__init__(data)
        self.headers = {} if total is None else {"Content-Length": str(total)}

    def info(self):
        return self.headers


class InterruptedResponse(DownloadResponse):
    """返回首段内容后抛出连接错误，模拟下载中断。"""

    def read(self, size=-1):
        if self.tell():
            raise ConnectionError("模拟下载中断")
        return super().read(size)


class DownloadTests(unittest.TestCase):
    def test_interrupted_download_can_be_retried_without_reusing_partial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pt"
            url = "https://example.invalid/model.pt"
            with patch("urllib.request.urlopen", return_value=InterruptedResponse(b"partial", 100)):
                with self.assertRaises(ConnectionError):
                    utils._download(url, directory)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            with patch("urllib.request.urlopen", return_value=DownloadResponse(b"complete", 8)):
                result = utils._download(url, directory)
            self.assertEqual(Path(result).read_bytes(), b"complete")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_short_response_is_not_saved_as_a_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", return_value=DownloadResponse(b"short", 100)):
                with self.assertRaises(RuntimeError):
                    utils._download("https://example.invalid/model.pt", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_response_without_a_length_header_can_be_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", return_value=DownloadResponse(b"complete")):
                path = utils._download("https://example.invalid/model.pt", directory)
            self.assertEqual(Path(path).read_bytes(), b"complete")

    def test_existing_cache_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pt"
            target.write_bytes(b"cached")
            with patch("urllib.request.urlopen", side_effect=AssertionError("已有缓存不应联网")):
                result = utils._download("https://example.invalid/model.pt", directory)
            self.assertEqual(Path(result).read_bytes(), b"cached")


class TextCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"
        self.config = dict(
            embed_dim=8, vocab_size=32,
            text_attention_probs_dropout_prob=0.0, text_hidden_act="gelu",
            text_hidden_dropout_prob=0.0, text_hidden_size=16,
            text_initializer_range=0.02, text_intermediate_size=32,
            text_max_position_embeddings=32, text_num_attention_heads=2,
            text_num_hidden_layers=1, text_type_vocab_size=2,
            tokenizer=SimpleNamespace(vocab={"[PAD]": 0}),
        )
        model = BERT(**self.config)
        torch.nn.init.normal_(model.text_projection, std=0.02)
        self.state = model.state_dict()

    def save_checkpoint(self, prefix):
        """将带指定键名前缀的微型权重写入临时文件，包含应当被滤除的视觉参数。"""
        state = {prefix + name: value for name, value in self.state.items()}
        state[prefix + "visual.example"] = torch.zeros(1)
        torch.save({"state_dict": state}, self.path)

    def test_text_info_accepts_both_checkpoint_key_formats(self):
        for prefix in ("module.", ""):
            with self.subTest(prefix=prefix):
                self.save_checkpoint(prefix)
                with redirect_stdout(io.StringIO()):
                    _, state = utils.get_text_info(
                        str(self.path), device="cpu", vision_model_name="RN50",
                        text_model_name="RBT3-chinese", input_resolution=224,
                    )
                self.assertEqual(set(state), set(self.state))
                for name, value in self.state.items():
                    torch.testing.assert_close(state[name], value)

    def test_text_encoder_accepts_both_checkpoint_key_formats(self):
        for prefix in ("module.", ""):
            with self.subTest(prefix=prefix):
                self.save_checkpoint(prefix)
                # 保留真实构造、转换和加载逻辑，只把大型号换成同类型的微型模型。
                with patch.object(utils, "BERT", side_effect=lambda **kwargs: BERT(**self.config)):
                    with redirect_stdout(io.StringIO()):
                        model = utils.get_text_encoder(
                            str(self.path), device="cpu", vision_model_name="RN50",
                            text_model_name="RBT3-chinese", input_resolution=224,
                        )
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, self.state[name].to(value.dtype))


if __name__ == "__main__":
    unittest.main()
