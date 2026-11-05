"""用微型真实 BERT 验证句级投影、填充掩码和反向传播。"""

from types import SimpleNamespace
import unittest

import torch

from uer.encoders.clip_bert import BERT


def make_model():
    """构造不下载权重的微型模型，返回可在 CPU 上快速运行的 BERT。"""
    torch.manual_seed(7)
    model = BERT(
        embed_dim=8, vocab_size=32,
        text_attention_probs_dropout_prob=0.0, text_hidden_act="gelu",
        text_hidden_dropout_prob=0.0, text_hidden_size=16,
        text_initializer_range=0.02, text_intermediate_size=32,
        text_max_position_embeddings=32, text_num_attention_heads=2,
        text_num_hidden_layers=1, text_type_vocab_size=2,
        tokenizer=SimpleNamespace(vocab={"[PAD]": 0}),
    )
    torch.nn.init.normal_(model.text_projection, std=0.02)
    return model.eval()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.src = torch.tensor([[2, 4, 5, 3, 0, 0], [2, 6, 7, 3, 0, 0]])
        self.spans = torch.tensor([[1, 3], [1, 3]])
        self.features = torch.eye(2, 8)

    def test_inference_uses_the_trained_sentence_head(self):
        with torch.no_grad():
            self.model.sim_linear_1.weight.zero_()
            self.model.sim_linear_1.bias.zero_()
            self.model.sim_layer_norm_1.bias.fill_(2.0)
            encoded = self.model.encode_text(self.src)
        torch.testing.assert_close(encoded, torch.full((2, 8), 2.0))

    def test_padding_does_not_change_training_losses(self):
        extended = torch.cat([self.src, torch.zeros(2, 3, dtype=torch.long)], dim=1)
        with torch.no_grad():
            original = self.model(self.src, self.spans, self.features, self.features, self.features)
            padded = self.model(extended, self.spans, self.features, self.features, self.features)
        for left, right in zip(original, padded):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)

    def test_all_three_losses_support_backward_with_mixed_target_precision(self):
        losses = self.model(
            self.src, self.spans, self.features.half(), self.features, self.features,
        )
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(torch.isfinite(loss).item() for loss in losses))
        sum(losses).backward()
        for parameter in (self.model.linear_2.weight, self.model.sim_linear_1.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
