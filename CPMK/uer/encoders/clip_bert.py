"""古诗文本编码器，保留历史检查点的参数名称及三项对齐损失。"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from Chinese_CLIP_master.cn_clip.clip import _tokenizer
from Chinese_CLIP_master.cn_clip.clip.configuration_bert import BertConfig
from Chinese_CLIP_master.cn_clip.clip.modeling_bert import BertModel


class LayerNorm(nn.LayerNorm):
    """用 float32 归一化输入，再恢复输入精度。"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class BERT(nn.Module):
    """根据 BERT 配置构造古诗编码器，输出句向量或三项训练损失。"""

    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float,
                 text_hidden_act: str,
                 text_hidden_dropout_prob: float,
                 text_hidden_size: int,
                 text_initializer_range: float,
                 text_intermediate_size: int,
                 text_max_position_embeddings: int,
                 text_num_attention_heads: int,
                 text_num_hidden_layers: int,
                 text_type_vocab_size: int,
                 tokenizer=_tokenizer,
                 use_flash_attention: bool = False):
        super().__init__()

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            max_position_embeddings=text_max_position_embeddings,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
            use_flash_attention=use_flash_attention,
        )
        self.bert = BertModel(self.bert_config)

        # 旧检查点包含这些参数，即使当前损失未使用，也保留名称与注册顺序。
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        self.tokenizer = tokenizer
        self.linear_1 = nn.Linear(self.bert_config.hidden_size, self.bert_config.vocab_size)
        self.layer_norm_1 = LayerNorm(self.bert_config.vocab_size)

        self.sim_linear_1 = nn.Linear(self.bert_config.hidden_size, embed_dim)
        self.sim_layer_norm_1 = LayerNorm(embed_dim)
        self.linear_2 = nn.Linear(self.bert_config.hidden_size, embed_dim)
        self.layer_norm_2 = LayerNorm(embed_dim)

    def encode_text(self, text):
        """将字符 token 编码为句向量，使用训练中的句级投影，不改变模型状态。"""
        output = self._encode_tokens(text)
        return self._project_sentence(output[:, 0, :])

    def _encode_tokens(self, text):
        # 训练和推理都屏蔽 PAD，避免填充长度改变有效字符的表示。
        attention_mask = text.ne(self.tokenizer.vocab['[PAD]']).float()
        return self.bert(text, attention_mask=attention_mask)[0]

    def _project_sentence(self, output_cls):
        return self.sim_layer_norm_1(self.gelu(self.sim_linear_1(output_cls)))

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, src, label, image_feature, translate, key_word_feature):
        """输入诗句、意象区间及三类目标特征，返回三项标量损失，保留反向传播图。"""
        output = self._encode_tokens(src)
        spans = label.tolist() if isinstance(label, torch.Tensor) else label
        extracted_vectors = torch.stack([
            output[index, start:end].mean(dim=0)
            for index, (start, end) in enumerate(spans)
        ])
        extracted_vectors = self.gelu(self.linear_2(extracted_vectors))
        extracted_vectors = self.layer_norm_2(extracted_vectors)

        # 固定用 float32 计算损失，兼容 CPU 及半精度的 CLIP 目标特征。
        labels = torch.arange(src.shape[0], device=src.device)
        similarity_matrix1 = extracted_vectors.float() @ image_feature.float().t()
        similarity_matrix2 = extracted_vectors.float() @ key_word_feature.float().t()
        output_cls = self._project_sentence(output[:, 0, :])
        similarity_matrix3 = output_cls.float() @ translate.float().t()
        loss1 = F.cross_entropy(similarity_matrix1 / 0.1, labels)
        loss2 = F.cross_entropy(similarity_matrix2 / 0.1, labels)
        loss3 = F.cross_entropy(similarity_matrix3 / 0.1, labels)
        return loss1, loss2, loss3
