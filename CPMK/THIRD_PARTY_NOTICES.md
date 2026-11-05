# 第三方来源与许可状态

本文件记录本地可核对的声明及上游线索，不替第三方授予许可。
目前没有覆盖整个 CPKG 的统一开源许可证。正式发布前，维护者需要确认
自己新增、修改部分的权利归属，并为可以授权的部分选定许可证。

## 已有明确声明

| 本地范围 | 声明与来源 |
| --- | --- |
| `Chinese_CLIP_master/` | 保留原有 `MIT-LICENSE.txt`，其中列有 OFA-Sys 与相关贡献者的版权声明；来自 Chinese-CLIP，包含本地修改，导入版本尚未记录 |
| `Chinese_CLIP_master/cn_clip/clip/bert_tokenizer.py` | 文件头明确声明 Apache-2.0，保留 Google AI Language Team Authors 版权 |
| `Chinese_CLIP_master/cn_clip/clip/configuration_bert.py`、`modeling_bert.py` | 文件头明确声明 Apache-2.0，保留 Google、Hugging Face、NVIDIA 版权 |
| `Chinese_CLIP_master/cn_clip/clip/utils.py` | 文件注明修改自 OpenAI CLIP；该上游采用 MIT，另保留其许可证正文 |

`pkuseg==0.0.25` 作为依赖单独安装，其发行包元数据将许可证分类标为
`Other/Proprietary License`。应独立核对该依赖及其分词模型的使用、分发条款，
不能将其视为继承本项目的许可证。上游为 <https://github.com/lancopku/pkuseg-python>。

补充的第三方许可证正文位于 `licenses/`。原有版权头和许可证不应在代码整理时删除，
也不应仅因当前上游的年份不同而覆盖本地历史声明。

官方来源：

- Chinese-CLIP：<https://github.com/OFA-Sys/Chinese-CLIP>
- OpenAI CLIP：<https://github.com/openai/CLIP>
- Apache-2.0 正文：<https://www.apache.org/licenses/LICENSE-2.0>

## 需要维护者确认的来源

`uer/` 目录没有随附整体 LICENSE 或 NOTICE。原项目的 PKG-Bert 名称、
历史路径和保留的优化器说明提供了来源线索，但不足以确定全部文件的导入版本和许可范围。

- 原项目的 `uer/model_builder.py`、`uer/models/model.py` 包含 PKG-Bert 名称，
  本次精简未将这两个旧框架文件纳入发布目录；删除文件不改变保留代码的来源。
  PKG-Bert 上游致谢 K-BERT，K-BERT 上游说明基于 UER。
- 官方 UER-py 采用 Apache-2.0，但不能据此直接认定整个本地 `uer/` 均已获得同样授权。
- 核查的 PKG-Bert 和 K-BERT 默认分支中未找到完整的许可证文件；
  K-BERT README 中的 MIT 徽章不能替代对本地来源和完整条款的核对。
- `uer/utils/optimizers.py` 明确注明实现来自 Hugging Face 的旧
  `pytorch-pretrained-BERT`。其上游历史实现有 Apache-2.0 声明，
  本地实际导入版本仍需确认。
- `uer/encoders/clip_bert.py` 从原项目当前使用的文本模型精简而来，含有与 CLIP 实现相近的代码；
  尚未逐文件确定复制来源和修改范围，不能将整个文件声明为全新原创代码。

上游核查入口：

- UER-py：<https://github.com/dbiir/UER-py>
- PKG-Bert：<https://github.com/liyuqing1/PKG-Bert>
- K-BERT：<https://github.com/autoliuweijie/K-BERT>
- 旧版优化器：<https://github.com/huggingface/transformers/tree/v0.6.2/pytorch_pretrained_bert>

核查日期：2026-11-05。PKG-Bert 当时核查的提交为
`6a5b3049ed42d7d06bb4cb69a3274e5826247b72`，
K-BERT 为 `da9358edb3b3f59e3ad6b5aab6af6df624b881ab`。
这些提交仅记录核查位置，不表示本地代码源自该提交。

## 数据与模型

图片、现代译文、知识释义、预训练权重和训练检查点应分别说明来源与分发依据。
古诗文本的存在不代表整套图片或译文数据可以随代码一起发布。
本次整理没有重新授权这些材料。仓库首页保留原有数据下载入口。

发布前应补齐：

1. 本地 `uer/` 实际取得的仓库与版本，以及自行新增或修改文件的范围。
2. PKG-Bert 衍生部分的完整许可依据；必要时取得原作者明确授权。
3. 原创部分采用的许可证，以及代码、图片、知识数据和权重各自的发布范围。
