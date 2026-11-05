# CPMK 代码

**面向古诗的多模态表征学习与图文检索。**

本目录提供 CPMK 仓库中的 CPKG 模型代码。CPKG 利用意象释义、意象图片和现代译文，
为古诗文本编码器提供训练信号。
项目基于 BERT 与 Chinese-CLIP，提供模型训练、检查点恢复、
诗句检索图片及图片检索诗句的评估流程。

> 知识图谱文件位于同级 [KG/](../KG/)，其余公开数据入口见
> [仓库首页](../README.md)。训练集、验证集和检索图片位于 [data/](data/)；
> 模型权重与训练所需的意象图片仍需另行准备。数据格式见 [数据说明](docs/data.md)，
> 许可状态见 [第三方来源与许可说明](THIRD_PARTY_NOTICES.md)。

## 方法概述

训练时，从原诗中匹配双字意象，取得对应释义和图片，使用冻结的
Chinese-CLIP 编码图片、释义和译文，再训练 CPKG 文本编码器进行对齐。

| 训练目标 | CPKG 表征 | Chinese-CLIP 目标特征 |
| --- | --- | --- |
| 意象与图片对齐 | 意象所在字符的聚合表征 | 意象图片特征 |
| 意象与释义对齐 | 意象所在字符的聚合表征 | 意象释义特征 |
| 诗句与译文对齐 | 诗句的句级表征 | 现代译文特征 |

当前实现将三项对比损失相加，保留每条诗句随机抽样五次、
每个意象选择首个释义的策略。检索时，以原诗和图片的归一化特征计算相似度。

## 环境安装

建议使用 Python 3.11 和独立虚拟环境。已验证的主要环境如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.14 |
| PyTorch / torchvision | 2.2.2 / 0.17.2 |
| NumPy | 1.26.4 |
| pkuseg | 0.0.25 |
| CUDA 运行时 | 11.8，本地 GPU 验证使用 |

下载或克隆仓库后，进入仓库内的代码目录。以下命令适用于 Windows PowerShell：

```powershell
cd CPMK
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 Cython==3.2.4
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt
.\.venv\Scripts\python.exe main.py --help
```

`pkuseg` 在该 Python 版本上需要源码编译，Windows 环境需安装 Visual Studio
的“使用 C++ 的桌面开发”工具。请先安装 NumPy 和 Cython，再安装项目依赖。
训练需要分词器；检索评估不调用分词器。

程序默认自动选择 CPU 或 CUDA，也可以通过 `--device cpu` 或 `--device cuda`
指定设备。需要特定 CUDA 版本时，应先安装相互匹配的 PyTorch 与 torchvision。

## 数据与权重

仓库包含训练、验证、检索及丰子恺图文数据，图片和 `data/kg/` 中的大型 JSON
通过 Git LFS 管理。已安装 Git LFS 的环境可在仓库根目录执行：

```powershell
git lfs install
git lfs pull
```

请使用 `git clone` 获取仓库，再执行上述命令；未拉取 LFS 文件时，本地图片和大型
JSON 可能只是指针文本。数据约 3.96 GiB，目录用途见 [数据说明](docs/data.md)。
模型权重与训练所需的意象图片仍需另行准备；`.gitignore` 排除 `outputs/`
及模型权重文件。

| 使用模式 | 需要的资源 |
| --- | --- |
| 训练 | 训练集、验证集、意象知识、意象图片、Chinese-CLIP 权重 |
| CPKG 评估 | 检索数据及图片、Chinese-CLIP 权重、CPKG 检查点 |
| 基线评估 | 检索数据及图片、Chinese-CLIP 权重 |

默认文件布局，`KG/` 与代码目录 `CPMK/` 同级：

```text
仓库根目录/
  KG/
    author_info.jsonl
    imagery_text_image_align.json # 默认意象知识文件
  CPMK/
    main.py
    data/
      tran_val/
        train.json
        val.json
      kg/
        PKG_Chinese.json
        KG_Imagery_with_tag.json
        imagery_text_image_align.json
        images/                   # 意象图片，需另行准备
      PI_retrieval_2/
        results_500_V1_c2e.json
        images/                   # 已包含，使用 Git LFS 下载
      PI_retrieval_1/
      PI_retrieval_banana/
      fzk/
    models/
      clip_cn_rn50.pt              # 默认 Chinese-CLIP 权重
    outputs/
      CPKG_checkpoint-1           # 完成第一轮训练后生成
```

训练集和验证集使用 UTF-8 JSON，例如：

```json
[
  {
    "1": {
      "诗句": "明月照山河",
      "译文": "明亮的月光照耀山河"
    }
  }
]
```

意象图片默认按 `图片根目录/tag.png` 查找，也可以在知识节点的 `image`
字段中指定相对路径。检索图片通过检索记录的 `image` 字段指定；
省略时采用“原诗前五个字符 + `.png`”的文件名。
完整示例与约束见 [数据说明](docs/data.md)。

默认 `--kg_node` 指向同级 `KG/imagery_text_image_align.json`。
若使用其他知识文件，可显式传入 `--kg_node`。
知识中的 `align`、`image_index` 字段不会自动决定图片路径；
请准备 `tag.png` 图片，或在节点中填写与实际图片对应的 `image`。

Chinese-CLIP 权重放入 `--model_dir` 指定目录，缺失时加载器会下载所选型号。
下载先写入 `.part` 临时文件，成功后才替换为正式缓存；有长度信息时会检查下载字节数。
如果已有旧版留下的残缺 `.pt` 缓存，请先删除该文件再重新运行。
CPKG 检查点通过本项目训练生成，或使用与模型配置匹配的已有检查点。
除同级知识文件外，默认路径均相对于本代码目录定位；
手工传入的相对路径相对于当前工作目录。下文命令均在 `CPMK/` 代码目录执行。

## 模型训练

准备好训练资源后，先使用小样本验证流程：

```powershell
.\.venv\Scripts\python.exe main.py --mode train `
  --image_root data/kg/images `
  --max_samples 50 `
  --epochs_num 1 `
  --eval_interval 1
```

上述命令分别限制训练集和验证集的原诗数量，完成后生成
`outputs/CPKG_checkpoint-1`。正式训练时去掉 `--max_samples`：

```powershell
.\.venv\Scripts\python.exe main.py --mode train --image_root data/kg/images --epochs_num 100
```

默认模型为 RN50，批大小为 16，序列长度为 52，其中正文最多 50 个字符。
只有截断范围内的双字意象参与训练；没有可用意象的诗句会被跳过并统计。
选中意象的图片缺失时，程序会报告对应路径。

目标特征预计算后保存在 CPU 内存中，训练时按批传入设备。
完整数据集仍可能占用较多系统内存，建议先用小样本检查资源占用。

### 恢复训练

每轮训练会保存权重、优化器状态、轮数、模型型号和序列长度。继续训练示例：

```powershell
.\.venv\Scripts\python.exe main.py --mode train `
  --image_root data/kg/images `
  --resume outputs/CPKG_checkpoint-1 `
  --epochs_num 10
```

`--epochs_num` 表示目标总轮数，而非追加轮数。恢复训练时应保持数据、
随机种子和模型配置一致。若要接着前面的小样本实验运行，
还应添加相同的 `--max_samples 50`。
旧检查点缺少优化器状态时，程序会告警并重建优化器。
改变 `--batch_size` 会同时改变验证批大小和对比损失的基准，
因此恢复训练时会重置最佳验证损失和早停计数；旧检查点未记录批大小时也会重置。
批大小相同时保留原有早停状态。
恢复训练不承诺与未中断运行逐位一致。

## 图文检索

准备好检索数据及图片后，使用训练得到的检查点评估 CPKG：

```powershell
.\.venv\Scripts\python.exe main.py --mode evaluate `
  --checkpoint outputs/CPKG_checkpoint-1 `
  --vi_model RN50
```

评估 Chinese-CLIP 基线：

```powershell
.\.venv\Scripts\python.exe main.py --mode baseline --vi_model RN50
```

两种模式均以原诗为文本查询，输出以下指标：

- 诗句检索图片：mAP@K、Recall@K。
- 图片检索诗句：mAP@K、Recall@K。

`evaluate` 使用 CPKG 训练中的句级投影，`baseline` 使用 Chinese-CLIP 原有文本编码器。
比较时应使用相同的检索数据、CLIP 型号、序列长度和 K。
默认 K 为 20，样本数不足时自动缩小并打印实际值。

当前指标要求诗句和图片一一对应；缺图或重复图片路径会直接报错。
设置 `--max_samples N` 时，仅检查和评估按 JSON 顺序选中的前 N 条记录，
不要求未选中的记录准备图片。
一对多检索需要另外准备相关性标注并调整指标实现。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | 必填 | `train`、`evaluate` 或 `baseline` |
| `--vi_model` | `RN50` | CLIP 型号，训练与 CPKG 评估需保持一致 |
| `--device` | `auto` | `auto`、`cpu` 或 `cuda` |
| `--kg_node` | 同级 `KG/imagery_text_image_align.json` | 意象知识文件，按仓库位置定位 |
| `--batch_size` | `16` | 每批样本数，训练时至少为 2 |
| `--seq_length` | `52` | 包含 CLS 和 SEP 的序列长度 |
| `--learning_rate` | `2e-5` | 训练学习率 |
| `--epochs_num` | `100` | 训练目标总轮数 |
| `--max_samples` | 不限制 | 每个数据集最多使用的原诗数量 |
| `--eval_interval` | `3` | 训练期间的验证间隔轮数 |
| `--early_stop` | `5` | 连续无改善的验证次数 |
| `--top_k` | `20` | 检索指标的 K |
| `--seed` | `7` | 随机种子 |

数据路径、权重目录和日志参数可通过 `main.py --help` 查看。

## 测试

运行基础回归测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用微型模型和临时文件，不下载预训练权重。
仅验证数据处理时，无需安装 PyTorch：

```powershell
python -m unittest discover -s tests -p test_poetry_data.py -v
```

本地已有 `models/clip_cn_rn50.pt` 时，可验证完整流程：

```powershell
.\.venv\Scripts\python.exe tests/smoke_local.py --device cuda
```

该检查使用临时诗句和合成图片，依次验证训练、保存、改变批大小后恢复训练、
早停状态重置、CPKG 评估及基线评估，并包含一条未选中的缺图记录来检查样本上限。
结束后自动清理临时检查点。
没有 CUDA 时改用 `--device cpu`。合成数据检查不用于评价模型效果。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `main.py` | 命令行入口、训练、检查点与检索评估 |
| `uer/utils/poetry_data.py` | 数据读取、字符编码、意象坐标和图片配对 |
| `uer/kg/kg.py` | 意象知识索引与 pkuseg 分词 |
| `uer/encoders/clip_bert.py` | 古诗文本模型和三项对齐损失 |
| `uer/utils/optimizers.py`、`seed.py` | 优化器与随机种子 |
| `Chinese_CLIP_master/` | 含本地修改的 Chinese-CLIP 运行核心、词表和配置 |
| `tests/` | 回归测试与完整流程检查 |
| `docs/data.md` | 数据格式与示例 |
| `licenses/` | 补充的第三方许可证正文 |

发布目录只保留当前训练和评估依赖，历史模型版本、备份及无关实验脚本未纳入。
文本模型去除了文件名中的版本后缀，保留原有权重参数名称与顺序。
本地 CLIP 目录包含主流程依赖的修改，不能直接用同名安装包替换。

## 验证与复现

2026-11-05 在精简后的发布目录中验证：35 项单元测试通过，23 个 Python 文件语法检查通过。
新增测试覆盖下载中断与短文件、两种权重键名前缀、小样本图片检查和早停状态恢复。
清理阶段的模型对照确认：
微型模型与清理前实现的输出、损失及梯度完全一致。
已有 CPKG 检查点在 CPU 上完成小样本检索，
真实 RN50 权重在 CUDA 上完成训练、恢复和两种检索评估。
详细环境和范围见 [验证记录](docs/release-plan.md#验证记录)。

目前未验证完整实验指标及其他 CLIP 型号。当前评估以原诗为查询；
历史脚本以译文运行的基线结果不能直接比较。
意象位置与填充掩码的修复也会改变训练行为，旧权重可加载并不代表恢复了历史实验结果。

## 致谢与许可

本项目使用 Chinese-CLIP，并保留 UER、PKG-Bert 等相关实现的来源线索。
具体路径、已有声明和待确认事项见 [第三方来源与许可说明](THIRD_PARTY_NOTICES.md)。

**项目整体许可证尚未确定。** 已有第三方代码继续适用其原有条款；
图片、译文、知识数据和模型权重的使用及分发范围需要分别确认。
