# Chinese-CLIP 运行依赖

本目录保留当前训练和检索入口需要的 Chinese-CLIP 代码、词表及模型配置。
来源为 [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)，
包含原项目已有的本地修改，实际导入版本尚未记录。

- `cn_clip/clip/`：模型、分词器、权重加载和图像预处理。
- `cn_clip/clip/model_configs/`：各型号的文本与视觉配置。
- `cn_clip/clip/vocab.txt`：文本分词词表。
- `MIT-LICENSE.txt`：保留的原始许可证及版权声明。

本次发布未包含上游训练、部署、评估工具、示例素材及本地实验脚本。
`cn_clip/clip/` 保留仍被加载器导入的模型变体。
本次针对 `utils.py` 修复下载中断缓存和检查点 `module.` 前缀兼容问题；
模型实现、词表、配置和原始许可证保留。
本项目依赖本地的 `get_text_info` 等接口，不能直接用同名安装包替换。

使用方式见 [代码说明](../README.md)，来源及许可范围见
[第三方说明](../THIRD_PARTY_NOTICES.md)。
