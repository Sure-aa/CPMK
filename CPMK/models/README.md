# 模型权重缓存

默认 Chinese-CLIP 权重文件为 `clip_cn_rn50.pt`，放在本目录中。
也可以通过主入口的 `--model_dir` 指定其他缓存目录。
运行时缺少所选型号的权重，加载器会尝试下载。

本目录不随源码分发模型权重。CPKG 训练检查点默认保存到 `outputs/`，
评估时通过 `--checkpoint` 指定。
