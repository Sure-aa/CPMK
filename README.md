Source code and data for the paper Large-Scale Multimodal Knowledge Graph about Classical Chinese Poetry: Fine-grained Method and Comprehensive Evaluation which is accepted by the Findings of ACL 2026.

Because of the data size, you can find the rest of the data at this link: https://huggingface.co/datasets/sadadsss/CPMK

## 代码与数据

- [CPMK/](CPMK/README.md)：古诗多模态表征学习代码，包含训练、恢复训练和图文检索评估。
- [KG/](KG/)：知识图谱数据，保留原有目录与文件。
- [数据格式说明](CPMK/docs/data.md)：训练诗句、意象知识及图片配对规则。

代码目录与数据目录同级。请先进入 `CPMK/`，再按其中的 README 安装依赖和运行程序：

```powershell
cd CPMK
```

默认意象知识文件为 `KG/imagery_text_image_align.json`，由程序按仓库位置定位。
训练集、图片与模型权重需按代码说明另行准备；知识节点中的 `image_index`
不会自动转换为图片路径。
