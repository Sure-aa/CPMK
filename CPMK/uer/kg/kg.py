"""读取训练诗句和意象知识，建立匹配索引及分词器。"""

import json

import pkuseg

from . import config
from uer.utils.poetry_data import load_poems


class ModalKnowledgeGraph:
    """从本地 JSON 构造知识索引，图像特征由主入口统一提取。"""

    def __init__(self, node_file, train_path, dev_path):
        """读取知识、训练和验证文件，初始化含意象词表的 pkuseg 分词器。"""
        self.train_data = load_poems(train_path)
        self.val_data = load_poems(dev_path)
        self.kg_nodes, self.lookup_table = self.create_lookup_table(node_file)
        segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(
            model_name="default", postag=False, user_dict=segment_vocab,
        )

    def create_lookup_table(self, node_file):
        """读取 UTF-8 知识文件，返回原始节点和意象到释义列表的映射。"""
        with open(node_file, encoding="utf-8") as file:
            words = json.load(file)
        lookup_table = {
            name: [node["Meaning"] for node in nodes]
            for name, nodes in words.items()
        }
        return words, lookup_table
