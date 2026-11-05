"""不依赖深度学习框架的诗句、知识图片与检索数据处理。"""

import json
from itertools import islice
from pathlib import Path


def load_poems(path):
    """读取 UTF-8 诗句 JSON，返回按编号索引的诗句与译文；不修改原文件。"""
    with open(path, encoding="utf-8") as reader:
        data = json.load(reader)
    records = data if isinstance(data, list) else [data]
    poems = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path} 中的记录必须是编号到诗句的字典")
        for poem_id, item in record.items():
            if poem_id in poems:
                raise ValueError(f"{path} 中存在重复编号：{poem_id}")
            if not isinstance(item, dict):
                raise ValueError(f"{path} 中编号 {poem_id} 的内容必须是字典")
            # 同时兼容随项目保存的数据和早期脚本使用的字段名称。
            poem = item.get("诗句", item.get("诗歌"))
            translation = item.get("译文", item.get("翻译/译文"))
            if not all(isinstance(value, str) and value.strip() for value in (poem, translation)):
                raise ValueError(f"{path} 中编号 {poem_id} 缺少有效的诗句或译文")
            poems[poem_id] = {"诗句": poem, "译文": translation}
    if not poems:
        raise ValueError(f"{path} 中没有诗句")
    return poems


def find_knowledge_spans(sentence, tokens, lookup_table, seq_length):
    """返回可见双字意象及其 token 区间；区间含 CLS 偏移，右端不包含在内。"""
    spans = []
    cursor = 0
    for word in tokens:
        # 分词器可能省略空白，必须在原文定位，不能只累加分词长度。
        start = sentence.find(word, cursor)
        if start < 0:
            raise ValueError(f"分词结果无法在原句定位：{word}")
        end = start + len(word)
        cursor = end
        if word in lookup_table and len(word) == 2 and end <= seq_length - 2:
            spans.append((word, (start + 1, end + 1)))
    return spans


def encode_poem(sentence, vocab, seq_length):
    """按字符编码原诗，返回固定长度列表，依次包含 CLS、正文、SEP 和填充。"""
    if seq_length < 4:
        raise ValueError("序列长度至少为 4，才能容纳 CLS、双字意象和 SEP")
    tokens = [vocab["[CLS]"]]
    tokens.extend(vocab.get(char, vocab["[UNK]"]) for char in sentence[:seq_length - 2])
    tokens.append(vocab["[SEP]"])
    return tokens + [vocab["[PAD]"]] * (seq_length - len(tokens))


def resolve_image_path(image_root, node):
    """根据节点 image 或 tag 定位已有图片，缺失时报告实际路径。"""
    filename = node.get("image")
    if not filename:
        tag = node.get("tag")
        if not tag:
            raise ValueError("知识节点需要 image 路径或 tag")
        filename = f"{tag}.png"
    path = Path(image_root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"缺少意象图片：{path}")
    return path


def load_retrieval_pairs(path, image_root, max_samples=None):
    """读取检索 JSON 的前 max_samples 条并检查图片，返回有序配对；None 表示全部。"""
    with open(path, encoding="utf-8") as reader:
        data = json.load(reader)
    if not isinstance(data, dict) or not data:
        raise ValueError("检索数据必须是非空的原诗到信息的字典")
    pairs = []
    seen = set()
    for poem, item in islice(data.items(), max_samples):
        if not poem.strip() or not isinstance(item, dict):
            raise ValueError("每条检索记录需要非空原诗和信息字典")
        # 兼容现有检索集的前五字文件名；重名时要求提供显式 image 字段。
        image = Path(image_root) / item.get("image", f"{poem[:5]}.png")
        if not image.is_file():
            raise FileNotFoundError(f"原诗 {poem!r} 缺少图片：{image}")
        if image.resolve() in seen:
            raise ValueError(f"检索图片重复：{image}；请使用 image 字段指定不同图片")
        seen.add(image.resolve())
        pairs.append((poem, image))
    return pairs
