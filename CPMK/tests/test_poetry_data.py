"""验证公开数据格式、意象坐标和检索样本的对应关系。"""

import json
from pathlib import Path
import tempfile
import unittest

from uer.utils.poetry_data import (
    encode_poem,
    find_knowledge_spans,
    load_poems,
    load_retrieval_pairs,
    resolve_image_path,
)


class PoetryDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_json(self, value):
        path = self.root / "data.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_existing_list_format(self):
        path = self.write_json([{"42": {"诗句": "明月照山河", "译文": "月光照耀山河"}}])
        self.assertEqual(load_poems(path), {"42": {"诗句": "明月照山河", "译文": "月光照耀山河"}})

    def test_accepts_previous_dictionary_fields(self):
        path = self.write_json({"42": {"诗歌": "明月", "翻译/译文": "月亮"}})
        self.assertEqual(load_poems(path), {"42": {"诗句": "明月", "译文": "月亮"}})

    def test_rejects_duplicate_ids_instead_of_overwriting(self):
        path = self.write_json([
            {"42": {"诗句": "明月", "译文": "月亮"}},
            {"42": {"诗句": "春风", "译文": "春天的风"}},
        ])
        with self.assertRaises(ValueError):
            load_poems(path)

    def test_rejects_missing_translation(self):
        with self.assertRaises(ValueError):
            load_poems(self.write_json([{"42": {"诗句": "明月"}}]))

    def test_positions_account_for_skipped_words_and_cls(self):
        spans = find_knowledge_spans("春明月照", ["春", "明月", "照"], {"春", "明月"}, 8)
        self.assertEqual(spans, [("明月", (2, 4))])

    def test_positions_account_for_whitespace(self):
        spans = find_knowledge_spans("春 明月", ["春", "明月"], {"明月"}, 8)
        self.assertEqual(spans, [("明月", (3, 5))])

    def test_drops_partial_or_fully_truncated_words(self):
        spans = find_knowledge_spans("春风明月", ["春风", "明月"], {"春风", "明月"}, 5)
        self.assertEqual(spans, [("春风", (1, 3))])

    def test_encoding_places_sep_before_padding(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "明": 4, "月": 5}
        self.assertEqual(encode_poem("明月", vocab, 6), [2, 4, 5, 3, 0, 0])
        self.assertEqual(encode_poem("明月未知", vocab, 5), [2, 4, 5, 1, 3])

    def test_resolves_images_from_node_tag_or_explicit_path(self):
        tagged = self.root / "明月_0.png"
        tagged.touch()
        self.assertEqual(resolve_image_path(self.root, {"tag": "明月_0"}), tagged)
        nested = self.root / "nested"
        nested.mkdir()
        explicit = nested / "moon.png"
        explicit.touch()
        self.assertEqual(
            resolve_image_path(self.root, {"tag": "明月_0", "image": "nested/moon.png"}),
            explicit,
        )

    def test_missing_knowledge_image_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            resolve_image_path(self.root, {"tag": "不存在"})

    def test_retrieval_keeps_original_poem_and_image_order(self):
        (self.root / "明月照山河.png").touch()
        (self.root / "spring.png").touch()
        path = self.write_json({
            "明月照山河": {"translation": "月光照耀"},
            "春风入故园": {"translation": "春风吹来", "image": "spring.png"},
        })
        self.assertEqual(load_retrieval_pairs(path, self.root), [
            ("明月照山河", self.root / "明月照山河.png"),
            ("春风入故园", self.root / "spring.png"),
        ])

    def test_missing_retrieval_image_cannot_shift_the_pairs(self):
        (self.root / "春风.png").touch()
        path = self.write_json({"明月": {}, "春风": {}})
        with self.assertRaises(FileNotFoundError):
            load_retrieval_pairs(path, self.root)

    def test_rejects_colliding_image_names(self):
        (self.root / "明月照山河.png").touch()
        path = self.write_json({"明月照山河一": {}, "明月照山河二": {}})
        with self.assertRaises(ValueError):
            load_retrieval_pairs(path, self.root)

    def test_sample_limit_skips_images_outside_the_selected_records(self):
        (self.root / "moon.png").touch()
        path = self.write_json({
            "明月照山河": {"image": "moon.png"},
            "春风入故园": {"image": "missing.png"},
        })
        self.assertEqual(
            load_retrieval_pairs(path, self.root, max_samples=1),
            [("明月照山河", self.root / "moon.png")],
        )
        with self.assertRaises(FileNotFoundError):
            load_retrieval_pairs(path, self.root)


if __name__ == "__main__":
    unittest.main()
