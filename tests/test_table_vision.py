"""Tests for table_vision (AI 视觉识别扫描页真实表格结构)."""
from __future__ import annotations

import unittest
from unittest import mock

from translate_app import table_vision


class ParseJsonTest(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(
            table_vision._parse_json('{"rows": [0.1, 0.2], "cols": [0.3, 0.4]}'),
            {"rows": [0.1, 0.2], "cols": [0.3, 0.4]},
        )

    def test_parses_fenced_json(self):
        self.assertEqual(
            table_vision._parse_json('```json\n{"rows": [1, 2]}\n```'),
            {"rows": [1, 2]},
        )

    def test_returns_none_on_garbage(self):
        self.assertIsNone(table_vision._parse_json("I cannot see any table"))
        self.assertIsNone(table_vision._parse_json(""))

    def test_parses_json_with_trailing_commas(self):
        # 常见 VLM 错误：数组/对象尾逗号 → 容错解析。
        self.assertEqual(
            table_vision._parse_json('{"rows": [0.1, 0.2,], "cols": [0.3, 0.4,],}'),
            {"rows": [0.1, 0.2], "cols": [0.3, 0.4]},
        )


class TablesFromGridTest(unittest.TestCase):
    def test_builds_rows_and_col_edges(self):
        tables = table_vision.tables_from_grid([0, 10, 20], [0, 50, 100])
        self.assertEqual(len(tables), 1)
        tb = tables[0]
        self.assertEqual(len(tb["rows"]), 2)          # 2 row gaps
        self.assertEqual(len(tb["rows"][0]), 2)       # 2 col gaps
        self.assertEqual(tb["col_edges"], [0, 50, 100])
        self.assertEqual(tb["bbox"].x1, 100)
        self.assertEqual(tb["bbox"].y1, 20)
        # A cell rect spans one row gap × one col gap.
        self.assertEqual((tb["rows"][0][0].x0, tb["rows"][0][0].x1), (0, 50))


class MakeLlmTableStructureTest(unittest.TestCase):
    def test_returns_none_for_non_vision_model(self):
        model = mock.Mock()
        model.vision = False
        self.assertIsNone(
            table_vision.make_llm_table_structure(model, page_width=100, page_height=100))

    def test_detector_scales_normalized_to_points(self):
        model = mock.Mock()
        model.vision = True
        model.api_key = None
        model.model = "m"
        model.client_kwargs.return_value = {"base_url": "http://x/v1", "api_key": "k"}
        model.request_params.return_value = {}

        fake_client = mock.Mock()
        fake_resp = mock.Mock()
        fake_resp.choices = [mock.Mock(message=mock.Mock(
            content='{"rows": [0.0, 0.5, 1.0], "cols": [0.0, 1.0], '
                    '"merged": [], "header_rows": [0], "header_cols": [0], '
                    '"align": [{"col": 1, "dir": "right"}]}'))]
        fake_client.chat.completions.create.return_value = fake_resp
        make = table_vision.make_llm_table_structure(
            model, page_width=200, page_height=400, client=fake_client, n_samples=1)
        styles = make(b"png")
        self.assertEqual(len(styles), 1)
        style = styles[0]
        self.assertEqual(style["rows_pts"], [0.0, 200.0, 400.0])
        self.assertEqual(style["cols_pts"], [0.0, 200.0])
        self.assertEqual(style["align"], [{"col": 1, "dir": "right"}])
        self.assertEqual(style["header_rows"], [0])

    def test_detector_returns_every_sample(self):
        # Layer-② 修正：返回**全部**候选（不再众数平均），由确定性打分选优。
        model = mock.Mock()
        model.vision = True
        model.model = "m"
        model.client_kwargs.return_value = {"base_url": "http://x/v1", "api_key": "k"}
        model.request_params.return_value = {}
        fake_client = mock.Mock()
        contents = [
            '{"rows": [0, 1], "cols": [0, 0.5, 1], "align": []}',
            '{"rows": [0, 1], "cols": [0, 0.3, 0.6, 1], "align": []}',
        ]
        fake_client.chat.completions.create.side_effect = [
            mock.Mock(choices=[mock.Mock(message=mock.Mock(content=c))]) for c in contents
        ]
        make = table_vision.make_llm_table_structure(
            model, page_width=100, page_height=100, client=fake_client, n_samples=2)
        styles = make(b"png")
        self.assertEqual(len(styles), 2)
        self.assertEqual([len(s["cols_pts"]) for s in styles], [3, 4])

    def test_single_retries_once_on_non_json(self):
        # 非 JSON 时用「纠正提示」重试一次，而不是直接丢弃该样本。
        model = mock.Mock()
        model.vision = True
        model.model = "m"
        model.client_kwargs.return_value = {"base_url": "http://x/v1", "api_key": "k"}
        model.request_params.return_value = {}
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = [
            mock.Mock(choices=[mock.Mock(message=mock.Mock(content="not json"))]),
            mock.Mock(choices=[mock.Mock(message=mock.Mock(
                content='{"rows": [0, 1], "cols": [0, 1], "align": []}'))]),
        ]
        make = table_vision.make_llm_table_structure(
            model, page_width=100, page_height=100, client=fake_client,
            log=lambda m: None, n_samples=1)
        styles = make(b"png")
        self.assertEqual(len(styles), 1)
        self.assertEqual(styles[0]["rows_pts"], [0.0, 100.0])
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)

    def test_detector_falls_back_on_parse_failure(self):
        model = mock.Mock()
        model.vision = True
        model.model = "m"
        model.client_kwargs.return_value = {"base_url": "http://x/v1", "api_key": "k"}
        model.request_params.return_value = {}
        fake_client = mock.Mock()
        fake_resp = mock.Mock()
        fake_resp.choices = [mock.Mock(message=mock.Mock(content="not json"))]
        fake_client.chat.completions.create.return_value = fake_resp
        make = table_vision.make_llm_table_structure(
            model, page_width=200, page_height=400, client=fake_client, log=lambda m: None)
        self.assertEqual(make(b"png"), [])


if __name__ == "__main__":
    unittest.main()
