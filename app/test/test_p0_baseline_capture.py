"""P0 基线采集在离线状态下的输入与外部搜索边界。"""

from contextlib import redirect_stderr
import asyncio
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts import capture_legacy_baseline as baseline


class BaselineCaptureTests(unittest.TestCase):
    def test_case_ids_are_unique_and_missing_products_are_not_sent_as_placeholders(self):
        cases = json.loads(baseline.CASES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

        rendered = baseline.render_cases(cases, None, None)
        self.assertTrue(any(case["status"] == "pending" for case in rendered))
        for case in rendered:
            if case["requires_products"]:
                self.assertEqual(case["status"], "skipped")
                self.assertEqual(case["turns"], [])

    def test_product_capture_needs_explicit_external_search_opt_in(self):
        args = [
            "capture_legacy_baseline", "--run-live", "--product-a", "测试商品",
            "--case-id", "product_operation", "--output", str(Path("/private/tmp/unused-p0-baseline.json")),
        ]
        with patch.object(sys, "argv", args), redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                baseline.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--allow-external-search", errors.getvalue())

    def test_local_only_replaces_external_call_and_restores_it(self):
        web_module = importlib.import_module(
            "app.query_process.agent.nodes.node_web_search_mcp"
        )
        original = web_module.mcp_call_streamable
        with baseline.external_search_context(local_only=True):
            self.assertIsNot(web_module.mcp_call_streamable, original)
            self.assertIsNone(asyncio.run(web_module.mcp_call_streamable("测试商品")))
        self.assertIs(web_module.mcp_call_streamable, original)


if __name__ == "__main__":
    unittest.main()
