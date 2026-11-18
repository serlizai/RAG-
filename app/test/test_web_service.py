"""联网服务的离线行为：工具校验、结果格式、失败和清理。"""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.query_process.agent.nodes import node_web_search_mcp as legacy_web
from app.query_process.services import web_service
from app.utils import task_utils


def raw_pages(pages):
    return SimpleNamespace(
        isError=False,
        content=[SimpleNamespace(text=json.dumps({"pages": pages}, ensure_ascii=False))],
    )


class FakeServer:
    def __init__(self, *, tools=None, response=None, connect_error=None, cleanup_error=None):
        self.tools = tools if tools is not None else [SimpleNamespace(name="bailian_web_search")]
        self.response = response if response is not None else raw_pages([])
        self.connect_error = connect_error
        self.cleanup_error = cleanup_error
        self.cleaned = False
        self.calls = []

    async def connect(self):
        if self.connect_error:
            raise self.connect_error

    async def list_tools(self):
        return self.tools

    async def call_tool(self, *, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.response

    async def cleanup(self):
        self.cleaned = True
        if self.cleanup_error:
            raise self.cleanup_error


class WebServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_checks_tool_and_cleans_up_then_builds_evidence(self):
        server = FakeServer(response=raw_pages([{
            "title": "官方文档", "url": "https://example.org/doc", "snippet": "开机说明",
        }]))
        raw = await web_service.call_mcp_raw(
            " 开机 ", server_factory=lambda **kwargs: server,
            base_url="https://example.org/mcp", api_key="test-key",
        )
        self.assertTrue(server.cleaned)
        self.assertEqual(server.calls, [("bailian_web_search", {"query": "开机", "count": 3})])
        result = web_service.parse_mcp_result(raw)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.items[0]["snippet"], "开机说明")
        self.assertEqual(result.evidence[0].url, "https://example.org/doc")

    async def test_missing_remote_tool_returns_distinct_error_and_cleans_up(self):
        server = FakeServer(tools=[SimpleNamespace(name="other")])
        with self.assertRaises(web_service.WebServiceError) as raised:
            await web_service.call_mcp_raw(
                "问题", server_factory=lambda **kwargs: server,
                base_url="https://example.org/mcp", api_key="test-key",
            )
        self.assertEqual(raised.exception.code, "TOOL_UNAVAILABLE")
        self.assertEqual(server.calls, [])
        self.assertTrue(server.cleaned)

    async def test_cleanup_failure_does_not_hide_original_failure(self):
        original = RuntimeError("连接失败")
        server = FakeServer(connect_error=original, cleanup_error=RuntimeError("清理失败"))
        with self.assertRaises(RuntimeError) as raised:
            await web_service.call_mcp_raw(
                "问题", server_factory=lambda **kwargs: server,
                base_url="https://example.org/mcp", api_key="test-key",
            )
        self.assertIs(raised.exception, original)
        self.assertTrue(server.cleaned)

    async def test_empty_invalid_and_partial_results_are_distinct(self):
        self.assertEqual(web_service.parse_mcp_result(raw_pages([])).status, "empty")
        self.assertEqual(web_service.parse_mcp_result(
            SimpleNamespace(isError=True, content=[])
        ).error_code, "REMOTE_ERROR")
        partial = web_service.parse_mcp_result(raw_pages([
            {"title": "有效", "url": "https://example.org", "snippet": "文本"},
            {"title": "缺少摘要"},
        ]))
        self.assertEqual(partial.status, "partial")
        self.assertEqual(partial.candidate_count, 2)
        self.assertEqual(len(partial.evidence), 1)

    async def test_service_converts_connection_and_timeout_to_errors(self):
        with patch.object(web_service, "call_mcp_raw", new=AsyncMock(
            side_effect=web_service.WebServiceError("TOOL_UNAVAILABLE")
        )):
            result = await web_service.search_web("问题")
        self.assertEqual((result.status, result.error_code), ("error", "TOOL_UNAVAILABLE"))
        with patch.object(web_service, "call_mcp_raw", new=AsyncMock(side_effect=TimeoutError())):
            result = await web_service.search_web("问题")
        self.assertEqual((result.error_code, result.retryable), ("TIMEOUT", True))


class LegacyWebNodeTests(unittest.TestCase):
    def test_legacy_node_preserves_output_and_finishes_progress(self):
        session_id = "web-service-test"
        self.addCleanup(task_utils.clear_task, session_id)
        response = raw_pages([{"title": "文档", "url": "https://example.org", "snippet": "内容"}])
        with patch.object(legacy_web, "mcp_call_streamable", new=AsyncMock(return_value=response)):
            result = legacy_web.node_web_search_mcp({
                "session_id": session_id, "rewritten_query": "问题", "is_stream": False,
            })
        self.assertEqual(result["web_search_docs"], [
            {"title": "文档", "url": "https://example.org", "snippet": "内容"},
        ])
        self.assertIn("node_web_search_mcp", task_utils._tasks_done_list[session_id])


if __name__ == "__main__":
    unittest.main()
