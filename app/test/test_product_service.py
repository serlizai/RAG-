"""P1 商品解析与旧节点适配的离线测试。"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.query_process.agent.nodes import node_item_name_confirm as legacy_node
from app.query_process.services import product_service as product
from app.query_process.services.schemas import ProductResolutionResult


class ProductServiceTests(unittest.TestCase):
    def test_extract_uses_passed_history_and_parses_json_fence(self):
        client = Mock()
        client.invoke.return_value = SimpleNamespace(
            content='```json\n{"item_names": ["设备 A"], "rewritten_query": "设备 A 怎么开机？"}\n```'
        )
        with (patch.object(product, "load_prompt", return_value="提取提示") as prompt,
              patch.object(product, "get_llm_client", return_value=client) as get_client):
            result = product.extract_product_info(
                "它怎么开机？", [{"role": "user", "text": "我用设备 A"}]
            )
        self.assertEqual(result["item_names"], ["设备 A"])
        self.assertEqual(result["rewritten_query"], "设备 A 怎么开机？")
        prompt.assert_called_once_with(
            "rewritten_query_and_itemnames",
            history_text="聊天角色：user: 聊天内容我用设备 A\n",
            query="它怎么开机？",
        )
        get_client.assert_called_once_with(json_mode=True)
        self.assertEqual(client.invoke.call_args.args[0][1].content, "提取提示")

    def test_search_keeps_each_extracted_name_with_its_matches(self):
        responses = [
            [[{"entity": {"item_name": "设备 A 标准型"}, "distance": 0.91}]],
            [[{"entity": {"item_name": "设备 B"}, "distance": 0.72}]],
        ]
        with (patch.object(product, "get_milvus_client", return_value=object()),
              patch.object(product, "generate_embeddings", return_value={
                  "dense": [[1], [2]], "sparse": [{1: 1}, {2: 1}],
              }) as embeddings,
              patch.object(product, "create_hybrid_search_requests", side_effect=["req-a", "req-b"]),
              patch.object(product, "hybrid_search", side_effect=responses) as search):
            results = product.search_matching_products(["设备 A", "设备 B"])
        embeddings.assert_called_once_with(["设备 A", "设备 B"])
        self.assertEqual([r["extracted_name"] for r in results], ["设备 A", "设备 B"])
        self.assertEqual(results[0]["matches"], [{"item_name": "设备 A 标准型", "score": 0.91}])
        self.assertEqual(results[1]["matches"], [{"item_name": "设备 B", "score": 0.72}])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args.kwargs["ranker_weights"], (0.7, 0.3))
        self.assertTrue(search.call_args.kwargs["norm_score"])

    def test_alignment_prefers_exact_high_match_and_limits_options(self):
        query_results = [
            {"extracted_name": "设备 A", "matches": [
                {"item_name": "设备 A Pro", "score": 0.95},
                {"item_name": "设备 A", "score": 0.86},
            ]},
            {"extracted_name": "设备 B", "matches": [
                {"item_name": "设备 B 旧款", "score": 0.62},
                {"item_name": "设备 B", "score": 0.79},
                {"item_name": "设备 B 新款", "score": 0.68},
            ]},
            {"extracted_name": "设备 A", "matches": [
                {"item_name": "设备 A", "score": 0.91},
            ]},
        ]
        original = copy.deepcopy(query_results)
        self.assertEqual(product.align_product_names(query_results), {
            "confirmed_item_names": ["设备 A"],
            "options_item_names": ["设备 B", "设备 B 新款"],
        })
        self.assertEqual(query_results, original)

    def test_milvus_failure_is_not_treated_as_zero_matches(self):
        with patch.object(product, "get_milvus_client", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "客户端不可用"):
                product.search_matching_products(["设备 A"])
        with (patch.object(product, "get_milvus_client", return_value=object()),
              patch.object(product, "generate_embeddings", return_value={
                  "dense": [[1]], "sparse": [{1: 1}],
              }),
              patch.object(product, "create_hybrid_search_requests", return_value=["request"]),
              patch.object(product, "hybrid_search", return_value=None)):
            with self.assertRaisesRegex(RuntimeError, "查询失败"):
                product.search_matching_products(["设备 A"])

    def test_resolve_distinguishes_confirmed_clarification_empty_and_error(self):
        with (patch.object(product, "extract_product_info", return_value={
                  "item_names": ["设备 A"], "rewritten_query": "设备 A 说明书",
              }),
              patch.object(product, "search_matching_products", return_value=[]),
              patch.object(product, "align_product_names", return_value={
                  "confirmed_item_names": ["设备 A"], "options_item_names": [],
              })):
            resolved = product.resolve_product("它的说明书", [])
        self.assertEqual(resolved.status, "ok")
        self.assertEqual(resolved.confirmed_item_names, ["设备 A"])
        self.assertEqual(resolved.rewritten_query, "设备 A 说明书")

        with (patch.object(product, "extract_product_info", return_value={
                  "item_names": ["设备 B"], "rewritten_query": "设备 B 说明书",
              }),
              patch.object(product, "search_matching_products", return_value=[]),
              patch.object(product, "align_product_names", return_value={
                  "confirmed_item_names": [], "options_item_names": ["设备 B 1", "设备 B 2"],
              })):
            resolved = product.resolve_product("设备 B", [])
        self.assertEqual(resolved.status, "needs_clarification")
        self.assertEqual(resolved.option_item_names, ["设备 B 1", "设备 B 2"])

        with (patch.object(product, "extract_product_info", return_value={
                  "item_names": [], "rewritten_query": "你好",
              }),
              patch.object(product, "search_matching_products") as search):
            resolved = product.resolve_product("你好", [])
        self.assertEqual(resolved.status, "empty")
        search.assert_not_called()

        with (patch.object(product, "extract_product_info", side_effect=TimeoutError),
              patch.object(product.logger, "exception")):
            resolved = product.resolve_product("设备 A", [])
        self.assertEqual(resolved.status, "error")
        self.assertEqual(resolved.error_code, "PRODUCT_EXTRACT_FAILED")
        self.assertTrue(resolved.retryable)

        with (patch.object(product, "extract_product_info", return_value={
                  "item_names": ["设备 A"], "rewritten_query": "设备 A",
              }),
              patch.object(product, "search_matching_products", side_effect=RuntimeError),
              patch.object(product.logger, "exception")):
            resolved = product.resolve_product("设备 A", [])
        self.assertEqual(resolved.status, "error")
        self.assertEqual(resolved.error_code, "PRODUCT_MATCH_FAILED")


class LegacyNodeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "session_id": "offline-product-test", "original_query": "它怎么开机？",
            "is_stream": False, "answer": "上轮旧答案", "item_names": [],
        }
        self.history = [{"role": "user", "text": "我用设备 A"}]

    def test_confirmed_result_updates_legacy_state_and_message(self):
        result = ProductResolutionResult(
            status="ok", rewritten_query="设备 A 怎么开机？",
            confirmed_item_names=["设备 A"],
        )
        with (patch.object(legacy_node, "get_recent_messages", return_value=self.history),
              patch.object(legacy_node, "save_chat_message", side_effect=["message-1", "message-1"]) as save,
              patch.object(legacy_node, "resolve_product", return_value=result) as resolve,
              patch.object(legacy_node, "add_running_task"),
              patch.object(legacy_node, "add_done_task")):
            output = legacy_node.node_item_name_confirm(self.state)
        resolve.assert_called_once_with("它怎么开机？", self.history)
        self.assertEqual(output["item_names"], ["设备 A"])
        self.assertEqual(output["rewritten_query"], "设备 A 怎么开机？")
        self.assertEqual(output["history"], self.history)
        self.assertNotIn("answer", output)
        self.assertEqual(save.call_count, 2)
        self.assertEqual(save.call_args.kwargs["message_id"], "message-1")
        self.assertEqual(save.call_args.kwargs["item_names"], ["设备 A"])

    def test_clarification_preserves_old_answer_text_and_stops_search(self):
        result = ProductResolutionResult(
            status="needs_clarification", rewritten_query="设备 B 说明书",
            option_item_names=["设备 B 1", "设备 B 2"],
        )
        with (patch.object(legacy_node, "get_recent_messages", return_value=self.history),
              patch.object(legacy_node, "save_chat_message", return_value="message-1"),
              patch.object(legacy_node, "resolve_product", return_value=result),
              patch.object(legacy_node, "add_running_task"),
              patch.object(legacy_node, "add_done_task")):
            output = legacy_node.node_item_name_confirm(self.state)
        self.assertEqual(output["answer"], "您是想问以下哪个产品：设备 B 1、设备 B 2？请明确一下型号")
        self.assertEqual(output["item_names"], [])

    def test_service_failure_finishes_legacy_node_progress(self):
        failure = ProductResolutionResult(
            status="error", rewritten_query="它怎么开机？",
            error_code="PRODUCT_MATCH_FAILED", retryable=True,
        )
        with (patch.object(legacy_node, "get_recent_messages", return_value=self.history),
              patch.object(legacy_node, "save_chat_message", return_value="message-1"),
              patch.object(legacy_node, "resolve_product", return_value=failure),
              patch.object(legacy_node, "add_running_task"),
              patch.object(legacy_node, "add_done_task") as done):
            with self.assertRaisesRegex(RuntimeError, "PRODUCT_MATCH_FAILED"):
                legacy_node.node_item_name_confirm(self.state)
        done.assert_called_once_with("offline-product-test", "node_item_name_confirm", False)


if __name__ == "__main__":
    unittest.main()
