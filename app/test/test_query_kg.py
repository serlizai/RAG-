"""第四步离线测试：不访问数据库、模型接口或下载模型。"""

import copy
import runpy
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from neo4j import READ_ACCESS

from app.clients import milvus_utils as milvus
from app.query_process.agent.nodes import node_query_kg as kg
from app.query_process.agent.nodes import node_rrf as rrf
from app.query_process.agent.state import create_query_default_state
from app.utils import task_utils


def entity(chunk_id, item_name="设备 A", content=None):
    return {"chunk_id": chunk_id, "item_name": item_name, "title": "使用说明",
            "content": content if content is not None else f"切片 {chunk_id} 的原文"}


def hit(chunk_id):
    return {"id": chunk_id, "entity": entity(chunk_id)}


class MilvusChunkLookupTests(unittest.TestCase):
    def test_ids_are_strict_int64_and_deduplicated_without_reordering(self):
        valid, invalid = milvus._coerce_int64_ids(
            [3, " 2 ", "03", 0, -1, -(2**63), 2**63 - 1,
             True, False, 1.8, "1.8", None, 2**63, -(2**63) - 1, "1] or true"]
        )
        self.assertEqual(valid, [3, 2, 0, -1, -(2**63), 2**63 - 1])
        self.assertEqual(len(invalid), 8)

    def test_lookup_restores_rank_across_batches_and_drops_missing_or_extra_rows(self):
        client = MagicMock()
        client.get.side_effect = [
            [entity(1), entity(3), entity(3), entity(999), {"content": "没有 ID"}],
            [entity(2)],  # 4 在 Milvus 中已不存在。
        ]
        fields = ["content"]
        rows = milvus.fetch_chunks_by_chunk_ids(
            client, "test_chunks", [3, 1, 3, 4, 2], output_fields=fields, batch_size=2,
        )
        self.assertEqual([r["chunk_id"] for r in rows], [3, 1, 2])
        self.assertEqual([c.kwargs["ids"] for c in client.get.call_args_list], [[3, 1], [4, 2]])
        self.assertEqual(fields, ["content"])
        self.assertEqual(client.get.call_args.kwargs["output_fields"], ["chunk_id", "content"])
        self.assertEqual(client.get.call_args.kwargs["timeout"], 5.0)
        client.query.assert_not_called()

    def test_get_failure_uses_numeric_filter_and_restores_rank(self):
        client = MagicMock()
        client.get.side_effect = RuntimeError("模拟 get 不可用")
        client.query.return_value = [entity(1), entity(2)]
        rows = milvus.fetch_chunks_by_chunk_ids(client, "test_chunks", [2, 1, "1] or true"])
        self.assertEqual([r["chunk_id"] for r in rows], [2, 1])
        self.assertEqual(client.query.call_args.kwargs["filter"], "chunk_id in [2, 1]")

    def test_failed_batch_is_skipped_but_later_batches_can_succeed(self):
        client = MagicMock()
        client.get.side_effect = [TimeoutError(), [entity(2)]]
        client.query.side_effect = TimeoutError()
        rows = milvus.fetch_chunks_by_chunk_ids(client, "test_chunks", [1, 2], batch_size=1)
        self.assertEqual([r["chunk_id"] for r in rows], [2])

    def test_empty_result_does_not_repeat_query_and_empty_inputs_do_not_connect(self):
        client = MagicMock()
        client.get.return_value = []
        self.assertEqual(milvus.fetch_chunks_by_chunk_ids(client, "test_chunks", [1]), [])
        client.query.assert_not_called()
        client.reset_mock()
        for collection, ids in (("", [1]), ("test_chunks", []), ("test_chunks", [False, None])):
            self.assertEqual(milvus.fetch_chunks_by_chunk_ids(client, collection, ids), [])
        client.get.assert_not_called()
        self.assertEqual(milvus.fetch_chunks_by_chunk_ids(None, "test_chunks", [1]), [])
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                milvus.fetch_chunks_by_chunk_ids(client, "test_chunks", [1], batch_size=size)


class KGRecallTests(unittest.TestCase):
    def setUp(self):
        self.session_id = "kg_recall_test_" + uuid.uuid4().hex
        self.addCleanup(task_utils.clear_task, self.session_id)
        self.state = create_query_default_state(
            session_id=self.session_id, item_names=["设备 A"], original_query="电源开关怎么检查？",
        )

    def test_query_normalization_fallback_and_required_scope(self):
        self.state.update(rewritten_query="  POWER 电源 开关  ", item_names=["设备 A", "设备 A", "", None])
        self.assertEqual(kg.step_1_prepare_kg_query(self.state), ("power电源开关", ["设备 A"]))
        self.state.update(rewritten_query=" ", item_names="设备 A")
        self.assertEqual(kg.step_1_prepare_kg_query(self.state), ("电源开关怎么检查？", []))

    def test_cypher_uses_bound_parameters_read_mode_and_timeout(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        expected = [{"chunk_id": 3, "item_name": "设备 A"}]
        session.run.return_value.data.return_value = expected
        query = "电源开关 ' RETURN 1 //"
        self.assertEqual(kg.step_2_query_chunk_ids(query, ["设备 A"], driver=driver, database="test"), expected)
        driver.session.assert_called_once_with(database="test", default_access_mode=READ_ACCESS)
        call = session.run.call_args
        self.assertEqual(call.args[0].timeout, kg.KG_QUERY_TIMEOUT)
        self.assertNotIn(query, call.args[0].text)
        self.assertEqual(call.kwargs["parameters"]["query"], query)
        driver.session.return_value.__exit__.assert_called_once()
        driver.close.assert_not_called()

    def test_node_returns_ranked_original_text_without_mutating_state_and_finishes_progress(self):
        original = copy.deepcopy(self.state)
        candidates = [{"chunk_id": n, "item_name": "设备 A"} for n in (3, 1)]
        client = MagicMock()
        client.get.return_value = [entity(1), entity(3)]
        with (patch.object(kg, "step_2_query_chunk_ids", return_value=candidates),
              patch.object(kg, "get_milvus_client", return_value=client),
              patch.object(kg.milvus_config, "chunks_collection", "test_chunks")):
            self.assertEqual(kg.node_query_kg(self.state), {"kg_chunks": [hit(3), hit(1)]})
        self.assertEqual(self.state, original)
        self.assertEqual(task_utils.get_running_task_list(self.session_id), [])
        self.assertIn("查询知识图谱", task_utils.get_done_task_list(self.session_id))

    def test_missing_scope_or_question_and_no_candidates_skip_connections(self):
        with (patch.object(kg, "get_neo4j_driver") as driver,
              patch.object(kg, "get_milvus_client") as client):
            for changes in ({"item_names": []}, {"original_query": ""}):
                self.assertEqual(kg.node_query_kg({**self.state, **changes}), {"kg_chunks": []})
            self.assertEqual(kg.step_3_fetch_kg_chunks([], ["设备 A"]), [])
            driver.assert_not_called()
            client.assert_not_called()

    def test_missing_stale_wrong_product_and_empty_text_are_skipped(self):
        candidates = [{"chunk_id": n, "item_name": "设备 A"} for n in (1, 2, 3, 4)]
        candidates.append({"chunk_id": 5, "item_name": "设备 B"})
        client = MagicMock()
        client.get.return_value = [entity(1), entity(2, "设备 B"), entity(3, content=" ")]
        actual = kg.step_3_fetch_kg_chunks(candidates, ["设备 A"], client=client, collection_name="test_chunks")
        self.assertEqual(actual, [hit(1)])
        self.assertEqual(client.get.call_args.kwargs["ids"], [1, 2, 3, 4])

    def test_database_exceptions_return_empty_and_finish_streaming_progress(self):
        candidates = [{"chunk_id": 1, "item_name": "设备 A"}]
        self.state["is_stream"] = True
        for failing_step in ("step_2_query_chunk_ids", "step_3_fetch_kg_chunks"):
            with (patch.object(kg, "step_2_query_chunk_ids", return_value=candidates),
                  patch.object(kg, failing_step, side_effect=TimeoutError("模拟超时")),
                  patch.object(task_utils, "push_to_session") as push):
                self.assertEqual(kg.node_query_kg(self.state), {"kg_chunks": []})
                self.assertEqual(push.call_args.args[2]["running_list"], [])
                self.assertIn("查询知识图谱", push.call_args.args[2]["done_list"])


class KGReferencePipelineTests(unittest.TestCase):
    def test_rrf_cross_source_votes_normalize_ids_and_count_each_source_once(self):
        a, b = hit(1), hit(2)
        sources = [([a, a], 1.0), ([b], 1.0), ([hit("2")], 0.5), (None, 1.0)]
        self.assertEqual(rrf.reciprocal_rank_fusion(sources), [b, a])
        self.assertEqual(rrf.reciprocal_rank_fusion([([{}, hit(0)], 1.0)]), [hit(0)])

    def test_rrf_accepts_kg_only_and_preserves_twenty_candidates_for_rerank(self):
        session_id = "rrf_" + uuid.uuid4().hex
        self.addCleanup(task_utils.clear_task, session_id)
        actual = rrf.node_rrf({"session_id": session_id, "kg_chunks": [hit(n) for n in range(21)]})
        self.assertEqual(actual["rrf_chunks"], [hit(n) for n in range(20)])

    def test_actual_main_graph_joins_all_branches_once_and_reaches_reranker(self):
        # 运行真实 main_graph 的连线；外部模型/联网节点替换为确定结果。
        # RRF、图谱节点、Reranker 输入格式转换仍执行生产代码。
        root = Path(__file__).resolve().parents[2]
        lm_stub = types.ModuleType("app.lm.reranker_utils")
        lm_stub.get_reranker_model = MagicMock(side_effect=AssertionError("不应加载模型"))
        with patch.dict(sys.modules, {lm_stub.__name__: lm_stub}):
            rerank_module = runpy.run_path(str(root / "app/query_process/agent/nodes/node_rerank.py"))
        merge_docs = rerank_module["step_1_merge_docs"]
        fused = MagicMock(wraps=rrf.node_rrf)
        rerank = MagicMock(side_effect=lambda s: {"reranked_docs": merge_docs(s)})
        stubs = {
            "node_item_name_confirm": lambda s: {"item_names": ["设备 A"]},
            "node_search_embedding": lambda s: {"embedding_chunks": [hit(1)]},
            "node_search_embedding_hyde": lambda s: {"hyde_embedding_chunks": [hit(2)]},
            "node_web_search_mcp": lambda s: {"web_search_docs": []},
            "node_rrf": fused, "node_rerank": rerank,
            "node_answer_output": lambda s: {"answer": "测试回答"},
        }
        modules = {}
        for name, callback in stubs.items():
            module = types.ModuleType(f"app.query_process.agent.nodes.{name}")
            setattr(module, name, callback)
            modules[module.__name__] = module
        client = MagicMock()
        client.get.return_value = [entity(3), entity(1)]
        session_id = "graph_" + uuid.uuid4().hex
        self.addCleanup(task_utils.clear_task, session_id)
        with (patch.dict(sys.modules, modules),
              patch.object(kg, "step_2_query_chunk_ids", return_value=[
                  {"chunk_id": 1, "item_name": "设备 A"}, {"chunk_id": 3, "item_name": "设备 A"}]),
              patch.object(kg, "get_milvus_client", return_value=client),
              patch.object(kg.milvus_config, "chunks_collection", "test_chunks")):
            graph = runpy.run_path(str(root / "app/query_process/agent/main_graph.py"))["query_app"]
            result = graph.invoke(create_query_default_state(session_id=session_id, original_query="电源开关"))
            self.assertEqual(result["answer"], "测试回答")
            self.assertEqual([c["id"] for c in result["rrf_chunks"]], [1, 2, 3])
            self.assertEqual([c["chunk_id"] for c in result["reranked_docs"]], [1, 2, 3])
            self.assertEqual(result["reranked_docs"][2]["text"], "切片 3 的原文")
            fused.assert_called_once()
            rerank.assert_called_once()
            # 缺少明确商品而已生成提示时，原有提前返回路线仍然有效。
            fused.reset_mock()
            result = graph.invoke(create_query_default_state(session_id=session_id, answer="请明确商品"))
            fused.assert_not_called()


if __name__ == "__main__":
    unittest.main()
