"""Offline behavior checks for the extracted retrieval service and adapters."""

import copy
import unittest
import uuid
from unittest.mock import MagicMock, patch

from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.services import retrieval_service as retrieval
from app.utils import task_utils


def hit(chunk_id, text="设备说明", item_name="设备 A"):
    return {"id": chunk_id, "distance": 0.9,
            "entity": {"chunk_id": chunk_id, "content": text,
                       "title": "说明书", "file_title": "manual.pdf", "item_name": item_name}}


class VectorSearchTests(unittest.TestCase):
    def test_scope_is_required_without_loading_models_or_connecting(self):
        with patch("app.lm.embedding_utils.generate_embeddings") as embed:
            result = retrieval.search_embedding("如何开机", [])
        self.assertEqual((result.status, result.error_code), ("needs_clarification", "PRODUCT_REQUIRED"))
        embed.assert_not_called()

    def test_milvus_result_preserves_hit_shape_and_quotes_product_filter(self):
        raw = hit(12)
        with (patch("app.lm.embedding_utils.generate_embeddings",
                    return_value={"dense": [[0.1]], "sparse": [{1: 0.2}]}) as embed,
              patch("app.clients.milvus_utils.create_hybrid_search_requests",
                    return_value=["dense", "sparse"]) as create,
              patch("app.clients.milvus_utils.hybrid_search", return_value=[[raw]]) as search,
              patch("app.clients.milvus_utils.get_milvus_client", return_value=object())):
            result = retrieval.search_embedding("如何开机", ['设备 "A"'])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.items, [raw])
        self.assertEqual(result.evidence[0].id, "local:12")
        self.assertEqual(result.evidence[0].retrieval_method, "embedding")
        self.assertEqual(result.candidate_count, 1)
        embed.assert_called_once_with(["如何开机"])
        self.assertEqual(create.call_args.kwargs["expr"], 'item_name in ["设备 \\"A\\""]')
        self.assertEqual(search.call_args.kwargs["ranker_weights"], (0.9, 0.1))
        self.assertTrue(search.call_args.kwargs["norm_score"])

    def test_empty_and_failed_search_are_distinct(self):
        with (patch("app.lm.embedding_utils.generate_embeddings",
                    return_value={"dense": [[0.1]], "sparse": [{1: 0.2}]}),
              patch("app.clients.milvus_utils.create_hybrid_search_requests", return_value=[]),
              patch("app.clients.milvus_utils.get_milvus_client", return_value=object()),
              patch("app.clients.milvus_utils.hybrid_search", side_effect=[[], None])):
            empty = retrieval.search_embedding("开机", ["设备 A"])
            failed = retrieval.search_embedding("开机", ["设备 A"])
        self.assertEqual(empty.status, "empty")
        self.assertEqual(empty.items, [])
        self.assertEqual((failed.status, failed.error_code, failed.retryable),
                         ("error", "VECTOR_SEARCH_FAILED", True))

    def test_hyde_passes_generated_text_and_reports_generation_error(self):
        expected = retrieval.SearchResult(status="empty")
        with (patch.object(retrieval, "generate_hypothetical_document", return_value="假设答案") as generate,
              patch.object(retrieval, "search_embedding", return_value=expected) as search):
            self.assertIs(retrieval.search_embedding_hyde("开机", ["设备 A"]), expected)
        generate.assert_called_once_with("开机")
        search.assert_called_once_with("开机", ["设备 A"], hyde_doc="假设答案", output_fields=None)
        with patch.object(retrieval, "generate_hypothetical_document", side_effect=TimeoutError):
            failed = retrieval.search_embedding_hyde("开机", ["设备 A"])
        self.assertEqual((failed.status, failed.error_code), ("error", "HYDE_GENERATION_FAILED"))


class RankingTests(unittest.TestCase):
    def test_rrf_counts_each_branch_once_and_returns_existing_hit(self):
        one, two = hit(1), hit(2)
        result = retrieval.fuse_local_results([one, one], [two], [hit("2")])
        self.assertEqual(result.items, [two, one])
        self.assertEqual([e.id for e in result.evidence], ["local:2", "local:1"])
        self.assertEqual(result.candidate_count, 2)

    def test_rerank_preserves_legacy_document_fields_and_cutoff(self):
        local = hit(1, "本地资料")
        web = {"title": "官网", "url": "https://example.com/doc", "snippet": "网页资料"}
        model = MagicMock()
        model.compute_score.return_value = [0.92, 0.12]
        with patch("app.lm.reranker_utils.get_reranker_model", return_value=model):
            result = retrieval.rerank_documents("如何开机", [local], [web])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.items, [{"text": "本地资料", "chunk_id": 1,
                                         "title": "说明书", "file_title": "manual.pdf",
                                         "item_name": "设备 A", "url": "",
                                         "source": "local", "score": 0.92}])
        self.assertEqual(result.evidence[0].id, "local:1")
        model.compute_score.assert_called_once_with(
            [["如何开机", "本地资料"], ["如何开机", "网页资料"]], normalize=True,
        )

    def test_web_evidence_keeps_url_and_single_score(self):
        model = MagicMock()
        model.compute_score.return_value = 0.7
        with patch("app.lm.reranker_utils.get_reranker_model", return_value=model):
            result = retrieval.rerank_documents("消息", [], [
                {"title": "消息来源", "url": "https://example.com/news", "snippet": "发布内容"},
            ])
        self.assertEqual(result.items[0]["source"], "web")
        self.assertEqual(result.evidence[0].url, "https://example.com/news")
        self.assertEqual(result.evidence[0].id, "web:https://example.com/news")

    def test_empty_does_not_load_reranker_and_error_is_explicit(self):
        with patch("app.lm.reranker_utils.get_reranker_model") as get_model:
            self.assertEqual(retrieval.rerank_documents("问题", [], []).status, "empty")
            get_model.assert_not_called()
        with patch("app.lm.reranker_utils.get_reranker_model", side_effect=TimeoutError):
            result = retrieval.rerank_documents("问题", [hit(1)], [])
        self.assertEqual((result.status, result.error_code, result.candidate_count),
                         ("error", "RERANK_FAILED", 1))


class LegacyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.session_id = "retrieval_" + uuid.uuid4().hex
        self.addCleanup(task_utils.clear_task, self.session_id)
        self.state = {"session_id": self.session_id, "is_stream": False,
                      "rewritten_query": "开机", "item_names": ["设备 A"],
                      "rrf_chunks": [hit(1)]}

    def test_nodes_only_map_service_items_and_finish_progress(self):
        original = copy.deepcopy(self.state)
        result = retrieval.SearchResult(status="ok", items=[hit(1)])
        with patch("app.query_process.agent.nodes.node_search_embedding.search_embedding", return_value=result):
            self.assertEqual(node_search_embedding(self.state), {"embedding_chunks": result.items})
        with patch.object(retrieval, "search_embedding_hyde", return_value=result):
            self.assertEqual(node_search_embedding_hyde(self.state), {"hyde_embedding_chunks": result.items})
        with patch("app.query_process.agent.nodes.node_rrf.fuse_local_results", return_value=result):
            self.assertEqual(node_rrf(self.state), {"rrf_chunks": result.items})
        with patch("app.query_process.agent.nodes.node_rerank.rerank_documents", return_value=result):
            self.assertEqual(node_rerank(self.state), {"reranked_docs": result.items})
        self.assertEqual(self.state, original)
        self.assertEqual(task_utils.get_running_task_list(self.session_id), [])

    def test_legacy_critical_failures_still_propagate_and_finish_progress(self):
        hyde_error = retrieval.SearchResult(
            status="error", error_code="HYDE_GENERATION_FAILED"
        )
        with patch.object(retrieval, "search_embedding_hyde", return_value=hyde_error):
            with self.assertRaisesRegex(RuntimeError, "HyDE 假设文档生成失败"):
                node_search_embedding_hyde(self.state)

        rerank_error = retrieval.SearchResult(status="error", error_code="RERANK_FAILED")
        with patch("app.query_process.agent.nodes.node_rerank.rerank_documents", return_value=rerank_error):
            with self.assertRaisesRegex(RuntimeError, "RERANK_FAILED"):
                node_rerank(self.state)
        self.assertEqual(task_utils.get_running_task_list(self.session_id), [])


if __name__ == "__main__":
    unittest.main()
