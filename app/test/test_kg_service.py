"""图谱服务离线测试：验证状态、证据和旧节点之外的调用方式。"""

import unittest
from unittest.mock import MagicMock, patch

from neo4j import READ_ACCESS

from app.query_process.services import kg_service as kg


def candidate(chunk_id, item_name="设备 A"):
    return {"chunk_id": chunk_id, "item_name": item_name}


def row(chunk_id, item_name="设备 A", content="原文"):
    return {"chunk_id": chunk_id, "item_name": item_name, "content": content, "title": "说明书"}


class KGServiceTests(unittest.TestCase):
    def test_standalone_lookup_preserves_rank_and_returns_source_evidence(self):
        driver, client = MagicMock(), MagicMock()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value.data.return_value = [candidate(3), candidate(1)]
        client.get.return_value = [row(1), row(3)]

        result = kg.search_knowledge_graph(
            "  电源 开关  ", ["设备 A", "设备 A"],
            driver=driver, client=client, database="test", collection_name="chunks",
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual([hit["id"] for hit in result.items], [3, 1])
        self.assertEqual([e.chunk_id for e in result.evidence], [3, 1])
        self.assertEqual(result.evidence[0].text, "原文")
        self.assertEqual(result.evidence[0].title, "说明书")
        self.assertEqual(result.evidence[0].source, "local")
        self.assertEqual(result.evidence[0].retrieval_method, "kg")
        self.assertEqual(result.evidence[0].item_name, "设备 A")
        driver.session.assert_called_once_with(database="test", default_access_mode=READ_ACCESS)
        params = session.run.call_args.kwargs["parameters"]
        self.assertEqual(params["query"], "电源开关")
        self.assertEqual(params["item_names"], ["设备 A"])
        self.assertEqual(params["seed_limit"], 10)
        self.assertEqual(params["chunk_limit"], 20)
        self.assertEqual(session.run.call_args.args[0].timeout, 5.0)
        self.assertEqual(client.get.call_args.kwargs["ids"], [3, 1])

    def test_no_scope_or_no_candidates_is_empty_without_milvus(self):
        driver, client = MagicMock(), MagicMock()
        self.assertEqual(kg.search_knowledge_graph("问题", [], driver=driver, client=client).status, "empty")
        driver.session.assert_not_called()
        client.get.assert_not_called()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value.data.return_value = []
        self.assertEqual(kg.search_knowledge_graph("问题", ["设备 A"], driver=driver, client=client).status, "empty")
        client.get.assert_not_called()

    def test_missing_original_or_wrong_product_is_partial(self):
        client = MagicMock()
        client.get.return_value = [row(1), row(2, "设备 B"), row(3, content="  ")]
        result = kg.search_knowledge_graph(
            "问题", ["设备 A"], client=client, collection_name="chunks",
            query_chunk_ids_fn=lambda *_: [candidate(n) for n in (1, 2, 3, 4)],
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.error_code, "MISSING_ORIGINAL_TEXT")
        self.assertFalse(result.retryable)
        self.assertEqual([hit["id"] for hit in result.items], [1])
        self.assertEqual(result.candidate_count, 4)

    def test_neo4j_failure_is_error_not_empty(self):
        result = kg.search_knowledge_graph(
            "问题", ["设备 A"],
            query_chunk_ids_fn=lambda *_: (_ for _ in ()).throw(TimeoutError()),
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "KG_QUERY_FAILED")
        self.assertTrue(result.retryable)
        self.assertEqual(result.items, [])

    def test_milvus_failed_batch_is_partial_when_another_batch_succeeds(self):
        client = MagicMock()
        client.get.side_effect = [TimeoutError(), [row(2)]]
        client.query.side_effect = TimeoutError()
        original_fetch = kg.fetch_chunks_by_chunk_ids

        def fetch_one_at_a_time(*args, **kwargs):
            return original_fetch(*args, **kwargs, batch_size=1)

        with patch.object(kg, "fetch_chunks_by_chunk_ids", side_effect=fetch_one_at_a_time):
            result = kg.search_knowledge_graph(
                "问题", ["设备 A"], client=client, collection_name="chunks",
                query_chunk_ids_fn=lambda *_: [candidate(1), candidate(2)],
            )
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.error_code, "MILVUS_LOOKUP_FAILED")
        self.assertTrue(result.retryable)
        self.assertEqual([hit["id"] for hit in result.items], [2])

    def test_milvus_fallback_success_and_total_failure_are_distinct(self):
        client = MagicMock()
        client.get.side_effect = TimeoutError()
        client.query.return_value = [row(1)]
        result = kg.search_knowledge_graph(
            "问题", ["设备 A"], client=client, collection_name="chunks",
            query_chunk_ids_fn=lambda *_: [candidate(1)],
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.retryable)

        client.query.side_effect = TimeoutError()
        result = kg.search_knowledge_graph(
            "问题", ["设备 A"], client=client, collection_name="chunks",
            query_chunk_ids_fn=lambda *_: [candidate(1)],
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "MILVUS_LOOKUP_FAILED")
        self.assertTrue(result.retryable)

    def test_milvus_unavailable_is_error_and_invalid_arguments_are_rejected(self):
        with patch.object(kg, "get_milvus_client", return_value=None):
            result = kg.search_knowledge_graph(
                "问题", ["设备 A"], query_chunk_ids_fn=lambda *_: [candidate(1)],
            )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "MILVUS_LOOKUP_FAILED")
        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(kg.search_knowledge_graph("问题", "设备 A").error_code, "INVALID_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
