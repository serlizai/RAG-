"""第四步真实数据库测试，必须显式指定测试地址，不读取业务集合名。

示例（Neo4j 地址替换为临时测试容器的端口）：
NEO4J_KG_TEST_URI=bolt://127.0.0.1:端口 NEO4J_KG_TEST_PASSWORD=测试密码 \
MILVUS_KG_TEST_URI=http://127.0.0.1:19530 \
.venv/bin/python -m unittest app.test.test_query_kg_integration -v

Milvus 新建随机集合；Neo4j 使用随机商品名；测试结束仅清理本次创建的数据。
不调用大模型，不依赖已导入的商品，不运行向量模型或最终答案生成。
"""

import os
import unittest
import uuid
from unittest.mock import patch

from neo4j import GraphDatabase
from pymilvus import DataType, MilvusClient

from app.clients.milvus_utils import fetch_chunks_by_chunk_ids
from app.import_process.agent.nodes.node_import_kg import step_3_save_to_neo4j
from app.query_process.agent.nodes import node_query_kg as kg
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import clear_task


@unittest.skipUnless(
    os.getenv("NEO4J_KG_TEST_URI") and os.getenv("MILVUS_KG_TEST_URI"),
    "需要显式指定 Neo4j 和 Milvus 测试地址",
)
class KGRecallIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = uuid.uuid4().hex
        cls.item_name = f"__kg_recall_test_{suffix} O'Reilly"
        cls.other_name = f"__kg_recall_other_{suffix}"
        cls.collection = "kg_recall_test_" + suffix
        cls.driver = GraphDatabase.driver(
            os.environ["NEO4J_KG_TEST_URI"],
            auth=("neo4j", os.environ["NEO4J_KG_TEST_PASSWORD"]),
        )
        cls.addClassCleanup(cls.driver.close)
        cls.driver.verify_connectivity()
        cls.client = MilvusClient(uri=os.environ["MILVUS_KG_TEST_URI"], timeout=10)
        cls.addClassCleanup(cls.client.close)

        # 主键与生产切片集合一样由 Milvus 自动生成。小向量仅满足建集合要求，
        # 本测试验证 ID 回查，完全不依赖向量的语义质量。
        schema = cls.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
        for name in ("content", "item_name", "title", "parent_title"):
            schema.add_field(name, DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=4)
        indexes = cls.client.prepare_index_params()
        indexes.add_index(field_name="dense_vector", index_type="FLAT", metric_type="L2")
        cls.client.create_collection(
            collection_name=cls.collection, schema=schema,
            index_params=indexes, consistency_level="Strong", timeout=20,
        )
        cls.addClassCleanup(cls.client.drop_collection, cls.collection, timeout=20)
        cls.addClassCleanup(cls.cleanup_graph)

        # 文档 0：直接命中“无法启动”；文档 1：只提到它的邻居“电源开关”。
        # 文档 2：需要再走一层才会遇到，查询“无法启动”时不应召回。
        # 文档 3/4：共用 Product 实体，验证不会沿商品节点扩散到无关温度说明。
        # 文档 5：另一商品的同名实体，验证不会串商品。
        specifications = [
            (cls.item_name, [("无法启动", "Fault"), ("电源开关", "Component")]),
            (cls.item_name, [("电源开关", "Component"), ("检查接线", "Operation")]),
            (cls.item_name, [("检查接线", "Operation"), ("断开电源", "Operation")]),
            (cls.item_name, [(cls.item_name, "Product"), ("无法启动", "Fault")]),
            (cls.item_name, [(cls.item_name, "Product"), ("温度设置", "Parameter")]),
            (cls.other_name, [("无法启动", "Fault"), ("电源开关", "Component")]),
        ]
        extractions = []
        for item_name, pairs in specifications:
            content = " 与 ".join(name for name, _ in pairs) + "相关，这是测试原文。"
            extractions.append({
                "item_name": item_name, "title": "测试说明", "file_title": "测试手册",
                "content": content,
                "entities": [
                    {"id": f"e{i}", "name": name, "type": kind, "evidence": name}
                    for i, (name, kind) in enumerate(pairs)
                ],
                "relations": [{"source": "e0", "target": "e1", "relation": "关联", "evidence": content}],
            })
        rows = [{**{k: d[k] for k in ("content", "title", "item_name")},
                 "parent_title": "测试手册", "dense_vector": [1.0, 0.0, 0.0, 0.0]}
                for d in extractions]
        result = cls.client.insert(cls.collection, rows, timeout=10)
        cls.ids = list(result["ids"])
        for doc, chunk_id in zip(extractions, cls.ids, strict=True):
            doc["chunk_id"] = chunk_id
        cls.client.flush(cls.collection, timeout=20)
        cls.docs = extractions
        # 使用真正的导入函数，确保召回 Cypher 与导入图结构兼容。
        step_3_save_to_neo4j(extractions, driver=cls.driver, database="neo4j")

    @classmethod
    def cleanup_graph(cls):
        with cls.driver.session(database="neo4j") as session:
            session.run(
                "MATCH (n) WHERE ((n:Chunk OR n:Entity) AND n.item_name IN $names) "
                "OR (n:Product AND n.name IN $names) DETACH DELETE n",
                names=[cls.item_name, cls.other_name],
            ).consume()

    def recall(self, query, names=None):
        return kg.step_2_query_chunk_ids(
            query, names if names is not None else [self.item_name],
            driver=self.driver, database="neo4j",
        )

    def test_direct_hits_rank_before_one_hop_neighbors_and_product_hub_is_excluded(self):
        rows = self.recall("无法启动怎么办")
        self.assertEqual([r["chunk_id"] for r in rows], [self.ids[0], self.ids[3], self.ids[1]])
        self.assertEqual([r["hop_count"] for r in rows], [0, 0, 1])
        self.assertEqual({r["item_name"] for r in rows}, {self.item_name})

    def test_relations_can_be_traversed_in_both_directions(self):
        rows = self.recall("怎样检查接线")
        self.assertEqual({r["chunk_id"] for r in rows}, {self.ids[0], self.ids[1], self.ids[2]})
        self.assertEqual(rows[-1]["chunk_id"], self.ids[0])
        self.assertEqual(rows[-1]["hop_count"], 1)

    def test_unmatched_product_only_and_out_of_scope_queries_return_empty(self):
        self.assertEqual(self.recall("液压系统如何维护"), [])
        self.assertEqual(self.recall(self.item_name.lower().replace(" ", "")), [])
        self.assertEqual(self.recall("无法启动", ["没有导入的商品"]), [])
        self.assertEqual(self.recall("无法启动", []), [])

    def test_multiple_confirmed_products_are_supported_without_merging_their_entities(self):
        rows = self.recall("无法启动", [self.item_name, self.other_name])
        self.assertEqual({r["chunk_id"] for r in rows}, {self.ids[i] for i in (0, 1, 3, 5)})
        other = next(r for r in rows if r["chunk_id"] == self.ids[5])
        self.assertEqual(other["item_name"], self.other_name)
        self.assertEqual(other["matched_entities"], 1)

    def test_cross_product_nodes_relations_and_proof_chunks_cannot_expand_scope(self):
        # 模拟历史脏数据：分别破坏目标实体、关系属性、关系证据的商品归属。
        # 即使这条边真的存在，查询也必须把它排除，而不是信任其中一个字段。
        cases = [
            (self.other_name, "电源开关", self.item_name, self.ids[0]),
            (self.item_name, "温度设置", self.other_name, self.ids[0]),
            (self.item_name, "温度设置", self.item_name, self.ids[5]),
        ]
        for target_item, target_name, relation_item, proof_id in cases:
            with self.subTest(target_item=target_item, relation_item=relation_item, proof_id=proof_id):
                marker = uuid.uuid4().hex
                with self.driver.session(database="neo4j") as session:
                    session.run(
                        "MATCH (s:Entity {item_name: $source_item, name: '无法启动'}), "
                        "(t:Entity {item_name: $target_item, name: $target_name}) "
                        "CREATE (s)-[:RELATED_TO {item_name: $relation_item, chunk_id: $proof_id, "
                        "test_marker: $marker}]->(t)",
                        source_item=self.item_name, target_item=target_item, target_name=target_name,
                        relation_item=relation_item, proof_id=proof_id, marker=marker,
                    ).consume()
                    try:
                        rows = self.recall("无法启动")
                        self.assertEqual([r["chunk_id"] for r in rows], [self.ids[0], self.ids[3], self.ids[1]])
                    finally:
                        session.run(
                            "MATCH ()-[r:RELATED_TO {test_marker: $marker}]->() DELETE r", marker=marker,
                        ).consume()

    def test_real_milvus_lookup_keeps_requested_order_and_skips_nonexistent_id(self):
        rows = fetch_chunks_by_chunk_ids(self.client, self.collection, [self.ids[3], -1, self.ids[0]], batch_size=2)
        self.assertEqual([r["chunk_id"] for r in rows], [self.ids[3], self.ids[0]])
        self.assertEqual(rows[0]["content"], self.docs[3]["content"])

    def test_real_milvus_query_fallback_uses_same_ids_and_original_text(self):
        # get 的异常是假故障；回退的 query 确实发送到 Milvus 服务器执行。
        with patch.object(self.client, "get", side_effect=RuntimeError("模拟 get 故障")):
            rows = fetch_chunks_by_chunk_ids(self.client, self.collection, [self.ids[1], self.ids[0]])
        self.assertEqual([r["chunk_id"] for r in rows], [self.ids[1], self.ids[0]])
        self.assertEqual(rows[0]["content"], self.docs[1]["content"])

    def test_full_kg_node_and_rrf_return_milvus_original_text(self):
        session_id = "kg_live_test_" + uuid.uuid4().hex
        self.addCleanup(clear_task, session_id)
        state = create_query_default_state(
            session_id=session_id, item_names=[self.item_name], original_query="无法启动怎么办",
            embedding_chunks=[{"id": self.ids[1], "entity": self.docs[1]}],
        )
        with (patch.object(kg, "get_neo4j_driver", return_value=self.driver),
              patch.object(kg, "get_milvus_client", return_value=self.client),
              patch.object(kg.milvus_config, "chunks_collection", self.collection),
              patch.dict(os.environ, {"NEO4J_DATABASE": "neo4j"})):
            state.update(kg.node_query_kg(state))
        self.assertEqual([c["id"] for c in state["kg_chunks"]], [self.ids[0], self.ids[3], self.ids[1]])
        self.assertEqual(state["kg_chunks"][-1]["entity"]["content"], self.docs[1]["content"])
        fused = node_rrf(state)["rrf_chunks"]
        self.assertEqual(len(fused), 3)
        # 文档 1 同时被向量路和图谱路命中，融合后应该排在第一位，且只出现一次。
        self.assertEqual(fused[0]["id"], self.ids[1])


if __name__ == "__main__":
    unittest.main()
