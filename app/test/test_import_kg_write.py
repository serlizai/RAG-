"""第三步测试；真实数据库测试需显式设置 NEO4J_KG_TEST_URI，默认只运行离线测试。"""

import copy
import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from neo4j import GraphDatabase

from app.import_process.agent.nodes import node_import_kg as kg
from app.import_process.agent.state import create_default_state
from app.test.test_import_kg_extraction import make_chunk, make_extraction


def make_result(chunk_id=101, item_name="示例设备"):
    result = {**make_chunk(chunk_id), **make_extraction()}
    result["item_name"] = item_name
    result["content"] = f"{item_name}配有电源开关。"
    result["entities"][0].update(name=item_name, evidence=item_name)
    result["relations"][0]["evidence"] = result["content"]
    return result


class KGWritePreparationTests(unittest.TestCase):
    def test_local_ids_are_resolved_per_chunk_and_entities_are_reused(self):
        first = make_result(101)
        second = make_result(102)
        # 第二个切片把两个局部编号调换，实际关系仍应连接同一对实体。
        second["entities"][0]["id"] = "e2"
        second["entities"][1].update(id="e1", name="  电源开关  ")
        second["relations"][0].update(source="e2", target="e1")
        original = copy.deepcopy([first, second])
        data = kg._prepare_neo4j_data([first, second])
        self.assertEqual(len(data["entities"]), 2)
        self.assertEqual(len(data["mentions"]), 4)
        self.assertEqual(len(data["relations"]), 2)
        self.assertEqual(data["relations"][0]["source_key"], data["relations"][1]["source_key"])
        self.assertEqual(data["relations"][0]["target_key"], data["relations"][1]["target_key"])
        self.assertEqual([first, second], original)

    def test_same_name_in_different_products_or_types_is_not_merged(self):
        first = make_result(101, "设备 A")
        second = make_result(102, "设备 B")
        first["entities"].append({"id": "e3", "name": "电源开关", "type": "Function", "evidence": "电源开关"})
        data = kg._prepare_neo4j_data([first, second])
        self.assertEqual(len(data["entities"]), 5)
        self.assertEqual(len({e["entity_key"] for e in data["entities"]}), 5)
        self.assertNotEqual(
            kg._make_entity_key("a|b", "c", "d"),
            kg._make_entity_key("a", "b|c", "d"),
        )

    def test_empty_entities_still_keep_chunk_and_empty_batch_does_not_connect(self):
        result = make_result()
        result.update(entities=[], relations=[])
        data = kg._prepare_neo4j_data([result])
        self.assertEqual(len(data["chunks"]), 1)
        self.assertEqual(data["entities"], [])
        driver = MagicMock()
        self.assertEqual(kg.step_3_save_to_neo4j([], driver=driver)["chunks"], 0)
        driver.session.assert_not_called()

    def test_all_inputs_are_validated_before_opening_database_session(self):
        cases = [None, [make_result(), make_result()]]
        for invalid_id in (True, "101", 2**63, -(2**63) - 1):
            result = make_result(102)
            result["chunk_id"] = invalid_id
            cases.append([make_result(), result])
        for field, value in (("item_name", " "), ("entities", None), ("relations", {}), ("title", ["标题"])):
            result = make_result(102)
            result[field] = value
            cases.append([make_result(), result])
        invalid_reference = make_result(102)
        invalid_reference["relations"][0]["target"] = "不存在"
        cases.append([make_result(), invalid_reference])
        duplicate_local_id = make_result(102)
        duplicate_local_id["entities"][1]["id"] = "e1"
        cases.append([make_result(), duplicate_local_id])

        for case in cases:
            with self.subTest(case=case):
                driver = MagicMock()
                with self.assertRaises(ValueError):
                    kg.step_3_save_to_neo4j(case, driver=driver)
                driver.session.assert_not_called()

    def test_commit_failure_propagates_and_session_is_closed(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        session.execute_write.side_effect = RuntimeError("测试写入失败")
        with self.assertRaisesRegex(RuntimeError, "测试写入失败"):
            kg.step_3_save_to_neo4j([make_result()], driver=driver, database="isolated_test")
        driver.session.assert_called_once_with(database="isolated_test")
        driver.session.return_value.__exit__.assert_called_once()
        driver.close.assert_not_called()

    def test_node_does_not_report_import_stats_after_write_failure(self):
        state = create_default_state(task_id="kg_write_failure", chunks=[make_chunk()])
        with (
            patch.object(kg, "step_2_extract_entities_relations", return_value=[make_result()]),
            patch.object(kg, "step_3_save_to_neo4j", side_effect=RuntimeError("提交失败")),
        ):
            with self.assertRaisesRegex(RuntimeError, "提交失败"):
                kg.node_import_kg(state)
        self.assertEqual(state["kg_import_stats"], {})
        self.assertEqual(state["kg_extractions"][0]["chunk_id"], 101)


@unittest.skipUnless(os.getenv("NEO4J_KG_TEST_URI"), "需要显式指定隔离的 Neo4j 测试库")
class KGWriteIntegrationTests(unittest.TestCase):
    """验证实际 Cypher 和事务行为。每个测试仅清理自己创建的随机商品数据。"""

    test_scope = "integration"

    @classmethod
    def setUpClass(cls):
        cls.driver = GraphDatabase.driver(
            os.environ["NEO4J_KG_TEST_URI"],
            auth=("neo4j", os.environ["NEO4J_KG_TEST_PASSWORD"]),
        )
        cls.driver.verify_connectivity()

    @classmethod
    def tearDownClass(cls):
        cls.driver.close()

    def setUp(self):
        self.item_name = "__kg_step3_test_" + uuid.uuid4().hex
        self.other_name = self.item_name + "_other"
        # 使用随机负数作为测试 ID，不依赖已有 Milvus 数据。
        self.chunk_id = -(uuid.uuid4().int % (2**60)) - 100

    def query(self, cypher, **parameters):
        return self.driver.execute_query(cypher, database_="neo4j", **parameters).records

    def save(self, results):
        return kg.step_3_save_to_neo4j(results, driver=self.driver, database="neo4j")

    def snapshot(self, item_name):
        nodes = self.query("""
            MATCH (n)
            WHERE (n:Product AND n.name = $item_name)
               OR ((n:Chunk OR n:Entity) AND n.item_name = $item_name)
            RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY id
        """, item_name=item_name)
        edges = self.query("""
            MATCH (n)-[r]->(target)
            WHERE (n:Product AND n.name = $item_name) OR n.item_name = $item_name
            RETURN elementId(n) AS source, elementId(target) AS target,
                   type(r) AS type, properties(r) AS properties ORDER BY source, target, type
        """, item_name=item_name)
        return [r.data() for r in nodes], [r.data() for r in edges]

    def tearDown(self):
        self.query("""
            MATCH (n)
            WHERE (n:Product AND n.name IN $names)
               OR ((n:Chunk OR n:Entity) AND n.item_name IN $names)
            DETACH DELETE n
        """, names=[self.item_name, self.other_name])

    def test_write_replay_and_original_evidence(self):
        result = make_result(self.chunk_id, self.item_name)
        stats = self.save([result])
        identities = self.query("""
            MATCH (e:Entity {item_name: $name})
            RETURN e.entity_key AS key, elementId(e) AS id ORDER BY key
        """, name=self.item_name)
        self.assertEqual(self.save([result]), stats)
        self.assertEqual(stats, {"products": 1, "chunks": 1, "entities": 2, "mentions": 2, "relations": 1})
        self.assertEqual(identities, self.query("""
            MATCH (e:Entity {item_name: $name})
            RETURN e.entity_key AS key, elementId(e) AS id ORDER BY key
        """, name=self.item_name))
        rows = self.query("""
            MATCH (p:Product {name: $name})-[:HAS_CHUNK]->(c:Chunk)-[m:MENTIONS]->(e:Entity)
            RETURN c.chunk_id AS chunk_id, e.name AS name, m.evidence AS evidence
        """, name=self.item_name)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["chunk_id"] for r in rows}, {self.chunk_id})
        self.assertEqual({r["evidence"] for r in rows}, {self.item_name, "电源开关"})
        facts = self.query("""
            MATCH (s:Entity)-[r:RELATED_TO]->(t:Entity)
            WHERE r.item_name = $name
            RETURN s.name AS source, t.name AS target, r.chunk_id AS chunk_id,
                   r.relation AS relation, r.evidence AS evidence
        """, name=self.item_name)
        self.assertEqual([r.data() for r in facts], [{
            "source": self.item_name, "target": "电源开关", "chunk_id": self.chunk_id,
            "relation": "包含", "evidence": result["content"],
        }])

    def test_replace_removes_stale_sources_and_preserves_other_product(self):
        old = make_result(self.chunk_id, self.item_name)
        old["entities"].append({"id": "e3", "name": "旧故障", "type": "Fault", "evidence": "旧故障"})
        second = make_result(self.chunk_id - 1, self.item_name)
        other = make_result(self.chunk_id - 2, self.other_name)
        self.save([old, second, other])
        other_before = self.snapshot(self.other_name)
        replacement = make_result(self.chunk_id - 3, self.item_name)
        self.save([replacement])
        nodes, edges = self.snapshot(self.item_name)
        self.assertEqual(len(nodes), 4)  # 商品 + 一个切片 + 两个实体
        self.assertEqual(len(edges), 4)  # HAS_CHUNK + 两条 MENTIONS + RELATED_TO
        self.assertEqual({n["properties"]["chunk_id"] for n in nodes if "Chunk" in n["labels"]}, {self.chunk_id - 3})
        self.assertNotIn("旧故障", {n["properties"].get("name") for n in nodes})
        self.assertEqual(self.snapshot(self.other_name), other_before)

    def test_empty_knowledge_replaces_old_entities_and_relations(self):
        self.save([make_result(self.chunk_id, self.item_name)])
        empty = make_result(self.chunk_id - 1, self.item_name)
        empty.update(entities=[], relations=[])
        self.save([empty])
        nodes, edges = self.snapshot(self.item_name)
        self.assertEqual(len(nodes), 2)
        self.assertEqual([e["type"] for e in edges], ["HAS_CHUNK"])

    def test_transaction_rolls_back_deletion_and_new_writes_together(self):
        self.save([make_result(self.chunk_id, self.item_name)])
        before = self.snapshot(self.item_name)
        data = kg._prepare_neo4j_data([make_result(self.chunk_id - 1, self.item_name)])

        def fail_after_writes(tx):
            kg._write_kg_transaction(tx, data)
            raise ValueError("故意触发回滚")

        with self.driver.session(database="neo4j") as session:
            with self.assertRaisesRegex(ValueError, "故意触发回滚"):
                session.execute_write(fail_after_writes)
        self.assertEqual(self.snapshot(self.item_name), before)

    def test_conflicting_chunk_id_cannot_overwrite_another_product(self):
        self.save([make_result(self.chunk_id, self.item_name)])
        before = self.snapshot(self.item_name)
        with self.assertRaisesRegex(ValueError, "已被其他商品"):
            self.save([make_result(self.chunk_id, self.other_name)])
        self.assertEqual(self.snapshot(self.item_name), before)
        self.assertEqual(self.snapshot(self.other_name), ([], []))

    def test_concurrent_replacements_do_not_mix_batches(self):
        self.save([make_result(self.chunk_id, self.item_name)])
        batch_a = [make_result(self.chunk_id - i, self.item_name) for i in (1, 2)]
        batch_b = [make_result(self.chunk_id - i, self.item_name) for i in (3, 4)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(self.save, [batch_a, batch_b]))
        rows = self.query("MATCH (c:Chunk {item_name: $name}) RETURN c.chunk_id AS id", name=self.item_name)
        ids = {r["id"] for r in rows}
        self.assertIn(ids, ({r["chunk_id"] for r in batch_a}, {r["chunk_id"] for r in batch_b}))
        facts = self.query("MATCH ()-[r:RELATED_TO]->() WHERE r.item_name = $name RETURN r.chunk_id AS id", name=self.item_name)
        self.assertEqual({r["id"] for r in facts}, ids)


if __name__ == "__main__":
    unittest.main()
