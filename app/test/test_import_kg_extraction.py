"""第二步的离线测试：用预设模型响应验证抽取流程，不访问模型接口或数据库。"""

import copy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.import_process.agent.nodes import node_import_kg as kg
from app.import_process.agent.state import ImportGraphState, create_default_state


def make_chunk(chunk_id=101):
    return {
        "chunk_id": chunk_id,
        "item_name": "示例设备",
        "title": "组成",
        "file_title": "示例设备说明书",
        "content": "示例设备配有电源开关。",
    }


def make_extraction():
    return {
        "entities": [
            {"id": "e1", "name": "示例设备", "type": "Product", "evidence": "示例设备"},
            {"id": "e2", "name": "电源开关", "type": "Component", "evidence": "电源开关"},
        ],
        "relations": [
            {"source": "e1", "target": "e2", "relation": "包含", "evidence": "示例设备配有电源开关。"},
        ],
    }


class FakeLLM:
    """按调用顺序返回指定结果，或抛出指定异常；记录请求用于检查来源绑定。"""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, AIMessage):
            return response
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        return AIMessage(content=response)


class KGExtractionTests(unittest.TestCase):
    def test_all_chunks_keep_real_source_metadata_and_full_text(self):
        chunks = [make_chunk(101), make_chunk(102)]
        chunks[1]["item_name"] = "另一设备"
        # 引号、换行和花括号必须原样传入模型，不能被提示词格式化或截断。
        chunks[1]["content"] = '另一设备配有电源开关。\n显示内容为 "{mode}"。'
        second = make_extraction()
        second["entities"][0].update(name="另一设备", evidence="另一设备")
        second["relations"][0]["evidence"] = "另一设备配有电源开关。"
        first = make_extraction()
        first.update(chunk_id=999, item_name="模型编造的商品")
        original = copy.deepcopy(chunks)
        llm = FakeLLM(first, second)

        results = kg.step_2_extract_entities_relations(chunks, llm=llm)

        self.assertEqual([r["chunk_id"] for r in results], [101, 102])
        self.assertEqual([r["item_name"] for r in results], ["示例设备", "另一设备"])
        self.assertEqual(results[0]["file_title"], chunks[0]["file_title"])
        self.assertEqual(results[0]["relations"][0]["evidence"], chunks[0]["content"])
        self.assertEqual(chunks, original)
        for chunk, messages in zip(chunks, llm.calls):
            self.assertIsInstance(messages[0], SystemMessage)
            self.assertIsInstance(messages[1], HumanMessage)
            self.assertNotIn("{entity_types}", messages[0].content)
            sent = json.loads(messages[1].content)
            self.assertEqual(set(sent), {"item_name", "title", "evidence_sources"})
            self.assertEqual(sent["item_name"], chunk["item_name"])
            self.assertEqual(sent["title"], chunk["title"])
            self.assertEqual("".join(source["text"] for source in sent["evidence_sources"]), chunk["content"])
            self.assertTrue(all(set(source) == {"id", "text"} for source in sent["evidence_sources"]))

    def test_empty_extraction_is_valid_and_retains_chunk_id(self):
        results = kg.step_2_extract_entities_relations(
            [make_chunk()], llm=FakeLLM({"entities": [], "relations": []}),
        )
        self.assertEqual(results[0]["chunk_id"], 101)
        self.assertEqual(results[0]["entities"], [])
        self.assertEqual(results[0]["relations"], [])
        self.assertEqual(kg.step_2_extract_entities_relations([]), [])

    def test_fenced_json_and_text_blocks_are_supported(self):
        text = json.dumps(make_extraction(), ensure_ascii=False)
        message = AIMessage(content=[{"type": "text", "text": f"```json\n{text}\n```"}])
        result = kg.step_2_extract_entities_relations([make_chunk()], llm=FakeLLM(message))
        self.assertEqual(result[0]["entities"], make_extraction()["entities"])

    def test_identical_relations_are_deduplicated(self):
        data = make_extraction()
        data["relations"].append(copy.deepcopy(data["relations"][0]))
        result = kg.step_2_extract_entities_relations([make_chunk()], llm=FakeLLM(data))
        self.assertEqual(len(result[0]["relations"]), 1)

    def test_bad_structure_references_and_evidence_raise_with_chunk_id(self):
        cases = [
            ("空响应", ""),
            ("非 JSON", "无法抽取"),
            ("截断 JSON", '{"entities": ['),
            ("非对象", "[]"),
            ("缺少列表", {}),
            ("错误列表类型", {"entities": {}, "relations": []}),
            ("非法实体", {"entities": ["电源开关"], "relations": []}),
        ]
        # 分别破坏一个必要条件，防止合法 JSON 绕过字段与原文证据校验。
        for field, value in (("name", " "), ("type", "Unknown"), ("evidence", "手册未提到的电池")):
            data = make_extraction()
            data["entities"][0][field] = value
            cases.append((f"实体 {field}", data))
        duplicate = make_extraction()
        duplicate["entities"][1]["id"] = "e1"
        cases.append(("重复实体 ID", duplicate))
        for field, value in (("source", "missing"), ("target", "missing"), ("relation", 123), ("evidence", "需要更换电池")):
            data = make_extraction()
            data["relations"][0][field] = value
            cases.append((f"关系 {field}", data))
        missing = make_extraction()
        del missing["relations"][0]["evidence"]
        cases.append(("缺少关系证据", missing))
        invalid_relation = make_extraction()
        invalid_relation["relations"] = [None]
        cases.append(("非法关系", invalid_relation))

        for label, response in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "chunk_id=101") as caught:
                    kg.step_2_extract_entities_relations(
                        [make_chunk()], llm=FakeLLM(*([response] * kg.KG_EXTRACTION_MAX_ATTEMPTS)),
                    )
                self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_metadata_cannot_be_used_as_original_evidence(self):
        chunk = make_chunk()
        data = make_extraction()
        data["entities"][0]["evidence"] = chunk["file_title"]
        with self.assertRaisesRegex(RuntimeError, "原文中找到"):
            kg.step_2_extract_entities_relations(
                [chunk], llm=FakeLLM(*([data] * kg.KG_EXTRACTION_MAX_ATTEMPTS)),
            )

    def test_model_failure_is_not_treated_as_empty_knowledge(self):
        with self.assertRaisesRegex(RuntimeError, "chunk_id=101") as caught:
            kg.step_2_extract_entities_relations([make_chunk()], llm=FakeLLM(TimeoutError("测试超时")))
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)

    def test_graph_preserves_extractions_and_original_chunks(self):
        chunk = make_chunk()
        chunk["dense_vector"] = [0.1, 0.2]
        state = create_default_state(task_id="kg_offline_success", chunks=[chunk])
        original_chunks = copy.deepcopy(state["chunks"])
        extract = kg.step_2_extract_entities_relations
        llm = FakeLLM(make_extraction())

        # 使用实际状态类型与节点，验证新字段确实能经过 LangGraph 返回。
        builder = StateGraph(ImportGraphState)
        builder.add_node("node_import_kg", kg.node_import_kg)
        builder.set_entry_point("node_import_kg")
        builder.add_edge("node_import_kg", END)
        graph = builder.compile()
        stats = {"products": 1, "chunks": 1, "entities": 2, "mentions": 2, "relations": 1}
        with (
            patch.object(kg, "step_2_extract_entities_relations", side_effect=lambda chunks: extract(chunks, llm=llm)),
            patch.object(kg, "step_3_save_to_neo4j", return_value=stats) as save,
        ):
            result = graph.invoke(state)

        self.assertEqual(result["kg_extractions"][0]["chunk_id"], 101)
        self.assertEqual(result["kg_extractions"][0]["relations"], make_extraction()["relations"])
        self.assertEqual(result["chunks"], original_chunks)
        self.assertEqual(result["kg_import_stats"], stats)
        save.assert_called_once_with(result["kg_extractions"])

    def test_failure_does_not_publish_partial_batch(self):
        state = create_default_state(task_id="kg_offline_failure", chunks=[make_chunk(101), make_chunk(102)])
        original = copy.deepcopy(state)
        extract = kg.step_2_extract_entities_relations
        llm = FakeLLM(make_extraction(), TimeoutError("第二个切片调用失败"))
        with patch.object(kg, "step_2_extract_entities_relations", side_effect=lambda chunks: extract(chunks, llm=llm)):
            with self.assertRaisesRegex(RuntimeError, "chunk_id=102"):
                kg.node_import_kg(state)
        self.assertEqual(state, original)

    def test_default_extraction_lists_are_independent(self):
        first = create_default_state()
        second = create_default_state()
        first["kg_extractions"].append({"chunk_id": 101})
        self.assertEqual(second["kg_extractions"], [])


if __name__ == "__main__":
    unittest.main()
