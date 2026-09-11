"""表格证据与局部重试的离线回归测试；不调用模型接口或数据库。"""

import copy
import json
import unittest
from unittest.mock import patch

from app.import_process.agent.nodes import node_import_kg as kg
from app.import_process.agent.state import create_default_state
from app.import_process.agent.nodes.kg_evidence_sources import SOURCE_MAX_CHARS, build_evidence_sources
from app.test.test_import_kg_extraction import FakeLLM, make_chunk, make_extraction


class KGEvidenceTests(unittest.TestCase):
    def test_exact_evidence_is_kept_verbatim(self):
        for evidence in ("电源\n  开关", "<td>&lt; / +</td>", "1 0V", "&#x2242;"):
            with self.subTest(evidence=evidence):
                self.assertEqual(kg._resolve_evidence(evidence, f"前文{evidence}后文", "实体"), evidence)

    def test_html_references_and_whitespace_restore_the_original_span(self):
        cases = [
            ("<td>&lt; / +</td>", "< / +", "&lt; / +"),
            ("<td>3&nbsp;秒</td>", "3 秒", "3&nbsp;秒"),
            ("<td>A&amp;B</td>", "A&B", "A&amp;B"),
            ("前文&#x4E2D;&#25991;后文", "中文", "&#x4E2D;&#25991;"),
            ("关机\r\n\t  状态", "关机 状态", "关机\r\n\t  状态"),
            ("关机 状态", "关机\n状态", "关机 状态"),
            ("3 &nbsp;\t秒", "3 秒", "3 &nbsp;\t秒"),
        ]
        for content, evidence, expected in cases:
            with self.subTest(content=content, evidence=evidence):
                actual = kg._resolve_evidence(evidence, content, "实体")
                self.assertEqual(actual, expected)
                self.assertIn(actual, content)

    def test_table_second_entity_no_longer_fails_on_escaped_button_name(self):
        # 复现用户切片中的格式差异：第二个按键的原文是 &lt; / +，
        # 模型以显示文字 < / + 引用时，仍应保存可直接追溯的原文证据。
        chunk = make_chunk()
        chunk["content"] = (
            '## �）按键操作说明\n\n<table><tr><td>按键</td><td>状态描述</td></tr>'
            '<tr><td rowspan="3">返回/取消</td><td>短按可取消任务。</td></tr>'
            '<tr><td>&lt; / +</td><td>短按可切换菜单。</td></tr></table>'
        )
        data = {
            "entities": [
                {"id": "e1", "name": "返回/取消", "type": "Component", "evidence": "返回/取消"},
                {"id": "e2", "name": "< / +", "type": "Component", "evidence": "< / +"},
            ],
            "relations": [],
        }
        original = copy.deepcopy(chunk)
        llm = FakeLLM(data)
        result = kg.step_2_extract_entities_relations([chunk], llm=llm)
        self.assertEqual(result[0]["entities"][1]["evidence"], "&lt; / +")
        self.assertEqual(result[0]["entities"][1]["name"], "< / +")
        self.assertEqual(chunk, original)
        self.assertEqual(len(llm.calls), 1)

    def test_relations_restore_original_html_and_deduplicate_after_restoration(self):
        content = "<table><tr><td>&lt; / +</td><td>切换菜单</td></tr></table>"
        data = {
            "entities": [
                {"id": "e1", "name": "< / +", "type": "Component", "evidence": "< / +"},
                {"id": "e2", "name": "切换菜单", "type": "Operation", "evidence": "切换菜单"},
            ],
            "relations": [
                {"source": "e1", "target": "e2", "relation": "用于", "evidence": "< / +</td><td>切换菜单"},
                {"source": "e1", "target": "e2", "relation": "用于", "evidence": "&lt; / +</td><td>切换菜单"},
            ],
        }
        parsed = kg._parse_and_validate_extraction(json.dumps(data), content)
        self.assertEqual(len(parsed["relations"]), 1)
        self.assertEqual(parsed["relations"][0]["evidence"], "&lt; / +</td><td>切换菜单")
        prepared = kg._prepare_neo4j_data([{**make_chunk(), **parsed}])
        for row in prepared["mentions"] + prepared["relations"]:
            self.assertIn(row["evidence"], content)

    def test_format_matching_does_not_accept_rewritten_or_joined_evidence(self):
        cases = [
            ("禁止使用", "允许使用"),
            ("短按,开机", "短按，开机"),
            ("## �）按键说明", "## 一）按键说明"),
            ("<td>电源</td><td>关机</td>", "电源关机"),
            ("<td>电源</td><td>关机</td>", "电源 关机"),
            ("1 0V", "10V"),
            ("先开机。中间步骤。后关机。", "先开机。后关机。"),
            ("&NotEqualTilde;", "\u2242"),  # 一个引用展开为两字符，不能只匹配一半。
            ("&amp;lt;", "<"),             # 不做递归解码。
            ("没有这段文字", "&nbsp;"),
        ]
        for content, evidence in cases:
            with self.subTest(content=content, evidence=evidence):
                with self.assertRaisesRegex(ValueError, "原文中找到"):
                    kg._resolve_evidence(evidence, content, "第 2 个实体")

    def test_validation_retry_only_repeats_failed_chunk_with_error_feedback(self):
        bad = make_extraction()
        bad["entities"][1]["evidence"] = "模型改写的证据"
        llm = FakeLLM(make_extraction(), bad, make_extraction())
        chunks = [make_chunk(101), make_chunk(102)]
        original = copy.deepcopy(chunks)
        results = kg.step_2_extract_entities_relations(chunks, llm=llm)
        self.assertEqual([r["chunk_id"] for r in results], [101, 102])
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(llm.calls[1][:2], llm.calls[2][:2])
        feedback = json.loads(llm.calls[2][-1].content)
        self.assertIn("第 2 个实体", feedback["validation_error"])
        self.assertIn("模型改写的证据", feedback["validation_error"])
        self.assertIn("完整 JSON", feedback["instruction"])
        self.assertIn("HTML", llm.calls[0][0].content)
        self.assertEqual(chunks, original)

    def test_repeated_bad_evidence_still_aborts_without_publishing_or_writing(self):
        bad = make_extraction()
        bad["entities"][1]["evidence"] = "不存在的电池"
        llm = FakeLLM(make_extraction(), *([bad] * kg.KG_EXTRACTION_MAX_ATTEMPTS))
        state = create_default_state(task_id="kg_evidence_failed", chunks=[make_chunk(101), make_chunk(102)])
        original = copy.deepcopy(state)
        extract = kg.step_2_extract_entities_relations
        with (
            patch.object(kg, "step_2_extract_entities_relations", side_effect=lambda chunks: extract(chunks, llm=llm)),
            patch.object(kg, "step_3_save_to_neo4j") as save,
        ):
            with self.assertRaisesRegex(RuntimeError, "chunk_id=102.*第 2 个实体.*不存在的电池") as caught:
                kg.node_import_kg(state)
            save.assert_not_called()
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(len(llm.calls), 1 + kg.KG_EXTRACTION_MAX_ATTEMPTS)
        self.assertEqual(state, original)

    def test_transport_errors_are_not_retried_as_validation_failures(self):
        llm = FakeLLM(TimeoutError("模型连接超时"), make_extraction())
        with self.assertRaisesRegex(RuntimeError, "chunk_id=101") as caught:
            kg.step_2_extract_entities_relations([make_chunk()], llm=llm)
        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
        self.assertEqual(len(llm.calls), 1)

    def test_invalid_json_can_be_corrected_without_skipping_chunk(self):
        llm = FakeLLM('{"entities": [', make_extraction())
        results = kg.step_2_extract_entities_relations([make_chunk()], llm=llm)
        self.assertEqual(results[0]["relations"], make_extraction()["relations"])
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("JSON", json.loads(llm.calls[-1][-1].content)["validation_error"])


def reference_extraction(start="s1", end="s1"):
    data = make_extraction()
    for row in data["entities"] + data["relations"]:
        row["evidence"] = {"start": start, "end": end}
    return data


class KGEvidenceReferenceTests(unittest.TestCase):
    def test_sources_cover_all_characters_with_bounded_size_and_exact_offsets(self):
        cases = [
            "", "\r\n", "首行\r\n\n第二行\r尾部", "## �）操作说明\n\n",
            "<table><tr><td>&lt; / +</td></tr><TR><td>关机</td></TR></table>",
            "![图片说明](http://localhost:9000/" + "a" * 1800 + ".png)\n\n不可退换。",
        ]
        for content in cases:
            with self.subTest(content=content[:50]):
                sources = build_evidence_sources(content)
                self.assertEqual("".join(s["text"] for s in sources), content)
                self.assertEqual(len({s["id"] for s in sources}), len(sources))
                previous_end = 0
                for source in sources:
                    self.assertEqual(source["start"], previous_end)
                    self.assertEqual(source["text"], content[source["start"]:source["end"]])
                    self.assertTrue(0 < len(source["text"]) <= SOURCE_MAX_CHARS)
                    previous_end = source["end"]
                self.assertEqual(previous_end, len(content))
                self.assertEqual(sources, build_evidence_sources(content))

    def test_single_source_reference_is_resolved_for_entities_and_relations(self):
        content = make_chunk()["content"]
        data = reference_extraction()
        parsed = kg._parse_and_validate_extraction(json.dumps(data), content)
        self.assertEqual([row["evidence"] for row in parsed["entities"] + parsed["relations"]], [content] * 3)
        # 中间模型输出的编号对象不会进入第三步，存储协议仍然是字符串证据。
        prepared = kg._prepare_neo4j_data([{**make_chunk(), **parsed}])
        self.assertTrue(all(row["evidence"] == content for row in prepared["mentions"] + prepared["relations"]))

    def test_cross_image_reference_keeps_all_urls_and_intermediate_content(self):
        long_url = "http://localhost:9000/knowledge-base-files/" + "image-" * 180 + ".png"
        content = (
            "## 首次安装打印机\n\n随机物品(拆除包装后):\n\n"
            f"![彩色打印头说明]({long_url})\n\n"
            "注意：不能在通电时拆除部件。\n\n"
            "![数据线图片](http://localhost:9000/usb.png)\nUSB数据线\n"
            "包装外的其他内容。"
        )
        sources = build_evidence_sources(content)
        first = next(s for s in sources if s["text"] == "随机物品(拆除包装后):\n")
        last = next(s for s in sources if s["text"] == "USB数据线\n")
        data = reference_extraction(first["id"], last["id"])
        parsed = kg._parse_and_validate_extraction(json.dumps(data), content)
        evidence = parsed["relations"][0]["evidence"]
        expected = content[content.index("随机物品"):content.index("USB数据线\n") + len("USB数据线\n")]
        self.assertEqual(evidence, expected)
        self.assertIn(long_url, evidence)
        self.assertIn("不能在通电时拆除部件", evidence)
        self.assertEqual(evidence.count("!["), 2)
        self.assertNotIn("包装外的其他内容", evidence)

    def test_ellipsis_evidence_is_rejected_then_reference_retry_can_succeed(self):
        chunk = make_chunk()
        chunk["content"] = "示例设备配件：\n![说明](http://localhost:9000/a.png)\n电源开关\n"
        bad = make_extraction()
        bad["relations"][0]["evidence"] = "示例设备配件：\n...\n电源开关"
        llm = FakeLLM(bad, reference_extraction("s1", "s3"))
        results = kg.step_2_extract_entities_relations([chunk], llm=llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(results[0]["relations"][0]["evidence"], chunk["content"])
        feedback = json.loads(llm.calls[-1][-1].content)
        self.assertIn("...", feedback["validation_error"])
        self.assertIn("start/end", feedback["instruction"])

    def test_invalid_unknown_reversed_and_blank_references_are_rejected(self):
        content = "第一行\n\n第三行\n"
        cases = [
            {}, {"start": "s1"}, {"start": "s1", "end": "s99"},
            {"start": "e1", "end": "e2"}, {"start": 1, "end": 1},
            {"start": True, "end": False}, {"start": "s3", "end": "s1"},
            {"start": "s2", "end": "s2"}, ["s1", "s3"],
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                data = make_extraction()
                data["entities"][0]["evidence"] = evidence
                with self.assertRaisesRegex(ValueError, "第 1 个实体"):
                    kg._parse_and_validate_extraction(json.dumps(data), content)

    def test_model_supplied_source_mapping_cannot_create_evidence(self):
        content = make_chunk()["content"]
        data = reference_extraction("s99", "s99")
        data["evidence_sources"] = [{"id": "s99", "text": "编造的来源", "start": 0, "end": len(content)}]
        with self.assertRaisesRegex(ValueError, "原文编号"):
            kg._parse_and_validate_extraction(json.dumps(data), content)
        data = reference_extraction()
        data["entities"][0]["evidence"]["text"] = "模型试图覆盖原文"
        parsed = kg._parse_and_validate_extraction(json.dumps(data), content)
        self.assertEqual(parsed["entities"][0]["evidence"], content)

    def test_source_ids_are_resolved_against_each_chunk_and_sent_without_offsets(self):
        chunks = [make_chunk(101), make_chunk(102)]
        chunks[1]["content"] = "第二个切片的独立原文。"
        llm = FakeLLM(reference_extraction(), reference_extraction())
        results = kg.step_2_extract_entities_relations(chunks, llm=llm)
        for result, chunk, call in zip(results, chunks, llm.calls, strict=True):
            self.assertEqual(result["relations"][0]["evidence"], chunk["content"])
            sent = json.loads(call[1].content)
            self.assertEqual(sent["evidence_sources"], [{"id": "s1", "text": chunk["content"]}])
            self.assertNotIn("content", sent)
            self.assertNotIn("chunk_id", sent)

    def test_reference_failures_remain_bounded_and_do_not_write_partial_graph(self):
        state = create_default_state(task_id="kg_reference_failed", chunks=[make_chunk()])
        original = copy.deepcopy(state)
        llm = FakeLLM(*([reference_extraction("s999", "s999")] * kg.KG_EXTRACTION_MAX_ATTEMPTS))
        extract = kg.step_2_extract_entities_relations
        with (
            patch.object(kg, "step_2_extract_entities_relations", side_effect=lambda chunks: extract(chunks, llm=llm)),
            patch.object(kg, "step_3_save_to_neo4j") as save,
        ):
            with self.assertRaisesRegex(RuntimeError, "chunk_id=101.*原文编号"):
                kg.node_import_kg(state)
            save.assert_not_called()
        self.assertEqual(state, original)
        self.assertEqual(len(llm.calls), kg.KG_EXTRACTION_MAX_ATTEMPTS)

    def test_node_passes_resolved_original_evidence_to_existing_writer(self):
        state = create_default_state(task_id="kg_reference_success", chunks=[make_chunk()])
        llm = FakeLLM(reference_extraction())
        extract = kg.step_2_extract_entities_relations
        stats = {"products": 1, "chunks": 1, "entities": 2, "mentions": 2, "relations": 1}
        with (
            patch.object(kg, "step_2_extract_entities_relations", side_effect=lambda chunks: extract(chunks, llm=llm)),
            patch.object(kg, "step_3_save_to_neo4j", return_value=stats) as save,
        ):
            result = kg.node_import_kg(state)
        saved = save.call_args.args[0][0]
        self.assertEqual(saved["relations"][0]["evidence"], make_chunk()["content"])
        self.assertEqual(result["kg_import_stats"], stats)
        self.assertEqual(result["chunks"][0], make_chunk())


if __name__ == "__main__":
    unittest.main()
