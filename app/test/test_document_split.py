"""文档切分回归测试：检查切片顺序、合并内容及本地备份，不访问外部服务。"""

import json
import tempfile
import unittest
from pathlib import Path

from app.import_process.agent.nodes import node_document_split as split
from app.import_process.agent.state import create_default_state
from app.utils.task_utils import clear_task


def section(title, content, parent_title=None, part=1):
    result = {"title": title, "content": content, "file_title": "测试手册"}
    if parent_title is not None:
        result.update(parent_title=parent_title, part=part)
    return result


class DocumentSplitTests(unittest.TestCase):
    def test_unmerged_chunks_are_kept_once_in_order(self):
        cases = [
            [section(name, name * 600) for name in ("A", "B", "C")],
            # 缺少父标题的短章节也必须各自保留，不能因为 None == None 合并。
            [section(name, name * 100) for name in ("A", "B", "C")],
            # 不同父标题，以及恰好达到合并阈值的块，都不应合并。
            [section(name, name * 100, parent_title=name) for name in ("A", "B", "C")],
            [section(name, name * 500, parent_title="同一章节") for name in ("A", "B", "C")],
        ]
        for chunks in cases:
            with self.subTest(chunks=[(c["title"], len(c["content"])) for c in chunks]):
                self.assertEqual(split.merge_short_sections(chunks), chunks)

    def test_merged_groups_are_kept_when_the_parent_changes(self):
        chunks = [
            section("A_1", "甲" * 100, "A", 1),
            section("A_2", "乙" * 100, "A", 2),
            section("B_1", "丙" * 100, "B", 1),
            section("B_2", "丁" * 500, "B", 2),
            section("C", "戊" * 600, "C"),
        ]
        result = split.merge_short_sections(chunks)
        self.assertEqual([c["title"] for c in result], ["A_1", "B_1", "C"])
        self.assertEqual([c["content"] for c in result], [
            "甲" * 100 + "\n\n" + "乙" * 100,
            "丙" * 100 + "\n\n" + "丁" * 500,
            "戊" * 600,
        ])
        self.assertEqual([c["parent_title"] for c in result], ["A", "B", "C"])

    def test_empty_single_and_final_merged_group_are_retained(self):
        self.assertEqual(split.merge_short_sections([]), [])
        single = [section("唯一章节", "唯一内容")]
        self.assertEqual(split.merge_short_sections(single), single)
        chunks = [
            section("A", "甲" * 600, "A"),
            section("B_1", "乙" * 100, "B", 1),
            section("B_2", "丙" * 100, "B", 2),
        ]
        result = split.merge_short_sections(chunks)
        self.assertEqual([c["content"] for c in result], [
            "甲" * 600, "乙" * 100 + "\n\n" + "丙" * 100,
        ])

    def test_only_long_sections_are_subdivided_and_all_text_is_retained(self):
        short = section("## 短章节", "短" * 600)
        long = section("## 长章节", "长" * 3500)
        sections = split.step_4_refine_chunks([short, long])
        self.assertEqual([c["title"] for c in sections], ["## 短章节", "## 长章节_1", "## 长章节_2"])
        self.assertEqual(sections[0]["content"], short["content"])
        self.assertEqual([len(c["content"]) for c in sections], [600, 2000, 1600])
        # 此无分隔符样本的相邻子块重叠 100 字符；扣除重叠后等于完整长章节。
        self.assertEqual(sections[1]["content"] + sections[2]["content"][100:], long["content"])
        self.assertEqual([c["parent_title"] for c in sections], ["## 短章节", "## 长章节", "## 长章节"])
        self.assertEqual([c["part"] for c in sections], [1, 1, 2])

    def test_node_and_backup_preserve_preamble_and_each_heading(self):
        task_id = "document_split_regression"
        self.addCleanup(clear_task, task_id)
        with tempfile.TemporaryDirectory() as directory:
            state = create_default_state(
                task_id=task_id,
                local_dir=directory,
                file_title="测试手册",
                md_content="前言说明\n# 第一章\n第一章正文\n## 第二章\n第二章正文\n## 第三章\n第三章正文",
            )
            result = split.node_document_split(state)
            chunks = result["chunks"]
            self.assertEqual([c["title"] for c in chunks], ["", "# 第一章", "## 第二章", "## 第三章"])
            self.assertEqual([c["content"] for c in chunks], [
                "前言说明", "# 第一章\n第一章正文", "## 第二章\n第二章正文", "## 第三章\n第三章正文",
            ])
            self.assertEqual(json.loads((Path(directory) / "chunks.json").read_text(encoding="utf-8")), chunks)


if __name__ == "__main__":
    unittest.main()
