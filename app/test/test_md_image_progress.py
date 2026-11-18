"""图片节点进度回归：跳过、失败及成功均结束进度，不访问外部服务。"""

import importlib.util
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from app.import_process.agent.state import create_default_state
from app.utils.task_utils import (
    add_running_task,
    clear_task,
    get_done_task_list,
    get_running_task_list,
)


# 单独加载被测节点，避免导入 MinIO 客户端时创建存储桶或设置访问策略。
_minio_stub = types.ModuleType("app.clients.minio_utils")
_minio_stub.get_minio_client = lambda: None
_source = Path(__file__).parents[1] / "import_process" / "agent" / "nodes" / "node_md_img.py"
_spec = importlib.util.spec_from_file_location("md_image_progress_under_test", _source)
images = importlib.util.module_from_spec(_spec)
_missing = object()
_previous_minio = sys.modules.get("app.clients.minio_utils", _missing)
sys.modules["app.clients.minio_utils"] = _minio_stub
try:
    _spec.loader.exec_module(images)
finally:
    # 仅恢复这一项，保留节点加载期间正常导入的第三方模块缓存。
    if _previous_minio is _missing:
        sys.modules.pop("app.clients.minio_utils", None)
    else:
        sys.modules["app.clients.minio_utils"] = _previous_minio


class MarkdownImageProgressTests(unittest.TestCase):
    def state(self, **overrides):
        task_id = "md_image_progress_" + uuid.uuid4().hex
        self.addCleanup(clear_task, task_id)
        # 图片节点结束不应清除其他节点的进度。
        add_running_task(task_id, "node_document_split")
        return create_default_state(task_id=task_id, **overrides)

    def assert_image_finished(self, state):
        task_id = state["task_id"]
        self.assertEqual(get_running_task_list(task_id), ["文档切分"])
        self.assertEqual(get_done_task_list(task_id), ["Markdown图片处理"])

    def test_missing_images_folder_finishes_progress_and_preserves_text(self):
        with tempfile.TemporaryDirectory() as directory:
            md_path = Path(directory) / "manual.md"
            md_path.write_text("# 测试手册\n正文内容", encoding="utf-8")
            state = self.state(md_path=str(md_path))

            result = images.node_md_img(state)

            self.assertIs(result, state)
            self.assertEqual(result["md_content"], md_path.read_text(encoding="utf-8"))
            self.assert_image_finished(state)

    def test_input_validation_errors_propagate_and_finish_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [("", ValueError), (str(Path(directory) / "missing.md"), FileNotFoundError)]
            for md_path, error in cases:
                with self.subTest(md_path=md_path):
                    state = self.state(md_path=md_path)
                    with self.assertRaises(error):
                        images.node_md_img(state)
                    self.assert_image_finished(state)

    def test_processing_error_propagates_and_finishes_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            md_path = Path(directory) / "manual.md"
            md_path.write_text("原始正文", encoding="utf-8")
            (Path(directory) / "images").mkdir()
            state = self.state(md_path=str(md_path))
            failure = RuntimeError("图片模型调用失败")

            with patch.object(images, "step_3_generate_img_summaries", side_effect=failure):
                with self.assertRaises(RuntimeError) as caught:
                    images.node_md_img(state)

            self.assertIs(caught.exception, failure)
            self.assertEqual(md_path.read_text(encoding="utf-8"), "原始正文")
            self.assert_image_finished(state)

    def test_success_saves_processed_markdown_and_finishes_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            md_path = Path(directory) / "manual.md"
            md_path.write_text("原始正文", encoding="utf-8")
            (Path(directory) / "images").mkdir()
            state = self.state(md_path=str(md_path))
            processed = "处理后的正文与图片链接"

            with (
                patch.object(images, "step_3_generate_img_summaries", return_value={}),
                patch.object(images, "step_4_upload_and_replace_images", return_value=processed),
            ):
                result = images.node_md_img(state)

            self.assertEqual(result["md_content"], processed)
            self.assertEqual(Path(result["md_path"]).read_text(encoding="utf-8"), processed)
            self.assertEqual(md_path.read_text(encoding="utf-8"), "原始正文")
            self.assert_image_finished(state)


if __name__ == "__main__":
    unittest.main()
