"""HTML/Markdown 图片导入及旧切片链接兼容，不访问数据库或模型。"""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from app.utils import document_image_utils as links
from app.query_process.services.retrieval_service import merge_documents
from app.test.test_md_image_progress import images


CONFIG = SimpleNamespace(endpoint="localhost:9000", bucket_name="files", minio_img_dir="/upload-images", minio_secure=False)
TITLE = "B520 用户手册-(Windows11,zh-cn)"


class DocumentImageTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.object(links, "minio_config", CONFIG)
        self.config.start()
        self.addCleanup(self.config.stop)

    def test_html_table_and_markdown_references_are_both_found(self):
        text = '<table><tr><td><img src="images/panel.jpg"/></td><td>电源键</td></tr></table>\n![外观](images/panel.jpg)'
        refs = list(links.image_references(text))
        self.assertEqual([r.url for r in refs], ["images/panel.jpg", "images/panel.jpg"])
        contexts = images.find_image_in_md_content(text, "panel.jpg")
        self.assertEqual(len(contexts), 2)
        self.assertIn("电源键", contexts[0][1])

    def test_legacy_url_with_parentheses_is_not_truncated(self):
        url = f"http://localhost:9000/files/upload-images/{TITLE}/panel.jpg"
        text = f"![说明]({url})"
        self.assertEqual([r.url for r in links.image_references(text)], [url])
        normalized = links.normalize_document_images(text)
        self.assertIn("%28Windows11%2Czh-cn%29/panel.jpg", normalized)
        self.assertNotIn(" 用户手册", normalized)
        self.assertEqual(len(list(links.image_references(normalized))), 1)

    def test_relative_images_are_scoped_by_document_and_html_is_preserved(self):
        html = '<table><tr><td><img src="images/panel.jpg" width="200"/></td><td>电源键</td></tr></table>'
        url = links.document_image_url(TITLE, "panel.jpg")
        self.assertEqual(links.normalize_document_images(html, TITLE), html.replace("images/panel.jpg", url))
        self.assertNotEqual(url, links.document_image_url("其他型号", "panel.jpg"))
        self.assertEqual(links.normalize_document_images(html), html)

    def test_http_urls_are_encoded_once_and_keep_query_parameters(self):
        url = "https://example.com/用户手册(一)/图.jpg?token=abc%2Fdef&v=1"
        resolved = links.resolve_image_url(url)
        self.assertIn("%28", resolved)
        self.assertTrue(resolved.endswith("?token=abc%2Fdef&v=1"))
        self.assertEqual(links.resolve_image_url(resolved), resolved)
        self.assertEqual(links.resolve_image_url("./images/panel.jpg", TITLE), links.document_image_url(TITLE, "panel.jpg"))

    def test_unknown_or_unsafe_relative_paths_are_not_exposed(self):
        for value in ["images/panel.jpg", "../panel.jpg", "javascript:alert(1)", "//example.com/a.jpg", "http://[bad/image.jpg"]:
            with self.subTest(value=value):
                self.assertIsNone(links.resolve_image_url(value))
        for value in ["images/../panel.jpg", "images/%2e%2e/panel.jpg"]:
            self.assertIsNone(links.resolve_image_url(value, TITLE))

    def test_scan_does_not_upload_unreferenced_images(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ["panel.jpg", "unused.jpg", "unsupported.txt"]:
                (folder / name).write_bytes(b"fixture")
            targets = images.step_2_scan_images('<table><img src="images/panel.jpg"/></table>', folder)
        self.assertEqual([item[0] for item in targets], ["panel.jpg"])

    def test_import_upload_key_and_render_url_match_and_preserve_table(self):
        client = MagicMock()
        client.list_objects.return_value = []
        client.remove_objects.return_value = []
        html = '<table><tr><td><img src="images/panel.jpg"/></td><td>电源键</td></tr></table>\n![原说明](images/panel.jpg)'
        with patch.object(images, "get_minio_client", return_value=client), patch.object(images, "minio_config", CONFIG):
            result = images.step_4_upload_and_replace_images(
                [("panel.jpg", "/tmp/panel.jpg", ("", ""))], {"panel.jpg": "面板[图]"}, TITLE, html,
            )
        key = client.fput_object.call_args.kwargs["object_name"]
        self.assertEqual(key, f"upload-images/{TITLE}/panel.jpg")
        url = links.document_image_url(TITLE, "panel.jpg")
        self.assertIn(f'<img src="{url}"/>', result)
        self.assertIn("<td>电源键</td>", result)
        self.assertIn("![面板\\[图\\]]", result)
        self.assertEqual([ref.url for ref in links.image_references(result)], [url, url])
        self.assertFalse(key.startswith("/"))

    def test_retrieval_normalizes_legacy_content_without_mutating_hits(self):
        hit = {"id": 1, "entity": {"chunk_id": 1, "content": '<img src="images/panel.jpg"/>', "file_title": TITLE, "item_name": "B520"}}
        original = copy.deepcopy(hit)
        result = merge_documents([hit], [])
        self.assertIn(links.document_image_url(TITLE, "panel.jpg"), result[0]["text"])
        self.assertEqual(hit, original)
        self.assertEqual(result[0]["file_title"], TITLE)


if __name__ == "__main__":
    unittest.main()
