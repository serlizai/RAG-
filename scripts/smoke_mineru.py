"""显式把指定 PDF 发到已配置的 MinerU，验证解析与下载；不写入知识库。"""

import argparse
import os
from pathlib import Path
import tempfile
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args(argv)
    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        parser.error("请指定存在的 PDF 文件")
    if not os.getenv("MINERU_BASE_URL") or not os.getenv("MINERU_API_TOKEN"):
        parser.error("缺少 MINERU_BASE_URL / MINERU_API_TOKEN")

    from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
    from app.import_process.agent.state import create_default_state
    from app.utils.task_utils import clear_task

    task_id = "mineru-smoke-" + uuid.uuid4().hex
    try:
        with tempfile.TemporaryDirectory(prefix="rag-mineru-smoke-") as directory:
            state = create_default_state(
                task_id=task_id, local_file_path=str(pdf), pdf_path=str(pdf),
                local_dir=directory, file_title=pdf.stem,
            )
            result = node_pdf_to_md(state)
            content = result.get("md_content") or ""
            if not content.strip() or not Path(result.get("md_path") or "").is_file():
                print("FAIL: MinerU 未返回有效的 Markdown 文件。")
                return 1
            print(f"PASS: MinerU 上传、解析、下载通过，Markdown {len(content)} 字符。")
            return 0
    except Exception as exc:
        print(f"FAIL: MinerU 解析失败（{type(exc).__name__}）；请检查凭证、服务地址和可用额度。")
        return 1
    finally:
        clear_task(task_id)


if __name__ == "__main__":
    raise SystemExit(main())
