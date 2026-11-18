"""手动导入演示；导入模块或收集测试时不运行工作流。"""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="需要导入的本地 PDF；会调用外部服务并写入知识库")
    args = parser.parse_args(argv)
    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file():
        parser.error("指定的 PDF 文件不存在")

    from app.core.logger import logger
    from app.import_process.agent.main_graph import kb_import_app
    from app.import_process.agent.state import create_default_state
    from app.utils.path_util import PROJECT_ROOT
    import uuid

    task_id = "import-demo-" + uuid.uuid4().hex
    output = PROJECT_ROOT / "output" / task_id
    output.mkdir(parents=True, exist_ok=True)
    state = create_default_state(task_id=task_id, local_file_path=str(pdf), local_dir=str(output))
    result = kb_import_app.invoke(state)
    logger.info("导入完成，图谱统计：{}", json.dumps(result.get("kg_import_stats", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
