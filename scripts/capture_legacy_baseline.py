"""显式调用旧查询图，保存固定验收问题的当前回答。

用法：.venv/bin/python -m scripts.capture_legacy_baseline --run-live \
    --local-only --product-a '库内实际商品名' --case-id product_operation

此脚本会访问配置的模型和数据库，并在 MongoDB 中保存测试会话。
旧查询图在商品匹配成功后还会将问题发送至外部联网搜索服务。
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from typing import Callable
from unittest.mock import patch
import uuid


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "app/test/fixtures/agent_acceptance_cases.json"
DEFAULT_OUTPUT = ROOT / "docs/agent_legacy_baseline_results.json"


async def _no_external_search(_query: str):
    """本地基线模式下返回空联网结果，不连接任何外部搜索服务。"""
    return None


@contextmanager
def external_search_context(local_only: bool):
    """隔离旧图的外部搜索调用；仅用于显式的本地基线模式。"""
    if not local_only:
        yield
        return
    web_module = importlib.import_module("app.query_process.agent.nodes.node_web_search_mcp")
    with patch.object(web_module, "mcp_call_streamable", _no_external_search):
        yield


def render_cases(cases: list[dict], product_a: str | None, product_b: str | None) -> list[dict]:
    """填入真实商品；缺少必需商品时标记跳过，不能把占位符发给模型。"""
    rendered = []
    for case in cases:
        required = case["requires_products"]
        missing = (required >= 1 and not product_a) or (required >= 2 and not product_b)
        item = {**case, "status": "skipped" if missing else "pending"}
        if missing:
            item["reason"] = "运行时未提供此场景所需的商品名称"
            item["turns"] = []
        else:
            item["turns"] = [
                turn.format(product_a=product_a or "", product_b=product_b or "")
                for turn in case["turns"]
            ]
        rendered.append(item)
    return rendered


def capture(cases: list[dict], on_progress: Callable[[], None] | None = None) -> list[dict]:
    """每个场景独立会话；一个场景中的追问共用 session_id。"""
    from app.query_process.agent.main_graph import query_app
    from app.query_process.agent.state import create_query_default_state

    for case in cases:
        if case["status"] == "skipped":
            continue
        print(f"开始场景：{case['id']}", flush=True)
        session_id = f"baseline-{uuid.uuid4().hex}"
        case["session_id"] = session_id
        case["results"] = []
        case["status"] = "running"
        if on_progress:
            on_progress()
        for turn in case["turns"]:
            try:
                state = create_query_default_state(session_id=session_id, original_query=turn)
                final_state = query_app.invoke(state)
                case["results"].append({
                    "query": turn,
                    "answer": final_state.get("answer", ""),
                    "item_names": final_state.get("item_names", []),
                    "answer_verified": final_state.get("answer_verified", False),
                })
            except Exception as exc:
                case["results"].append({
                    "query": turn,
                    "error_type": type(exc).__name__,
                })
                case["status"] = "failed"
                if on_progress:
                    on_progress()
                break
            if on_progress:
                on_progress()
        if case["status"] == "running":
            case["status"] = "completed"
            if on_progress:
                on_progress()
        print(f"场景结束：{case['id']} ({case['status']})", flush=True)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="明确同意访问模型和数据库")
    search_mode = parser.add_mutually_exclusive_group()
    search_mode.add_argument(
        "--allow-external-search", action="store_true",
        help="明确允许旧查询图将商品问题发送到外部联网搜索服务",
    )
    search_mode.add_argument(
        "--local-only", action="store_true",
        help="将旧图的外部搜索调用替换为空结果；输出标记为 local_only",
    )
    parser.add_argument("--case-id", action="append", help="只运行指定场景，可重复传入")
    parser.add_argument("--product-a", help="库内实际存在的商品名称")
    parser.add_argument("--product-b", help="第二个库内实际商品名称")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.run_live:
        parser.error("此脚本会真实调用模型和数据库，请显式传入 --run-live")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    known_ids = {case["id"] for case in cases}
    unknown_ids = set(args.case_id or []) - known_ids
    if unknown_ids:
        parser.error(f"未知场景: {', '.join(sorted(unknown_ids))}")
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected_ids]

    results = render_cases(cases, args.product_a, args.product_b)
    if not (args.allow_external_search or args.local_only) and any(
        case["status"] == "pending" and case["requires_products"] >= 1
        for case in results
    ):
        parser.error("商品场景可能调用外部联网搜索，请显式传入 --allow-external-search")
    output = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "legacy",
        "search_mode": "local_only" if args.local_only else "normal",
        "cases": results,
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    with external_search_context(args.local_only):
        capture(results, on_progress=save)
    save()
    print(f"已保存 {len(results)} 个场景：{args.output}")


if __name__ == "__main__":
    main()
