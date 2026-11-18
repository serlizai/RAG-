"""旧查询图的联网检索节点，负责进度与旧状态格式的适配。"""

import asyncio

from app.core.logger import logger
from app.query_process.services.web_service import call_mcp_raw, parse_mcp_result
from app.utils.task_utils import add_done_task, add_running_task


async def mcp_call_streamable(query):
    """兼容旧调用接口：失败时返回 None，不阻断其余召回通道。"""
    try:
        return await call_mcp_raw(query)
    except Exception as exc:
        logger.warning(f"联网召回失败，本次按空结果继续：{type(exc).__name__}")
        return None


def node_web_search_mcp(state):
    """将异步服务结果包装为旧图使用的 `web_search_docs`。"""
    node_name = "node_web_search_mcp"
    add_running_task(state["session_id"], node_name, state.get("is_stream", False))
    try:
        query = state.get("rewritten_query") or state.get("original_query") or ""
        raw = asyncio.run(mcp_call_streamable(query))
        result = parse_mcp_result(raw)
        if result.status == "error":
            logger.warning(f"联网结果不可用：{result.error_code}")
        logger.info(f"联网召回完成：原始候选 {result.candidate_count} 条，有效 {len(result.items)} 条")
        return {"web_search_docs": result.items}
    finally:
        add_done_task(state["session_id"], node_name, state.get("is_stream", False))
