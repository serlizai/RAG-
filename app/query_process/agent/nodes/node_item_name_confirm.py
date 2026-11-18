"""旧查询图的商品确认节点；解析逻辑位于 product_service。"""

import json
import sys

from app.core.logger import logger
from app.query_process.services.product_service import (
    align_product_names as step_5_align_item_names,
    extract_product_info as step_3_extract_info,
    resolve_product,
    search_matching_products as step_4_vectorize_and_query,
)
from app.utils.task_utils import add_done_task, add_running_task


def get_recent_messages(*args, **kwargs):
    """将 Mongo 导入留在旧节点的运行路径，离线导入服务时无需连接。"""
    from app.clients.mongo_history_utils import get_recent_messages as load_messages

    return load_messages(*args, **kwargs)


def save_chat_message(*args, **kwargs):
    from app.clients.mongo_history_utils import save_chat_message as persist_message

    return persist_message(*args, **kwargs)


def step_6_deal_list(state, align_result, history, rewritten_query):
    """将服务返回的商品名称映射到旧图状态及澄清回答。"""
    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options_item_names", [])
    if confirmed:
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        state["history"] = history
        state.pop("answer", None)
    elif options:
        state["answer"] = f"您是想问以下哪个产品：{'、'.join(options)}？请明确一下型号"
    else:
        state["answer"] = "抱歉，我没有找到您要查询的商品。"
    return state


def node_item_name_confirm(state):
    """读取/保存旧会话，调用无会话副作用的商品解析服务。"""
    node_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], node_name, state["is_stream"])
    try:
        history = get_recent_messages(session_id=state["session_id"], limit=10)
        message_id = save_chat_message(
            session_id=state["session_id"], role="user", text=state["original_query"]
        )
        logger.debug("Node: 用户消息已初始保存, ID: {}", message_id)

        result = resolve_product(state["original_query"], history)
        if result.status == "error":
            raise RuntimeError(f"商品名称解析失败: {result.error_code}")

        state = step_6_deal_list(
            state,
            {
                "confirmed_item_names": result.confirmed_item_names,
                "options_item_names": result.option_item_names,
            },
            history,
            result.rewritten_query,
        )
        # 澄清回答仍由 node_answer_output 统一输出并保存。
        save_chat_message(
            session_id=state["session_id"],
            role="user",
            text=state["original_query"],
            rewritten_query=result.rewritten_query,
            item_names=state.get("item_names", []),
            message_id=message_id,
        )
        return state
    finally:
        add_done_task(state["session_id"], node_name, state["is_stream"])


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "万用表RS-12怎么用？",
        "is_stream": False,
    }
    try:
        result_state = node_item_name_confirm(mock_state)
        print(json.dumps(result_state, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[FAIL] 测试运行出错: {exc}")
