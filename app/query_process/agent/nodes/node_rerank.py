"""Legacy graph adapter for document merging and reranking."""

from app.core.logger import logger
from app.query_process.services.retrieval_service import (
    RERANK_GAP_ABS,
    RERANK_GAP_RATIO,
    RERANK_MAX_TOPK,
    RERANK_MIN_TOPK,
    merge_documents,
    rerank_documents,
    score_documents,
    select_top_k,
)
from app.utils.task_utils import add_done_task, add_running_task


def step_1_merge_docs(state):
    return merge_documents(state.get("rrf_chunks"), state.get("web_search_docs"))


def step_2_rerank_docs(state, doc_items):
    query = state.get("rewritten_query") or state.get("original_query")
    return score_documents(query, doc_items)


def step_3_topk(scored_docs):
    return select_top_k(scored_docs)


def node_rerank(state):
    session_id = state["session_id"]
    is_stream = state.get("is_stream")
    add_running_task(session_id, "node_rerank", is_stream)
    try:
        query = state.get("rewritten_query") or state.get("original_query")
        result = rerank_documents(
            query, state.get("rrf_chunks"), state.get("web_search_docs"),
        )
        if result.status == "error":
            logger.warning("文档重排未完成: {}", result.error_code)
            # 旧图的重排异常原本会终止本轮查询，不能悄悄当作零证据。
            raise RuntimeError(f"文档重排失败: {result.error_code}")
        return {"reranked_docs": result.items}
    finally:
        add_done_task(session_id, "node_rerank", is_stream)
