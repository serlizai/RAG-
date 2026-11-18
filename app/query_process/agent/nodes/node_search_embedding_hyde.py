"""Legacy graph adapter for HyDE vector search."""

from app.core.logger import logger
from app.query_process.services import retrieval_service
from app.utils.task_utils import add_done_task, add_running_task


def step_1_create_hyde_doc(rewritten_query):
    """Compatibility entry point for callers of the former node helper."""
    return retrieval_service.generate_hypothetical_document(rewritten_query)


def step_2_search_embedding_hyde(
    rewritten_query, hyde_doc, item_names, req_limit=10, top_k=5,
    ranker_weights=(0.8, 0.2), norm_score=True,
    output_fields=None,
):
    """Compatibility entry point; ranking remains the old Milvus default."""
    result = retrieval_service.search_embedding_hyde(
        rewritten_query, item_names, hyde_doc=hyde_doc, output_fields=output_fields,
    )
    return result.items


def node_search_embedding_hyde(state):
    session_id = state["session_id"]
    is_stream = state.get("is_stream")
    add_running_task(session_id, "node_search_embedding_hyde", is_stream)
    try:
        result = retrieval_service.search_embedding_hyde(
            state.get("rewritten_query"), state.get("item_names"),
        )
        if result.status == "error":
            logger.warning("HyDE 检索未完成: {}", result.error_code)
            if result.error_code == "HYDE_GENERATION_FAILED":
                # 旧节点的模型生成异常会向上传播；数据库召回异常仍按空路继续。
                raise RuntimeError("HyDE 假设文档生成失败")
        return {"hyde_embedding_chunks": result.items}
    finally:
        add_done_task(session_id, "node_search_embedding_hyde", is_stream)
