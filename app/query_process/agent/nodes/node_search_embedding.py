"""Legacy graph adapter for session-independent vector search."""

from app.core.logger import logger
from app.query_process.services.retrieval_service import search_embedding
from app.utils.task_utils import add_done_task, add_running_task


def node_search_embedding(state):
    session_id = state["session_id"]
    is_stream = state.get("is_stream")
    add_running_task(session_id, "node_search_embedding", is_stream)
    try:
        result = search_embedding(state.get("rewritten_query"), state.get("item_names"))
        if result.status == "error":
            logger.warning("向量检索未完成: {}", result.error_code)
        return {"embedding_chunks": result.items}
    finally:
        add_done_task(session_id, "node_search_embedding", is_stream)
