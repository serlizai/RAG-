"""Legacy graph adapter for local recall fusion."""

from app.query_process.services.retrieval_service import (
    fuse_local_results,
    reciprocal_rank_fusion,
)
from app.utils.task_utils import add_done_task, add_running_task


def node_rrf(state):
    session_id = state["session_id"]
    is_stream = state.get("is_stream")
    add_running_task(session_id, "node_rrf", is_stream)
    try:
        result = fuse_local_results(
            state.get("embedding_chunks"),
            state.get("hyde_embedding_chunks"),
            state.get("kg_chunks"),
        )
        return {"rrf_chunks": result.items}
    finally:
        add_done_task(session_id, "node_rrf", is_stream)
