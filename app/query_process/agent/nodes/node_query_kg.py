"""旧查询图的知识图谱节点适配层；实际检索位于 kg_service。"""

from app.clients.milvus_utils import get_milvus_client
from app.clients.neo4j_utils import get_neo4j_driver
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.query_process.services import kg_service
from app.query_process.services.kg_service import (
    KG_CHUNK_LIMIT, KG_QUERY_TIMEOUT, KG_RECALL_QUERY, KG_SEED_LIMIT,
)
from app.utils.task_utils import add_done_task, add_running_task


def step_1_prepare_kg_query(state: QueryGraphState):
    """兼容旧节点的改写问题优先、原问题兜底规则。"""
    query = state.get("rewritten_query")
    if not isinstance(query, str) or not query.strip():
        query = state.get("original_query") or ""
    return kg_service.prepare_kg_query(query, state.get("item_names") or [])


def step_2_query_chunk_ids(query, item_names, *, driver=None, database=None):
    """保留旧辅助函数名，查询逻辑统一交给服务。"""
    if not query or not item_names:
        return []
    return kg_service.query_chunk_ids(
        query, item_names,
        driver=driver if driver is not None else get_neo4j_driver(),
        database=database,
    )


def step_3_fetch_kg_chunks(candidates, item_names, *, client=None, collection_name=None, errors=None):
    """保留旧辅助函数名和列表输出，供原查询图与测试继续使用。"""
    if not any(row.get("item_name") in set(item_names) for row in candidates):
        return []
    return kg_service.fetch_kg_chunks(
        candidates, item_names,
        client=client if client is not None else get_milvus_client(),
        collection_name=collection_name or milvus_config.chunks_collection,
        errors=errors,
    )


def node_query_kg(state: QueryGraphState):
    """旧图只接收 kg_chunks；异常详情由服务保留给未来的 Agent 工具。"""
    node_name = "node_query_kg"
    add_running_task(state["session_id"], node_name, state.get("is_stream"))
    try:
        query, item_names = step_1_prepare_kg_query(state)
        result = kg_service.search_knowledge_graph(
            query, item_names,
            query_chunk_ids_fn=step_2_query_chunk_ids,
            fetch_kg_chunks_fn=step_3_fetch_kg_chunks,
        )
        # 并行节点只返回自己更新的字段，不覆盖其他召回节点的 state。
        return {"kg_chunks": result.items}
    except Exception as exc:
        logger.warning(f"知识图谱节点适配失败，本次按空结果继续：{type(exc).__name__}")
        return {"kg_chunks": []}
    finally:
        add_done_task(state["session_id"], node_name, state.get("is_stream"))
