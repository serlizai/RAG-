"""知识图谱召回服务：Neo4j 定位切片，Milvus 回查可引用的原文。"""

import os
from collections.abc import Callable

from neo4j import Query, READ_ACCESS

from app.clients.milvus_utils import fetch_chunks_by_chunk_ids
from app.clients.milvus_utils import get_milvus_client
from app.clients.neo4j_utils import get_neo4j_driver
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.query_process.services.schemas import Evidence, SearchResult


KG_SEED_LIMIT = 10       # 最多从问题中匹配 10 个实体，避免展开过多节点
KG_CHUNK_LIMIT = 20      # 最多回查 20 个切片，后续仍由 RRF 和 Reranker 筛选
KG_QUERY_TIMEOUT = 5.0  # Neo4j 查询执行超时；不含驱动建立连接的时间

# 起点实体及关系证据必须属于已确认商品。商品名只限定范围，不作为起点。
# 先排直接命中，再排相邻实体；同层按命中数和 chunk_id 稳定排序。
# 所有用户输入都通过参数传入，不拼接进查询文本。
KG_RECALL_QUERY = """
MATCH (seed:Entity)
WHERE seed.item_name IN $item_names AND seed.type <> 'Product'
WITH seed, replace(toLower(trim(seed.name)), ' ', '') AS entity_name
WHERE size(entity_name) >= 2 AND $query CONTAINS entity_name
WITH seed
ORDER BY size(seed.name) DESC, seed.entity_key ASC
LIMIT $seed_limit
MATCH path = (seed)-[:RELATED_TO*0..1]-(entity:Entity)
WHERE all(node IN nodes(path)
          WHERE node.item_name = seed.item_name AND node.type <> 'Product')
  AND all(rel IN relationships(path)
          WHERE rel.item_name = seed.item_name AND EXISTS {
              MATCH (proof:Chunk {chunk_id: rel.chunk_id})
              WHERE proof.item_name = seed.item_name
          })
MATCH (:Product {name: seed.item_name})-[:HAS_CHUNK]->(chunk:Chunk)-[:MENTIONS]->(entity)
WHERE chunk.item_name = seed.item_name
WITH chunk, seed, min(length(path)) AS hops
WITH chunk, min(hops) AS hop_count, count(*) AS matched_entities
RETURN chunk.chunk_id AS chunk_id, chunk.item_name AS item_name,
       hop_count, matched_entities
ORDER BY hop_count ASC, matched_entities DESC, chunk_id ASC
LIMIT $chunk_limit
"""


def prepare_kg_query(query: str, item_names: list[str] | tuple[str, ...]):
    """规范化实体匹配问题和商品范围，不读取图状态。"""
    normalized_query = "".join(query.lower().split()) if isinstance(query, str) else ""
    if not isinstance(item_names, (list, tuple)):
        return normalized_query, []
    names = list(dict.fromkeys(
        name.strip() for name in item_names if isinstance(name, str) and name.strip()
    ))
    return normalized_query, names


def query_chunk_ids(query, item_names, *, driver=None, database=None):
    """只读查询有序的切片标识和商品归属，不把实体名称当作回答原文。"""
    if not query or not item_names:
        return []
    if driver is None:
        driver = get_neo4j_driver()
    database = database or os.getenv("NEO4J_DATABASE") or "neo4j"
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        # 在 session 关闭前消费结果；全局复用的 driver 不由这里关闭。
        return session.run(
            Query(KG_RECALL_QUERY, timeout=KG_QUERY_TIMEOUT),
            parameters={
                "query": query, "item_names": item_names,
                "seed_limit": KG_SEED_LIMIT, "chunk_limit": KG_CHUNK_LIMIT,
            },
        ).data()


def fetch_kg_chunks(candidates, item_names, *, client=None, collection_name=None, errors=None):
    """按图谱排名回查原文，并包装为旧 RRF 使用的 {id, entity} 结构。

    `errors` 是可选的批次失败收集器；旧调用仍只得到切片列表。
    """
    # 二次检查商品归属，防止图谱中的旧 ID 指向另一商品的原文。
    allowed_names = set(item_names)
    candidates = [row for row in candidates if row.get("item_name") in allowed_names]
    if not candidates:
        return []
    if client is None:
        client = get_milvus_client()
    if client is None:
        raise ConnectionError("Milvus client unavailable")
    collection_name = collection_name or milvus_config.chunks_collection
    if not collection_name:
        raise ValueError("Milvus collection unavailable")
    expected_items = {row["chunk_id"]: row["item_name"] for row in candidates}
    lookup_kwargs = {
        "output_fields": ["chunk_id", "content", "item_name", "title", "file_title"],
    }
    if errors is not None:
        lookup_kwargs["on_error"] = lambda exc, batch: errors.append((type(exc).__name__, tuple(batch)))
    entities = fetch_chunks_by_chunk_ids(
        client, collection_name, list(expected_items), **lookup_kwargs,
    )
    chunks = []
    for entity in entities:
        chunk_id = entity["chunk_id"]
        content = entity.get("content")
        if entity.get("item_name") != expected_items.get(chunk_id):
            logger.warning(f"图谱与 Milvus 的商品归属不一致，跳过切片 {chunk_id}")
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        # 不伪造 distance：图谱使用自身排名，RRF 只看排名。
        chunks.append({"id": chunk_id, "entity": entity})
    if len(chunks) < len(expected_items):
        logger.warning(f"图谱候选 {len(expected_items)} 条，成功回查原文 {len(chunks)} 条；请检查两库数据是否同步")
    return chunks


def search_knowledge_graph(
    query: str,
    item_names: list[str] | tuple[str, ...],
    *,
    driver=None,
    client=None,
    database=None,
    collection_name=None,
    query_chunk_ids_fn: Callable | None = None,
    fetch_kg_chunks_fn: Callable | None = None,
) -> SearchResult:
    """供 Agent 调用的无状态图谱检索入口。

    返回实际原文与结构化状态；不会读写会话、发送进度或访问历史。
    回调参数只用于让旧节点复用历史辅助函数及其测试替身。
    """
    if query is not None and not isinstance(query, str):
        return SearchResult(status="error", error_code="INVALID_ARGUMENT")
    if not isinstance(item_names, (list, tuple)):
        return SearchResult(status="error", error_code="INVALID_ARGUMENT")
    normalized_query, names = prepare_kg_query(query or "", item_names)
    if not normalized_query or not names:
        return SearchResult(status="empty")

    try:
        if query_chunk_ids_fn is None:
            candidates = query_chunk_ids(normalized_query, names, driver=driver, database=database)
        else:
            candidates = query_chunk_ids_fn(normalized_query, names)
    except Exception as exc:
        logger.warning(f"知识图谱候选查询失败：{type(exc).__name__}")
        return SearchResult(status="error", error_code="KG_QUERY_FAILED", retryable=True)
    if not candidates:
        return SearchResult(status="empty")

    lookup_errors = []
    try:
        if fetch_kg_chunks_fn is None:
            chunks = fetch_kg_chunks(
                candidates, names, client=client, collection_name=collection_name,
                errors=lookup_errors,
            )
        else:
            chunks = fetch_kg_chunks_fn(candidates, names, errors=lookup_errors)
    except Exception as exc:
        logger.warning(f"知识图谱原文回查失败：{type(exc).__name__}")
        return SearchResult(
            status="error", error_code="MILVUS_LOOKUP_FAILED", retryable=True,
            candidate_count=len(candidates),
        )

    evidence = [
        Evidence(
            id=f"local:{hit['id']}",
            text=hit["entity"]["content"],
            title=hit["entity"].get("title") or "",
            source="local",
            retrieval_method="kg",
            item_name=hit["entity"].get("item_name"),
            chunk_id=hit["id"],
        )
        for hit in chunks
    ]
    if lookup_errors:
        status = "partial" if chunks else "error"
        error_code = "MILVUS_LOOKUP_FAILED"
    elif len(chunks) < len(candidates):
        status = "partial"
        error_code = "MISSING_ORIGINAL_TEXT"
    else:
        status = "ok"
        error_code = None
    logger.info(f"知识图谱召回完成：候选 {len(candidates)} 条，原文 {len(chunks)} 条")
    return SearchResult(
        status=status, items=chunks, evidence=evidence, error_code=error_code,
        retryable=bool(lookup_errors), candidate_count=len(candidates),
    )
