"""Session-independent retrieval, fusion, and reranking for the query graph.

The ``items`` in each result retain the legacy graph's Milvus/web shapes. The
parallel ``evidence`` field is the stable, source-labelled view used by Agent
tools. No function in this module reads graph state or publishes SSE events.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from app.core.logger import logger
from app.query_process.services.schemas import Evidence, SearchResult
from app.utils.document_image_utils import normalize_document_images


CHUNK_FIELDS = ["chunk_id", "content", "item_name", "title", "file_title"]
RERANK_MAX_TOPK = 10
RERANK_MIN_TOPK = 1
RERANK_GAP_RATIO = 0.25
RERANK_GAP_ABS = 0.5


def _scope(item_names: Sequence[str] | None) -> list[str]:
    if not isinstance(item_names, (list, tuple)):
        return []
    return list(dict.fromkeys(name for name in item_names if isinstance(name, str) and name.strip()))


def _evidence_id(source: str, chunk_id: Any, text: str, url: str = "") -> str:
    if source == "local" and chunk_id is not None:
        return f"local:{chunk_id}"
    if source == "web" and url:
        return f"web:{url}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}:text:{digest}"


def evidence_from_hits(items: list[dict], method: str) -> list[Evidence]:
    evidence = []
    for hit in items:
        if not isinstance(hit, dict):
            continue
        entity = hit.get("entity") or hit
        if not isinstance(entity, dict):
            continue
        content = entity.get("content") or entity.get("text") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        chunk_id = entity.get("chunk_id")
        if chunk_id is None:
            chunk_id = hit.get("id")
        source = "web" if entity.get("source") == "web" else "local"
        url = str(entity.get("url") or "") if source == "web" else ""
        evidence.append(Evidence(
            id=_evidence_id(source, chunk_id, content, url),
            text=content,
            title=str(entity.get("title") or entity.get("file_title") or entity.get("item_name") or ""),
            source=source,
            retrieval_method=method,
            item_name=entity.get("item_name") if source == "local" else None,
            chunk_id=chunk_id if source == "local" else None,
            url=url or None,
        ))
    return evidence


def search_embedding(query: str, item_names: Sequence[str] | None, *,
                     hyde_doc: str | None = None,
                     output_fields: Sequence[str] | None = None) -> SearchResult:
    """Run the existing dense/sparse Milvus search within confirmed products."""
    names = _scope(item_names)
    if not names:
        return SearchResult(status="needs_clarification", error_code="PRODUCT_REQUIRED")
    if not isinstance(query, str) or not query.strip():
        return SearchResult(status="error", error_code="INVALID_QUERY")
    if hyde_doc is not None and not isinstance(hyde_doc, str):
        return SearchResult(status="error", error_code="INVALID_QUERY")

    try:
        # Imports stay inside the I/O boundary so fusion and rerank tests need
        # neither a Milvus connection nor an embedding model at import time.
        from app.clients.milvus_utils import (create_hybrid_search_requests,
                                              get_milvus_client, hybrid_search)
        from app.conf.milvus_config import milvus_config
        from app.lm.embedding_utils import generate_embeddings

        text = f"{query} {hyde_doc}" if hyde_doc is not None else query
        embeddings = generate_embeddings([text])
        expr = "item_name in [" + ", ".join(
            json.dumps(name, ensure_ascii=False) for name in names
        ) + "]"
        requests = create_hybrid_search_requests(
            embeddings["dense"][0], embeddings["sparse"][0], expr=expr,
        )
        response = hybrid_search(
            client=get_milvus_client(),
            collection_name=milvus_config.chunks_collection,
            reqs=requests,
            ranker_weights=(0.9, 0.1),
            norm_score=hyde_doc is None,
            output_fields=list(output_fields) if output_fields is not None else CHUNK_FIELDS,
        )
    except Exception:
        logger.exception("向量检索失败")
        return SearchResult(status="error", error_code="VECTOR_SEARCH_FAILED", retryable=True)

    # ``hybrid_search`` returns None after catching its own connection errors.
    if response is None:
        return SearchResult(status="error", error_code="VECTOR_SEARCH_FAILED", retryable=True)
    try:
        items = list(response[0]) if response else []
    except (TypeError, IndexError):
        logger.warning("向量检索返回了无法解析的结果")
        return SearchResult(status="error", error_code="INVALID_VECTOR_RESPONSE")
    method = "hyde" if hyde_doc is not None else "embedding"
    return SearchResult(status="ok" if items else "empty", items=items,
                        evidence=evidence_from_hits(items, method),
                        candidate_count=len(items))


def generate_hypothetical_document(query: str) -> str:
    """Generate the HyDE text without needing a graph state or session."""
    from langchain_core.messages import HumanMessage

    from app.core.load_prompt import load_prompt
    from app.lm.lm_utils import get_llm_client

    prompt = load_prompt("hyde_prompt", rewritten_query=query)
    result = get_llm_client().invoke([HumanMessage(content=prompt)])
    document = result.content
    if not isinstance(document, str) or not document.strip():
        raise ValueError("HyDE model returned no text")
    logger.info("假设文档生成完成, 长度: {} 字符", len(document))
    return document


def search_embedding_hyde(query: str, item_names: Sequence[str] | None,
                          *, hyde_doc: str | None = None,
                          output_fields: Sequence[str] | None = None) -> SearchResult:
    """Generate a hypothesis, then search Milvus with query plus hypothesis."""
    if not _scope(item_names):
        return SearchResult(status="needs_clarification", error_code="PRODUCT_REQUIRED")
    if not isinstance(query, str) or not query.strip():
        return SearchResult(status="error", error_code="INVALID_QUERY")
    if hyde_doc is None:
        try:
            hyde_doc = generate_hypothetical_document(query)
        except Exception:
            logger.exception("HyDE 假设文档生成失败")
            return SearchResult(status="error", error_code="HYDE_GENERATION_FAILED", retryable=True)
    return search_embedding(query, item_names, hyde_doc=hyde_doc,
                            output_fields=output_fields)


def reciprocal_rank_fusion(source_with_weight: Sequence[tuple[Sequence[dict] | None, float]],
                           top_k: int = 5) -> list[dict]:
    """Fuse ranked sources. A duplicate within one source casts one vote."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for source, weight in source_with_weight:
        seen: set[str] = set()
        for chunk in source or []:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("id")
            if chunk_id is None:
                chunk_id = (chunk.get("entity") or {}).get("chunk_id")
            if chunk_id is None:
                logger.warning("缺少 chunk_id 的数据: {}", chunk)
                continue
            key = str(chunk_id)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + weight / (len(seen) + 60)
            chunks.setdefault(key, chunk)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    result = [chunks[key] for key in ranked]
    logger.info("RRF 融合完成，Top {}，保留 {} 个切片", top_k, len(result))
    return result


def fuse_local_results(embedding_chunks: Sequence[dict] | None,
                       hyde_embedding_chunks: Sequence[dict] | None,
                       kg_chunks: Sequence[dict] | None,
                       *, top_k: int = 20) -> SearchResult:
    """Combine the three legacy local recall branches with equal weights."""
    items = reciprocal_rank_fusion([
        (embedding_chunks, 1.0), (hyde_embedding_chunks, 1.0), (kg_chunks, 1.0),
    ], top_k=top_k)
    return SearchResult(status="ok" if items else "empty", items=items,
                        evidence=evidence_from_hits(items, "rrf"),
                        candidate_count=len(items))


def merge_documents(rrf_chunks: Sequence[dict] | None,
                    web_search_docs: Sequence[dict] | None) -> list[dict]:
    """Adapt Milvus hits and web results to the existing reranker document shape."""
    documents = []
    for chunk in rrf_chunks or []:
        entity = chunk.get("entity") or {}
        documents.append({
            "text": normalize_document_images(entity.get("content", ""), entity.get("file_title") or ""),
            "chunk_id": entity.get("chunk_id") or entity.get("id"),
            "title": entity.get("title") or entity.get("item_name") or "",
            "file_title": entity.get("file_title") or "",
            "item_name": entity.get("item_name") or "",
            "url": "",
            "source": "local",
        })
    for doc in web_search_docs or []:
        documents.append({
            "text": (doc.get("snippet") or doc.get("content") or "").strip(),
            "chunk_id": None,
            "title": (doc.get("title") or "").strip(),
            "url": (doc.get("url") or "").strip(),
            "source": "web",
        })
    logger.info("合并文档完成, 总数: {}", len(documents))
    return documents


def score_documents(query: str, doc_items: Sequence[dict]) -> list[dict]:
    """Compute normalized reranker scores and sort, without mutating inputs."""
    if not doc_items:
        return []
    from app.lm.reranker_utils import get_reranker_model

    pairs = [[query, item.get("text")] for item in doc_items]
    scores = get_reranker_model().compute_score(pairs, normalize=True)
    if isinstance(scores, (int, float)):
        scores = [scores]
    if len(scores) != len(doc_items):
        raise ValueError("reranker score count does not match documents")
    ranked = [{**item, "score": score} for item, score in zip(doc_items, scores)]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    logger.info("Rerank 重排序完成, 总数: {}", len(ranked))
    return ranked


def select_top_k(scored_docs: Sequence[dict]) -> list[dict]:
    """Keep the existing dynamic top-k cliff detection unchanged."""
    topk = min(RERANK_MAX_TOPK, len(scored_docs))
    if topk > RERANK_MIN_TOPK:
        for index in range(RERANK_MIN_TOPK - 1, topk - 1):
            left = scored_docs[index].get("score", 0.0)
            right = scored_docs[index + 1].get("score", 0.0)
            gap = left - right
            relative = gap / abs(left + 1e-6)
            if gap >= RERANK_GAP_ABS or relative >= RERANK_GAP_RATIO:
                logger.info("动态 TopK 截断，位置={}，差值={:.4f}", index + 1, gap)
                topk = index + 1
                break
    logger.info("最终 TopK 结果, topk={}", topk)
    return list(scored_docs[:topk])


def rerank_documents(query: str, rrf_chunks: Sequence[dict] | None,
                     web_search_docs: Sequence[dict] | None) -> SearchResult:
    """Merge, rerank, and return the final legacy documents plus Agent evidence."""
    try:
        documents = merge_documents(rrf_chunks, web_search_docs)
    except (AttributeError, TypeError):
        logger.warning("重排输入包含无法解析的文档")
        return SearchResult(status="error", error_code="INVALID_DOCUMENTS")
    if not documents:
        return SearchResult(status="empty")
    if not isinstance(query, str) or not query.strip():
        return SearchResult(status="error", error_code="INVALID_QUERY",
                            candidate_count=len(documents))
    try:
        items = select_top_k(score_documents(query, documents))
    except Exception:
        logger.exception("文档重排失败")
        return SearchResult(status="error", error_code="RERANK_FAILED", retryable=True,
                            candidate_count=len(documents))
    return SearchResult(status="ok" if items else "empty", items=items,
                        evidence=evidence_from_hits(items, "rerank"),
                        candidate_count=len(documents))
