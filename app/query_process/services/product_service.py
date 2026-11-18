"""商品名称解析服务。调用方提供历史记录；这里不管理会话或节点进度。"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.services.schemas import ProductResolutionResult


def extract_product_info(original_query: str, history: list[dict]) -> dict[str, Any]:
    """根据本次问题和已提供的历史，提取商品名并改写检索问题。"""
    history_text = "".join(
        f"聊天角色：{msg['role']}: 聊天内容{msg['text']}\n" for msg in history
    )
    prompt = load_prompt(
        "rewritten_query_and_itemnames", history_text=history_text, query=original_query
    )
    response = get_llm_client(json_mode=True).invoke(
        [
            SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
            HumanMessage(content=prompt),
        ]
    )
    content = response.content
    if not isinstance(content, str):
        raise ValueError("商品提取模型未返回 JSON 文本")
    content = content.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").removesuffix("```").strip()
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("商品提取模型未返回 JSON 对象")

    item_names = result.get("item_names", [])
    if not isinstance(item_names, list):
        raise ValueError("商品提取模型的 item_names 不是列表")
    result["item_names"] = [name for name in item_names if isinstance(name, str) and name]
    rewritten_query = result.get("rewritten_query", original_query)
    result["rewritten_query"] = rewritten_query if isinstance(rewritten_query, str) else original_query
    return result


def search_matching_products(item_names: list[str]) -> list[dict]:
    """逐个匹配模型提取的商品名，保留原来的混合搜索权重和评分。"""
    if not item_names:
        return []

    client = get_milvus_client()
    if client is None:
        raise RuntimeError("商品名称库客户端不可用")
    embeddings = generate_embeddings(item_names)
    results = []
    for index, item_name in enumerate(item_names):
        requests = create_hybrid_search_requests(
            embeddings["dense"][index], embeddings["sparse"][index]
        )
        response = hybrid_search(
            client=client,
            collection_name=milvus_config.item_name_collection,
            reqs=requests,
            ranker_weights=(0.7, 0.3),
            norm_score=True,
        )
        # hybrid_search 以 None 表示查询异常，空列表才是正常的零命中。
        if response is None:
            raise RuntimeError("商品名称库查询失败")
        matches = []
        if response:
            for hit in response[0]:
                hit_name = hit.get("entity", {}).get("item_name", "")
                if hit_name:
                    matches.append({"item_name": hit_name, "score": hit.get("distance", 0)})
        results.append({"extracted_name": item_name, "matches": matches})
    return results


def align_product_names(query_results: list[dict]) -> dict[str, list[str]]:
    """高分自动确认；中分返回候选。阈值与旧节点一致。"""
    confirmed_item_names = []
    options_item_names = []
    for item in query_results:
        extracted_name = item.get("extracted_name", "")
        matches = sorted(item.get("matches", []), key=lambda match: match.get("score", 0), reverse=True)
        high = [match for match in matches if match.get("score", 0) >= 0.85]
        middle = [match for match in matches if 0.6 <= match.get("score", 0) < 0.85]
        if high:
            same_name = next(
                (match for match in high if match.get("item_name") == extracted_name), None
            )
            confirmed_item_names.append((same_name or high[0]).get("item_name", ""))
        elif middle:
            options_item_names.extend(match.get("item_name", "") for match in middle[:2])
        else:
            logger.info("没有匹配的商品名: {}", extracted_name)
    return {
        "confirmed_item_names": list(dict.fromkeys(confirmed_item_names)),
        "options_item_names": list(dict.fromkeys(options_item_names)),
    }


def resolve_product(original_query: str, history: list[dict]) -> ProductResolutionResult:
    """解析商品名称。不会读取历史、保存消息或更新 LangGraph 状态。"""
    try:
        extracted = extract_product_info(original_query, history)
    except Exception:
        logger.exception("商品名称提取失败")
        return ProductResolutionResult(
            status="error", rewritten_query=original_query,
            error_code="PRODUCT_EXTRACT_FAILED", retryable=True,
        )

    rewritten_query = extracted["rewritten_query"]
    item_names = extracted["item_names"]
    if not item_names:
        return ProductResolutionResult(status="empty", rewritten_query=rewritten_query)

    try:
        matches = search_matching_products(item_names)
        aligned = align_product_names(matches)
    except Exception:
        logger.exception("商品名称匹配失败")
        return ProductResolutionResult(
            status="error", rewritten_query=rewritten_query,
            error_code="PRODUCT_MATCH_FAILED", retryable=True,
        )

    confirmed = aligned["confirmed_item_names"]
    options = aligned["options_item_names"]
    status = "ok" if confirmed else "needs_clarification" if options else "empty"
    return ProductResolutionResult(
        status=status,
        rewritten_query=rewritten_query,
        confirmed_item_names=confirmed,
        option_item_names=options,
    )
