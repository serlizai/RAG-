import sys

from app.core.logger import logger
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task

load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5


def step_1_merge_docs(state):
    """
    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式
    格式：
    [
        rrf = {id:chunk_id, distance:0.xxx, entity:{}}
        mcp = {"title": title, "url": url, "snippet": snippet}
        合并后的格式：
        {
            text: snippet、content，
            chunk_id：rrf才有，mcp为none，
            title,
            url：rrf为none,mcp有
            source:web->mcp, local->rrf
        }
    ]
    :param state:
    :return:
    """
    # 获取数据
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    # 准备列表容器
    chunks_list = []
    # 循环添加数据
    # rrf local
    for chunk in rrf_chunks:
        entity = chunk.get("entity")
        chunks_list.append({
            "text": entity.get("content", ""),
            "chunk_id": entity.get("chunk_id") or entity.get("id"),
            "title": entity.get("title") or entity.get("item_name") or "",
            "url": "",
            "source": "local"
        })
    # mcp web
    for doc in web_search_docs:
        chunks_list.append({
            "text": (doc.get("snippet") or doc.get("content") or "").strip(),
            "chunk_id": None,  # # 联网结果无固定 ID
            "title": (doc.get("title", "")).strip(),
            "url": (doc.get("url", "")).strip(),
            "source": "web"
        })
    
    logger.info(f"Step 1: 合并文档完成, 总数: {len(chunks_list)}")
    # 普通的内部处理函数不是节点所以不返回state
    return chunks_list


def step_2_rerank_docs(state, doc_items):
    """
    对文档进行重排序
    格式：
    [
        {
            text: snippet、content，
            chunk_id：rrf才有，mcp为none，
            title,
            url：rrf为none,mcp有
            source:web->mcp, local->rrf，
            score: rerank打分
        }
    ]
    :param state: 
    :param doc_items: 
    :return: 
    """
    # 获取问题
    query = state.get("rewritten_query") or state.get("original_query")
    # 获取问题答案
    answers = [doc.get("text") for doc in doc_items]
    # 加载rerank模型
    rerank = get_reranker_model()
    # 处理数据: 问题+答案成对装到列表再打分
    # [问题，答案]/(问题，答案) -> 一对限制
    # 512
    pairs = [[query, answer] for answer in answers]
    # normalize=False 分数范围不确定分差很大，true就会限制到0-1
    scores = rerank.compute_score(pairs, normalize=True)  # 一对按顺序对应一个分：[0.9，0.89，...]
    # 原数据添加分数
    doc_with_scores = []
    for score, item in zip(scores, doc_items):
        item["score"] = score
        doc_with_scores.append(item)

    # 排序
    doc_with_scores.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"Step 2: Rerank重排序完成, 总数: {len(doc_with_scores)}")
    return doc_with_scores


def step_3_topk(scored_docs):
    """
    对rerank模型打分以后的有序集合进行再次算法筛选，动态取出topk

    [
        {
            text: snippet、content，
            chunk_id：rrf才有，mcp为none，
            title:title,
            url：rrf为none,mcp有
            source:web->mcp, local->rrf，
            score: rerank打分
        }
    ]
    :param scored_docs:
    :return:
    """
    # 双指针
    topk = min(RERANK_MAX_TOPK, len(scored_docs))

    if topk > RERANK_MIN_TOPK:
        # min-1 topk-1
        for index in range(RERANK_MIN_TOPK - 1, topk - 1):
            left = scored_docs[index].get("score", 0.0)
            right = scored_docs[index + 1].get("score", 0.0)
            # 分数差值
            gap = left - right
            # 相对差值
            relative = gap / (abs(left + 1e-6))  # 防止分母为0和分数出现负数(例如打分的时候没有normalize)
            if gap >= RERANK_GAP_ABS or relative >= RERANK_GAP_RATIO:
                # 断崖出现，截断topk
                logger.info(f"Step 3: 动态TopK截断, 原topk={topk}, 截断位置={index + 1}, left_score={left:.4f}, right_score={right:.4f}, gap={gap:.4f}, relative_gap={relative:.4f}")
                topk = index + 1  # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                break
    # else:
        # min_topk = topk  不用管
        # min_topk > topk  只有list是空的才会出现这个状况，取topk就行

    topk_score_list = scored_docs[:topk]

    logger.info(f"Step 3: 最终TopK结果, topk={topk}")
    return topk_score_list


def node_rerank(state):
    """
    节点功能：使用rerank模型对RRF后+MCP网络搜索的结果进行精确打分重排用防断崖算法取top_k(最多10最少1)

    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 合并rrf和mcp结果
    # 阶段一：合并文档
    doc_items = step_1_merge_docs(state)
    # 阶段二：对文档进行重排序,rerank精排
    scored_docs = step_2_rerank_docs(state, doc_items)
    # 阶段三：动态 TopK,用算法进行防断崖和取top_k
    topk_docs = step_3_topk(scored_docs)
    logger.info(f"最终文档:{topk_docs}")

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs": topk_docs}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
