import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def reciprocal_rank_fusion(source_with_weight, top_k:int=5):
    """
    进行同源数据排名+权重处理
    :param top_k:
    :param source_with_weight: [(集合, 权重), (集合, 权重)]
    :return:
    """
    # 准备记录历史得分的容器
    score_dict = {}  # id | chunk_id -> key || 计算得分 -> value，如果chunk_id重复则累加得分
    # 记录chunk片段
    chunk_dict = {}  # id | chunk_id -> key || chunk -> value，重复了不存，保证不重复

    # 3.循环处理每个集合中的数据进行积分计算
    for source, weight in source_with_weight:
        # 嵌套便利具体的数据和chunk
        for rank, chunk in enumerate(source, start=1):  # 已经排过序了从大到小
            chunk_id = chunk.get("id") or chunk.get("entity").get("chunk_id")
            if not chunk_id:
                logger.warning(f"缺少 chunk_id 的数据: {chunk}")
                continue

            # 计算当前chunk RRF 得分
            score_dict[chunk_id] = score_dict.get(chunk_id, 0) + 1.0 * weight / (rank + 60)  # 累加
            # chunk_dict[chunk_id] = chunk  # 重复会覆盖
            chunk_dict.setdefault(chunk_id, chunk)  # 有的时候不会添加，没有才会加入

    # 分和chunk的融合+排序
    merged = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        merged.append((chunk, score))
    merged.sort(key=lambda x: x[1], reverse=True)

    merged = merged[:top_k]
    logger.info(f"RRF 融合完成，Top {top_k} 结果: {[{'chunk_id': c.get('id') or c.get('entity').get('chunk_id'), 'score': s} for c, s in merged]}")

    return [chunk for chunk, score in merged]


def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点

    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。

    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]
    """
    print("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 获取同源数据
    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    # 单列查询 data = [向量1，向量2] -> [[向量1], [向量2]]
    # 混合查询 reps = [AnnSearchRequest] -> [[]] -> [{id:实体主键, distance:得分, entity:{}},{}...]
    # 数据进行整合(捏到一起)
    # 权重后续方便动态调整选择相同的1.0
    source_with_weight = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0),
    ]
    # rrf算法进行数据排序处理
    rrf_response = reciprocal_rank_fusion(source_with_weight)
    # 将排序后的数据添加到state[rrf_chunks]属性即可

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # state["rrf_chunks"] = rrf_response

    return {"rrf_chunks": rrf_response}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果
    mock_embedding_chunks = [
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.9
        },
        {
            "id": "doc_2",
            "pk": "pk_2",
            "file_title": "维修指南.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容2：遇到故障请联系...",
            "score": 0.8
        },
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.7
        }
    ]

    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.85
        },
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.82
        },
        {
            "id": "doc_4",
            "pk": "pk_4",
            "file_title": "安全须知.docx",
            "item_name": "HAK 180 烫金机",
            "content": "内容4：操作时请佩戴手套...",
            "score": 0.75
        }
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)

        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]

        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")

        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")