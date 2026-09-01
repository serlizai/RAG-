import sys
import os

from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    主要作用: 多路召回的向量检索--->问题->查询chunks切片
    目标：{"embedding_chunks": [查到的chunks]}
    参数：重写的问题、item_names
    """
    print("--向量内容检索 开始处理--")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state.get("is_stream"))

    # 从state中获取参数数据
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 将重写问题生成对应稠密和稀疏向量
    embeddings = generate_embeddings([rewritten_query])
    # 进行向量数据库的混合查询
    # 创建混合查询请求;查询条件：content内容、主体名字item_name
    # 对每个商品名添加双引号，拼接为Milvus支持的in语法格式
    quoted = ", ".join(f'"{v}"' for v in item_names)
    # 构造最终过滤表达式
    expr = f"item_name in [{quoted}]"
    logger.info(f"创建搜索请求过滤表达式: {expr}")

    search_requests = create_hybrid_search_requests(
        embeddings["dense"][0],
        embeddings["sparse"][0],
        expr=expr
    )
    # 混合查询触发
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=search_requests,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        output_fields=["chunk_id", "content", "item_name"]  # 指定返回的业务字段
    )
    # 处理查询结果赋值embedding_chunks属性
    result_chunks = resp[0] if resp and len(resp) > 0 else []

    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state.get("is_stream"))

    print("--向量内容检索 处理结束--")
    return {"embedding_chunks": result_chunks}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "万用表RS-12的使用",  # 模拟改写后的查询
        "item_names": ["万用表RS-12"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)