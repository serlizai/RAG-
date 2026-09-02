from langchain_core.messages import HumanMessage

# HyDE节点
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # 先用 find_dotenv() 明确找到 .env 的路径再加载，方便调试


def step_1_create_hyde_doc(rewritten_query):
    """
    调用模型根据问题生成假设答案
    :param rewritten_query: 重写的问题
    :return: 答案
    """
    llm = get_llm_client()

    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)

    messages = [HumanMessage(content=hyde_prompt)]

    response = llm.invoke(messages)
    hyde_doc = response.content

    logger.info(f"Step 1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")
    logger.debug(f"Step 1: 文档预览: {hyde_doc[:50]}...")

    return hyde_doc


def step_2_search_embedding_hyde(
        rewritten_query,
        hyde_doc,
        item_names,
        req_limit: int = 10,
        top_k: int = 5,
        ranker_weights=(0.8, 0.2),  # 调整默认权重以偏向稠密向量 (0.8, 0.2)
        norm_score: bool = True,    # 默认开启归一化
        output_fields= ["chunk_id", "content", "item_name"],
):
    """
    利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片
    :param rewritten_query:
    :param hyde_doc:
    :param item_names:
    :param req_limit:
    :param top_k:
    :param ranker_weights:
    :param norm_score:
    :param output_fields:
    :return:
    """
    # 拼接重写问题和假设答案
    combined_text = rewritten_query + " " + hyde_doc
    # 生成拼接字符串向量
    embeddings = generate_embeddings([combined_text])
    # 创建AnnSearchRequest混合请求
    item_name = ','.join(f'"{item}"' for item in item_names)
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name}]",
    )
    # 混合查询
    milvus_client = get_milvus_client()
    # AnnSearchRequest.limit >= hybrid_search.limit
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        output_fields=output_fields,
    )
    # 返回
    return resp[0] if resp else []
    logger.info(f"Step 2: HyDE混合检索完成, 返回 {len(resp[0]) if resp else 0} 条结果")



def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案+问题进行向量检索，提高召回率。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 提取参数 item_names+rewritten_query
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    # 用模型生成假设答案
    logger.info("Step 1: 开始生成假设性文档 (HyDE Doc)...")
    hyde_doc = step_1_create_hyde_doc(rewritten_query)
    # 问题+答案进行混合检索
    logger.info("Step 2: 基于假设文档执行 Milvus 混合检索...")
    res = step_2_search_embedding_hyde(
        rewritten_query=rewritten_query,
        hyde_doc=hyde_doc,
        item_names=item_names,
    )

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")

    # 返回和赋值
    return {"hyde_embedding_chunks": res}


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "万用表RS-12怎么操作？",
        "rewritten_query": "万用表RS-12的具体操作步骤是什么？",
        "item_names": ["万用表RS-12"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")