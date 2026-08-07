import copy

from pyparsing import original_text_for
from typing_extensions import TypedDict
from typing import List


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    kg_chunks: list  # 图谱检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记


# 定义图状态的默认初始值
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "kg_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False
}

# 创建默认状态(可覆盖)
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建一个默认的 QueryGraphState，并允许覆盖默认值。
    :param overrides: 可选的关键字参数，用于覆盖默认值
    :return: 返回一个完整的 QueryGraphState 字典
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# 获取干净状态
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


# 状态复制函数
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == '__main__':
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为p60怎么样",
        is_stream=False
    )
    print("测试创建默认状态：", state)

    # 测试状态复制
    new_state = copy_query_state(state, original_query="修改后的问题")
    print("测试状态复制：", new_state)