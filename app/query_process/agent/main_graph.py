from langgraph.graph import StateGraph, END

from app.query_process.agent import state
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数（从具体文件导入函数，避免拿到模块对象）
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

builder = StateGraph(QueryGraphState)

builder.add_node("node_item_name_confirm", node_item_name_confirm) # 确认商品
builder.add_node("node_search_embedding", node_search_embedding)   # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)                             # 排序
builder.add_node("node_rerank", node_rerank)                       # 重排
builder.add_node("node_answer_output", node_answer_output)         # 生成

# 添加边
builder.set_entry_point("node_item_name_confirm")

# node_item_name_confirm可能出现没有明确主题item_name(查到打分低)，提前结束返回用户提示，让用户明确内容
# node_item_name_confirm->多路召回｜答案生成反馈给前端
# 条件边，通过state中的answer字段判断是否继续执行后续节点
def route_after_item_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        return "node_answer_output"
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


builder.add_conditional_edges("node_item_name_confirm", route_after_item_name_confirm,
                              {  # 只有写了这个静态ascii图才能看到这些节点，不写逻辑不影响但是静态图看不到
                                  "node_answer_output": "node_answer_output",
                                  "node_search_embedding": "node_search_embedding",
                                  "node_search_embedding_hyde": "node_search_embedding_hyde",
                                  "node_web_search_mcp": "node_web_search_mcp"
                              })


builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")  # 只是将结果让入到rrf节点的state不参与粗排
builder.add_edge("node_rrf", "node_rerank")  # mcp网络搜索在这里和前两个搜索排序结果进行重排
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

query_app = builder.compile()