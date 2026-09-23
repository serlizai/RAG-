"""查询流程中的资料适用性校验节点。

在主图中的位置：node_rerank → node_evidence_guard → node_answer_output。
本文件负责调用校验逻辑、记录摘要和更新节点进度；具体规则在 evidence_guard.py。
这里检查的是“生成答案前能使用哪些资料”，生成后的答案复核由输出节点另行调用。
"""

from app.query_process.agent.evidence_guard import prepare_evidence
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def node_evidence_guard(state):
    """筛选可用参考资料，或为输出节点准备固定的拒答/澄清回复。

    参数：
        state：查询图的当前状态，主要使用以下字段：
            session_id：本轮会话标识，用于进度管理和日志追踪。
            is_stream：是否通过 SSE 向前端推送进度，缺省为 False。
            original_query、history、item_names：原始问题、历史及商品范围。
            reranked_docs：重排后待校验的参考片段。

    返回：
        交给 LangGraph 合并到 state 的字段更新，而非完整 state：
        - 允许生成时：返回筛选后的 reranked_docs 和 evidence_policy。
        - 无单机依据、版本待确认、资料不足或校验失败时：额外返回固定 answer，
          并清空参考片段及图片，后续输出节点直接发送该回复，不再生成新答案。

        evidence_policy.status 记录校验结果，blocked_docs 记录逐片校验的排除项。
        单机问题提前拦截等分支可能没有逐片排除记录，不能据此认为所有资料都适用。
    """
    # 与主图注册的节点名保持一致，进度工具会将它映射为“核对资料适用范围”。
    node_name = "node_evidence_guard"

    # 标记本节点开始处理；SSE 模式下还会将最新进度推送给前端。
    add_running_task(state["session_id"], node_name, state.get("is_stream", False))
    try:
        # 核心规则统一由 prepare_evidence 执行：单机依据限制、系统版本匹配，
        # 以及资料语义校验。预期的模型调用/解析失败会转换为固定兜底回复。
        result = prepare_evidence(state)
        policy = result["evidence_policy"]

        # 记录会话、校验状态及片段数量，便于定位回答被拦截或资料被排除的原因。
        logger.info(f"回答依据校验：session_id={state['session_id']}，status={policy['status']}，"
                    f"保留={len(result['reranked_docs'])}，排除={len(policy['blocked_docs'])}")

        # 只返回本节点负责的字段；商品信息、历史记录等继续保留在图状态中。
        return result
    finally:
        # 正常返回、提前拒答或异常向外传播时都结束进度，避免前端一直显示处理中。
        # “节点结束”不代表“校验通过”；是否通过应查看 evidence_policy.status。
        add_done_task(state["session_id"], node_name, state.get("is_stream", False))
