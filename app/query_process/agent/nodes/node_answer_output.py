import time
import sys
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_done_task, add_running_task, set_task_result


def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出
    :param state:
    :return:
    """
    print("---node_answer_output处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    base_answer = state.get("base_answer") or f"这是关于[{state.get('original_query', '当前问题')}]的测试回答，流式输出"
    final_text = ""

    if is_stream:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.03)

        image_urls = ["https://example.com/demo-1.png", "https://example.com/demo-2.png"]
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": final_text,
                "status": "completed",
                "image_urls": image_urls
            }
        )
        logger.info(f"流式输出完成，总长度: {len(final_text)}")
    else:
        final_text = base_answer
        # 图最后一个节点执行完毕，存储结果
        set_task_result(session_id, "answer" , final_text)

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("--node_answer_output 节点处理结束--")
    return {"answer": final_text}