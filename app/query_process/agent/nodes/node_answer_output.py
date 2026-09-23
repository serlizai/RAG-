import json
import sys

from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
from app.query_process.agent.evidence_guard import review_answer
from app.query_process.agent.nodes.node_evidence_guard import node_evidence_guard
import re
import time

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000  # 限制prompt的最大字符数，避免过长导致大模型处理缓慢或失败


def step_1_check_answer(state):
    """
    判断第一个节点 有没有明确的answer回答(item_name明不明确)
    - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
    - 若不存在：返回 False
    :param state:
    :return:
    """
    # 获取answer、is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)

    if answer:
        if is_stream:
            # 有且是流式->push_to_session
            push_to_session(state['session_id'], SSEEvent.DELTA, {"delta": answer})
        else:
            # 有但不是流式->set_task_result
            set_task_result(state['session_id'], "answer", answer)
        return True
    # 返回true有答案，false没答案
    else:
        return False


def step_2_construct_prompt(state):
    """
    加载模型润色答案的提示词
    :param state:
    :return:
    """
    # 获取数据
    question = state.get("original_query") or state.get("rewritten_query") or ""
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    history = state.get("history", [])

    # 处理chunk块的内容->context
    """
        格式：
        [1][source][url][title][score]
        [text]
        [2]...
    """
    docs = []
    context_docs = []
    used_length = 0  # 使用的长度
    for i, doc in enumerate(reranked_docs, start=1):
        text = doc.get("text")
        chunk_id = doc.get("chunk_id")
        source = doc.get("source")
        url = doc.get("url")
        title = doc.get("title")
        score = doc.get("score")
        content = f"[{i}][chunk_id={chunk_id}][source={source}][url={url}][title={title}][score={score}]\n[text={text}]"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            logger.info(f"达到最大上下文长度限制，停止添加更多文档。当前使用长度: {used_length}, 文档长度: {len(content)}, 最大限制: {MAX_CONTEXT_CHARS}")
            break
        docs.append(content)
        context_docs.append(doc)
        used_length += len(content) + 2
    final_context = "\n\n".join(docs)
    # 复核和图片提取只能使用本次实际送入生成模型的片段。
    state["reranked_docs"] = context_docs
    # 处理history聊天记录内容
    history_str = ""
    if history and len(history) > 0:
        for i, message in enumerate(history, start=1):
            role = message.get("role")
            text = message.get("text")
            current_history = ""
            if role == "user" and text:
                current_history = f"用户: {text}\n"
            # 旧助手回答可能含未核验断言，不再作为生成依据。
            if used_length + len(current_history) > MAX_CONTEXT_CHARS:
                logger.info(f"达到最大上下文长度限制，停止添加更多历史记录。当前使用长度: {used_length}, 最大限制: {MAX_CONTEXT_CHARS}")
                break
            used_length += len(current_history)
            history_str += current_history
    else:
        history_str = "无历史对话记录"
    # 处理item_name
    item_names_str = ",".join(item_names)
    # 处理问题question
    final_prompt = load_prompt("answer_out",
                           context=final_context,
                           history=history_str,
                           item_names=item_names_str,
                           policy=json.dumps(state.get("evidence_policy", {}), ensure_ascii=False),
                           question=question)
    logger.info(f"构建Prompt完成，长度: {len(final_prompt)}, 内容:{final_prompt}")
    return final_prompt


def step_3_generate_response(state, prompt):
    """
    生成完整草稿，核验通过后才发送；SSE 保留进度事件，不提前发送草稿。
    :param state:
    :param prompt:
    :return:
    """
    # 获取模型对象和客户端
    model = get_llm_client()
    # 获取流式状态
    is_stream = state.get("is_stream", False)
    response = model.invoke(prompt)
    review_node = "answer_evidence_review"
    add_running_task(state['session_id'], review_node, is_stream)
    try:
        answer, verified = review_answer(state, response.text)
    finally:
        add_done_task(state['session_id'], review_node, is_stream)
    state["answer_verified"] = verified
    if not verified:
        state["reranked_docs"] = []
        state["image_urls"] = []
    if is_stream:
        push_to_session(state['session_id'], SSEEvent.DELTA, {"delta": answer})
    else:
        set_task_result(state['session_id'], "answer", answer)
    # state赋值和返回结果
    state["answer"] = answer
    logger.info(f"回答核验完成，verified={verified}，长度: {len(answer)}, 内容: {answer}")
    return answer


def step_4_write_history(state, image_urls):
    """
    保存本轮助手回复及图片，用户提问已由 node_item_name_confirm 保存。
    正常答案和入口节点生成的澄清提示都在这里统一写入。
    :param state:
    :param image_urls:
    :return:
    """
    session_id = state.get("session_id")
    answer = state.get("answer")
    item_names = state.get("item_names", [])

    if answer:
        # assistant
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query="",
            image_urls=image_urls,
            message_id=None
        )

    logger.info(f"写入MongoDB历史记录完成，session_id: {session_id}, answer长度: {len(answer)}, 图片数量: {len(image_urls)}")


def _extract_images_from_docs(state):
    """
    从local的chunk的content中提取
    从web的url中提取图片
    :param state:
    :return:
    """
    image_urls = []  # 有先后顺序
    set_images = set()  # 去重

    # 正则
    image_reg = re.compile(r"!\[.*?]\((.*?)\)")
    # 处理切片
    reranked_docs = state.get("reranked_docs", [])
    for doc in reranked_docs:
        url = doc.get("url")
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in set_images:
                    image_urls.append(url)
                    set_images.add(url)
        text = doc.get("text")
        if text:
            # 从text里面提取图片
            # findall 机制解释：
            # 正则表达式 r'!\[.*?\]\((.*?)\)' 中包含一个捕获组 (.*?)
            # 当存在捕获组时，findall 只返回括号内匹配到的内容（即 URL），而不是整个 ![...](...) 字符串
            # 示例：
            # 输入 text: "参考图片 ![面板图](http://img.com/1.jpg) 如下"
            # 返回 matches: ['http://img.com/1.jpg']
            matches = image_reg.findall(text)
            for match in matches:
                if match not in set_images:
                    image_urls.append(match)
                    set_images.add(match)

    logger.info(f"提取图片URL完成，数量: {len(image_urls)}, 内容: {image_urls}")
    state["image_urls"] = image_urls
    return image_urls


def node_answer_output(state):
    """
    宏观：topk内容->大模型润色->润色结果->sse输出到前端  push_to_session流式、set_task_result非流式
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成润色答案
    3 调用大模型生成并复核答案，通过后在 SSE 模式发送完整文本，图片列表写入state
    4 将本轮助手回复及图片写入MongoDB，用户提问由入口节点保存
    5 通过state返回答案和图片；查询图完成后，由接口层发送final事件
    接口层发送的最终结果格式：
     {
        "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
        "status": "completed",
        "image_urls": [
            "http://local-server/images/panel_view.jpg",
            "http://local-server/images/button_detail.jpg"
        ]
      }
    节点功能：进行过处理可以是流式输出可以整体输出
    :param state:
    :return:
    """
    logger.info("---node_answer_output (答案生成) 节点开始处理---")
    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 单独调用本节点也不能绕过核验；主图正常路径已经完成此步骤。
    if not state.get("answer") and not state.get("evidence_policy"):
        state.update(node_evidence_guard(state))

    # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
    answer_exists = step_1_check_answer(state)

    # 阶段二  如果没有answer则 构建 Prompt
    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt

        # 阶段三：  如果没有answer则 调用大模型输出答案
        step_3_generate_response(state, prompt)

    # 提取图片URL（用于历史记录和前端展示）
    image_urls = _extract_images_from_docs(state)

    # 阶段四：统一保存助手回复（正式答案或已有的澄清提示）及图片
    if state.get("answer"):
        logger.info("---写入MongoDB历史记录---")
        step_4_write_history(state, image_urls=image_urls)

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 阶段五：非流式保存图片供 /query 返回；流式 final 由接口层在图完成后发送
    if not state.get("is_stream"):
        # 任务结果只存字符串，因此先将图片列表编码为 JSON
        set_task_result(state['session_id'], "image_urls", json.dumps(image_urls, ensure_ascii=False))

    logger.info("---node_answer_output 节点处理结束---")
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://www.baidu.com/img/bd_logo.png)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案: {answer}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
