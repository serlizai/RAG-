import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from urllib import response

from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_extract_info(original_query, history):
    """
    根据历史记录识别item_name和重写问题
    :param original_query: 原始问题
    :param history: 聊天记录
    :return: {
            "item_names": List[str],  # 提取出的商品名称
            "rewritten_query": str,  # 改写后的问题
        }
    """
    # 1. 准备提示词
    history_text = ""
    for msg in history:
        history_text += f"聊天角色：{msg['role']}: 聊天内容{msg['text']}\n"
    logger.info(f"Step 3: 历史上下文准备完成 (长度: {len(history_text)})")

    """
          为了让 Python 把大括号当作 “普通字符” 保留下来，f-string 规定：
          用双大括号 {{ 表示普通的左大括号 {，双大括号 }} 表示普通的右大括号 }。
    """
    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)
    logger.info(f"Step 3: 提示词加载成功")
    # 2. 模型调用
    llm_client = get_llm_client(json_mode=True)
    # system->模型角色边界  human->每次任务提示
    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]
    response = llm_client.invoke(messages)
    # 3. 结果解析
    content = response.content
    # 处理LLM可能返回的代码块格式（如```json ... ```），去除包裹符
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    # 将处理后的文本转为JSON字典，解析LLM返回结果
    result = json.loads(content)
    # 健壮性处理：确保返回结果包含item_names字段，无则设为空列表
    if "item_names" not in result:
        result["item_names"] = []
    # 健壮性处理：确保返回结果包含rewritten_query字段，无则复用原始查询
    if "rewritten_query" not in result:
        result["rewritten_query"] = original_query
    logger.info(f"Step 3: LLM提取结果解析完成, item_names: {result['item_names']}, rewritten_query: {result['rewritten_query']}")
    # 4. 封装返回
    return result


def step_4_vectorize_and_query(item_names):
    """
           把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
           :param item_names: 列表[字符串] - step3提取的商品名列表（如["苹果15", "华为P60"]）
           :return: 列表[字典] - 每个商品名的向量化+搜索结果，格式：
                [
                    {
                        "extracted_name": "提取的原始商品名",  # 如"苹果15"
                        "matches": [                          # 该商品名的TopN匹配结果，无则空列表
                            {
                                "item_name": "数据库中的商品名",  # Milvus中存储的标准化商品名
                                "score": 0.98                  # 混合搜索的相似度评分（0-1，越高越相似）
                            },
                            ...
                        ]
                    },
                    ...
                ]
        """
    final_results = []
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # item_names稠密、稀疏向量化【不循环是因为item_names大概率不超过8192维度】
    # embeddings格式：{"dense": [向量1, 向量2,...], "sparse": [向量1, 向量2,...]}
    logger.info("Step 4: 正在生成向量...")
    embeddings = generate_embeddings(item_names)
    logger.info(f"Step 4: 已生成 {len(item_names)} 个商品名的向量。开始 Milvus 搜索...")
    # 混合查询(创建稠密、稀疏的AnnSearchRequest；设置权重重排；进行混合查询)
    for index, item_name in enumerate(item_names):
        # 获取当前item_name对应的向量
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        # 拼接对应的AnnSearchRequest
        reqs = create_hybrid_search_requests(dense_vector, sparse_vector)
        # 定义权重重排
        # 混合检索
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.7, 0.3),
            norm_score=True  # 0-1
        )
        logger.info(f"Step 4: '{item_names[index]}' 搜索完成。找到 {len(response[0]) if response else 0} 个匹配项。")

        """
            [
                [
                    {id:xxx, distance:0.xxxx, entity:{item_name:xxx}},
                    {id:xxx, distance:0.xxxx, entity:{item_name:xxx}},
                    ...
                ]
            ]
        """
        # 结果解析
        matches = []  # 当前item对应的匹配结果matches
        if response and len(response) > 0:
            for hit in response[0]:  # 取对应的第一个列表,混合查询只会有一个结果因为会合并
                entity = hit.get("entity", {})
                hit_name = entity.get("item_name", "")
                score = hit.get("distance", 0)
                if hit_name:
                    matches.append(
                        {
                            "item_name": hit_name,
                            "score": score
                        }
                    )
    # 提取查询结果封装返回的数据格式
        final_results.append(
            {
                "extracted_name": item_name,  # 模型给的
                "matches": matches  # 查询到的
            }
        )

    logger.info(f"Step 4: 所有商品名的向量化和Milvus搜索完成，返回结果数量: {len(final_results)}，结果为{final_results}")
    return final_results


def step_5_align_item_names(query_results):
    """
    通过向量数据库查询的分数结果处理出确定的和可选的item_name
    :param query_results: [{extracted_name: item_name, matches: [{item_name: str, score: float}, ...], ...},
                            ...]
    :return:{confirmed_item_names: [item_name, ...], options_item_names: [item_name, ...]}
    评分规则：
    确定
    0.85(根据权重和数据可以进行调整)
    可选
    0.6
    忽略
    思路：循环处理每个item_name列表，高分就要一个，可选可以要2个
    """
    # 1.准备确认和可选列表
    confirmed_item_names = []
    options_item_names = []
    # 2.处理query_results
    for item in query_results:
        extracted_name = item.get("extracted_name", "")
        matches = item.get("matches", [])
        # 3.分数排序：列表推导提取0.85-0.6的item_name
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        # >=0.85，没有高分才处理低分
        high_score_matches = [x for x in matches if x.get("score", 0) >= 0.85]
        middle_score_matches = [x for x in matches if 0.6 <= x.get("score", 0) < 0.85]
        # 只有一个高分就获取一个
        if len(high_score_matches) == 1:
            confirmed_item_names.append(high_score_matches[0].get("item_name", ""))
            continue
        elif len(high_score_matches) > 1:
            # 名字相同可能分数不是最高的
            # 如果有多个高分，优先考虑名字相同，再选择分数最高的
            same_name_item = [x for x in high_score_matches if x.get("item_name") == extracted_name]
            if same_name_item:
                confirmed_item_names.append(same_name_item[0].get("item_name", ""))
                continue
            else:
                confirmed_item_names.append(high_score_matches[0].get("item_name", ""))
                continue
        # 没有高分的，处理可选分数列表，可以多带几个
        if len(middle_score_matches) > 0:
            # 取前两个可选
            for item in middle_score_matches[:2]:
                options_item_names.append(item.get("item_name", ""))
            continue
        logger.info(f"没有匹配的item_name: {extracted_name}")

    # 去重，因为多个“模型提取名称”可能最终匹配到同一个标准商品名
    result = {
        # set会打乱顺序，dict可以保留第一次插入的顺序
        "confirmed_item_names": list(dict.fromkeys(confirmed_item_names)),
        "options_item_names": list(dict.fromkeys(options_item_names))
    }

    logger.info(f"处理结果为:{result}")

    return result


def step_6_deal_list(state, align_result, history, rewritten_query):
    """
    根据集合类型中数据判定是否要赋值answer内容，根据对齐结果更新会话状态（State），决定后续流程分支
    :param rewritten_query:
    :param state:
    :param align_result:
    :param history:
    :return:
    """
    # 1. 获取两个集合确认、可选
    confirmed_item_names = align_result.get("confirmed_item_names", [])
    options_item_names = align_result.get("options_item_names", [])
    # 2. 确认集合有数据
    if len(confirmed_item_names) > 0:
        state['item_names'] = confirmed_item_names
        state['rewritten_query'] = rewritten_query
        state['history'] = history
        if "answer" in state:  # 若状态中存在旧答案，删除（避免干扰后续流程）
            del state["answer"]
        logger.info(f"有确定的item_name:{confirmed_item_names}")
        return state
    # 3. 确认没数据处理可选集合，因为直接跳过后续所以不用给state的history赋值
    if len(options_item_names) > 0:
        options_names = '、'.join(options_item_names)
        answer = f"您是想问以下哪个产品：{options_names}？请明确一下型号"
        logger.info(f"有可选的item_name:{options_item_names}")
        state["answer"] = answer
        return state
    # 4. 处理都没数据的情况
    state["answer"] = "抱歉，我没有找到您要查询的商品。"
    logger.info("没有找到匹配的商品。")
    return state

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    目标：
    1.提取 item_name  大模型从 历史对话+本次提问 提取item_name->向量库搜索->打分
    2.利用模型充血用户问题确保后续查询召回率更高
    参数：state['original_query']、session_id
    响应：item_names: List[str]  # 提取出的商品名称
          rewritten_query: str  # 改写后的问题
          history: list  # 历史对话记录

    1.获取历史聊天记录(依据)
    2.保存包括当前聊天的聊天记录
    3.用LLM提取item_name和重写提问内容
    4.用item_name去向量库查询
    5.对查询结果进行打分和分类处理：确认集合和可选集合
    6.处理确认和可选集合：确认=》继续下一个节点||有可选或没有item_name->赋值answer
    7.补充state状态
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 1.获取历史聊天记录(依据)
    history = get_recent_messages(session_id=state["session_id"], limit=10)
    # 2.保存包括当前聊天的聊天记录(提问)
    message_id = save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        # rewritten_query=state.get("rewritten_query", ""),
        # item_names=state.get("item_names", []),
        # image_urls=state.get("image_urls", [])
    )
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")
    # 3.用LLM提取item_name和重写提问内容
    # 重写原因：消除指代起义，明确主体；补全上下文；去掉口语和冗余；润色问题增加召回率
    extract_res = step_3_extract_info(state["original_query"], history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", state["original_query"])
    # 4.用item_name去向量库查询
    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        # 通过查询结果处理出确定的和可选的item_name
        # 参数：query_results 返回：{确定item_name:[], 可选item_name:[]}
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到商品名，跳过向量检索")

    # 6.处理确认和可选集合：确认 =》继续下一个节点 | | 有可选或没有item_name->赋值answer
    state = step_6_deal_list(state, align_result, history, rewritten_query)

    # 7.记录本次聊天对话(answer)
    if state.get("answer"):
        # 对话结束
        save_chat_message(
            session_id=state["session_id"],
            role="assistant",
            text=state["answer"],
            item_names=state.get("item_names", []),
            image_urls=[],
            rewritten_query = ""  # 助手消息不需要重写
        )

    # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
    save_chat_message(
        session_id=state["session_id"],  # 会话ID，关联所属会话
        role="user",  # 消息角色：用户
        text=state["original_query"],  # 消息内容：用户原始查询
        rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
        item_names=state.get("item_names", []),  # 补充关联的商品名列表
        message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
    )

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    # 7.补充state状态
    return state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "万用表RS-12怎么用？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")