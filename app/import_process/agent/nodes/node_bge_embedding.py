import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger


def step_1_validate_input(state):
    chunks = state.get('chunks')
    if not chunks or not isinstance(chunks, list):
        raise ValueError("输入数据无效，缺少有效的 chunks 列表。请确保前置节点已正确生成切片数据。")
    return chunks

# 核心词前置，item_name拼接到前面，因为权重会衰减
# 批量处理的量需要看模型序列长度(上下文窗口大小)
def step_2_generate_vectors(texts_to_embed):
    final_chunks = []  # 最终处理结果
    batch_size = 5  # 每次处理多少个chunk

    for i in range(0, len(texts_to_embed), batch_size):  # i+batch
        try:
            # 本次批量处理的chunk
            batch = texts_to_embed[i:i + batch_size]

            # 计算当前批次的起止索引，用于日志展示（方便看从1开始，也不获取下标，没有影响）
            start_idx, end_idx = i + 1, min(i + len(batch), len(texts_to_embed))

            # 当前批次的字符串
            current_text = []
            for chunk in batch:
                item_name = chunk.get('item_name')
                chunk_content = chunk.get('content')

                chunk_text = f"主体：{item_name}，内容：{chunk_content}"
                current_text.append(chunk_text)

            # 当前批次生成向量
            result = generate_embeddings(current_text)
            # 当前批次的chunk添加向量
            for j, item in enumerate(batch):
                chunk_item = item.copy()  # 防止用到原结构的地方报错
                chunk_item['dense_vector'] = result['dense'][j]
                chunk_item['sparse_vector'] = result['sparse'][j]
                final_chunks.append(chunk_item)

            logger.info(f"第{start_idx}-{end_idx}条切片：双向量生成成功")
        except Exception as e:
            logger.error(f"第{i + 1}-{i + len(batch)}条切片：双向量生成失败，错误信息：{str(e)}", exc_info=True)
            # 异常批次保留原切片数据，保证数据完整性，后续可人工排查
            final_chunks.extend(batch)
            continue

    return final_chunks


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        # 步骤1：输入数据校验，核心chunks无效则抛出异常
        texts_to_embed = step_1_validate_input(state)

        # 步骤2：给每个chunks里面的chunk的content、item_name生成向量，因为问题一般会带主语，否则会查错content

        output_data = step_2_generate_vectors(texts_to_embed)

        # 步骤3: 完善chunks的属性 添加稠密和稀疏向量
        state['chunks'] = output_data
        logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(output_data)} 条文本切片 ---")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"{current_node}节点执行失败：{str(e)}", exc_info=True)
    finally:
        logger.info(f">>> [Stub] 完成节点: {current_node},当前状态为{state}")
        add_done_task(state.get("task_id", ""), current_node)

    # 返回更新后的状态对象，传递至下游节点
    return state


