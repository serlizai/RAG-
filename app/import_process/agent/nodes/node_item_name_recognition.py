# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from gc import enable
from typing import List, Dict, Any, Tuple

from langchain_core import messages
from narwhals import expr
# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus.client.types import MetricType

from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

"""
主要目标：
    1.利用文本大模型识别当前chunks对应的item_name，用于区分不同文档
    2.用嵌入式模型将item_name生成向量存储到向量数据库
    3.修改state[chunks]里面的属性
步骤：
    1.参数校验和取值
    2.构建上下文环境 chunks->前几个拼接成context
    3.调用模型，拼接提示词，识别chunks对应的item_name
    4.修改state[chunks]
    5.item_name生成向量稠密和稀疏
    6.存储向量 file_title item_name
"""


def step_1_get_chunks(state):
    """
    获取chunks和file_title
    :param state:
    :return:
    """

    chunks = state.get("chunks")
    file_title = state.get("file_title")

    if not chunks:
        raise ValueError("chunks没有值，无法继续执行")
    if not file_title:
        # file_title没有值可以从路径中获取
        file_title = os.path.splitext(os.path.basename(state.get('md_path')))[0]
        logger.warning(f"state中无有效file_title，已从md路径中提取兜底标题: {file_title}")
        state["file_title"] = file_title

    logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks


def step_2_build_context(chunks):
    """
    根据chunks的content内容进行拼接
    内容限制：最多截取前五个，最多字符不超过 CONTEXT_TOTAL_MAX_CHARS
    内容处理：
        切片：{1}, 标题:{title}, 内容:{content}\n\n
        切片：{2}, 标题:{title}, 内容:{content}\n\n
    :param chunks:
    :return:
    """
    # 前置准备
    parts = []  # 存储处理后的切片
    total_chars = 0  # 记录已经加入列表的字符串数量
    # 循环处理content+判断
    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        # 提取切片标题和内容，去首尾空格，过滤无效字符
        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()

        # 标题和内容均为空，跳过该无效切片
        if not (chunk_title or chunk_content):
            logger.debug(f"第{index}个切片为空白内容，已过滤")
            continue

        # 单切片内容截断：防止单个切片内容过长占满上下文
        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"第{index}个切片内容过长，已截断至{SINGLE_CHUNK_CONTENT_MAX_LEN}字符")

        data = f"切片：{index}, 标题:{chunk_title}, 内容:{chunk_content}\n\n"
        parts.append(data)
        # 累计字符数，包含分隔符
        total_chars += len(data)
        # 总字符数超限时立即停止拼接，避免大模型输入超限
        if total_chars > CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"上下文总字符数即将超限（{CONTEXT_TOTAL_MAX_CHARS}），已停止拼接后续切片")
            break
    # 结果转化
    context = "\n\n".join(parts)
    final_context = context[:CONTEXT_TOTAL_MAX_CHARS]
    # 返回结果
    logger.info(f"步骤2：上下文构建完成，最终长度{len(final_context)}字符")
    return final_context


def step_3_call_llm(context, file_title):
    """
    调用大模型识别item_name,file_title兜底
    :param context:
    :param file_title:
    :return:
    """
    try:
        # 1.构建提示词
        human_prompt = load_prompt("item_name_recognition", file_title = file_title, context = context)
        system_prompt = load_prompt("product_recognition_system")
        # 2.获取模型对象
        llm = get_llm_client(json_mode=False)
        # 3.执行调用
        messages = [
            HumanMessage(content=human_prompt),
            SystemMessage(content=system_prompt)
        ]
        response = llm.invoke(messages)
        # 4.结果判断和兜底
        item_name = getattr(response, "content", "").strip()  # 获取response的content属性，没有返回空字符串
        # 清洗返回结果：过滤空格、换行、回车、制表符等无效字符
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

        # 清洗后结果为空，使用文件标题兜底
        if not item_name:
            logger.warning("大模型返回空内容，使用文件标题作为商品名称兜底")
            return file_title

        # 5.返回结果
        logger.info(f"步骤3：大模型识别商品名称成功，结果为：{item_name}")
        return item_name

    # 捕获所有异常：大模型调用超时、网络错误、格式错误等，均不中断主流程
    except Exception as e:
        # 日志消息会自动附加当前异常的完整堆栈跟踪信息
        logger.error(f"步骤3：大模型调用失败，原因：{str(e)}", exc_info=True)
        # 异常时返回文件标题兜底，保证流程继续执行
        return file_title

def step_4_update_chunks_and_state(state, item_name, chunks):
    """
    state的item_name赋值，chunks里面的元数据item_name赋值
    :param state:
    :param item_name:
    :param chunks:
    :return:
    """
    # 1.更新state的item_name
    state["item_name"] = item_name
    # 2.更新chunks里面的元数据item_name
    for chunk in chunks:
        chunk["item_name"] = item_name
    # 同步更新state里面的chunks
    state["chunks"] = chunks
    logger.info(f"步骤4：state和chunks的item_name已更新为：{item_name}")


def step_5_generate_embeddings(item_name):
    """
    根据item_name生成向量
    :param item_name:
    :return: dense_vector, sparse_vector
    """
    if not item_name:
        logger.warning("item_name为空，跳过向量生成，返回空向量")
        return None, None
    """
        generate_embeddings函数返回一个字典，包含稠密向量和稀疏向量
        参数是一个列表允许传入多个字符串 ["1","2","3",...]
        结果：
            result = {
                "dense": [1的稠密, 2的稠密...],  # 嵌套列表，与输入文本一一对应
                 # 字典列表
                "sparse": [
                    {15: 0.56, 1024: 0.32, 4096: 0.78},   # "你好" 的稀疏向量（词权重）
                    {20: 0.44, 315: 0.91, 5000: 0.23}     # "世界" 的稀疏向量 
                ]
            }
    """
    logger.info(f"开始执行步骤5：为商品名称[{item_name}]生成BGE-M3双向量")
    try:
        vectors = generate_embeddings([item_name])
        # 向量生成结果非空，才进行后续解析
        if vectors and "dense" in vectors and "sparse" in vectors:
            # 稠密向量解析：取批量结果第一个因为就一个字符串，为Python列表（Milvus存储要求）
            dense_vector = vectors["dense"][0]
            # 稀疏向量解析：取批量结果第一个因为就一个字符串，CSR矩阵解析为字典格式
            sparse_vector = vectors["sparse"][0]
            logger.info("步骤5：BGE-M3稠密+稀疏向量生成成功")
        else:
            logger.warning("步骤5：向量生成工具返回空结果，无法提取双向量")
            dense_vector, sparse_vector = None, None
    except Exception as e:
        logger.error(f"步骤5：向量生成失败，原因：{str(e)}", exc_info=True)
        dense_vector, sparse_vector = None, None

    return dense_vector, sparse_vector


def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
    将item_name的向量存储到向量数据库
    :param file_title:
    :param item_name:
    :param dense_vector:
    :param sparse_vector:
    :return:
    """
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 判断是否存在集合，不存在就创建
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 创建集合对应的列
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True  # 允许插入 schema 里没定义的字段
        )
        # 将属性加入列
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 为了查询向量快建立索引，根据谁查就给谁索引
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  # 哪个列加索引
            index_name="dense_vector_index",  # 索引名字
            index_type="HNSW",  # 查找算法 一般是IVF或HNSW(图)系列
            metric_type="COSINE",  # 向量匹配和对比 IP/cosine 稠密一般是cosine
            params={
                # 10000 M-16,efConstruction-200;50000 M-32,efConstruction-300;100000 M-64,efConstruction-400
                "M": 16,  # 每个节点在构建时连接的最大邻居数
                "efConstruction": 200  # 构建阶段搜索的候选队列大小
            }
        )

        index_params.add_index(
            field_name="sparse_vector",  # 哪个列加索引
            index_name="sparse_vector_index",  # 索引名字
            index_type="SPARSE_INVERTED_INDEX",  # 查找算法 稀疏只有这一个
            metric_type="IP",  # 向量匹配和对比 稀疏一般用内积IP
            params={
                # 动态维护一个最低入围分数线,利用每个词的最大贡献计算出文档分数的天花板,天花板低于当前分数线就直接跳过该文档
                "inverted_index_algo": "DAAT_MAXSCORE"
            }
        )

        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )

    # 删除之前存在的item_name 数据，因为Milvus的向量数据不可原地修改
    # 要加载之后才会有数据，才能进行搜索和改动
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)  # 加载全部collection字段
    # 新api是filter不是expr
    milvus_client.delete(collection_name=milvus_config.item_name_collection, filter=f"item_name=='{item_name}'")
    # 向集合插入最新的item_name数据和向量
    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    # item要求是list
    res = milvus_client.insert(collection_name=milvus_config.item_name_collection, data=[item])
    milvus_client.flush(collection_name=milvus_config.item_name_collection)
    logger.info(f"插入结果：{res}")
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)

    logger.info(f"步骤6：[{item_name}]成功存入Milvus集合[{milvus_config.item_name_collection}]，数据：{list(item.keys())}")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}")
    add_running_task(state['task_id'], function_name)

    try:
        # 1.校验和取值(file_title chunks) file_title是兜底用的
        file_title, chunks = step_1_get_chunks(state)
        # 2.chunks->前几个拼接成context
        context = step_2_build_context(chunks)
        # 3.调用LLM,拼接提示词识别item_name
        item_name = step_3_call_llm(context, file_title)
        # 4.修改state[chunks]里面的属性
        step_4_update_chunks_and_state(state, item_name, chunks)
        # 5.item_name生成向量
        dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
        # 6.向量存储到数据库
        step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

    except Exception as e:
        logger.error(f">>>[{function_name}] 执行出错: {e}")
        raise  # 终止工作流
    finally:
        logger.info(f">>> [Stub] 完成节点: {function_name},当前状态为{state}")
        add_done_task(state['task_id'], function_name)

    return state


def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            # 在 res = milvus_client.query(...) 之前加上
            stats = milvus_client.get_collection_stats(collection_name)
            logger.info(f"集合统计：{stats}")

            # 再无条件查一下
            # all_data = milvus_client.query(
            #     collection_name=collection_name,
            #     filter="item_name != ''",
            #     output_fields=["item_name"],
            #     limit=10
            # )
            # logger.info(f"集合中所有数据：{all_data}")

            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()