import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
from scipy.signal import step

# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_1_validate(state):
    # 提取待入库的切片数据
    chunks_json_data = state.get("chunks")
    # 校验1：chunks非空
    if not chunks_json_data:
        logger.error("Milvus入库校验失败：state中chunks字段为空")
        raise ValueError("错误: chunks为空，无法执行Milvus入库")
    # 校验2：chunks为非空列表
    if not isinstance(chunks_json_data, list) or len(chunks_json_data) == 0:
        logger.error("Milvus入库校验失败：chunks非列表类型或为空列表")
        raise ValueError("错误: chunks数据格式不正确，必须为非空列表")
    # 校验3：切片包含dense_vector字段（向量化节点核心产出）
    first_chunk = chunks_json_data[0]
    if 'dense_vector' not in first_chunk:
        logger.error("Milvus入库校验失败：切片缺失dense_vector字段，上游向量化节点可能执行失败")
        raise ValueError("错误: 数据中缺失dense_vector字段，请检查上游向量化节点执行状态")

    logger.info(f"Milvus入库校验通过，待入库切片数：{len(chunks_json_data)}")

    return chunks_json_data


def step_2_prepare_collections(state):
    """
    创建chunks对应的集合
    :param state:
    :return:
    """
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 判断是否存在集合，不存在就创建
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 创建集合对应的列
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True  # 允许插入 schema 里没定义的字段
        )
        # 将属性加入列
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

        # 为了查询向量快建立索引，根据谁查就给谁索引
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",  # 哪个列加索引
            index_name="dense_vector_index",  # 索引名字
            index_type="HNSW",  # 查找算法 一般是IVF或HNSW(图)系列
            metric_type="COSINE",  # 向量匹配和对比 IP/cosine 稠密一般是cosine
            params={
                # 10000 M-16,efConstruction-200;50000 M-32,efConstruction-300;100000 M-64,efConstruction-400
                "M": 32,  # 每个节点在构建时连接的最大邻居数
                "efConstruction": 300  # 构建阶段搜索的候选队列大小
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
            collection_name=milvus_config.chunks_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )

    return milvus_client


def step_3_delete_old_data(milvus_client, item_name):
    """
    根据item_name删除旧数据
    :param milvus_client:
    :param item_name:
    :return:
    """
    safe_name = escape_milvus_string(item_name)
    milvus_client.delete(collection_name=milvus_config.chunks_collection, filter=f'item_name == "{safe_name}"')
    milvus_client.flush(collection_name=milvus_config.chunks_collection)
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)


def step_4_insert_collections(milvus_client, chunks):
    """
    插入集合数据
    :param milvus_client:
    :param chunks:
    :return:
    """
    result = milvus_client.insert(collection_name=milvus_config.chunks_collection, data=chunks)
    milvus_client.flush(collection_name=milvus_config.chunks_collection)
    insert_count = result.get("insert_count", 0)
    logger.info(f"Milvus入库成功，插入切片数：{insert_count}")

    # 获取回显的ids
    ids = result.get("ids", [])

    # id赋值给chunks
    if ids and len(ids) == len(chunks):
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[i]

    return chunks


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)

    try:
        # 1.校验输入数据
        chunks = step_1_validate(state)
        # 2.创建集合collection、fields、索引
        milvus_client = step_2_prepare_collections(state)
        # 3.删除旧数据
        step_3_delete_old_data(milvus_client, chunks[0]['item_name'])  # chunks的item_name相同
        # 4.插入chunks数据
        with_id_chunks = step_4_insert_collections(milvus_client, chunks)
        state["chunks"] = with_id_chunks
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"{current_node}节点执行失败：{str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f">>> [Stub] 完成节点: {current_node},当前状态为{state}")
        add_done_task(state.get("task_id", ""), current_node)

    # 返回更新后的状态对象，传递至下游节点
    return state


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")