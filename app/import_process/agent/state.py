from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如state["session_id"]、state.get("embedding_chunks")）
    """
    task_id: str          # 任务唯一ID，用于追踪日志

    # --- 流程控制标记 ---
    is_md_read_enabled: bool   # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径


    # --- 切块相关 ---
    is_normal_split_enabled: bool  #常规切块
    is_silicon_flow_api_enabled: bool # 调用硅基流动（SiliconFlow）云服务进行智能切片
    is_advanced_split_enabled: bool  # 高级切块：更精细的语义分割策略
    is_vllm_enabled: bool  # vllm使用本地大模型进行语义理解后切割

    # --- 路径相关 ---
    local_dir: str        # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径，前端页面赋值，测试写死
    file_title: str       # 文件标题（文件名去后缀）
    pdf_path: str         # PDF 文件路径 (如果输入是PDF)
    md_path: str          # Markdown 文件路径 (转换后或直接输入的)
    split_path: str       # 分块后的文件路径
    embeddings_path: str  # 向量数据库文件路径

    # --- 内容数据 ---
    md_content: str       # Markdown 的全文内容
    chunks: list          # 切片后的文本列表，包含 metadata
    item_name: str        # 识别出的主体名称 (如: "万用表")，用于增强检索

    # --- 数据库相关 ---
    embeddings_content: list # 包含向量数据的列表，准备写入 Milvus

# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {  # ImportGraphState限定了必须有哪些 key，每个 key 是什么类型
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

# 定义时：把多个关键字参数打包成一个 dict，解包就拆成关键字参数
def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)  # 浅拷贝字典里面的列表还是同一份不会复制，所以要用深拷贝
    # 用 overrides 覆盖默认值
    # dict.update(other_dict)或dict.update(key1=value1, key2=value2)
    state.update(overrides)  # update将另一个字典的键值对合并进来。如果 key 已存在就覆盖，不存在就新增
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)