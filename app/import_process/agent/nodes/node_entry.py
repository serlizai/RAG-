import os
from pathlib import Path
import sys

from RAG.app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState

# 函数名不能改因为要涉及到和前端的联动,具体在task_utils.py中有调用
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    设计的state: is_pdf_read_enabled is_md_read_enabled md_path pdf_path local_file_path file_title
    未来要实现:
    进入节点和结束节点的日志输出(节点+核心参数)
    记录任务状态(哪个任务开始了/结束了)->给前端推送信息(埋点)
    参数校验(local_file_path -> 没有传入文件 -> end  local_dir -> 没有传入输出文件夹 -> 创建一个临时的文件夹)
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled, md_path/pdf_path = local_file_path)。
    """

    # sys._getframe()获取当前调用栈的这一层，也就是“当前代码运行到哪里了”
    # f_code拿到当前这一层对应的代码对象，.co_name拿到当前代码对象的函数名
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行; 现在状态为{state}")  
    add_running_task(state["task_id"], function_name)  # 记录任务状态(哪个任务开始了/结束了)->给前端推送信息(埋点)

    # 参数校验
    locals_file_path = state['local_file_path']
    if not locals_file_path:
        logger.error(f">>> [{function_name}] 错误: local_file_path 为空;")  
        return state  # 直接返回,条件边路由函数会直接路由到end
    
    # state属性赋值
    if locals_file_path.endswith(".pdf"):
        # 处理pdf
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = locals_file_path
    elif locals_file_path.endswith(".md"):
        # 处理md
        state["is_md_read_enabled"] = True
        state["md_path"] = locals_file_path
    else:
        logger.error(f">>> [{function_name}] 错误: local_file_path 不是 PDF 或 MD 文件;")  

    # 提取file_title，去掉路径和后缀，为后期没有识别出item_name时提供兜底
    # file_title = locals_file_path.split("/")[-1].split(".")[0]
    # file_title = os.path.basename(locals_file_path).split(".")[0]  # 去掉路径和后缀
    file_title = Path(locals_file_path).stem  # stem最后一级文件名去掉最后一个后缀
    state["file_title"] = file_title

    logger.info(f">>> [{function_name}] 执行结束; 现在状态为{state}")  
    add_done_task(state["task_id"], function_name)

    # # 模拟简单的路由逻辑，防止报错 (仅 node_entry 需要)
    # if "local_file_path" in state:
    #     path = state["local_file_path"]
    #     if path.endswith(".pdf"):
    #         state["is_pdf_read_enabled"] = True
    #     elif path.endswith(".md"):
    #         state["is_md_read_enabled"] = True

    return state