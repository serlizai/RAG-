import sys

from RAG.app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    进入和结束的日志和任务状态的配置
    参数校验 local_dir没有的话要有默认值 local_file_path完成字面意思的校验，深入校验文件是否真的存在
    1. 调用 MinerU (magic-pdf) 工具，返回一个下载文件的url地址
    2. 将 PDF 转换成 Markdown 格式。
    3. 根据返回的下载文件的url地址，下载zip包，赋值md_path
    4.将结果保存到 state["md_content"]。
    这里需要try-except，因为涉及到了第三方
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> {function_name} 开始执行; 当前状态为{state}")
    add_running_task(state["task_id"], function_name)  

    try:
        # 参数校验，返回校验完成可以直接使用的路径对象
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state["pdf_path"], state["local_dir"])
        # 调用mineru进行pdf解析(local_file_path)返回一个下载文件的url地址
        # 参数是要解析的pdf文件路径
        zip_url = step_2_upload_poll(pdf_path_obj)
        # 下载zip包并解压提取，返回md_path
        # 参数：要下载的地址 解压的文件夹 文件名
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    except Exception as e:
        # 处理异常，记录日志并更新状态
        logger.error(f">>> [{function_name}] 执行MinerU解析异常: {e}")
        raise # TODO: 这里直接抛出异常，终止工作流，后续可以考虑更优雅的处理方式
    finally:
        # 结束的日志和任务状态的配置
        logger.info(f">>> [{function_name}] 执行结束; 现在状态为{state}")  
        add_done_task(state["task_id"], function_name)

    return state