import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette import status

# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具


# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


# 8080/ -> import.html
@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    import_html_path = PROJECT_ROOT/"app"/"import_process"/"page"/"import.html"
    if not import_html_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(path=import_html_path, media_type="text/html")


# 定义调用import_graph的异步任务函数
# local_file_path task_id  local_dir
def run_import_graph(task_id, local_file_path, local_dir):
    """
    开启图的执行和调用
    :param task_id: 每次的标识
    :param local_file_path: 文件地址
    :param local_dir: 输出文件夹的地址
    :return:
    """
    try:
        # key:task_id, value: 任务状态
        # add_running和add_done是针对任务中每个节点的状态，update_task_status是针对整个任务的状态
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_file_path"] = local_file_path
        init_state["local_dir"] = local_dir

        # 执行图
        for even in kb_import_app.stream(init_state):  # 默认mode是value，每一步之后，都会输出当时完整的图状态
            # even是一个字典，包含当前节点的名字和状态
            for node_name, result in even.items():
                logger.info(f"节点{node_name}完成，执行结果：{result}")

        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"任务{task_id}图执行完成")
    except Exception as e:
        logger.exception("==============图执行失败发生异常==============")
        update_task_status(task_id, TASK_STATUS_FAILED)


# 8080/upload post -> 文件上传+开启导入流程
"""
    1.接收文件存储到output文件夹  /output/当天日期/uuid(taskid)/文件名
    2.异步开启import_graph图的执行【1.整个任务状态(开始和结束) 2.每个节点状态 add_running/add_done
"""
@app.post("/upload")
# ...代表没有默认值，必须提供
async def upload_file(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    :param background_tasks:
    :param files: 文件列表，传一个也是列表
    :return:
    """
    # 整理输出位置
    today_str = datetime.now().strftime("%Y%m%d")
    base_out_path = PROJECT_ROOT/"output"/today_str
    # 记录每个文件上传的任务id [task_id, ...]
    task_ids = []
    # 循环处理每个上传的文件(存储到本地) + 异步图任务调用
    for file in files:
        # file->UploadFile (.file 上传文件的输入流 .filename 上传文件名 .read 可直接读取 .contentType 获取文件mime类型)
        # 32位 时区+时间戳+网卡地址结合 不重复
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        # 记录文件上传
        add_running_task(task_id, "upload_file")
        # 文件的dir和local
        # output/YYYYMMDD/TaskID
        dir_path = base_out_path/task_id
        os.makedirs(dir_path, exist_ok=True)
        local_file_path = dir_path/file.filename

        # 上传的文件写入到local
        with local_file_path.open(mode="wb") as buffer:
            # 一边读一边写（默认每次 16KB）
            shutil.copyfileobj(file.file, buffer)  # 将file写进buffer

        # 异步调用图的执行
        # 参数1:执行的方法  参数2: *args 参数列表->执行的方法
        background_tasks.add_task(run_import_graph, task_id, str(local_file_path), str(dir_path))
        add_done_task(task_id, "upload_file")
        logger.info(f"任务{task_id}文件上传完成，异步调用图执行，文件路径：{local_file_path}")

    # 返回结果
    return {
        "code": 200,
        "message": f"文件上传完成, 文件数量: {len(files)}",
        "task_ids": task_ids,
    }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8001/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info


# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8001端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8001  # 服务端口
    )
