# 6个接口 api状态、返回初始页面、发起提问、SSE流式处理(长连接)、查看历史对话、清空历史对话

from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 接收参数的类型
class QueryRequest(BaseModel):
    # title是字段标题，description是字段描述
    query: str = Field(..., description="用户的查询问题")
    session_id: str = Field(None, description="会话id, 可以不传递后台会自动通过uuid生成")
    is_stream: bool = Field(False, description="是否流式输出，默认False")

# api状态
@app.get("/health")
async def health_check():
    logger.info("触发后台检测检查接口")
    return {"status": "ok"}

@app.get("/chat.html")
async def chat_html():
    chat_html_path = PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found")

    return FileResponse(chat_html_path)


def run_query_graph(query: str, session_id: str, is_stream: bool):
    # 调用main_graph执行
    # 两种模式最终都会调用update，所以放在这里
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)  # is_stream为true就将结果加入到队列，sse能取到

    state = create_query_default_state(session_id=session_id, original_query=query, is_stream=is_stream)
    try:
        # 因为是节点自己调用push_to_session,只要state中is_stream为true就会流式输出，和这里关系不大
        query_app.invoke(state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.exception(f"=======session_id:{session_id}，查询发生异常: {str(e)}==============")
        # 告诉前端发生错误
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            # 告诉前端具体是什么错误
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
            # 流式后台任务到这里结束
            return None
        # 非流式请求没有SSE，异常必须交回API层
        raise


# 客户端->问题->开启graph->RAG查询结果->返回
@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    :param request: 请求参数
    :param background_tasks: 异步执行函数 is_stream来决定是否异步
    :return:
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream

    # 先阻止同一会话并发执行
    if get_task_status(session_id) == TASK_STATUS_PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="当前会话已有查询正在处理中"
        )

    # 清除上一轮任务状态
    clear_task(session_id)

    # 是否是流式处理 异步->先返回一个结果 开始处理，后台运行图，结果向前端推送
    if is_stream:
        # 只要开启流式处理业务就是将数据插入到队列中：{session_id,queue[update_task_status,add_running_task,add_done_list]}
        # 防止插入数据时队列还未创建——_session_stream
        create_sse_queue(session_id)
        # 流式就是异步  立即返回结果给前端，中间过程sse来一点点推送
        background_tasks.add_task(run_query_graph, query, session_id, is_stream)
        logger.info(f"query: {query} 已经开启异步和流式处理")

        return {
            "message": "本次查询处理中...",
            "session_id": session_id,
        }

    else:
        # 同步执行
        run_query_graph(query, session_id, is_stream)
        # 获取最后一个节点answer的结果
        answer = get_task_result(session_id, "answer")  # 获取会话结果函数
        logger.info(f"query:{query}开启同步处理，处理结果为:{answer}")
        return {
            "message": "本次查询完成",
            "answer": answer,
            "session_id": session_id,
            "done_list": []
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """
    :param session_id:
    :param request: 前端原生请求对象，可以判断是否断开连接
    :return:
    """
    logger.info("session_id: {}，已经和后台建立sse长连接".format(session_id))
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # 告诉浏览器和代理服务器不要直接使用缓存的 SSE 响应
            "Connection": "keep-alive",  # 保持连接，不要发送完一次就断开
            "X-Accel-Buffering": "no"  # 告诉 Nginx 不要缓冲响应否则看上去不是流式
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录，_id是objectID不能序列化
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        logger.info(f"session_id={session_id}查询历史对话成功,数据为:{items}")

        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    # 删除历史对话
    count = clear_history(session_id)
    logger.info(f"session_id={session_id}删除历史对话成功,删除数量:{count}")
    return {"message":f"{session_id}聊天记录删除成功", "deleted_count":count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)