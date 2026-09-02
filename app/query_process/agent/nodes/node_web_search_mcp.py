import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerStreamableHttp
from app.core.logger import logger


async def mcp_call_streamable(query):
    """
    异步调用百炼MCP搜索服务的核心函数。

    该函数负责初始化MCP客户端，建立SSE连接，调用远程工具，并返回原始结果。

    :param query: 搜索查询词（通常是经过改写后的精准Query）
    :return: MCP返回的原始结果对象 (包含 content, isError 等字段)
    """

    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 30
        },
        max_retry_attempts=3
    )

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_config.mcp_base_url}")
        # 建立与MCP服务的http连接（异步方法，需await）
        await search_mcp.connect()

        # 获取工具列表
        tools = await search_mcp.list_tools()
        print(f"[MCP] 可用工具列表: {tools}")

        logger.info(f"[MCP] 连接成功，正在调用工具 'bailian_web_search' 查询: {query}")
        # 调用百炼MCP的搜索工具（核心步骤）
        # tool_name: "bailian_web_search" 是百炼官方定义的工具名称
        # arguments: 工具所需的参数，这里需要 "query" (查询词) 和 "count" (返回数量)
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 3}
        )
        logger.info("[MCP] 工具调用完成，已获取返回结果")
        return result

    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)
        return None

    finally:
        # 无论调用成功/失败，最终都关闭MCP连接（释放资源，异步方法）
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    # 获取重写后的问题
    query = state.get("rewritten_query")
    # 调用网络搜索方法
    result = asyncio.run(mcp_call_streamable(query))
    # 结果处理
    # {
    #   "isError": false,
    #   "content": [
    #     {
    #       "text": "{\"pages\":[{\"snippet\":\"金华今日天气 (原标题:金华今日天气) 中国天气网讯 今天是9月1日星期二,一起来看天气。 今晨6时,金华晴,气温26℃,东风4-5级,相对湿度91%。 预计,今天白天中雨,最高气温32.1℃,微风,今天夜间小雨,最低气温24.5℃,微风。 空气质量方面,今晨6时,AQI指数29,空气质量为优,未来24小时空气质量持续良好。适宜开窗通风和进行户外运动。 近期重点天气提示: 降水提醒:预计今明天有中雨转小雨。出门记得带伞并注意交通安全。 天气信息就是这么多啦~想看更多内容,请访问中国天气网移动网站:https://e.weather.com.cn!\",\"hostname\":\"网易\",\"hostlogo\":\"https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=1534926245,1016405979&fm=195&app=88&f=JPEG?w=200&h=200\",\"title\":\"金华今日天气\",\"url\":\"https://www.163.com/news/article/L5NL52U200019TLK.html\"},{\"snippet\":\"北京时 2026-08-26 05:30更新 | 数据来源中央气象台 今天 周末 7天 8-15天 今天白天 26日(周三) 阴 23 ℃ <3级 日出 06:26 今天夜间 26日(周三) 阴 10 ℃ <3级 日落 17:52 明天白天 27日(周四) 多云 19 ℃ <3级 日出 06:25 逐3小时预报 08时 11时 14时 17时 20时 23时 02时 05时 东风 <3级 北风 <3级 西北风 <3级 西北风 <3级 东北风 <3级 东北风 <3级 东北风 <3级 东北风 <3级 15°c 21°c 23°c 22°c 17°c 14°c 11°c 10°c 周边地区 巴特沃斯 17 / 12\",\"hostname\":\"天气网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i2/O1CN01MJh8gM28s6MYPHsm6_!!6000000007987-2-tps-32-32.png\",\"title\":\"【中央天气】中央今天天气预报,今天,今天天气,7天,15天天气预报,天气预报一周,天气预报15天查询\",\"url\":\"https://www.weather.com.cn/weather1dn/302050100.shtml\"}],\"request_id\":\"3301c643-2d23-4974-a863-babe379c11ec\",\"tools\":[{\"result\":\"北京市北京市2026-09-02基本天气信息：\\n天气：晴\\n温度：27\\n风向：南风\\n风力：4-5级\\n近期天气信息：2026-09-01，天气：晴、温度：30/18℃、风向：西南风;2026-09-02，天气：晴、温度：29/18℃、风向：西南风;2026-09-03，天气：晴、温度：31/19℃、风向：西南风;2026-09-04，天气：晴、温度：32/19℃、风向：西南风;2026-09-05，天气：多云、温度：31/21℃、风向：东风;2026-09-06，天气：多云、温度：30/20℃、风向：南风;2026-09-07，天气：阴、温度：29/16℃、风向：北风;2026-09-08，天气：小雨、温度：27/17℃、风向：西风;\",\"type\":\"weather\"}],\"status\":0}",
    #       "type": "text"
    #     }
    #   ]
    # }
    docs = []
    if result and not result.isError and result.content:
        # 解析MCP原始结果：提取文本内容并转为JSON对象
        # result.content 通常是一个列表，第一项包含文本结果
        raw_text = result.content[0].text
        data = json.loads(raw_text)
        pages = data.get("pages") or []

        logger.info(f"MCP 返回原始页面数量: {len(pages)}")

        # 遍历结果，统一封装为结构化格式
        for item in pages:
            snippet = (item.get("snippet") or "").strip()
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()

            # 过滤无核心摘要的结果
            if not snippet:
                continue

            docs.append({"title": title, "url": url, "snippet": snippet})
    logger.info(f"[MCP] Web搜索结果解析完成,结果为：{docs}")


    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    print("---node-web-search-mcp处理结束---")
    # 和另外的搜索并行的，不能直接返回state，每个并行节点只返回自己负责的字段，否则会覆盖一些数据
    return {"web_search_docs": docs}


if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")