"""异步联网检索服务；不依赖 LangGraph 状态、会话历史或 SSE。"""

import asyncio
import json
from typing import Any, Callable

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger
from app.query_process.services.schemas import Evidence, SearchResult


WEB_TOOL_NAME = "bailian_web_search"


class WebServiceError(Exception):
    def __init__(self, code: str, *, retryable: bool = False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


async def call_mcp_raw(
    query: str,
    *,
    server_factory: Callable[..., Any] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """执行一次 MCP 查询，校验工具存在并保证清理连接。"""
    if not isinstance(query, str) or not query.strip():
        raise WebServiceError("INVALID_ARGUMENT")
    if base_url is None:
        base_url = mcp_config.mcp_base_url
    if api_key is None:
        api_key = mcp_config.api_key
    if not base_url or not api_key:
        raise WebServiceError("NOT_CONFIGURED")

    server_factory = server_factory or MCPServerStreamableHttp
    server = server_factory(
        name="search_mcp",
        params={
            "url": base_url,
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": 30,
        },
        max_retry_attempts=3,
    )
    try:
        await server.connect()
        available_tools = await server.list_tools()
        if WEB_TOOL_NAME not in {_tool_name(tool) for tool in available_tools}:
            raise WebServiceError("TOOL_UNAVAILABLE")
        return await server.call_tool(
            tool_name=WEB_TOOL_NAME,
            arguments={"query": query.strip(), "count": 3},
        )
    finally:
        try:
            await server.cleanup()
        except Exception as exc:
            # 清理失败不覆盖连接、列举工具或查询阶段的原始错误。
            logger.warning(f"MCP 连接清理失败：{type(exc).__name__}")


def parse_mcp_result(result: Any) -> SearchResult:
    """将 MCP 原始页面转换为旧节点形状和统一证据形状。"""
    if result is None:
        return SearchResult(status="empty")
    if getattr(result, "isError", False):
        return SearchResult(status="error", error_code="REMOTE_ERROR", retryable=True)

    content = getattr(result, "content", None) or []
    if not content:
        return SearchResult(status="empty")
    raw_text = getattr(content[0], "text", None)
    if not isinstance(raw_text, str):
        return SearchResult(status="error", error_code="INVALID_RESPONSE")
    try:
        pages = json.loads(raw_text).get("pages") or []
    except (TypeError, ValueError, AttributeError):
        return SearchResult(status="error", error_code="INVALID_RESPONSE")
    if not isinstance(pages, list):
        return SearchResult(status="error", error_code="INVALID_RESPONSE")

    items: list[dict] = []
    evidence: list[Evidence] = []
    invalid_count = 0
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            invalid_count += 1
            continue
        snippet = page.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            invalid_count += 1
            continue
        title = page.get("title") if isinstance(page.get("title"), str) else ""
        url = page.get("url") if isinstance(page.get("url"), str) else ""
        item = {"title": title.strip(), "url": url.strip(), "snippet": snippet.strip()}
        items.append(item)
        evidence.append(Evidence(
            id=f"web:{item['url'] or index}",
            text=item["snippet"], title=item["title"], source="web",
            url=item["url"] or None, retrieval_method="web",
        ))

    if invalid_count:
        status = "partial"
    elif items:
        status = "ok"
    else:
        status = "empty"
    return SearchResult(
        status=status, items=items, evidence=evidence, candidate_count=len(pages)
    )


async def search_web(query: str, *, server_factory: Callable[..., Any] | None = None) -> SearchResult:
    """供未来 Agent 直接 await 的入口；故障与正常空结果分开返回。"""
    try:
        raw = await call_mcp_raw(query, server_factory=server_factory)
        if raw is None:
            return SearchResult(status="error", error_code="EMPTY_RESPONSE", retryable=True)
        return parse_mcp_result(raw)
    except WebServiceError as exc:
        return SearchResult(status="error", error_code=exc.code, retryable=exc.retryable)
    except (asyncio.TimeoutError, TimeoutError):
        return SearchResult(status="error", error_code="TIMEOUT", retryable=True)
    except Exception as exc:
        logger.warning(f"联网检索失败：{type(exc).__name__}")
        return SearchResult(status="error", error_code="UNAVAILABLE", retryable=True)
