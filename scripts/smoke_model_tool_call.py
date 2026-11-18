"""Explicit, opt-in smoke check for the configured model's tool-call round trip.

Run from the repository root:

    .venv/bin/python -m scripts.smoke_model_tool_call

Importing this module never contacts the model service. Running it makes two
model requests: one tool call and one answer after the tool result.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


ROOT = Path(__file__).resolve().parents[1]


class ProbeError(Exception):
    """The configured model could not complete a valid tool round trip."""


@tool
def make_probe_token(a: int, b: int) -> str:
    """Return a deterministic test token for two integers without external effects."""
    return hashlib.sha256(f"rag-tool-probe:{a}:{b}".encode()).hexdigest()[:16]


def validate_config(environ: Mapping[str, str]) -> None:
    """Fail before constructing a client when required local settings are absent."""
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_DEFAULT_TEMPERATURE")
               if not environ.get(name)]
    if missing:
        raise ProbeError("缺少配置：" + ", ".join(missing) + "。请检查项目根目录的 .env。")


def configured_client():
    # The existing config module reads .env when imported. Load the repository's
    # .env explicitly so this command also works from a different working directory.
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    validate_config(os.environ)
    from app.lm.lm_utils import get_llm_client

    # Tool calling should not inherit the project's forced JSON response mode.
    return get_llm_client(json_mode=False)


def model_failure(phase: str, exc: Exception) -> ProbeError:
    """Describe capability failures without exposing upstream response bodies."""
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError"}:
        return ProbeError(
            f"{phase}无法连接模型服务（{name}）；工具调用和 extra_body 兼容性尚未验证。"
            "请检查 OPENAI_BASE_URL、网络及服务状态。"
        )
    status = getattr(exc, "status_code", None)
    if status == 401:
        return ProbeError(f"{phase}模型服务拒绝认证（HTTP 401）；请检查 OPENAI_API_KEY。")
    if status == 400:
        return ProbeError(
            f"{phase}模型服务拒绝请求（HTTP 400）；请检查 tools 及现有客户端 extra_body 参数兼容性。"
        )
    return ProbeError(f"{phase}模型调用失败（{name}）；请检查模型服务日志。")


async def run_probe(client, *, timeout_seconds: float = 45.0) -> str:
    """Exercise tool binding, async model calls, ToolMessage pairing and answer."""
    try:
        model_with_tools = client.bind_tools([make_probe_token])
    except Exception as exc:
        raise ProbeError(f"工具绑定失败（{type(exc).__name__}）；检查模型客户端的 bind_tools 支持。") from exc

    messages = [
        SystemMessage(content="Call the supplied tool exactly once, then repeat its returned token exactly."),
        HumanMessage(content="Use make_probe_token with a=6 and b=7. After receiving the tool result, answer with the token."),
    ]
    try:
        first = await asyncio.wait_for(model_with_tools.ainvoke(messages), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise ProbeError("首轮异步模型调用超时；检查模型服务连接和异步调用支持。") from exc
    except Exception as exc:
        raise model_failure("首轮", exc) from exc

    if not isinstance(first, AIMessage):
        raise ProbeError(f"首轮返回 {type(first).__name__}，预期为 AIMessage。")
    calls = first.tool_calls
    if first.invalid_tool_calls:
        raise ProbeError("首轮包含无法解析的工具调用；检查模型返回的工具参数格式。")
    if not calls:
        raise ProbeError("首轮没有 AIMessage.tool_calls；检查模型的工具调用支持。")
    if len(calls) != 1:
        raise ProbeError(f"预期一次工具调用，实际收到 {len(calls)} 次。")

    call = calls[0]
    if call.get("name") != make_probe_token.name:
        raise ProbeError(f"模型选择了非预期工具：{call.get('name')!r}。")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ProbeError("工具调用缺少有效 tool_call_id。")
    args = call.get("args")
    if not isinstance(args, dict) or args != {"a": 6, "b": 7}:
        raise ProbeError(f"工具参数解析结果错误：{args!r}，预期 a=6、b=7。")

    try:
        value = make_probe_token.invoke(args)
    except Exception as exc:
        raise ProbeError(f"本地工具执行失败（{type(exc).__name__}）。") from exc
    expected_token = hashlib.sha256(b"rag-tool-probe:6:7").hexdigest()[:16]
    if value != expected_token:
        raise ProbeError("本地工具结果错误；探测令牌不匹配。")
    tool_message = ToolMessage(content=value, name=make_probe_token.name, tool_call_id=call_id)
    if tool_message.tool_call_id != first.tool_calls[0]["id"]:
        raise ProbeError("ToolMessage 与 AIMessage.tool_calls 的 ID 不匹配。")

    # Keep the originating AIMessage and its paired ToolMessage together. The
    # second request uses the ordinary client so the model must give an answer.
    messages.extend((first, tool_message))
    try:
        final = await asyncio.wait_for(client.ainvoke(messages), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise ProbeError("工具结果回传后的异步模型调用超时。") from exc
    except Exception as exc:
        raise model_failure("工具结果回传后", exc) from exc

    if not isinstance(final, AIMessage):
        raise ProbeError(f"最终返回 {type(final).__name__}，预期为 AIMessage。")
    if final.tool_calls or final.invalid_tool_calls:
        raise ProbeError("最终响应仍包含工具调用，未形成普通回答。")
    answer = final.text.strip()
    if expected_token not in answer:
        raise ProbeError(f"最终回答未包含工具返回的令牌：{answer[:120]!r}。")
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=45.0, help="每次模型调用的超时秒数（默认 45）")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    try:
        client = configured_client()
        answer = asyncio.run(run_probe(client, timeout_seconds=args.timeout))
    except ProbeError as exc:
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:
        # Avoid printing the client configuration or upstream error body, which
        # could contain credentials supplied by a custom OpenAI-compatible API.
        print(f"FAIL: 初始化模型客户端失败（{type(exc).__name__}）；检查 .env 与依赖版本。")
        return 1
    print("PASS: 异步工具调用、参数解析、ToolMessage 配对和最终回答均通过。")
    print(f"最终回答：{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
