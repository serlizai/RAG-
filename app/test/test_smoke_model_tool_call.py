"""Offline checks for the opt-in model tool-call smoke script."""

import unittest

from langchain_core.messages import AIMessage, ToolMessage

from scripts.smoke_model_tool_call import ProbeError, make_probe_token, model_failure, run_probe, validate_config


EXPECTED_TOKEN = make_probe_token.invoke({"a": 6, "b": 7})


def tool_call(*, name="make_probe_token", args=None, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name,
        "args": {"a": 6, "b": 7} if args is None else args,
        "id": call_id,
    }])


class FakeBoundModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = list(messages)
        return self.response


class FakeClient:
    def __init__(self, first, final=None):
        self.bound = FakeBoundModel(first)
        self.final = final or AIMessage(content=EXPECTED_TOKEN)
        self.tools = None
        self.messages = None

    def bind_tools(self, tools):
        self.tools = tools
        return self.bound

    async def ainvoke(self, messages):
        self.messages = list(messages)
        return self.final


class ToolCallSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_preserves_ai_tool_call_and_paired_tool_message(self):
        client = FakeClient(tool_call(), AIMessage(content=f"结果是 {EXPECTED_TOKEN}。"))

        result = await run_probe(client)

        self.assertEqual(result, f"结果是 {EXPECTED_TOKEN}。")
        self.assertEqual([tool.name for tool in client.tools], ["make_probe_token"])
        self.assertEqual(len(client.messages), 4)
        self.assertIs(client.messages[2], client.bound.response)
        self.assertIsInstance(client.messages[3], ToolMessage)
        self.assertEqual(client.messages[3].tool_call_id, client.messages[2].tool_calls[0]["id"])
        self.assertEqual(client.messages[3].content, EXPECTED_TOKEN)

    async def test_missing_tool_call_explains_unsupported_response(self):
        client = FakeClient(AIMessage(content="无法使用工具"))
        with self.assertRaisesRegex(ProbeError, "没有 AIMessage.tool_calls"):
            await run_probe(client)
        self.assertIsNone(client.messages)

    async def test_malformed_tool_call_is_rejected(self):
        client = FakeClient(AIMessage(content="", invalid_tool_calls=[{
            "name": "make_probe_token", "args": "{bad-json", "id": "call-2", "error": "invalid JSON",
        }]))
        with self.assertRaisesRegex(ProbeError, "无法解析的工具调用"):
            await run_probe(client)
        self.assertIsNone(client.messages)

    async def test_wrong_arguments_stop_before_tool_execution(self):
        client = FakeClient(tool_call(args={"a": 6, "b": 8}))
        with self.assertRaisesRegex(ProbeError, "工具参数解析结果错误"):
            await run_probe(client)
        self.assertIsNone(client.messages)

    async def test_missing_tool_id_stops_before_follow_up(self):
        client = FakeClient(tool_call(call_id=None))
        with self.assertRaisesRegex(ProbeError, "tool_call_id"):
            await run_probe(client)
        self.assertIsNone(client.messages)

    async def test_final_answer_must_use_tool_result(self):
        client = FakeClient(tool_call(), AIMessage(content="无法计算"))
        with self.assertRaisesRegex(ProbeError, "未包含工具返回的令牌"):
            await run_probe(client)

    def test_missing_configuration_has_specific_diagnostic(self):
        with self.assertRaisesRegex(ProbeError, "OPENAI_API_KEY.*OPENAI_BASE_URL"):
            validate_config({"LLM_DEFAULT_TEMPERATURE": "0.1"})

    def test_connection_error_does_not_claim_tool_incompatibility(self):
        APIConnectionError = type("APIConnectionError", (Exception,), {})
        message = str(model_failure("首轮", APIConnectionError()))
        self.assertIn("尚未验证", message)
        self.assertIn("OPENAI_BASE_URL", message)


if __name__ == "__main__":
    unittest.main()
