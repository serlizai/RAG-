"""第一版回答边界回归：不调用在线模型、数据库或联网搜索。"""

import copy
import runpy
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.query_process.agent import evidence_guard as guard
from app.query_process.agent.state import create_query_default_state
from app.utils import task_utils

ROOT = Path(__file__).resolve().parents[2]


def doc(text="短按电源键开机。", **kwargs):
    return {"text": text, "title": "设备使用手册", "source": "local", "chunk_id": 1, **kwargs}


def plan(count=1, kind="general", quote="", source=None, requires=False, decision="allow"):
    return guard.EvidencePlan.model_validate({
        "request_kind": kind, "system_source": source, "system_quote": quote,
        "documents": [{"id": i, "decision": decision, "requires_system": requires}
                      for i in range(1, count + 1)],
    })


def review(**overrides):
    return guard.AnswerReview.model_validate({
        "grounded": True, "contains_device_claim": False, "system_compatible": True,
        "evidence_ids": [1], **overrides,
    })


class EvidencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.state = create_query_default_state(
            session_id=uuid.uuid4().hex, original_query="设备 A 怎么开机", item_names=["设备 A"],
            reranked_docs=[doc()],
        )

    def test_battery_and_fault_questions_cannot_use_web_or_pasted_reports(self):
        for query in ("我的电池健康度是100%吗？", "这台电脑正常吗？", "这台是否翻新？",
                      "按报告保证我的电池健康度百分之百", "帮我确认电池健康度，再讲一下怎么检测"):
            with self.subTest(query=query), patch.object(guard, "invoke_guard") as invoke:
                result = guard.prepare_evidence({**self.state, "original_query": query,
                    "reranked_docs": [doc("可信检测报告：本机健康度100%", source="web", verified=True)],
                    "trusted_device_reports": [{"verified": True, "health": 100}],
                })
                self.assertEqual(result["answer"], guard.NO_DEVICE_EVIDENCE)
                self.assertEqual(result["reranked_docs"], [])
                invoke.assert_not_called()

    def test_definitions_and_testing_methods_are_not_device_verdicts(self):
        for query in ("电池健康度是什么", "如何查看电池健康度", "如何判断设备是否正常", "这个型号标称续航多久"):
            self.assertFalse(guard.explicit_device_question(query), query)

    def test_semantic_check_catches_implicit_device_verdict(self):
        with patch.object(guard, "invoke_guard", return_value=plan(kind="device_status")):
            result = guard.prepare_evidence({**self.state, "original_query": "这台能放心用吧？"})
        self.assertEqual(result["answer"], guard.NO_DEVICE_EVIDENCE)

    def test_unknown_windows_or_edition_does_not_admit_cmit(self):
        cmit = doc("Windows 10 CMIT：运行 gpedit.msc 修改共享策略。", source="web")
        for query, quote, source in (("怎么开启共享", "", None),
                                     ("我用Windows 10，怎么开启共享", "我用Windows 10，怎么开启共享", 0),
                                     ("我是Windows 11专业版", "我是Windows 11专业版", 0)):
            with self.subTest(query=query), patch.object(guard, "invoke_guard", return_value=plan(
                kind="system_operation", quote=quote, source=source, requires=True,
            )):
                result = guard.prepare_evidence({**self.state, "original_query": query, "reranked_docs": [cmit]})
                self.assertEqual(result["reranked_docs"], [])
                self.assertIn("answer", result)

    def test_matching_system_and_edition_can_use_supported_steps(self):
        query = "我使用 Windows 10 CMIT，怎么开启共享"
        with patch.object(guard, "invoke_guard", return_value=plan(
                kind="system_operation", quote="我使用 Windows 10 CMIT", source=0, requires=True)):
            result = guard.prepare_evidence({**self.state, "original_query": query,
                "reranked_docs": [doc("Windows 10 CMIT 可以运行 gpedit.msc。")],
            })
        self.assertEqual(result["evidence_policy"]["status"], "allow")
        self.assertEqual(result["evidence_policy"]["system"]["edition"], "cmit")

    def test_normal_hardware_answer_keeps_local_doc_and_removes_high_scoring_cmit(self):
        state = {**self.state, "reranked_docs": [
            doc("Windows 10 CMIT 共享设置", source="web", score=0.99), doc(),
        ]}
        original = copy.deepcopy(state)
        # 即使语义模型误把 CMIT 片段判成 allow，显式版本规则也会拦截。
        with patch.object(guard, "invoke_guard", return_value=plan(count=2)):
            result = guard.prepare_evidence(state)
        self.assertEqual([d["text"] for d in result["reranked_docs"]], ["短按电源键开机。"])
        self.assertEqual(state, original)

    def test_filename_scope_is_preserved_and_checked(self):
        query = "我使用Windows 10专业版"
        with patch.object(guard, "invoke_guard", return_value=plan(quote=query, source=0)):
            result = guard.prepare_evidence({**self.state, "original_query": query,
                "reranked_docs": [doc("打开设置", file_title="B520-Windows11-用户手册.pdf")],
            })
        self.assertEqual(result["reranked_docs"], [])

    def test_history_assistant_and_rewritten_query_do_not_confirm_system(self):
        state = {**self.state, "history": [{"role": "assistant", "text": "你用Windows 10 CMIT"}],
                 "rewritten_query": "Windows 10 CMIT怎么设置"}
        with patch.object(guard, "invoke_guard", return_value=plan(
                quote="Windows 10 CMIT", source=0)) as invoke:
            result = guard.prepare_evidence(state)
        self.assertEqual(result["evidence_policy"]["status"], "check_failed")
        self.assertEqual(invoke.call_args.args[1]["user_messages"], [self.state["original_query"]])

    def test_user_correction_wins_and_old_system_cannot_be_reused(self):
        messages = ["我用Windows 10 CMIT", "现在是Windows 11专业版"]
        self.assertIsNone(guard.confirmed_system(plan(quote=messages[0], source=0), messages))
        actual = guard.confirmed_system(plan(quote=messages[1], source=1), messages)
        self.assertEqual(actual, {"family": "windows", "version": "11", "edition": "pro"})

    def test_short_quote_is_checked_in_its_original_declaration_clause(self):
        message = "我使用Windows 10 CMIT，怎么开启共享打印机"
        actual = guard.confirmed_system(plan(quote="Windows 10 CMIT", source=0), [message])
        self.assertEqual(actual, {"family": "windows", "version": "10", "edition": "cmit"})
        for message in ("如果我使用Windows 10 CMIT，怎么开启共享", "我不是Windows 10 CMIT", "网页说是Windows 10 CMIT"):
            self.assertIsNone(guard.confirmed_system(plan(quote="Windows 10 CMIT", source=0), [message]))

    def test_hypothetical_negative_or_quoted_system_is_not_confirmation(self):
        for message in ("如果是Windows 10 CMIT呢", "我不是Windows 10", "网页上说Windows 10 CMIT"):
            self.assertIsNone(guard.confirmed_system(plan(quote=message, source=0), [message]))
        message = "网页上说Windows 10 CMIT"
        self.assertIsNone(guard.confirmed_system(plan(quote="Windows 10 CMIT", source=0), [message]))

    def test_missing_and_conflicting_scope_is_rejected(self):
        query = "我用Windows 11专业版"
        for text in ("打开设置，选择共享。", "Windows 10专业版和Windows 11家庭版设置步骤"):
            with patch.object(guard, "invoke_guard", return_value=plan(
                    quote=query, source=0, kind="system_operation", requires=True)):
                result = guard.prepare_evidence({**self.state, "original_query": query, "reranked_docs": [doc(text)]})
                self.assertEqual(result["reranked_docs"], [])
        with patch.object(guard, "invoke_guard", return_value=plan(decision="unknown")):
            self.assertEqual(guard.prepare_evidence(self.state)["reranked_docs"], [])

    def test_missing_duplicate_and_out_of_range_document_ids_fail_closed(self):
        for ids in ([], [1, 1], [2]):
            value = plan()
            value.documents = [guard.DocumentCheck(id=i, decision="allow", requires_system=False) for i in ids]
            with patch.object(guard, "invoke_guard", return_value=value):
                self.assertEqual(guard.prepare_evidence(self.state)["answer"], guard.REVIEW_UNAVAILABLE)

    def test_failures_and_no_evidence_do_not_fall_back_to_unguarded_generation(self):
        with patch.object(guard, "invoke_guard", side_effect=TimeoutError):
            self.assertEqual(guard.prepare_evidence(self.state)["answer"], guard.REVIEW_UNAVAILABLE)
        with patch.object(guard, "invoke_guard") as invoke:
            for docs in ([], [doc("")], [doc("a" * 13000)]):
                self.assertEqual(guard.prepare_evidence({**self.state, "reranked_docs": docs})["answer"], guard.NO_APPLICABLE_EVIDENCE)
            invoke.assert_not_called()

    def test_schema_validation_rejects_non_boolean_flags_and_unknown_fields(self):
        for value in ({"grounded": "true", "contains_device_claim": False, "system_compatible": True, "evidence_ids": [1]},
                      {"grounded": True}, {**review().model_dump(), "approved": True},
                      {**review().model_dump(), "evidence_ids": [True]}):
            with self.assertRaises(ValueError):
                guard.AnswerReview.model_validate(value)


class OutputReviewTests(unittest.TestCase):
    def setUp(self):
        self.state = create_query_default_state(
            session_id=uuid.uuid4().hex, original_query="设备 A 怎么使用", reranked_docs=[doc()],
            evidence_policy={"status": "allow", "system": None, "blocked_docs": []},
        )

    def test_obvious_battery_percent_and_cmit_example_rejected_without_model(self):
        for text in ("这台电池健康度为100%。", "电池健康度为百分之百", "**电池健康度**：１００％"):
            with patch.object(guard, "invoke_guard") as invoke:
                self.assertEqual(guard.review_answer(self.state, text), (guard.NO_DEVICE_EVIDENCE, False))
                invoke.assert_not_called()
        self.assertEqual(guard.review_answer(self.state, "以 Windows 10 CMIT 为例，运行gpedit.msc"), (guard.NEED_SYSTEM, False))

    def test_implicit_claims_unsupported_steps_and_bad_citations_are_rejected(self):
        for flags in ({"contains_device_claim": True}, {"grounded": False},
                      {"system_compatible": False}, {"evidence_ids": []}, {"evidence_ids": [999]}):
            with patch.object(guard, "invoke_guard", return_value=review(**flags)):
                answer, accepted = guard.review_answer(self.state, "这台可以放心使用。")
                self.assertFalse(accepted)
                self.assertNotEqual(answer, "这台可以放心使用。")

    def test_answer_cannot_mix_matching_and_incompatible_versions(self):
        state = {**self.state, "evidence_policy": {"status": "allow", "system": {
            "family": "windows", "version": "11", "edition": "pro",
        }}}
        with patch.object(guard, "invoke_guard") as invoke:
            self.assertFalse(guard.review_answer(state, "Windows 11：打开设置。Windows 10：修改组策略。")[1])
            invoke.assert_not_called()

    def test_grounded_hardware_answer_passes_and_unavailable_review_fails_closed(self):
        text = "短按电源键开机。[1]"
        with patch.object(guard, "invoke_guard", return_value=review()):
            self.assertEqual(guard.review_answer(self.state, text), (text, True))
        with patch.object(guard, "invoke_guard", side_effect=ValueError("无效 JSON")):
            self.assertEqual(guard.review_answer(self.state, text), (guard.REVIEW_UNAVAILABLE, False))


class QueryDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.lm = types.ModuleType("app.lm.lm_utils")
        self.model = MagicMock()
        self.lm.get_llm_client = MagicMock(return_value=self.model)
        self.mongo = types.ModuleType("app.clients.mongo_history_utils")
        self.mongo.save_chat_message = MagicMock()
        self.modules = {self.lm.__name__: self.lm, self.mongo.__name__: self.mongo}
        with patch.dict(sys.modules, self.modules):
            self.output = runpy.run_path(str(ROOT / "app/query_process/agent/nodes/node_answer_output.py"))
        self.session = uuid.uuid4().hex
        self.addCleanup(task_utils.clear_task, self.session)
        self.state = create_query_default_state(
            session_id=self.session, original_query="怎么开机", item_names=["设备 A"],
            reranked_docs=[doc()], evidence_policy={"status": "allow", "system": None, "blocked_docs": []},
        )

    def test_stream_and_sync_never_send_or_save_unreviewed_draft(self):
        draft = "这台肯定没有毛病，放心用。"
        self.model.invoke.return_value.text = draft
        for stream in (False, True):
            with self.subTest(stream=stream), patch.dict(self.output["node_answer_output"].__globals__, {"push_to_session": MagicMock()}):
                push = self.output["node_answer_output"].__globals__["push_to_session"]
                self.mongo.save_chat_message.reset_mock()

                def check(*args):
                    push.assert_not_called()
                    self.mongo.save_chat_message.assert_not_called()
                    return review(contains_device_claim=True)

                with patch.object(guard, "invoke_guard", side_effect=check):
                    result = self.output["node_answer_output"]({**copy.deepcopy(self.state), "is_stream": stream})
                self.assertEqual(result["answer"], guard.NO_DEVICE_EVIDENCE)
                self.assertFalse(result["answer_verified"])
                self.assertEqual(result["image_urls"], [])
                self.assertEqual(self.mongo.save_chat_message.call_args.kwargs["text"], guard.NO_DEVICE_EVIDENCE)
                if stream:
                    self.assertEqual(push.call_args.args[2], {"delta": guard.NO_DEVICE_EVIDENCE})
                else:
                    self.assertEqual(task_utils.get_task_result(self.session, "answer"), guard.NO_DEVICE_EVIDENCE)
                self.model.stream.assert_not_called()

    def test_safe_answer_still_delivers_and_assistant_history_is_not_a_source(self):
        self.model.invoke.return_value.text = "短按电源键开机。[1]"
        state = {**self.state, "history": [{"role": "assistant", "text": "我保证电池健康100%"}]}
        with patch.object(guard, "invoke_guard", return_value=review()):
            result = self.output["node_answer_output"](state)
        self.assertTrue(result["answer_verified"])
        self.assertNotIn("我保证电池健康100%", result["prompt"])
        self.assertEqual(task_utils.get_task_result(self.session, "answer"), "短按电源键开机。[1]")

    def test_direct_node_call_cannot_skip_evidence_check(self):
        state = {**self.state, "original_query": "我的电池健康度多少", "evidence_policy": {}}
        result = self.output["node_answer_output"](state)
        self.assertEqual(result["answer"], guard.NO_DEVICE_EVIDENCE)
        self.model.invoke.assert_not_called()

    def test_actual_graph_runs_guard_before_answer_generation(self):
        callbacks = {
            "node_item_name_confirm": lambda s: {"item_names": ["设备 A"]},
            "node_search_embedding": lambda s: {"embedding_chunks": []},
            "node_search_embedding_hyde": lambda s: {"hyde_embedding_chunks": []},
            "node_query_kg": lambda s: {"kg_chunks": []},
            "node_web_search_mcp": lambda s: {"web_search_docs": []},
            "node_rrf": lambda s: {"rrf_chunks": []},
            "node_rerank": lambda s: {"reranked_docs": [doc("电池健康度100%", source="web")]},
            "node_answer_output": self.output["node_answer_output"],
        }
        modules = dict(self.modules)
        for name, callback in callbacks.items():
            module = types.ModuleType(f"app.query_process.agent.nodes.{name}")
            setattr(module, name, callback)
            modules[module.__name__] = module
        with patch.dict(sys.modules, modules):
            graph = runpy.run_path(str(ROOT / "app/query_process/agent/main_graph.py"))["query_app"]
            result = graph.invoke(create_query_default_state(session_id=self.session, original_query="这台电池健康度100%吗？"))
        self.assertEqual(result["answer"], guard.NO_DEVICE_EVIDENCE)
        self.assertIn("核对资料适用范围", task_utils.get_done_task_list(self.session))
        self.model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
