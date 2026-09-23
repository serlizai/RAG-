"""回答证据边界（第一版）。

当前没有接入可信的单机检测服务，因此不存在“检索文本自称检测报告即可放行”
的路径。模型负责语义识别，程序负责拒绝单机结论、匹配已确认的系统和失败兜底。
"""

import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.load_prompt import load_prompt
from app.core.logger import logger


NO_DEVICE_EVIDENCE = (
    "目前没有这台设备经过核验的单机检测记录，无法确认其电池健康度、实际性能或故障状态，"
    "也不能作出正常、无故障等保证。型号手册、网页资料和聊天中的描述不能替代本机检测。"
    "请通过售后查询或完成本机检测；如需了解检测方法，可以继续说明设备型号和系统版本。"
)
NEED_SYSTEM = (
    "目前还不能确认这些操作步骤适用于当前设备。请说明当前设备的操作系统名称、版本及版本类型"
    "（例如 Windows 11 专业版或 Windows 10 CMIT），以便核对适用步骤。"
)
NO_APPLICABLE_EVIDENCE = (
    "当前检索资料不足以支持适用于这台设备的回答。请补充具体问题、设备型号及相关系统版本，"
    "或提供对应版本的官方说明。"
)
REVIEW_UNAVAILABLE = "本次回答未能通过依据核验，暂时无法给出可靠结论，请稍后重试或联系人工客服核实。"
MAX_EVIDENCE_CHARS = 12000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DocumentCheck(StrictModel):
    id: int = Field(ge=1)
    decision: Literal["allow", "mismatch", "unknown"]
    requires_system: bool


class EvidencePlan(StrictModel):
    request_kind: Literal["general", "device_status", "system_operation"]
    system_source: int | None
    system_quote: str
    documents: list[DocumentCheck]


class AnswerReview(StrictModel):
    grounded: bool
    contains_device_claim: bool
    system_compatible: bool
    evidence_ids: list[int]


def invoke_guard(prompt_name, payload, schema):
    # 惰性导入，离线策略测试不初始化模型客户端。
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.lm.lm_utils import get_llm_client

    response = get_llm_client(json_mode=True).invoke([
        SystemMessage(content=load_prompt(prompt_name)),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], temperature=0, timeout=30)
    # JSON 模式也必须校验字段、类型与枚举；缺字段/字符串布尔值等都不能放行。
    return schema.model_validate(json.loads(response.text))


def user_messages(state):
    """只读用户原话；助手历史和改写问题不能确认设备状态或系统版本。"""
    messages = [m["text"] for m in state.get("history", [])
                if m.get("role") == "user" and isinstance(m.get("text"), str)]
    return messages + [state.get("original_query") or ""]


def plain_text(text):
    # 图片/链接 URL 中的 Windows11 文件夹名不能被误当成用户的系统声明。
    text = re.sub(r"https?://[^\s)\]>]+", "", str(text))
    return unicodedata.normalize("NFKC", text).casefold()


def system_tags(text):
    """显式系统标记。未知版本保留为空，不从型号、文件名或 CMIT 推断版本。"""
    text = plain_text(text)
    patterns = {
        "windows": r"(?:windows|win)[\s_-]*(11|10|8\.1|8|7|vista|xp)(?![\d.])",
        "macos": r"mac\s*os[\s_-]*(\d+(?:\.\d+)*)",
        "android": r"(?:android|安卓)[\s_-]*(\d+(?:\.\d+)*)",
        "ios": r"ios[\s_-]*(\d+(?:\.\d+)*)",
        "harmonyos": r"(?:harmony\s*os|鸿蒙)[\s_-]*(\d+(?:\.\d+)*)",
        "ubuntu": r"ubuntu[\s_-]*(\d+(?:\.\d+)*)",
    }
    tags = []
    for family, pattern in patterns.items():
        versions = re.findall(pattern, text)
        if family == "windows":
            editions = [name for name, pattern in (
                ("cmit", r"cmit|神州网信"), ("pro", r"专业版|\bpro\b"),
                ("home", r"家庭版|\bhome\b"), ("enterprise", r"企业版|\benterprise\b"),
                ("education", r"教育版|\beducation\b"),
            ) if re.search(pattern, text)]
            if not versions and re.search(r"windows|\bwin\b|cmit|神州网信", text):
                versions = [""]
        else:
            editions = []
        for version in versions:
            for edition in editions or [""]:
                tag = {"family": family, "version": version, "edition": edition}
                if tag not in tags:
                    tags.append(tag)
    return tags


def confirmed_system(plan, messages):
    index = plan.system_source
    quote = plan.system_quote.strip()
    if index is None:
        if quote:
            raise ValueError("系统声明缺少原话编号")
        return None
    if not 0 <= index < len(messages) or not quote or quote not in messages[index]:
        raise ValueError("系统声明无法追溯到用户原话")
    # 模型可能只引用“Windows 10 CMIT”，检查它所在的完整分句，避免既误拒绝
    # “我使用 Windows 10 CMIT”，又误接受“网页提到 Windows 10 CMIT”。
    source_text = messages[index]
    start = source_text.index(quote)
    left = re.split(r"[，,。;；!?？！\n]", source_text[:start])[-1]
    right = re.split(r"[，,。;；!?？！\n]", source_text[start + len(quote):])[0]
    clause = left + quote + right
    if re.search(r"如果|假如|假设|是不是|是否|可能|不确定|不知道|计划|准备|打算|升级到|"
                 r"网页|文章|你说|不是|以前|之前|原来|[?？]", clause):
        return None
    if quote != source_text.strip() and not re.search(r"我用|是|使用|用的|装的|运行|系统为|系统[:：]", clause):
        return None
    tags = system_tags(quote)
    # 多设备、多个版本、只说 Windows 没有版本，均不能认为已经确认。
    if len(tags) != 1 or not tags[0]["version"]:
        return None
    # 后续用户提到不同系统时不沿用旧版本（包括询问升级/对比，保守地重新确认）。
    for message in messages[index + 1:]:
        if any(tag != tags[0] for tag in system_tags(message)):
            return None
    return tags[0]


def system_matches(requirements, system):
    if not requirements:
        return True
    if system is None:
        return False
    return any(
        tag["family"] == system["family"]
        and (not tag["version"] or tag["version"] == system["version"])
        and (not tag["edition"] or tag["edition"] == system["edition"])
        for tag in requirements
    )


def explicit_device_question(query):
    """常见单机断言提前拦截；其他表达还会经过语义分类与输出复核。"""
    text = plain_text(query)
    metric = re.search(
        r"健康度|电池健康|实际容量|循环次数|老化|损耗|续航|寿命|故障|正常|没问题|"
        r"全新|翻新|维修过|进水|原装|battery\s*health|defective|faulty", text,
    )
    if not metric:
        return False
    # 定义、查询方法与通用排查可以回答；夹带单机判断/保证的混合请求仍拦截。
    method = re.search(r"怎么|如何|怎样|哪里|方法|原理|是什么|什么意思|定义|标准|排查|"
                       r"多少算|标称|额定|设计|规格|官方参数|通常|一般|how to|what is", text)
    verdict = re.search(r"保证|确定|肯定|确认|是不是|是否|有没有|到底|正常吗|没问题吧|is my", text)
    if re.search(r"(?:如何|怎么|怎样).{0,8}(?:查看|检测|检查|判断|排查|测量)", text) and not re.search(r"保证|肯定|直接.*说|帮我确认|(?:再|同时|顺便|并).{0,6}确认", text):
        return False
    return not method or bool(verdict)


def blocked_result(status, answer, blocked_docs=None):
    return {"answer": answer, "reranked_docs": [], "image_urls": [],
            "evidence_policy": {"status": status, "system": None,
                                "blocked_docs": blocked_docs or []}}


def prepare_evidence(state):
    """给后续生成节点提供允许使用的完整片段，或程序固定的兜底回答。"""
    if explicit_device_question(state.get("original_query") or ""):
        return blocked_result("device_evidence_missing", NO_DEVICE_EVIDENCE)
    docs, size = [], 0
    for doc in state.get("reranked_docs", []):
        text = doc.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        # 不截断片段，以免移除片段末尾的适用条件。
        length = len(json.dumps(doc, ensure_ascii=False))
        if size + length > MAX_EVIDENCE_CHARS:
            continue
        size += length
        docs.append(doc)
    if not docs:
        return blocked_result("no_evidence", NO_APPLICABLE_EVIDENCE)
    messages = user_messages(state)
    try:
        plan = invoke_guard("evidence_check", {
            "user_messages": messages, "item_names": state.get("item_names", []),
            "documents": [{**doc, "id": i} for i, doc in enumerate(docs, 1)],
            "trusted_device_reports": [],
        }, EvidencePlan)
        if plan.request_kind == "device_status":
            return blocked_result("device_evidence_missing", NO_DEVICE_EVIDENCE)
        if sorted(check.id for check in plan.documents) != list(range(1, len(docs) + 1)):
            raise ValueError("资料校验编号缺失、重复或越界")
        system = confirmed_system(plan, messages)
        if plan.request_kind == "system_operation" and system is None:
            return blocked_result("needs_system", NEED_SYSTEM)
        allowed, blocked = [], []
        checks = {check.id: check for check in plan.documents}
        for i, doc in enumerate(docs, 1):
            check = checks[i]
            requirements = system_tags("\n".join(str(doc.get(k) or "")
                                                 for k in ("title", "file_title", "text")))
            permitted = check.decision == "allow" and system_matches(requirements, system)
            if check.requires_system and (system is None or len(requirements) != 1):
                permitted = False
            if permitted:
                # 第一版所有检索源都是通用资料，忽略文本自称的报告/可信标记。
                allowed.append({**doc, "evidence_type": "general_reference"})
            else:
                blocked.append({"title": doc.get("title", ""), "chunk_id": doc.get("chunk_id"),
                                "reason": check.decision if check.decision != "allow" else "system_scope"})
        if not allowed:
            answer = NEED_SYSTEM if system is None and any(c.requires_system for c in plan.documents) else NO_APPLICABLE_EVIDENCE
            return blocked_result("no_applicable_evidence", answer, blocked)
        return {"reranked_docs": allowed, "evidence_policy": {
            "status": "allow", "system": system, "request_kind": plan.request_kind,
            "blocked_docs": blocked,
        }}
    except Exception as exc:
        logger.warning(f"依据适用性校验未完成：{type(exc).__name__}")
        return blocked_result("check_failed", REVIEW_UNAVAILABLE)


def review_answer(state, answer):
    """复核完整草稿。任何失败均返回程序固定文本，草稿不能进入 SSE 或历史记录。"""
    if state.get("evidence_policy", {}).get("status") != "allow":
        return REVIEW_UNAVAILABLE, False
    if not isinstance(answer, str) or not answer.strip():
        return REVIEW_UNAVAILABLE, False
    system = state["evidence_policy"].get("system")
    # 即使复核模型误放行，也拒绝显式的版本不匹配和健康度具体数值。
    if any(not system_matches([tag], system) for tag in system_tags(answer)):
        return NEED_SYSTEM if system is None else NO_APPLICABLE_EVIDENCE, False
    normalized = re.sub(r"[*_`\s]", "", plain_text(answer))
    if re.search(r"(?:健康度|电池健康|batteryhealth).{0,20}(?:\d+(?:\.\d+)?%|百分之[零一二三四五六七八九十百\d]+)", normalized):
        return NO_DEVICE_EVIDENCE, False
    try:
        docs = state.get("reranked_docs", [])
        review = invoke_guard("answer_review", {
            "user_messages": user_messages(state),
            "system": system, "trusted_device_reports": [],
            "documents": [{**doc, "id": i} for i, doc in enumerate(docs, 1)],
            "answer": answer,
        }, AnswerReview)
        if review.contains_device_claim:
            return NO_DEVICE_EVIDENCE, False
        valid_ids = set(range(1, len(docs) + 1))
        if (not review.grounded or not review.system_compatible or not review.evidence_ids
                or any(type(i) is not int or i not in valid_ids for i in review.evidence_ids)):
            return NO_APPLICABLE_EVIDENCE, False
        return answer, True
    except Exception as exc:
        logger.warning(f"回答依据复核未完成：{type(exc).__name__}")
        return REVIEW_UNAVAILABLE, False
