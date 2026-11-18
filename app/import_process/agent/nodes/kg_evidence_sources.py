"""为模型提供可引用的原文片段；编号只在当前切片内部有效。"""

import re


# 优先在换行、HTML 表格行结束处划分；特别长的一行再按长度切开。
# 划分只添加编号，不删除空白、图片 URL、标签或 OCR 字符。
_SOURCE_BOUNDARY = re.compile(r"\r\n|[\r\n]|</tr\s*>", re.IGNORECASE)
SOURCE_MAX_CHARS = 800


def build_evidence_sources(content: str) -> list[dict]:
    """返回覆盖全文且有序的 {id, text, start, end} 列表。

    start/end 是 Python 字符位置，仅供程序还原原文，模型只需要看到 id/text。
    将各片段 text 依次拼接，必须恰好等于 content。
    """
    sources = []
    start = 0
    boundaries = [match.end() for match in _SOURCE_BOUNDARY.finditer(content)]
    if not boundaries or boundaries[-1] != len(content):
        boundaries.append(len(content))
    for boundary in boundaries:
        while start < boundary:
            end = min(start + SOURCE_MAX_CHARS, boundary)
            sources.append({
                "id": f"s{len(sources) + 1}",
                "text": content[start:end],
                "start": start,
                "end": end,
            })
            start = end
    return sources


def resolve_evidence_reference(reference: dict, content: str, sources: dict, label: str) -> str:
    """校验模型选择的起止编号，截取包含中间所有内容的连续原文。

    即使证据跨越图片、表格或多个段落，也不需要模型复制长 URL。
    只接受当前切片提供的编号；不能把模型传入的正文、位置当作真实来源。
    """
    for key in ("start", "end"):
        value = reference.get(key)
        if not isinstance(value, str) or value not in sources:
            raise ValueError(f"{label}的 evidence.{key} 必须引用当前切片的原文编号：{str(value)[:80]!r}")
    first = sources[reference["start"]]
    last = sources[reference["end"]]
    if first["start"] > last["start"]:
        raise ValueError(f"{label}的 evidence 起止编号顺序颠倒")
    # 截取完整区间，保留中间图片、否定条件等内容，禁止用省略号拼接首尾。
    evidence = content[first["start"]:last["end"]]
    if not evidence.strip():
        raise ValueError(f"{label}的 evidence 不能只引用空白原文")
    return evidence
