"""
知识图谱入库节点：为后续把文档知识写入 Neo4j 做准备。

这个节点在导入流程中的位置：
    商品名称识别 → BGE 向量化 → Milvus 入库 → 当前节点 → END

数据来源：
    商品识别节点给切片补充 item_name；Milvus 入库节点生成并回填 chunk_id。
    当前节点通过共享的 state 读取这些切片，不需要重新读取 PDF 或重新切分文档。

当前实现范围：
    读取切片 → 校验必要字段 → 逐片抽取实体、关系 → 写入 Neo4j。
    state["kg_extractions"] 保存抽取结果，state["kg_import_stats"] 保存本批入库数量。
"""

# json 用于把输入字典转换为模型可读的文本，以及解析模型返回的 JSON。
import json
import hashlib
import os
import re
from html import unescape

from langchain_core.messages import SystemMessage, HumanMessage

# 统一日志工具：把处理进度、提示等信息交给项目现有日志配置输出。
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.import_process.agent.nodes.kg_evidence_sources import build_evidence_sources, resolve_evidence_reference
# 导入流程的状态类型：说明 state 中有哪些字段，便于编辑器提示和阅读代码。
# 类型注解本身不会检查运行时传入的数据，因此下面仍需要手动校验。
from app.import_process.agent.state import ImportGraphState
# 任务进度工具：登记某个导入任务当前运行到哪个节点。
from app.utils.task_utils import add_running_task, add_done_task


# 固定实体分类，防止同一类对象被模型随意写成不同类型。
# 对应：商品、部件、故障、操作、参数、材料、功能、条件、其他。
# 写入 Neo4j 时统一使用 Entity 标签，这些分类保存在实体的 type 属性中。
KG_ENTITY_TYPES = {
    "Product", "Component", "Fault", "Operation", "Parameter",
    "Material", "Function", "Condition", "Other",
}

# 模型输出通过 JSON 解析但内容校验失败时，附带原因再抽取一次。
# 次数包含首次请求；连接错误、超时等交给客户端处理，不在此额外重复调用。
KG_EXTRACTION_MAX_ATTEMPTS = 2

# 一次识别一个 HTML 字符引用或普通字符，保留每个字符在原文中的位置。
# 例如 &lt; 显示为 <，但最后保存的 evidence 仍应来自带 &lt; 的原文。
_EVIDENCE_TOKENS = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#[xX][0-9a-fA-F]+);|.", re.DOTALL)


def step_1_get_chunks(state: ImportGraphState) -> list[dict]:
    """
    获取用于构建知识图谱的切片。

    前置条件：
    1. 商品识别已完成，切片具有 item_name。
    2. Milvus 入库已完成，切片具有 chunk_id。

    参数 state：整个导入流程共享的字典，state["chunks"] 是切片列表。
    每个切片也是一个字典，通常包含原文、标题、商品名称、向量和切片 ID。

    返回值 list[dict]：一个新的列表，每个字典只保留图谱处理需要的字段。
    本函数不修改原 state，也不连接 Neo4j；发现不合格的切片时抛出 ValueError。
    """
    # 1. 读取上游节点处理好的全部切片。
    # get("chunks") 在字段不存在时返回 None，方便下面统一报出明确的错误。
    chunks = state.get("chunks")

    # isinstance 检查类型；not chunks 检查列表是否为空。
    # or 表示两个条件满足任意一个就报错，避免后续遍历错误的数据。
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("图谱入库失败：chunks 必须是非空列表")

    # 2. 用新列表收集整理后的切片，不直接删除原切片中的向量字段。
    # 这是当前函数的局部变量，与查询流程 state 中的 kg_chunks 字段没有自动关联。
    kg_chunks = []

    # 3. 逐个校验切片。enumerate 同时返回序号和元素；从 1 编号便于定位错误。
    for index, chunk in enumerate(chunks, start=1):
        # 先确认是字典，后面才能安全地调用 chunk.get(...)。
        if not isinstance(chunk, dict):
            raise ValueError(f"第 {index} 个切片不是字典")

        # chunk_id：Milvus 切片的主键，后续用于从图谱关系追溯到原文。
        # item_name：优先取当前切片的商品名，缺失或为空字符串时使用 state 中的商品名。
        # content：切片原文，后续由大模型从中抽取实体和关系。
        chunk_id = chunk.get("chunk_id")
        item_name = chunk.get("item_name") or state.get("item_name")
        content = chunk.get("content")

        # 当前项目的 Milvus 主键类型是 INT64。
        # Python 中 bool 是 int 的子类，isinstance(True, int) 也会返回 True，
        # 所以需要先排除布尔值。这里检查整数类型，不负责检查 INT64 范围或数据库记录是否存在。
        # 不在这里生成新 ID：Neo4j 中的引用必须与 Milvus 已有切片的 ID 一致。
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise ValueError(
                f"第 {index} 个切片缺少有效的整数 chunk_id，"
                "请先完成 Milvus 入库"
            )

        # 商品名应是非空字符串；strip() 去掉首尾空白，避免把 "   " 当成有效名称。
        # or 会短路：如果不是字符串，就不会执行后面的 .strip()。
        if not isinstance(item_name, str) or not item_name.strip():
            raise ValueError(f"第 {index} 个切片缺少 item_name")

        # 没有原文就没有可供抽取的依据，因此空文本应在调用大模型前被拦截。
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"第 {index} 个切片的 content 为空")

        # 4. 创建新的切片字典，再追加到结果列表。
        # 图谱抽取使用原文，不需要把 dense_vector、sparse_vector 传给大模型。
        # 商品名去掉首尾空白；content 保留原文，便于后续核对抽取依据。
        # title 是章节标题；file_title 是文件标题，缺失时按“切片 → state → 空字符串”兜底。
        kg_chunks.append({
            "chunk_id": chunk_id,
            "item_name": item_name.strip(),
            "content": content,
            "title": chunk.get("title") or "",
            "file_title": (
                chunk.get("file_title")
                or state.get("file_title")
                or ""
            ),
        })

    # 全部切片通过校验后才返回；任意一条抛错都会中断本函数，不会静默跳过。
    return kg_chunks


def _required_text(data: dict, key: str, label: str) -> str:
    """内部校验工具：指定字段必须是非空字符串，返回去掉首尾空白后的值。"""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 的 {key} 必须是非空字符串")
    return value.strip()


def _evidence_view(text: str) -> tuple[str, list[tuple[int, int]]]:
    """生成只用于比较的文本和原文位置映射；不改变原始切片。

    只解码一层 HTML 字符引用，并把连续空白折叠成一个空格。
    不删除 HTML 标签、不改标点、不纠正 OCR，也不把不同单元格拼成句子。
    """
    characters = []
    offsets = []
    for match in _EVIDENCE_TOKENS.finditer(text):
        token = match.group()
        decoded = unescape(token) if token.startswith("&") and token.endswith(";") else token
        for char in decoded:
            char = " " if char.isspace() else char
            if char == " " and characters and characters[-1] == " ":
                # 连续空白只保留一个比较字符，但原文范围覆盖全部空白。
                offsets[-1] = (offsets[-1][0], match.end())
            else:
                characters.append(char)
                offsets.append((match.start(), match.end()))
    return "".join(characters), offsets


def _resolve_evidence(evidence: str, content: str, label: str) -> str:
    """返回可在原文中逐字找到的证据；格式等价时通过位置映射还原原文。"""
    if evidence in content:
        return evidence

    source_view, offsets = _evidence_view(content)
    evidence_view, _ = _evidence_view(evidence)
    if evidence_view.strip():
        position = source_view.find(evidence_view)
        while position >= 0:
            # 用匹配位置取回原文，而不是将模型改写的文本存入数据库。
            start = offsets[position][0]
            end = offsets[position + len(evidence_view) - 1][1]
            original = content[start:end]
            # 一个 HTML 引用可能展开成多个字符，不能只命中其中一部分就接受。
            if _evidence_view(original)[0] == evidence_view:
                return original
            position = source_view.find(evidence_view, position + 1)

    # 错误中保留有限长度的失败证据，便于定位具体差异，也作为重新抽取的反馈。
    raise ValueError(f"{label}的 evidence 无法在切片原文中找到：{evidence[:120]!r}")


def _read_evidence(data: dict, content: str, sources: dict, label: str) -> str:
    """新输出用起止编号引用证据；旧的字符串输出仍按原有规则严格验证。"""
    evidence = data.get("evidence")
    if isinstance(evidence, dict):
        return resolve_evidence_reference(evidence, content, sources, label)
    # 兼容旧响应，但不会因为引用不到原文就直接放行或删除事实。
    return _resolve_evidence(_required_text(data, "evidence", label), content, label)


def _parse_and_validate_extraction(text: str, content: str) -> dict:
    """
    把模型输出转换为经过校验的实体、关系列表。

    JSON 模式主要约束输出格式，不能保证字段、实体引用和证据正确，
    因此还要在 Python 中逐项检查。编号证据由程序还原为连续原文；旧字符串
    证据允许 HTML/空白等价匹配。这不等于证明模型理解的事实一定正确。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型返回的抽取内容为空")
    text = text.strip()

    # 部分兼容接口仍会给 JSON 套上 Markdown 代码块，只移除最外层包装。
    # 不用全局 replace，避免误删原文证据中本来就存在的反引号。
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("模型输出不是合法的 JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("抽取结果必须是 JSON 对象")
    for key in ("entities", "relations"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"抽取结果的 {key} 必须是列表，允许为空列表")

    # 编号和位置由同一份原文确定性生成，不能使用模型自行返回的映射表。
    sources = {source["id"]: source for source in build_evidence_sources(content)}
    entities = []
    entity_ids = set()
    for index, entity in enumerate(data["entities"], start=1):
        label = f"第 {index} 个实体"
        if not isinstance(entity, dict):
            raise ValueError(f"{label}必须是对象")

        # e1、e2 等 id 只用于当前切片内的关系引用，不是 Neo4j 或 Milvus 的主键。
        # 不同切片可以重复使用 e1；跨切片的实体合并留给后续图谱写入步骤。
        clean = {
            key: _required_text(entity, key, label)
            for key in ("id", "name", "type")
        }
        if clean["id"] in entity_ids:
            raise ValueError(f"{label}的 id 重复：{clean['id']}")
        if clean["type"] not in KG_ENTITY_TYPES:
            raise ValueError(f"{label}使用了不支持的实体类型：{clean['type']}")
        clean["evidence"] = _read_evidence(entity, content, sources, label)

        entity_ids.add(clean["id"])
        entities.append(clean)

    relations = []
    seen_relations = set()
    for index, relation in enumerate(data["relations"], start=1):
        label = f"第 {index} 条关系"
        if not isinstance(relation, dict):
            raise ValueError(f"{label}必须是对象")
        clean = {
            key: _required_text(relation, key, label)
            for key in ("source", "target", "relation")
        }
        # source 和 target 必须引用上面 entities 列表中真实存在的局部 id。
        if clean["source"] not in entity_ids or clean["target"] not in entity_ids:
            raise ValueError(f"{label}引用了不存在的实体")
        clean["evidence"] = _read_evidence(relation, content, sources, label)

        # 相同端点、关系含义和证据的重复结果只保留一次；不同证据仍分别保留。
        relation_key = tuple(clean[key] for key in ("source", "target", "relation", "evidence"))
        if relation_key not in seen_relations:
            seen_relations.add(relation_key)
            relations.append(clean)

    # 只返回已校验字段，不接受模型自行提供的 chunk_id、商品归属等元数据。
    return {"entities": entities, "relations": relations}


def step_2_extract_entities_relations(chunks: list[dict], *, llm=None) -> list[dict]:
    """
    对第一步返回的每个切片调用模型，抽取实体与关系，保留原文出处。

    输入：step_1_get_chunks 整理后的切片列表。
    输出：每个切片一条结果，包含 chunk_id、item_name、title、file_title、
          entities、relations。实体和关系的 evidence 均来自该条切片原文。

    llm：正常运行时无需传入，会复用项目已有客户端；测试时可以传入假模型，
         验证输出处理逻辑而不调用外部接口。
    """
    if not chunks:
        return []

    if llm is None:
        # 按需导入，只有真实抽取时才读取模型配置、创建客户端。
        from app.lm.lm_utils import get_llm_client
        llm = get_llm_client(json_mode=True)

    # 提示词单独保存在 prompts 中；修改抽取规则时可以直接调整该文件。
    # 模板中的 JSON 示例使用双大括号，避免被 load_prompt 的 format 当成变量。
    system_prompt = load_prompt(
        "kg_extraction",
        entity_types=", ".join(sorted(KG_ENTITY_TYPES)),
    )
    # 便于重启后确认服务已加载新协议；不把原文或模型配置打印到日志中。
    logger.info("图谱抽取证据模式：原文编号引用（evidence.start/end）")
    extractions = []

    # 第一版逐片处理全部切片，便于定位异常，也让每条证据只对应当前原文。
    # 不沿用商品识别“只取前几个切片”的策略，否则会遗漏手册后面的知识。
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk["chunk_id"]
        logger.info("开始抽取第 {}/{} 个切片，chunk_id={}", index, len(chunks), chunk_id)
        try:
            # 完整原文按顺序编号后交给模型，不让它逐字复制跨图片的长证据。
            # 模型只返回起止编号，程序根据位置截取原文。每个切片重新从 s1 编号。
            # 分段不改写原文；不重复发送一份 content，以免输入长度翻倍。
            evidence_sources = build_evidence_sources(chunk["content"])
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps({
                    "item_name": chunk["item_name"],
                    "title": chunk.get("title", ""),
                    "evidence_sources": [
                        {"id": source["id"], "text": source["text"]}
                        for source in evidence_sources
                    ],
                }, ensure_ascii=False)),
            ]
            for attempt in range(1, KG_EXTRACTION_MAX_ATTEMPTS + 1):
                response = llm.invoke(messages)
                # LangChain 的 text 属性统一提取文本，也能处理 content 是文本块列表的情况。
                response_text = response.text
                try:
                    extracted = _parse_and_validate_extraction(response_text, chunk["content"])
                    break
                except ValueError as exc:
                    if attempt == KG_EXTRACTION_MAX_ATTEMPTS:
                        raise
                    logger.warning(
                        "chunk_id={} 第 {}/{} 次抽取校验失败，将附带原因重新抽取：{}",
                        chunk_id, attempt, KG_EXTRACTION_MAX_ATTEMPTS, exc,
                    )
                    # 只重试当前失败切片，原始规则和原文保持不变。
                    # 反馈以数据形式提供；要求重新生成整份结果，不能丢弃失败实体后假装成功。
                    messages = messages[:2] + [HumanMessage(content=json.dumps({
                        "validation_error": str(exc),
                        "instruction": (
                            "上次结果未通过校验，请基于前面的 evidence_sources 重新抽取并返回完整 JSON。"
                            "所有 evidence 使用含 start/end 的对象，引用当前切片已有的 s 编号。"
                            "跨图片或表格时选择有序的起止编号，程序会保留中间全部原文。"
                            "不要复制长 URL、改写或省略原文，不要通过返回空结果规避校验。"
                        ),
                    }, ensure_ascii=False))]
        except Exception as exc:
            # 调用失败或格式错误都不是“没有知识”，不能悄悄变成空结果。
            # 报错带上真实 chunk_id；from exc 保留具体原因，便于定位并重试。
            raise RuntimeError(f"chunk_id={chunk_id} 的实体关系抽取失败：{exc}") from exc

        # 商品归属和出处始终取自输入切片，避免模型编造 ID 或把其他商品的知识混进来。
        # entities 中的 id 只能在本条结果内使用，后续写入时需结合商品、类型和名称合并实体。
        extractions.append({
            "chunk_id": chunk_id,
            "item_name": chunk["item_name"],
            "title": chunk.get("title", ""),
            "file_title": chunk.get("file_title", ""),
            "entities": extracted["entities"],
            "relations": extracted["relations"],
        })
        logger.info(
            "chunk_id={} 抽取完成：{} 个实体，{} 条关系",
            chunk_id, len(extracted["entities"]), len(extracted["relations"]),
        )

    return extractions


def _make_entity_key(item_name: str, entity_type: str, name: str) -> str:
    """用“所属商品 + 类型 + 名称”生成可重复计算的实体标识。"""
    # 先序列化成 JSON 列表，避免直接用 | 拼接时，名称本身含有 | 导致组合产生歧义。
    # SHA-256 将这个组合转换成固定长度的字符串；相同组合始终得到相同 entity_key。
    # 这不是 e1/e2 临时编号，也不是 Neo4j 内部 ID，而是应用自己维护的业务标识。
    identity = json.dumps([item_name, entity_type, name], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _prepare_neo4j_data(extractions: list[dict]) -> dict:
    """
    把按切片组织的抽取结果，整理成适合批量写入的节点、关系记录。

    在连接数据库之前完成必要校验，避免不完整输入触发旧数据替换。
    原文证据是否存在已由第二步检查；这里再次检查写库所需字段和实体引用。
    """
    if not isinstance(extractions, list):
        raise ValueError("Neo4j 入库数据必须是列表")

    item_names = set()
    chunks = {}
    entities = {}
    mentions = {}
    relations = {}

    for index, extraction in enumerate(extractions, start=1):
        label = f"第 {index} 条图谱抽取结果"
        if not isinstance(extraction, dict):
            raise ValueError(f"{label}必须是字典")
        item_name = _required_text(extraction, "item_name", label)
        chunk_id = extraction.get("chunk_id")
        # Neo4j 和当前 Milvus 集合均使用有符号 64 位整数保存切片 ID。
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or not -(2**63) <= chunk_id < 2**63:
            raise ValueError(f"{label}的 chunk_id 必须是 INT64 范围内的整数")
        if chunk_id in chunks:
            raise ValueError(f"本批次存在重复的 chunk_id：{chunk_id}")
        for field in ("entities", "relations"):
            if not isinstance(extraction.get(field), list):
                raise ValueError(f"{label}的 {field} 必须是列表")

        chunk_row = {"chunk_id": chunk_id, "item_name": item_name}
        for field in ("title", "file_title"):
            value = extraction.get(field) or ""
            if not isinstance(value, str):
                raise ValueError(f"{label}的 {field} 必须是字符串")
            chunk_row[field] = value
        chunks[chunk_id] = chunk_row
        item_names.add(item_name)

        # 每处理一个切片都重新创建映射。两个切片中的 e1 可以代表不同的实体。
        local_entity_keys = {}
        for entity in extraction["entities"]:
            if not isinstance(entity, dict):
                raise ValueError(f"{label}中的实体必须是字典")
            local_id = _required_text(entity, "id", label)
            if local_id in local_entity_keys:
                raise ValueError(f"chunk_id={chunk_id} 存在重复的实体编号：{local_id}")
            # 第一版只统一多余空白，不自动合并同义词，也不更改型号的大小写。
            name = " ".join(_required_text(entity, "name", label).split())
            entity_type = _required_text(entity, "type", label)
            if entity_type not in KG_ENTITY_TYPES:
                raise ValueError(f"不支持的实体类型：{entity_type}")
            evidence = _required_text(entity, "evidence", label)
            entity_key = _make_entity_key(item_name, entity_type, name)
            local_entity_keys[local_id] = entity_key
            entities[entity_key] = {
                "entity_key": entity_key, "item_name": item_name,
                "name": name, "type": entity_type,
            }
            # 证据属于“哪个切片提到了哪个实体”，所以存放在 MENTIONS 边上。
            # 多个切片共享实体时，各自的证据不会相互覆盖。
            mentions[(chunk_id, entity_key, evidence)] = {
                "chunk_id": chunk_id, "entity_key": entity_key, "evidence": evidence,
            }

        for relation in extraction["relations"]:
            if not isinstance(relation, dict):
                raise ValueError(f"{label}中的关系必须是字典")
            source = _required_text(relation, "source", label)
            target = _required_text(relation, "target", label)
            if source not in local_entity_keys or target not in local_entity_keys:
                raise ValueError(f"chunk_id={chunk_id} 的关系引用了不存在的实体")
            source_key = local_entity_keys[source]
            target_key = local_entity_keys[target]
            predicate = _required_text(relation, "relation", label)
            evidence = _required_text(relation, "evidence", label)
            # 把临时编号换成稳定实体标识；同一事实的不同切片、不同证据分别保留。
            key = (source_key, target_key, predicate, chunk_id, evidence)
            relations[key] = {
                "source_key": source_key, "target_key": target_key,
                "relation": predicate, "chunk_id": chunk_id,
                "evidence": evidence, "item_name": item_name,
            }

    return {
        "item_names": sorted(item_names),
        "chunks": list(chunks.values()),
        "entities": list(entities.values()),
        "mentions": list(mentions.values()),
        "relations": list(relations.values()),
    }


def _write_kg_transaction(tx, data: dict) -> None:
    """
    在同一个 Neo4j 事务中替换本批商品的图谱内容。

    tx 是驱动传入的事务对象。下面所有 run 都成功才提交；任意一条失败则一起回滚。
    驱动可能自动重试整个函数，所以这里不调用大模型、不修改外部 state。
    """
    # Product 表示手册所属商品；Entity 表示原文中提到的对象，二者用途不同。
    # 写一个临时属性再移除，可以持有商品节点的写锁直到事务结束。
    # 同一商品的两个图谱写入因此会串行执行，避免“删旧、写新”的过程交错。
    # 多商品时固定加锁顺序，降低相互等待的机会。
    tx.run("""
        UNWIND $item_names AS item_name
        WITH item_name ORDER BY item_name
        MERGE (p:Product {name: item_name})
        SET p._kg_write_lock = true
        REMOVE p._kg_write_lock
    """, item_names=data["item_names"]).consume()

    # chunk_id 是整个 Milvus 切片集合的主键，不能把另一商品的已有 ID 占过来。
    conflict = tx.run("""
        UNWIND $chunks AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        WHERE c.item_name IS NULL OR c.item_name <> row.item_name
        RETURN c.chunk_id AS chunk_id LIMIT 1
    """, chunks=data["chunks"]).single()
    if conflict is not None:
        raise ValueError(f"chunk_id={conflict['chunk_id']} 已被其他商品或未归属切片使用")

    # 当前 Milvus 按商品名整体替换旧切片，Neo4j 使用相同的替换范围。
    # 先删除该商品的旧事实关系；其出处 chunk_id 可能已在 Milvus 中失效。
    tx.run("""
        MATCH (e:Entity)-[r:RELATED_TO]->(:Entity)
        WHERE e.item_name IN $item_names
        DELETE r
    """, item_names=data["item_names"]).consume()

    # DETACH DELETE 删除旧切片节点及其 HAS_CHUNK、MENTIONS 等连接。
    # Entity 暂时保留：新切片仍提到的实体可继续复用原节点。
    tx.run("""
        MATCH (c:Chunk)
        WHERE c.item_name IN $item_names
        DETACH DELETE c
    """, item_names=data["item_names"]).consume()

    # UNWIND 将参数列表展开为一行一条；MERGE 表示“存在则复用，不存在则创建”。
    # SET 更新普通属性。原文全文仍放在 Milvus，Neo4j 的 Chunk 保存其 ID 和标题。
    tx.run("""
        UNWIND $chunks AS row
        MATCH (p:Product {name: row.item_name})
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.item_name = row.item_name, c.title = row.title, c.file_title = row.file_title
        MERGE (p)-[:HAS_CHUNK]->(c)
    """, chunks=data["chunks"]).consume()

    tx.run("""
        UNWIND $entities AS row
        MERGE (e:Entity {entity_key: row.entity_key})
        SET e.name = row.name, e.type = row.type, e.item_name = row.item_name
    """, entities=data["entities"]).consume()

    tx.run("""
        UNWIND $mentions AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MATCH (e:Entity {entity_key: row.entity_key})
        MERGE (c)-[:MENTIONS {evidence: row.evidence}]->(e)
    """, mentions=data["mentions"]).consume()

    # 固定使用 RELATED_TO；模型给出的“包含”“可能导致”等词存入 relation 属性。
    # 所有动态值都经参数传入，不把模型输出拼接成 Cypher 语句。
    tx.run("""
        UNWIND $relations AS row
        MATCH (source:Entity {entity_key: row.source_key})
        MATCH (target:Entity {entity_key: row.target_key})
        MERGE (source)-[r:RELATED_TO {
            relation: row.relation, chunk_id: row.chunk_id, evidence: row.evidence
        }]->(target)
        SET r.item_name = row.item_name
    """, relations=data["relations"]).consume()

    # 清除本次商品中已经没有原文支持的旧实体，不影响其他商品。
    # 空实体/关系列表也会执行到这里，因此能够移除旧版本留下的知识。
    tx.run("""
        MATCH (e:Entity)
        WHERE e.item_name IN $item_names
          AND NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(e) }
        DETACH DELETE e
    """, item_names=data["item_names"]).consume()


def step_3_save_to_neo4j(extractions: list[dict], *, driver=None, database: str | None = None) -> dict:
    """
    第三步：写入 Product、Chunk、Entity 及它们之间的关系，返回本批入库数量。

    每个商品必须传入本轮完整抽取结果，因为当前策略与 Milvus 一致：按商品整体替换。
    这不是追加单个切片的增量接口。空列表不操作数据库，也不会删除已有数据。
    driver、database 可用于指定隔离测试库；正常调用复用项目的 Neo4j 客户端。
    """
    data = _prepare_neo4j_data(extractions)
    # 统计的是本批去重后的对象数量，不是数据库总量，也不是本次新创建的数量。
    stats = {
        "products": len(data["item_names"]),
        "chunks": len(data["chunks"]),
        "entities": len(data["entities"]),
        "mentions": len(data["mentions"]),
        "relations": len(data["relations"]),
    }
    if not data["chunks"]:
        return stats

    if driver is None:
        from app.clients.neo4j_utils import get_neo4j_driver
        driver = get_neo4j_driver()
    database = database or os.getenv("NEO4J_DATABASE") or "neo4j"

    # 唯一约束保证业务标识不会重复，也为 MERGE 查找提供索引。
    # IF NOT EXISTS 允许重复执行；建约束与数据更新使用不同事务。
    constraints = [
        "CREATE CONSTRAINT kg_product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT kg_entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
    ]
    # 每次调用独立创建 session，离开 with 时关闭；不关闭全局共享的 driver。
    with driver.session(database=database) as session:
        for statement in constraints:
            session.run(statement).consume()
        session.execute_write(_write_kg_transaction, data)

    # 只有数据事务提交成功才返回统计值；失败时异常交给节点和接口层处理。
    # 这个事务只覆盖 Neo4j，不能回滚前面已经完成的 Milvus 入库。
    return stats


def node_import_kg(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph 调用的节点入口：负责安排各个处理步骤和登记任务进度。

    main_graph.py 注册的是这个函数；step_1、step_2、step_3 都是由它调用的内部步骤，
    无需再把它们单独注册成图节点。

    完成输入校验、实体关系抽取与 Neo4j 写入；任何一步失败都会向上层抛出异常。
    全部切片抽取成功后保留抽取结果，Neo4j 提交成功后再保存入库统计。
    """
    # 节点名称与 main_graph.py 中 add_node 的名称保持一致，便于进度展示和排查。
    current_node = "node_import_kg"
    # task_id 标识当前这一轮导入任务；get 的第二个参数是字段缺失时的默认值。
    task_id = state.get("task_id", "")
    # 将节点登记到当前任务的“正在运行”列表。
    add_running_task(task_id, current_node)

    try:
        # 调用上面的校验函数，得到后续抽取步骤需要的简化切片列表。
        chunks = step_1_get_chunks(state)

        # 第二步：调用模型并校验结果，单独保存到导入状态的 kg_extractions 字段。
        # 这个字段已在 ImportGraphState 中声明，LangGraph 才能保留它并传给后续步骤。
        state["kg_extractions"] = step_2_extract_entities_relations(chunks)

        # 第三步：把实体、关系及原文出处写入 Neo4j；成功后才记录入库统计。
        state["kg_import_stats"] = step_3_save_to_neo4j(state["kg_extractions"])
        logger.info(
            "Neo4j 图谱入库完成，本批统计：{}",
            state["kg_import_stats"],
        )
        # 返回带有抽取结果的 state；原 state["chunks"] 中的原文和向量保持不变。
        return state
    finally:
        # finally 在正常返回前或异常向上传播前都会执行；它本身不会吞掉校验异常。
        # add_done_task 会移除“正在运行”记录，并把节点加入“已完成”列表。
        # 这是当前的进度登记方式：即使校验失败也会执行，不能据此判断 Neo4j 写入成功。
        add_done_task(task_id, current_node)
