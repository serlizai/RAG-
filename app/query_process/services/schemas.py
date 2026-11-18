"""查询服务的统一结果结构，供下一阶段的 Agent 工具层使用。"""

from dataclasses import dataclass, field
from typing import Literal


ServiceStatus = Literal["ok", "empty", "partial", "error", "needs_clarification"]
EvidenceSource = Literal["local", "web"]


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    title: str
    source: EvidenceSource
    retrieval_method: str
    item_name: str | None = None
    chunk_id: int | str | None = None
    url: str | None = None


@dataclass
class SearchResult:
    """`items` 保留旧节点需要的原始形状，`evidence` 供新工具消费。"""

    status: ServiceStatus
    items: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    error_code: str | None = None
    retryable: bool = False
    candidate_count: int = 0


@dataclass
class ProductResolutionResult:
    status: ServiceStatus
    rewritten_query: str
    confirmed_item_names: list[str] = field(default_factory=list)
    option_item_names: list[str] = field(default_factory=list)
    error_code: str | None = None
    retryable: bool = False
