"""文档图片引用：兼容 Markdown 与 MinerU 表格内的 HTML img。"""

from dataclasses import dataclass
from html import escape, unescape
from pathlib import PurePosixPath
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from app.conf.minio_config import minio_config


# 兼容旧 Markdown URL 内文件名的括号；新生成的 URL 会编码这些字符。
_IMAGE_REFERENCE = re.compile(
    r'!\[(?:\\.|[^\]\\\n])*\]\(\s*(?P<markdown><[^>\n]+>|(?:[^()\n]|\([^()\n]*\))+?)\s*\)'
    r'|<img\b[^>]*?\bsrc\s*=\s*(?P<quote>[\"\'])(?P<html>.*?)(?P=quote)[^>]*>',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageReference:
    start: int
    end: int
    url_start: int
    url_end: int
    url: str
    kind: str


def image_references(text: str):
    for match in _IMAGE_REFERENCE.finditer(text or ""):
        kind = "markdown" if match.group("markdown") is not None else "html"
        raw = match.group(kind)
        yield ImageReference(
            match.start(), match.end(), match.start(kind), match.end(kind),
            unescape(raw.strip().removeprefix("<").removesuffix(">")), kind,
        )


def document_image_object_name(file_title: str, image_file: str) -> str:
    """写入对象的 key 与读取 URL 使用同一个规范，均不带前导斜线。"""
    if not file_title or "/" in file_title or "\\" in file_title or file_title in (".", ".."):
        raise ValueError("图片缺少有效的文档标题")
    path = PurePosixPath(image_file)
    if not image_file or path.is_absolute() or any(part in (".", "..") for part in image_file.split("/")):
        raise ValueError("图片路径必须位于文档的 images 目录内")
    return "/".join(part for part in (minio_config.minio_img_dir.strip("/"), file_title, image_file) if part)


def document_image_url(file_title: str, image_file: str) -> str:
    key = document_image_object_name(file_title, image_file)
    scheme = "https" if minio_config.minio_secure else "http"
    endpoint = minio_config.endpoint.rstrip("/")
    base = endpoint if endpoint.startswith(("http://", "https://")) else f"{scheme}://{endpoint}"
    return f"{base}/{quote(minio_config.bucket_name, safe='')}/{quote(key, safe='/')}"


def resolve_image_url(url: str, file_title: str = "") -> str | None:
    """旧相对引用按文档归属解析；不把未知相对路径传给浏览器。"""
    url = unescape(str(url or "").strip())
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, quote(parsed.path, safe="/%:@-._~"), parsed.query, parsed.fragment))
    relative = unquote(url).replace("\\", "/")
    if relative.startswith("./"):
        relative = relative[2:]
    if parsed.scheme or parsed.netloc or not relative.startswith("images/") or not file_title:
        return None
    try:
        return document_image_url(file_title, relative[len("images/"):])
    except ValueError:
        return None


def replace_image_references(text: str, replacements: dict[str, str], summaries=None) -> str:
    """按 URL 精确替换，保留 HTML 表格标签及其余原文。"""
    summaries = summaries or {}
    result = text
    for ref in reversed(list(image_references(text))):
        url = replacements.get(ref.url)
        if not url:
            continue
        summary = summaries.get(ref.url)
        if ref.kind == "markdown" and summary is not None:
            alt = str(summary).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
            result = result[:ref.start] + f"![{alt}]({url})" + result[ref.end:]
        else:
            value = escape(url, quote=True) if ref.kind == "html" else url
            result = result[:ref.url_start] + value + result[ref.url_end:]
    return result


def normalize_document_images(text: str, file_title: str = "") -> str:
    replacements = {}
    for ref in image_references(text):
        url = resolve_image_url(ref.url, file_title)
        if url:
            replacements[ref.url] = url
    return replace_image_references(text, replacements)
