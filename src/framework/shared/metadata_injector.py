"""
生成文件元数据注入与校验工具

功能
- 为自动生成的文件头部注入 `<!-- generated: <command> @ <timestamp> -->`
- 支持幂等注入（已存在则更新时间或保持一致）
- 批量验证与批量注入

使用
- 在生成器/脚本中调用 `inject_header(path, command)`
- 在 CI 中调用 `verify_file(path)` 或 `verify_directory(root)`
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional

GENERATED_RE = re.compile(
    r"^\s*<!--\s*generated:\s*(?P<command>.+?)\s*@\s*(?P<timestamp>[^-]+?)\s*-->\s*$"
)
HEADER_TEMPLATE = "<!-- generated: {command} @ {timestamp} -->\n"


@dataclass(frozen=True)
class VerificationResult:
    path: Path
    has_header: bool
    command: str | None
    timestamp: str | None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_header(command: str, ts: str | None = None) -> str:
    return HEADER_TEMPLATE.format(command=command.strip(), timestamp=(ts or _now_iso()))


def parse_header_line(line: str) -> tuple[str, str] | None:
    m = GENERATED_RE.match(line)
    if not m:
        return None
    return m.group("command").strip(), m.group("timestamp").strip()


def inject_header(file_path: str | Path, command: str, update_if_exists: bool = False) -> bool:
    """
    为文件注入生成头。如果已存在：
    - update_if_exists=True 则更新时间戳
    - 否则保持不变
    返回：是否发生写入
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))

    content = p.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines(keepends=True)

    header = format_header(command)

    if lines:
        parsed = parse_header_line(lines[0])
        if parsed:
            # 已有头
            if update_if_exists:
                lines[0] = format_header(command)
                p.write_text("".join(lines), encoding="utf-8")
                return True
            return False

    # 无头，注入
    new_content = header + content
    p.write_text(new_content, encoding="utf-8")
    return True


def verify_file(file_path: str | Path) -> VerificationResult:
    """校验单个文件是否包含生成头"""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return VerificationResult(p, False, None, None)

    with p.open(encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        parsed = parse_header_line(first_line)
        if not parsed:
            return VerificationResult(p, False, None, None)
        command, ts = parsed
        return VerificationResult(p, True, command, ts)


def verify_directory(
    root: str | Path,
    include_globs: Iterable[str] = ("**/*.md",),
    exclude_globs: Iterable[str] = (".git/**",),
) -> list[VerificationResult]:
    """校验目录下文件的生成头，返回结果列表"""
    root_path = Path(root)
    results: list[VerificationResult] = []

    matched: set[Path] = set()
    for pattern in include_globs:
        for p in root_path.glob(pattern):
            matched.add(p)

    # 排除
    excluded: set[Path] = set()
    for pattern in exclude_globs:
        excluded.update(root_path.glob(pattern))

    targets = [p for p in matched if p.is_file() and p not in excluded]
    for p in targets:
        results.append(verify_file(p))
    return results


def bulk_inject(
    root: str | Path,
    command: str,
    include_globs: Iterable[str] = ("**/*.md",),
    exclude_globs: Iterable[str] = (".git/**",),
    update_if_exists: bool = False,
) -> list[Path]:
    """为目录下的匹配文件批量注入（或更新）生成头"""
    root_path = Path(root)
    changed: list[Path] = []

    matched: set[Path] = set()
    for pattern in include_globs:
        for p in root_path.glob(pattern):
            matched.add(p)

    excluded: set[Path] = set()
    for pattern in exclude_globs:
        excluded.update(root_path.glob(pattern))

    targets = [p for p in matched if p.is_file() and p not in excluded]
    for p in targets:
        if inject_header(p, command, update_if_exists=update_if_exists):
            changed.append(p)

    return changed


__all__ = [
    "VerificationResult",
    "inject_header",
    "verify_file",
    "verify_directory",
    "bulk_inject",
    "format_header",
    "parse_header_line",
]

# 向后兼容：推荐使用统一的 MetadataInjector
# from src.framework.shared.metadata_injector_unified import MetadataInjector

