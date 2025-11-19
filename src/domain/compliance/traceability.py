"""
Spec-to-Code 追溯性管理模块

提供任务 ID ↔ 提交文件的映射功能，生成追溯性报告。
基于 SQLModel 实现数据持久化，集成到 ComplianceReport 中。
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlmodel import JSON, Field, Session, SQLModel, create_engine, select

from src.domain.compliance.models import ComplianceReport


@dataclass
class TraceabilityRecord:
    """追溯性记录项"""
    task_id: str
    file_path: str
    file_hash: str
    file_type: str  # spec, code, doc, test
    operation: str  # created, modified, deleted
    timestamp: datetime
    commit_id: str | None = None
    author: str | None = None
    message: str | None = None


@dataclass
class TraceabilityGap:
    """追溯性缺口"""
    requirement_id: str
    implementation_files: list[str]
    status: str  # missing, partial, complete
    confidence: float
    description: str


class TraceabilityManager:
    """追溯性管理器"""

    def __init__(self, db_path: str = "data/persistence/traceability.db"):
        """
        初始化追溯性管理器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self._create_tables()

    def _create_tables(self) -> None:
        """创建数据库表"""
        SQLModel.metadata.create_all(self.engine)

    def calculate_file_hash(self, file_path: Path) -> str:
        """计算文件内容的 SHA256 哈希值"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return "unknown"

    def scan_spec_files(self, spec_dir: str = "specs") -> list[TraceabilityRecord]:
        """扫描规范文件并创建追溯性记录"""
        records = []
        spec_path = Path(spec_dir)

        if not spec_path.exists():
            logger.warning(f"规范目录不存在: {spec_dir}")
            return records

        # 扫描所有规范文件
        for file_path in spec_path.rglob("*.md"):
            if file_path.is_file():
                file_hash = self.calculate_file_hash(file_path)

                # 推断任务 ID（基于文件路径）
                task_id = f"spec_{file_path.stem}_{file_hash[:8]}"

                record = TraceabilityRecord(
                    task_id=task_id,
                    file_path=str(file_path),
                    file_hash=file_hash,
                    file_type="spec",
                    operation="created",
                    timestamp=datetime.fromtimestamp(file_path.stat().st_mtime)
                )
                records.append(record)

        logger.info(f"扫描到 {len(records)} 个规范文件")
        return records

    def scan_code_files(self, code_dir: str = "src") -> list[TraceabilityRecord]:
        """扫描代码文件并创建追溯性记录"""
        records = []
        code_path = Path(code_dir)

        if not code_path.exists():
            logger.warning(f"代码目录不存在: {code_dir}")
            return records

        # 扫描所有 Python 文件
        for file_path in code_path.rglob("*.py"):
            if file_path.is_file():
                file_hash = self.calculate_file_hash(file_path)

                # 推断任务 ID（基于文件路径和模块结构）
                rel_path = file_path.relative_to(code_path)
                task_id = f"code_{rel_path.parent.name}_{rel_path.stem}_{file_hash[:8]}"

                record = TraceabilityRecord(
                    task_id=task_id,
                    file_path=str(file_path),
                    file_hash=file_hash,
                    file_type="code",
                    operation="created",
                    timestamp=datetime.fromtimestamp(file_path.stat().st_mtime)
                )
                records.append(record)

        logger.info(f"扫描到 {len(records)} 个代码文件")
        return records

    def scan_documentation_files(self, docs_dir: str = "docs") -> list[TraceabilityRecord]:
        """扫描文档文件并创建追溯性记录"""
        records = []
        docs_path = Path(docs_dir)

        if not docs_path.exists():
            logger.warning(f"文档目录不存在: {docs_dir}")
            return records

        # 扫描所有文档文件
        for file_path in docs_path.rglob("*.md"):
            if file_path.is_file():
                file_hash = self.calculate_file_hash(file_path)

                # 推断任务 ID
                rel_path = file_path.relative_to(docs_path)
                task_id = f"doc_{rel_path.parent.name}_{rel_path.stem}_{file_hash[:8]}"

                record = TraceabilityRecord(
                    task_id=task_id,
                    file_path=str(file_path),
                    file_hash=file_hash,
                    file_type="doc",
                    operation="created",
                    timestamp=datetime.fromtimestamp(file_path.stat().st_mtime)
                )
                records.append(record)

        logger.info(f"扫描到 {len(records)} 个文档文件")
        return records

    def analyze_traceability_gaps(self,
                                 spec_records: list[TraceabilityRecord],
                                 code_records: list[TraceabilityRecord]) -> list[TraceabilityGap]:
        """分析追溯性缺口"""
        gaps = []

        # 构建文件映射
        spec_files = {r.file_path: r for r in spec_records}
        code_files = {r.file_path: r for r in code_records}

        # 分析规范到代码的映射
        for spec_path, spec_record in spec_files.items():
            spec_name = Path(spec_path).stem

            # 查找相关的代码文件
            related_codes = []
            for code_path in code_files.keys():
                code_name = Path(code_path).stem
                # 简单的名称匹配逻辑
                if spec_name.lower() in code_name.lower() or code_name.lower() in spec_name.lower():
                    related_codes.append(code_path)

            # 确定追溯性状态
            if not related_codes:
                status = "missing"
                confidence = 0.0
            elif len(related_codes) < 3:  # 至少需要3个相关文件
                status = "partial"
                confidence = 0.5
            else:
                status = "complete"
                confidence = 1.0

            gap = TraceabilityGap(
                requirement_id=spec_record.task_id,
                implementation_files=related_codes,
                status=status,
                confidence=confidence,
                description=f"规范文件 {spec_path} 的实现状态"
            )
            gaps.append(gap)

        logger.info(f"分析完成，发现 {len(gaps)} 个追溯性缺口")
        return gaps

    def generate_traceability_report(self, submission_id: str) -> dict[str, Any]:
        """生成追溯性报告"""
        logger.info(f"生成追溯性报告: {submission_id}")

        # 扫描各类文件
        spec_records = self.scan_spec_files()
        code_records = self.scan_code_files()
        doc_records = self.scan_documentation_files()

        # 分析追溯性缺口
        gaps = self.analyze_traceability_gaps(spec_records, code_records)

        # 计算统计信息
        total_specs = len(spec_records)
        total_codes = len(code_records)
        total_docs = len(doc_records)

        complete_gaps = [g for g in gaps if g.status == "complete"]
        partial_gaps = [g for g in gaps if g.status == "partial"]
        missing_gaps = [g for g in gaps if g.status == "missing"]

        # 生成报告
        report = {
            "submission_id": submission_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_specs": total_specs,
                "total_codes": total_codes,
                "total_docs": total_docs,
                "traceability_score": len(complete_gaps) / max(len(gaps), 1) * 100,
                "gaps_count": {
                    "complete": len(complete_gaps),
                    "partial": len(partial_gaps),
                    "missing": len(missing_gaps)
                }
            },
            "spec_files": [
                {
                    "task_id": r.task_id,
                    "path": r.file_path,
                    "type": r.file_type,
                    "hash": r.file_hash,
                    "modified": r.timestamp.isoformat()
                }
                for r in spec_records
            ],
            "code_files": [
                {
                    "task_id": r.task_id,
                    "path": r.file_path,
                    "type": r.file_type,
                    "hash": r.file_hash,
                    "modified": r.timestamp.isoformat()
                }
                for r in code_records
            ],
            "documentation_files": [
                {
                    "task_id": r.task_id,
                    "path": r.file_path,
                    "type": r.file_type,
                    "hash": r.file_hash,
                    "modified": r.timestamp.isoformat()
                }
                for r in doc_records
            ],
            "traceability_gaps": [
                {
                    "requirement_id": g.requirement_id,
                    "implementation_files": g.implementation_files,
                    "status": g.status,
                    "confidence": g.confidence,
                    "description": g.description
                }
                for g in gaps
            ],
            "recommendations": self._generate_recommendations(gaps)
        }

        # 保存报告到数据库
        compliance_report = ComplianceReport(
            report_id=submission_id,
            report_type="traceability",
            status="completed",
            total_checks=len(gaps),
            passed_checks=len(complete_gaps),
            failed_checks=len(missing_gaps),
            summary=f"追溯性检查完成，完整度: {report['summary']['traceability_score']:.1f}%",
            details=report
        )

        with Session(self.engine) as session:
            session.add(compliance_report)
            session.commit()

        # 保存详细报告到文件
        self._save_traceability_report_file(submission_id, report)

        logger.info(f"追溯性报告生成完成: {submission_id}")
        return report

    def _generate_recommendations(self, gaps: list[TraceabilityGap]) -> list[str]:
        """生成改进建议"""
        recommendations = []

        missing_gaps = [g for g in gaps if g.status == "missing"]
        partial_gaps = [g for g in gaps if g.status == "partial"]

        if missing_gaps:
            recommendations.append(f"为 {len(missing_gaps)} 个缺失实现的规范创建对应的代码文件")

        if partial_gaps:
            recommendations.append(f"完善 {len(partial_gaps)} 个部分实现的规范，增加更多相关代码文件")

        complete_gaps = [g for g in gaps if g.status == "complete"]
        if len(complete_gaps) < len(gaps) * 0.8:
            recommendations.append("整体追溯性完整度较低，建议建立更严格的规范-代码映射流程")

        if not recommendations:
            recommendations.append("追溯性状态良好，继续保持规范与代码的对应关系")

        return recommendations

    def _save_traceability_report_file(self, submission_id: str, report: dict[str, Any]) -> None:
        """保存追溯性报告到文件"""
        output_dir = Path("data/persistence/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 JSON 格式
        json_file = output_dir / f"traceability_report_{submission_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 保存 Markdown 格式摘要
        md_file = output_dir / f"traceability_report_{submission_id}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 追溯性报告\n\n")
            f.write(f"**提交 ID:** {submission_id}\n")
            f.write(f"**生成时间:** {report['generated_at']}\n\n")

            f.write("## 摘要\n\n")
            f.write(f"- 规范文件: {report['summary']['total_specs']} 个\n")
            f.write(f"- 代码文件: {report['summary']['total_codes']} 个\n")
            f.write(f"- 文档文件: {report['summary']['total_docs']} 个\n")
            f.write(f"- 追溯性评分: {report['summary']['traceability_score']:.1f}%\n\n")

            f.write("## 追溯性缺口统计\n\n")
            gaps_count = report['summary']['gaps_count']
            f.write(f"- 完整: {gaps_count['complete']} 个\n")
            f.write(f"- 部分: {gaps_count['partial']} 个\n")
            f.write(f"- 缺失: {gaps_count['missing']} 个\n\n")

            f.write("## 改进建议\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")

        logger.info(f"追溯性报告文件已保存: {json_file}, {md_file}")


# 全局追溯性管理器实例
_traceability_manager: TraceabilityManager | None = None


def get_traceability_manager() -> TraceabilityManager:
    """获取全局追溯性管理器实例"""
    global _traceability_manager
    if _traceability_manager is None:
        _traceability_manager = TraceabilityManager()
    return _traceability_manager


def generate_traceability_report(submission_id: str) -> dict[str, Any]:
    """
    生成追溯性报告的便捷函数

    Args:
        submission_id: 提交 ID

    Returns:
        追溯性报告数据
    """
    manager = get_traceability_manager()
    return manager.generate_traceability_report(submission_id)
