"""
文档分类器与 Token 守卫模块

实现文档自动分类（Agent/Developer/External）并与 DocumentProfile 联动，
对 Agent 文档执行 token 限制提示。
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.domain.compliance.models import ComplianceReport
from src.domain.doc_review.models import DocumentReview, ReviewMetric


@dataclass
class DocumentClassification:
    """文档分类结果"""
    doc_type: str  # agent, developer, external
    confidence: float
    reasoning: str
    keywords: list[str]
    metadata: dict[str, Any]


@dataclass
class TokenAnalysis:
    """Token 分析结果"""
    estimated_tokens: int
    token_limit: int
    usage_percentage: float
    warnings: list[str]
    recommendations: list[str]


class DocumentClassifier:
    """文档分类器"""

    def __init__(self):
        """初始化分类器"""
        # Agent 文档关键词
        self.agent_keywords = {
            'high': [
                'prompt', 'instruction', 'guidance', 'workflow', 'procedure',
                'howto', 'tutorial', 'manual', 'guide', 'assistant',
                'ai', 'llm', 'chatbot', 'bot', 'agent'
            ],
            'medium': [
                'example', 'template', 'pattern', 'best practice',
                'standard', 'guideline', 'policy', 'rule'
            ]
        }

        # Developer 文档关键词
        self.developer_keywords = {
            'high': [
                'api', 'interface', 'endpoint', 'schema', 'protocol',
                'implementation', 'code', 'library', 'framework',
                'sdk', 'integration', 'architecture', 'design'
            ],
            'medium': [
                'function', 'class', 'method', 'parameter', 'return',
                'type', 'validation', 'error', 'exception'
            ]
        }

        # External 文档关键词
        self.external_keywords = {
            'high': [
                'user', 'customer', 'client', 'end-user', 'consumer',
                'documentation', 'help', 'support', 'faq', 'troubleshoot',
                'installation', 'setup', 'configuration', 'usage'
            ],
            'medium': [
                'feature', 'requirement', 'specification', 'overview',
                'introduction', 'getting started', 'quick start'
            ]
        }

        # Token 限制配置
        self.token_limits = {
            'agent': 2000,      # Agent 文档限制 2000 tokens
            'developer': 5000,  # Developer 文档限制 5000 tokens
            'external': 3000    # External 文档限制 3000 tokens
        }

    def calculate_keyword_score(self, content: str, keywords: dict[str, list[str]]) -> float:
        """计算关键词匹配分数"""
        content_lower = content.lower()
        score = 0.0

        for level, words in keywords.items():
            weight = 2.0 if level == 'high' else 1.0
            for word in words:
                # 使用正则表达式进行单词边界匹配
                if re.search(r'\b' + re.escape(word) + r'\b', content_lower):
                    score += weight

        return score

    def classify_document(self, content: str, file_path: str | None = None) -> DocumentClassification:
        """对文档进行分类"""
        # 计算各类别的分数
        agent_score = self.calculate_keyword_score(content, self.agent_keywords)
        developer_score = self.calculate_keyword_score(content, self.developer_keywords)
        external_score = self.calculate_keyword_score(content, self.external_keywords)

        # 归一化分数
        total_score = agent_score + developer_score + external_score
        if total_score == 0:
            # 如果没有匹配到关键词，基于文件路径推断
            if file_path:
                path_lower = file_path.lower()
                if 'api' in path_lower or 'dev' in path_lower:
                    developer_score = 1.0
                elif 'user' in path_lower or 'guide' in path_lower:
                    external_score = 1.0
                else:
                    external_score = 1.0  # 默认为外部文档
            else:
                external_score = 1.0  # 默认为外部文档

        # 确定分类
        scores = {
            'agent': agent_score,
            'developer': developer_score,
            'external': external_score
        }

        doc_type = max(scores, key=scores.get)
        confidence = scores[doc_type] / max(total_score, 1.0)

        # 生成推理说明
        reasoning_parts = []
        for category, score in scores.items():
            if score > 0:
                reasoning_parts.append(f"{category}: {score:.1f}")

        reasoning = f"基于关键词匹配: {', '.join(reasoning_parts)}"

        # 提取关键词
        content_lower = content.lower()
        matched_keywords = []
        for level, words in self.agent_keywords.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', content_lower):
                    matched_keywords.append(word)

        for level, words in self.developer_keywords.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', content_lower):
                    matched_keywords.append(word)

        for level, words in self.external_keywords.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', content_lower):
                    matched_keywords.append(word)

        return DocumentClassification(
            doc_type=doc_type,
            confidence=confidence,
            reasoning=reasoning,
            keywords=list(set(matched_keywords)),  # 去重
            metadata={
                'scores': scores,
                'file_path': file_path,
                'analysis_timestamp': datetime.now().isoformat()
            }
        )

    def estimate_tokens(self, content: str) -> int:
        """估算文本的 token 数量"""
        # 简单的 token 估算：英文约 4 字符 = 1 token，中文约 1.5 字符 = 1 token
        english_chars = len(re.findall(r'[a-zA-Z\s]', content))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        other_chars = len(content) - english_chars - chinese_chars

        estimated_tokens = (english_chars / 4) + (chinese_chars / 1.5) + (other_chars / 3)
        return int(estimated_tokens)

    def analyze_token_usage(self, content: str, doc_type: str) -> TokenAnalysis:
        """分析 token 使用情况"""
        estimated_tokens = self.estimate_tokens(content)
        token_limit = self.token_limits.get(doc_type, 3000)
        usage_percentage = (estimated_tokens / token_limit) * 100

        warnings = []
        recommendations = []

        if usage_percentage > 100:
            warnings.append(f"文档超出 token 限制 {estimated_tokens - token_limit} tokens")
            recommendations.append("考虑将文档拆分为多个部分")
            recommendations.append("移除冗余内容和重复信息")
            recommendations.append("使用更简洁的表达方式")
        elif usage_percentage > 80:
            warnings.append("文档接近 token 限制，建议优化内容")
            recommendations.append("检查是否有不必要的详细描述")
            recommendations.append("考虑将示例代码移至附录")

        if doc_type == 'agent' and estimated_tokens > token_limit:
            warnings.append("Agent 文档过长可能影响 AI 助手的处理效果")
            recommendations.append("将复杂的 Agent 文档分解为多个专门的子文档")
            recommendations.append("为 Agent 提供结构化的问题-答案格式")

        return TokenAnalysis(
            estimated_tokens=estimated_tokens,
            token_limit=token_limit,
            usage_percentage=usage_percentage,
            warnings=warnings,
            recommendations=recommendations
        )


class DocumentProfileManager:
    """文档档案管理器"""

    def __init__(self, db_path: str = "data/persistence/documents.db"):
        """
        初始化文档档案管理器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.classifier = DocumentClassifier()
        self.engine = None
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        try:
            from sqlmodel import create_engine
            self.engine = create_engine(f"sqlite:///{self.db_path}")
            # 创建表（如果不存在）
            # 这里简化处理，实际应该使用 SQLModel 的 metadata.create_all()
        except ImportError:
            logger.warning("SQLModel 未安装，将使用文件存储")

    def analyze_document(self, file_path: str) -> dict[str, Any]:
        """分析单个文档"""
        try:
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 读取文件内容
            content = path.read_text(encoding='utf-8', errors='ignore')

            # 文档分类
            classification = self.classifier.classify_document(content, file_path)

            # Token 分析
            token_analysis = self.classifier.analyze_token_usage(content, classification.doc_type)

            # 计算文档哈希
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # 生成分析结果
            profile = {
                "file_path": file_path,
                "file_size": len(content),
                "content_hash": content_hash,
                "classification": {
                    "type": classification.doc_type,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords,
                    "metadata": classification.metadata
                },
                "token_analysis": {
                    "estimated_tokens": token_analysis.estimated_tokens,
                    "token_limit": token_analysis.token_limit,
                    "usage_percentage": token_analysis.usage_percentage,
                    "warnings": token_analysis.warnings,
                    "recommendations": token_analysis.recommendations
                },
                "analysis_metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "analyzer_version": "1.0.0",
                    "content_lines": len(content.splitlines())
                }
            }

            # 保存分析结果
            self._save_document_profile(file_path, profile)

            # 如果是 Agent 文档且有警告，记录到合规报告
            if classification.doc_type == 'agent' and token_analysis.warnings:
                self._create_compliance_record(file_path, profile)

            logger.info(f"文档分析完成: {file_path} -> {classification.doc_type} (置信度: {classification.confidence:.2f})")
            return profile

        except Exception as e:
            logger.error(f"文档分析失败 {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "analyzed_at": datetime.now().isoformat()
            }

    def analyze_batch(self, pattern: str) -> dict[str, Any]:
        """批量分析文档"""
        from glob import glob

        files = glob(pattern, recursive=True)
        results = {
            "pattern": pattern,
            "total_files": len(files),
            "analyzed_files": [],
            "classification_summary": {
                "agent": 0,
                "developer": 0,
                "external": 0
            },
            "token_issues": [],
            "analysis_timestamp": datetime.now().isoformat()
        }

        for file_path in files:
            if file_path.endswith('.md'):
                profile = self.analyze_document(file_path)
                results["analyzed_files"].append(profile)

                # 更新分类统计
                if "classification" in profile:
                    doc_type = profile["classification"]["type"]
                    results["classification_summary"][doc_type] += 1

                # 收集 Token 问题
                if "token_analysis" in profile and profile["token_analysis"]["warnings"]:
                    results["token_issues"].append({
                        "file": file_path,
                        "warnings": profile["token_analysis"]["warnings"],
                        "tokens": profile["token_analysis"]["estimated_tokens"],
                        "limit": profile["token_analysis"]["token_limit"]
                    })

        # 保存批次分析结果
        self._save_batch_analysis_results(results)

        logger.info(f"批次分析完成: {len(files)} 个文件，发现 {len(results['token_issues'])} 个 Token 问题")
        return results

    def _save_document_profile(self, file_path: str, profile: dict[str, Any]) -> None:
        """保存文档档案到文件"""
        output_dir = Path("data/persistence/documents")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成档案文件名
        file_hash = profile.get("content_hash", "unknown")
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', Path(file_path).name)
        profile_file = output_dir / f"profile_{safe_filename}_{file_hash[:8]}.json"

        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def _save_batch_analysis_results(self, results: dict[str, Any]) -> None:
        """保存批次分析结果"""
        output_dir = Path("data/persistence/documents")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = output_dir / f"batch_analysis_{timestamp}.json"

        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _create_compliance_record(self, file_path: str, profile: dict[str, Any]) -> None:
        """创建合规记录"""
        try:
            from sqlmodel import Session

            from src.domain.compliance.models import ComplianceReport

            if not self.engine:
                return

            # 创建合规报告
            compliance_report = ComplianceReport(
                report_id=f"doc_review_{profile.get('content_hash', '')[:8]}",
                report_type="doc_review",
                status="warning",
                total_checks=1,
                passed_checks=0,
                failed_checks=1,
                summary=f"Agent 文档 '{file_path}' 存在 Token 使用问题",
                details={
                    "file_path": file_path,
                    "classification": profile["classification"],
                    "token_analysis": profile["token_analysis"],
                    "issues": profile["token_analysis"]["warnings"]
                }
            )

            with Session(self.engine) as session:
                session.add(compliance_report)
                session.commit()

            logger.info(f"已创建合规记录: {file_path}")

        except Exception as e:
            logger.error(f"创建合规记录失败: {e}")


# 全局文档档案管理器实例
_document_profile_manager: DocumentProfileManager | None = None


def get_document_profile_manager() -> DocumentProfileManager:
    """获取全局文档档案管理器实例"""
    global _document_profile_manager
    if _document_profile_manager is None:
        _document_profile_manager = DocumentProfileManager()
    return _document_profile_manager


def analyze_document(file_path: str) -> dict[str, Any]:
    """
    分析单个文档的便捷函数

    Args:
        file_path: 文档文件路径

    Returns:
        分析结果
    """
    manager = get_document_profile_manager()
    return manager.analyze_document(file_path)


def analyze_document_batch(pattern: str) -> dict[str, Any]:
    """
    批量分析文档的便捷函数

    Args:
        pattern: 文件匹配模式

    Returns:
        批次分析结果
    """
    manager = get_document_profile_manager()
    return manager.analyze_batch(pattern)
