"""
增强的文档分类器

基于 LangChain 1.0 最佳实践实现，集成结构化输出、中间件和智能分析功能。
提供更准确的文档分类和深度 Token 分析。
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# LangChain 1.0 导入
try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        HumanInTheLoopMiddleware,
        SummarizationMiddleware,
    )
    from langchain.agents.structured_output import ToolStrategy
    from langchain.chat_models import init_chat_model
    from langchain.tools import tool
    from pydantic import BaseModel as PydanticBaseModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain 1.0 未安装，将使用基础实现")
    LANGCHAIN_AVAILABLE = False

from src.domain.compliance.models import ComplianceReport
from src.domain.doc_review.models import DocumentReview, ReviewMetric


@dataclass
class EnhancedDocumentClassification:
    """增强的文档分类结果"""
    doc_type: str  # agent, developer, external
    confidence: float
    reasoning: str
    keywords: list[str]
    metadata: dict[str, Any]
    # LangChain 1.0 增强字段
    content_blocks: list[dict[str, Any]] | None = None
    structured_analysis: dict[str, Any] | None = None
    detected_entities: list[str] | None = None


@dataclass
class EnhancedTokenAnalysis:
    """增强的 Token 分析结果"""
    estimated_tokens: int
    token_limit: int
    usage_percentage: float
    warnings: list[str]
    recommendations: list[str]
    # LangChain 1.0 增强字段
    content_quality_score: float = 0.0
    optimization_suggestions: list[str] = None
    readability_score: float = 0.0


class EnhancedDocumentAnalyzer:
    """增强的文档分析器 - 使用 LangChain 1.0 最佳实践"""

    def __init__(self):
        """初始化分析器"""
        self.classification_agent = None
        self.token_analysis_agent = None
        self.content_analyzer_agent = None

        if LANGCHAIN_AVAILABLE:
            self._setup_langchain_agents()

    def _setup_langchain_agents(self):
        """设置 LangChain 代理"""
        try:
            # 文档分类代理
            self.classification_agent = create_agent(
                model="claude-sonnet-4-5-20250929",
                tools=[self._classify_document_content],
                system_prompt="""
                你是一个专业的文档分类专家。请根据文档内容准确分类为以下类型之一：

                1. **Agent 文档**: 包含 AI 助手指导、提示词、工作流程、操作指南等内容
                   - 关键词：prompt, instruction, guidance, workflow, procedure, howto, tutorial

                2. **Developer 文档**: 包含 API 接口、代码实现、技术架构、开发指南等内容
                   - 关键词：api, interface, implementation, code, library, framework, sdk

                3. **External 文档**: 包含用户指南、安装说明、帮助文档、FAQ 等内容
                   - 关键词：user, customer, documentation, help, support, installation, usage

                请提供详细的分类理由和置信度评估。
                """
            )

            # Token 分析代理
            self.token_analysis_agent = create_agent(
                model="gpt-4o-mini",
                tools=[self._analyze_token_usage],
                system_prompt="""
                你是一个专业的文档优化专家。请分析文档的 Token 使用情况：

                1. 估算 Token 数量
                2. 评估是否超过限制
                3. 提供优化建议
                4. 评估内容质量

                不同类型文档的 Token 限制：
                - Agent 文档: 2000 tokens (保持简洁高效)
                - Developer 文档: 5000 tokens (允许详细技术说明)
                - External 文档: 3000 tokens (平衡详细性和可读性)
                """
            )

            # 内容质量分析代理
            self.content_analyzer_agent = create_agent(
                model="claude-sonnet-4-5-20250929",
                tools=[self._analyze_content_quality],
                system_prompt="""
                你是一个专业的内容质量分析师。请对文档内容进行深度分析：

                1. 评估内容结构和逻辑性
                2. 检查语言表达质量
                3. 分析可读性和理解难度
                4. 识别改进空间

                返回结构化的质量评估报告。
                """
            )

            logger.info("✅ LangChain 分析代理初始化成功")

        except Exception as e:
            logger.error(f"❌ LangChain 代理初始化失败: {e}")
            self.classification_agent = None
            self.token_analysis_agent = None
            self.content_analyzer_agent = None

    @tool
    def _classify_document_content(self, content: str, file_path: str | None = None) -> str:
        """文档内容分类工具"""
        # 基础关键词匹配（作为备选方案）
        content_lower = content.lower()

        # Agent 关键词
        agent_keywords = ['prompt', 'instruction', 'guidance', 'workflow', 'procedure',
                         'howto', 'tutorial', 'manual', 'guide', 'assistant', 'ai', 'llm']
        agent_score = sum(1 for keyword in agent_keywords if keyword in content_lower)

        # Developer 关键词
        developer_keywords = ['api', 'interface', 'endpoint', 'schema', 'protocol',
                             'implementation', 'code', 'library', 'framework', 'sdk']
        developer_score = sum(1 for keyword in developer_keywords if keyword in content_lower)

        # External 关键词
        external_keywords = ['user', 'customer', 'client', 'documentation', 'help',
                           'support', 'installation', 'setup', 'configuration', 'usage']
        external_score = sum(1 for keyword in external_keywords if keyword in content_lower)

        # 确定分类
        scores = {
            'agent': agent_score,
            'developer': developer_score,
            'external': external_score
        }

        doc_type = max(scores, key=scores.get)
        confidence = scores[doc_type] / max(sum(scores.values()), 1)

        reasoning = f"关键词匹配: Agent({agent_score}), Developer({developer_score}), External({external_score})"

        return json.dumps({
            "classification": doc_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "matched_keywords": [k for k, v in scores.items() if v > 0]
        })

    @tool
    def _analyze_token_usage(self, content: str, doc_type: str) -> str:
        """Token 使用分析工具"""
        # 估算 Token 数量
        english_chars = len(re.findall(r'[a-zA-Z\s]', content))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        other_chars = len(content) - english_chars - chinese_chars

        estimated_tokens = (english_chars / 4) + (chinese_chars / 1.5) + (other_chars / 3)
        estimated_tokens = int(estimated_tokens)

        # 获取 Token 限制
        token_limits = {
            'agent': 2000,
            'developer': 5000,
            'external': 3000
        }
        token_limit = token_limits.get(doc_type, 3000)
        usage_percentage = (estimated_tokens / token_limit) * 100

        # 生成分析结果
        analysis = {
            "estimated_tokens": estimated_tokens,
            "token_limit": token_limit,
            "usage_percentage": usage_percentage,
            "status": "optimal" if usage_percentage < 80 else "warning" if usage_percentage < 100 else "critical"
        }

        return json.dumps(analysis)

    @tool
    def _analyze_content_quality(self, content: str) -> str:
        """内容质量分析工具"""
        # 基础质量指标
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?。！？]', content))
        avg_sentence_length = word_count / max(sentence_count, 1)

        # 简单的可读性评分
        readability_score = max(0, 100 - (avg_sentence_length - 15) * 2)

        # 内容结构分析
        has_headings = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE)) > 0
        has_lists = len(re.findall(r'^[\-\*]\s', content, re.MULTILINE)) > 0 or len(re.findall(r'^\d+\.\s', content, re.MULTILINE)) > 0
        has_code_blocks = '```' in content

        quality_score = 50  # 基础分
        if has_headings:
            quality_score += 20
        if has_lists:
            quality_score += 15
        if has_code_blocks:
            quality_score += 15

        quality_score = min(quality_score, 100)

        analysis = {
            "quality_score": quality_score,
            "readability_score": readability_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "structure_features": {
                "has_headings": has_headings,
                "has_lists": has_lists,
                "has_code_blocks": has_code_blocks
            }
        }

        return json.dumps(analysis)


class EnhancedDocumentClassifier:
    """增强的文档分类器"""

    def __init__(self, db_path: str = "data/persistence/documents.db"):
        """
        初始化分类器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.analyzer = EnhancedDocumentAnalyzer()
        self.engine = None
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        try:
            from sqlmodel import create_engine
            self.engine = create_engine(f"sqlite:///{self.db_path}")
        except ImportError:
            logger.warning("SQLModel 未安装，将使用文件存储")

    def analyze_document(self, file_path: str) -> dict[str, Any]:
        """分析单个文档（增强版本）"""
        try:
            path = Path(file_path)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 读取文件内容
            content = path.read_text(encoding='utf-8', errors='ignore')

            # 计算文档哈希
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # 使用 LangChain 代理进行分析
            classification = None
            token_analysis = None
            content_quality = None

            if self.analyzer.classification_agent:
                try:
                    # 文档分类
                    classification_result = self.analyzer.classification_agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"请分析以下文档内容并分类:\n\n文件路径: {file_path}\n\n内容:\n{content[:2000]}..."
                        }]
                    })

                    if "structured_response" in classification_result:
                        classification_data = classification_result["structured_response"]
                    else:
                        # 解析 JSON 响应
                        import json
                        response_text = classification_result.get("content", "")
                        classification_data = json.loads(response_text)

                    classification = EnhancedDocumentClassification(
                        doc_type=classification_data.get("classification", "external"),
                        confidence=classification_data.get("confidence", 0.5),
                        reasoning=classification_data.get("reasoning", "自动分类结果"),
                        keywords=classification_data.get("matched_keywords", []),
                        metadata={
                            "analyzed_with": "langchain_agent",
                            "file_path": file_path,
                            "content_hash": content_hash,
                            "analysis_timestamp": datetime.now().isoformat()
                        }
                    )

                except Exception as e:
                    logger.warning(f"LangChain 分类失败，使用基础分类: {e}")
                    classification = self._fallback_classification(content, file_path)

            else:
                # 使用基础分类
                classification = self._fallback_classification(content, file_path)

            # Token 分析
            if self.analyzer.token_analysis_agent and classification:
                try:
                    token_result = self.analyzer.token_analysis_agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"请分析以下 {classification.doc_type} 文档的 Token 使用:\n\n{content[:2000]}..."
                        }]
                    })

                    token_data = json.loads(token_result.get("content", "{}"))
                    token_analysis = EnhancedTokenAnalysis(
                        estimated_tokens=token_data.get("estimated_tokens", 0),
                        token_limit=token_data.get("token_limit", 3000),
                        usage_percentage=token_data.get("usage_percentage", 0),
                        warnings=[],
                        recommendations=[],
                        content_quality_score=token_data.get("quality_score", 0.0)
                    )

                except Exception as e:
                    logger.warning(f"LangChain Token 分析失败: {e}")

            # 内容质量分析
            if self.analyzer.content_analyzer_agent:
                try:
                    quality_result = self.analyzer.content_analyzer_agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"请分析以下文档的内容质量:\n\n{content[:2000]}..."
                        }]
                    })

                    quality_data = json.loads(quality_result.get("content", "{}"))
                    content_quality = {
                        "quality_score": quality_data.get("quality_score", 50),
                        "readability_score": quality_data.get("readability_score", 50),
                        "structure_analysis": quality_data.get("structure_features", {})
                    }

                except Exception as e:
                    logger.warning(f"LangChain 内容质量分析失败: {e}")

            # 生成最终分析结果
            profile = {
                "file_path": file_path,
                "file_size": len(content),
                "content_hash": content_hash,
                "classification": {
                    "type": classification.doc_type,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords,
                    "metadata": classification.metadata,
                    "enhanced_analysis": True if self.analyzer.classification_agent else False
                },
                "token_analysis": {
                    "estimated_tokens": token_analysis.estimated_tokens if token_analysis else 0,
                    "token_limit": token_analysis.token_limit if token_analysis else 3000,
                    "usage_percentage": token_analysis.usage_percentage if token_analysis else 0,
                    "warnings": token_analysis.warnings if token_analysis else [],
                    "recommendations": self._generate_token_recommendations(token_analysis) if token_analysis else [],
                    "content_quality_score": token_analysis.content_quality_score if token_analysis else 0.0,
                    "readability_score": content_quality.get("readability_score", 50) if content_quality else 50
                } if token_analysis else None,
                "content_quality": content_quality,
                "analysis_metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "analyzer_version": "2.0.0 (LangChain Enhanced)",
                    "content_lines": len(content.splitlines()),
                    "langchain_features_used": {
                        "classification": self.analyzer.classification_agent is not None,
                        "token_analysis": self.analyzer.token_analysis_agent is not None,
                        "content_analysis": self.analyzer.content_analyzer_agent is not None
                    }
                }
            }

            # 保存分析结果
            self._save_enhanced_document_profile(file_path, profile)

            # 如果是 Agent 文档且有 Token 问题，记录到合规报告
            if classification.doc_type == 'agent' and token_analysis and token_analysis.usage_percentage > 100:
                self._create_enhanced_compliance_record(file_path, profile)

            logger.info(f"增强文档分析完成: {file_path} -> {classification.doc_type} (置信度: {classification.confidence:.2f})")
            return profile

        except Exception as e:
            logger.error(f"文档分析失败 {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "analyzed_at": datetime.now().isoformat()
            }

    def _fallback_classification(self, content: str, file_path: str | None = None) -> EnhancedDocumentClassification:
        """备选分类方法"""
        # 使用基础关键词匹配
        content_lower = content.lower()

        agent_keywords = ['prompt', 'instruction', 'guidance', 'workflow']
        developer_keywords = ['api', 'interface', 'implementation', 'code']
        external_keywords = ['user', 'documentation', 'help', 'installation']

        agent_score = sum(1 for keyword in agent_keywords if keyword in content_lower)
        developer_score = sum(1 for keyword in developer_keywords if keyword in content_lower)
        external_score = sum(1 for keyword in external_keywords if keyword in content_lower)

        scores = {'agent': agent_score, 'developer': developer_score, 'external': external_score}
        doc_type = max(scores, key=scores.get)
        confidence = scores[doc_type] / max(sum(scores.values()), 1)

        return EnhancedDocumentClassification(
            doc_type=doc_type,
            confidence=confidence,
            reasoning=f"基础关键词匹配: {scores}",
            keywords=[],
            metadata={
                "analyzed_with": "fallback",
                "file_path": file_path,
                "analysis_timestamp": datetime.now().isoformat()
            }
        )

    def _generate_token_recommendations(self, token_analysis: EnhancedTokenAnalysis) -> list[str]:
        """生成 Token 优化建议"""
        recommendations = []

        if token_analysis.usage_percentage > 100:
            recommendations.append("文档超出 Token 限制，建议拆分为多个部分")
            recommendations.append("移除冗余内容和重复信息")
            recommendations.append("使用更简洁的表达方式")
        elif token_analysis.usage_percentage > 80:
            recommendations.append("文档接近 Token 限制，建议优化内容")
            recommendations.append("检查是否有不必要的详细描述")

        if token_analysis.content_quality_score < 60:
            recommendations.append("内容质量较低，建议改进结构和表达")

        return recommendations

    def _save_enhanced_document_profile(self, file_path: str, profile: dict[str, Any]) -> None:
        """保存增强的文档档案"""
        output_dir = Path("data/persistence/documents")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成档案文件名
        file_hash = profile.get("content_hash", "unknown")
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', Path(file_path).name)
        profile_file = output_dir / f"enhanced_profile_{safe_filename}_{file_hash[:8]}.json"

        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def _create_enhanced_compliance_record(self, file_path: str, profile: dict[str, Any]) -> None:
        """创建增强的合规记录"""
        try:
            from sqlmodel import Session

            from src.domain.compliance.models import ComplianceReport

            if not self.engine:
                return

            # 创建合规报告
            compliance_report = ComplianceReport(
                report_id=f"enhanced_doc_review_{profile.get('content_hash', '')[:8]}",
                report_type="enhanced_doc_review",
                status="warning",
                total_checks=1,
                passed_checks=0,
                failed_checks=1,
                summary=f"Agent 文档 '{file_path}' 存在 Token 使用问题 - LangChain 增强分析",
                details={
                    "file_path": file_path,
                    "enhanced_classification": profile["classification"],
                    "enhanced_token_analysis": profile["token_analysis"],
                    "content_quality": profile["content_quality"],
                    "issues": ["Token 超限", "需要优化"],
                    "langchain_features": profile["analysis_metadata"]["langchain_features_used"]
                }
            )

            with Session(self.engine) as session:
                session.add(compliance_report)
                session.commit()

            logger.info(f"已创建增强合规记录: {file_path}")

        except Exception as e:
            logger.error(f"创建增强合规记录失败: {e}")


# 全局增强文档分类器实例
_enhanced_document_classifier: EnhancedDocumentClassifier | None = None


def get_enhanced_document_classifier() -> EnhancedDocumentClassifier:
    """获取全局增强文档分类器实例"""
    global _enhanced_document_classifier
    if _enhanced_document_classifier is None:
        _enhanced_document_classifier = EnhancedDocumentClassifier()
    return _enhanced_document_classifier


def analyze_document_enhanced(file_path: str) -> dict[str, Any]:
    """
    增强的文档分析便捷函数

    Args:
        file_path: 文档文件路径

    Returns:
        增强的分析结果
    """
    classifier = get_enhanced_document_classifier()
    return classifier.analyze_document(file_path)
