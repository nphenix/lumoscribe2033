"""
增强的元数据注入器

基于 LangChain 1.0 最佳实践实现，集成结构化输出、中间件和智能分析功能。
提供更智能的元数据生成和验证能力。
"""

import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# LangChain 1.0 导入
try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        HumanInTheLoopMiddleware,
        PIIMiddleware,
        SummarizationMiddleware,
    )
    from langchain.chat_models import init_chat_model
    from langchain.tools import tool
    from pydantic import BaseModel as PydanticBaseModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain 1.0 未安装，将使用基础实现")
    LANGCHAIN_AVAILABLE = False

from src.framework.shared.utils import get_git_info


@dataclass
class EnhancedMetadata:
    """增强的元数据结构"""
    command: str
    timestamp: str
    git_commit: str | None = None
    git_branch: str | None = None
    author: str | None = None
    # LangChain 1.0 增强字段
    content_summary: str | None = None
    content_type: str | None = None
    quality_score: float | None = None
    detected_issues: list[str] | None = None
    recommendations: list[str] | None = None
    processing_metadata: dict[str, Any] | None = None


class EnhancedMetadataGenerator:
    """增强的元数据生成器 - 使用 LangChain 1.0 最佳实践"""

    def __init__(self):
        """初始化生成器"""
        self.summary_agent = None
        self.quality_agent = None
        self.issue_detector_agent = None

        if LANGCHAIN_AVAILABLE:
            self._setup_langchain_agents()

    def _setup_langchain_agents(self):
        """设置 LangChain 代理"""
        try:
            # 内容摘要生成代理
            self.summary_agent = create_agent(
                model="claude-sonnet-4-5-20250929",
                tools=[self._generate_content_summary],
                system_prompt="""
                你是一个专业的文档摘要生成专家。请为以下内容生成简洁的摘要：

                要求：
                1. 突出核心内容和关键信息
                2. 保持客观和准确
                3. 长度控制在 2-3 句话
                4. 使用中文表达

                返回格式：
                {
                    "summary": "内容摘要",
                    "key_points": ["关键点1", "关键点2"],
                    "content_type": "文档类型"
                }
                """
            )

            # 质量评估代理
            self.quality_agent = create_agent(
                model="gpt-4o-mini",
                tools=[self._assess_content_quality],
                system_prompt="""
                你是一个专业的内容质量评估专家。请对以下内容进行质量评估：

                评估维度：
                1. 内容完整性 (0-25分)
                2. 逻辑结构 (0-25分)
                3. 语言表达 (0-25分)
                4. 技术准确性 (0-25分)

                返回格式：
                {
                    "overall_score": 85,
                    "detailed_scores": {
                        "completeness": 20,
                        "structure": 22,
                        "language": 20,
                        "accuracy": 23
                    },
                    "strengths": ["优点1", "优点2"],
                    "weaknesses": ["缺点1", "缺点2"]
                }
                """
            )

            # 问题检测代理
            self.issue_detector_agent = create_agent(
                model="claude-sonnet-4-5-20250929",
                tools=[self._detect_content_issues],
                system_prompt="""
                你是一个专业的文档问题检测专家。请检测以下内容中的潜在问题：

                检查维度：
                1. 技术错误和不一致
                2. 格式问题
                3. 语言表达问题
                4. 结构完整性问题
                5. 安全和合规问题

                返回格式：
                {
                    "issues": [
                        {"type": "technical", "severity": "high", "description": "问题描述", "line": 10}
                    ],
                    "recommendations": [
                        {"priority": "high", "action": "建议操作", "description": "详细说明"}
                    ]
                }
                """
            )

            logger.info("✅ LangChain 元数据生成代理初始化成功")

        except Exception as e:
            logger.error(f"❌ LangChain 代理初始化失败: {e}")
            self.summary_agent = None
            self.quality_agent = None
            self.issue_detector_agent = None

    @tool
    def _generate_content_summary(self, content: str, file_path: str | None = None) -> str:
        """生成内容摘要工具"""
        # 基础摘要生成（备选方案）
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if len(non_empty_lines) == 0:
            summary = "空文档"
            content_type = "unknown"
        elif len(non_empty_lines) <= 10:
            summary = content[:200] + "..." if len(content) > 200 else content
            content_type = self._detect_content_type(content)
        else:
            # 提取前几行、中间关键行和后几行
            first_lines = non_empty_lines[:3]
            middle_lines = non_empty_lines[len(non_empty_lines)//2-1:len(non_empty_lines)//2+2] if len(non_empty_lines) > 10 else []
            last_lines = non_empty_lines[-3:]

            key_content = " ".join(first_lines + middle_lines + last_lines)
            summary = key_content[:300] + "..." if len(key_content) > 300 else key_content
            content_type = self._detect_content_type(content)

        return json.dumps({
            "summary": summary,
            "key_points": self._extract_key_points(content),
            "content_type": content_type
        })

    @tool
    def _assess_content_quality(self, content: str, file_path: str | None = None) -> str:
        """评估内容质量工具"""
        # 基础质量评分
        score = 50  # 基础分

        # 内容长度检查
        word_count = len(content.split())
        if word_count < 50:
            score -= 10
        elif word_count > 5000:
            score -= 5

        # 结构检查
        has_headings = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE)) > 0
        has_lists = len(re.findall(r'^[\-\*]\s', content, re.MULTILINE)) > 0
        has_code_blocks = '```' in content

        if has_headings:
            score += 15
        if has_lists:
            score += 10
        if has_code_blocks:
            score += 10

        # 语言检查
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', content)) / max(len(content), 1)
        if chinese_ratio > 0.1:  # 包含中文
            score += 5

        # 格式检查
        has_proper_formatting = bool(re.search(r'\*\*.*\*\*|_.*_|\[.*\]\(.*\)', content))
        if has_proper_formatting:
            score += 5

        quality_result = {
            "overall_score": min(score, 100),
            "detailed_scores": {
                "completeness": min(25, score // 4 + 5),
                "structure": 25 if has_headings else 15,
                "language": 20 if chinese_ratio > 0.1 else 15,
                "accuracy": 20  # 基础准确性评分
            },
            "strengths": self._get_quality_strengths(content, score),
            "weaknesses": self._get_quality_weaknesses(content, score)
        }

        return json.dumps(quality_result)

    @tool
    def _detect_content_issues(self, content: str, file_path: str | None = None) -> str:
        """检测内容问题工具"""
        issues = []
        recommendations = []

        # 技术问题检测
        if 'TODO' in content or 'FIXME' in content or 'XXX' in content:
            issues.append({
                "type": "technical",
                "severity": "medium",
                "description": "发现待办事项标记",
                "content": content[content.find('TODO'):content.find('TODO')+50] if 'TODO' in content else ""
            })
            recommendations.append({
                "priority": "medium",
                "action": "完成待办事项",
                "description": "移除或完成文档中的 TODO/FIXME 标记"
            })

        # 格式问题检测
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 200:  # 过长行
                issues.append({
                    "type": "format",
                    "severity": "low",
                    "description": f"第 {i+1} 行过长 ({len(line)} 字符)",
                    "line": i + 1
                })
                recommendations.append({
                    "priority": "low",
                    "action": "分行长行",
                    "description": f"将第 {i+1} 行拆分为多行以提高可读性"
                })

        # 语言问题检测
        if content.count('。') > 0 and content.count('.') == 0:
            # 纯中文内容检查
            if '，' in content and content.count('，') > content.count('。') * 2:
                issues.append({
                    "type": "language",
                    "severity": "medium",
                    "description": "逗号使用过多，建议优化句子结构",
                    "content": content[:100]
                })

        return json.dumps({
            "issues": issues,
            "recommendations": recommendations
        })

    def _detect_content_type(self, content: str) -> str:
        """检测内容类型"""
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ['api', 'endpoint', 'interface']):
            return "api_documentation"
        elif any(keyword in content_lower for keyword in ['guide', 'tutorial', 'howto']):
            return "user_guide"
        elif any(keyword in content_lower for keyword in ['spec', 'specification', 'requirement']):
            return "specification"
        elif any(keyword in content_lower for keyword in ['plan', 'roadmap', 'strategy']):
            return "plan"
        elif any(keyword in content_lower for keyword in ['code', 'implementation', 'source']):
            return "implementation"
        else:
            return "general_documentation"

    def _extract_key_points(self, content: str) -> list[str]:
        """提取关键点"""
        key_points = []

        # 提取标题
        headings = re.findall(r'^#{1,3}\s+(.+)', content, re.MULTILINE)
        key_points.extend(headings[:3])  # 最多提取3个标题

        # 提取列表项
        list_items = re.findall(r'^[\-\*]\s+(.{1,100})', content, re.MULTILINE)
        key_points.extend(list_items[:2])  # 最多提取2个列表项

        return key_points

    def _get_quality_strengths(self, content: str, score: int) -> list[str]:
        """获取质量优点"""
        strengths = []

        if score >= 80:
            strengths.append("内容质量优秀")
        elif score >= 60:
            strengths.append("内容质量良好")

        if '```' in content:
            strengths.append("包含代码示例")

        if re.findall(r'^#{1,3}\s+', content, re.MULTILINE):
            strengths.append("结构清晰")

        if len(content.split()) > 100:
            strengths.append("内容详实")

        return strengths

    def _get_quality_weaknesses(self, content: str, score: int) -> list[str]:
        """获取质量缺点"""
        weaknesses = []

        if score < 60:
            weaknesses.append("内容质量需要改进")

        if len(content.split()) < 50:
            weaknesses.append("内容过于简短")

        if not re.findall(r'^#{1,3}\s+', content, re.MULTILINE):
            weaknesses.append("缺乏结构化标题")

        # 功能性 TODO 不计入弱点
        if any(pattern in content for pattern in ['TODO:', 'FIXME:', 'XXX:']):
            weaknesses.append("包含未完成的待办事项")

        return weaknesses


class EnhancedMetadataInjector:
    """增强的元数据注入器"""

    def __init__(self):
        """初始化注入器"""
        self.generator = EnhancedMetadataGenerator()
        self.enhanced_header_template = """<!-- generated: {command} @ {timestamp} -->
<!-- git-commit: {git_commit} -->
<!-- git-branch: {git_branch} -->
<!-- author: {author} -->
<!-- content-summary: {content_summary} -->
<!-- content-type: {content_type} -->
<!-- quality-score: {quality_score} -->
<!-- detected-issues: {detected_issues} -->
<!-- langchain-enhanced: true -->

"""
        self.basic_header_template = """<!-- generated: {command} @ {timestamp} -->
<!-- git-commit: {git_commit} -->
<!-- git-branch: {git_branch} -->
<!-- author: {author} -->

"""

    def inject_metadata(self, file_path: str, command: str = "enhanced_metadata_injector") -> bool:
        """
        注入增强元数据到文件

        Args:
            file_path: 文件路径
            command: 生成命令

        Returns:
            是否成功注入
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False

            # 读取文件内容
            content = path.read_text(encoding='utf-8', errors='ignore')

            # 如果已经有元数据头，先移除
            content = self._remove_existing_metadata(content)

            # 生成增强元数据
            metadata = self._generate_enhanced_metadata(content, command, file_path)

            # 根据 LangChain 可用性选择模板
            if LANGCHAIN_AVAILABLE and metadata.content_summary:
                header = self.enhanced_header_template.format(
                    command=command,
                    timestamp=metadata.timestamp,
                    git_commit=metadata.git_commit or "unknown",
                    git_branch=metadata.git_branch or "unknown",
                    author=metadata.author or "unknown",
                    content_summary=metadata.content_summary,
                    content_type=metadata.content_type or "unknown",
                    quality_score=metadata.quality_score or 0,
                    detected_issues=len(metadata.detected_issues or [])
                )
            else:
                header = self.basic_header_template.format(
                    command=command,
                    timestamp=metadata.timestamp,
                    git_commit=metadata.git_commit or "unknown",
                    git_branch=metadata.git_branch or "unknown",
                    author=metadata.author or "unknown"
                )

            # 写入新内容
            new_content = header + content
            path.write_text(new_content, encoding='utf-8')

            logger.info(f"✅ 增强元数据注入成功: {file_path}")
            return True

        except Exception as e:
            logger.error(f"❌ 元数据注入失败 {file_path}: {e}")
            return False

    def verify_metadata(self, file_path: str) -> dict[str, Any]:
        """
        验证文件的元数据

        Args:
            file_path: 文件路径

        Returns:
            验证结果
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"valid": False, "error": "文件不存在"}

            content = path.read_text(encoding='utf-8', errors='ignore')

            # 提取元数据头
            metadata_match = re.match(r'<!-- generated: (.+?) -->\s*' +
                                    r'<!-- git-commit: (.+?) -->\s*' +
                                    r'<!-- git-branch: (.+?) -->\s*' +
                                    r'<!-- author: (.+?) -->\s*' +
                                    r'(?:<!-- content-summary: (.+?) -->\s*)?' +
                                    r'(?:<!-- content-type: (.+?) -->\s*)?' +
                                    r'(?:<!-- quality-score: (.+?) -->\s*)?' +
                                    r'(?:<!-- langchain-enhanced: (.+?) -->\s*)?', content)

            if not metadata_match:
                return {"valid": False, "error": "未找到元数据头"}

            # 验证 Git 信息
            git_info = get_git_info()
            git_commit_match = metadata_match.group(2) == git_info.get('commit', 'unknown')
            git_branch_match = metadata_match.group(3) == git_info.get('branch', 'unknown')

            # 验证时间戳格式
            timestamp = metadata_match.group(1).split(' @ ')[1] if ' @ ' in metadata_match.group(1) else None
            timestamp_valid = self._validate_timestamp(timestamp)

            # 验证内容完整性
            content_after_header = content[metadata_match.end():]
            content_integrity = self._verify_content_integrity(content_after_header)

            # LangChain 增强验证
            langchain_enhanced = metadata_match.group(8) == "true" if metadata_match.group(8) else False
            enhanced_fields_present = bool(metadata_match.group(5) and metadata_match.group(6))

            verification_result = {
                "valid": True,
                "file_path": file_path,
                "command": metadata_match.group(1),
                "timestamp": timestamp,
                "git_commit": metadata_match.group(2),
                "git_branch": metadata_match.group(3),
                "author": metadata_match.group(4),
                "git_commit_match": git_commit_match,
                "git_branch_match": git_branch_match,
                "timestamp_valid": timestamp_valid,
                "content_integrity": content_integrity,
                "langchain_enhanced": langchain_enhanced,
                "enhanced_fields_present": enhanced_fields_present,
                "overall_valid": all([
                    git_commit_match,
                    git_branch_match,
                    timestamp_valid,
                    content_integrity
                ])
            }

            # 如果是 LangChain 增强版本，添加额外验证
            if langchain_enhanced:
                verification_result.update({
                    "content_summary": metadata_match.group(5),
                    "content_type": metadata_match.group(6),
                    "quality_score": metadata_match.group(7),
                    "enhanced_verification": enhanced_fields_present
                })

            return verification_result

        except Exception as e:
            logger.error(f"元数据验证失败 {file_path}: {e}")
            return {"valid": False, "error": str(e)}

    def _generate_enhanced_metadata(self, content: str, command: str, file_path: str) -> EnhancedMetadata:
        """生成增强元数据"""
        # 基础 Git 信息
        git_info = get_git_info()
        timestamp = datetime.now().isoformat()

        # LangChain 增强分析
        summary = None
        content_type = None
        quality_score = None
        detected_issues = []
        recommendations = []

        if LANGCHAIN_AVAILABLE:
            try:
                # 内容摘要生成
                if self.generator.summary_agent:
                    summary_result = self.generator.summary_agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"请为以下内容生成摘要:\n\n文件路径: {file_path}\n\n内容:\n{content[:2000]}..."
                        }]
                    })

                    summary_data = json.loads(summary_result.get("content", "{}"))
                    summary = summary_data.get("summary")
                    content_type = summary_data.get("content_type")

                # 质量评估
                if self.generator.quality_agent:
                    quality_result = self.generator.quality_agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"请评估以下内容的质量:\n\n{content[:1000]}..."
                        }]
                    })

                    quality_data = json.loads(quality_result.get("content", "{}"))
                    quality_score = quality_data.get("overall_score")

                # 问题检测
                if self.generator.issue_detector_agent:
                    issue_result = self.generator.issue_detector_agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": f"请检测以下内容中的问题:\n\n文件路径: {file_path}\n\n内容:\n{content[:1500]}..."
                        }]
                    })

                    issue_data = json.loads(issue_result.get("content", "{}"))
                    detected_issues = issue_data.get("issues", [])
                    recommendations = issue_data.get("recommendations", [])

            except Exception as e:
                logger.warning(f"LangChain 增强分析失败: {e}")
                # 使用基础分析作为备选

        return EnhancedMetadata(
            command=command,
            timestamp=timestamp,
            git_commit=git_info.get('commit'),
            git_branch=git_info.get('branch'),
            author=git_info.get('author'),
            content_summary=summary,
            content_type=content_type,
            quality_score=quality_score,
            detected_issues=detected_issues,
            recommendations=recommendations,
            processing_metadata={
                "langchain_features_used": {
                    "summary": self.generator.summary_agent is not None,
                    "quality": self.generator.quality_agent is not None,
                    "issue_detection": self.generator.issue_detector_agent is not None
                },
                "content_length": len(content),
                "word_count": len(content.split())
            }
        )

    def _remove_existing_metadata(self, content: str) -> str:
        """移除现有元数据头"""
        # 移除增强元数据头
        enhanced_pattern = r'<!-- generated: .+? -->\s*' + \
                          r'<!-- git-commit: .+? -->\s*' + \
                          r'<!-- git-branch: .+? -->\s*' + \
                          r'<!-- author: .+? -->\s*' + \
                          r'(?:<!-- content-summary: .+? -->\s*)?' + \
                          r'(?:<!-- content-type: .+? -->\s*)?' + \
                          r'(?:<!-- quality-score: .+? -->\s*)?' + \
                          r'(?:<!-- langchain-enhanced: .+? -->\s*)?' + \
                          r'\s*'

        content = re.sub(enhanced_pattern, '', content, count=1)

        # 移除基础元数据头
        basic_pattern = r'<!-- generated: .+? -->\s*' + \
                       r'<!-- git-commit: .+? -->\s*' + \
                       r'<!-- git-branch: .+? -->\s*' + \
                       r'<!-- author: .+? -->\s*'

        content = re.sub(basic_pattern, '', content, count=1)

        return content.strip()

    def _validate_timestamp(self, timestamp: str | None) -> bool:
        """验证时间戳格式"""
        if not timestamp:
            return False

        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False

    def _verify_content_integrity(self, content: str) -> bool:
        """验证内容完整性"""
        # 检查内容是否为空
        if not content.strip():
            return False

        # 检查是否有明显的损坏
        if content.count('�') > 0:
            return False

        return True


# 全局增强元数据注入器实例
_enhanced_metadata_injector: EnhancedMetadataInjector | None = None


def get_enhanced_metadata_injector() -> EnhancedMetadataInjector:
    """获取全局增强元数据注入器实例"""
    global _enhanced_metadata_injector
    if _enhanced_metadata_injector is None:
        _enhanced_metadata_injector = EnhancedMetadataInjector()
    return _enhanced_metadata_injector


def inject_enhanced_metadata(file_path: str, command: str = "enhanced_metadata_injector") -> bool:
    """
    增强元数据注入便捷函数

    Args:
        file_path: 文件路径
        command: 生成命令

    Returns:
        是否成功
    """
    injector = get_enhanced_metadata_injector()
    return injector.inject_metadata(file_path, command)


def verify_enhanced_metadata(file_path: str) -> dict[str, Any]:
    """
    增强元数据验证便捷函数

    Args:
        file_path: 文件路径

    Returns:
        验证结果
    """
    injector = get_enhanced_metadata_injector()
    return injector.verify_metadata(file_path)
