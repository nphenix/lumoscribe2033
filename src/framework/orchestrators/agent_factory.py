"""
AI 代理工厂

基于 LangChain 1.0 Runnable + AgentExecutor 创建和管理各类 AI 代理。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class AgentFactory:
    """AI 代理工厂"""

    DEFAULT_SYSTEM_PROMPTS = {
        "speckit": (
            "你是一个专业的软件项目规划专家。"
            "基于用户的需求，生成高质量的 speckit 章程、规范、计划和任务列表，"
            "确保输出格式规范、内容完整、可执行性强。"
        ),
        "doc_review": (
            "你是一个严谨的文档质量审查专家，需要从语法、逻辑、格式和完整性等"
            "角度对文档进行逐条分析，输出结构化的改进建议。"
        ),
        "compliance": (
            "你是一个资深的代码合规性审查专家。请结合项目规范与行业最佳实践，"
            "评估代码/文档的风险点，并给出清晰的整改建议与优先级。"
        ),
    }

    # ------------------------------------------------------------------ #
    # 通用 Agent 构建
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_agent_executor(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        system_prompt: str,
        *,
        extra_instructions: str | None = None,
        tags: Sequence[str] | None = None,
        max_iterations: int = 25,
        verbose: bool = False,
    ) -> AgentExecutor:
        """使用 LangChain 1.0 官方 Tool Calling Agent 创建 AgentExecutor。"""
        prompt_messages = [
            ("system", AgentFactory._compose_system_prompt(system_prompt, extra_instructions)),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        agent = create_tool_calling_agent(llm, list(tools), prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=list(tools),
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            tags=list(tags) if tags else None,
        )
        return executor

    @staticmethod
    def create_speckit_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        *,
        system_prompt: str | None = None,
        extra_instructions: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> AgentExecutor:
        """创建 Speckit 代理"""
        prompt = system_prompt or AgentFactory.DEFAULT_SYSTEM_PROMPTS["speckit"]
        return AgentFactory.create_agent_executor(
            llm,
            tools,
            prompt,
            extra_instructions=extra_instructions,
            tags=tags or ["speckit-agent"],
        )

    @staticmethod
    def create_doc_review_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        *,
        system_prompt: str | None = None,
        extra_instructions: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> AgentExecutor:
        """创建文档审查代理"""
        prompt = system_prompt or AgentFactory.DEFAULT_SYSTEM_PROMPTS["doc_review"]
        return AgentFactory.create_agent_executor(
            llm,
            tools,
            prompt,
            extra_instructions=extra_instructions,
            tags=tags or ["doc-review-agent"],
        )

    @staticmethod
    def create_compliance_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        *,
        system_prompt: str | None = None,
        extra_instructions: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> AgentExecutor:
        """创建合规检查代理"""
        prompt = system_prompt or AgentFactory.DEFAULT_SYSTEM_PROMPTS["compliance"]
        return AgentFactory.create_agent_executor(
            llm,
            tools,
            prompt,
            extra_instructions=extra_instructions,
            tags=tags or ["compliance-agent"],
        )

    # ------------------------------------------------------------------ #
    # 基础链构建
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_simple_chain(
        llm: BaseChatModel,
        template: str,
    ) -> Runnable:
        """创建最简 Runnable 链 (Prompt -> LLM)。"""
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | llm

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_system_prompt(base_prompt: str, extra: str | None) -> str:
        if extra:
            return f"{base_prompt}\n\n补充要求：{extra.strip()}"
        return base_prompt
