"""
生成器模块（结合检索上下文）

实现 RAG（Retrieval-Augmented Generation）生成器，
将检索到的文档作为上下文注入到大语言模型中。
"""

from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod

from .retriever import BaseRetriever


class BaseGenerator(ABC):
    """
    生成器基类。

    所有生成器实现都应继承此类。
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        生成回答。

        Args:
            query: 查询文本
            context: 上下文文档列表
            **kwargs: 其他生成参数

        Returns:
            生成的回答
        """
        pass


class RAGGenerator(BaseGenerator):
    """
    RAG 生成器：结合检索与生成。

    工作流程：
    1. 使用检索器获取相关文档
    2. 将文档格式化为上下文
    3. 调用 LLM 生成回答

    Args:
        retriever: 检索器实例
        llm_fn: LLM 调用函数
        prompt_template: Prompt 模板
        max_context_length: 最大上下文长度
        top_k: 检索文档数量
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm_fn: Optional[Callable] = None,
        prompt_template: str = "",
        max_context_length: int = 4000,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm_fn = llm_fn or self._default_llm_fn
        self.prompt_template = prompt_template or (
            "基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明。\n\n"
            "上下文：\n{context}\n\n"
            "问题：{question}\n\n"
            "回答："
        )
        self.max_context_length = max_context_length
        self.top_k = top_k

    def _default_llm_fn(self, prompt: str, **kwargs) -> str:
        """默认 LLM 函数（占位）。"""
        return f"[模拟回答] 收到 Prompt: {prompt[:100]}..."

    def _format_context(self, documents: List[Dict]) -> str:
        """
        将检索到的文档格式化为上下文。

        Args:
            documents: 文档列表

        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            context_parts.append(f"[文档{i+1}] {text}")

        context = "\n\n".join(context_parts)

        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."

        return context

    def retrieve_context(self, query: str) -> List[Dict]:
        """
        检索相关上下文。

        Args:
            query: 查询文本

        Returns:
            检索到的文档列表
        """
        if self.retriever is None:
            return []
        return self.retriever.retrieve(query, top_k=self.top_k)

    def generate(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        生成回答。

        Args:
            query: 查询文本
            context: 可选的预检索上下文
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            生成的回答
        """
        if context is None:
            context = self.retrieve_context(query)

        context_text = self._format_context(context)
        prompt = self.prompt_template.format(
            context=context_text,
            question=query,
        )

        response = self.llm_fn(prompt, **kwargs)
        return response

    def generate_with_sources(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict:
        """
        生成回答并附带引用来源。

        Args:
            query: 查询文本
            context: 可选的预检索上下文
            **kwargs: 其他参数

        Returns:
            包含 answer 和 sources 的字典
        """
        if context is None:
            context = self.retrieve_context(query)

        answer = self.generate(query, context=context, **kwargs)

        sources = []
        for doc in context:
            sources.append({
                "id": doc.get("id"),
                "text": doc.get("text", "")[:200],
                "score": doc.get("score", 0),
            })

        return {
            "answer": answer,
            "sources": sources,
        }


class MultiQueryGenerator(BaseGenerator):
    """
    多查询生成器：对同一问题生成多种查询变体并合并结果。

    Args:
        generator: 基础 RAG 生成器
        num_variants: 查询变体数量
    """

    def __init__(
        self,
        generator: RAGGenerator,
        num_variants: int = 3,
    ):
        self.generator = generator
        self.num_variants = num_variants

    def _expand_query(self, query: str) -> List[str]:
        """
        扩展查询为多个变体。

        Args:
            query: 原始查询

        Returns:
            查询变体列表
        """
        variants = [query]
        templates = [
            "关于'{query}'的相关信息",
            "请解释'{query}'",
            "'{query}'是什么？",
        ]
        for template in templates[:self.num_variants - 1]:
            variants.append(template.format(query=query))
        return variants

    def generate(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        使用多查询策略生成回答。

        Args:
            query: 查询文本
            context: 可选上下文
            **kwargs: 其他参数

        Returns:
            生成的回答
        """
        variants = self._expand_query(query)
        all_contexts = []

        for variant in variants:
            ctx = self.generator.retrieve_context(variant)
            all_contexts.extend(ctx)

        seen_ids = set()
        unique_contexts = []
        for ctx in all_contexts:
            if ctx.get("id") not in seen_ids:
                seen_ids.add(ctx.get("id"))
                unique_contexts.append(ctx)

        return self.generator.generate(query, context=unique_contexts, **kwargs)