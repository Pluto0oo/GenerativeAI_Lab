"""
Prompt 策略实现类

提供高级 Prompt 优化策略，包括：
- 基础策略（直接调用）
- Few-Shot 策略（示例选择）
- CoT 策略（思维链增强）
- ToT 策略（思维树搜索）
- Self-Consistency 策略（自洽性投票）
- 组合策略（多策略融合）
"""

import random
from typing import List, Dict, Optional, Callable
from .templates import (
    PromptTemplate,
    ZERO_SHOT_QA,
    FEW_SHOT_QA,
    COT_TEMPLATE,
    COT_WITH_EXAMPLES,
    TOT_TEMPLATE,
    SELF_CONSISTENCY_TEMPLATE,
)


class BasePromptStrategy:
    """
    基础 Prompt 策略接口。

    所有 Prompt 策略都应继承此类并实现 build_prompt 方法。
    """

    def __init__(self, template: Optional[PromptTemplate] = None):
        self.template = template

    def build_prompt(self, **kwargs) -> str:
        """
        构建最终 Prompt。

        Args:
            **kwargs: 模板变量

        Returns:
            格式化后的 Prompt 字符串
        """
        if self.template is None:
            raise NotImplementedError("子类必须实现 build_prompt 或设置 template")
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(template={self.template.name if self.template else 'None'})"


class ZeroShotStrategy(BasePromptStrategy):
    """
    零样本策略：直接使用模板，无需示例。
    """

    def __init__(self, template: Optional[PromptTemplate] = None):
        super().__init__(template or ZERO_SHOT_QA)


class FewShotStrategy(BasePromptStrategy):
    """
    少样本策略：从示例库中选择示例注入模板。

    Attributes:
        examples: 示例列表
        max_examples: 最大示例数
        selection_strategy: 示例选择策略（random/first/most_similar）
    """

    def __init__(
        self,
        template: Optional[PromptTemplate] = None,
        examples: Optional[List[Dict]] = None,
        max_examples: int = 3,
        selection_strategy: str = "first",
    ):
        super().__init__(template or FEW_SHOT_QA)
        self.examples = examples or []
        self.max_examples = max_examples
        self.selection_strategy = selection_strategy

    def _select_examples(self, query: str = "") -> List[Dict]:
        """
        根据策略选择示例。

        Args:
            query: 查询文本（用于相似度选择）

        Returns:
            选中的示例列表
        """
        if not self.examples:
            return []

        if self.selection_strategy == "random":
            return random.sample(
                self.examples, min(self.max_examples, len(self.examples))
            )
        elif self.selection_strategy == "most_similar":
            query_words = set(query.lower().split())
            scored = [
                (len(query_words & set(str(e).lower().split())), e)
                for e in self.examples
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:self.max_examples]]
        else:
            return self.examples[:self.max_examples]

    def build_prompt(self, query: str = "", **kwargs) -> str:
        """
        构建少样本 Prompt。

        Args:
            query: 查询文本
            **kwargs: 其他模板变量

        Returns:
            格式化后的 Prompt
        """
        selected = self._select_examples(query)
        examples_text = "\n".join(
            f"示例{i+1}：{ex}" for i, ex in enumerate(selected)
        )
        merged = {"examples": examples_text, **kwargs}
        return self.template.format(**merged)


class ChainOfThoughtStrategy(BasePromptStrategy):
    """
    思维链策略：引导模型逐步推理。

    Attributes:
        with_examples: 是否在 CoT 中使用示例
        cot_examples: CoT 示例列表
    """

    def __init__(
        self,
        template: Optional[PromptTemplate] = None,
        with_examples: bool = False,
        cot_examples: Optional[List[str]] = None,
    ):
        default_template = COT_WITH_EXAMPLES if with_examples else COT_TEMPLATE
        super().__init__(template or default_template)
        self.with_examples = with_examples
        self.cot_examples = cot_examples or []

    def build_prompt(self, **kwargs) -> str:
        """
        构建 CoT Prompt。
        """
        if self.with_examples and self.cot_examples:
            examples_text = "\n".join(self.cot_examples)
            return self.template.format(examples=examples_text, **kwargs)
        return self.template.format(**kwargs)


class TreeOfThoughtStrategy(BasePromptStrategy):
    """
    思维树策略：探索多条推理路径并择优。

    Attributes:
        num_branches: 分支数量
        evaluator: 分支评估函数
    """

    def __init__(
        self,
        template: Optional[PromptTemplate] = None,
        num_branches: int = 3,
        evaluator: Optional[Callable] = None,
    ):
        super().__init__(template or TOT_TEMPLATE)
        self.num_branches = num_branches
        self.evaluator = evaluator

    def build_prompt(self, **kwargs) -> str:
        """
        构建 ToT Prompt。
        """
        return self.template.format(num_branches=self.num_branches, **kwargs)

    def evaluate_thoughts(self, thoughts: List[str]) -> int:
        """
        评估各分支，返回最佳分支索引。

        Args:
            thoughts: 分支内容列表

        Returns:
            最佳分支索引
        """
        if self.evaluator:
            return self.evaluator(thoughts)
        return 0


class SelfConsistencyStrategy(BasePromptStrategy):
    """
    自洽性策略：通过多条独立路径生成答案，最终投票。

    Attributes:
        num_paths: 采样路径数量
        temperature: 采样温度
    """

    def __init__(
        self,
        template: Optional[PromptTemplate] = None,
        num_paths: int = 3,
        temperature: float = 0.7,
    ):
        super().__init__(template or SELF_CONSISTENCY_TEMPLATE)
        self.num_paths = num_paths
        self.temperature = temperature

    def build_prompt(self, **kwargs) -> str:
        """
        构建 Self-Consistency Prompt。
        """
        return self.template.format(num_paths=self.num_paths, **kwargs)

    def aggregate_answers(self, answers: List[str]) -> str:
        """
        对多条路径的答案进行多数投票。

        Args:
            answers: 各路径的答案列表

        Returns:
            最终答案
        """
        if not answers:
            return ""
        from collections import Counter
        counter = Counter(answers)
        return counter.most_common(1)[0][0]


class CompositeStrategy(BasePromptStrategy):
    """
    组合策略：将多个策略串联使用。

    Attributes:
        strategies: 策略列表
    """

    def __init__(self, strategies: List[BasePromptStrategy]):
        super().__init__()
        self.strategies = strategies

    def build_prompt(self, **kwargs) -> str:
        """
        依次应用所有策略构建 Prompt。
        """
        result = kwargs
        for strategy in self.strategies:
            if hasattr(strategy, 'build_prompt'):
                result = strategy.build_prompt(**result)
                if isinstance(result, str):
                    break
        if isinstance(result, str):
            return result
        return str(result)

    def __repr__(self) -> str:
        return f"CompositeStrategy({self.strategies})"