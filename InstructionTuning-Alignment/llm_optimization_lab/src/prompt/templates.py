"""
Prompt 模板库

提供多种 Prompt 模板，包括：
- Zero-Shot 模板
- Few-Shot 模板
- Chain-of-Thought (CoT) 模板
- Tree-of-Thought (ToT) 模板
- Self-Consistency 模板
- Role-based 模板
"""

from string import Template
from typing import List, Dict, Optional


class PromptTemplate:
    """
    Prompt 模板类，用于管理和格式化 Prompt 模板。

    Attributes:
        name: 模板名称
        template: 模板字符串（支持 {variable} 占位符）
        description: 模板描述
    """

    def __init__(self, name: str, template: str, description: str = ""):
        self.name = name
        self.template = template
        self.description = description
        self._variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """提取模板中的变量名。"""
        import re
        return list(set(re.findall(r'\{(\w+)\}', self.template)))

    def format(self, **kwargs) -> str:
        """
        使用提供的参数格式化模板。

        Args:
            **kwargs: 模板变量的键值对

        Returns:
            格式化后的字符串
        """
        missing = set(self._variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"缺少模板变量: {missing}")
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name}, variables={self._variables})"


# ============ Zero-Shot 模板 ============

ZERO_SHOT_CLASSIFICATION = PromptTemplate(
    name="zero_shot_classification",
    template=(
        "请对以下文本进行情感分类，类别包括：{labels}。\n\n"
        "文本：{text}\n\n"
        "情感类别："
    ),
    description="零样本文本分类模板"
)

ZERO_SHOT_QA = PromptTemplate(
    name="zero_shot_qa",
    template=(
        "问题：{question}\n\n"
        "请回答上述问题。"
    ),
    description="零样本问答模板"
)

ZERO_SHOT_SUMMARY = PromptTemplate(
    name="zero_shot_summary",
    template=(
        "请为以下文本生成一段摘要：\n\n"
        "原文：{text}\n\n"
        "摘要："
    ),
    description="零样本摘要生成模板"
)

# ============ Few-Shot 模板 ============

FEW_SHOT_CLASSIFICATION = PromptTemplate(
    name="few_shot_classification",
    template=(
        "以下是一些文本情感分类的示例：\n\n"
        "{examples}\n\n"
        "现在请对以下文本进行分类：\n"
        "文本：{text}\n"
        "情感类别："
    ),
    description="少样本分类模板"
)

FEW_SHOT_QA = PromptTemplate(
    name="few_shot_qa",
    template=(
        "以下是一些问答示例：\n\n"
        "{examples}\n\n"
        "问题：{question}\n"
        "答案："
    ),
    description="少样本问答模板"
)

# ============ Chain-of-Thought (CoT) 模板 ============

COT_TEMPLATE = PromptTemplate(
    name="chain_of_thought",
    template=(
        "问题：{question}\n\n"
        "请一步步思考（Chain-of-Thought），然后给出最终答案。\n\n"
        "思考过程："
    ),
    description="思维链模板，引导模型逐步推理"
)

COT_WITH_EXAMPLES = PromptTemplate(
    name="chain_of_thought_with_examples",
    template=(
        "以下是一些使用思维链推理的示例：\n\n"
        "{examples}\n\n"
        "问题：{question}\n\n"
        "请一步步思考，然后给出最终答案。\n\n"
        "思考过程："
    ),
    description="带示例的思维链模板"
)

# ============ Tree-of-Thought (ToT) 模板 ============

TOT_TEMPLATE = PromptTemplate(
    name="tree_of_thought",
    template=(
        "问题：{question}\n\n"
        "请提出 {num_branches} 种不同的解题思路，"
        "评估每种思路的优劣，然后选择最佳方案并给出最终答案。\n\n"
        "思路1：\n"
        "思路2：\n"
        "思路3：\n\n"
        "最佳方案："
    ),
    description="思维树模板，探索多条推理路径"
)

TOT_EVALUATION = PromptTemplate(
    name="tree_of_thought_evaluation",
    template=(
        "问题：{question}\n\n"
        "以下是几种可能的解法：\n{thoughts}\n\n"
        "请评估每种解法的正确性和可行性，选出最优解法。\n\n"
        "评估结果："
    ),
    description="思维树评估模板"
)

# ============ Self-Consistency 模板 ============

SELF_CONSISTENCY_TEMPLATE = PromptTemplate(
    name="self_consistency",
    template=(
        "问题：{question}\n\n"
        "请用 {num_paths} 种不同的方法解决此问题，"
        "然后综合得出最终答案。\n\n"
        "方法1答案：\n"
        "方法2答案：\n"
        "方法3答案：\n\n"
        "最终答案（多数投票）："
    ),
    description="自洽性模板，通过多条路径投票"
)

# ============ Role-based 模板 ============

ROLE_EXPERT = PromptTemplate(
    name="role_expert",
    template=(
        "你是一位{role}专家。请以专业的角度回答以下问题：\n\n"
        "问题：{question}\n\n"
        "回答："
    ),
    description="角色扮演模板"
)

ROLE_TEACHER = PromptTemplate(
    name="role_teacher",
    template=(
        "你是一位经验丰富的教师。请针对以下问题给出详细讲解，"
        "包括核心概念、解题步骤和常见误区。\n\n"
        "问题：{question}\n\n"
        "讲解："
    ),
    description="教师角色模板"
)

# ============ 常用示例库 ============

SENTIMENT_EXAMPLES = [
    {"text": "这部电影太棒了，演员表演出色！", "label": "正面"},
    {"text": "产品质量很差，完全不值这个价。", "label": "负面"},
    {"text": "还可以，没有特别惊喜也没有失望。", "label": "中性"},
]

QA_EXAMPLES = [
    {"question": "中国的首都是哪里？", "answer": "北京"},
    {"question": "地球绕太阳一周需要多长时间？", "answer": "一年（约365.25天）"},
]


def get_template(name: str) -> Optional[PromptTemplate]:
    """
    按名称获取预定义的模板。

    Args:
        name: 模板名称

    Returns:
        PromptTemplate 实例，未找到返回 None
    """
    templates = {
        t.name: t for t in globals().values()
        if isinstance(t, PromptTemplate)
    }
    return templates.get(name)


def list_templates() -> List[str]:
    """
    列出所有可用的模板名称。

    Returns:
        模板名称列表
    """
    return [
        t.name for t in globals().values()
        if isinstance(t, PromptTemplate)
    ]