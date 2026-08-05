"""
幻觉检测模块

提供 LLM 生成文本中的幻觉检测功能：
- 基于证据的检测（RAG 对齐）
- 基于语言统计的检测
- 基于一致性的检测
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter


class HallucinationDetector:
    """
    幻觉检测器基类。

    综合多种方法检测 LLM 生成文本中的幻觉。
    """

    def __init__(self):
        self._detection_results: List[Dict] = []

    def detect(
        self,
        generated_text: str,
        reference_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """
        执行幻觉检测。

        Args:
            generated_text: 生成的文本
            reference_texts: 参考文本列表
            **kwargs: 其他参数

        Returns:
            检测结果字典
        """
        results = {
            "text_length": len(generated_text),
            "contradiction_score": 0.0,
            "hallucination_score": 0.0,
            "unverified_claims": [],
            "confidence": 0.0,
        }

        if reference_texts:
            evidence_result = self._check_evidence_alignment(
                generated_text, reference_texts
            )
            results.update(evidence_result)

        repetition_result = self._check_repetition(generated_text)
        results["repetition_score"] = repetition_result["score"]

        claim_result = self._extract_and_verify_claims(
            generated_text, reference_texts
        )
        results["unverified_claims"] = claim_result["unverified"]
        results["hallucination_score"] = self._compute_hallucination_score(results)
        results["confidence"] = 1.0 - results["hallucination_score"]

        self._detection_results.append(results)
        return results

    def _check_evidence_alignment(
        self,
        generated_text: str,
        reference_texts: List[str],
    ) -> Dict:
        """
        检查生成文本与参考文本的对齐程度。
        """
        ref_combined = " ".join(reference_texts).lower()
        gen_lower = generated_text.lower()

        gen_words = set(gen_lower.split())
        ref_words = set(ref_combined.split())

        if not gen_words:
            return {"evidence_coverage": 0.0, "contradiction_score": 0.0}

        covered = gen_words & ref_words
        coverage = len(covered) / len(gen_words) if gen_words else 0

        contradictions = 0
        negation_words = {"不是", "不对", "错误", "相反", "不", "没", "无", "非",
                          "not", "is not", "never", "no", "false"}

        gen_sentences = re.split(r'[。.！？?!\n]', generated_text)
        for sentence in gen_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            has_negation = any(w in sentence.lower() for w in negation_words)
            if has_negation:
                for ref_text in reference_texts:
                    ref_lower = ref_text.lower()
                    if any(w in ref_lower for w in negation_words[:5]):
                        contradictions += 1
                        break

        contradiction_score = min(1.0, contradictions / max(len(gen_sentences), 1))

        return {
            "evidence_coverage": coverage,
            "contradiction_score": contradiction_score,
        }

    def _check_repetition(self, text: str) -> Dict:
        """
        检查文本重复度。
        """
        sentences = re.split(r'[。.！？?!\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return {"score": 0.0, "unique_ratio": 1.0}

        unique_sentences = set(sentences)
        unique_ratio = len(unique_sentences) / len(sentences)

        score = 1.0 - unique_ratio
        return {"score": score, "unique_ratio": unique_ratio}

    def _extract_and_verify_claims(
        self,
        generated_text: str,
        reference_texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        提取声明并验证。
        """
        claims = self._extract_claims(generated_text)
        unverified = []

        if reference_texts:
            for claim in claims:
                if not self._verify_claim(claim, reference_texts):
                    unverified.append(claim)

        return {"unverified": unverified, "total_claims": len(claims)}

    def _extract_claims(self, text: str) -> List[str]:
        """
        从文本中提取声明。
        """
        sentences = re.split(r'[。.！？?!\n]', text)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5 and not sentence.endswith('?') and not sentence.endswith('？'):
                claims.append(sentence)
        return claims

    def _verify_claim(
        self,
        claim: str,
        reference_texts: List[str],
    ) -> bool:
        """
        验证单个声明。
        """
        claim_words = set(claim.lower().split())
        if not claim_words:
            return True

        max_overlap = 0.0
        for ref in reference_texts:
            ref_words = set(ref.lower().split())
            overlap = len(claim_words & ref_words) / len(claim_words)
            max_overlap = max(max_overlap, overlap)

        return max_overlap > 0.3

    def _compute_hallucination_score(self, results: Dict) -> float:
        """
        计算综合幻觉分数。
        """
        scores = []

        if "contradiction_score" in results:
            scores.append(results["contradiction_score"] * 0.4)

        if "evidence_coverage" in results:
            scores.append((1.0 - results["evidence_coverage"]) * 0.3)

        if "repetition_score" in results:
            scores.append(results["repetition_score"] * 0.1)

        if "unverified_claims" in results:
            total = results.get("total_claims", 1)
            unverified_ratio = len(results["unverified_claims"]) / max(total, 1)
            scores.append(unverified_ratio * 0.2)

        if not scores:
            return 0.0

        return min(1.0, sum(scores))


class LLMSelfConsistencyDetector(HallucinationDetector):
    """
    基于 LLM 自洽性的幻觉检测器。

    通过多次采样生成，检查答案的一致性。
    """

    def detect_with_consistency(
        self,
        outputs: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> Dict:
        """
        基于多次采样结果检测幻觉。

        Args:
            outputs: 多次采样的输出列表
            reference_texts: 参考文本

        Returns:
            检测结果
        """
        if len(outputs) < 2:
            return self.detect(outputs[0] if outputs else "", reference_texts)

        from collections import Counter

        lengths = [len(o.split()) for o in outputs]
        length_consistency = 1.0 - float(np.std(lengths) / max(np.mean(lengths), 1))

        all_claims = []
        for output in outputs:
            claims = self._extract_claims(output)
            all_claims.extend(claims)

        claim_counter = Counter(all_claims)
        total_claims = len(all_claims)
        unique_claims = len(claim_counter)
        claim_diversity = unique_claims / max(total_claims, 1)

        consistency_score = 1.0 - claim_diversity * 0.5 - (1.0 - length_consistency) * 0.5

        base_result = self.detect(outputs[0], reference_texts)
        base_result["consistency_score"] = consistency_score
        base_result["hallucination_score"] = min(
            1.0,
            (base_result.get("hallucination_score", 0) + (1.0 - consistency_score)) / 2,
        )
        base_result["confidence"] = 1.0 - base_result["hallucination_score"]

        return base_result


def compute_hallucination_rate(
    detector: HallucinationDetector,
    generated_texts: List[str],
    reference_texts: Optional[List[str]] = None,
) -> float:
    """
    计算一组文本的幻觉率。

    Args:
        detector: 幻觉检测器
        generated_texts: 生成文本列表
        reference_texts: 参考文本列表

    Returns:
        平均幻觉率 (0-1)
    """
    scores = []
    for text in generated_texts:
        result = detector.detect(text, reference_texts)
        scores.append(result.get("hallucination_score", 0.0))
    return float(np.mean(scores))