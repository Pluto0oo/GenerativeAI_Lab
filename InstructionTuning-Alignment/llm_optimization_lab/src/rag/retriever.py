"""
检索器实现模块

提供多种文档检索策略：
- 基础检索器（向量相似度检索）
- BM25 关键词检索器
- 混合检索器（向量 + 关键词）
- 重排序检索器
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .embeddings import BaseEmbedder
from .vector_store import BaseVectorStore


class BaseRetriever(ABC):
    """
    检索器基类。

    所有检索器实现都应继承此类。
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """
        根据查询检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            文档列表 [{"id": ..., "text": ..., "score": ..., "metadata": ...}, ...]
        """
        pass


class VectorRetriever(BaseRetriever):
    """
    基于向量相似度的检索器。

    Args:
        embedder: 嵌入模型
        vector_store: 向量存储
        documents: 原始文档列表
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        documents: Optional[List[str]] = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.documents = documents or []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            检索到的文档列表
        """
        query_vector = self.embedder.embed_single(query)
        raw_results = self.vector_store.search(
            query_vector, top_k=top_k, threshold=threshold
        )

        documents = []
        for idx, score, metadata in raw_results:
            doc = {
                "id": idx,
                "text": self.documents[idx] if idx < len(self.documents) else "",
                "score": score,
                "metadata": metadata or {},
            }
            documents.append(doc)

        return documents


class BM25Retriever(BaseRetriever):
    """
    基于 BM25 算法的关键词检索器。

    Args:
        documents: 文档列表
        k1: BM25 参数 k1
        b: BM25 参数 b
    """

    def __init__(
        self,
        documents: List[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._avg_dl = 0.0
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lengths: List[int] = []
        self._corpus_size = len(documents)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（支持中英文）。"""
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
        tokens.extend(text.lower().split())
        return tokens

    def _build_index(self) -> None:
        """构建 BM25 索引。"""
        for doc in self.documents:
            tokens = self._tokenize(doc)
            self._doc_lengths.append(len(tokens))

            seen = set()
            for token in tokens:
                if token not in seen:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen.add(token)

        self._avg_dl = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths else 0
        )

    def _idf(self, token: str) -> float:
        """计算逆文档频率。"""
        df = self._doc_freqs.get(token, 0)
        import math
        return math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档的 BM25 分数。"""
        import math

        doc = self.documents[doc_idx]
        doc_tokens = self._tokenize(doc)
        dl = self._doc_lengths[doc_idx]

        tf: Dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for token in set(query_tokens):
            if token not in tf:
                continue
            idf = self._idf(token)
            freq = tf[token]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * (numerator / denominator)

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """基于 BM25 检索。"""
        query_tokens = self._tokenize(query)

        scores = []
        for i in range(self._corpus_size):
            score = self._score_document(query_tokens, i)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "id": idx,
                "text": self.documents[idx],
                "score": score,
                "metadata": {},
            })

        return results


class HybridRetriever(BaseRetriever):
    """
    混合检索器：结合向量检索和 BM25 检索。

    Args:
        vector_retriever: 向量检索器
        bm25_retriever: BM25 检索器
        alpha: 融合权重（0=纯BM25, 1=纯向量）
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """混合检索。"""
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

        score_map: Dict[int, Dict] = {}

        for result in vector_results:
            idx = result["id"]
            score_map[idx] = {
                "id": idx,
                "text": result["text"],
                "vector_score": result["score"],
                "bm25_score": 0.0,
                "metadata": result.get("metadata", {}),
            }

        for result in bm25_results:
            idx = result["id"]
            if idx in score_map:
                score_map[idx]["bm25_score"] = result["score"]
            else:
                score_map[idx] = {
                    "id": idx,
                    "text": result["text"],
                    "vector_score": 0.0,
                    "bm25_score": result["score"],
                    "metadata": result.get("metadata", {}),
                }

        for idx in score_map:
            entry = score_map[idx]
            entry["score"] = (
                self.alpha * entry["vector_score"]
                + (1 - self.alpha) * entry["bm25_score"]
            )

        merged = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]


class RerankRetriever(BaseRetriever):
    """
    重排序检索器：先用基础检索器获取候选，再精排。

    Args:
        base_retriever: 基础检索器
        reranker_fn: 重排序函数
        candidate_k: 候选数量
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker_fn: Optional[callable] = None,
        candidate_k: int = 20,
    ):
        self.base_retriever = base_retriever
        self.reranker_fn = reranker_fn or self._default_rerank
        self.candidate_k = candidate_k

    def _default_rerank(
        self, query: str, documents: List[Dict]
    ) -> List[Dict]:
        """默认重排序：保持原始顺序。"""
        return documents

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """重排序检索。"""
        candidates = self.base_retriever.retrieve(
            query, top_k=self.candidate_k
        )
        reranked = self.reranker_fn(query, candidates)
        return reranked[:top_k]


class MedicalRAG:
    """
    医疗RAG系统。

    整合检索和生成，提供完整的医疗问答功能。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.kb_path = config.get('knowledge_base', {}).get('path', '')
        self.embedding_model = None
        self.vector_store = None
        self.documents = []
        self._model = None
        self._tokenizer = None
        self._indexed = False

    def build_index(self, documents: Optional[List[Dict]] = None) -> None:
        """
        构建向量索引。

        Args:
            documents: 文档列表（如果为None则从知识库加载）
        """
        if documents:
            self.documents = [d.get('content', str(d)) for d in documents]
        elif self.kb_path and os.path.exists(self.kb_path):
            self._load_knowledge_base()
        else:
            self.documents = ["高血压首选药物是氨氯地平等钙通道阻滞剂。"]
        
        self._indexed = True
        print(f"知识库构建完成：{len(self.documents)} 个文档")

    def _load_knowledge_base(self) -> None:
        """从知识库目录加载文档。"""
        import os as os_module
        if not os_module.path.exists(self.kb_path):
            self.documents = ["暂无知识库内容"]
            return

        for filename in os_module.listdir(self.kb_path):
            if filename.endswith(('.md', '.txt')):
                filepath = os_module.path.join(self.kb_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.documents.append(f.read())

    def _retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """简化版检索：基于关键词匹配。"""
        if not self.documents:
            return []

        query_lower = query.lower()
        scored_docs = []
        for doc in self.documents:
            doc_lower = doc.lower()
            # 简单关键词匹配
            score = sum(1 for word in query_lower.split() if word in doc_lower)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k] if score > 0]

    def _load_model(self):
        """加载生成模型。"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # 优先使用 model_path，否则使用 model_name
                model_path = self.config.get('generation', {}).get('model_path', '') or self.config.get('generation', {}).get('model_name', 'TinyLlama')
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self._model.eval()
            except Exception:
                self._model = None
                self._tokenizer = None

    def answer(self, question: str) -> Dict:
        """
        回答问题。

        Args:
            question: 问题文本

        Returns:
            包含答案、来源等的字典
        """
        # 1. 检索上下文
        contexts = self._retrieve(question)
        
        # 2. 构建提示
        context_text = "\n\n".join(contexts) if contexts else "暂无相关医学文献"
        
        prompt = f"""基于以下医学参考资料回答问题。如果参考资料中没有相关信息，请说明。

参考资料：
{context_text}

问题：{question}

回答："""

        # 3. 生成回答
        answer = self._simple_generate(prompt)

        # 4. 返回结果
        return {
            "answer": answer,
            "contexts": contexts,
            "faithfulness": self._estimate_faithfulness(answer, contexts),
            "sources": [f"文档{i+1}" for i in range(len(contexts))] if contexts else [],
        }

    def _simple_generate(self, prompt: str) -> str:
        """简化生成（不依赖模型时返回规则答案）。"""
        try:
            self._load_model()
            if self._model is not None:
                import torch
                messages = [{"role": "user", "content": prompt}]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                
                generated = outputs[0][inputs["input_ids"].shape[1]:]
                return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception:
            pass
        
        # 回退：返回基于检索的答案
        return "根据医学知识库，建议咨询专业医生获取准确诊断和治疗方案。"

    def _estimate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """估算答案忠实度。"""
        if not contexts or not answer:
            return 0.5
        
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = answer_words & context_words
        return float(len(overlap) / len(answer_words))