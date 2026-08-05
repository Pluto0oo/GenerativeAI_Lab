"""
嵌入模型封装模块

提供统一的嵌入模型接口，支持多种嵌入模型后端：
- HuggingFace 本地模型
- Sentence-Transformers
- 本地 TF-IDF 嵌入（备选方案）
"""

import numpy as np
from typing import List, Optional, Union
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    嵌入模型基类。

    所有嵌入模型实现都应继承此类。
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为嵌入向量。

        Args:
            texts: 文本列表

        Returns:
            嵌入矩阵，形状为 (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """
        将单条文本转换为嵌入向量。

        Args:
            text: 输入文本

        Returns:
            嵌入向量，形状为 (embedding_dim,)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """返回嵌入维度。"""
        pass


class HFEmbedder(BaseEmbedder):
    """
    基于 HuggingFace Transformers 的嵌入模型。

    使用 AutoModel 和 AutoTokenizer 加载模型。

    Args:
        model_name: 模型名称或路径
        max_length: 最大 token 长度
        device: 运行设备（cpu/cuda）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        max_length: int = 512,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tokenizer = None
        self._embedding_dim = 0
        self._load_model()

    def _load_model(self) -> None:
        """加载嵌入模型和分词器。"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            self._embedding_dim = self._model.config.hidden_size
        except ImportError:
            raise ImportError("请安装 transformers 库: pip install transformers")

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文本。"""
        import torch

        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """嵌入单条文本。"""
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    基于 Sentence-Transformers 的嵌入模型。

    Args:
        model_name: 模型名称
        device: 运行设备
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._embedding_dim = 0
        self._load_model()

    def _load_model(self) -> None:
        """加载 Sentence-Transformers 模型。"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文本。"""
        return self._model.encode(texts, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """嵌入单条文本。"""
        return self._model.encode(text, show_progress_bar=False)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class TfidfEmbedder(BaseEmbedder):
    """
    基于 TF-IDF 的轻量级嵌入（备选方案，无需深度学习模型）。

    Args:
        max_features: 最大特征数
    """

    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self._vectorizer = None
        self._embedding_dim = max_features
        self._vocab_built = False

    def _ensure_vocab(self, texts: List[str]) -> None:
        """确保 TF-IDF 词汇表已构建。"""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=self.max_features)
            self._vectorizer.fit(texts)
            self._embedding_dim = len(self._vectorizer.vocabulary_)
            self._vocab_built = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文本。"""
        self._ensure_vocab(texts)
        return self._vectorizer.transform(texts).toarray()

    def embed_single(self, text: str) -> np.ndarray:
        """嵌入单条文本。"""
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


def create_embedder(
    backend: str = "hf",
    model_name: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
) -> BaseEmbedder:
    """
    工厂函数：创建嵌入模型实例。

    Args:
        backend: 后端类型（hf/sentence_transformers/tfidf）
        model_name: 模型名称
        device: 设备
        **kwargs: 其他参数

    Returns:
        嵌入模型实例
    """
    if backend == "hf":
        return HFEmbedder(model_name=model_name or "BAAI/bge-small-zh-v1.5",
                          device=device, **kwargs)
    elif backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=model_name or "all-MiniLM-L6-v2", device=device
        )
    elif backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    else:
        raise ValueError(f"未知的后端类型: {backend}")