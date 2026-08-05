"""
向量存储模块

提供统一的向量存储接口，支持：
- FAISS 向量存储（高性能，适合大规模数据）
- Chroma 向量存储（轻量级，适合快速原型）
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """
    向量存储基类。

    所有向量存储实现都应继承此类。
    """

    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加向量到存储。

        Args:
            vectors: 向量矩阵，形状 (n, dim)
            metadata: 每个向量的元数据列表
            ids: 向量 ID 列表

        Returns:
            分配的向量 ID 列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """
        搜索最相似的向量。

        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果
            threshold: 相似度阈值

        Returns:
            列表 [(id, score, metadata), ...]
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存向量存储到磁盘。

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从磁盘加载向量存储。

        Args:
            path: 加载路径
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """返回存储的向量数量。"""
        pass


class FaissVectorStore(BaseVectorStore):
    """
    基于 FAISS 的向量存储。

    FAISS 提供高效的相似度搜索，支持内积和 L2 距离。

    Args:
        dimension: 向量维度
        metric: 距离度量（l2/ip/cosine）
    """

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self._index = None
        self._metadata: List[Optional[Dict]] = []
        self._ids: List[str] = []
        self._next_id = 0
        self._init_index()

    def _init_index(self) -> None:
        """初始化 FAISS 索引。"""
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装 faiss: pip install faiss-cpu")

        if self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "cosine":
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"未知的度量方式: {self.metric}")

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加向量到 FAISS 索引。"""
        import faiss

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        self._index.add(vectors.astype(np.float32))

        assigned_ids = []
        for i in range(len(vectors)):
            vid = ids[i] if ids else str(self._next_id)
            assigned_ids.append(vid)
            self._ids.append(vid)
            self._metadata.append(metadata[i] if metadata else None)
            self._next_id += 1

        return assigned_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """搜索最相似的向量。"""
        import faiss

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)

        actual_k = min(top_k, self._index.ntotal)
        if actual_k == 0:
            return []

        scores, indices = self._index.search(
            query_vector.astype(np.float32), actual_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if threshold is not None:
                if self.metric == "l2" and score > threshold:
                    continue
                elif self.metric in ("ip", "cosine") and score < threshold:
                    continue
            results.append((int(idx), float(score), self._metadata[int(idx)]))

        return results

    def save(self, path: str) -> None:
        """保存 FAISS 索引。"""
        import faiss
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self._index, path)

        meta_path = path + ".meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "ids": self._ids,
                "metadata": self._metadata,
                "next_id": self._next_id,
                "dimension": self.dimension,
                "metric": self.metric,
            }, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """加载 FAISS 索引。"""
        import faiss
        import json

        self._index = faiss.read_index(path)
        meta_path = path + ".meta.json"

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._ids = data["ids"]
                self._metadata = data["metadata"]
                self._next_id = data["next_id"]

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0


class ChromaVectorStore(BaseVectorStore):
    """
    基于 Chroma 的向量存储。

    Args:
        collection_name: 集合名称
        persist_directory: 持久化目录
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._metadata: List[Optional[Dict]] = []
        self._ids: List[str] = []
        self._init_client()

    def _init_client(self) -> None:
        """初始化 Chroma 客户端。"""
        try:
            import chromadb
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")

        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加向量到 Chroma 集合。"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n = len(vectors)
        assigned_ids = ids if ids else [str(len(self._ids) + i) for i in range(n)]
        metadatas = metadata if metadata else [{} for _ in range(n)]

        self._collection.add(
            ids=assigned_ids,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )

        self._ids.extend(assigned_ids)
        self._metadata.extend(metadatas)

        return assigned_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """搜索最相似的向量。"""
        if query_vector.ndim > 1:
            query_vector = query_vector[0]

        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, (vid, distance, meta) in enumerate(zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
            )):
                score = 1.0 - distance if distance is not None else 0.0
                if threshold is not None and score < threshold:
                    continue
                idx = self._ids.index(vid) if vid in self._ids else i
                output.append((idx, score, meta))

        return output

    def save(self, path: str) -> None:
        """Chroma 已自动持久化，此操作保留兼容接口。"""
        pass

    def load(self, path: str) -> None:
        """Chroma 已自动加载，此操作保留兼容接口。"""
        pass

    @property
    def size(self) -> int:
        return self._collection.count() if self._collection else 0


def create_vector_store(
    backend: str = "faiss",
    dimension: int = 768,
    **kwargs,
) -> BaseVectorStore:
    """
    工厂函数：创建向量存储实例。

    Args:
        backend: 后端类型（faiss/chroma）
        dimension: 向量维度
        **kwargs: 其他参数

    Returns:
        向量存储实例
    """
    if backend == "faiss":
        return FaissVectorStore(dimension=dimension, **kwargs)
    elif backend == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"未知的后端类型: {backend}")