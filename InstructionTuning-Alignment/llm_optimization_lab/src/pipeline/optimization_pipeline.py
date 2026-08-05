"""
完整优化 Pipeline 模块

提供一站式 LLM 优化流水线，整合：
1. 模型加载与配置
2. Prompt 工程优化
3. RAG 增强
4. 模型压缩（量化/剪枝/蒸馏）
5. LoRA 微调
6. 综合评估
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from ..utils.seed import set_seed
from ..utils.logger import setup_logger, get_timestamp
from ..utils.io_utils import save_json, ensure_dir

from ..models.model_factory import ModelFactory, ModelConfig, save_model
from ..models.adapters import (
    LoRAConfig, apply_lora, save_lora_weights, load_lora_weights
)

from ..prompt.templates import list_templates, get_template
from ..prompt.strategies import (
    BasePromptStrategy, ZeroShotStrategy, FewShotStrategy,
    ChainOfThoughtStrategy, TreeOfThoughtStrategy,
    SelfConsistencyStrategy, CompositeStrategy,
)
from ..prompt.evaluator import PromptEvaluator, PromptABTest

from ..rag.embeddings import create_embedder, BaseEmbedder
from ..rag.vector_store import create_vector_store, BaseVectorStore
from ..rag.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from ..rag.generator import RAGGenerator, MultiQueryGenerator

from ..compression.quantizer import create_quantizer, QuantizationConfig
from ..compression.pruner import create_pruner, PruningConfig
from ..compression.distiller import create_distillation_loss, DistillationConfig

from ..evaluation.evaluator import ComprehensiveEvaluator, EvaluationReport
from ..evaluation.metrics import PerformanceMetrics


@dataclass
class PipelineConfig:
    """
    Pipeline 配置数据类。

    整合所有子模块的配置参数。
    """
    experiment_name: str = "llm_optimization"
    seed: int = 42
    output_dir: str = "./outputs"

    model_config: Optional[Dict] = None
    lora_config: Optional[Dict] = None
    prompt_strategy: str = "zero_shot"
    rag_enabled: bool = False
    compression_config: Optional[Dict] = None
    evaluation_enabled: bool = True

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "model_config": self.model_config,
            "lora_config": self.lora_config,
            "prompt_strategy": self.prompt_strategy,
            "rag_enabled": self.rag_enabled,
            "compression_config": self.compression_config,
            "evaluation_enabled": self.evaluation_enabled,
        }


class OptimizationPipeline:
    """
    LLM 优化 Pipeline。

    完整的 LLM 优化流水线，支持从模型加载到评估的完整流程。

    Usage:
        pipeline = OptimizationPipeline(config)
        pipeline.run(train_data, eval_data)
        pipeline.save_results()
    """

    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
        self.config = config or PipelineConfig(**kwargs)
        self.logger = setup_logger(
            f"pipeline_{self.config.experiment_name}",
            log_file=f"{self.config.output_dir}/{self.config.experiment_name}_{get_timestamp()}.log",
        )

        set_seed(self.config.seed)

        self._model: Optional[nn.Module] = None
        self._tokenizer: Any = None
        self._prompt_strategy: Optional[BasePromptStrategy] = None
        self._rag_generator: Optional[RAGGenerator] = None
        self._embedder: Optional[BaseEmbedder] = None
        self._vector_store: Optional[BaseVectorStore] = None
        self._evaluator: Optional[ComprehensiveEvaluator] = None
        self._results: List[EvaluationReport] = []
        self._execution_log: List[Dict] = []

        self.logger.info(f"Pipeline 初始化完成: {self.config.experiment_name}")

    def run(
        self,
        train_data: Optional[List[Dict]] = None,
        eval_data: Optional[List[Dict]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整优化流程。

        Args:
            train_data: 训练数据
            eval_data: 评估数据
            query: 单条查询

        Returns:
            执行结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("开始执行优化 Pipeline")
        self.logger.info("=" * 60)

        start_time = time.time()
        result = {"experiment_name": self.config.experiment_name}

        try:
            self._step_load_model()
            self._step_setup_prompt_strategy()

            if self.config.rag_enabled:
                self._step_setup_rag(train_data)

            if self.config.compression_config:
                self._step_apply_compression()

            if self.config.lora_config and train_data:
                self._step_apply_lora()

            if query:
                result["response"] = self._step_generate(query)

            if eval_data and self.config.evaluation_enabled:
                eval_report = self._step_evaluate(eval_data)
                result["evaluation"] = eval_report.to_dict()
                self._results.append(eval_report)

            result["status"] = "success"

        except Exception as e:
            self.logger.error(f"Pipeline 执行失败: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        elapsed = time.time() - start_time
        result["elapsed_seconds"] = elapsed
        self._execution_log.append(result)

        self.logger.info(f"Pipeline 完成，耗时 {elapsed:.2f}s")
        return result

    def _step_load_model(self) -> None:
        """步骤 1: 加载模型。"""
        self.logger.info("[Step 1] 加载模型...")

        if self.config.model_config:
            model_config = ModelConfig.from_dict(self.config.model_config)
            self._model, self._tokenizer = ModelFactory.create_model(model_config)
            self.logger.info(f"模型加载完成: {model_config.model_name_or_path}")
        else:
            self.logger.warning("未配置模型，跳过加载步骤")

    def _step_setup_prompt_strategy(self) -> None:
        """步骤 2: 配置 Prompt 策略。"""
        self.logger.info("[Step 2] 配置 Prompt 策略...")

        strategy_name = self.config.prompt_strategy
        strategies = {
            "zero_shot": ZeroShotStrategy,
            "few_shot": FewShotStrategy,
            "cot": ChainOfThoughtStrategy,
            "tot": TreeOfThoughtStrategy,
            "self_consistency": SelfConsistencyStrategy,
        }

        strategy_class = strategies.get(strategy_name, ZeroShotStrategy)
        self._prompt_strategy = strategy_class()
        self.logger.info(f"Prompt 策略: {strategy_name}")

    def _step_setup_rag(self, documents: Optional[List[Dict]] = None) -> None:
        """步骤 3: 设置 RAG 系统。"""
        self.logger.info("[Step 3] 设置 RAG 系统...")

        self._embedder = create_embedder(backend="tfidf")

        self._vector_store = create_vector_store(
            backend="faiss", dimension=self._embedder.embedding_dim
        )

        docs_text = []
        if documents:
            for item in documents:
                if isinstance(item, dict):
                    docs_text.append(item.get("text", str(item)))
                else:
                    docs_text.append(str(item))

            if docs_text:
                embeddings = self._embedder.embed(docs_text)
                self._vector_store.add_vectors(
                    embeddings,
                    metadata=documents if isinstance(documents[0], dict) else None,
                )

        retriever = VectorRetriever(
            self._embedder, self._vector_store, docs_text
        )
        self._rag_generator = RAGGenerator(retriever=retriever)

        self.logger.info(f"RAG 系统设置完成，文档数: {len(docs_text)}")

    def _step_apply_compression(self) -> None:
        """步骤 4: 应用模型压缩。"""
        self.logger.info("[Step 4] 应用模型压缩...")

        if self._model is None or not self.config.compression_config:
            return

        comp_config = self.config.compression_config
        method = comp_config.get("method", "quantization")

        if method == "quantization":
            q_config = QuantizationConfig(
                bits=comp_config.get("bits", 8),
                method=comp_config.get("quant_method", "dynamic"),
            )
            quantizer = create_quantizer(q_config)
            self._model = quantizer.quantize(self._model)
            self.logger.info(f"模型量化完成: {q_config.bits}-bit {q_config.method}")

        elif method == "pruning":
            p_config = PruningConfig(
                method=comp_config.get("prune_method", "unstructured"),
                amount=comp_config.get("amount", 0.3),
            )
            pruner = create_pruner(p_config)
            self._model = pruner.prune(self._model, amount=p_config.amount)
            self.logger.info(f"模型剪枝完成: {p_config.amount * 100:.0f}%")

    def _step_apply_lora(self) -> None:
        """步骤 5: 应用 LoRA 适配器。"""
        self.logger.info("[Step 5] 应用 LoRA 适配器...")

        if self._model is None or not self.config.lora_config:
            return

        lora_config = LoRAConfig(**self.config.lora_config)
        self._model = apply_lora(self._model, lora_config)
        self.logger.info(f"LoRA 适配器应用完成: rank={lora_config.r}")

    def _step_generate(self, query: str) -> str:
        """步骤 6: 生成回答。"""
        self.logger.info(f"[Step 6] 生成回答: {query[:50]}...")

        if self._rag_generator:
            return self._rag_generator.generate(query)

        if self._prompt_strategy:
            prompt = self._prompt_strategy.build_prompt(question=query)
            return f"[生成回答] 基于 Prompt: {prompt[:100]}..."

        return f"[生成回答] {query}"

    def _step_evaluate(self, eval_data: List[Dict]) -> EvaluationReport:
        """步骤 7: 评估。"""
        self.logger.info("[Step 7] 执行评估...")

        self._evaluator = ComprehensiveEvaluator(
            model_name=self.config.experiment_name,
        )

        predictions = []
        references = []

        for item in eval_data:
            query = item.get("question", item.get("text", ""))
            reference = item.get("answer", item.get("label", ""))
            prediction = self._step_generate(query)
            predictions.append(prediction)
            references.append(reference)

        report = self._evaluator.evaluate(
            predictions=predictions,
            references=references,
            include_hallucination=True,
        )

        self.logger.info(f"评估完成: accuracy={report.metrics.get('accuracy', 0):.4f}")
        return report

    def optimize_prompt(
        self,
        query: str,
        reference: str,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        优化 Prompt 策略。

        Args:
            query: 查询文本
            reference: 参考答案
            strategies: 要测试的策略列表

        Returns:
            各策略的评估结果
        """
        if strategies is None:
            strategies = ["zero_shot", "few_shot", "cot"]

        evaluator = PromptEvaluator()
        results = {}

        for strategy_name in strategies:
            self._step_setup_prompt_strategy()
            prompt = self._prompt_strategy.build_prompt(question=query)
            result = self._step_generate(query)

            eval_result = evaluator.compute_accuracy(
                strategy_name, [reference], [result]
            )
            results[strategy_name] = {
                "prompt": prompt[:100],
                "accuracy": eval_result.value,
            }

        return results

    def save_results(self, path: Optional[str] = None) -> str:
        """
        保存所有结果到文件。

        Args:
            path: 保存路径

        Returns:
            保存的文件路径
        """
        save_dir = path or f"{self.config.output_dir}/{self.config.experiment_name}"
        ensure_dir(save_dir)

        results_data = {
            "config": self.config.to_dict(),
            "execution_log": self._execution_log,
            "evaluations": [r.to_dict() for r in self._results],
        }

        filepath = f"{save_dir}/results_{get_timestamp()}.json"
        save_json(results_data, filepath)
        self.logger.info(f"结果已保存: {filepath}")
        return filepath

    def save_model(self, path: Optional[str] = None) -> None:
        """保存模型。"""
        if self._model is None:
            self.logger.warning("没有可保存的模型")
            return

        save_dir = path or f"{self.config.output_dir}/{self.config.experiment_name}/model"
        if self.config.lora_config:
            save_lora_weights(self._model, save_dir)
        else:
            save_model(self._model, self._tokenizer, save_dir)

        self.logger.info(f"模型已保存: {save_dir}")

    def get_status(self) -> Dict[str, Any]:
        """获取 Pipeline 当前状态。"""
        return {
            "experiment_name": self.config.experiment_name,
            "model_loaded": self._model is not None,
            "prompt_strategy": str(self._prompt_strategy),
            "rag_enabled": self._rag_generator is not None,
            "lora_applied": any(
                hasattr(m, 'lora_A') for _, m in self._model.named_modules()
            ) if self._model else False,
            "num_evaluations": len(self._results),
            "execution_log_length": len(self._execution_log),
        }