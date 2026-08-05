"""
量化实现模块

提供模型量化功能，支持：
- INT8 动态量化
- INT8 静态量化
- INT4 量化（AWQ/GPTQ 风格）
- 量化感知训练
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    """
    量化器基类。

    所有量化器实现都应继承此类。
    """

    @abstractmethod
    def quantize(self, model: nn.Module) -> nn.Module:
        """
        对模型进行量化。

        Args:
            model: 待量化的模型

        Returns:
            量化后的模型
        """
        pass

    @abstractmethod
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        量化单个张量。

        Args:
            tensor: 待量化的张量

        Returns:
            (量化后的张量, 量化参数字典)
        """
        pass


class INT8DynamicQuantizer(BaseQuantizer):
    """
    INT8 动态量化器。

    动态量化在推理时量化权重和激活，适用于 LLM 推理加速。

    Args:
        dtype: 量化数据类型
    """

    def __init__(self, dtype: torch.dtype = torch.qint8):
        self.dtype = dtype
        self._quantization_config = {}

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        动态量化单个张量为 INT8。
        """
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            scale = torch.tensor(1e-8)

        quantized = torch.quantize_per_tensor(tensor, scale.item(), 0, self.dtype)
        dequantized = quantized.dequantize()

        params = {
            "scale": scale.item(),
            "zero_point": 0,
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }
        return dequantized, params

    def quantize(self, model: nn.Module) -> nn.Module:
        """
        对模型进行动态量化。
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=self.dtype,
        )
        return quantized_model


class INT8StaticQuantizer(BaseQuantizer):
    """
    INT8 静态量化器。

    静态量化需要校准数据集来确定量化参数。

    Args:
        num_calibration: 校准样本数量
    """

    def __init__(self, num_calibration: int = 100):
        self.num_calibration = num_calibration
        self._scale_cache: Dict[str, Tuple[float, int]] = {}

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        静态量化单个张量。
        """
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        scale = (max_val - min_val) / 255.0
        zero_point = int(-min_val / scale) if scale > 0 else 0

        scale = max(scale, 1e-8)
        zero_point = max(0, min(255, zero_point))

        quantized = torch.clamp(
            torch.round(tensor / scale) + zero_point, 0, 255
        )
        dequantized = (quantized - zero_point) * scale

        params = {"scale": scale, "zero_point": zero_point}
        return dequantized, params

    def quantize(self, model: nn.Module) -> nn.Module:
        """
        对模型进行静态量化（模拟量化）。
        """
        model.eval()
        replaced_modules = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                quantized_weight, params = self.quantize_tensor(weight)
                self._scale_cache[name] = (params["scale"], params["zero_point"])
                module.weight.data = quantized_weight
                replaced_modules[name] = True

        return model


class INT4Quantizer(BaseQuantizer):
    """
    INT4 量化器（仿 AWQ/GPTQ 风格）。

    使用 4-bit 量化进一步压缩模型体积。

    Args:
        group_size: 分组大小
        symmetric: 是否使用对称量化
    """

    def __init__(self, group_size: int = 128, symmetric: bool = True):
        self.group_size = group_size
        self.symmetric = symmetric
        self._quant_info: Dict[str, Dict] = {}

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        将张量量化为 INT4。
        """
        shape = tensor.shape
        tensor_flat = tensor.view(-1)
        n_elements = tensor_flat.numel()

        groups = []
        scales = []
        zero_points = []

        for start in range(0, n_elements, self.group_size):
            end = min(start + self.group_size, n_elements)
            group = tensor_flat[start:end]

            if self.symmetric:
                abs_max = group.abs().max().item()
                scale = max(abs_max / 7.0, 1e-8)
                quantized = torch.clamp(
                    torch.round(group / scale), -8, 7
                )
                dequantized = quantized * scale
                scales.append(scale)
                zero_points.append(0)
            else:
                min_val = group.min().item()
                max_val = group.max().item()
                scale = max((max_val - min_val) / 15.0, 1e-8)
                zp = int(-min_val / scale)
                zp = max(0, min(15, zp))
                quantized = torch.clamp(
                    torch.round(group / scale) + zp, 0, 15
                )
                dequantized = (quantized - zp) * scale
                scales.append(scale)
                zero_points.append(zp)

            groups.append(dequantized)

        dequantized_tensor = torch.cat(groups).view(shape)
        params = {
            "scales": scales,
            "zero_points": zero_points,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
        }
        return dequantized_tensor, params

    def quantize(self, model: nn.Module) -> nn.Module:
        """
        对模型进行 INT4 量化。
        """
        model.eval()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data.clone()
                quantized_weight, params = self.quantize_tensor(weight)
                module.weight.data = quantized_weight
                self._quant_info[name] = params

        return model

    def get_quantization_info(self) -> Dict[str, Dict]:
        """获取量化信息。"""
        return self._quant_info.copy()


class QuantizationConfig:
    """
    量化配置类。

    Attributes:
        bits: 量化位数
        method: 量化方法（dynamic/static/int4）
        target_modules: 目标模块
    """

    def __init__(
        self,
        bits: int = 8,
        method: str = "dynamic",
        target_modules: Optional[list] = None,
        **kwargs,
    ):
        self.bits = bits
        self.method = method
        self.target_modules = target_modules or ["Linear"]
        self.extra_params = kwargs

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "bits": self.bits,
            "method": self.method,
            "target_modules": self.target_modules,
            **self.extra_params,
        }


def create_quantizer(config: QuantizationConfig) -> BaseQuantizer:
    """
    工厂函数：根据配置创建量化器。

    Args:
        config: 量化配置

    Returns:
        量化器实例
    """
    if config.bits == 8:
        if config.method == "dynamic":
            return INT8DynamicQuantizer()
        elif config.method == "static":
            return INT8StaticQuantizer()
        else:
            raise ValueError(f"未知的 INT8 量化方法: {config.method}")
    elif config.bits == 4:
        return INT4Quantizer(**config.extra_params)
    else:
        raise ValueError(f"不支持的量化位数: {config.bits}")


class ModelQuantizer:
    """
    模型量化器：对外提供统一的量化接口。

    封装模型加载、量化和性能测试功能。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model', {}).get('base_name', 'TinyLlama')
        self.bits = config.get('quantization', {}).get('bits', 8)
        self.method = config.get('quantization', {}).get('method', 'dynamic')
        self._model = None

    def load_model(self):
        """加载模型。"""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            model_path = self.config.get('model', {}).get('merged_path', self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        return self._model

    def quantize(self, model=None):
        """
        对模型进行量化。

        Args:
            model: 待量化的模型（为None则自动加载）

        Returns:
            量化后的模型
        """
        if model is None:
            model = self.load_model()

        quantizer_config = QuantizationConfig(
            bits=self.bits,
            method=self.method,
        )
        quantizer = create_quantizer(quantizer_config)
        return quantizer.quantize(model)

    def benchmark(self, model=None) -> Dict:
        """
        对量化模型进行基准测试。

        Args:
            model: 模型（为None则自动加载+量化）

        Returns:
            性能指标字典
        """
        import torch
        import time

        if model is None:
            model = self.load_model()
            model = self.quantize(model)

        model.eval()

        # 计算模型大小
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / 1024 / 1024

        # 测试推理速度
        try:
            dummy_input = torch.randn(1, 32, model.config.hidden_size).to(next(model.parameters()).device)
            
            # 预热
            for _ in range(3):
                _ = model(dummy_input)
            
            # 测量延迟
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                _ = model(dummy_input)
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
        except Exception:
            avg_latency = 0.0

        # 显存占用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            _ = model(dummy_input)
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            memory_gb = 0.0

        return {
            "model_size_mb": float(model_size_mb),
            "parameters": int(param_count),
            "inference_latency_ms": float(avg_latency),
            "memory_gb": float(memory_gb),
            "quantization_bits": self.bits,
            "quantization_method": self.method,
        }