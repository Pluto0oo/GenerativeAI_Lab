#!/usr/bin/env python3
"""
基础配置模块测试
"""
import os
import sys
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestConfiguration:
    """测试配置加载和验证"""
    
    def test_base_config_exists(self):
        """测试基础配置文件存在"""
        config_path = "configs/base.yaml"
        assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
    
    def test_sft_config_exists(self):
        """测试SFT配置文件存在"""
        config_path = "configs/experiment/sft_llama3.yaml"
        assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
    
    def test_dpo_config_exists(self):
        """测试DPO配置文件存在"""
        config_path = "configs/experiment/dpo_alignment.yaml"
        assert os.path.exists(config_path), f"配置文件不存在: {config_path}"
    
    def test_config_can_be_loaded(self):
        """测试配置文件可以被加载"""
        import yaml
        config_path = "configs/base.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert 'experiment' in config
        assert 'model' in config
        assert 'training' in config
    
    def test_experiment_id_auto_generation(self):
        """测试实验ID自动生成"""
        from src.utils.logger import get_timestamp
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS format

class TestDirectoryStructure:
    """测试目录结构完整性"""
    
    def test_configs_directory(self):
        """测试configs目录存在"""
        assert os.path.isdir("configs")
        assert os.path.isdir("configs/experiment")
    
    def test_data_directories(self):
        """测试数据目录存在"""
        assert os.path.isdir("data")
        assert os.path.isdir("data/raw")
        assert os.path.isdir("data/processed")
    
    def test_source_directories(self):
        """测试源码目录存在"""
        assert os.path.isdir("src")
        assert os.path.isdir("src/data")
        assert os.path.isdir("src/models")
        assert os.path.isdir("src/training")
        assert os.path.isdir("src/evaluation")
        assert os.path.isdir("src/utils")
        assert os.path.isdir("src/pipeline")
    
    def test_scripts_directory(self):
        """测试脚本目录存在"""
        assert os.path.isdir("scripts")
        assert os.path.exists("scripts/run_experiment.py")
        assert os.path.exists("scripts/run_comparison.py")
        assert os.path.exists("scripts/aggregate_results.py")
        assert os.path.exists("scripts/generate_report.py")
    
    def test_output_directories(self):
        """测试输出目录存在"""
        assert os.path.isdir("results")
        assert os.path.isdir("logs")
        assert os.path.isdir("reports")

class TestGPUAvailability:
    """测试GPU可用性"""
    
    def test_cuda_available(self):
        """测试CUDA是否可用"""
        try:
            import torch
            # 注意：实际实验需要GPU，但CI环境可能没有
            # 此测试仅记录状态，不强制要求通过
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                pytest.skip("CUDA not available - GPU tests skipped")
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_torch_version(self):
        """测试PyTorch版本"""
        try:
            import torch
            version = torch.__version__
            parts = version.split('.')
            major_version = int(parts[0])
            assert major_version >= 2, f"PyTorch 2.0+ required, got {version}"
        except ImportError:
            pytest.skip("PyTorch not installed")

class TestDependencies:
    """测试必要依赖是否安装"""
    
    def test_transformers_available(self):
        """测试transformers库"""
        try:
            import transformers
            assert transformers.__version__ >= '4.36.0'
        except ImportError:
            pytest.skip("transformers not installed")
    
    def test_peft_available(self):
        """测试peft库"""
        try:
            import peft
        except ImportError:
            pytest.skip("peft not installed")
    
    def test_trl_available(self):
        """测试trl库"""
        try:
            import trl
        except ImportError:
            pytest.skip("trl not installed")
    
    def test_yaml_available(self):
        """测试yaml库"""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
