import os
import time
import logging
from datetime import datetime
import torch
import matplotlib.pyplot as plt

from src.utils.seed import seed_everything
from src.utils.io_utils import (
    save_config, save_metrics_csv, save_metrics_json,
    generate_summary_md, create_exp_dirs, load_config
)
from src.utils.logger import get_timestamp
from src.data.dataset import InstructionDataset, PreferenceDataset
from src.models.model_factory import ModelFactory
from src.training.sft_trainer import SFTTrainerWrapper
from src.training.dpo_trainer import DPOTrainerWrapper
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class AlignmentPipeline:
    """完整的指令微调与对齐Pipeline"""
    
    def __init__(self, config_path: str, exp_id: str = None):
        """初始化Pipeline
        
        Args:
            config_path: 配置文件路径
            exp_id: 实验ID，如果为None则自动生成
        """
        self.config = load_config(config_path)
        self.exp_id = exp_id or f"exp_{get_timestamp()}"
        self.config['exp_id'] = self.exp_id
        
        # 设置输出目录
        base_output_dir = self.config['output']['save_dir']
        self.exp_paths = create_exp_dirs(base_output_dir, self.exp_id)
        
        # 初始化组件
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.training_history = []
        self.final_metrics = {}
        
        # 验证GPU
        self._check_gpu()
        
        # 设置随机种子
        seed = self.config['experiment']['seed']
        seed_everything(seed)
        
        logger.info(f"Pipeline initialized for experiment: {self.exp_id}")
        logger.info(f"Output directory: {self.exp_paths['root']}")
    
    def _check_gpu(self):
        """检查GPU可用性"""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available! This project requires GPU. "
                "Please ensure you have a CUDA-capable GPU and proper drivers installed."
            )
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {cuda_version}")
    
    def prepare_data(self):
        """准备数据集"""
        training_method = self.config['training']['method']
        
        logger.info(f"Preparing data for training method: {training_method}")
        
        if training_method == 'dpo':
            # DPO使用偏好数据集
            dataset_handler = PreferenceDataset(self.config)
            dataset_handler.load()
            self.train_dataset, self.eval_dataset = dataset_handler.get_datasets()
        else:
            # SFT使用指令数据集
            dataset_handler = InstructionDataset(self.config)
            dataset_handler.load()
            self.train_dataset, self.eval_dataset = dataset_handler.get_datasets()
        
        logger.info(f"Data prepared. Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
        return self
    
    def prepare_model(self):
        """准备模型"""
        logger.info("Loading model...")
        self.model, self.tokenizer = ModelFactory.create(self.config)
        
        # DPO需要参考模型
        if self.config['training']['method'] == 'dpo':
            logger.info("Loading reference model for DPO...")
            self.ref_model = ModelFactory.get_ref_model(
                self.config['model']['name'],
                torch_dtype=torch.bfloat16
            )
        
        logger.info("Model preparation complete.")
        return self
    
    def train(self):
        """执行训练"""
        training_method = self.config['training']['method']
        training_start = time.time()
        
        logger.info(f"Starting {training_method.upper()} training...")
        
        # 创建trainer
        if training_method == 'sft':
            self.trainer = SFTTrainerWrapper(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                config=self.config,
                output_dir=self.exp_paths['root']
            )
        elif training_method == 'dpo':
            self.trainer = DPOTrainerWrapper(
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                config=self.config,
                output_dir=self.exp_paths['root']
            )
        else:
            raise ValueError(f"Unknown training method: {training_method}")
        
        # 运行训练
        self.trainer.setup()
        train_metrics = self.trainer.train()
        
        # 获取训练历史
        self.training_history = self.trainer.get_history()
        
        # 计算训练时间
        training_time = time.time() - training_start
        train_metrics['training_time_seconds'] = training_time
        
        logger.info(f"Training completed in {training_time:.1f}s")
        return train_metrics
    
    def evaluate(self):
        """执行评估（使用Trainer内置评估 + 简单生成测试）"""
        logger.info("Running final evaluation...")

        # 1. 使用Trainer内置评估
        eval_results = {}
        if self.trainer is not None:
            try:
                eval_results = self.trainer.evaluate() or {}
                logger.info(f"Trainer evaluation: {eval_results}")
            except Exception as e:
                logger.warning(f"Trainer evaluation failed: {e}")

        # 2. 简单的指令跟随测试（仅生成少量样本）
        instruction_test = {"status": "skipped", "samples": []}
        try:
            evaluator = ModelEvaluator(self.model, self.tokenizer, self.config)
            test_prompts = ["什么是机器学习？", "用Python写一个hello world"]
            predictions = evaluator.generate(test_prompts, max_new_tokens=64)
            instruction_test = {
                "status": "completed",
                "samples": [{"prompt": p, "response": r} for p, r in zip(test_prompts, predictions)]
            }
        except Exception as e:
            logger.warning(f"Instruction following test failed: {e}")
            instruction_test = {"status": "failed", "error": str(e), "samples": []}

        # 提取指标
        eval_loss = eval_results.get('eval_loss', 0)
        train_loss = eval_results.get('eval_loss', 0)
        # 从训练历史中提取最终train_loss
        for log in reversed(self.training_history):
            if 'loss' in log:
                train_loss = log['loss']
                break
            elif 'train_loss' in log:
                train_loss = log['train_loss']
                break

        self.final_metrics = {
            'experiment_id': self.exp_id,
            'completed_at': datetime.now().isoformat(),
            'training_method': self.config['training']['method'],
            'metrics': {
                'train_loss': float(train_loss) if train_loss else 0,
                'eval_loss': float(eval_loss) if eval_loss else 0,
                'accuracy': 0.0,  # 简化，实际需要专门评估
                'bleu': 0.0,
            },
            'model_info': {
                'base_model': self.config['model']['name'],
                'training_method': self.config['training']['method'],
            },
            'hardware': {
                'gpu_model': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
            },
            'instruction_test': instruction_test,
        }

        logger.info("Evaluation complete.")
        return self.final_metrics
    
    def _generate_plots(self):
        """生成训练过程图表"""
        plots_dir = self.exp_paths['plots']
        if not self.training_history:
            return
            
        # 提取指标
        steps = []
        train_losses = []
        eval_losses = []
        
        for log in self.training_history:
            step = log.get('step', len(steps))
            steps.append(step)
            
            if 'loss' in log:
                train_losses.append(log['loss'])
            elif 'train_loss' in log:
                train_losses.append(log['train_loss'])
            
            if 'eval_loss' in log:
                eval_losses.append((step, log['eval_loss']))
        
        # 绘制Loss曲线
        if train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(steps[:len(train_losses)], train_losses, 'b-', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Training Loss')
            plt.title(f'Training Loss Curve - {self.exp_id}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=150)
            plt.close()
        
        # 绘制Eval Loss曲线
        if eval_losses:
            eval_steps, eval_values = zip(*eval_losses)
            plt.figure(figsize=(10, 6))
            plt.plot(eval_steps, eval_values, 'r-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Step')
            plt.ylabel('Validation Loss')
            plt.title(f'Validation Loss Curve - {self.exp_id}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'eval_loss_curve.png'), dpi=150)
            plt.close()
    
    def save_results(self):
        """保存所有结果"""
        logger.info("Saving experiment results...")
        
        # 1. 保存实际使用的配置
        save_config(self.config, os.path.join(self.exp_paths['root'], 'config_used.yaml'))
        
        # 2. 保存训练历史(CSV)
        if self.training_history:
            save_metrics_csv(self.training_history, os.path.join(self.exp_paths['root'], 'metrics.csv'))
        
        # 3. 保存最终指标(JSON)
        save_metrics_json(self.final_metrics, os.path.join(self.exp_paths['root'], 'metrics.json'))
        
        # 4. 生成摘要(Markdown)
        generate_summary_md(
            self.final_metrics,
            self.config,
            os.path.join(self.exp_paths['root'], 'summary.md')
        )
        
        # 5. 生成图表
        self._generate_plots()
        
        logger.info(f"Results saved to {self.exp_paths['root']}")
        return self
    
    def run_full_pipeline(self):
        """运行完整Pipeline"""
        logger.info(f"{'='*60}")
        logger.info(f"Starting experiment: {self.exp_id}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 1. 准备数据
            self.prepare_data()
            
            # 2. 准备模型
            self.prepare_model()
            
            # 3. 训练
            train_metrics = self.train()
            
            # 4. 评估
            eval_metrics = self.evaluate()
            
            # 5. 保存结果
            self.save_results()
            
            total_time = time.time() - start_time
            logger.info(f"{'='*60}")
            logger.info(f"Experiment {self.exp_id} completed successfully!")
            logger.info(f"Total time: {total_time:.1f}s")
            logger.info(f"Results saved to: {self.exp_paths['root']}")
            logger.info(f"{'='*60}")
            
            return self.final_metrics
            
        except Exception as e:
            logger.error(f"Experiment {self.exp_id} failed: {e}")
            raise
    
    def run_with_repeats(self, repeat_times: int = None):
        """多次重复实验"""
        if repeat_times is None:
            repeat_times = self.config['experiment']['repeat_times']
        
        logger.info(f"Running {repeat_times} repetitions for experiment {self.exp_id}")
        
        all_metrics = []
        original_seed = self.config['experiment']['seed']
        
        for i in range(repeat_times):
            logger.info(f"\n{'*'*40}")
            logger.info(f"Repeat {i+1}/{repeat_times}")
            logger.info(f"{'*'*40}")
            
            # 为每次重复设置不同的随机种子
            self.config['experiment']['seed'] = original_seed + i * 1000
            seed_everything(self.config['experiment']['seed'])
            
            # 创建重复实验目录
            repeat_dir = os.path.join(self.exp_paths['repeats'], f'repeat_{i:03d}')
            os.makedirs(repeat_dir, exist_ok=True)
            
            # 重置状态
            self.model = None
            self.tokenizer = None
            self.ref_model = None
            self.train_dataset = None
            self.eval_dataset = None
            self.trainer = None
            self.training_history = []
            
            # 运行单次实验（简化版）
            self.prepare_data()
            self.prepare_model()
            train_metrics = self.train()
            
            # 保存该次重复的结果
            save_config(self.config, os.path.join(repeat_dir, 'config_used.yaml'))
            if self.training_history:
                save_metrics_csv(self.training_history, os.path.join(repeat_dir, 'metrics.csv'))
            
            all_metrics.append(train_metrics)
        
        # 恢复原始种子
        self.config['experiment']['seed'] = original_seed
        
        # 计算统计汇总
        self._compute_repeat_statistics(all_metrics)
        
        return all_metrics
    
    def _compute_repeat_statistics(self, all_metrics: list):
        """计算多次实验的统计信息"""
        logger.info("Computing statistics across repetitions...")
        
        calculator = MetricsCalculator(self.config)
        stats = calculator.compute_statistics(all_metrics)
        
        stats_path = self.exp_paths['stats']
        os.makedirs(stats_path, exist_ok=True)
        
        # 保存统计结果
        save_metrics_json(stats, os.path.join(stats_path, 'aggregated_metrics.json'))
        
        # 生成统计摘要
        summary = "# 多次实验统计汇总\n\n"
        summary += f"**实验ID**: {self.exp_id}\n\n"
        summary += f"**重复次数**: {len(all_metrics)}\n\n"
        summary += "## 指标统计\n\n"
        summary += "| 指标 | 均值 | 标准差 | 最小值 | 最大值 |\n"
        summary += "|------|------|--------|--------|--------|\n"
        
        for metric_name, metric_stats in stats.items():
            summary += f"| {metric_name} | {metric_stats['mean']:.4f} | {metric_stats['std']:.4f} | {metric_stats['min']:.4f} | {metric_stats['max']:.4f} |\n"
        
        with open(os.path.join(stats_path, 'statistical_summary.md'), 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Statistics saved to {stats_path}")
