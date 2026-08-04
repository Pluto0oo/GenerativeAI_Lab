import logging
import torch
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)

class DPOTrainerWrapper:
    """DPO训练器封装 (兼容 TRL 1.x)"""

    def __init__(self, model, ref_model, tokenizer, train_dataset, eval_dataset, config, output_dir):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.output_dir = output_dir
        self.training_config = config['training']
        self.trainer = None

    def setup(self):
        """设置DPO训练器（DPOTrainer TRL 1.x: beta/max_length放DPOConfig, processing_class传tokenizer）"""
        beta = self.training_config.get('beta', 0.1)
        max_length = self.training_config.get('max_length', 512)

        dpo_config_kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config['epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            lr_scheduler_type=self.training_config['lr_scheduler'],
            warmup_ratio=self.training_config['warmup_ratio'],
            weight_decay=self.training_config['weight_decay'],
            max_grad_norm=self.training_config['max_grad_norm'],
            fp16=self.training_config.get('fp16', True),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
            logging_steps=self.training_config['logging_steps'],
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            save_total_limit=self.config['output']['save_total_limit'],
            load_best_model_at_end=self.config['output'].get('save_best_model', False),
            # load_best_model_at_end 要求 save_strategy == eval_strategy；
            # 小数据集总步数往往 < save_steps，用 epoch 策略确保每轮有 checkpoint 可加载
            save_strategy="epoch" if self.config['output'].get('save_best_model', False) else "steps",
            eval_strategy="epoch" if self.config['output'].get('save_best_model', False) else "steps",
            report_to="wandb" if self.config['logging'].get('use_wandb', False) else "none",
            remove_unused_columns=False,
            # DPOConfig 特有参数：beta 和 max_length 放在 config 里（TRL 1.x）
            beta=beta,
            max_length=max_length,
        )

        # 创建 DPOConfig（TRL 1.x 专用）
        try:
            training_args = DPOConfig(**dpo_config_kwargs)
            logger.info(f"DPOConfig OK: beta={beta}, max_length={max_length}")
        except TypeError as e:
            logger.warning(f"DPOConfig failed ({e}), fallback to TrainingArguments")
            # 回退：把 DPOConfig 特有参数从 kwargs 中移除
            fallback_kwargs = {k: v for k, v in dpo_config_kwargs.items() if k not in ['beta', 'max_length']}
            training_args = TrainingArguments(**fallback_kwargs)

        # 创建 DPOTrainer（TRL 1.x: 必须传 processing_class）
        # DPOTrainer 自己从 config 里取 beta/max_length
        trainer_kwargs = dict(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        self.trainer = DPOTrainer(**trainer_kwargs)
        logger.info(f"DPOTrainer initialized (TRL 1.x: processing_class=tokenizer, beta={beta})")
        return self

    def train(self):
        """开始训练"""
        logger.info("Starting DPO training...")
        train_result = self.trainer.train()

        metrics = train_result.metrics
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info(f"DPO Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")
        return metrics

    def evaluate(self):
        """评估模型"""
        logger.info("Running DPO evaluation...")
        eval_results = self.trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results

    def get_history(self):
        """获取训练历史"""
        return self.trainer.state.log_history
