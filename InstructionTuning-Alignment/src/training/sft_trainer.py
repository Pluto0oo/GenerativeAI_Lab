import logging
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

class SFTTrainerWrapper:
    """SFT训练器封装 (兼容 TRL 1.x)"""

    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.output_dir = output_dir
        self.training_config = config['training']
        self.trainer = None

    def _formatting_func(self):
        """返回一个将原始数据格式化为文本的函数"""
        tokenizer = self.tokenizer
        preprocessing = self.config['data'].get('preprocessing', {})

        def format_example(example):
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')

            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: "
            else:
                prompt = f"Instruction: {instruction}\nResponse: "

            if preprocessing.get('chat_template', True) and hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                text = prompt + output
            return text

        return format_example

    def setup(self):
        """设置SFT训练器"""
        # 兼容TRL 1.x: 使用SFTConfig
        sft_config_kwargs = dict(
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
            packing=False,
        )

        # TRL 1.x SFTConfig 使用 max_seq_length
        training_args = None
        for max_len_key in ['max_length', 'max_seq_length']:
            try:
                training_args = SFTConfig(**sft_config_kwargs, **{max_len_key: self.config['data']['max_length']})
                logger.info(f"SFTConfig created with {max_len_key}={self.config['data']['max_length']}")
                break
            except (TypeError, Exception) as e:
                logger.debug(f"SFTConfig with {max_len_key} failed: {e}")
                continue

        if training_args is None:
            # 回退到 TrainingArguments
            training_args = TrainingArguments(
                **{k: v for k, v in sft_config_kwargs.items() if k != 'packing'}
            )

        # 创建SFTTrainer
        trainer_kwargs = dict(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            formatting_func=self._formatting_func(),
        )

        # TRL 1.x 使用 processing_class
        self.trainer = None
        for attempt in [
            lambda: SFTTrainer(processing_class=self.tokenizer, **trainer_kwargs),
            lambda: SFTTrainer(tokenizer=self.tokenizer, **trainer_kwargs),
        ]:
            try:
                self.trainer = attempt()
                break
            except TypeError as e:
                logger.debug(f"SFTTrainer attempt failed: {e}")
                continue

        if self.trainer is None:
            raise RuntimeError("Failed to initialize SFTTrainer with any API variant")

        logger.info("SFT Trainer initialized successfully")
        return self

    def train(self):
        """开始训练"""
        logger.info("Starting SFT training...")
        train_result = self.trainer.train()

        metrics = train_result.metrics
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info(f"Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")
        return metrics

    def evaluate(self):
        """评估模型"""
        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results

    def get_history(self):
        """获取训练历史"""
        return self.trainer.state.log_history
