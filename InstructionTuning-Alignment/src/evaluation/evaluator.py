import logging
import torch
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.eval_config = config['evaluation']
        self.device = model.device
    
    def generate(self, prompts: List[str], max_new_tokens: int = None) -> List[str]:
        """生成回复"""
        if max_new_tokens is None:
            max_new_tokens = self.eval_config['generation']['max_new_tokens']
        
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': self.eval_config['generation']['temperature'],
            'top_p': self.eval_config['generation']['top_p'],
            'do_sample': self.eval_config['generation']['do_sample'],
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        
        self.model.eval()
        generated_texts = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating responses"):
                # 格式化prompt
                if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted_prompt = f"Instruction: {prompt}\nResponse: "
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.config['data'].get('max_length', 2048),
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # 解码输出（只取生成部分）
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        return generated_texts
    
    def evaluate_on_dataset(self, eval_dataset) -> Dict:
        """在评估数据集上进行评估"""
        logger.info("Evaluating model on validation dataset...")
        
        # 提取prompts和参考答案
        prompts = []
        references = []
        
        for example in eval_dataset:
            instruction = example.get('instruction', example.get('prompt', ''))
            input_text = example.get('input', '')
            output = example.get('output', example.get('chosen', ''))
            
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\n"
            else:
                prompt = f"Instruction: {instruction}\n"
            
            prompts.append(prompt)
            references.append(output)
        
        # 生成回复
        predictions = self.generate(prompts)
        
        # 计算指标
        from src.evaluation.metrics import MetricsCalculator
        calculator = MetricsCalculator(self.config)
        metrics = calculator.compute(predictions, references)
        
        metrics['predictions'] = predictions
        metrics['references'] = references
        
        return metrics
    
    def test_instruction_following(self) -> Dict:
        """测试指令跟随能力"""
        test_prompts = [
            "请用Python写一个快速排序函数。",
            "解释一下什么是机器学习中的过拟合。",
            "帮我翻译这句话：Hello, how are you?",
            "写一首关于秋天的诗。",
            "分析一下这段代码的时间复杂度：def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        ]
        
        logger.info("Running instruction following test...")
        predictions = self.generate(test_prompts, max_new_tokens=256)
        
        results = {
            'test_prompts': test_prompts,
            'predictions': predictions,
        }
        
        # 打印结果
        for prompt, pred in zip(test_prompts, predictions):
            logger.info(f"Prompt: {prompt[:50]}...")
            logger.info(f"Response: {pred[:100]}...")
            logger.info("---")
        
        return results
