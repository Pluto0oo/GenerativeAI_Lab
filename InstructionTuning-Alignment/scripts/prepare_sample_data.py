#!/usr/bin/env python3
"""生成示例指令数据集和偏好数据集

生成小规模数据用于快速验证Pipeline流程
- SFT数据: instruction/input/output 格式
- DPO数据: prompt/chosen/rejected 格式
"""
import os
import json
import random

random.seed(42)

# ============ SFT 指令数据集 ============
sft_data = [
    {"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进。主要包括监督学习、无监督学习和强化学习三种范式。"},
    {"instruction": "用Python写一个冒泡排序", "input": "", "output": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
    {"instruction": "什么是过拟合？如何避免？", "input": "", "output": "过拟合是指模型在训练数据上表现很好，但在测试数据上表现较差的现象。避免方法包括：1)增加训练数据；2)正则化(L1/L2)；3)Dropout；4)早停法；5)交叉验证。"},
    {"instruction": "翻译以下英文", "input": "Hello, how are you?", "output": "你好，你怎么样？"},
    {"instruction": "解释Transformer架构", "input": "", "output": "Transformer是一种基于自注意力机制的神经网络架构，由编码器和解码器组成。其核心是多头自注意力机制，能够并行处理序列数据，避免了RNN的序列依赖问题。"},
    {"instruction": "写一首关于春天的诗", "input": "", "output": "春风拂面柳丝长，\n桃花杏花满园香。\n燕子归来筑新巢，\n万物复苏好时光。"},
    {"instruction": "什么是梯度下降法", "input": "", "output": "梯度下降法是一种优化算法，通过沿损失函数的负梯度方向迭代更新参数，以最小化损失函数。学习率是其关键超参数，决定了步长大小。"},
    {"instruction": "解释什么是API", "input": "", "output": "API(应用程序编程接口)是一组定义了软件组件之间如何交互的规则。它允许不同的软件系统进行通信，常见的有REST API、GraphQL等。"},
    {"instruction": "写一个SQL查询", "input": "查询年龄大于20岁的学生", "output": "SELECT * FROM students WHERE age > 20;"},
    {"instruction": "什么是Docker", "input": "", "output": "Docker是一个开源的容器化平台，它将应用程序及其依赖打包到容器中，确保应用在任何环境中都能一致运行。主要概念包括镜像、容器和仓库。"},
    {"instruction": "解释递归", "input": "", "output": "递归是一种编程技术，函数在其定义中调用自身。每个递归函数需要有基线条件(base case)来终止递归，否则会导致栈溢出。例如阶乘计算：def fact(n): return 1 if n<=1 else n*fact(n-1)"},
    {"instruction": "什么是神经网络", "input": "", "output": "神经网络是一种模仿生物神经系统的计算模型，由神经元(节点)和连接组成。典型的神经网络包含输入层、隐藏层和输出层，通过反向传播算法学习权重。"},
    {"instruction": "写一个Python函数计算斐波那契数列", "input": "", "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b"},
    {"instruction": "解释什么是JSON", "input": "", "output": "JSON(JavaScript对象表示法)是一种轻量级的数据交换格式。它使用键值对和数组结构，易于人阅读和编写，也易于机器解析和生成。"},
    {"instruction": "什么是Git", "input": "", "output": "Git是一个分布式版本控制系统，用于跟踪文件变化并协调多人协作。主要操作包括clone、commit、push、pull、branch和merge。"},
    {"instruction": "解释RESTful API", "input": "", "output": "RESTful API是一种基于HTTP协议的API设计风格，使用GET、POST、PUT、DELETE等方法操作资源。资源通过URI标识，数据通常以JSON格式传输。"},
    {"instruction": "写一个正则表达式匹配邮箱", "input": "", "output": "import re\npattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\nemail = 'test@example.com'\nif re.match(pattern, email):\n    print('Valid')"},
    {"instruction": "什么是缓存", "input": "", "output": "缓存是一种临时存储技术，将频繁访问的数据保存在快速访问的存储介质中，以减少数据访问延迟。常见的有CPU缓存、浏览器缓存、Redis等。"},
    {"instruction": "解释什么是微服务架构", "input": "", "output": "微服务架构是一种将应用拆分为多个小型、独立服务的架构风格。每个服务围绕业务能力构建，独立部署，通过轻量级机制通信。优点是可扩展性和独立性。"},
    {"instruction": "写一个Python装饰器示例", "input": "", "output": "def timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'Took {time.time()-start:.2f}s')\n        return result\n    return wrapper"},
]

# ============ DPO 偏好数据集 ============
dpo_data = [
    {"prompt": "解释什么是机器学习", "chosen": "机器学习是人工智能的一个分支，它使计算机系统能够通过数据自动改进性能，主要包括监督学习、无监督学习和强化学习。", "rejected": "机器学习就是让电脑学习。"},
    {"prompt": "什么是Python", "chosen": "Python是一种高级、解释型、通用的编程语言，以简洁的语法和丰富的库生态著称，广泛用于Web开发、数据科学、人工智能等领域。", "rejected": "Python是一种编程语言。"},
    {"prompt": "解释什么是深度学习", "chosen": "深度学习是机器学习的一个分支，使用多层神经网络模拟人脑处理信息的方式。它在图像识别、自然语言处理等领域取得了突破性进展。", "rejected": "深度学习就是很深的学习。"},
    {"prompt": "写一个冒泡排序", "chosen": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr", "rejected": "arr.sort()"},
    {"prompt": "什么是过拟合", "chosen": "过拟合是指模型在训练集上表现优异，但在测试集上泛化能力差的现象。可通过正则化、Dropout、数据增强等方法缓解。", "rejected": "过拟合就是模型太好了。"},
    {"prompt": "解释什么是API", "chosen": "API(应用程序编程接口)是一组规则和协议，定义了软件组件之间如何交互。它允许不同系统进行通信，如REST API使用HTTP协议传输数据。", "rejected": "API就是接口。"},
    {"prompt": "什么是Docker", "chosen": "Docker是一个开源的容器化平台，将应用及其依赖打包到容器中，确保在任何环境中一致运行。核心概念包括镜像、容器和仓库。", "rejected": "Docker是一个软件。"},
    {"prompt": "解释什么是递归", "chosen": "递归是一种编程技术，函数在其定义中调用自身。必须包含基线条件以避免无限递归。典型应用包括阶乘、斐波那契数列、树遍历等。", "rejected": "递归就是自己调用自己。"},
    {"prompt": "什么是JSON", "chosen": "JSON(JavaScript Object Notation)是一种轻量级数据交换格式，易于人读写和机器解析。它基于键值对和数组结构，广泛用于Web API数据传输。", "rejected": "JSON是一种数据格式。"},
    {"prompt": "解释什么是Git", "chosen": "Git是一个分布式版本控制系统，用于跟踪文件变化和协作开发。主要概念包括仓库、提交、分支和合并，支持离线工作和非线性开发。", "rejected": "Git是一个工具。"},
    {"prompt": "什么是神经网络", "chosen": "神经网络是模仿生物神经系统的计算模型，由相互连接的神经元组成。典型结构包括输入层、隐藏层和输出层，通过反向传播算法学习。", "rejected": "神经网络就是网络。"},
    {"prompt": "写一个Python函数", "chosen": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b", "rejected": "def f(n): return n"},
    {"prompt": "解释RESTful API", "chosen": "RESTful API是基于HTTP协议的API设计风格，使用标准HTTP方法(GET/POST/PUT/DELETE)操作资源，资源通过URI标识，无状态、可缓存。", "rejected": "REST就是API。"},
    {"prompt": "什么是缓存", "chosen": "缓存是一种临时存储技术，将频繁访问的数据保存在快速访问介质中，减少延迟。常见类型包括CPU缓存、浏览器缓存和分布式缓存(如Redis)。", "rejected": "缓存就是存东西。"},
    {"prompt": "解释什么是微服务", "chosen": "微服务架构将应用拆分为多个独立的小型服务，每个服务围绕业务能力构建，独立部署和扩展，通过轻量级机制(如HTTP/REST)通信。", "rejected": "微服务就是小服务。"},
]

# 保存SFT数据
sft_dir = "data/processed/alpaca_cleaned"
os.makedirs(sft_dir, exist_ok=True)
# 划分训练/验证
split = int(len(sft_data) * 0.9)
with open(os.path.join(sft_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(sft_data[:split], f, ensure_ascii=False, indent=2)
with open(os.path.join(sft_dir, "validation.json"), "w", encoding="utf-8") as f:
    json.dump(sft_data[split:], f, ensure_ascii=False, indent=2)
print(f"SFT数据: {len(sft_data[:split])}训练, {len(sft_data[split:])}验证 -> {sft_dir}")

# 保存DPO数据
dpo_dir = "data/processed/preference_data"
os.makedirs(dpo_dir, exist_ok=True)
split = int(len(dpo_data) * 0.9)
with open(os.path.join(dpo_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(dpo_data[:split], f, ensure_ascii=False, indent=2)
with open(os.path.join(dpo_dir, "validation.json"), "w", encoding="utf-8") as f:
    json.dump(dpo_data[split:], f, ensure_ascii=False, indent=2)
print(f"DPO数据: {len(dpo_data[:split])}训练, {len(dpo_data[split:])}验证 -> {dpo_dir}")

print("\n示例数据生成完成！")
