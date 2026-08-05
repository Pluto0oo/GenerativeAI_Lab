#!/usr/bin/env python3
"""构建医疗RAG知识库

使用方法:
    python scripts/build_knowledge_base.py
"""
import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_knowledge_base():
    """构建医疗知识库"""
    kb_dir = os.path.join(project_root, 'knowledge_base', 'medical_guidelines')
    embeddings_dir = os.path.join(project_root, 'knowledge_base', 'embeddings')
    os.makedirs(kb_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    print("构建医疗知识库...")
    
    # 医学指南内容
    medical_guidelines = {
        'hypertension.md': """# 高血压诊疗指南

## 定义
高血压是指在未使用降压药物的情况下，收缩压≥140mmHg和/或舒张压≥90mmHg。

## 诊断标准
- 正常血压：收缩压<120mmHg和舒张压<80mmHg
- 正常高值：收缩压120-139mmHg和/或舒张压80-89mmHg
- 高血压1级：收缩压140-159mmHg和/或舒张压90-99mmHg
- 高血压2级：收缩压160-179mmHg和/或舒张压100-109mmHg
- 高血压3级：收缩压≥180mmHg和/或舒张压≥110mmHg

## 治疗药物
1. 钙通道阻滞剂(CCB)：氨氯地平、硝苯地平
2. 血管紧张素转换酶抑制剂(ACEI)：卡托普利、依那普利
3. 血管紧张素II受体拮抗剂(ARB)：氯沙坦、缬沙坦
4. β受体阻滞剂：美托洛尔、阿替洛尔
5. 利尿剂：氢氯噻嗪、呋塞米
""",
        
        'diabetes.md': """# 糖尿病诊疗指南

## 定义
糖尿病是一种以慢性高血糖为特征的代谢性疾病。

## 诊断标准
- 随机血糖≥11.1 mmol/L + 典型糖尿病症状
- 空腹血糖≥7.0 mmol/L
- OGTT 2小时血糖≥11.1 mmol/L
- 糖化血红蛋白HbA1c≥6.5%

## 正常血糖范围
- 空腹血糖：3.9-6.1 mmol/L
- 餐后2小时：<7.8 mmol/L
- 糖化血红蛋白：4.0-6.0%

## 治疗药物
1. 二甲双胍：首选药物
2. 磺脲类：格列本脲、格列齐特
3. α-糖苷酶抑制剂：阿卡波糖
4. 胰岛素增敏剂：吡格列酮
5. 胰岛素：1型糖尿病必需
""",

        'mi.md': """# 心肌梗死诊疗指南

## 症状
- 持续性胸骨后压榨性疼痛，可放射至左臂、下颌、背部
- 持续时间>20分钟，含服硝酸甘油不缓解
- 伴随症状：大汗、恶心、呕吐、呼吸困难

## 急救处理
1. 立即拨打120
2. 让患者保持安静，避免活动
3. 给予阿司匹林300mg嚼服
4. 监测心电图和生命体征

## 诊断
- 典型胸痛症状
- 心电图ST段改变
- 肌钙蛋白阳性
- 冠状动脉造影显示堵塞

## 治疗
- 再灌注治疗：溶栓或PCI
- 抗血小板治疗：阿司匹林+氯吡格雷
- 抗凝治疗：肝素
- β受体阻滞剂
""",

        'peptic_ulcer.md': """# 消化性溃疡诊疗指南

## 定义
消化性溃疡是指发生在胃或十二指肠的慢性溃疡。

## 症状
- 上腹痛：周期性、节律性
- 胃溃疡：进食后痛
- 十二指肠溃疡：饥饿痛，进食后缓解
- 伴随症状：反酸、嗳气、恶心

## 并发症
1. 上消化道出血（最常见）
2. 幽门梗阻
3. 穿孔
4. 癌变（胃溃疡）

## 治疗
1. 质子泵抑制剂：奥美拉唑
2. 胃黏膜保护剂：硫糖铝
3. 根除幽门螺杆菌：PPI+两种抗生素
4. 手术治疗：并发症或药物无效时
""",

        'vitamin.md': """# 维生素指南

## 脂溶性维生素
- 维生素A：来源-动物肝脏、胡萝卜；功能-视觉、免疫
- 维生素D：来源-阳光、鱼肝油；功能-钙磷代谢、骨骼健康
- 维生素E：来源-植物油、坚果；功能-抗氧化
- 维生素K：来源-绿叶蔬菜；功能-凝血

## 水溶性维生素
- 维生素B1：来源-全谷物、瘦肉；功能-能量代谢
- 维生素B12：来源-肉类、蛋类；功能-造血、神经
- 维生素C：来源-柑橘、蔬菜；功能-抗氧化、胶原

## 每日推荐摄入量
- 维生素A：800μg
- 维生素D：10μg
- 维生素C：100mg
- 维生素B12：2.4μg
"""
    }
    
    for filename, content in medical_guidelines.items():
        filepath = os.path.join(kb_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  创建: {filename}")
    
    print(f"\n知识库创建完成！")
    print(f"  文件数: {len(medical_guidelines)}")
    print(f"  路径: {kb_dir}")
    print(f"\n注意: 首次使用RAG前需运行向量嵌入构建")
    print(f"  python -c \"from src.rag.embeddings import EmbeddingModel; em = EmbeddingModel(); em.build_index('{kb_dir}')\"")


if __name__ == '__main__':
    build_knowledge_base()
