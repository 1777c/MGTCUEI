"""MGTCUEI 使用受控上层证据注入器预测药物–疾病关联。

训练与评估入口位于 mgtcuei.pipeline，配对评分入口位于 mgtcuei.inference。主要模块涵盖图编码、记忆、读取器、残差注入、语言模型、读出、指标、检查点和训练引擎。config、data 和 pipeline 分别提供实验配置、冻结数据读取和实验编排。
"""

__version__ = "1.0.0"
