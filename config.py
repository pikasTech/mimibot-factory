"""
OpenAI API配置文件
"""

# OpenAI API配置
OPENAI_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
OPENAI_API_KEY = "2f1646dd-0c30-4541-897a-96219d3009de"  # 替换为实际API Key
# MODEL_NAME = "ep-20250204211231-lhx4t" # R1
MODEL_NAME = "ep-20250204211807-p9dzg" # V3
TEMPERATURE = 0.1  # 低温度以获得一致的评分
MAX_TOKENS = 1024  # 最大生成令牌数
# BASE_MODEL = 'output/mimibot_l3d_v1.1'
BASE_MODEL = 'models/Tifa-DeepsexV2-7b-Cot-0317-F16'
LORA_OUTPUT = "./results/mimibot_tifa"

LORA_PATH = "results/mimibot_l3/checkpoint-1600"
