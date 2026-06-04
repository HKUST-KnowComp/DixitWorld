"""
Dixit游戏配置文件
包含游戏常量、评分规则和其他配置参数
"""

# 游戏设置
NUM_PLAYERS = 4
CARDS_PER_PLAYER = 4  # 每人4张牌
TARGET_SCORE = 30  # 游戏结束分数

# 评分规则
SCORING_RULES = {
    'storyteller_win': 3,        # 讲述人赢了：讲述人得3分
    'guesser_correct': 3,        # 猜对的人得3分
    'others_when_storyteller_loses': 2,  # 讲述人输了：其他人各得2分
    'card_selected_bonus': 1     # 听众的牌被选择：每次得1分
}

# 图像设置
IMAGE_FOLDER = "images"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png']

# VLM设置
VLM_CONFIG = {
    'model_name': 'claude35h',  # 使用Claude 3.5 Haiku，支持多模态
    'api_endpoint': 'https://openrouter.ai/api/v1/chat/completions',
    'max_tokens': 100,
    'temperature': 0.7,
    'enable_vlm': True  # 启用真实VLM调用
}

# 日志设置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 游戏轮次设置
DEFAULT_ROUNDS = 10
