import os
import sys
import datetime
import logging
import json


class TeeOutput:
    """同时将输出发送到文件和终端的类"""

    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def setup_logging():
    """设置日志记录，将输出重定向到日志文件和终端"""
    # 创建日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f"{log_dir}/{timestamp}_log.txt"

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 重定向标准输出和标准错误
    log_file_handler = open(log_file, "a")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(log_file_handler, original_stdout)
    sys.stderr = TeeOutput(log_file_handler, original_stderr)

    print(f"日志系统初始化完成，日志文件: {log_file}")

    # 返回需要恢复的对象和时间戳
    return {
        "log_file": log_file,
        "log_file_handler": log_file_handler,
        "original_stdout": original_stdout,
        "original_stderr": original_stderr,
        "timestamp": timestamp
    }


def setup_reward_logger(timestamp, reward_type):
    """
    设置奖励日志记录器

    Args:
        timestamp: 时间戳，与主日志保持一致
        reward_type: 奖励类型名称（用于日志文件名）

    Returns:
        日志记录器对象
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = f"{log_dir}/{timestamp}_{reward_type}.jsonl"

    # 创建一个专门的记录器，避免与根记录器冲突
    reward_logger = logging.getLogger(f"reward_{reward_type}")
    reward_logger.setLevel(logging.INFO)

    # 确保处理器不重复添加
    if not reward_logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        reward_logger.addHandler(file_handler)

    print(f"{reward_type}奖励日志初始化完成，日志文件: {log_file}")
    # 不输出 INFO 级别的日志到终端
    reward_logger.propagate = False
    return reward_logger


def cleanup_logging(logging_context):
    """清理日志设置，恢复原始输出流"""
    sys.stdout = logging_context["original_stdout"]
    sys.stderr = logging_context["original_stderr"]
    logging_context["log_file_handler"].close()
    print(f"日志系统已关闭，日志文件: {logging_context['log_file']}")


def extract_response_from_thinking(completion):
    """
    从思维链输出中提取最终回答部分
    Args:
        completion: 可能包含思维链的完整回答字符串
    Returns:
        提取出的回答部分
    """
    if "</think>" in completion:
        response = completion.split("</think>")[-1]
    else:
        # 只保留最后50个字符
        response = completion[-50:]
    return response


def clean_user_message(user_message):
    """
    清理用户消息，移除前缀标识符
    Args:
        user_message: 原始用户消息
    Returns:
        清理后的用户消息
    """
    if ">:" in user_message:
        user_message = user_message.split(">:")[-1].strip()
    if "：" in user_message:
        user_message = user_message.split("：")[-1].strip()
    return user_message


def log_reward_data(logger, prompt, answer, reward):
    """
    记录奖励数据到日志文件

    Args:
        logger: 日志记录器对象
        prompt: 提示文本
        answer: 生成的回答
        similarity: 计算的奖励值
    """
    data = {
        "prompt": prompt,
        "answer": answer,
        "similarity": float(reward)
    }
    logger.info(json.dumps(data, ensure_ascii=False))


def apply_template(prompt, tokenizer, system_prompt=None):
    """应用聊天模板到提示文本
    
    参数:
        prompt: 用户输入的提示文本
        tokenizer: 使用的分词器
        system_prompt: 可选的系统提示，默认为None
    
    返回:
        应用模板后的提示文本
    """
    # 构建消息列表
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # 应用聊天模板
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return formatted_prompt
