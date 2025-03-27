from typing import List, Tuple
import re

PROMPT_EXAMPLE = """
<|im_start|>system

    你是mimi波特，你要回复群聊，带上你的思考。【示例输出】
<think>我看到群友在聊...所以我想回复...</think>你的回复 
    <|im_end|>
<|im_start|>user
【任务目标】
你是mimi波特，回复最新消息，多使用颜文字，禁止重复历史消息

【示例输出】
<think>我看到群友在聊...所以我想回复...</think>你的回复

【历史消息】:
<亡霊>:今年都推了家里三个相亲了
<495>:啊啊啊啊好

【最新消息】
<帽>:我跟爸妈打玉玉牌，现在已经不催了<|im_end|>
<|im_start|>assistant
"""

def extract_history(prompt: str) -> List[str]:
    """从prompt中提取历史消息列表"""
    match = re.search(r'【历史消息】:\s*(.*?)\s*【最新消息】', prompt, re.DOTALL)
    if not match:
        return []
    history_text = match.group(1).strip()
    return [line.strip() for line in history_text.split('\n') if line.strip()]


def extract_latest(prompt: str) -> str:
    """从prompt中提取最新消息"""
    match = re.search(r'【最新消息】\s*(.*?)$', prompt, re.DOTALL)
    return match.group(1).strip() if match else ''


def parse_prompt(promt:str) -> Tuple[str, List[str]]:
    """
    返回一个元组，包含两个元素
    第一个是用户的输入
    第二个是历史消息
    """
    # 提取 user_message
    # user_message = re.findall(r"<\|im_start\|>user(.*?)<\|im_end\|>", promt, re.DOTALL)[0].strip()
    user_message = re.findall(r"<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>", promt, re.DOTALL)[0].strip()
    # 提取 
    latest_message = extract_latest(user_message)
    history_message = extract_history(user_message)
    return latest_message, history_message
    


if __name__ == "__main__":
    new_msg, history_msg = parse_prompt(PROMPT_EXAMPLE)
    print(f"New message: {new_msg}")
    print(f"History messages: {history_msg}")



