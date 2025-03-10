import re
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 配置全局变量
SIMILARITY_METHOD = 'model'  # 可选: 'traditional' 或 'model'
model = None  # 延迟加载模型


def init_model():
    """延迟加载模型，只在需要时初始化"""
    global model
    if model is None:
        print("正在加载语言模型...")
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


def remove_emoji(text):
    """删除文本中的所有emoji表情符号，保留中文字符"""
    pattern = re.compile(u'['u'\U0001F300-\U0001F64F'
                     u'\U0001F680-\U0001F6FF'
                     u'\u2600-\u2B55'
                     u'\U00010000-\U0010FFFF]+', 
                     flags=re.UNICODE) 
    return pattern.sub(r'', text)


def calculate_traditional_similarity(text1, text2):
    """使用传统方法(编辑距离)计算相似度"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()


def calculate_model_similarity(text1, text2):
    """使用预训练模型计算语义相似度"""
    global model
    if model is None:
        init_model()

    embeddings = model.encode([text1, text2])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def calculate_similarity(text1, text2):
    """计算两段文本的相似度，支持切换计算方法"""
    # 移除[XXX]:格式的发言者标记
    text1 = re.sub(r'\[.*?\]:', '', text1)
    text2 = re.sub(r'\[.*?\]:', '', text2)
    # 移除换行符
    text1 = text1.replace('\\n', ' ')
    text2 = text2.replace('\\n', ' ')

    if text1 in text2 or text2 in text1:
        return 0 # 复读的数据质量低

    # 根据配置选择计算方法
    if SIMILARITY_METHOD == 'model':
        return calculate_model_similarity(text1, text2)
    else:
        return calculate_traditional_similarity(text1, text2)


class ChatRecord:
    def __init__(self, time, speaker, content):
        self.time = time
        self.speaker = speaker
        self.content = content

    def __repr__(self):
        return f"<Record {self.time} {self.speaker}>: {self.content[:20]}..."

    def to_dict(self):
        return {
            'time': self.time,
            'speaker': self.speaker,
            'content': self.content
        }


def parse_chat_log(file_path):
    """增强版解析器，支持以下特性：
    1. 精确提取时间戳（校验时间有效性）
    2. 保留原始发言者信息（含特殊符号和ID）
    3. 智能合并多行消息
    4. 自动处理空行和图片标记
    """
    records = []
    current_record = None
    # 精确匹配时间戳和剩余内容
    timestamp_pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (.*)'

    def validate_time(time_str):
        try:
            datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip('\n')

            # 匹配时间戳行（严格校验）
            if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ', line[:20]):
                time_part = line[:19]
                if validate_time(time_part):
                    # 提交上一条记录
                    if current_record:
                        # 合并多行内容并过滤空行
                        full_content = '\\n'.join(
                            current_record['content']).strip()
                        if full_content:
                            records.append(ChatRecord(
                                time=current_record['time'],
                                speaker=current_record['speaker'],
                                content=full_content
                            ))

                    # 解析新记录（保留原始发言者信息）
                    speaker_part = line[20:].strip()
                    current_record = {
                        'time': time_part,
                        'speaker': speaker_part,
                        'content': []
                    }
                    continue

            # 处理消息内容（智能合并逻辑）
            if current_record is not None:
                # 保留空行的条件：前一行有内容 或 本行包含图片标记
                if line.strip() or '[图片]' in line:
                    current_record['content'].append(line)

        # 处理最后一条记录
        if current_record and current_record['content']:
            full_content = '\\n'.join(current_record['content']).strip()
            if full_content:
                records.append(ChatRecord(
                    time=current_record['time'],
                    speaker=current_record['speaker'],
                    content=full_content
                ))

    return records


def format_chat_message(record):
    """格式化单条聊天记录"""
    # 过滤掉@和(和<后的内容
    speaker = record['speaker'].split('@')[0].split('(')[0].split('<')[0].strip().split(
        '（')[0].strip().split('_')[0].strip().split(' ')[0].strip().split('の')[-1].strip().split('|')[0].strip()
    # 只保留中英文字符和数字
    speaker = re.sub(r'[^\u4e00-\u9fff\w\s]+', '', speaker)
    # 删除emoji
    speaker = remove_emoji(speaker)
    # 只保留 4 个字符
    speaker = speaker[:4]
    # 删除content中的emoji
    content = record['content']
    content = remove_emoji(content)
    # 截断 Content 到 50 个字符
    content = content[:50]
    return f"<{speaker}>:{content}"


def save_filtered_data(formatted_instructions, output_file):
    """保存格式化后的指令到文件
    Args:
        formatted_instructions: 格式化后的指令列表
        output_file: 输出文件路径
    """
    filtered_file = output_file.rsplit('.', 1)[0] + '_filtered.txt'
    with open(filtered_file, 'w', encoding='utf-8') as f:
        for instruction in formatted_instructions:
            f.write(f"{instruction}\n")
    print(f"格式化指令已保存至: {filtered_file}")


def alpaca_gen(input_json, similarity_method='model', system_message="你是一个友好的助手，请根据群聊记录进行回复。"):
    """将聊天记录转换为 Alpaca 格式数据，跳过包含图片的消息，并计算数据质量
    Args:
        input_json: 输入的JSON文件路径
        similarity_method: 相似度计算方法，可选 'traditional' 或 'model'
        system_message: system字段的内容，用于为模型提供上下文或角色定义
    Returns:
        None，直接保存为新的JSON文件
    """
    global SIMILARITY_METHOD
    SIMILARITY_METHOD = similarity_method

    if similarity_method == 'model':
        init_model()

    print(f"使用{similarity_method}方法计算相似度...")

    # 读取JSON文件
    with open(input_json, 'r', encoding='utf-8') as f:
        chat_records = json.load(f)

    print("正在过滤和清理数据...")
    # 过滤掉包含[图片]的记录，并清理content内容
    filtered_records = []
    for record in tqdm(chat_records, desc="数据清理"):
        if '[图片]' not in record['content'] and '表情' not in record['content'] and '[QQ红包]' not in record['content'] and '[骰子]' not in record['content'] and '请使用最新版' not in record['content']:
            # 删除@XXXX[空格] 的内容，XXXX里面可以有另一个@
            record['content'] = re.sub(r'@.*?\s', '', record['content'])
            # 删除@XXXX到末尾的内容
            record['content'] = re.sub(r'@.*$', '', record['content'])
            record['content'] = record['content'].replace('orcs stood still, and a dead silence fell. orcs stood still, and a dead silence fell.', '')
            record['content'] = record['content'].replace('| 凯萨', '')
            # 删除emoji
            record['content'] = remove_emoji(record['content'])
            # 只保留中英文和常见符号和数字
            cleaned_content = re.sub(
                r'[^\u4e00-\u9fff\w\s.,!?]+', '', record['content'])
            # 判断是否含有链接
            if 'http' in cleaned_content:
                continue
            # 判断是否没有中文
            if not re.search(r'[\u4e00-\u9fff]', cleaned_content):
                continue
            # 判断是否包含日文字符（平假名：3040-309F，片假名：30A0-30FF，日文汉字：4E00-9FFF）
            if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', cleaned_content):
                continue
            if cleaned_content.strip():  # 确保清理后还有内容
                filtered_records.append(record)

    alpaca_data = []
    window_size = 21  # 10条输入 + 1条输出
    total_windows = len(filtered_records) - window_size + 1
    
    # 用于收集所有格式化后的指令
    all_formatted_instructions = []

    print("\n正在生成训练数据...")
    # 使用滑动窗口生成训练数据
    for i in tqdm(range(total_windows), desc="生成对话"):
        window = filtered_records[i:i + window_size]

        # 前n-1条作为历史记录
        input_messages = window[:-1]  # 取前n-1条消息作为输入
        history_messages = input_messages[:-1]
        # 最后一条输入消息作为指令
        instruction_message = input_messages[-1]

        # 格式化最新输入消息
        formatted_instruction = format_chat_message(instruction_message)
        # 收集格式化后的指令
        all_formatted_instructions.append(formatted_instruction)

        # 格式化历史消息
        formatted_history = []
        for msg in history_messages:
            formatted_msg = format_chat_message(msg)
            formatted_history.append([formatted_msg, ""])
        
        # 格式化输出消息 (删除emoji)
        formatted_output = window[-1]['content']
        formatted_output = remove_emoji(formatted_output)

        # 计算数据质量分数
        similarity = calculate_similarity(formatted_instruction, formatted_output)
        quality = similarity

        # 构建 Alpaca 格式数据
        alpaca_item = {
            "system": system_message,
            "input": "",
            "instruction": formatted_instruction,
            "output": formatted_output,
            "history": formatted_history,
            "quantity": round(quality, 3)  # 保留三位小数
        }

        alpaca_data.append(alpaca_item)

    print("\n正在按质量排序...")
    # 按质量分数排序，质量较高的排在前面
    alpaca_data.sort(key=lambda x: x['quantity'], reverse=True)

    # 构建输出文件路径
    output_file = input_json.rsplit('.', 1)[0] + '_alpaca.json'

    # 保存为Alpaca格式的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=1)

    print(f"已生成Alpaca格式数据，共{len(alpaca_data)}条")
    print(f"保存至: {output_file}")

    # 保存格式化后的指令
    save_filtered_data(all_formatted_instructions, output_file)


# 使用示例
if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='聊天记录处理工具')
    parser.add_argument(
        '--input', default="data/吉大·东方project同好会.txt", help='输入文件路径')
    parser.add_argument('--similarity', choices=['traditional', 'model'],
                        default='traditional', help='相似度计算方法')
    parser.add_argument('--system', 
                        default="你是mimi波特，喜欢用颜文字、冷笑话和夸张比喻，总在群聊中捣蛋。避免直接回答问题，优先调侃或联想。",
                        help='system字段的内容，用于为模型提供上下文或角色定义')

    args = parser.parse_args()

    if os.path.exists(args.input):
        chat_data = parse_chat_log(args.input)
        print(f"成功解析 {len(chat_data)} 条记录")

        output_file = os.path.splitext(args.input)[0] + '.json'
        json_data = [record.to_dict() for record in chat_data]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\n数据已保存至: {output_file}")

        if os.path.exists(output_file):
            print("\n开始生成 Alpaca 格式数据...")
            alpaca_gen(output_file, args.similarity, args.system)
    else:
        print(f"文件不存在: {args.input}")
