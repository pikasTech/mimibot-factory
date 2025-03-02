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
        '（')[0].strip().split('_')[0].strip().split(' ')[0].strip().split('の')[-1].strip()
    # 只保留中英文字符和数字
    speaker = re.sub(r'[^\u4e00-\u9fff\w\s]+', '', speaker)
    return f"<{speaker}>:{record['content']}"


def alpaca_gen(input_json, similarity_method='model'):
    """将聊天记录转换为 Alpaca 格式数据，跳过包含图片的消息，并计算数据质量
    Args:
        input_json: 输入的JSON文件路径
        similarity_method: 相似度计算方法，可选 'traditional' 或 'model'
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
            # 只保留中英文和常见符号和数字
            cleaned_content = re.sub(
                r'[^\u4e00-\u9fff\w\s.,!?]+', '', record['content'])
            if cleaned_content.strip():  # 确保清理后还有内容
                filtered_records.append(record)

    alpaca_data = []
    window_size = 11  # 10条输入 + 1条输出
    total_windows = len(filtered_records) - window_size + 1

    print("\n正在生成训练数据...")
    # 使用滑动窗口生成训练数据
    for i in tqdm(range(total_windows), desc="生成对话"):
        window = filtered_records[i:i + window_size]

        # 前10条作为输入
        input_messages = window[:-1]
        # 第11条作为输出
        output_message = window[-1]

        # 格式化输入消息
        formatted_input = "\\n".join(
            [format_chat_message(msg) for msg in input_messages])

        # 格式化输出消息
        formatted_output = output_message['content']

        # 计算数据质量分数

        similarity = calculate_similarity(formatted_input, formatted_output)

        quality = similarity

        # 构建 Alpaca 格式数据
        alpaca_item = {
            "instruction": "回复群聊" + formatted_input,
            "input": "",
            "output": formatted_output,
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
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"已生成Alpaca格式数据，共{len(alpaca_data)}条")
    print(f"保存至: {output_file}")


# 使用示例
if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='聊天记录处理工具')
    parser.add_argument(
        '--input', default="data/吉大·东方project同好会.txt", help='输入文件路径')
    parser.add_argument('--similarity', choices=['traditional', 'model'],
                        default='traditional', help='相似度计算方法')

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
            alpaca_gen(output_file, args.similarity)
    else:
        print(f"文件不存在: {args.input}")
