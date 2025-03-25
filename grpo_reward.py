import re
import os
import numpy as np
import sentence_transformers
from prompt_utils import parse_prompt
from utils import extract_response_from_thinking, clean_user_message, setup_reward_logger, log_reward_data
import logging

# 全局变量，存储语义模型
semantic_model = None

# 初始化语义模型


def init_semantic_model(model_path):
    """
    初始化语义模型
    Args:
        model_path: 模型路径或Hugging Face模型ID
    Returns:
        初始化成功返回True，失败返回False
    """
    global semantic_model

    print(f"正在加载语义模型: {model_path}")

    # 检查是否为本地路径
    is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)

    try:
        if is_local_path:
            print(f"检测到本地模型路径: {model_path}")
        else:
            # 使用Hugging Face Hub加载
            print(f"正在尝试从Hugging Face Hub加载语义模型: {model_path}")
        semantic_model = sentence_transformers.SentenceTransformer(model_path)
        print(f"语义模型加载成功: {model_path}")
        return True
    except Exception as e:
        print(f"加载模型失败，尝试使用默认方式: {str(e)}")
        # 回退到默认的Hugging Face模型
        try:
            fallback_model = 'sentence-transformers/all-MiniLM-L6-v2'
            semantic_model = sentence_transformers.SentenceTransformer(
                fallback_model)
            print(f"已加载默认模型: {fallback_model}")
            return True
        except Exception as e:
            print(f"加载默认模型失败: {str(e)}")
            return False


def reward_len_response(completions, answer=None, **kwargs):
    """
    检查回复长度，如果长度和标准回复长度相差过大，则给予惩罚
    相差不大则容忍，不惩罚
    Args:
        completions: 模型生成的回复列表
        answer: 标准答案列表，可选
    Returns:
        奖励值列表
    """
    use_debug = False
    rewards = []

    # 提取实际回复内容
    responses = [extract_response_from_thinking(c) for c in completions]
    response_lengths = [len(r) for r in responses]

    if answer and len(answer) > 0:
        # 一对一比较每个回复与对应的标准答案
        target_lengths = [len(extract_response_from_thinking(a))
                          for a in answer]

        # 确保有足够的标准答案进行比较
        if len(target_lengths) < len(response_lengths):
            if use_debug:
                print(
                    f"警告: 标准答案数量({len(target_lengths)})少于回复数量({len(response_lengths)})")
            # 对超出的回复使用已有标准答案的中位数长度
            median_target = np.median(target_lengths)
            target_lengths.extend(
                [median_target] * (len(response_lengths) - len(target_lengths)))
    else:
        # 如果没有标准答案，使用所有回复的中位数长度
        median_length = np.median(response_lengths)
        target_lengths = [median_length] * len(response_lengths)
        if use_debug:
            print(f"回复中位数长度: {median_length}")

    # 定义长度差异的可接受范围
    acceptable_range = 0.3  # 30%的差异视为可接受
    # 添加绝对差异容忍阈值，对短文本更宽松
    absolute_tolerance = 5  # 少于5个字符的差异可以接受

    for i, length in enumerate(response_lengths):
        # 获取当前回复对应的目标长度
        target_length = target_lengths[min(i, len(target_lengths)-1)]

        # 计算绝对差异
        abs_diff = abs(length - target_length)
        
        # 计算与目标长度的差异比例
        if target_length == 0:
            # 避免除零错误
            ratio = float('inf') if length > 0 else 0
        else:
            ratio = abs_diff / target_length
            
        # 对短文本使用更宽松的标准
        # 如果标准答案很短（小于20个字符），并且绝对差异小于阈值，直接视为可接受
        is_acceptable = (ratio <= acceptable_range) or (target_length < 20 and abs_diff <= absolute_tolerance)

        # 根据差异比例计算奖励/惩罚
        if is_acceptable:
            # 在可接受范围内，给予奖励
            reward = 50
        else:
            # 计算惩罚因子，考虑短文本情况
            # 动态调整较短文本的可接受范围
            dynamic_range = acceptable_range
            if target_length < 50:
                # 对更短的文本逐渐放宽标准
                dynamic_range = acceptable_range + (0.5 * (50 - target_length) / 50)
            
            penalty_factor = min(5, ratio / dynamic_range - 1)  # 限制最大惩罚因子
            reward = -50 * penalty_factor

            # 对极短或极长的回复给予额外惩罚，但调整极短的判断标准
            if (target_length >= 20 and length < target_length * 0.2) or length > target_length * 3:
                reward -= 50

        rewards.append(reward)

        if use_debug:
            print(
                f"回复{i}长度: {length}, 目标长度: {target_length}, 绝对差异: {abs_diff}, 差异比例: {ratio:.2f}, 是否接受: {is_acceptable}, 奖励值: {reward}")

    return rewards


# 检查格式是否正确
def reward_format(completions, **kwargs):
    rewared = []
    for c in completions:
        c1 = c.count("<think>")
        c2 = c.count("</think>")
        # res = 1 if c1 == 1 and c2 == 1 else min(0.1 * (c1 + c2), 0.4)
        res = 1 if c2 == 1 else min(0.1 * (c2), 0.4)
        ans = extract_response_from_thinking(c)

        # c_not_true = ans.count('<') + ans.count('>') + ans.count(':') + ans.count('：') + ans.count('【') + ans.count('】') + ans.count('?') + ans.count('？') + ans.count('\n') + ans.count(' ') + ans.count('\r') + ans.count('\t') + ans.count('吗')

        # res -= 0.1 * c_not_true

        # 惩罚非中文或非ASCII字符
        # for char in ans:
        #     if not (char >= u'\u4e00' and char <= u'\u9fa5') and not char.isascii():
        #         res -= 0.1

        rewared.append(res * 100)
    return rewared

# 重复惩罚奖励函数


def reward_no_repetition(completions, prompts=None, **kwargs):
    rewared = []
    use_debug = False
    for i, completion in enumerate(completions):
        # 获取输入提示
        prompt = prompts[i] if prompts else ""

        # 解析提示中的用户消息和历史消息
        user_message, history_messages = parse_prompt(prompt)

        if use_debug:
            print(f"completion：{completion}")

        response = extract_response_from_thinking(completion)

        if use_debug:
            print("用户消息：", user_message)
            print("历史消息：", history_messages)

        # 转换为纯文本进行比较
        def to_pure_text(text):
            return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text).strip().lower()

        response_pure = to_pure_text(response)
        user_message_pure = to_pure_text(user_message)
        if use_debug:
            print("用户消息（纯文本）：", user_message_pure)
            print("回复（纯文本）：", response_pure)

        # 计算惩罚分数
        penalty = 0

        # Combine user message and history messages into a single list
        all_messages = [user_message] + history_messages
        # 去掉 <xxx>: 前缀
        all_messages = [message[message.find(
            ":")+1:].strip() for message in all_messages]

        # Check for repetition with all messages
        for message in all_messages:
            message_pure = to_pure_text(message)
            if use_debug:
                print("message_pure:", message_pure)

            # Check for complete repetition
            if response_pure == message_pure:
                penalty -= 100  # Severe penalty for exact repetition
                break

        # 3. 检查是否包含"禁止重复"等提示词
        if "禁止重复" in response or "不要重复" in response:
            penalty -= 40

        # 4. 检查回复是否以问号结尾
        if response.strip().endswith('?') or response.strip().endswith('？'):
            penalty -= 30

        # 5. 检查是否包含"历史消息"字样
        if "历史消息" in response:
            penalty -= 50

        # 6. 检查"帽"出现的次数是否过多
        if response.count("帽") >= 3:
            penalty -= 30

        rewared.append(penalty * 50)

    return rewared

# 用户相似度奖励函数


# 全局变量存储日志记录器
user_similarity_logger = None


def init_user_similarity_logger(timestamp):
    """初始化用户相似度奖励的日志记录器"""
    global user_similarity_logger
    user_similarity_logger = setup_reward_logger(timestamp, "similarity")
    return user_similarity_logger is not None


def reward_user_similarity(completions, prompts=None, **kwargs):
    global semantic_model
    if semantic_model is None:
        print("警告：语义模型未初始化，无法计算相似度")
        return [0] * len(completions)

    rewards = []
    use_debug = True

    # 提取回复和用户消息
    user_messages = []
    responses = []

    for i, completion in enumerate(completions):
        # 获取输入提示
        prompt = prompts[i] if prompts else ""

        # 解析提示中的用户消息并清理
        user_message, _ = parse_prompt(prompt)
        user_message = clean_user_message(user_message)
        user_messages.append(user_message)

        # 提取回复部分
        response = extract_response_from_thinking(completion)
        responses.append(response)

    # 使用语义模型计算相似度
    try:
        similarities = sentence_transformers.util.cos_sim(
            semantic_model.encode(responses),
            semantic_model.encode(user_messages)
        ).diagonal().tolist()

        # 计算奖励
        for sim in similarities:
            # 相似度作为基础奖励
            reward = sim * 100

            # 相似度较高时给予额外奖励
            if sim > 0.3:
                reward += 50
            if sim > 0.5:
                reward += 100
            if sim > 0.7:
                reward -= 150

            rewards.append(reward)

        # 如果和用户消息重复，奖励清零
        for i, (user_message, response) in enumerate(zip(user_messages, responses)):
            if user_message == response:
                rewards[i] = 0

        if use_debug and len(rewards) > 0:
            # 打印最高奖励的示例
            best_idx = rewards.index(max(rewards))
            print(f"用户消息: {user_messages[best_idx]}")
            print(f"模型回复: {responses[best_idx]}")
            print(f"相似度: {similarities[best_idx]}")
            print(f"奖励值: {rewards[best_idx]}")

    except Exception as e:
        print(f"计算相似度时出错: {str(e)}")
        rewards = [0] * len(completions)

    return rewards


def reward_similarity(completions, prompts=None, answer=None, **kwargs):
    global semantic_model
    global user_similarity_logger
    if semantic_model is None:
        print("警告：语义模型未初始化，无法计算相似度")
        return [0] * len(completions)

    use_debug = True
    responses = []
    for completion in completions:
        response = extract_response_from_thinking(completion)
        responses.append(response)
    answers = []
    for ans in answer:
        response = extract_response_from_thinking(ans)
        response = clean_user_message(response)
        answers.append(response)
    rewards = []
    similarity = sentence_transformers.util.cos_sim(
        semantic_model.encode(responses), semantic_model.encode(answers))
    similarity = similarity.diagonal().tolist()
    for sim in similarity:
        reward = 0
        reward += sim * 500
        if sim > 0.5:
            reward += sim * 1000
        if sim > 0.7:
            reward += sim * 1500
        if sim > 0.9:
            reward += sim * 2000
        rewards.append(reward)
    if use_debug:
        # 打印前4个reward最多的
        best_indexs = sorted(range(len(rewards)),
                             key=lambda i: rewards[i], reverse=True)[:4]
        # 倒着顺序打印
        best_indexs.reverse()
        for i in best_indexs:
            print(f"<think>{completions[i]}")
            print(f"回答: {responses[i]}")
            print(f"标准: {answers[i]}")
            print(f"Similarity: {similarity[i]}")
            print(f"Reward: {rewards[i]}")
            print('-' * 10)
        print(f"======== 处理了 {len(completions)} 个回答 ========")

    # 记录奖励数据到日志
    if user_similarity_logger:
        # 创建字典来存储唯一的(prompt, ans)对及其相似度列表
        unique_pairs = {}
        for prompt, ans, sim in zip(prompts, answers, similarity):
            pair_key = (prompt, ans)
            if pair_key in unique_pairs:
                unique_pairs[pair_key].append(sim)
            else:
                unique_pairs[pair_key] = [sim]
        
        # 对于每个唯一的对，计算平均相似度并记录
        for (prompt, ans), sim_list in unique_pairs.items():
            avg_sim = sum(sim_list) / len(sim_list)
            log_reward_data(user_similarity_logger, prompt, ans, avg_sim)

    return rewards
