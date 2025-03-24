import re
import os
import sentence_transformers
from prompt_utils import parse_prompt
from utils import extract_response_from_thinking

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
    """


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

        # 解析提示中的用户消息
        user_message, _ = parse_prompt(prompt)
        if ">:" in user_message:
            user_message = user_message.split(">:")[-1].strip()
        if "：" in user_message:
            user_message = user_message.split("：")[-1].strip()
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

# 与标准答案的相似度奖励函数


def reward_similarity(completions, prompts=None, answer=None, **kwargs):
    global semantic_model
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
        if ">:" in response:
            response = response.split(">:")[-1].strip()
        if "：" in response:
            response = response.split("：")[-1].strip()
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
    return rewards
