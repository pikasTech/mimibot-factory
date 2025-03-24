from datasets import load_dataset, Dataset
from utils import parse_prompt
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os
import json
import argparse
import sentence_transformers
import re
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRPO训练脚本')
    parser.add_argument('--dataset_type', type=str, default='alpaca', choices=['tldr', 'alpaca'], 
                        help='数据集类型: tldr或alpaca')
    parser.add_argument('--alpaca_path', type=str, default='data/alpaca_data_reward_sort.json', 
                        help='Alpaca数据集路径')
    parser.add_argument('--semantic_model_path', type=str, 
                        default='shibing624/text2vec-base-chinese',  # 修改默认值为HF模型ID
                        help='Sentence Transformer模型路径')
    args = parser.parse_args()

    # 加载语义相似度模型
    semantic_model_path = args.semantic_model_path
    print(f"正在加载语义模型: {semantic_model_path}")
    
    # 检查是否为本地路径
    is_local_path = os.path.exists(semantic_model_path) and os.path.isdir(semantic_model_path)
    
    try:
        if is_local_path:
            print(f"检测到本地模型路径: {semantic_model_path}")
        else:
            # 使用Hugging Face Hub加载
            print(f"正在尝试从Hugging Face Hub加载语义模型: {semantic_model_path}")
        semantic_model = sentence_transformers.SentenceTransformer(semantic_model_path)
        print(f"语义模型加载成功: {semantic_model_path}")
    except Exception as e:
        print(f"加载模型失败，尝试使用默认方式: {str(e)}")
        # 回退到默认的Hugging Face模型
        fallback_model = 'sentence-transformers/all-MiniLM-L6-v2'
        semantic_model = sentence_transformers.SentenceTransformer(fallback_model)
        print(f"已加载默认模型: {fallback_model}")

    # 设置环境变量以避免内存碎片
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    SYSTEM_PROMPT = """
    你是mimi波特，你要回复群聊，带上你的思考。【示例输出】\n<think>我看到群友在聊...所以我想回复...</think>你的回复 
    """

    # MODEL_PATH = "/root/autodl-fs/models/Tifa-DeepsexV2-7b-Cot-0317-F16"
    # MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"
    # MODEL_PATH = "/root/autodl-tmp/models/Qwen2-0.5B-Instruct"
    # MODEL_PATH = "/root/autodl-fs/models/Llama3.1-8B-Chinese-Chat"
    # MODEL_PATH = "/root/autodl-fs/models/Tifa-DeepsexV2-7b-Cot-0317-F16"
    # MODEL_PATH = '/root/autodl-fs/models/mimibot_tifa_v1.2'
    # MODEL_PATH = '/root/autodl-fs/models/mimibot_l3_v0.9'
    MODEL_PATH = 'output/mimibot_tifa_v1.2'

    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower
    max_data_length = 1024

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        load_in_4bit = False,
        load_in_8bit = False, # False for LoRA 16bit
        max_seq_length = max_seq_length,
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )

    # 加载Alpaca格式数据集
    def load_alpaca_dataset(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[:max_data_length]
        
        processed_data = {
            "prompt": [],
            "completion": []
        }
        
        for item in data:
            # 处理历史对话
            history_text = ""
            if item["history"]:
                for turn in item["history"]:
                    if turn[0] and turn[1]:
                        history_text += f"{turn[0]}\n{turn[1]}\n"
                    elif turn[0]:
                        history_text += f"{turn[0]}\n"
            
            # 构建提示
            prompt = "【任务目标】\n你是mimi波特，回复最新消息，禁止重复历史消息\n\n【示例输出】\n<think>我看到群友在聊...所以我想回复...</think>你的回复\n\n"
            if history_text:
                prompt += f"【历史消息】\n{history_text}\n\n"
            
            prompt += "【最新消息】\n" + f"{item['input']}"
            
            # 提取输出
            completion = item["output"]
            
            processed_data["prompt"].append(prompt)
            processed_data["completion"].append(completion)
        
        return Dataset.from_dict(processed_data)

    # 根据选择加载不同的数据集
    if args.dataset_type == 'tldr':
        dataset = load_dataset("trl-lib/tldr", split="train")
    else:  # alpaca
        dataset = load_alpaca_dataset(args.alpaca_path)

    # 应用模板的函数，处理不同格式的数据
    def apply_template(x):
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x['prompt']}
            ]
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt + "<think>",
            "answer": x['completion'],
        }
    
    def get_promt_dataset(dataset) -> Dataset:
        # 数据集已经是训练集，不需要再次拆分
        data = dataset # type: ignore
        data = data.map(apply_template) # type: ignore
        return data # type: ignore

    dataset = get_promt_dataset(dataset=dataset)
    # 打印一个 dataset 例子
    print(dataset[0])
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    # Dummy reward function: count the number of unique characters in the completions
    def reward_num_unique_chars(completions, **kwargs):
        # 奖励中文回答，惩罚英文回答
        rewared = []
        for c in completions:
            c = c.replace("<think>", "")
            c = c.replace("</think>", "")
            count_en = 0
            count_zh = 0
            res = c.split("</think>")[-1]
            for char in c:
                if char >= u'\u4e00' and char <= u'\u9fa5':
                    count_zh += 1
                elif char.isalpha():
                    count_en += 1
            ans = - count_en - abs(250 - count_zh) - abs(270 - len(c)) - abs(50 - len(res)) # 惩罚英文回答，奖励长度为55的中文回答，奖励总长度为55的回答
            rewared.append(ans)
        return rewared

    def reward_format(completions, **kwargs):
        rewared = []
        for c in completions:
            c1 = c.count("<think>")
            c2 = c.count("</think>")
            # res = 1 if c1 == 1 and c2 == 1 else min(0.1 * (c1 + c2), 0.4)
            res = 1 if c2 == 1 else min(0.1 * (c2), 0.4)
            if "</think>" in c:
                ans = c.split("</think>")[-1]
            else:
                ans = c
            
            # c_not_true = ans.count('<') + ans.count('>') + ans.count(':') + ans.count('：') + ans.count('【') + ans.count('】') + ans.count('?') + ans.count('？') + ans.count('\n') + ans.count(' ') + ans.count('\r') + ans.count('\t') + ans.count('吗')

            # res -= 0.1 * c_not_true 

            # 惩罚非中文或非ASCII字符
            # for char in ans:
            #     if not (char >= u'\u4e00' and char <= u'\u9fa5') and not char.isascii():
            #         res -= 0.1

            rewared.append(res * 100)
        return rewared

    # 新增重复惩罚奖励函数
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

            if "</think>" in completion:
                response = completion.split("</think>")[-1]
            else:
                response = completion

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
            all_messages = [message[message.find(":")+1:].strip() for message in all_messages]

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

    # 新增用户相似度奖励函数
    def reward_user_similarity(completions, prompts=None, **kwargs):
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
            if "</think>" in completion:
                response = completion.split("</think>")[-1]
            else:
                response = completion
                
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

    def reward_similarity(completions, prompts=None, answer=None,**kwargs):
        use_debug = True
        responses = []
        for completion in completions:
            if "</think>" in completion:
                response = completion.split("</think>")[-1]
            else:
                response = completion
            responses.append(response)
        answers = []
        for ans in answer:
            if "</think>" in ans:
                response = ans.split("</think>")[-1]
            else:
                response = ans
            if ">:" in response:
                response = response.split(">:")[-1].strip()
            if "：" in response:
                response = response.split("：")[-1].strip()
            answers.append(response)
        rewards = []
        similarity = sentence_transformers.util.cos_sim(semantic_model.encode(responses), semantic_model.encode(answers))
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
            best_indexs = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)[:4]
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

    # 使用GRPOConfig而非TrainingArguments
    training_args = GRPOConfig(
        learning_rate= 3e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.001,
        output_dir="./results/mimibot_tifa",  # 添加必要的output_dir参数
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_generations=4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        max_steps=10000,
        max_grad_norm = 0.1,
        max_completion_length = 256
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_no_repetition, reward_similarity],
        train_dataset=dataset,
        args=training_args
    )
    trainer.train()
