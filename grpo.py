from datasets import load_dataset, Dataset
from prompt_utils import parse_prompt
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import os
import json
import argparse
import sentence_transformers
import re
import sys
import datetime
import logging
from transformers import AutoModel, AutoTokenizer
# 导入自定义工具模块
from utils import setup_logging, cleanup_logging, setup_reward_logger, apply_template
# 导入奖励函数
from grpo_reward import (
    init_semantic_model,
    reward_format,
    reward_no_repetition,
    reward_similarity,
    reward_user_similarity,
    reward_len_response,
    reward_EXTERNAL,
    init_external_reward_server,
    shutdown_external_reward_server,
    init_user_similarity_logger
)
# 导入OpenAI兼容API服务器
from api_server.openai_compatible_server import OpenAICompatibleCallback, OpenAICompatibleServer

if __name__ == "__main__":
    # 设置日志记录
    logging_context = setup_logging()

    # 记录脚本开始运行
    print(f"GRPO训练脚本开始运行")

    # 初始化用户相似度奖励的日志记录器
    init_user_similarity_logger(logging_context["timestamp"])

    parser = argparse.ArgumentParser(description='GRPO训练脚本')
    parser.add_argument('--dataset_type', type=str, default='alpaca', choices=['tldr', 'alpaca'],
                        help='数据集类型: tldr或alpaca')
    parser.add_argument('--alpaca_path', type=str, default='data/grpo_sorted.json',
                        help='Alpaca数据集路径')
    parser.add_argument('--semantic_model_path', type=str,
                        default='shibing624/text2vec-base-chinese',
                        help='Sentence Transformer模型路径')
    parser.add_argument('--load_lora_path', type=str,
                        default=None,
                        help='要加载的LoRA模型路径（用于继续训练）')
    # 添加API服务器相关参数
    parser.add_argument('--enable_api', type=bool, default=True,
                        help='启用OpenAI兼容的API服务器')
    parser.add_argument('--api_port', type=int, default=8099,
                        help='API服务器端口')
    parser.add_argument('--api_host', type=str, default='0.0.0.0',
                        help='API服务器主机地址')
    # 添加外部奖励服务器参数
    parser.add_argument('--external_reward_port', type=int, default=5678,
                        help='外部奖励服务器端口')
    args = parser.parse_args()

    # 加载语义相似度模型并初始化奖励模块
    semantic_model_path = args.semantic_model_path
    if not init_semantic_model(semantic_model_path):
        print("语义模型初始化失败，程序退出")
        exit(1)
        
    # 初始化外部奖励服务器
    print(f"初始化外部奖励服务器，端口: {args.external_reward_port}")
    if not init_external_reward_server(args.external_reward_port):
        print("警告: 外部奖励服务器初始化失败，但程序将继续运行")

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
    # MODEL_PATH = 'output/mimibot_tifa_v1.2'
    # MODEL_PATH = 'results/mimibot_tifa/checkpoint-1500'
    # MODEL_PATH = 'output/mimibot_tifa_v3.0'
    # MODEL_PATH = 'models/Tifa-DeepsexV2-7b-Cot-0317-F16'
    # MODEL_PATH = 'output/mimibot_tifa_v3.6'
    MODEL_PATH = 'output/mimibot_l3_v1.1'

    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 64
    max_data_length = 4096 # 1k examples

    print(f"加载模型: {MODEL_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        load_in_4bit=False,
        load_in_8bit=False,  # False for LoRA 16bit
        max_seq_length=max_seq_length,
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 加载Alpaca格式数据集
    def load_alpaca_dataset(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[:max_data_length]

        processed_data = {
            "prompt": [],
            "completion": []
        }

        if "prompt" in data[0] and "completion" in data[0]:
            # 如果数据集已经是prompt-completion格式，直接返回
            for item in data:
                processed_data["prompt"].append(item["prompt"])
                processed_data["completion"].append(item["completion"])
            return Dataset.from_dict(processed_data)

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
    def get_promt_dataset(dataset) -> Dataset:
        # 数据集已经是训练集，不需要再次拆分
        if "prompt" in dataset[0] and "answer" in dataset[0]:
            return dataset
        data = dataset  # type: ignore

        # 使用apply_template函数处理数据
        def transform_example(x):
            formatted_prompt = apply_template(
                x['prompt'], tokenizer, SYSTEM_PROMPT) + "<think>"
            return {
                "prompt": formatted_prompt,
                "answer": x['completion'],
            }

        data = data.map(transform_example)  # type: ignore
        return data  # type: ignore

    dataset = get_promt_dataset(dataset=dataset)
    # 打印一个 dataset 例子
    print(dataset[0])

    # 使用GRPOConfig而非TrainingArguments
    training_args = GRPOConfig(
        # learning_rate=1e-4,
        # learning_rate=3e-5, # 1epoch 后训飞
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.01,
        output_dir="./results/mimibot_l3",  # 添加必要的output_dir参数
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_generations=8,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        max_steps=2000,
        max_grad_norm=0.1,
        max_completion_length=512,
        # reward_weights=[1e-3, 1e-3, 1e-3, 1e-3] # normalize to 1 not need
        beta=0.1,  # 由 0.04 增大到 0.1 保留更多原模型能力
    )

    # 创建回调列表
    callbacks = []

    def get_trainer():
        return trainer

    # 如果启用API服务器，添加OpenAI兼容API回调
    if args.enable_api:
        print(f"启用OpenAI兼容API服务器: 端口={args.api_port}, 主机={args.api_host}")
        api_callback = OpenAICompatibleCallback(
            port=args.api_port,
            simulation_mode=False,
            trainer_getter=get_trainer,
            system_prompt=SYSTEM_PROMPT,  # 传入系统提示
            tokenizer=tokenizer,
        )
        callbacks.append(api_callback)

    # 使用导入的奖励函数
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_format,
            reward_no_repetition,
            reward_similarity,
            reward_len_response,
            reward_EXTERNAL,
        ],
        train_dataset=dataset,
        args=training_args,
        callbacks=callbacks  # 添加回调列表
    )

    # 开始训练
    trainer.train()

    # 关闭外部奖励服务器
    print("关闭外部奖励服务器...")
    shutdown_external_reward_server()

    # 清理日志设置
    cleanup_logging(logging_context)
