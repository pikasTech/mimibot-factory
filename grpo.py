from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os

if __name__ == "__main__":
    # 设置环境变量以避免内存碎片
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    SYSTEM_PROMPT = """
    你是一个有感情的人类
    <user>用户提问</user>
    回复模板:
    <think>
    你的自主思考
    </think>
    你的回答...
    
    """


    # MODEL_PATH = "/root/autodl-fs/models/Tifa-DeepsexV2-7b-Cot-0317-F16"
    # MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"
    # MODEL_PATH = "/root/autodl-tmp/models/Qwen2-0.5B-Instruct"
    MODEL_PATH = "/root/autodl-fs/models/Llama3.1-8B-Chinese-Chat"
    # MODEL_PATH = "/root/autodl-fs/models/Tifa-DeepsexV2-7b-Cot-0317-F16"

    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        load_in_4bit = False,
        load_in_8bit = False, # False for LoRA 16bit
        max_seq_length = max_seq_length,
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )


    dataset = load_dataset("trl-lib/tldr", split="train")
    # uncomment middle messages for 1-shot prompting
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
        # print(prompt)
        return {
            "prompt": prompt,
            "completion": x['completion'],
        }
    
    def get_promt_dataset(dataset) -> Dataset:
        # 数据集已经是训练集，不需要再次拆分
        data = dataset # type: ignore
        data = data.map(apply_template) # type: ignore
        return data # type: ignore

    dataset = get_promt_dataset(dataset=dataset)
    
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
        # return [-abs(100 - len(c)) for c in completions]
        # 奖励中文回答，惩罚英文回答
        rewared = []
        for c in completions:
            # c = c.split("</think>")[-1]
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

    def reward_one_line(completions, **kwargs):
        rewared = []
        print(completions[0])
        print('-' * 10)
        print(completions[1])
        print('-' * 10)
        print(completions[2])
        print('-' * 10)
        for c in completions:
            c1 = c.count("<think>")
            c2 = c.count("</think>")
            res = 1 if c1 == 1 and c2 == 1 else min(0.1 * (c1 + c2), 0.4)
            rewared.append(res * 100)
        return rewared


    # 使用GRPOConfig而非TrainingArguments
    training_args = GRPOConfig(
        learning_rate=1e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.01,
        output_dir="./results",  # 添加必要的output_dir参数
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        per_device_train_batch_size=12,
        gradient_accumulation_steps=1,
        num_generations=12,
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        max_steps=1000,
        max_grad_norm = 0.1,
        max_completion_length = 1024
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_num_unique_chars, reward_one_line],
        train_dataset=dataset,
        args=training_args
    )
    trainer.train()
