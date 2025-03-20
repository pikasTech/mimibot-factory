from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os

# 设置环境变量以避免内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

dataset = load_dataset("trl-lib/tldr", split="train")

# 加载模型和tokenizer，使用bf16精度加载模型以减少内存使用
model, tokenizer = (
    AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    ),
    AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
)

# 配置LoRA - 降低参数以减少内存需求
lora_config = LoraConfig(
    r=8,  # 降低秩以减少内存使用
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

# 使用GRPOConfig而非TrainingArguments
training_args = GRPOConfig(
    output_dir="./results",  # 添加必要的output_dir参数
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    fp16=True,
    optim="adamw_torch",
    logging_steps=1,
    save_strategy="steps",
    save_steps=500,
    learning_rate=5e-5,
    max_steps=1000,
    warmup_steps=100,
    num_generations=2
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,  # 为PeftModel添加tokenizer
    compute_metrics=None,  # 明确设置compute_metrics为None
)
trainer.train()
