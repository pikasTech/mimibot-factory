#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="与本地Hugging Face模型对话")
    parser.add_argument("--model_path", type=str, required=True, help="本地Hugging Face模型的路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="运行模型的设备 (cuda/cpu)")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大token数")
    parser.add_argument("--use_chat_template", action="store_true", default=True, 
                      help="使用模型的聊天模板（如果有）")
    parser.add_argument("--load_in_8bit", action="store_true", help="使用8位量化加载模型")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4位量化加载模型")
    parser.add_argument("--system_prompt", type=str, default="你是一个有帮助的AI助手。",
                      help="系统提示词")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"正在从 {args.model_path} 加载模型")
    print(f"使用设备: {args.device}")
    
    # 设置量化配置
    quantization_config = None
    if args.load_in_8bit or args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            if args.load_in_8bit:
                print("以8位精度加载模型")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif args.load_in_4bit:
                print("以4位精度加载模型")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
        except ImportError:
            print("警告: bitsandbytes不可用，使用默认精度")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        
        # 如果不存在则添加padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.unk_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=args.device,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        # 将模型设置为评估模式
        model.eval()
        
        print("模型加载成功。开始聊天会话...")
        print("输入 'exit'、'quit' 或 'q' 结束会话。")
        
        # 检查模型是否有聊天模板
        has_chat_template = hasattr(tokenizer, "apply_chat_template")
        if args.use_chat_template and not has_chat_template:
            print("警告: 模型没有聊天模板，回退到基本提示格式")
        
        # 使用系统提示词初始化对话（如果提供）
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        
        # 聊天循环
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("结束对话。")
                break
            
            # 将用户输入添加到消息中
            messages.append({"role": "user", "content": user_input})
            
            # 准备模型输入
            if args.use_chat_template and has_chat_template:
                # 使用模型的聊天模板（如果有）
                try:
                    inputs = tokenizer.apply_chat_template(
                        messages, 
                        return_tensors="pt",
                        add_generation_prompt=True
                    ).to(args.device)
                except Exception as e:
                    print(f"应用聊天模板时出错: {e}")
                    print("回退到基本提示格式")
                    args.use_chat_template = False
                    has_chat_template = False
            
            if not (args.use_chat_template and has_chat_template):
                # 为没有聊天模板的模型使用基本提示格式
                prompt = ""
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        prompt += f"系统: {content}\n"
                    elif role == "user":
                        prompt += f"用户: {content}\n"
                    elif role == "assistant":
                        prompt += f"助手: {content}\n"
                prompt += "助手: "
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True if args.temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码生成的回复
            if args.use_chat_template and has_chat_template:
                response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            else:
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 提取助手的回复
                if "助手: " in full_response:
                    response_parts = full_response.split("助手: ")
                    response_text = response_parts[-1].strip()
                else:
                    # 备选方案
                    response_text = full_response.replace(prompt, "").strip()
            
            print(f"\n助手: {response_text}")
            
            # 将助手回复添加到消息中
            messages.append({"role": "assistant", "content": response_text})
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
