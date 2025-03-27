import os
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_base(
    base_model_path,
    lora_model_path,
    alpha=1.0,
    output_precision=None,
    device_map="auto",
    output_dir=None,
    offload_folder="tmp_offload"  # 修改：默认值不为None，而是一个默认路径
):
    """
    将LoRA模型合并到基础模型

    Args:
        base_model_path: 基础模型的路径
        lora_model_path: LoRA模型的路径
        alpha: LoRA权重合并比例，默认为1.0
        output_precision: 输出模型的精度，可选值为 "fp16", "bf16" 或 None (保持原精度)
        device_map: 模型加载的设备映射策略
        output_dir: 输出目录路径
        offload_folder: 模型卸载目录路径，为了支持自动设备映射必须提供

    Returns:
        合并后模型的保存路径
    """
    start_time = time.time()
    print(f"开始合并过程...")
    print(f"基础模型: {base_model_path}")
    print(f"LoRA模型: {lora_model_path}")
    print(f"合并系数: {alpha}")
    print(f"输出精度: {output_precision if output_precision else '原始精度'}")
    print(f"设备映射: {device_map}")
    print(f"卸载目录: {offload_folder}")
    print("-" * 50)

    # 始终确保offload_folder存在
    if offload_folder:
        os.makedirs(offload_folder, exist_ok=True)
        print(f"创建或确认卸载目录: {offload_folder}")

    # 准备模型加载配置
    model_kwargs = {
        "device_map": device_map,
        "offload_folder": offload_folder,  # 总是提供offload_folder
    }

    # 配置精度选项
    if output_precision == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    elif output_precision == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = "auto"

    try:
        print(f"正在加载基础模型: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
    except Exception as e:
        print(f"加载基础模型时出错: {str(e)}")
        raise

    try:
        print(f"正在加载LoRA模型: {lora_model_path}")
        # 确保为PeftModel提供必要的参数
        peft_kwargs = {
            "device_map": device_map,
            "offload_dir": offload_folder,  # 总是提供offload_dir
        }
        
        model = PeftModel.from_pretrained(base_model, lora_model_path, **peft_kwargs)
    except Exception as e:
        print(f"加载LoRA模型时出错: {str(e)}")
        raise

    try:
        print(f"正在合并LoRA权重到基础模型 (alpha={alpha})...")
        # 如果提供了自定义合并比例
        if alpha != 1.0:
            model = model.merge_and_unload(alpha=alpha)
        else:
            model = model.merge_and_unload()
    except Exception as e:
        print(f"合并模型时出错: {str(e)}")
        raise

    # 获取LoRA模型的名称用作输出文件夹名
    lora_model_name = os.path.basename(os.path.normpath(lora_model_path))

    # 获取基础模型所在的目录
    base_model_dir = os.path.dirname(os.path.normpath(base_model_path))

    # 合并后的模型输出路径
    if output_dir is None:
        output_dir = os.path.join(base_model_dir, lora_model_name)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"正在保存合并后的模型到: {output_dir}")
        # 保存模型
        model.save_pretrained(
            output_dir,
            safe_serialization=True,  # 使用safetensors格式保存
        )

        # 同时保存tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)

        # 保存模型配置文件
        if hasattr(model, 'config'):
            # 记录合并信息
            model.config.merged_from_lora = True
            model.config.lora_model_path = lora_model_path
            model.config.base_model_path = base_model_path
            model.config.alpha = alpha
            model.config.merged_at = time.strftime("%Y-%m-%d %H:%M:%S")

            # 保存修改后的配置
            model.config.save_pretrained(output_dir)

    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        raise

    end_time = time.time()
    print("-" * 50)
    print(f"合并完成! 模型已保存到: {output_dir}")
    print(f"合并过程耗时: {end_time - start_time:.2f} 秒")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA模型合并工具")
    parser.add_argument("--base_model_path", type=str,
                        help="基础模型的路径", 
                        default="output/mimibot_l3_v1.0")
    parser.add_argument("--lora_model_path", type=str,
                        help="LoRA模型的路径", default="results/mimibot_l3/checkpoint-1000")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LoRA权重合并比例，默认为1.0")
    parser.add_argument("--output_dir", type=str,
                        help="合并后模型的保存路径", 
                        default="output/mimibot_l3_v1.1")
    parser.add_argument("--output_precision", type=str, 
                        default="none",
                        choices=["fp16", "bf16", "none"],
                        help="输出模型的精度，可选值为fp16, bf16或none(保持原精度)")
    parser.add_argument("--device_map", type=str, default="cuda",
                        help="模型加载的设备映射策略")
    parser.add_argument("--offload_folder", type=str, default="tmp_offload",    
                        help="模型卸载目录的路径，用于处理大模型(必须提供)")

    args = parser.parse_args()
    
    # 始终确保offload_folder存在
    os.makedirs(args.offload_folder, exist_ok=True)
    print(f"使用卸载目录: {args.offload_folder}")
        
    # 处理"none"字符串为None对象
    output_precision = None if args.output_precision == "none" else args.output_precision

    merge_lora_to_base(
        args.base_model_path,
        args.lora_model_path,
        alpha=args.alpha,
        output_precision=output_precision,
        device_map=args.device_map,
        output_dir=args.output_dir,
        offload_folder=args.offload_folder
    )
