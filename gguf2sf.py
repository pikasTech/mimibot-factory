"""
GGUF to HuggingFace Converter (v2024.6+)
Requires: transformers>=4.40.0, safetensors
"""

import argparse
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_gguf_to_hf(input_gguf: str, output_dir: str):
    """
    利用Transformers原生支持转换GGUF模型
    """
    # 创建输出目录
    # 分离文件名和路径
    #if '/' in input_gguf:
    #    file_name, path = os.path.split(input_gguf)
    #else:
    file_name = input_gguf
    path = '.'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 加载GGUF模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            gguf_file=file_name
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            gguf_file=file_name
        )

        # 保存为HuggingFace格式
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(
            output_path,
            safe_serialization=True  # 默认使用safetensors
        )

        print(f"转换成功！保存路径: {output_path}")
        # print("??  注意：量化模型转换可能导致精度损失")

    except Exception as e:
        print(f"? 转换失败: {str(e)}")
        if "gguf_file" in str(e):
            print("请确认：\n1. Transformers版本≥4.40\n2. 模型架构受支持")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GGUF转HuggingFace格式工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入的GGUF文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        default="./hf_output",
        help="输出目录路径"
    )

    args = parser.parse_args()

    convert_gguf_to_hf(
        input_gguf=args.input,
        output_dir=args.output
    )


# 示例：python gguf2hf.py -i gguf_model.gguf -o ./hf_output
