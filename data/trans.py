import json

def process_dataset(input_path, output_path):
    alpaca_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # 直接使用原始问答内容，不做任何处理
            alpaca_data.append({
                "instruction": data["question"],
                "input": "",
                "output": data["answer"]
            })
    
    # 写入目标文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

# 使用示例
process_dataset('Beautiful-Chinese.jsonl', 'alpaca_Beautiful-Chinese.json')