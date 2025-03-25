import requests
import json
import argparse
import time

def test_chat_completions(base_url="http://localhost:8000/v1", model="default-model"):
    """测试聊天完成API"""
    url = f"{base_url}/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个有用的AI助手。"
            },
            {
                "role": "user",
                "content": "请简要介绍人工智能的发展历史。"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    print(f"发送请求到 {url}...")
    start_time = time.time()
    
    response = requests.post(url, headers=headers, json=data)
    
    elapsed_time = time.time() - start_time
    print(f"请求耗时: {elapsed_time:.2f}秒")
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n响应内容:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n生成的文本:")
        print(result["choices"][0]["message"]["content"])
        
        print("\n令牌统计:")
        print(f"输入令牌: {result['usage']['prompt_tokens']}")
        print(f"输出令牌: {result['usage']['completion_tokens']}")
        print(f"总令牌数: {result['usage']['total_tokens']}")
    else:
        print("错误:", response.text)

def test_completions(base_url="http://localhost:8000/v1", model="default-model"):
    """测试文本完成API"""
    url = f"{base_url}/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "prompt": "讲解一下量子计算的基本原理：",
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    print(f"发送请求到 {url}...")
    start_time = time.time()
    
    response = requests.post(url, headers=headers, json=data)
    
    elapsed_time = time.time() - start_time
    print(f"请求耗时: {elapsed_time:.2f}秒")
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n响应内容:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n生成的文本:")
        print(result["choices"][0]["text"])
        
        print("\n令牌统计:")
        print(f"输入令牌: {result['usage']['prompt_tokens']}")
        print(f"输出令牌: {result['usage']['completion_tokens']}")
        print(f"总令牌数: {result['usage']['total_tokens']}")
    else:
        print("错误:", response.text)

def test_models(base_url="http://localhost:8000/v1"):
    """测试模型列表API"""
    url = f"{base_url}/models"
    
    print(f"发送请求到 {url}...")
    
    response = requests.get(url)
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n可用模型:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("错误:", response.text)

def main():
    parser = argparse.ArgumentParser(description='测试OpenAI兼容API')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000/v1', help='API基础URL')
    parser.add_argument('--model', type=str, default='default-model', help='使用的模型名称')
    parser.add_argument('--api', type=str, choices=['chat', 'completion', 'models', 'all'], 
                        default='all', help='要测试的API类型')
    
    args = parser.parse_args()
    
    if args.api == 'chat' or args.api == 'all':
        print("\n===== 测试聊天完成API =====")
        test_chat_completions(args.base_url, args.model)
    
    if args.api == 'completion' or args.api == 'all':
        print("\n===== 测试文本完成API =====")
        test_completions(args.base_url, args.model)
    
    if args.api == 'models' or args.api == 'all':
        print("\n===== 测试模型列表API =====")
        test_models(args.base_url)

if __name__ == "__main__":
    main()