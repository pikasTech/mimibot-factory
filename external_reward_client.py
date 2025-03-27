"""
外部奖励客户端工具
在实际训练过程中使用此工具可以提供自定义的外部奖励
"""
import re
import socket
import json
import time
import argparse
import threading
import sys
import signal
import os
# 导入OpenAI评估器
from openai_evaluator import OpenAIEvaluator

# 全局变量
running = True
connected = False
processed_requests = 0

def signal_handler(sig, frame):
    """处理中断信号"""
    global running
    print("\n检测到中断，正在关闭客户端...")
    running = False
    sys.exit(0)

def calculate_reward(responses, prompts_origin=None, answers=None):
    """
    计算自定义奖励 - 使用OpenAI API评估回复质量
    
    Args:
        responses: 模型生成的回复列表
        prompts: 输入提示列表
        answers: 标准答案列表
    
    Returns:
        奖励值列表
    """
    # 创建OpenAI评估器 - 设置批次大小为8
    evaluator = OpenAIEvaluator(batch_size=8)  # 增加到8个回复一批

    prompts = []
    for prompt in prompts_origin:
        user_message = re.findall(r"<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>", prompt, re.DOTALL)[0].strip()
        prompts.append(user_message)
    
    # 检查评估器是否可用
    if not evaluator.is_available():
        print("警告: OpenAI API密钥未设置，将使用备用评分方法")
        return calculate_backup_reward(responses, prompts, answers)
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 按prompt和answer分组
        grouped_data = {}
        for i, response in enumerate(responses):
            prompt = prompts[i] if prompts and i < len(prompts) else "未提供"
            answer = answers[i] if answers and i < len(answers) else None
            
            # 使用(prompt, answer)作为分组键
            group_key = (prompt, str(answer))  # 将answer转为字符串以便作为字典键
            
            if group_key not in grouped_data:
                grouped_data[group_key] = {
                    "prompt": prompt,
                    "answer": answer,
                    "responses": [],
                    "indices": []  # 记录原始索引位置
                }
            
            grouped_data[group_key]["responses"].append(response)
            grouped_data[group_key]["indices"].append(i)
        
        # 记录所有回复的奖励值
        all_rewards = [0] * len(responses)
        
        # 对每组数据分别评估
        print(f"将{len(responses)}个回复分为{len(grouped_data)}个组进行评估")
        
        for group_key, group_data in grouped_data.items():
            group_prompt = group_data["prompt"]
            group_answer = group_data["answer"]
            group_responses = group_data["responses"]
            group_indices = group_data["indices"]
            
            # 只有在有标准答案时才使用相对评分
            if group_answer:
                print(f"评估组({len(group_responses)}个回复)，提示: {group_prompt[:30]}...")
                try:
                    # 使用evaluate_responses批量评估
                    relative_scores, _, _ = evaluator.evaluate_responses(
                        group_prompt, group_responses, group_answer, simplified=True
                    )
                    
                    # 将分数分配到对应的位置
                    for idx, score in zip(group_indices, relative_scores):
                        all_rewards[idx] = score
                except Exception as e:
                    print(f"组评估失败: {str(e)}，使用备用评分方法...")
                    # 使用备用方法评估该组
                    backup_scores = calculate_backup_reward(
                        group_responses, 
                        [group_prompt] * len(group_responses), 
                        [group_answer] * len(group_responses)
                    )
                    # 将备用分数分配到对应的位置
                    for idx, score in zip(group_indices, backup_scores):
                        all_rewards[idx] = score
            else:
                # 无标准答案时使用备用评分
                print(f"组({len(group_responses)}个回复)缺少标准答案，使用备用评分方法...")
                backup_scores = calculate_backup_reward(
                    group_responses, 
                    [group_prompt] * len(group_responses),
                    None
                )
                # 将备用分数分配到对应的位置
                for idx, score in zip(group_indices, backup_scores):
                    all_rewards[idx] = score
        
        # 计算总耗时
        elapsed_time = time.time() - start_time
        print(f"评估 {len(responses)} 个回复总耗时: {elapsed_time:.2f}秒, 平均每个回复: {elapsed_time/len(responses):.2f}秒")
        
        return all_rewards
    except Exception as e:
        print(f"使用OpenAI评估器时出错: {str(e)}")
        print("使用备用评分方法...")
        return calculate_backup_reward(responses, prompts, answers)

def calculate_backup_reward(responses, prompts=None, answers=None):
    """
    备用奖励计算方法 - 当OpenAI API不可用时使用
    """
    rewards = []
    
    for i, response in enumerate(responses):
        # 基础奖励
        reward = 50.0
        
        # 根据回复长度给予奖励
        length_factor = min(len(response) / 100, 2.0)
        reward += length_factor * 20
        
        # 如果有标准答案，可以基于相似度计算奖励
        if answers and i < len(answers):
            answer = answers[i]
            # 简单的相似度计算示例 - 这里仅基于长度差异
            length_diff = abs(len(response) - len(answer))
            similarity = max(0, 1 - (length_diff / max(len(answer), 1, 100)))
            reward += similarity * 50
        
        rewards.append(round(reward, 2))
    
    return rewards

def run_client(host, port, client_id=1):
    """
    运行奖励客户端
    
    Args:
        host: 服务器主机名
        port: 服务器端口
        client_id: 客户端ID
    """
    global running, connected, processed_requests
    
    print(f"启动外部奖励客户端 #{client_id}...")
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    client = None
    retry_count = 0
    max_retries = 10
    
    while running and retry_count < max_retries:
        try:
            # 连接到服务器
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(5.0)
            print(f"尝试连接到服务器 {host}:{port}...")
            client.connect((host, port))
            
            # 接收欢迎消息
            welcome = client.recv(1024).decode('utf-8').strip()
            print(f"收到欢迎消息: {welcome}")
            
            connected = True
            print(f"已连接到服务器 {host}:{port}")
            
            # 重置重试计数
            retry_count = 0
            
            # 处理请求循环
            while running:
                try:
                    # 接收请求
                    request_data = ""
                    client.settimeout(None)  # 无限等待请求
                    
                    while running:
                        chunk = client.recv(4096).decode('utf-8')
                        if not chunk:
                            print("服务器断开连接")
                            connected = False
                            break
                        
                        request_data += chunk
                        if '\n' in request_data:
                            break
                    
                    if not connected:
                        break
                    
                    # 解析请求
                    try:
                        request = json.loads(request_data)
                    except json.JSONDecodeError:
                        print(f"无法解析请求: {request_data}")
                        continue
                    
                    # 处理奖励请求
                    if request["type"] == "reward_request":
                        completions = request["data"]["completions"]
                        responses = request["data"]["responses"]
                        prompts = request["data"]["prompts"]
                        answers = request["data"].get("answers", [])
                        
                        print(f"\n收到奖励请求 #{processed_requests + 1}")
                        print(f"回复数量: {len(responses)}")
                        
                        # 计算奖励
                        rewards = calculate_reward(responses, prompts, answers)
                        
                        # 发送响应
                        response = {"rewards": rewards}
                        client.send((json.dumps(response) + "\n").encode('utf-8'))
                        
                        processed_requests += 1
                        print(f"已发送奖励响应: {rewards}")
                        print(f"已处理请求总数: {processed_requests}")
                    else:
                        print(f"未知请求类型: {request.get('type', '未知')}")
                
                except socket.timeout:
                    # 超时，继续等待
                    continue
                except Exception as e:
                    print(f"处理请求时出错: {str(e)}")
                    connected = False
                    break
            
            # 如果连接断开但程序仍在运行，尝试重新连接
            if running and not connected:
                print("连接断开，正在尝试重新连接...")
                retry_count += 1
                time.sleep(2)  # 等待一段时间后重试
        
        except ConnectionRefusedError:
            print(f"连接被拒绝，服务器可能未启动或端口不正确 ({retry_count+1}/{max_retries})")
            retry_count += 1
            time.sleep(2)
        except socket.timeout:
            print(f"连接超时 ({retry_count+1}/{max_retries})")
            retry_count += 1
            time.sleep(2)
        except Exception as e:
            print(f"连接出错: {str(e)} ({retry_count+1}/{max_retries})")
            retry_count += 1
            time.sleep(2)
        finally:
            if client:
                try:
                    client.close()
                except:
                    pass
    
    if retry_count >= max_retries:
        print(f"达到最大重试次数 ({max_retries})，客户端退出")
    
    print("外部奖励客户端已关闭")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行外部奖励客户端")
    parser.add_argument("--host", default="localhost", help="服务器主机名")
    parser.add_argument("--port", type=int, default=5678, help="服务器端口")
    parser.add_argument("--id", type=int, default=1, help="客户端ID")
    
    args = parser.parse_args()
    
    try:
        # 在主线程中运行客户端
        run_client(args.host, args.port, args.id)
    except KeyboardInterrupt:
        print("\n检测到中断，正在关闭客户端...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
