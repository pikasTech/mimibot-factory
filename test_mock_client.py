import socket
import json
import time
import argparse
import random

def run_mock_client(host='localhost', port=5678, client_id=1, disconnect_after=None):
    """
    运行一个模拟的外部奖励客户端
    Args:
        host: 服务器主机名
        port: 服务器端口
        client_id: 客户端ID
        disconnect_after: 处理多少请求后断开
    """
    try:
        # 连接到服务器
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        
        # 接收欢迎消息
        welcome = client.recv(1024).decode('utf-8')
        welcome_data = json.loads(welcome)
        print(f"收到欢迎消息: {welcome_data}")
        
        print(f"模拟客户端 {client_id} 已连接到 {host}:{port}")
        
        # 计数器，用于可选的断开连接测试
        counter = 0
        
        # 持续接收和响应请求
        while True:
            try:
                # 接收请求
                request = ""
                while True:
                    chunk = client.recv(4096).decode('utf-8')
                    if not chunk:
                        print("连接已关闭")
                        return  # 连接已关闭
                    
                    request += chunk
                    if '\n' in request:
                        break
                
                # 解析请求
                request_data = json.loads(request)
                
                # 检查请求类型
                if request_data["type"] != "reward_request":
                    print(f"收到未知请求类型: {request_data['type']}")
                    continue
                
                # 处理奖励请求
                completions = request_data["data"]["completions"]
                responses = request_data["data"]["responses"]
                prompts = request_data["data"]["prompts"]
                answers = request_data["data"]["answers"]
                
                print(f"收到请求，包含 {len(completions)} 个回复")
                
                # 模拟奖励计算
                # 在实际应用中，这里可以根据responses和answers进行自定义比较
                # 这里简单地生成随机奖励，并对较长回复给予较高奖励
                rewards = []
                for i, (resp, comp) in enumerate(zip(responses, completions)):
                    # 基础奖励 (随机成分)
                    base_reward = random.uniform(10, 50)
                    
                    # 长度奖励 (较长的回复获得更高奖励)
                    length_factor = min(len(resp) / 100, 2.0)
                    
                    # 思考过程奖励
                    thinking_bonus = 20 if "<think>" in comp and "</think>" in comp else 0
                    
                    # 如果有标准答案，根据长度相似度给予奖励
                    ans_bonus = 0
                    if i < len(answers):
                        ans_length = len(answers[i])
                        resp_length = len(resp)
                        # 长度差异比例 (值越小越好)
                        length_diff_ratio = abs(ans_length - resp_length) / max(ans_length, 1)
                        # 长度相似奖励
                        ans_bonus = 30 * (1 - min(length_diff_ratio, 1))
                    
                    # 总奖励
                    total_reward = base_reward + (length_factor * 20) + thinking_bonus + ans_bonus
                    rewards.append(round(total_reward, 2))
                
                print(f"计算的奖励值: {rewards}")
                
                # 发送响应
                response = {
                    "rewards": rewards
                }
                client.send((json.dumps(response) + "\n").encode('utf-8'))
                
                # 递增计数器
                counter += 1
                print(f"已处理 {counter} 个请求")
                
                # 如果设置了断开连接条件并达到条件，则断开
                if disconnect_after and counter >= disconnect_after:
                    print(f"模拟客户端 {client_id} 已达到计数 {counter}，准备断开")
                    break
                
            except Exception as e:
                print(f"处理请求异常: {str(e)}")
                break
    except Exception as e:
        print(f"连接异常: {str(e)}")
    finally:
        try:
            client.close()
        except:
            pass
        print(f"模拟客户端 {client_id} 已断开连接")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行模拟的外部奖励客户端")
    parser.add_argument("--host", default="localhost", help="服务器主机名")
    parser.add_argument("--port", type=int, default=5678, help="服务器端口")
    parser.add_argument("--client-id", type=int, default=1, help="客户端ID")
    parser.add_argument("--disconnect-after", type=int, help="处理多少请求后断开")
    
    args = parser.parse_args()
    
    run_mock_client(
        host=args.host, 
        port=args.port, 
        client_id=args.client_id, 
        disconnect_after=args.disconnect_after
    )
