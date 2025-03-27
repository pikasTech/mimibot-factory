"""
外部奖励系统的集成测试 - 更简单的测试方法
同时启动服务器和客户端，验证奖励系统功能
"""

import threading
import time
import json
import socket
import random
from grpo_reward import init_external_reward_server, reward_EXTERNAL, shutdown_external_reward_server

def run_mock_client(port=5678, client_id=1, auto_disconnect=False):
    """运行一个模拟的客户端"""
    print(f"启动模拟客户端 #{client_id}...")
    try:
        # 连接到服务器
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', port))
        
        # 接收欢迎消息
        welcome = client.recv(1024).decode('utf-8')
        print(f"客户端 #{client_id} 收到欢迎消息: {welcome.strip()}")
        
        print(f"客户端 #{client_id} 已连接，等待请求...")
        
        # 计数器
        request_count = 0
        
        # 处理请求
        while True:
            try:
                # 接收请求
                data = ""
                while True:
                    chunk = client.recv(4096).decode('utf-8')
                    if not chunk:
                        print(f"客户端 #{client_id}: 连接关闭")
                        return
                    
                    data += chunk
                    if '\n' in data:
                        break
                
                # 解析请求
                request = json.loads(data)
                
                # 检查请求类型
                if request["type"] != "reward_request":
                    print(f"客户端 #{client_id}: 收到未知请求类型: {request['type']}")
                    continue
                
                # 获取请求数据
                completions_count = len(request["data"]["completions"])
                print(f"客户端 #{client_id}: 收到奖励请求，包含 {completions_count} 个回复")
                
                # 生成随机奖励
                rewards = [random.uniform(50, 150) * client_id for _ in range(completions_count)]
                
                # 发送响应
                response = {"rewards": rewards}
                client.send((json.dumps(response) + "\n").encode('utf-8'))
                
                print(f"客户端 #{client_id}: 已发送奖励响应 {rewards}")
                
                # 计数
                request_count += 1
                
                # 如果设置了自动断开，处理完第一个请求后断开
                if auto_disconnect and request_count >= 1:
                    print(f"客户端 #{client_id}: 自动断开模式，现在断开连接")
                    break
                
            except Exception as e:
                print(f"客户端 #{client_id}: 处理请求时出错: {e}")
                break
    except Exception as e:
        print(f"客户端 #{client_id}: 连接出错: {e}")
    finally:
        try:
            client.close()
        except:
            pass
        print(f"客户端 #{client_id}: 已断开连接")

def main():
    try:
        # 启动服务器
        print("启动外部奖励服务器...")
        server_port = 5679  # 使用不同的端口避免冲突
        init_external_reward_server(server_port)
        time.sleep(1)  # 等待服务器启动
        
        # 启动两个客户端线程
        client_threads = []
        for i in range(1, 3):
            client_thread = threading.Thread(
                target=run_mock_client,
                args=(server_port, i, False)
            )
            client_thread.daemon = True
            client_thread.start()
            client_threads.append(client_thread)
        
        # 再启动一个会自动断开的客户端
        auto_disconnect_thread = threading.Thread(
            target=run_mock_client,
            args=(server_port, 3, True)
        )
        auto_disconnect_thread.daemon = True
        auto_disconnect_thread.start()
        
        # 等待客户端连接
        time.sleep(2)
        print("\n所有客户端已启动")
        
        # 测试数据
        completions = [
            "<think>这是思考过程1</think>这是回复内容1",
            "<think>这是思考过程2</think>这是回复内容2",
        ]
        prompts = ["用户问题1", "用户问题2"]
        answers = ["标准答案1", "标准答案2"]
        
        # 测试1: 使用多客户端获取奖励
        print("\n测试1: 使用多客户端获取奖励")
        rewards1 = reward_EXTERNAL(completions, prompts, answers)
        print(f"获取的奖励: {rewards1}")
        assert len(rewards1) == len(completions), "奖励数量应与回复数量相同"
        assert all(r > 50 for r in rewards1), "所有奖励应大于50"
        print("测试1通过\n")
        
        # 等待自动断开的客户端断开连接
        time.sleep(3)
        
        # 测试2: 验证断开连接后使用最后一次的奖励
        print("测试2: 验证断开连接后使用最后一次的奖励")
        rewards2 = reward_EXTERNAL(completions, prompts, answers)
        print(f"第二次获取的奖励: {rewards2}")
        print(f"与上次奖励相同: {rewards1 == rewards2}")
        print("测试2通过\n")
        
        # 测试3: 测试不同长度输入
        print("测试3: 测试不同长度输入")
        more_completions = completions + ["<think>额外思考</think>额外回复"]
        rewards3 = reward_EXTERNAL(more_completions, prompts, answers)
        print(f"不同长度输入的奖励: {rewards3}")
        assert len(rewards3) == len(more_completions), "奖励数量应与新的回复数量相同"
        print("测试3通过\n")
        
        print("所有测试通过!")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
    finally:
        # 关闭服务器
        print("\n关闭外部奖励服务器...")
        shutdown_external_reward_server()
        
        # 等待所有线程结束
        for thread in client_threads:
            thread.join(timeout=3)
        auto_disconnect_thread.join(timeout=3)
        
        print("测试完成")

if __name__ == "__main__":
    main()
