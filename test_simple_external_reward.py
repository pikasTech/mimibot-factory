"""
简化的外部奖励系统测试脚本
在同一进程中控制服务器和客户端，避免连接问题
"""

import socket
import threading
import json
import time
import signal
import sys
import random
from contextlib import contextmanager

# 导入需要测试的函数
from grpo_reward import (
    init_external_reward_server, 
    reward_external, 
    shutdown_external_reward_server
)

# 全局变量
MOCK_CLIENT_CONNECTED = False
SERVER_PORT = 5680  # 使用不冲突的端口
server_started = False
client_socket = None

# 超时处理装饰器
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({seconds}秒)")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class MockClient(threading.Thread):
    """模拟的外部奖励客户端"""
    
    def __init__(self, port, client_id=1):
        super().__init__()
        self.port = port
        self.client_id = client_id
        self.daemon = True
        self.socket = None
        self.connected = False
        self.running = True
    
    def run(self):
        global MOCK_CLIENT_CONNECTED
        
        try:
            # 尝试连接5次
            for attempt in range(5):
                try:
                    print(f"客户端 #{self.client_id}: 尝试连接 (尝试 {attempt+1}/5)")
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.settimeout(2.0)
                    self.socket.connect(('localhost', self.port))
                    self.connected = True
                    MOCK_CLIENT_CONNECTED = True
                    break
                except (ConnectionRefusedError, socket.timeout) as e:
                    print(f"客户端 #{self.client_id}: 连接失败: {e}")
                    if self.socket:
                        self.socket.close()
                        self.socket = None
                    if attempt < 4:
                        time.sleep(1)  # 等待后重试
            
            if not self.connected:
                print(f"客户端 #{self.client_id}: 无法连接到服务器")
                return
            
            print(f"客户端 #{self.client_id}: 已连接到服务器")
            
            # 接收欢迎消息
            welcome = self.socket.recv(1024).decode('utf-8')
            print(f"客户端 #{self.client_id}: 收到欢迎消息: {welcome.strip()}")
            
            # 处理请求
            while self.running:
                try:
                    # 接收请求
                    request_data = ""
                    self.socket.settimeout(5.0)  # 设置读取超时
                    
                    while True:
                        chunk = self.socket.recv(4096).decode('utf-8')
                        if not chunk:
                            if self.running:
                                print(f"客户端 #{self.client_id}: 连接已关闭")
                            return
                        
                        request_data += chunk
                        if '\n' in request_data:
                            break
                    
                    # 解析请求
                    request = json.loads(request_data)
                    
                    # 处理奖励请求
                    if request["type"] != "reward_request":
                        print(f"客户端 #{self.client_id}: 未知请求类型: {request['type']}")
                        continue
                    
                    # 获取回复和答案
                    completions = request["data"]["completions"]
                    responses = request["data"]["responses"]
                    answers = request["data"].get("answers", [])
                    
                    # 生成奖励
                    rewards = []
                    for i, completion in enumerate(completions):
                        # 基础奖励
                        base_reward = 50.0 * self.client_id
                        
                        # 如果有答案，则基于相似度给予额外奖励
                        if i < len(answers):
                            # 简化的"相似度"，这里只是示例
                            resp_len = len(responses[i])
                            ans_len = len(answers[i])
                            similarity_bonus = 20.0 * (1.0 - abs(resp_len - ans_len) / max(resp_len, ans_len, 1))
                            base_reward += similarity_bonus
                        
                        rewards.append(round(base_reward, 2))
                    
                    # 发送响应
                    response = {"rewards": rewards}
                    self.socket.send((json.dumps(response) + "\n").encode('utf-8'))
                    
                    print(f"客户端 #{self.client_id}: 已处理请求，奖励: {rewards}")
                    
                except socket.timeout:
                    # 读取超时，只是继续循环
                    continue
                except Exception as e:
                    if self.running:
                        print(f"客户端 #{self.client_id}: 处理请求时出错: {e}")
                    break
        
        except Exception as e:
            print(f"客户端 #{self.client_id}: 运行时出错: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止客户端"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        print(f"客户端 #{self.client_id}: 已停止")


def run_test():
    """运行测试"""
    global server_started
    
    try:
        # 确保服务器关闭
        shutdown_external_reward_server()
        time.sleep(0.5)
        
        print("\n==== 开始测试外部奖励系统 ====")
        
        # 启动服务器
        print("\n1. 启动服务器...")
        server_started = init_external_reward_server(SERVER_PORT)
        assert server_started, "服务器启动失败"
        print(f"服务器已在端口 {SERVER_PORT} 上启动")
        
        # 启动客户端
        print("\n2. 启动测试客户端...")
        client1 = MockClient(SERVER_PORT, 1)
        client2 = MockClient(SERVER_PORT, 2)
        
        client1.start()
        time.sleep(0.5)
        client2.start()
        
        # 等待客户端连接
        timeout = 10
        start_time = time.time()
        while not MOCK_CLIENT_CONNECTED and time.time() - start_time < timeout:
            time.sleep(0.5)
            print("等待客户端连接...")
        
        if not MOCK_CLIENT_CONNECTED:
            print("超时：客户端无法连接到服务器")
            return False
        
        # 准备测试数据
        completions = [
            "<think>这是思考过程1</think>回复1", 
            "<think>这是思考过程2</think>回复2"
        ]
        prompts = ["提示1", "提示2"]
        answers = ["标准答案1", "标准答案2"]
        
        # 测试1: 基本奖励计算
        try:
            with time_limit(10):
                print("\n3. 测试1: 获取外部奖励...")
                rewards = reward_external(completions, prompts, answers)
                print(f"获取的奖励: {rewards}")
                assert len(rewards) == len(completions), "奖励数量与回复数量不匹配"
                assert all(r > 0 for r in rewards), "应返回正奖励值"
                print("测试1通过")
        except Exception as e:
            print(f"测试1失败: {e}")
            return False
        
        # 测试2: 客户端断开后使用缓存奖励
        try:
            with time_limit(10):
                print("\n4. 测试2: 客户端断开后使用缓存奖励...")
                # 关闭客户端1
                client1.stop()
                time.sleep(1)
                
                # 再次获取奖励
                new_rewards = reward_external(completions, prompts, answers)
                print(f"客户端断开后获取的奖励: {new_rewards}")
                
                # 应仍有奖励（来自客户端2）
                assert all(r > 0 for r in new_rewards), "客户端断开后奖励应仍为正值"
                print("测试2通过")
        except Exception as e:
            print(f"测试2失败: {e}")
            return False
        
        # 测试3: 所有客户端断开后使用最后一次奖励
        try:
            with time_limit(10):
                print("\n5. 测试3: 所有客户端断开后使用最后一次奖励...")
                # 关闭客户端2
                client2.stop()
                time.sleep(1)
                
                # 记住上次的奖励值
                last_rewards = new_rewards
                
                # 再次获取奖励
                final_rewards = reward_external(completions, prompts, answers)
                print(f"所有客户端断开后获取的奖励: {final_rewards}")
                
                # 应使用最后一次的奖励
                assert final_rewards == last_rewards, "所有客户端断开后应使用最后一次奖励"
                print("测试3通过")
        except Exception as e:
            print(f"测试3失败: {e}")
            return False
        
        print("\n所有测试通过！")
        return True
    
    except Exception as e:
        print(f"测试过程中出错: {e}")
        return False
    finally:
        # 确保清理
        print("\n清理测试环境...")
        
        # 关闭服务器
        shutdown_external_reward_server()
        time.sleep(1)


if __name__ == "__main__":
    try:
        success = run_test()
        if success:
            print("\n==== 外部奖励系统测试成功 ====")
            sys.exit(0)
        else:
            print("\n==== 外部奖励系统测试失败 ====")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        # 确保清理
        if server_started:
            shutdown_external_reward_server()
        sys.exit(1)
