import unittest
import threading
import socket
import json
import time
import sys
import os
import random

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要测试的模块
from grpo_reward import init_external_reward_server, reward_EXTERNAL, shutdown_external_reward_server

class TestExternalReward(unittest.TestCase):
    """测试外部奖励功能"""
    
    server_port = 5678
    server_initialized = False

    @classmethod
    def setUpClass(cls):
        """在所有测试之前运行，启动服务器"""
        # 确保关闭任何可能已经在运行的服务器
        shutdown_external_reward_server()
        time.sleep(0.5)  # 等待可能的关闭完成
        
        # 启动外部奖励服务器
        print("开始初始化测试服务器...")
        cls.server_initialized = init_external_reward_server(cls.server_port)
        print(f"服务器初始化状态: {cls.server_initialized}")
        time.sleep(2)  # 给服务器更多时间启动

    @classmethod
    def tearDownClass(cls):
        """在所有测试之后运行，关闭服务器"""
        print("测试完成，准备关闭服务器...")
        shutdown_external_reward_server()
        time.sleep(1)  # 等待服务器完全关闭

    def setUp(self):
        """每个测试前的准备工作"""
        # 检查服务器是否正常初始化
        if not self.server_initialized:
            self.skipTest("服务器初始化失败，跳过测试")

    def test_no_clients(self):
        """测试没有客户端连接时的行为"""
        completions = ["<think>思考过程</think>回复1", "<think>思考过程</think>回复2"]
        prompts = ["提示1", "提示2"]
        answers = ["标准答案1", "标准答案2"]
        
        # 确保没有客户端连接
        time.sleep(0.5)
        
        # 调用奖励函数
        rewards = reward_EXTERNAL(completions, prompts, answers)
        
        # 验证结果
        self.assertEqual(len(rewards), len(completions), "奖励值数量应与完成数量匹配")
        self.assertEqual(rewards, [0, 0], "无客户端时应返回零奖励")

    def test_single_client(self):
        """测试单个客户端连接的情况"""
        # 启动模拟客户端线程
        client_connected = threading.Event()
        client_thread = threading.Thread(
            target=self._mock_client, 
            args=(1, None, client_connected)
        )
        client_thread.daemon = True
        client_thread.start()
        
        # 等待客户端连接成功或超时
        if not client_connected.wait(5):
            self.skipTest("客户端连接失败，跳过测试")
            return
            
        # 准备测试数据
        completions = ["<think>思考过程1</think>回复1", "<think>思考过程2</think>回复2"]
        prompts = ["提示1", "提示2"]
        answers = ["标准答案1", "标准答案2"]
        
        # 调用奖励函数
        rewards = reward_EXTERNAL(completions, prompts, answers)
        
        # 验证结果
        self.assertEqual(len(rewards), len(completions), "奖励值数量应与完成数量匹配")
        self.assertTrue(all(r > 0 for r in rewards), "应收到非零奖励")
        
        # 等待客户端线程结束
        client_thread.join(timeout=5)

    def test_multiple_clients(self):
        """测试多个客户端连接的情况"""
        # 启动多个模拟客户端线程
        client_threads = []
        client_events = []
        
        for i in range(3):
            connected_event = threading.Event()
            client_events.append(connected_event)
            
            client_thread = threading.Thread(
                target=self._mock_client, 
                args=(i+1, None, connected_event)
            )
            client_thread.daemon = True
            client_thread.start()
            client_threads.append(client_thread)
        
        # 等待所有客户端连接或超时
        all_connected = True
        for i, event in enumerate(client_events):
            if not event.wait(5):
                print(f"客户端 {i+1} 连接失败")
                all_connected = False
        
        if not all_connected:
            self.skipTest("一个或多个客户端连接失败，跳过测试")
            return
        
        # 准备测试数据
        completions = ["<think>思考过程1</think>回复1", "<think>思考过程2</think>回复2"]
        prompts = ["提示1", "提示2"]
        answers = ["标准答案1", "标准答案2"]
        
        # 调用奖励函数
        rewards = reward_EXTERNAL(completions, prompts, answers)
        
        # 验证结果 - 多个客户端的奖励应当累加
        self.assertEqual(len(rewards), len(completions), "奖励值数量应与完成数量匹配")
        self.assertTrue(all(r > 50 for r in rewards), "多客户端累加奖励应较大")
        
        # 等待客户端线程结束
        for thread in client_threads:
            thread.join(timeout=5)

    def test_client_disconnect(self):
        """测试客户端断开连接后使用上次奖励的情况"""
        # 启动一个会断开的客户端
        connected_event = threading.Event()
        client_thread = threading.Thread(
            target=self._mock_client,
            args=(999, 1, connected_event)  # client_id=999, disconnect_after=1
        )
        client_thread.daemon = True
        client_thread.start()
        
        # 等待客户端连接成功或超时
        if not connected_event.wait(5):
            self.skipTest("客户端连接失败，跳过测试")
            return
        
        # 准备测试数据
        completions = ["<think>思考过程1</think>回复1", "<think>思考过程2</think>回复2"]
        prompts = ["提示1", "提示2"]
        answers = ["标准答案1", "标准答案2"]
        
        # 第一次调用奖励函数，获取初始奖励
        initial_rewards = reward_EXTERNAL(completions, prompts, answers)
        
        # 等待客户端断开连接
        time.sleep(3)
        
        # 再次调用奖励函数，应使用上次的奖励
        second_rewards = reward_EXTERNAL(completions, prompts, answers)
        
        # 验证结果
        self.assertEqual(initial_rewards, second_rewards, "断开连接后应使用上次的奖励")
        
        # 等待客户端线程结束
        client_thread.join(timeout=5)

    def _mock_client(self, client_id=1, disconnect_after=None, connected_event=None):
        """模拟一个外部奖励客户端"""
        client = None
        try:
            # 多次尝试连接，最多重试5次
            for attempt in range(5):
                try:
                    # 连接到服务器
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.settimeout(2.0)  # 设置连接超时
                    client.connect(('localhost', self.server_port))
                    # 连接成功，跳出重试循环
                    break
                except (socket.timeout, ConnectionRefusedError) as e:
                    print(f"模拟客户端 {client_id} 连接尝试 {attempt+1} 失败: {str(e)}")
                    if client:
                        client.close()
                    client = None
                    # 如果不是最后一次尝试，继续尝试
                    if attempt < 4:
                        time.sleep(1)  # 等待1秒后重试
                    else:
                        print(f"模拟客户端 {client_id} 达到最大重试次数，连接失败")
                        if connected_event:
                            connected_event.clear()
                        return
            
            # 验证是否成功连接
            if not client:
                print(f"模拟客户端 {client_id} 无法连接到服务器")
                if connected_event:
                    connected_event.clear()
                return
                
            # 接收欢迎消息
            welcome = client.recv(1024).decode('utf-8')
            welcome_data = json.loads(welcome)
            
            print(f"模拟客户端 {client_id} 已连接并收到欢迎消息")
            
            # 通知连接已建立
            if connected_event:
                connected_event.set()
            
            # 计数器，用于可选的断开连接测试
            counter = 0
            
            # 持续接收和响应请求
            while True:
                try:
                    # 接收请求
                    request = ""
                    client.settimeout(10.0)  # 设置读取超时
                    while True:
                        chunk = client.recv(4096).decode('utf-8')
                        if not chunk:
                            return  # 连接已关闭
                        
                        request += chunk
                        if '\n' in request:
                            break
                    
                    # 解析请求
                    request_data = json.loads(request)
                    
                    # 生成随机奖励
                    completions = request_data["data"]["completions"]
                    rewards = [random.uniform(30, 100) * client_id for _ in completions]
                    
                    # 发送响应
                    response = {
                        "rewards": rewards
                    }
                    client.send((json.dumps(response) + "\n").encode('utf-8'))
                    
                    # 递增计数器
                    counter += 1
                    print(f"模拟客户端 {client_id} 已处理请求 #{counter}")
                    
                    # 如果设置了断开连接条件并达到条件，则断开
                    if disconnect_after and counter >= disconnect_after:
                        print(f"模拟客户端 {client_id} 已达到计数 {counter}，准备断开")
                        break
                        
                    # 小延迟以模拟处理时间
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"模拟客户端 {client_id} 处理请求异常: {str(e)}")
                    break
        except Exception as e:
            print(f"模拟客户端 {client_id} 连接或处理异常: {str(e)}")
            if connected_event:
                connected_event.clear()
        finally:
            try:
                if client:
                    client.close()
            except:
                pass
            print(f"模拟客户端 {client_id} 已断开连接")


if __name__ == '__main__':
    unittest.main(verbosity=2)
