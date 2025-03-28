"""
极简版的外部奖励系统测试，专注于基本功能
"""
import sys
import time
import socket
import threading
import json
from grpo_reward import (
    init_external_reward_server,
    reward_external,
    shutdown_external_reward_server
)

# 使用不常用端口避免冲突
TEST_PORT = 7890
DEBUG = True

def log(message):
    """打印调试日志"""
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

def run_test_client():
    """运行一个简单的测试客户端"""
    log("启动测试客户端...")
    client = None
    
    try:
        # 等待一会儿确保服务器已启动
        time.sleep(2)
        
        # 创建并连接客户端
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10.0)  # 设置较长的超时时间
        
        log(f"客户端尝试连接到 localhost:{TEST_PORT}")
        client.connect(('localhost', TEST_PORT))
        log("客户端已连接到服务器")
        
        # 接收欢迎消息
        welcome_data = client.recv(1024).decode('utf-8')
        log(f"客户端收到欢迎消息: {welcome_data.strip()}")
        
        # 等待奖励请求，接收后回复
        while True:
            try:
                # 接收请求
                request_data = ""
                while True:
                    chunk = client.recv(4096).decode('utf-8')
                    if not chunk:
                        log("服务器已关闭连接")
                        return
                    
                    request_data += chunk
                    if '\n' in request_data:
                        break
                
                # 解析请求
                request = json.loads(request_data)
                
                if request["type"] == "reward_request":
                    log("收到奖励请求，处理中...")
                    # 简单生成固定奖励
                    completions = request["data"]["completions"]
                    rewards = [50.0 for _ in completions]
                    
                    # 发送响应
                    response = {"rewards": rewards}
                    client.send((json.dumps(response) + "\n").encode('utf-8'))
                    log(f"已发送奖励响应: {rewards}")
                else:
                    log(f"收到未知请求类型: {request['type']}")
            
            except Exception as e:
                log(f"处理请求时出错: {e}")
                break
    
    except Exception as e:
        log(f"客户端运行时出错: {e}")
    finally:
        if client:
            try:
                client.close()
            except:
                pass
        log("客户端已关闭")

def main():
    """主测试函数"""
    log("\n==== 开始简化版外部奖励测试 ====")
    
    # 确保服务器关闭
    shutdown_external_reward_server()
    time.sleep(1)
    
    # 第1步：启动服务器
    log("\n1. 启动外部奖励服务器")
    if not init_external_reward_server(TEST_PORT):
        log("服务器启动失败，测试终止")
        return False
    
    # 第2步：启动客户端线程
    log("\n2. 启动测试客户端")
    client_thread = threading.Thread(target=run_test_client)
    client_thread.daemon = True
    client_thread.start()
    
    # 等待客户端连接
    time.sleep(3)
    
    # 第3步：测试奖励功能
    try:
        log("\n3. 测试奖励功能")
        completions = ["<think>测试思考</think>测试回复"]
        prompts = ["测试提示"]
        answers = ["测试答案"]
        
        rewards = reward_external(completions, prompts, answers)
        log(f"获取的奖励: {rewards}")
        
        if len(rewards) == len(completions) and all(r > 0 for r in rewards):
            log("奖励功能测试通过")
            result = True
        else:
            log("奖励功能测试失败")
            result = False
    
    except Exception as e:
        log(f"测试过程中出错: {e}")
        result = False
    
    finally:
        # 关闭服务器
        log("\n4. 关闭服务器")
        shutdown_external_reward_server()
        time.sleep(1)
        
        # 等待客户端线程结束
        client_thread.join(timeout=3)
        
        log("\n==== 简化版外部奖励测试" + ("成功" if result else "失败") + " ====")
        return result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
