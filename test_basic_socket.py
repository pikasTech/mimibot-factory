"""
极简测试文件，只测试最基本的服务器启动和客户端连接
"""
import socket
import threading
import time
import json
import sys

# 全局变量
server = None
server_running = False
client_connected = False
PORT = 6789  # 使用一个不太常用的端口避免冲突

def run_server():
    """运行最简单的服务器"""
    global server, server_running
    
    print("\n===== 启动测试服务器 =====")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 绑定端口和启动
    try:
        server.bind(('0.0.0.0', PORT))
        server.listen(5)
        server_running = True
        print(f"服务器已启动，监听端口 {PORT}")
        
        # 设置超时以防止无限阻塞
        server.settimeout(0.5)
        
        # 接受连接的循环
        while server_running:
            try:
                client_socket, client_address = server.accept()
                print(f"客户端已连接: {client_address}")
                
                # 发送欢迎消息
                welcome = {"message": "Welcome to the test server"}
                client_socket.send((json.dumps(welcome) + "\n").encode('utf-8'))
                print(f"已向客户端发送欢迎消息")
                
                # 在独立线程中处理客户端
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                # 正常超时，继续循环
                continue
            except Exception as e:
                if server_running:
                    print(f"接受连接时出错: {e}")
                    time.sleep(0.5)
    except Exception as e:
        print(f"服务器启动或运行时出错: {e}")
    finally:
        if server:
            try:
                server.close()
            except:
                pass
        server_running = False
        print("服务器已关闭")

def handle_client(client_socket, address):
    """非常简单的客户端处理函数"""
    global client_connected
    
    client_connected = True
    try:
        # 保持连接并接收数据，简单打印
        while server_running:
            try:
                client_socket.settimeout(1.0)
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    print(f"客户端 {address} 断开连接")
                    break
                
                print(f"从客户端 {address} 收到: {data.strip()}")
                
                # 发送回复
                response = {"status": "ok", "received": data.strip()}
                client_socket.send((json.dumps(response) + "\n").encode('utf-8'))
                
            except socket.timeout:
                # 超时，继续循环
                continue
            except Exception as e:
                print(f"处理客户端 {address} 数据时出错: {e}")
                break
    finally:
        try:
            client_socket.close()
        except:
            pass
        print(f"客户端 {address} 连接已关闭")

def run_client():
    """运行一个简单的测试客户端"""
    global client_connected
    
    print("\n===== 启动测试客户端 =====")
    time.sleep(1)  # 确保服务器先启动
    
    try:
        # 创建客户端套接字
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)  # 设置较长的超时
        
        # 连接到服务器
        print(f"尝试连接到服务器 localhost:{PORT}")
        client.connect(('localhost', PORT))
        
        # 接收欢迎消息
        welcome_data = client.recv(1024).decode('utf-8')
        print(f"收到服务器欢迎消息: {welcome_data.strip()}")
        
        # 发送测试数据
        test_message = {"type": "test", "message": "Hello from client"}
        client.send((json.dumps(test_message) + "\n").encode('utf-8'))
        print("已发送测试消息到服务器")
        
        # 接收响应
        response_data = client.recv(1024).decode('utf-8')
        print(f"收到服务器响应: {response_data.strip()}")
        
        client_connected = True
        
        # 关闭连接
        client.close()
        print("客户端已关闭连接")
        
    except Exception as e:
        print(f"客户端运行时出错: {e}")
        client_connected = False
    
def main():
    """运行测试"""
    global server_running
    
    # 启动服务器线程
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(2)
    
    if not server_running:
        print("服务器启动失败，退出测试")
        return False
    
    # 运行客户端
    client_thread = threading.Thread(target=run_client)
    client_thread.daemon = True
    client_thread.start()
    
    # 等待客户端完成
    client_thread.join(timeout=10)
    
    # 检查结果
    if client_connected:
        print("\n===== 基本套接字测试通过 =====")
        success = True
    else:
        print("\n===== 基本套接字测试失败 =====")
        success = False
    
    # 关闭服务器
    server_running = False
    time.sleep(1)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
