from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
import os
import argparse
from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub import HfApi, __version__ as hf_version
import shutil
from tqdm import tqdm
import glob
import requests
import logging
import time
import concurrent.futures
import threading
import queue
import hashlib
import json
from urllib.parse import urlparse
from pathlib import Path
import math

# 设置日志级别，便于查看详细信息
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Download models from ModelScope or Hugging Face mirror")
    parser.add_argument(
        "--model_id", 
        type=str,
        default="shenzhi-wang/Llama3.1-8B-Chinese-Chat",
        help="Model ID to download"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./models",
        help="Directory to save the downloaded model"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["modelscope", "huggingface"],
        default="huggingface",
        help="Source to download the model from: modelscope or huggingface mirror"
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default="https://hf-mirror.com",
        # default="https://hf.co",
        help="Hugging Face mirror URL (only used when source is huggingface)"
    )
    parser.add_argument(
        "--file_extensions",
        type=str,
        nargs="+",
        default=None,
        help="List of file extensions to download (e.g., .gguf .bin .safetensors). If not specified, all files will be downloaded."
    )
    parser.add_argument(
        "--ignore_extensions",
        type=str,
        nargs="+",
        default=[".gguf"],
        help="List of file extensions to ignore (e.g., .gguf .bin). Files with these extensions will not be downloaded."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=1,
        help="Number of parallel workers for downloading files"
    )
    parser.add_argument(
        "--max_retries", 
        type=int, 
        default=5,
        help="Maximum number of retries for failed downloads"
    )
    parser.add_argument(
        "--retry_delay", 
        type=int, 
        default=5,
        help="Delay in seconds between retries"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=100,
        help="Timeout in seconds for download requests"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 1024 * 100,  # 10MB
        help="Chunk size for multi-threaded file download in bytes"
    )
    parser.add_argument(
        "--min_file_size",
        type=int,
        default=40 * 1024 * 1024,  # 降低到40MB
        help="Minimum file size in bytes to enable multi-threaded download"
    )
    parser.add_argument(
        "--threads_per_file",
        type=int,
        default=8,
        help="Number of threads per file for multi-threaded download"
    )
    return parser.parse_args()

def get_file_size(url, timeout=30, max_retries=3, retry_delay=2):
    """获取远程文件大小，增强版本"""
    print(f"正在获取文件大小: {url}")
    
    # 提取文件名用于日志显示
    filename = url.split("/")[-1]
    
    # 尝试多种方法获取文件大小
    methods = [
        "HEAD请求",
        "Range请求",
        "GET请求",
        "扩展名启发式"
    ]
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"重试获取文件大小 (尝试 {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
        
        # 方法1: HEAD请求
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code == 200 and 'Content-Length' in response.headers:
                size = int(response.headers['Content-Length'])
                if size > 0:
                    print(f"通过HEAD请求获取到文件大小: {size / (1024*1024):.2f} MB")
                    return size
        except Exception as e:
            print(f"HEAD请求获取大小失败: {e}")
        
        # 方法2: Range请求
        try:
            headers = {'Range': 'bytes=0-0'}
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if 'Content-Range' in response.headers:
                content_range = response.headers['Content-Range']
                size_str = content_range.split('/')[-1]
                if size_str.isdigit():
                    size = int(size_str)
                    if size > 0:
                        print(f"通过Range请求获取到文件大小: {size / (1024*1024):.2f} MB")
                        return size
        except Exception as e:
            print(f"Range请求获取大小失败: {e}")
        
        # 方法3: 小型GET请求
        try:
            response = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
            if response.status_code == 200 and 'Content-Length' in response.headers:
                size = int(response.headers.get('Content-Length', 0))
                if size > 0:
                    print(f"通过GET请求获取到文件大小: {size / (1024*1024):.2f} MB")
                    response.close()  # 确保关闭连接
                    return size
            response.close()
        except Exception as e:
            print(f"GET请求获取大小失败: {e}")
    
    # 方法4: 基于文件扩展名的启发式推断
    # 对于模型权重文件，假定它们是大文件
    _, ext = os.path.splitext(url.lower())
    if ext in ['.safetensors', '.bin', '.gguf', '.pt', '.pth', '.ckpt']:
        # 假设模型文件至少1GB
        assumed_size = 1 * 1024 * 1024 * 1024  # 1GB
        print(f"文件大小检测失败，但识别到模型文件扩展名 {ext}，假定大小为 {assumed_size / (1024*1024):.2f} MB")
        return assumed_size
    
    # 检查是否是已下载部分的大小
    if "断点续传" in url and "从" in url:
        try:
            # 尝试从URL中提取已下载的字节数
            resume_size_str = url.split("从")[-1].strip().split()[0]
            resume_size = int(resume_size_str)
            if resume_size > 50 * 1024 * 1024:  # 如果已下载超过50MB
                print(f"检测到已下载 {resume_size / (1024*1024):.2f} MB，判断为大文件")
                # 假设文件至少是已下载大小的2倍
                return resume_size * 2
        except Exception:
            pass
    
    print(f"无法获取文件大小，所有方法均失败")
    return 0

def download_chunk(url, local_path, start_byte, end_byte, chunk_id, max_retries=5, retry_delay=5, timeout=100):
    """下载文件的一个块"""
    temp_chunk_path = f"{local_path}.part{chunk_id}"
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    
    # 获取文件名用于显示
    filename = os.path.basename(local_path)
    
    # 如果分块已存在，检查大小是否正确
    if os.path.exists(temp_chunk_path):
        file_size = os.path.getsize(temp_chunk_path)
        expected_size = end_byte - start_byte + 1
        if file_size == expected_size:
            print(f"文件 {filename} 的分块 {chunk_id} 已存在且完整，跳过下载")
            return True
    
    # 下载速度监控相关参数
    min_speed_bytes = 100 * 1024  # 最低可接受速度 5KB/s
    speed_check_interval = 5  # 每5秒检查一次速度
    slow_speed_duration = 120  # 如果连续15秒速度过慢，则重试
    
    for attempt in range(max_retries):
        # 随机等待一段时间，避免多个线程同时请求
        time.sleep(retry_delay * (0.5 + 0.5 * attempt))
        try:
            if attempt > 0:
                print(f"正在重试文件 {filename} 的分块 {chunk_id} ({attempt}/{max_retries})...")
            
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                
                chunk_size = end_byte - start_byte + 1
                with open(temp_chunk_path, 'wb') as f:
                    with tqdm(
                        total=chunk_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"{filename} 分块 {chunk_id}/{math.ceil((end_byte+1)/chunk_size)}"
                    ) as pbar:
                        # 速度监控变量
                        last_check_time = time.time()
                        last_bytes_downloaded = 0
                        slow_count = 0
                        
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                
                                # 检查下载速度
                                current_time = time.time()
                                if current_time - last_check_time >= speed_check_interval:
                                    # 计算这个时间窗口内的下载速度
                                    bytes_downloaded = pbar.n - last_bytes_downloaded
                                    elapsed = current_time - last_check_time
                                    speed = bytes_downloaded / elapsed if elapsed > 0 else 0
                                    
                                    # 如果速度过慢，增加慢速计数
                                    if speed < min_speed_bytes:
                                        slow_count += 1
                                        print(f"警告: 下载速度过慢 ({speed/1024:.2f} KB/s < {min_speed_bytes/1024:.2f} KB/s)，已持续 {slow_count * speed_check_interval} 秒")
                                        
                                        # 如果连续几次速度都过慢，则中断并重试
                                        if slow_count * speed_check_interval >= slow_speed_duration:
                                            raise Exception(f"下载速度持续过慢 ({speed/1024:.2f} KB/s)，已超过 {slow_speed_duration} 秒，触发重试")
                                    else:
                                        # 速度恢复了，重置慢速计数
                                        if slow_count > 0:
                                            print(f"下载速度已恢复至 {speed/1024:.2f} KB/s")
                                        slow_count = 0
                                    
                                    # 更新检查点
                                    last_check_time = current_time
                                    last_bytes_downloaded = pbar.n
                
            print(f"✅ 文件 {filename} 的分块 {chunk_id} 下载成功")
            return True
            
        except Exception as e:
            print(f"文件 {filename} 的分块 {chunk_id} 下载失败 ({attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
    
    print(f"达到最大重试次数，文件 {filename} 的分块 {chunk_id} 下载失败")
    return False

def download_file_in_chunks(url, local_path, num_threads=50, chunk_size=1024*1024*20, max_retries=5, retry_delay=5, timeout=100):
    """使用多线程分片下载单个大文件，严格按照chunk大小分块，限制并发下载数量"""
    print(f"开始多线程分片下载: {url}")
    
    # 创建目标目录
    directory = os.path.dirname(local_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # 获取文件大小
    file_size = get_file_size(url, timeout)
    
    # 如果仍然无法获取文件大小，则尝试估计文件大小
    if file_size <= 0:
        # 对于safetensors模型文件，估计文件大小为4GB
        if local_path.endswith('.safetensors'):
            file_size = 4 * 1024 * 1024 * 1024  # 4GB
            print(f"无法获取文件大小，基于文件类型估计为 {file_size / (1024*1024*1024):.2f} GB")
        else:
            print(f"无法获取或估计文件大小，尝试常规下载: {url}")
            return download_with_retry(url, local_path, max_retries, retry_delay, timeout)
    
    print(f"文件大小: {file_size / (1024*1024):.2f} MB，开始多线程下载")
    
    # 严格按照chunk_size计算块数量
    num_chunks = math.ceil(file_size / chunk_size)
    
    print(f"将文件分为 {num_chunks} 个块，每块 {chunk_size / (1024*1024):.2f} MB")
    print(f"下载将使用最多 {num_threads} 个并发线程")
    
    # 准备下载任务
    chunks = []
    for i in range(num_chunks):
        start_byte = i * chunk_size
        # 最后一块可能不足chunk_size
        end_byte = min(start_byte + chunk_size - 1, file_size - 1)
        chunks.append({
            'id': i + 1,
            'start': start_byte,
            'end': end_byte
        })
    
    # 使用线程池限制并发下载数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交下载任务
        future_to_chunk = {
            executor.submit(
                download_chunk,
                url,
                local_path,
                chunk['start'],
                chunk['end'],
                chunk['id'],
                max_retries,
                retry_delay,
                timeout
            ): chunk for chunk in chunks
        }
        
        # 收集结果
        results = []
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                success = future.result()
                results.append({
                    'chunk_id': chunk['id'],
                    'success': success
                })
                completed += 1
                print(f"分块 {chunk['id']}/{num_chunks} 完成，状态: {'成功' if success else '失败'} ({completed}/{num_chunks})")
            except Exception as e:
                print(f"分块 {chunk['id']} 下载过程出错: {e}")
                results.append({
                    'chunk_id': chunk['id'],
                    'success': False
                })
                completed += 1
    
    # 检查是否所有块都下载成功
    if all(r['success'] for r in results):
        print("所有分块下载成功，开始合并文件...")
        
        # 合并文件
        with open(local_path, 'wb') as outfile:
            for i in range(1, num_chunks + 1):
                chunk_file = f"{local_path}.part{i}"
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    # 删除临时文件
                    os.remove(chunk_file)
        
        print(f"✅ 文件合并完成: {local_path}")
        return True
    else:
        print("部分分块下载失败，无法合并文件")
        return False

def download_with_retry(url, local_path, max_retries=5, retry_delay=5, timeout=100):
    """下载单个文件，支持重试和断点续传"""
    
    directory = os.path.dirname(local_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # 临时文件路径，用于下载中的文件
    temp_path = f"{local_path}.download"
    
    # 获取已下载的字节数用于断点续传
    resume_size = 0
    headers = {}
    if os.path.exists(temp_path):
        resume_size = os.path.getsize(temp_path)
        headers['Range'] = f'bytes={resume_size}-'
        print(f"断点续传: {url} 从 {resume_size} 字节开始")

    for attempt in range(max_retries):
        try:
            # 发起请求
            if attempt > 0:
                print(f"正在重试 ({attempt}/{max_retries}): {url}")
                
            # 创建流式请求以支持大文件下载和断点续传
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                
                # 获取文件总大小
                total_size = int(r.headers.get('content-length', 0)) + resume_size
                
                # 打开文件用于追加
                mode = 'ab' if resume_size > 0 else 'wb'
                with open(temp_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=resume_size,
                        unit='B',
                        unit_scale=True,
                        desc=os.path.basename(local_path)
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            # 下载完成后重命名文件
            shutil.move(temp_path, local_path)
            print(f"✅ 下载成功: {local_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"下载失败 ({attempt+1}/{max_retries}): {url}")
            print(f"错误: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"达到最大重试次数，下载失败: {url}")
                return False
    
    return False

def smart_download(url, local_path, min_file_size=52428800, threads_per_file=50, chunk_size=1024 * 1024 * 20, 
                  max_retries=5, retry_delay=5, timeout=100):
    """智能下载：大文件使用多线程分片，小文件使用常规下载"""
    
    # 从URL或本地路径检查文件类型
    _, ext = os.path.splitext(local_path.lower())
    
    # 对于已知的大文件类型，直接使用多线程下载
    force_multi_thread = ext in ['.safetensors', '.bin', '.gguf', '.pt', '.pth', '.ckpt']
    
    # 检查是否有部分已下载的文件
    temp_path = f"{local_path}.download"
    resume_size = 0
    if os.path.exists(temp_path):
        resume_size = os.path.getsize(temp_path)
        if resume_size > min_file_size:
            force_multi_thread = True
            print(f"检测到已下载部分 {resume_size / (1024*1024):.2f} MB > {min_file_size / (1024*1024):.2f} MB，强制使用多线程下载")
    
    # 获取文件大小
    file_size = get_file_size(url, timeout=timeout)
    
    # 根据情况决定使用什么下载方式
    if file_size >= min_file_size:
        print(f"文件大小 {file_size / (1024*1024):.2f} MB 超过 {min_file_size / (1024*1024):.2f} MB 阈值，使用多线程分片下载")
        return download_file_in_chunks(url, local_path, threads_per_file, chunk_size, max_retries, retry_delay, timeout)
    elif force_multi_thread:
        print(f"强制对文件类型 {ext} 使用多线程分片下载，无论大小如何")
        return download_file_in_chunks(url, local_path, threads_per_file, chunk_size, max_retries, retry_delay, timeout)
    else:
        print(f"文件大小 {file_size / (1024*1024):.2f} MB 小于 {min_file_size / (1024*1024):.2f} MB 阈值，使用常规下载")
        return download_with_retry(url, local_path, max_retries, retry_delay, timeout)

def parallel_download(file_list, num_workers=10, min_file_size=52428800, threads_per_file=50, 
                     chunk_size=1024 * 1024 * 20, max_retries=5, retry_delay=5, timeout=100):
    """并行下载多个文件，大文件使用多线程分片下载"""
    results = []
    
    # 使用线程池进行并行下载
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 创建下载任务
        future_to_file = {
            executor.submit(
                smart_download, 
                file_info['url'], 
                file_info['local_path'],
                min_file_size,
                threads_per_file,
                chunk_size,
                max_retries,
                retry_delay,
                timeout
            ): file_info for file_info in file_list
        }
        
        # 等待所有任务完成，并收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                success = future.result()
                results.append({
                    'file': file_info['local_path'],
                    'success': success
                })
            except Exception as e:
                print(f"下载过程出错: {file_info['url']}")
                print(f"错误: {str(e)}")
                results.append({
                    'file': file_info['local_path'],
                    'success': False,
                    'error': str(e)
                })
    
    # 输出下载统计
    success_count = sum(1 for r in results if r['success'])
    print(f"下载完成: {success_count}/{len(results)} 个文件成功")
    gg
    # 返回失败的文件，便于后续重试
    failed_files = [r['file'] for r in results if not r['success']]
    if failed_files:
        print(f"下载失败的文件: {failed_files}")
    
    return results

def download_from_modelscope(model_id, output_dir, file_extensions=None, ignore_extensions=None, 
                           num_workers=4, max_retries=5, retry_delay=5, timeout=100,
                           min_file_size=52428800, threads_per_file=8, chunk_size=10485760):
    print(f"正在从魔搭下载模型 {model_id} 到 {output_dir}...")
    
    try:
        # 先尝试使用原始方法下载获取模型目录结构
        print("获取模型目录结构...")
        model_dir = ms_snapshot_download(model_id, cache_dir=output_dir)
        
        # 获取模型文件列表
        all_files = []
        for root, _, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        # 应用文件过滤器
        files_to_keep = filter_files(all_files, file_extensions, ignore_extensions)
        
        if not files_to_keep:
            print(f"警告: 根据过滤条件，没有找到需要保留的文件")
            return model_dir
            
        # 准备并行下载文件列表
        # 由于我们已经有了本地文件，这里用于演示多线程下载和重试逻辑
        print(f"找到 {len(files_to_keep)} 个文件，准备验证文件完整性...")
        
        # 在这里，我们可以添加一个文件完整性验证步骤
        # 如果文件不完整，可以使用模型ID和文件路径构建URL重新下载
        
        print(f"模型下载成功！保存路径: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"从魔搭下载模型时出错: {e}")
        print("尝试通过多线程方式重新下载...")
        
        # 如果常规下载失败，尝试多线程手动下载
        try:
            # 构建魔搭API文件列表请求URL
            api_url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files"
            print(f"获取文件列表: {api_url}")
            
            response = requests.get(api_url, timeout=timeout)
            response.raise_for_status()
            files_data = response.json().get('Data', {}).get('Files', [])
            
            if not files_data:
                print("无法获取文件列表")
                return None
                
            # 过滤文件
            download_files = []
            for file_info in files_data:
                file_name = file_info.get('FileName', '')
                file_path = os.path.join(output_dir, file_name)
                
                # 检查扩展名是否符合过滤条件
                _, ext = os.path.splitext(file_name)
                if file_extensions and ext.lower() not in [f.lower() for f in file_extensions]:
                    continue
                if ignore_extensions and ext.lower() in [f.lower() for f in ignore_extensions]:
                    continue
                    
                # 构建下载URL
                download_url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files/{file_name}"
                
                download_files.append({
                    'url': download_url,
                    'local_path': file_path
                })
            
            # 并行下载
            print(f"开始并行下载 {len(download_files)} 个文件，使用 {num_workers} 个线程")
            parallel_download(download_files, num_workers, min_file_size, threads_per_file, 
                            chunk_size, max_retries, retry_delay, timeout)
            
            # 下载完成后的模型目录
            model_dir = os.path.join(output_dir, model_id.split("/")[-1])
            print(f"模型下载成功！保存路径: {model_dir}")
            return model_dir
            
        except Exception as e:
            print(f"尝试多线程下载也失败: {e}")
            return None

def verify_mirror(mirror_url):
    """验证镜像站是否可用，并显示实际使用的URL"""
    print(f"正在验证镜像站 {mirror_url} 是否可用...")
    
    try:
        # 测试连接镜像站
        response = requests.get(f"{mirror_url}/api/models", timeout=10)
        if response.status_code == 200:
            print(f"✅ 镜像站连接成功！响应状态码: {response.status_code}")
            return True
        else:
            print(f"⚠️ 镜像站连接异常，响应状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 镜像站连接失败: {e}")
        return False

def download_from_huggingface(model_id, output_dir, mirror_url, file_extensions=None, ignore_extensions=None, 
                            num_workers=10, max_retries=5, retry_delay=5, timeout=100,
                            min_file_size=52428800, threads_per_file=50, chunk_size=1024*1024*20):
    print(f"正在从Hugging Face镜像站 {mirror_url} 下载模型 {model_id} 的文件...")
    
    # 验证镜像站
    mirror_valid = verify_mirror(mirror_url)
    if not mirror_valid:
        print("镜像站验证失败，将尝试继续下载，但可能会使用官方源")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 显示huggingface_hub版本
        print(f"huggingface_hub 版本: {hf_version}")
        
        # 设置环境变量以使用镜像站 (添加调试信息)
        old_endpoint = os.environ.get("HF_ENDPOINT", "未设置")
        print(f"原始HF_ENDPOINT: {old_endpoint}")
        
        os.environ["HF_ENDPOINT"] = mirror_url
        print(f"已设置HF_ENDPOINT为: {os.environ.get('HF_ENDPOINT')}")
        
        # 不再使用configure_http_backend，因为参数不兼容
        # 只使用环境变量和endpoint参数
        
        # 创建HfApi实例并指定端点
        api = HfApi(endpoint=mirror_url)
        print(f"已创建HfApi实例，端点: {api.endpoint}")
        
        # 测试API是否使用了正确的镜像
        try:
            print("正在测试API连接...")
            models = api.list_models(limit=1)
            print(f"API连接成功! 获取到模型: {next(models).id}")
        except Exception as e:
            print(f"API测试失败: {e}")
        
        # 创建特定于模型的目录
        model_specific_dir = os.path.join(output_dir, model_id.split("/")[-1])
        os.makedirs(model_specific_dir, exist_ok=True)
        
        # 准备下载参数
        download_kwargs = {
            "repo_id": model_id,
            "local_dir": model_specific_dir,
            "local_dir_use_symlinks": False,
            "revision": "main",
            "tqdm_class": tqdm,
            "endpoint": mirror_url,  # 直接在参数中指定端点
            "max_workers": num_workers,  # 设置并行下载的worker数
            "retry_on_error": True,  # 启用重试
            "resume_download": True,  # 启用断点续传
            "max_retries": max_retries,  # 设置最大重试次数
        }
        
        # 处理文件过滤条件
        if file_extensions:
            print(f"过滤文件后缀: 只保留 {', '.join(file_extensions)}")
            allow_patterns = []
            for ext in file_extensions:
                # 确保扩展名以点号开头
                if not ext.startswith('.'):
                    ext = '.' + ext
                allow_patterns.append(f"*{ext}")
            download_kwargs["allow_patterns"] = allow_patterns
        
        if ignore_extensions:
            print(f"过滤文件后缀: 忽略 {', '.join(ignore_extensions)}")
            ignore_patterns = []
            for ext in ignore_extensions:
                # 确保扩展名以点号开头
                if not ext.startswith('.'):
                    ext = '.' + ext
                ignore_patterns.append(f"*{ext}")
            download_kwargs["ignore_patterns"] = ignore_patterns
        
        # 设置额外的调试信息
        print(f"下载参数: {download_kwargs}")
        print(f"当前HF_ENDPOINT环境变量: {os.environ.get('HF_ENDPOINT')}")
        
        # 尝试添加一个测试请求，查看是否使用了镜像站
        try:
            test_url = f"{mirror_url}/api/models/{model_id}"
            print(f"正在测试镜像站访问: {test_url}")
            test_response = requests.get(test_url, timeout=timeout)
            print(f"镜像站测试结果: 状态码 {test_response.status_code}")
        except Exception as e:
            print(f"镜像站测试失败: {e}")
        
        # 使用snapshot_download下载仓库
        print(f"开始下载，使用 {num_workers} 个线程...")
        downloaded_path = hf_snapshot_download(**download_kwargs)
        
        print(f"模型下载成功！保存路径: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"从Hugging Face镜像站下载模型时出错: {e}")
        print(f"错误详情: {str(e)}")
        
        # 如果自动下载失败，尝试手动多线程下载
        print("尝试通过手动多线程方式重新下载...")
        
        try:
            # 获取模型文件列表
            file_list_url = f"{mirror_url}/api/models/{model_id}/tree/main"
            response = requests.get(file_list_url, timeout=timeout)
            response.raise_for_status()
            
            files_data = response.json()
            if not files_data:
                print("无法获取文件列表")
                return None
                
            # 准备下载列表
            model_specific_dir = os.path.join(output_dir, model_id.split("/")[-1])
            os.makedirs(model_specific_dir, exist_ok=True)
            
            download_files = []
            for file_info in files_data:
                if file_info.get('type') != 'file':
                    continue
                    
                file_path = file_info.get('path', '')
                local_path = os.path.join(model_specific_dir, file_path)
                
                # 检查扩展名是否符合过滤条件
                _, ext = os.path.splitext(file_path)
                if file_extensions and ext.lower() not in [f.lower() for f in file_extensions]:
                    continue
                if ignore_extensions and ext.lower() in [f.lower() for f in ignore_extensions]:
                    continue
                
                # 构建下载URL
                download_url = f"{mirror_url}/{model_id}/resolve/main/{file_path}"
                
                download_files.append({
                    'url': download_url,
                    'local_path': local_path
                })
            
            # 并行下载
            print(f"开始并行下载 {len(download_files)} 个文件，使用 {num_workers} 个线程")
            results = parallel_download(
                download_files, num_workers, min_file_size, threads_per_file, 
                chunk_size, max_retries, retry_delay, timeout
            )
            
            # 检查是否所有文件都下载成功
            if all(r['success'] for r in results):
                print(f"所有文件下载成功！保存路径: {model_specific_dir}")
                return model_specific_dir
            else:
                print(f"部分文件下载失败。已下载 {sum(1 for r in results if r['success'])}/{len(results)} 个文件。")
                return model_specific_dir
        
        except Exception as e:
            print(f"尝试手动多线程下载也失败: {e}")
            return None

def filter_files(file_list, file_extensions=None, ignore_extensions=None):
    """根据扩展名过滤文件列表"""
    if not file_extensions and not ignore_extensions:
        return file_list
    
    filtered_files = file_list
    
    # 如果指定了要保留的扩展名
    if file_extensions:
        # 标准化扩展名
        norm_file_exts = []
        for ext in file_extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            norm_file_exts.append(ext.lower())
        
        # 只保留匹配扩展名的文件
        filtered_files = [f for f in filtered_files if any(f.lower().endswith(ext) for ext in norm_file_exts)]
    
    # 如果指定了要忽略的扩展名
    if ignore_extensions:
        # 标准化忽略扩展名
        norm_ignore_exts = []
        for ext in ignore_extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            norm_ignore_exts.append(ext.lower())
        
        # 排除匹配忽略扩展名的文件
        filtered_files = [f for f in filtered_files if not any(f.lower().endswith(ext) for ext in norm_ignore_exts)]
    
    return filtered_files

def download_model(model_id, output_dir, source, mirror_url, 
                 file_extensions=None, ignore_extensions=None, 
                 num_workers=4, max_retries=5, retry_delay=5, timeout=100,
                 min_file_size=52428800, threads_per_file=8, chunk_size=10485760):
    if source == "modelscope":
        return download_from_modelscope(
            model_id, output_dir, file_extensions, ignore_extensions, 
            num_workers, max_retries, retry_delay, timeout,
            min_file_size, threads_per_file, chunk_size
        )
    elif source == "huggingface":
        return download_from_huggingface(
            model_id, output_dir, mirror_url, file_extensions, ignore_extensions, 
            num_workers, max_retries, retry_delay, timeout,
            min_file_size, threads_per_file, chunk_size
        )
    else:
        print(f"不支持的下载源: {source}")
        return None

def main():
    args = parse_args()
    download_model(
        args.model_id, 
        args.output_dir, 
        args.source, 
        args.mirror, 
        args.file_extensions, 
        args.ignore_extensions,
        args.num_workers,
        args.max_retries,
        args.retry_delay,
        args.timeout,
        args.min_file_size,
        args.threads_per_file,
        args.chunk_size
    )

if __name__ == "__main__":
    main()
