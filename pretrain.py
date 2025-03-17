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

# 设置日志级别，便于查看详细信息
logging.basicConfig(level=logging.INFO)

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
    return parser.parse_args()

def download_from_modelscope(model_id, output_dir, file_extensions=None, ignore_extensions=None):
    print(f"正在从魔搭下载模型 {model_id} 到 {output_dir}...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 先下载所有文件
        model_dir = ms_snapshot_download(model_id, cache_dir=output_dir)
        
        # 处理文件过滤
        if file_extensions or ignore_extensions:
            temp_dir = model_dir + "_temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 获取所有文件
            all_files = []
            for root, _, files in os.walk(model_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            # 确定要保留的文件
            files_to_keep = []
            
            # 如果指定了要保留的扩展名
            if file_extensions:
                print(f"过滤文件后缀: 只保留 {', '.join(file_extensions)}")
                # 创建匹配模式列表
                patterns = []
                for ext in file_extensions:
                    # 确保扩展名以点号开头
                    if not ext.startswith('.'):
                        ext = '.' + ext
                    patterns.append(f"*{ext}")
                
                # 找到所有匹配的文件
                for pattern in patterns:
                    files_to_keep.extend(glob.glob(os.path.join(model_dir, "**", pattern), recursive=True))
            else:
                # 如果没有指定要保留的扩展名，则默认保留所有文件
                files_to_keep = all_files
            
            # 如果指定了要忽略的扩展名
            if ignore_extensions:
                print(f"过滤文件后缀: 忽略 {', '.join(ignore_extensions)}")
                # 标准化忽略扩展名
                norm_ignore_exts = []
                for ext in ignore_extensions:
                    if not ext.startswith('.'):
                        ext = '.' + ext
                    norm_ignore_exts.append(ext.lower())
                
                # 从要保留的文件中移除忽略的文件
                files_to_keep = [f for f in files_to_keep if not any(f.lower().endswith(ext) for ext in norm_ignore_exts)]
            
            # 如果没有找到要保留的文件，给出警告
            if not files_to_keep:
                print(f"警告: 根据过滤条件，没有找到需要保留的文件")
                return model_dir
            
            # 复制匹配的文件到临时目录，保持相对路径结构
            for file_path in files_to_keep:
                rel_path = os.path.relpath(file_path, model_dir)
                dest_path = os.path.join(temp_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
                print(f"已复制文件: {rel_path}")
            
            # 删除原始目录并移动临时目录
            shutil.rmtree(model_dir)
            shutil.move(temp_dir, model_dir)
            
        print(f"模型下载成功！保存路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"从魔搭下载模型时出错: {e}")
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

def download_from_huggingface(model_id, output_dir, mirror_url, file_extensions=None, ignore_extensions=None):
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
            "endpoint": mirror_url  # 直接在参数中指定端点
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
            test_response = requests.get(test_url, timeout=10)
            print(f"镜像站测试结果: 状态码 {test_response.status_code}")
        except Exception as e:
            print(f"镜像站测试失败: {e}")
        
        # 使用snapshot_download下载仓库
        print(f"开始下载，请检查下面的URL确认是否使用了镜像站 {mirror_url}...")
        downloaded_path = hf_snapshot_download(**download_kwargs)
        
        print(f"模型下载成功！保存路径: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"从Hugging Face镜像站下载模型时出错: {e}")
        print(f"错误详情: {str(e)}")
        return None

def download_model(model_id, output_dir, source, mirror_url, file_extensions=None, ignore_extensions=None):
    if source == "modelscope":
        return download_from_modelscope(model_id, output_dir, file_extensions, ignore_extensions)
    elif source == "huggingface":
        return download_from_huggingface(model_id, output_dir, mirror_url, file_extensions, ignore_extensions)
    else:
        print(f"不支持的下载源: {source}")
        return None

def main():
    args = parse_args()
    download_model(args.model_id, args.output_dir, args.source, args.mirror, 
                  args.file_extensions, args.ignore_extensions)

if __name__ == "__main__":
    main()
