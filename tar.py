#!/usr/bin/env python3
import os
import argparse
import zipfile
import tarfile
import requests
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

# ################### 压缩功能 ####################

def compress_folder(input_dir, output_path, compression_format='tar.gz'):
    """压缩文件夹到指定格式（带进度条）"""
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()

    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录: {input_dir}")

    # 预扫描文件列表
    file_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            file_list.append(file_path)
    
    print(f"正在压缩: {input_dir} -> {output_path}")
    print(f"文件总数: {len(file_list)}")
    print(f"压缩格式: {compression_format}\n")

    try:
        with tqdm(total=len(file_list), desc="压缩进度", unit="file") as pbar:
            if compression_format == 'zip':
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in file_list:
                        rel_path = file_path.relative_to(input_dir)
                        zipf.write(file_path, arcname=rel_path)
                        pbar.update(1)
                        pbar.set_postfix_str(f"添加: {rel_path}")

            elif compression_format in ['tar', 'tar.gz']:
                mode = 'w:gz' if compression_format == 'tar.gz' else 'w'
                with tarfile.open(output_path, mode) as tar:
                    for file_path in file_list:
                        rel_path = file_path.relative_to(input_dir)
                        tar.add(file_path, arcname=rel_path)
                        pbar.update(1)
                        pbar.set_postfix_str(f"添加: {rel_path}")

        print(f"\n压缩成功！压缩包大小: {output_path.stat().st_size/1024/1024:.2f} MB")
    
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"压缩失败: {str(e)}") from e

# ################### 解压功能 ####################

def download_with_progress(url, dest_path):
    """带进度条的下载函数"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192

        with open(dest_path, 'wb') as f, tqdm(
            desc="下载进度",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        raise RuntimeError(f"下载失败: {str(e)}") from e

def handle_remote_file(url, output_dir):
    """处理远程压缩文件"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            temp_path = Path(tmp_file.name)
        
        print(f"开始下载远程文件: {url}")
        if not download_with_progress(url, temp_path):
            return False

        print(f"下载完成，临时文件: {temp_path}")
        return extract_local_file(temp_path, output_dir, is_temp=True)
    
    finally:
        if temp_path.exists():
            temp_path.unlink()
            print("已清理临时下载文件")

def extract_local_file(input_file, output_dir, is_temp=False):
    """解压本地文件"""
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_file.exists():
        raise ValueError(f"压缩文件不存在: {input_file}")

    # 自动识别格式
    suffixes = input_file.suffixes
    ext_map = {
        ('.zip',): 'zip',
        ('.tar', '.gz'): 'tar.gz',
        ('.tgz',): 'tar.gz',
        ('.tar',): 'tar'
    }

    matched_format = None
    for ext, fmt in ext_map.items():
        if tuple(suffixes[-len(ext):]) == ext:
            matched_format = fmt
            break

    if not matched_format:
        raise ValueError("不支持的压缩文件格式")

    print(f"识别格式: {matched_format.upper()}")
    print(f"解压到目录: {output_dir}\n")

    try:
        if matched_format == 'zip':
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                with tqdm(total=len(file_list), desc="解压进度", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, output_dir)
                        pbar.update(1)
                        pbar.set_postfix_str(f"解压: {file.filename}")
        
        elif matched_format in ['tar', 'tar.gz']:
            mode = 'r:gz' if matched_format == 'tar.gz' else 'r'
            with tarfile.open(input_file, mode) as tar_ref:
                members = tar_ref.getmembers()
                with tqdm(total=len(members), desc="解压进度", unit="file") as pbar:
                    for member in members:
                        tar_ref.extract(member, output_dir)
                        pbar.update(1)
                        pbar.set_postfix_str(f"解压: {member.name}")

        print(f"\n解压成功！目标目录大小: {sum(f.stat().st_size for f in output_dir.glob('**/*') if f.is_file())/1024/1024:.2f} MB")
        return True
    
    except Exception as e:
        if output_dir.exists() and not is_temp:
            shutil.rmtree(output_dir)
        raise RuntimeError(f"解压失败: {str(e)}") from e

def extract_file(input_path, output_dir):
    """统一解压入口"""
    if str(input_path).startswith(('http://', 'https://')):
        return handle_remote_file(input_path, output_dir)
    return extract_local_file(input_path, output_dir)

# ################### 命令行接口 ####################

def main():
    parser = argparse.ArgumentParser(
        description="智能压缩解压工具 (支持远程文件)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='操作指令')

    # 压缩命令
    compress_parser = subparsers.add_parser('compress', help='压缩文件夹')
    compress_parser.add_argument('input_dir', help='要压缩的文件夹路径')
    compress_parser.add_argument('output_file', help='输出压缩文件路径')
    compress_parser.add_argument('-f', '--format',
                                choices=['zip', 'tar', 'tar.gz'],
                                default='tar.gz',
                                help='压缩格式')

    # 解压命令
    extract_parser = subparsers.add_parser('extract', help='解压文件')
    extract_parser.add_argument('input_path', help='要解压的压缩文件路径或URL')
    extract_parser.add_argument('output_dir', help='解压目标目录路径')

    args = parser.parse_args()

    try:
        if args.command == 'compress':
            compress_folder(args.input_dir, args.output_file, args.format)
        elif args.command == 'extract':
            extract_file(args.input_path, args.output_dir)
    
    except Exception as e:
        print(f"\n错误: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()