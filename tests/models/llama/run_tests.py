#!/usr/bin/env python
"""
Llama模型測試運行腳本

此腳本用於設置測試環境並運行所有Llama模型相關的測試。
它會在測試前設置必要的環境變量以禁用GPU要求，並模擬必要的資源。
"""

import os
import sys
import subprocess
import argparse

# 設置環境變量
os.environ['UNSLOTH_DISABLE_GPU_CHECK'] = '1'
os.environ['DISABLE_UNSLOTH_FLASH_ATTN'] = '1'
os.environ['UNSLOTH_DISABLE_FLASH_ATTN'] = '1'
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', 'C:/fake_cuda')

def run_tests(test_files=None, verbose=True):
    """
    運行指定的測試文件，或全部測試
    
    Args:
        test_files: 要運行的測試文件列表，如果為None則運行所有測試
        verbose: 是否顯示詳細輸出
    """
    # 設置測試目錄
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 構建命令
    cmd = [sys.executable, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    
    if test_files:
        # 運行指定的測試文件
        for file in test_files:
            file_path = os.path.join(test_dir, file)
            cmd_with_file = cmd + [file_path]
            print(f"運行測試: {file}")
            subprocess.run(cmd_with_file)
    else:
        # 運行所有測試文件
        all_files = [
            "test_rotary_embeddings.py",
            "test_attention.py",
            "test_fast_llama_model.py",
            "test_optimization.py",
            "test_integration.py"
        ]
        
        for file in all_files:
            file_path = os.path.join(test_dir, file)
            cmd_with_file = cmd + [file_path]
            print(f"\n運行測試: {file}")
            subprocess.run(cmd_with_file)
    
    print("\n所有測試完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="運行Llama模型測試")
    parser.add_argument('--test-file', nargs='*', help='指定要運行的測試文件')
    parser.add_argument('--quiet', action='store_true', help='禁用詳細輸出')
    
    args = parser.parse_args()
    
    # 確保fake_cuda目錄存在
    cuda_dir = os.environ['CUDA_HOME']
    if not os.path.exists(cuda_dir):
        os.makedirs(cuda_dir, exist_ok=True)
    
    run_tests(args.test_file, not args.quiet) 