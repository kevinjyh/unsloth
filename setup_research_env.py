#!/usr/bin/env python
"""
Unsloth 研究環境設置腳本
啟動適合代碼研究的環境，修復常見問題
"""

import os
import sys
from unsloth_patches import apply_patches
from simulate_cuda import setup_fake_cuda_env

def setup_research_environment():
    """設置用於研究的環境"""
    print("設置 Unsloth 研究環境...")
    
    # 1. 首先設置模擬 CUDA 環境
    original_cuda = setup_fake_cuda_env()
    
    # 2. 應用所有補丁
    apply_patches()
    
    # 3. 設置環境變量
    os.environ['DISABLE_UNSLOTH_FLASH_ATTN'] = '1'
    os.environ['UNSLOTH_DISABLE_FLASH_ATTN'] = '1'
    
    print("\n環境已準備就緒！您現在可以開始研究 Unsloth 代碼。")
    print("提示：在導入 unsloth 模組前，請確保已運行此腳本。")
    
    # 返回原始 CUDA 對象，以便日後需要時恢復
    return original_cuda

if __name__ == "__main__":
    # 設置環境
    original_cuda = setup_research_environment()
    
    # 如果是交互模式
    if sys.flags.interactive:
        print("\n您正處於交互模式。已設置好研究環境，可以開始導入和研究 unsloth 了。")
        print("完成後，可以使用以下代碼恢復原始環境：")
        print("from simulate_cuda import cleanup_fake_cuda_env")
        print("cleanup_fake_cuda_env(original_cuda)")
    else:
        print("\n要開始研究，請在 Python 交互環境中執行此腳本：")
        print("python -i setup_research_env.py") 