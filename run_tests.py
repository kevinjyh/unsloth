import os
import sys
from dotenv import load_dotenv
import unittest
import pytest
import torch

# 載入環境變數
load_dotenv()

# 設置低資源消耗的環境變量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def is_cuda_available_and_working():
    """檢查CUDA是否可用且功能正常"""
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return False
        
    try:
        # 嘗試執行一個簡單的CUDA運算
        x = torch.zeros(10, device="cuda")
        y = x + 1
        return True
    except Exception as e:
        print(f"CUDA測試失敗: {e}")
        return False

def print_system_info():
    """打印系統信息"""
    print("\n系統信息:")
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA是否可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"CUDA設備數量: {torch.cuda.device_count()}")
        print(f"當前CUDA設備: {torch.cuda.current_device()}")
        print(f"CUDA設備名稱: {torch.cuda.get_device_name()}")
        print(f"CUDA設備能力: {torch.cuda.get_device_capability()}")
        
        # 嘗試獲取cuDNN版本
        try:
            if torch.backends.cudnn.is_available():
                print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            else:
                print(f"cuDNN: 不可用")
        except:
            print(f"cuDNN: 檢測時發生錯誤")
    else:
        print("CUDA不可用，將使用CPU進行測試")

# 如果腳本被直接執行
if __name__ == "__main__":
    print_system_info()
    
    # 檢查CUDA是否正常工作
    cuda_working = is_cuda_available_and_working()
    if cuda_working:
        print("CUDA環境正常工作，將進行測試")
    else:
        print("CUDA環境不可用或無法正常工作，測試將在CPU上執行")
    
    # 運行pytest測試
    args = sys.argv[1:] or ["-xvs"]
    sys.exit(pytest.main(args)) 