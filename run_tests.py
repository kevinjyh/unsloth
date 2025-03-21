import os
import sys
from dotenv import load_dotenv
import unittest
import pytest

# 載入環境變數
load_dotenv()

# 模擬CUDA環境
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', 'C:/fake_cuda')
os.environ['DISABLE_UNSLOTH_FLASH_ATTN'] = '1'
os.environ['UNSLOTH_DISABLE_FLASH_ATTN'] = '1'

# 創建假的CUDA目錄（如果不存在）
fake_cuda_dir = os.environ['CUDA_HOME']
if not os.path.exists(fake_cuda_dir):
    os.makedirs(fake_cuda_dir, exist_ok=True)

# 在導入torch之前設置環境
import torch

# 重寫torch.cuda函數，避免實際檢查CUDA
class FakeCUDA:
    @staticmethod
    def is_available():
        return True
    
    @staticmethod
    def device_count():
        return 1
    
    @staticmethod
    def current_device():
        return 0
    
    @staticmethod
    def get_device_name(device=None):
        return "NVIDIA GeForce RTX Fake GPU"
    
    @staticmethod
    def get_device_capability(device=None):
        return (8, 0)  # 模擬RTX 3090等Ampere架構的能力
    
    @staticmethod
    def is_bf16_supported():
        return True
    
    @staticmethod
    def is_fp16_supported():
        return True
    
    @staticmethod
    def get_rng_state(device=None):
        return torch.ByteTensor(8)
    
    @staticmethod
    def get_rng_state_all():
        return [torch.ByteTensor(8)]
    
    @staticmethod
    def set_rng_state(new_state, device=None):
        pass
    
    @staticmethod
    def set_rng_state_all(new_states):
        pass
    
    @staticmethod
    def synchronize(device=None):
        pass
    
    @staticmethod
    def empty_cache():
        pass
    
    @staticmethod
    def memory_allocated(device=None):
        return 0
    
    @staticmethod
    def max_memory_allocated(device=None):
        return 0
    
    class Stream:
        def __init__(self, device=None):
            pass
        
        def wait_stream(self, stream):
            pass

# 備份原始的torch.cuda
original_cuda = torch.cuda

# 替換為模擬版本
torch.cuda = FakeCUDA()

# 如果腳本被直接執行
if __name__ == "__main__":
    print("模擬CUDA環境已設置")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA設備數量: {torch.cuda.device_count()}")
    print(f"當前CUDA設備: {torch.cuda.current_device()}")
    print(f"CUDA設備名稱: {torch.cuda.get_device_name()}")
    print(f"CUDA設備能力: {torch.cuda.get_device_capability()}")
    print(f"支援BF16: {torch.cuda.is_bf16_supported()}")
    
    # 運行pytest測試
    args = sys.argv[1:] or ["-xvs"]
    sys.exit(pytest.main(args)) 