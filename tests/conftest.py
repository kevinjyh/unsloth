import os
import sys
import pytest
import torch

# 將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 設置環境變量
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', 'C:/fake_cuda')
os.environ['DISABLE_UNSLOTH_FLASH_ATTN'] = '1'
os.environ['UNSLOTH_DISABLE_FLASH_ATTN'] = '1'

# 確保fake_cuda目錄存在
fake_cuda_dir = os.environ['CUDA_HOME']
if not os.path.exists(fake_cuda_dir):
    os.makedirs(fake_cuda_dir, exist_ok=True)

# 模擬CUDA
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

# 保存原始的torch.cuda
original_cuda = torch.cuda

@pytest.fixture(autouse=True, scope="session")
def mock_cuda():
    """自動模擬CUDA環境，在所有測試用例中生效"""
    # 替換為模擬版本
    torch.cuda = FakeCUDA()
    print("\n模擬CUDA環境已設置")
    
    yield
    
    # 恢復原始版本（測試結束後）
    torch.cuda = original_cuda 