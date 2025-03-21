import os
import sys
import pytest
import torch
import importlib
from types import ModuleType

# 阻止 unsloth 檢測 GPU 環境
os.environ['UNSLOTH_DISABLE_GPU_CHECK'] = '1'
os.environ['DISABLE_UNSLOTH_FLASH_ATTN'] = '1'
os.environ['UNSLOTH_DISABLE_FLASH_ATTN'] = '1'

# 將项目根目錄添加到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)

# 模擬 BlockDiagonalCausalMask 類型
class MockBlockDiagonalCausalMask:
    """模擬 BlockDiagonalCausalMask 類型"""
    pass

# 模擬 CUDA
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

# 創建模擬模組
class MockModule(ModuleType):
    """模擬模組，用於替代無法導入的模組"""
    def __init__(self, name):
        super().__init__(name)
        self.__name__ = name
    
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(f"'{self.__name__}' has no attribute '{attr}'")
        return MockModule(f"{self.__name__}.{attr}")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """設置測試環境，這個 fixture 會在所有測試開始前自動運行"""
    # 保存原始的 torch.cuda
    original_cuda = torch.cuda
    
    # 應用模擬 CUDA
    torch.cuda = FakeCUDA()
    
    # 添加模擬模組
    sys.modules['xformers'] = MockModule('xformers')
    sys.modules['xformers.attn_bias'] = MockModule('xformers.attn_bias')
    sys.modules['xformers.attn_bias.BlockDiagonalCausalMask'] = MockBlockDiagonalCausalMask
    sys.modules['BlockDiagonalCausalMask'] = MockBlockDiagonalCausalMask
    
    # 如果 flash_attn 不存在，創建模擬模組
    if 'flash_attn' not in sys.modules:
        sys.modules['flash_attn'] = MockModule('flash_attn')
    
    yield
    
    # 恢復原始環境
    torch.cuda = original_cuda 