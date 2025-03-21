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
os.environ['UNSLOTH_DISABLE_GPU_CHECK'] = '1'  # 新增: 禁用 GPU 檢查

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

# 創建用於修補初始化檢查的函數
def patch_unsloth_init():
    """修補 unsloth 的初始化檢查"""
    try:
        # 添加模擬模組到系統模組中
        import importlib
        from types import ModuleType
        
        class MockModule(ModuleType):
            """模擬模組，用於替代無法導入的模組"""
            def __init__(self, name):
                super().__init__(name)
                self.__name__ = name
            
            def __getattr__(self, attr):
                if attr.startswith('__'):
                    raise AttributeError(f"'{self.__name__}' has no attribute '{attr}'")
                return MockModule(f"{self.__name__}.{attr}")
        
        # 模擬不存在的或需要的模組
        sys.modules['flash_attn'] = MockModule('flash_attn')
        sys.modules['xformers'] = MockModule('xformers')
        
        print("已添加模擬模組")
    except Exception as e:
        print(f"修補時發生錯誤: {e}")

@pytest.fixture(autouse=True, scope="session")
def mock_cuda():
    """自動模擬CUDA環境，在所有測試用例中生效"""
    # 替換為模擬版本
    torch.cuda = FakeCUDA()
    print("\n模擬CUDA環境已設置")
    
    # 應用修補
    patch_unsloth_init()
    
    yield
    
    # 恢復原始版本（測試結束後）
    torch.cuda = original_cuda 