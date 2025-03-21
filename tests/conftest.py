import os
import sys
import pytest
import torch

# 將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 設置低資源消耗的環境變量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def is_cuda_available_and_working():
    """檢查CUDA是否可用且運作正常"""
    if not torch.cuda.is_available():
        return False
        
    try:
        # 嘗試進行簡單的CUDA運算
        x = torch.rand(10, device="cuda")
        y = x + x
        return True
    except Exception as e:
        print(f"CUDA測試失敗: {e}")
        return False

@pytest.fixture(scope="session")
def cuda_device():
    """提供CUDA設備，如果不可用則跳過相關測試"""
    if is_cuda_available_and_working():
        return torch.device("cuda")
    pytest.skip("需要可用的CUDA設備")
    
@pytest.fixture(scope="session")
def cpu_device():
    """提供CPU設備"""
    return torch.device("cpu")
    
@pytest.fixture(autouse=True, scope="session")
def setup_testing_environment():
    """設置測試環境"""
    # 打印CUDA信息
    if torch.cuda.is_available():
        print(f"\nCUDA可用: 是")
        print(f"CUDA設備數量: {torch.cuda.device_count()}")
        print(f"當前CUDA設備: {torch.cuda.current_device()}")
        print(f"CUDA設備名稱: {torch.cuda.get_device_name()}")
        print(f"CUDA設備能力: {torch.cuda.get_device_capability()}")
        print(f"支援BF16: {torch.cuda.is_bf16_supported()}")
    else:
        print("\nCUDA不可用，僅執行CPU測試")
    
    # 添加需要的模組處理
    try:
        # 嘗試導入所需模組，若不存在則處理例外
        import flash_attn
    except ImportError:
        print("flash_attn 模組不可用，部分功能可能受限")
        
    yield
    
    # 清理CUDA緩存（測試結束後）
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 