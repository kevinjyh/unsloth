import pytest
import torch
import os
import sys

# 將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from unsloth.models.loader import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("無法導入Unsloth模組，跳過相關測試")

@pytest.mark.skipif(not UNSLOTH_AVAILABLE, reason="Unsloth未正確安裝")
class TestModelLoader:
    """測試模型加載器基本功能"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA環境")
    def test_cuda_environment(self):
        """測試CUDA環境設置"""
        # 確認CUDA可用
        assert torch.cuda.is_available()
        
        # 創建一個小型張量並測試基本運算
        x = torch.ones(10, device="cuda")
        y = x + 1
        assert torch.all(y == 2)
    
    def test_model_utils_import(self):
        """測試能否導入Unsloth模型工具函數"""
        try:
            # 測試能否導入關鍵工具函數
            from unsloth.models._utils import get_device
            assert callable(get_device)
        except ImportError as e:
            pytest.skip(f"無法導入模型工具函數: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA環境")
    def test_device_detection(self):
        """測試設備檢測函數"""
        try:
            from unsloth.models._utils import get_device
            device = get_device()
            assert isinstance(device, torch.device)
            assert device.type in ["cuda", "cpu"]
            
            # 如果CUDA可用，應該返回CUDA設備
            if torch.cuda.is_available():
                assert device.type == "cuda"
        except ImportError:
            pytest.skip("無法導入get_device函數")
        except Exception as e:
            pytest.fail(f"設備檢測失敗: {e}") 