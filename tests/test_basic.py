import unittest
import torch
import os
import sys
import pytest

# 需要將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 嘗試導入unsloth (可能會失敗，取決於項目結構)
try:
    from unsloth.models._utils import get_device
except ImportError:
    print("警告: 無法導入unsloth.models._utils，跳過相關測試")

class TestBasicEnvironment(unittest.TestCase):
    def test_torch_available(self):
        """測試PyTorch版本和環境"""
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '不可用'}")
    
    @unittest.skipIf(not torch.cuda.is_available(), "需要CUDA環境")
    def test_cuda_functionality(self):
        """測試CUDA功能是否正常工作"""
        # 創建CUDA張量並進行簡單運算
        x = torch.rand(100, 100, device="cuda")
        y = torch.rand(100, 100, device="cuda")
        z = x @ y  # 矩陣乘法
        self.assertEqual(z.shape, (100, 100))
        self.assertEqual(z.device.type, "cuda")
    
    def test_environment_setup(self):
        """測試環境設置是否適合低資源CUDA環境"""
        self.assertIn('PYTORCH_CUDA_ALLOC_CONF', os.environ)
        
        # 如果CUDA可用，測試記憶體分配
        if torch.cuda.is_available():
            # 分配一個小型張量
            x = torch.rand(10, 10, device="cuda")
            # 確認記憶體已分配但數量很小
            allocated_memory = torch.cuda.memory_allocated()
            print(f"已分配CUDA記憶體: {allocated_memory} bytes")
            self.assertGreater(allocated_memory, 0)
            # 清理
            del x
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main() 