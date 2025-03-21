import unittest
import torch
import os
import sys

# 需要將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 嘗試導入unsloth (可能會失敗，取決於項目結構)
try:
    from unsloth.models._utils import get_device
except ImportError:
    print("警告: 無法導入unsloth.models._utils，跳過相關測試")

class TestBasicEnvironment(unittest.TestCase):
    def test_torch_cuda_available(self):
        """測試模擬的CUDA環境是否正常工作"""
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.get_device_name(), "NVIDIA GeForce RTX Fake GPU")
    
    def test_environment_variables(self):
        """測試環境變量是否正確設置"""
        self.assertIn('CUDA_HOME', os.environ)
        self.assertIn('DISABLE_UNSLOTH_FLASH_ATTN', os.environ)
        self.assertEqual(os.environ['DISABLE_UNSLOTH_FLASH_ATTN'], '1')

if __name__ == '__main__':
    unittest.main() 