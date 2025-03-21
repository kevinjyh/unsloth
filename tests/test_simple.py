import os
import sys
import unittest
import torch

# 將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSimpleEnvironment(unittest.TestCase):
    """簡單測試環境設置"""
    
    def test_environment_variables(self):
        """測試環境變量設置"""
        self.assertIn('CUDA_HOME', os.environ)
        self.assertIn('DISABLE_UNSLOTH_FLASH_ATTN', os.environ)
        self.assertEqual(os.environ['DISABLE_UNSLOTH_FLASH_ATTN'], '1')
    
    def test_cuda_mock(self):
        """測試CUDA模擬是否工作"""
        self.assertTrue(hasattr(torch, 'cuda'))
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.device_count(), 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertEqual(torch.cuda.get_device_name(), "NVIDIA GeForce RTX Fake GPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 0))
        self.assertTrue(torch.cuda.is_bf16_supported())

if __name__ == '__main__':
    unittest.main() 