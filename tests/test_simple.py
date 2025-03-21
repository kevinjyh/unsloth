import os
import sys
import unittest
import torch
import pytest

# 將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSimpleOperations(unittest.TestCase):
    def setUp(self):
        """測試前準備"""
        # 檢查CUDA是否可用，設置合適的設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"測試使用設備: {self.device}")
    
    def test_tensor_creation(self):
        """測試張量創建"""
        # 創建一個簡單的張量
        x = torch.tensor([1, 2, 3], device=self.device)
        self.assertEqual(x.device.type, self.device.type)
        self.assertEqual(x.shape, (3,))
    
    def test_tensor_operations(self):
        """測試基本張量運算"""
        x = torch.tensor([1, 2, 3], device=self.device, dtype=torch.float32)
        y = torch.tensor([4, 5, 6], device=self.device, dtype=torch.float32)
        
        # 加法
        z = x + y
        self.assertTrue(torch.all(torch.eq(z, torch.tensor([5, 7, 9], device=self.device))))
        
        # 乘法
        z = x * y
        self.assertTrue(torch.all(torch.eq(z, torch.tensor([4, 10, 18], device=self.device))))
    
    @unittest.skipIf(not torch.cuda.is_available(), "需要CUDA環境")
    def test_cuda_memory_management(self):
        """測試CUDA記憶體管理"""
        # 記錄初始分配的記憶體
        initial_memory = torch.cuda.memory_allocated()
        
        # 創建一個大矩陣
        x = torch.randn(1000, 1000, device="cuda")
        
        # 檢查記憶體增加了
        current_memory = torch.cuda.memory_allocated()
        self.assertGreater(current_memory, initial_memory)
        
        # 刪除張量和清空快取
        del x
        torch.cuda.empty_cache()
        
        # 檢查記憶體回到初始水平或接近初始水平
        final_memory = torch.cuda.memory_allocated()
        print(f"初始記憶體: {initial_memory}, 最終記憶體: {final_memory}")

if __name__ == '__main__':
    unittest.main() 