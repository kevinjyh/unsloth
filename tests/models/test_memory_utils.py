import pytest
import torch
import os
import sys
import gc

# 將項目根目錄添加到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class TestMemoryUtilities:
    """測試記憶體管理相關功能"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA環境")
    def test_memory_efficient_operations(self):
        """測試記憶體高效的運算"""
        # 確保環境變量設置正確
        assert 'PYTORCH_CUDA_ALLOC_CONF' in os.environ
        
        # 清理快取
        torch.cuda.empty_cache()
        gc.collect()
        
        # 記錄初始記憶體
        initial_memory = torch.cuda.memory_allocated()
        print(f"初始CUDA記憶體使用量: {initial_memory} bytes")
        
        # 創建中等大小的張量並執行操作
        size = 1000
        try:
            x = torch.rand(size, size, device="cuda")
            y = torch.rand(size, size, device="cuda")
            
            # 記錄操作前記憶體
            after_alloc_memory = torch.cuda.memory_allocated()
            print(f"分配後CUDA記憶體使用量: {after_alloc_memory} bytes")
            print(f"增加量: {after_alloc_memory - initial_memory} bytes")
            
            # 執行矩陣乘法
            z = torch.matmul(x, y)
            
            # 記錄操作後記憶體
            after_op_memory = torch.cuda.memory_allocated()
            print(f"操作後CUDA記憶體使用量: {after_op_memory} bytes")
            print(f"增加量: {after_op_memory - after_alloc_memory} bytes")
            
            # 清理
            del x, y, z
            torch.cuda.empty_cache()
            gc.collect()
            
            # 最終記憶體狀態
            final_memory = torch.cuda.memory_allocated()
            print(f"最終CUDA記憶體使用量: {final_memory} bytes")
            
            # 確認記憶體成功釋放
            assert final_memory <= initial_memory * 1.1  # 允許有10%的誤差
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                pytest.skip("顯卡記憶體不足，跳過測試")
            else:
                raise
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA環境")
    def test_progressive_memory_allocation(self):
        """測試逐步分配記憶體時的使用情況"""
        torch.cuda.empty_cache()
        gc.collect()
        
        # 記錄初始記憶體
        initial_memory = torch.cuda.memory_allocated()
        
        # 逐步增加張量大小，觀察記憶體使用
        tensors = []
        max_memory = 0
        
        try:
            for i in range(1, 6):  # 由小到大逐步增加
                size = 200 * i  # 每次增加200
                print(f"\n創建大小為 {size}x{size} 的張量")
                
                tensor = torch.rand(size, size, device="cuda")
                tensors.append(tensor)
                
                current_memory = torch.cuda.memory_allocated()
                max_memory = max(max_memory, current_memory)
                
                print(f"當前記憶體: {current_memory} bytes")
                print(f"從初始增加: {current_memory - initial_memory} bytes")
            
            # 清理
            for tensor in tensors:
                del tensor
            tensors = []
            torch.cuda.empty_cache()
            gc.collect()
            
            # 最終記憶體
            final_memory = torch.cuda.memory_allocated()
            print(f"\n最大使用記憶體: {max_memory} bytes")
            print(f"最終記憶體: {final_memory} bytes")
            
            # 檢查記憶體已釋放
            assert final_memory <= initial_memory * 1.1  # 允許有10%的誤差
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # 清理所有張量
                for tensor in tensors:
                    del tensor
                tensors = []
                torch.cuda.empty_cache()
                gc.collect()
                
                print(f"在分配到 {len(tensors) + 1} 個張量時記憶體不足")
                pytest.skip("顯卡記憶體不足，無法完成全部測試") 