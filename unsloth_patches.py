"""
Unsloth 修補程式
用於修復開發環境中的一些常見問題
"""

import sys
import importlib.util
from types import ModuleType
from typing import Optional, List, Tuple, Union

class MockModule(ModuleType):
    """模擬模組，用於替代無法導入的模組"""
    def __init__(self, name):
        super().__init__(name)
        self.__name__ = name
    
    def __getattr__(self, attr):
        # 返回一個空函數或模擬對象
        if attr.startswith('__'):
            raise AttributeError(f"'{self.__name__}' has no attribute '{attr}'")
        
        # 對於token模組，特別處理get_token函數
        if self.__name__ == 'huggingface_hub.utils.token' and attr == 'get_token':
            # 導入huggingface_hub並嘗試找到正確的get_token路徑
            try:
                import huggingface_hub
                if hasattr(huggingface_hub.utils, 'get_token'):
                    return huggingface_hub.utils.get_token
                elif hasattr(huggingface_hub.utils, '_token') and hasattr(huggingface_hub.utils._token, 'get_token'):
                    return huggingface_hub.utils._token.get_token
            except:
                pass
            
            # 如果無法找到，返回一個虛擬函數
            def mock_get_token(*args, **kwargs):
                return None
            return mock_get_token
        
        # 對於flash_attn，返回一個模擬模組
        if self.__name__ == 'flash_attn':
            return MockModule(f"{self.__name__}.{attr}")
        
        # 一般情況下返回一個虛擬對象
        return MockModule(f"{self.__name__}.{attr}")

# 用於修復缺少的 BlockDiagonalCausalMask 型別
class MockBlockDiagonalCausalMask:
    """模擬 BlockDiagonalCausalMask 類型"""
    pass

def fix_type_annotations():
    """修復型別註解問題"""
    
    # 檢查是否已導入 xformers
    if 'xformers' in sys.modules and sys.modules['xformers'] is not None:
        xformers = sys.modules['xformers']
        
        # 確保 BlockDiagonalCausalMask 可用
        if not hasattr(sys.modules, 'BlockDiagonalCausalMask'):
            if hasattr(xformers, 'attn_bias') and hasattr(xformers.attn_bias, 'BlockDiagonalCausalMask'):
                sys.modules['BlockDiagonalCausalMask'] = xformers.attn_bias.BlockDiagonalCausalMask
            else:
                sys.modules['BlockDiagonalCausalMask'] = MockBlockDiagonalCausalMask
    else:
        # 如果沒有 xformers，則使用模擬類型
        sys.modules['BlockDiagonalCausalMask'] = MockBlockDiagonalCausalMask

def fix_fast_inner_training_loop():
    """修復 _fast_inner_training_loop 問題"""
    
    # 檢查是否已導入 transformers
    if 'transformers' in sys.modules:
        transformers = sys.modules['transformers']
        
        # 檢查 Trainer 類
        if hasattr(transformers, 'Trainer'):
            Trainer = transformers.Trainer
            
            # 如果 _fast_inner_training_loop 尚未定義
            if not hasattr(globals(), '_fast_inner_training_loop'):
                # 創建一個模擬函數
                def _fast_inner_training_loop(*args, **kwargs):
                    # 調用原始的 _inner_training_loop
                    if hasattr(Trainer, '_inner_training_loop'):
                        return Trainer._inner_training_loop(*args, **kwargs)
                    return None
                
                # 將其添加到全局命名空間
                globals()['_fast_inner_training_loop'] = _fast_inner_training_loop

def apply_patches():
    """應用所有補丁"""
    
    # 1. 修復 huggingface_hub.utils.token 問題
    if 'huggingface_hub.utils.token' not in sys.modules:
        sys.modules['huggingface_hub.utils.token'] = MockModule('huggingface_hub.utils.token')
    
    # 2. 修復 flash_attn 問題
    if 'flash_attn' not in sys.modules:
        sys.modules['flash_attn'] = MockModule('flash_attn')
    
    # 3. 修復型別註解問題
    fix_type_annotations()
    
    # 4. 修復 _fast_inner_training_loop 問題
    fix_fast_inner_training_loop()
    
    print("Unsloth 補丁已應用！")

if __name__ == "__main__":
    apply_patches() 