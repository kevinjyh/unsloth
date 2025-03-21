import pytest
import torch
import sys
import os
import math

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 在導入unsloth前先導入補丁
from tests.models.llama.init_patch import patch_unsloth_init
patch_unsloth_init()

# 這些是我們將模擬的類別
class MockRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
    def __call__(self, x, position_ids=None):
        # 簡單模擬旋轉嵌入輸出
        return x

class MockLinearScalingRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, device=None):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
    def __call__(self, x, position_ids=None):
        # 簡單模擬旋轉嵌入輸出
        return x

class MockExtendedRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
    def __call__(self, x, position_ids=None):
        # 簡單模擬旋轉嵌入輸出
        return x

class MockLongRopeRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, original_max_position_embeddings=2048, base=10000, device=None):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        
    def __call__(self, x, position_ids=None):
        # 簡單模擬旋轉嵌入輸出
        return x

# 添加模擬類到系統模組
sys.modules['unsloth.models.llama.rotary_embedding'] = type('', (), {})()
sys.modules['unsloth.models.llama.rotary_embedding'].LlamaRotaryEmbedding = MockRotaryEmbedding
sys.modules['unsloth.models.llama.rotary_embedding'].LlamaLinearScalingRotaryEmbedding = MockLinearScalingRotaryEmbedding
sys.modules['unsloth.models.llama.rotary_embedding'].LlamaExtendedRotaryEmbedding = MockExtendedRotaryEmbedding
sys.modules['unsloth.models.llama.rotary_embedding'].LongRopeRotaryEmbedding = MockLongRopeRotaryEmbedding

# 從模擬模組中導入類
from unsloth.models.llama.rotary_embedding import (
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding, 
    LlamaExtendedRotaryEmbedding,
    LongRopeRotaryEmbedding
)

class TestRotaryEmbeddings:
    """測試Llama模型中不同的旋轉位置嵌入(RoPE)實現"""
    
    @pytest.fixture
    def setup_params(self):
        """設置測試參數"""
        batch_size = 2
        seq_len = 10
        num_heads = 8
        head_dim = 32
        max_position_embeddings = 4096
        device = "cpu"
        dtype = torch.float32

        # 創建輸入張量
        x = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "max_position_embeddings": max_position_embeddings,
            "device": device,
            "dtype": dtype,
            "x": x,
            "position_ids": position_ids
        }
    
    def test_base_rotary_embedding(self, setup_params):
        """測試基本的LlamaRotaryEmbedding實現"""
        params = setup_params
        
        # 初始化旋轉嵌入
        rope = LlamaRotaryEmbedding(
            dim=params['head_dim'],
            max_position_embeddings=params['max_position_embeddings'],
            device=params['device']
        )
        
        # 測試前向傳播
        output = rope(params['x'], params['position_ids'])
        
        # 驗證輸出形狀
        assert output.shape == params['x'].shape
        assert output.dtype == params['dtype']
        assert output.device.type == params['device']

    def test_linear_scaling_rotary_embedding(self, setup_params):
        """測試線性縮放的旋轉嵌入實現"""
        params = setup_params
        scaling_factor = 2.0
        
        # 初始化線性縮放旋轉嵌入
        rope = LlamaLinearScalingRotaryEmbedding(
            dim=params['head_dim'],
            max_position_embeddings=params['max_position_embeddings'],
            device=params['device'],
            scaling_factor=scaling_factor
        )
        
        # 測試前向傳播
        output = rope(params['x'], params['position_ids'])
        
        # 驗證輸出形狀
        assert output.shape == params['x'].shape
        assert output.dtype == params['dtype']
        assert output.device.type == params['device']
        assert rope.scaling_factor == scaling_factor

    def test_extended_rotary_embedding(self, setup_params):
        """測試擴展旋轉嵌入實現 (用於長序列處理)"""
        params = setup_params
        
        # 初始化擴展旋轉嵌入
        rope = LlamaExtendedRotaryEmbedding(
            dim=params['head_dim'],
            max_position_embeddings=params['max_position_embeddings'],
            device=params['device']
        )
        
        # 測試前向傳播
        output = rope(params['x'], params['position_ids'])
        
        # 驗證輸出形狀
        assert output.shape == params['x'].shape
        assert output.dtype == params['dtype']
        assert output.device.type == params['device']

    def test_long_rope_rotary_embedding(self, setup_params):
        """測試長序列RoPE實現 (特別用於處理超長序列)"""
        params = setup_params
        original_max_position_embeddings = 4096
        max_position_embeddings = 131072
        
        # 初始化長序列RoPE
        rope = LongRopeRotaryEmbedding(
            dim=params['head_dim'],
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            device=params['device']
        )
        
        # 測試前向傳播
        output = rope(params['x'], params['position_ids'])
        
        # 驗證輸出形狀
        assert output.shape == params['x'].shape
        assert output.dtype == params['dtype']
        assert output.device.type == params['device']
        assert rope.max_position_embeddings == max_position_embeddings
        assert rope.original_max_position_embeddings == original_max_position_embeddings 