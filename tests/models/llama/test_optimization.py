import pytest
import torch
import sys
import os
import unittest.mock as mock

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 在導入unsloth前先導入補丁
from tests.models.llama.init_patch import patch_unsloth_init
patch_unsloth_init()

try:
    from unsloth.models.llama import (
        fast_swiglu_inference,
        fast_rms_layernorm_inference,
        fast_rms_layernorm_inference_gemma,
        fast_layernorm_compiled,
        LlamaModel_fast_forward_inference,
        CausalLM_fast_forward,
        unsloth_fast_generate,
        get_grouped_params,
        linear_warmup_cosine_decay,
        linear_warmup_constant,
        cosine_decay,
    )
except (ImportError, NotImplementedError) as e:
    print(f"無法導入 unsloth.models.llama 模組: {e}")
    print("使用模擬模組替代...")
    
    # 從 init_patch 中獲取模擬函數
    from unsloth.models.llama import (
        fast_swiglu_inference,
        fast_rms_layernorm_inference,
        fast_rms_layernorm_inference_gemma,
        fast_layernorm_compiled,
        LlamaModel_fast_forward_inference,
        CausalLM_fast_forward,
        unsloth_fast_generate,
        get_grouped_params,
        linear_warmup_cosine_decay,
        linear_warmup_constant,
        cosine_decay,
    )

class TestOptimizationFunctions:
    """測試Llama模型中的優化函數實現"""
    
    @pytest.fixture
    def setup_params(self):
        """設置測試的基本參數"""
        batch_size = 2
        seq_len = 32
        hidden_size = 64
        device = 'cpu'
        dtype = torch.float32
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'device': device,
            'dtype': dtype
        }
    
    def test_fast_swiglu_inference(self, setup_params):
        """測試快速SwiGLU激活函數計算"""
        params = setup_params
        
        # 創建模擬MLP模組
        mock_mlp = mock.MagicMock()
        mock_mlp.gate_proj = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'] * 2,
            device=params['device'], dtype=params['dtype']
        ))
        mock_mlp.up_proj = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'] * 2,
            device=params['device'], dtype=params['dtype']
        ))
        mock_mlp.down_proj = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'],
            device=params['device'], dtype=params['dtype']
        ))
        
        # 輸入張量
        X = torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'],
            device=params['device'], dtype=params['dtype']
        )
        
        # 測試fast_swiglu_inference
        output = fast_swiglu_inference(mock_mlp, X)
        
        # 驗證輸出形狀
        assert output.shape == X.shape
        
        # 驗證函數調用
        mock_mlp.gate_proj.assert_called_once()
        mock_mlp.up_proj.assert_called_once()
        mock_mlp.down_proj.assert_called_once()
    
    def test_fast_rms_layernorm_inference(self, setup_params):
        """測試快速RMS層歸一化計算"""
        params = setup_params
        
        # 創建模擬LayerNorm模組
        mock_layernorm = mock.MagicMock()
        mock_layernorm.weight = torch.ones(
            params['hidden_size'], 
            device=params['device'], 
            dtype=params['dtype']
        )
        mock_layernorm.variance_epsilon = 1e-6
        
        # 輸入張量
        X = torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'],
            device=params['device'], dtype=params['dtype']
        )
        
        # 測試fast_rms_layernorm_inference
        output = fast_rms_layernorm_inference(mock_layernorm, X)
        
        # 驗證輸出形狀
        assert output.shape == X.shape
    
    def test_fast_rms_layernorm_inference_gemma(self, setup_params):
        """測試Gemma模型的快速RMS層歸一化計算"""
        params = setup_params
        
        # 創建模擬LayerNorm模組
        mock_layernorm = mock.MagicMock()
        mock_layernorm.weight = torch.ones(
            params['hidden_size'], 
            device=params['device'], 
            dtype=params['dtype']
        )
        mock_layernorm.variance_epsilon = 1e-6
        
        # 輸入張量
        X = torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'],
            device=params['device'], dtype=params['dtype']
        )
        
        # 測試fast_rms_layernorm_inference_gemma
        output = fast_rms_layernorm_inference_gemma(mock_layernorm, X)
        
        # 驗證輸出形狀
        assert output.shape == X.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA環境")
    def test_fast_layernorm_compiled(self, setup_params):
        """測試編譯優化的層歸一化計算 (需要CUDA環境)"""
        params = setup_params
        
        # 在非CUDA環境中使用CPU進行測試
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 創建模擬LayerNorm模組
        mock_layernorm = mock.MagicMock()
        mock_layernorm.weight = torch.ones(
            params['hidden_size'], 
            device=device, 
            dtype=params['dtype']
        )
        mock_layernorm.variance_epsilon = 1e-6
        mock_layernorm.forward = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'],
            device=device, dtype=params['dtype']
        ))
        
        # 輸入張量
        X = torch.randn(
            params['batch_size'], params['seq_len'], params['hidden_size'],
            device=device, dtype=params['dtype']
        )
        
        # 測試fast_layernorm_compiled
        with mock.patch('torch.compile', return_value=lambda func: lambda x: func(x)) as mock_compile:
            output = fast_layernorm_compiled(mock_layernorm, X)
            
            # 驗證輸出形狀
            assert output.shape == X.shape or isinstance(output, mock.MagicMock)
            
            # 驗證編譯器調用
            mock_compile.assert_called_once()
            
            # 驗證forward調用 (需通過編譯後的函數調用)
            mock_layernorm.forward.assert_called_once_with(X)
    
    def test_llama_model_fast_forward_inference(self, setup_params):
        """測試LlamaModel快速推理的前向計算"""
        params = setup_params
        
        # 創建模擬LlamaModel
        mock_model = mock.MagicMock()
        mock_model.embed_tokens = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], 1, params['hidden_size'],
            device=params['device'], dtype=params['dtype']
        ))
        
        # 模擬模型層
        mock_model.layers = []
        for i in range(3):  # 創建3層
            layer = mock.MagicMock()
            layer.self_attn = mock.MagicMock()
            # 設置fast_forward_inference方法
            layer.self_attn.fast_forward_inference = mock.MagicMock(return_value=(
                torch.randn(
                    params['batch_size'], 1, params['hidden_size'],
                    device=params['device'], dtype=params['dtype']
                ),
                (torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1))
            ))
            mock_model.layers.append(layer)
        
        # 設置模型的處理層
        mock_model.norm = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], 1, params['hidden_size'],
            device=params['device'], dtype=params['dtype']
        ))
        
        # 輸入數據
        input_ids = torch.randint(0, 1000, (params['batch_size'], 1), device=params['device'])
        position_ids = torch.zeros((params['batch_size'], 1), dtype=torch.long, device=params['device'])
        past_key_values = [(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)) for _ in range(3)]
        
        # 測試LlamaModel_fast_forward_inference
        output, new_past_key_values = LlamaModel_fast_forward_inference(
            mock_model,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids
        )
        
        # 驗證輸出
        assert output.shape == (params['batch_size'], 1, params['hidden_size'])
        assert len(new_past_key_values) == len(mock_model.layers)
    
    def test_causal_lm_fast_forward(self, setup_params):
        """測試CausalLM的快速前向函數工廠方法"""
        params = setup_params
        
        # 創建模擬前向推理函數
        mock_fast_forward_inference = mock.MagicMock(return_value=(
            torch.randn(
                params['batch_size'], 1, params['hidden_size'],
                device=params['device'], dtype=params['dtype']
            ),
            [(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)) for _ in range(3)]
        ))
        
        # 創建模擬CausalLM模型
        mock_model = mock.MagicMock()
        mock_model.model = mock.MagicMock()
        mock_model.lm_head = mock.MagicMock(return_value=torch.randn(
            params['batch_size'], 1, 1000,  # 輸出詞彙大小為1000
            device=params['device'], dtype=params['dtype']
        ))
        
        # 生成CausalLM_fast_forward函數
        fast_forward_func = CausalLM_fast_forward(mock_fast_forward_inference)
        
        # 輸入數據
        input_ids = torch.randint(0, 1000, (params['batch_size'], 1), device=params['device'])
        position_ids = torch.zeros((params['batch_size'], 1), dtype=torch.long, device=params['device'])
        past_key_values = [(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)) for _ in range(3)]
        
        # 測試生成的快速前向函數
        output = fast_forward_func(
            mock_model,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True
        )
        
        # 驗證函數調用
        mock_fast_forward_inference.assert_called_once()
        mock_model.lm_head.assert_called_once()
    
    def test_unsloth_fast_generate(self, setup_params):
        """測試unsloth_fast_generate方法，用於加速文本生成"""
        params = setup_params
        
        # 創建模擬模型
        mock_model = mock.MagicMock()
        mock_model._old_generate = mock.MagicMock(return_value=["生成的文本"])
        
        # 修補函數
        with mock.patch('unsloth.models.llama.FastLlamaModel.for_inference'):
            with mock.patch('unsloth.models.llama.FastLlamaModel.for_training'):
                with mock.patch('torch.inference_mode'):
                    with mock.patch('torch.autocast'):
                        # 調用unsloth_fast_generate
                        output = unsloth_fast_generate(
                            mock_model,
                            input_ids=torch.randint(0, 1000, (params['batch_size'], 10), device=params['device']),
                            max_length=20
                        )
                        
                        # 驗證結果 (允許返回列表或字符串)
                        assert output == ["生成的文本"] or output == "生成的文本"
                        mock_model._old_generate.assert_called_once()

    def setup_method(self):
        # 創建一個簡單的模型來測試參數分組
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.bias = torch.nn.Parameter(torch.zeros(10))
                self.layernorm = torch.nn.LayerNorm(10)
                
            def named_parameters(self):
                return [
                    ("linear.weight", self.linear.weight),
                    ("linear.bias", self.linear.bias),
                    ("bias", self.bias),
                    ("layernorm.weight", self.layernorm.weight),
                    ("layernorm.bias", self.layernorm.bias)
                ]
        
        self.model = SimpleModel()
    
    def test_get_grouped_params(self):
        """測試參數分組函數"""
        # 導入要測試的函數
        from unsloth.models.llama.optimization import get_grouped_params
        
        # 測試默認參數
        param_groups = get_grouped_params(self.model, weight_decay=0.01)
        
        # 驗證有兩個參數組
        assert len(param_groups) == 2
        
        # 驗證權重衰減設置
        assert param_groups[0]["weight_decay"] == 0.01
        assert param_groups[1]["weight_decay"] == 0.0
        
    def test_linear_warmup_cosine_decay(self):
        """測試線性預熱餘弦衰減學習率調度器"""
        # 導入要測試的函數
        from unsloth.models.llama.optimization import linear_warmup_cosine_decay
        
        # 測試預熱階段
        lr_warmup = linear_warmup_cosine_decay(
            step=5, 
            max_steps=100, 
            warmup_steps=10, 
            learning_rate=0.001,
            min_lr=1e-5,
            warmup_lr=0.0
        )
        
        # 預熱階段應該是線性增長
        expected_warmup = 0.0 + 5 * (0.001 - 0.0) / 10
        assert abs(lr_warmup - expected_warmup) < 1e-6
        
        # 測試衰減階段
        lr_decay = linear_warmup_cosine_decay(
            step=50, 
            max_steps=100, 
            warmup_steps=10, 
            learning_rate=0.001,
            min_lr=1e-5
        )
        
        # 驗證衰減階段的學習率
        assert lr_decay <= 0.001
        assert lr_decay >= 1e-5
        
    def test_linear_warmup_constant(self):
        """測試線性預熱恆定學習率調度器"""
        # 導入要測試的函數
        from unsloth.models.llama.optimization import linear_warmup_constant
        
        # 測試預熱階段
        lr_warmup = linear_warmup_constant(
            step=5, 
            max_steps=100, 
            warmup_steps=10, 
            learning_rate=0.001
        )
        
        # 預熱階段應該是線性增長
        expected_warmup = 0.0 + 5 * (0.001 - 0.0) / 10
        assert abs(lr_warmup - expected_warmup) < 1e-6
        
        # 測試恆定階段
        lr_constant = linear_warmup_constant(
            step=50, 
            max_steps=100, 
            warmup_steps=10, 
            learning_rate=0.001
        )
        
        # 恆定階段學習率應該保持不變
        assert lr_constant == 0.001
        
    def test_cosine_decay(self):
        """測試餘弦衰減學習率調度器"""
        # 導入要測試的函數
        from unsloth.models.llama.optimization import cosine_decay
        
        # 測試衰減開始階段
        lr_start = cosine_decay(
            step=0, 
            max_steps=100, 
            learning_rate=0.001,
            min_lr=1e-5
        )
        
        # 開始時應該是最大學習率
        assert lr_start == 0.001
        
        # 測試衰減中間階段
        lr_middle = cosine_decay(
            step=50, 
            max_steps=100, 
            learning_rate=0.001,
            min_lr=1e-5
        )
        
        # 中間應該介於最大和最小學習率之間
        assert lr_middle < 0.001
        assert lr_middle > 1e-5
        
        # 測試衰減結束階段
        lr_end = cosine_decay(
            step=100, 
            max_steps=100, 
            learning_rate=0.001,
            min_lr=1e-5
        )
        
        # 結束時應該是最小學習率
        assert abs(lr_end - 1e-5) < 1e-6 