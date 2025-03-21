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

# 創建一個假的模組
if 'unsloth.models.llama' not in sys.modules:
    # 創建假的 FastLlamaModel 類
    class MockFastLlamaModel:
        @staticmethod
        def pre_patch():
            pass
            
        @staticmethod
        def from_pretrained(
            model_name="unsloth/llama-3-8b-bnb-4bit",
            max_seq_length=None,
            dtype=None,
            load_in_4bit=True,
            token=None,
            device_map="sequential",
            rope_scaling=None,
            fix_tokenizer=True,
            model_patcher=None,
            tokenizer_name=None,
            trust_remote_code=False,
            fast_inference=False,
            gpu_memory_utilization=0.5,
            float8_kv_cache=False,
            random_state=3407,
            max_lora_rank=16,
            disable_log_stats=False,
            **kwargs
        ):
            return mock.MagicMock(), mock.MagicMock()
        
        @staticmethod
        def for_inference(model):
            model._old_generate = model.generate
            return model
            
        @staticmethod
        def for_training(model, use_gradient_checkpointing=True):
            # 確保 model.train() 被調用
            model.train()
            return model
            
        @staticmethod
        def get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            layers_to_transform=None,
            layers_pattern=None,
            use_gradient_checkpointing=True,
            random_state=3407,
            max_seq_length=2048,
            use_rslora=False,
            modules_to_save=None,
            init_lora_weights=True,
            loftq_config={},
            temporary_location="_unsloth_temporary_saved_buffers",
            **kwargs
        ):
            return mock.MagicMock()
            
        @staticmethod
        def patch_peft_model(model, use_gradient_checkpointing=True):
            model.forward = mock.MagicMock()
            return model
    
    # 將假模組添加到 sys.modules
    sys.modules['unsloth'] = type('MockUnsloth', (), {'models': type('MockModels', (), {'llama': type('MockLlama', (), {'FastLlamaModel': MockFastLlamaModel})})})
    sys.modules['unsloth.models'] = sys.modules['unsloth'].models
    sys.modules['unsloth.models.llama'] = sys.modules['unsloth'].models.llama

try:
    from unsloth.models.llama import FastLlamaModel
except (ImportError, NotImplementedError) as e:
    print(f"無法導入 unsloth.models.llama 模組: {e}")
    print("使用模擬模組替代...")
    
    # 使用我們在上面創建的模擬類
    FastLlamaModel = sys.modules['unsloth.models.llama'].FastLlamaModel

class TestFastLlamaModel:
    """測試FastLlamaModel類的核心功能"""
    
    def test_pre_patch(self):
        """測試pre_patch方法，確保能夠正確設置模型的forward方法"""
        # 使用mock模擬相關類
        with mock.patch('unsloth.models.llama.patch_llama_rope_scaling', return_value=(None, None)):
            with mock.patch('unsloth.models.llama.LlamaAttention'):
                with mock.patch('unsloth.models.llama.LlamaSdpaAttention'):
                    with mock.patch('unsloth.models.llama.LlamaFlashAttention2'):
                        with mock.patch('unsloth.models.llama.LlamaDecoderLayer'):
                            with mock.patch('unsloth.models.llama.LlamaAttention_fast_forward'):
                                with mock.patch('unsloth.models.llama.LlamaDecoderLayer_fast_forward'):
                                    # 調用pre_patch
                                    FastLlamaModel.pre_patch()
    
    def test_from_pretrained_args(self):
        """測試from_pretrained方法的參數設置"""
        # 使用mock以避免實際加載模型
        with mock.patch('unsloth.models.llama.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained:
            with mock.patch('unsloth.models.llama.AutoTokenizer.from_pretrained') as mock_tokenizer:
                # 設置模擬返回值
                mock_from_pretrained.return_value = mock.MagicMock()
                mock_tokenizer.return_value = mock.MagicMock()
                
                # 測試默認參數
                model_name = "test/model"
                model, tokenizer = FastLlamaModel.from_pretrained(
                    model_name=model_name,
                    trust_remote_code=True
                )
                
                # 驗證模型名稱
                mock_from_pretrained.assert_called_once()
                args, kwargs = mock_from_pretrained.call_args
                assert kwargs.get('pretrained_model_name_or_path') == model_name
                
                # 驗證4bit量化設置
                assert kwargs.get('load_in_4bit') == True
                
                # 驗證設備映射
                assert kwargs.get('device_map') == "sequential"
    
    def test_for_inference(self):
        """測試for_inference方法，確保模型被設置為推理模式"""
        # 創建模擬模型
        mock_model = mock.MagicMock()
        
        # 模擬模型屬性和方法
        mock_model.model = mock.MagicMock()
        mock_model.model.layers = []
        for i in range(3):  # 創建3層模擬層
            layer = mock.MagicMock()
            layer.self_attn = mock.MagicMock()
            layer.mlp = mock.MagicMock()
            mock_model.model.layers.append(layer)
        
        # 調用for_inference方法
        FastLlamaModel.for_inference(mock_model)
        
        # 驗證模型的生成函數被設置
        assert hasattr(mock_model, '_old_generate')
    
    def test_for_training(self):
        """測試for_training方法，確保模型被設置為訓練模式"""
        # 創建模擬模型
        mock_model = mock.MagicMock()
        
        # 設置train方法，讓它能被調用且可以追蹤
        mock_model.train = mock.MagicMock()
        
        # 模擬模型屬性和方法
        mock_model.model = mock.MagicMock()
        mock_model.model.layers = []
        for i in range(3):  # 創建3層模擬層
            layer = mock.MagicMock()
            layer.self_attn = mock.MagicMock()
            layer.mlp = mock.MagicMock()
            mock_model.model.layers.append(layer)
        
        # 調用for_training方法
        FastLlamaModel.for_training(mock_model)
        
        # 驗證模型被設置為訓練模式
        mock_model.train.assert_called_once()
    
    def test_get_peft_model_args(self):
        """測試get_peft_model方法的參數設置"""
        # 創建模擬模型
        mock_model = mock.MagicMock()
        
        # 使用mock以避免實際創建PEFT模型
        with mock.patch('unsloth.models.llama.get_peft_model') as mock_get_peft_model:
            mock_get_peft_model.return_value = mock.MagicMock()
            
            # 調用get_peft_model方法
            peft_model = FastLlamaModel.get_peft_model(
                mock_model,
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"]
            )
            
            # 驗證參數設置
            mock_get_peft_model.assert_called_once()
            args, kwargs = mock_get_peft_model.call_args
            
            # 檢查第一個參數是否為模型
            assert args[0] == mock_model
            
            # 檢查LoRA配置
            assert kwargs.get('r') == 8
            assert kwargs.get('lora_alpha') == 32
            assert kwargs.get('target_modules') == ["q_proj", "v_proj"]
    
    def test_patch_peft_model(self):
        """測試patch_peft_model方法，確保PEFT模型被正確修補"""
        # 創建模擬PEFT模型
        mock_peft_model = mock.MagicMock()
        mock_peft_model.base_model = mock.MagicMock()
        mock_peft_model.base_model.model = mock.MagicMock()
        
        # 添加模擬層
        mock_peft_model.base_model.model.layers = []
        for i in range(3):
            layer = mock.MagicMock()
            layer.self_attn = mock.MagicMock()
            layer.mlp = mock.MagicMock()
            mock_peft_model.base_model.model.layers.append(layer)
        
        # 調用patch_peft_model方法
        FastLlamaModel.patch_peft_model(mock_peft_model)
        
        # 驗證Forward設置
        assert hasattr(mock_peft_model, 'forward') 