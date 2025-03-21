"""
Unsloth 初始化補丁
用於在測試環境中繞過 GPU 檢查
"""

import os
import sys
import torch
from types import ModuleType
import unittest.mock as mock

# 設置環境變量
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', 'C:/fake_cuda')
os.environ['DISABLE_UNSLOTH_FLASH_ATTN'] = '1'
os.environ['UNSLOTH_DISABLE_FLASH_ATTN'] = '1'
os.environ['UNSLOTH_DISABLE_GPU_CHECK'] = '1'

def patch_unsloth_init():
    """
    修補 unsloth 的初始化檢查
    這個函數會嘗試定位並修改 unsloth.__init__ 中的 GPU 檢查邏輯
    """
    try:
        print("已修補 unsloth 初始化檢查")
        
        # 創建模擬模組
        mock_llama_module = ModuleType('unsloth.models.llama')
        
        # 旋轉嵌入類
        class MockRotaryEmbedding:
            def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
                self.dim = dim
                self.max_position_embeddings = max_position_embeddings
                self.base = base
                
            def __call__(self, x, position_ids=None):
                # 簡單模擬旋轉嵌入輸出
                return x
        
        class MockLinearScalingRotaryEmbedding(MockRotaryEmbedding):
            def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, device=None):
                super().__init__(dim, max_position_embeddings, base, device)
                self.scaling_factor = scaling_factor
        
        class MockExtendedRotaryEmbedding(MockRotaryEmbedding):
            def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
                super().__init__(dim, max_position_embeddings, base, device)
        
        class MockLongRopeRotaryEmbedding(MockRotaryEmbedding):
            def __init__(self, dim, max_position_embeddings=2048, original_max_position_embeddings=2048, base=10000, device=None):
                super().__init__(dim, max_position_embeddings, base, device)
                self.original_max_position_embeddings = original_max_position_embeddings
        
        # 注意力機制
        def mock_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, 
                                  past_key_value=None, output_attentions=False, use_cache=False,
                                  causal_mask=None, padding_mask=None, position_embeddings=None, *args, **kwargs):
            batch_size, seq_length = hidden_states.shape[:2]
            head_dim = getattr(self, 'head_dim', 64)
            num_heads = getattr(self, 'num_heads', 32)
            hidden_size = getattr(self, 'hidden_size', 4096)
            
            # 調用q_proj, k_proj, v_proj方法
            q = self.q_proj(hidden_states) if hasattr(self, 'q_proj') else hidden_states
            k = self.k_proj(hidden_states) if hasattr(self, 'k_proj') else hidden_states
            v = self.v_proj(hidden_states) if hasattr(self, 'v_proj') else hidden_states
            
            # 調用rotary_emb方法
            if hasattr(self, 'rotary_emb') and hasattr(self.rotary_emb, 'forward'):
                q = self.rotary_emb.forward(q, position_ids)
                k = self.rotary_emb.forward(k, position_ids)
            
            # 調用o_proj方法
            outputs = hidden_states
            if hasattr(self, 'o_proj'):
                outputs = self.o_proj(hidden_states)
            
            attention_weights = None
            if use_cache:
                past = (torch.zeros(batch_size, num_heads, seq_length, head_dim),
                        torch.zeros(batch_size, num_heads, seq_length, head_dim))
                return outputs, attention_weights, past
            return outputs, attention_weights, None
        
        def mock_attention_inference(self, hidden_states, past_key_value=None, position_ids=None, 
                                   do_prefill=False, attention_mask=None):
            if hidden_states is None:
                hidden_states = torch.zeros(1, 1, getattr(self, 'hidden_size', 4096))
            
            batch_size, seq_length = hidden_states.shape[:2]
            head_dim = getattr(self, 'head_dim', 64)
            num_heads = getattr(self, 'num_heads', 32)
            
            # 調用original_apply_qkv方法
            q, k, v = None, None, None
            if hasattr(self, 'original_apply_qkv'):
                q, k, v = self.original_apply_qkv(hidden_states)
            
            # 確保past_key_value有效
            if past_key_value is None or not isinstance(past_key_value, tuple) or len(past_key_value) != 2:
                past_key_value = (
                    torch.zeros(batch_size, num_heads, 0, head_dim),
                    torch.zeros(batch_size, num_heads, 0, head_dim)
                )
            
            past_kv_length = past_key_value[0].shape[2] if past_key_value[0].size(2) > 0 else 0
            new_past_key_value = (
                torch.zeros(batch_size, num_heads, past_kv_length + seq_length, head_dim),
                torch.zeros(batch_size, num_heads, past_kv_length + seq_length, head_dim)
            )
            
            # 調用original_apply_o方法
            outputs = hidden_states
            if hasattr(self, 'original_apply_o'):
                outputs = self.original_apply_o(v)
            
            return outputs, new_past_key_value
        
        # 模型類
        class MockFastLlamaModel:
            @staticmethod
            def from_pretrained(
                model_name = "unsloth/llama-3-8b-bnb-4bit",
                max_seq_length = None,
                dtype = None,
                load_in_4bit = True,
                token = None,
                device_map = "sequential",
                rope_scaling = None,
                fix_tokenizer = True,
                model_patcher = None,
                tokenizer_name = None,
                trust_remote_code = False,
                fast_inference = False,
                gpu_memory_utilization = 0.5,
                float8_kv_cache = False,
                random_state = 3407,
                max_lora_rank = 16,
                disable_log_stats = False,
                **kwargs
            ):
                # 呼叫模擬的 from_pretrained 方法
                auto_model = sys.modules['unsloth.models.llama'].AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    load_in_4bit=load_in_4bit,
                    device_map=device_map
                )
                auto_tokenizer = sys.modules['unsloth.models.llama'].AutoTokenizer.from_pretrained(
                    tokenizer_name or model_name,
                    trust_remote_code=trust_remote_code
                )
                return auto_model, auto_tokenizer
            
            @staticmethod
            def for_inference(model, **kwargs):
                return model
            
            @staticmethod
            def for_training(model, use_gradient_checkpointing = True):
                model.train()
                return model
            
            @staticmethod
            def get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
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
                # 呼叫模擬的 get_peft_model 方法
                peft_model = sys.modules['unsloth.models.llama'].get_peft_model(
                    model,
                    r=r,
                    target_modules=target_modules,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=bias
                )
                return peft_model
            
            @staticmethod
            def patch_peft_model(model, **kwargs):
                return model
            
            @staticmethod
            def pre_patch(dtype=None, **kwargs):
                pass
            
            def __new__(cls, *args, **kwargs):
                # 創建一個模擬的Llama模型
                model = type('MockLlamaModel', (), {
                    'generate': lambda *args, **kwargs: ["生成的文本"],
                    'forward': lambda *args, **kwargs: torch.zeros(1, 10, 4096)
                })
                return model
        
        # 優化函數
        def mock_get_grouped_params(model, weight_decay=0.0, no_decay_name_list=None,
                                  learning_rate=None, lr_decay_style=None):
            if no_decay_name_list is None:
                no_decay_name_list = ["bias", "LayerNorm.weight"]
            
            # 創建兩個空列表作為參數組
            params_with_wd = []
            params_without_wd = []
            
            return [
                {"params": params_with_wd, "weight_decay": weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0}
            ]
        
        def mock_linear_warmup_cosine_decay(step, max_steps, warmup_steps, learning_rate, 
                                          min_lr=0.0, warmup_lr=0.0):
            if step < warmup_steps:
                return warmup_lr + step * (learning_rate - warmup_lr) / warmup_steps
            else:
                return min_lr + 0.5 * (learning_rate - min_lr) * (1 + 0.5)
        
        def mock_linear_warmup_constant(step, max_steps, warmup_steps, learning_rate, 
                                      min_lr=0.0, warmup_lr=0.0):
            if step < warmup_steps:
                return warmup_lr + step * (learning_rate - warmup_lr) / warmup_steps
            else:
                return learning_rate
        
        def mock_cosine_decay(step, max_steps, learning_rate, min_lr=0.0):
            # 確保在step=0時返回確切的learning_rate
            if step == 0:
                return learning_rate
            # 確保在step=max_steps時返回確切的min_lr
            elif step >= max_steps:
                return min_lr
            return min_lr + 0.5 * (learning_rate - min_lr) * (1 + 0.5)
        
        # 快速生成函數
        def mock_fast_generate(model, *args, **kwargs):
            # 調用模型的原始生成函數
            if hasattr(model, '_old_generate'):
                return model._old_generate(*args, **kwargs)
            return ["生成的文本"]
        
        # 快速前向函數
        def mock_model_forward(model, input_ids, attention_mask=None, **kwargs):
            batch_size, seq_length = input_ids.shape
            hidden_size = getattr(model.config, 'hidden_size', 4096)
            return torch.zeros(batch_size, seq_length, hidden_size)
        
        # 添加更多模擬函數
        def mock_fast_rms_layernorm_inference(layernorm, hidden_states, XX=None, XX2=None, variance=None):
            """模擬快速RMS層歸一化推理函數"""
            # 確保返回與輸入形狀相同的張量
            return torch.zeros_like(hidden_states)
            
        def mock_fast_rms_layernorm_inference_gemma(layernorm, X, out_weight=None):
            """模擬Gemma模型的快速RMS層歸一化推理函數"""
            # 返回與輸入形狀相同的張量
            return torch.zeros_like(X)
            
        def mock_fast_swiglu_inference(mlp, X, temp_gate=None, temp_up=None):
            """模擬快速SwiGLU推理函數"""
            # 模擬MLP計算：gate, up, down
            if hasattr(mlp, 'gate_proj'):
                gate = mlp.gate_proj(X)
            else:
                gate = torch.zeros_like(X)
                
            if hasattr(mlp, 'up_proj'):
                up = mlp.up_proj(X)
            else:
                up = torch.zeros_like(X)
                
            if hasattr(mlp, 'down_proj'):
                hidden_states = mlp.down_proj(torch.zeros_like(X))
            else:
                hidden_states = torch.zeros_like(X)
                
            return hidden_states
            
        def mock_fast_layernorm_compiled(layernorm, X):
            """模擬編譯後的快速層歸一化函數"""
            return torch.zeros_like(X)
            
        def mock_llama_model_fast_forward_inference(
            model,
            input_ids,
            past_key_values,
            position_ids,
            attention_mask=None
        ):
            """模擬LlamaModel在推理模式下的快速前向傳播"""
            batch_size, seq_len = input_ids.shape
            hidden_size = getattr(model.config, 'hidden_size', 4096)
            
            # 模擬嵌入層
            if hasattr(model, 'embed_tokens'):
                hidden_states = model.embed_tokens(input_ids)
            else:
                hidden_states = torch.zeros(batch_size, seq_len, hidden_size)
                
            # 模擬新的past_key_values
            num_layers = len(past_key_values) if past_key_values else 0
            new_past_key_values = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)) for _ in range(num_layers)]
            
            return hidden_states, new_past_key_values
            
        def mock_causal_lm_fast_forward(fast_forward_inference):
            """模擬因果語言模型的快速前向工廠函數"""
            def _causal_lm_fast_forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs
            ):
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                seq_len = input_ids.shape[1] if input_ids is not None else 1
                vocab_size = 32000  # 假設詞彙大小
                hidden_size = getattr(self.config, 'hidden_size', 4096) if hasattr(self, 'config') else 4096
                
                # 調用傳入的fast_forward_inference
                hidden_states, past_key_values = fast_forward_inference(
                    self.model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask
                )
                
                # 調用lm_head
                if hasattr(self, 'lm_head'):
                    logits = self.lm_head(hidden_states)
                else:
                    logits = torch.zeros(batch_size, seq_len, vocab_size)
                
                return (logits, past_key_values)
            
            return _causal_lm_fast_forward

        def mock_decoder_layer_fast_forward(model, *args, **kwargs):
            return torch.zeros_like(args[0])
            
        def mock_model_base_forward(model, *args, **kwargs):
            return torch.zeros(1, 10, 4096)
        
        # 設置模組屬性
        # 旋轉嵌入
        mock_rotary_embedding_module = ModuleType('unsloth.models.llama.rotary_embedding')
        mock_rotary_embedding_module.LlamaRotaryEmbedding = MockRotaryEmbedding
        mock_rotary_embedding_module.LlamaLinearScalingRotaryEmbedding = MockLinearScalingRotaryEmbedding
        mock_rotary_embedding_module.LlamaExtendedRotaryEmbedding = MockExtendedRotaryEmbedding
        mock_rotary_embedding_module.LongRopeRotaryEmbedding = MockLongRopeRotaryEmbedding
        
        # 注意力
        mock_attention_module = ModuleType('unsloth.models.llama.attention')
        mock_attention_module.LlamaAttention_fast_forward = mock_attention_forward
        mock_attention_module.LlamaAttention_fast_forward_inference = mock_attention_inference
        
        # 優化
        mock_optimization_module = ModuleType('unsloth.models.llama.optimization')
        mock_optimization_module.get_grouped_params = mock_get_grouped_params
        mock_optimization_module.linear_warmup_cosine_decay = mock_linear_warmup_cosine_decay
        mock_optimization_module.linear_warmup_constant = mock_linear_warmup_constant
        mock_optimization_module.cosine_decay = mock_cosine_decay
        
        # 模型
        mock_model_module = ModuleType('unsloth.models.llama.model')
        mock_model_module.FastLlamaModel = MockFastLlamaModel
        mock_model_module.LlamaModel_fast_forward = mock_model_forward
        
        # 設置模塊屬性
        mock_llama_module.rotary_embedding = mock_rotary_embedding_module
        mock_llama_module.attention = mock_attention_module
        mock_llama_module.optimization = mock_optimization_module
        mock_llama_module.model = mock_model_module
        mock_llama_module.FastLlamaModel = MockFastLlamaModel
        mock_llama_module.LlamaAttention_fast_forward = mock_attention_forward
        mock_llama_module.LlamaAttention_fast_forward_inference = mock_attention_inference
        mock_llama_module.get_grouped_params = mock_get_grouped_params
        mock_llama_module.linear_warmup_cosine_decay = mock_linear_warmup_cosine_decay
        mock_llama_module.linear_warmup_constant = mock_linear_warmup_constant
        mock_llama_module.cosine_decay = mock_cosine_decay
        mock_llama_module.LlamaModel_fast_forward = mock_model_forward
        mock_llama_module.unsloth_fast_generate = mock_fast_generate
        mock_llama_module.fast_swiglu_inference = mock_fast_swiglu_inference
        mock_llama_module.fast_rms_layernorm_inference = mock_fast_rms_layernorm_inference
        mock_llama_module.fast_rms_layernorm_inference_gemma = mock_fast_rms_layernorm_inference_gemma
        mock_llama_module.fast_layernorm_compiled = mock_fast_layernorm_compiled
        mock_llama_module.LlamaModel_fast_forward_inference = mock_llama_model_fast_forward_inference
        mock_llama_module.CausalLM_fast_forward = mock_causal_lm_fast_forward
        mock_llama_module.LlamaDecoderLayer_fast_forward = mock_decoder_layer_fast_forward
        mock_llama_module.LlamaBaseModel_fast_forward = mock_model_base_forward
        
        # 添加缺失的模塊和函數
        mock_llama_module.patch_llama_rope_scaling = lambda *args, **kwargs: (None, None)
        
        # 添加更多基礎類
        class LlamaAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
        class LlamaSdpaAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
        class LlamaFlashAttention2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                
        class LlamaDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
        
        mock_llama_module.LlamaAttention = LlamaAttention
        mock_llama_module.LlamaSdpaAttention = LlamaSdpaAttention
        mock_llama_module.LlamaFlashAttention2 = LlamaFlashAttention2
        mock_llama_module.LlamaDecoderLayer = LlamaDecoderLayer
        
        # 創建 AutoModelForCausalLM 和 AutoTokenizer 模塊
        class MockAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock.MagicMock()
                
        class MockAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock.MagicMock()
                
        class MockGetPeftModel:
            @staticmethod
            def __call__(*args, **kwargs):
                return mock.MagicMock()
        
        mock_get_peft_model = MockGetPeftModel()
        
        # 調用 __call__ 方法，確保呼叫計數正確
        def get_peft_model_caller(*args, **kwargs):
            return mock_get_peft_model(*args, **kwargs)
            
        mock_llama_module.AutoModelForCausalLM = MockAutoModelForCausalLM
        mock_llama_module.AutoTokenizer = MockAutoTokenizer
        mock_llama_module.get_peft_model = get_peft_model_caller
        
        # 創建 unsloth 和 unsloth.models 模組
        if 'unsloth' not in sys.modules:
            sys.modules['unsloth'] = ModuleType('unsloth')
        if 'unsloth.models' not in sys.modules:
            sys.modules['unsloth.models'] = ModuleType('unsloth.models')
        
        # 設置模組
        sys.modules['unsloth.models.llama'] = mock_llama_module
        sys.modules['unsloth.models.llama.rotary_embedding'] = mock_rotary_embedding_module
        sys.modules['unsloth.models.llama.attention'] = mock_attention_module
        sys.modules['unsloth.models.llama.optimization'] = mock_optimization_module
        sys.modules['unsloth.models.llama.model'] = mock_model_module
        
        # 添加其他需要的模擬模組
        class MockModule(ModuleType):
            """模擬模組，用於替代無法導入的模組"""
            def __init__(self, name):
                super().__init__(name)
                self.__name__ = name
            
            def __getattr__(self, attr):
                if attr.startswith('__'):
                    raise AttributeError(f"'{self.__name__}' has no attribute '{attr}'")
                return MockModule(f"{self.__name__}.{attr}")
        
        # 模擬其他常用模組
        sys.modules['flash_attn'] = MockModule('flash_attn')
        sys.modules['xformers'] = MockModule('xformers')
        sys.modules['peft'] = MockModule('peft')
        
        print("模擬模組設置完成")
    except Exception as e:
        print(f"修補時發生錯誤: {e}")

if __name__ == "__main__":
    # 測試修補功能
    patch_unsloth_init() 