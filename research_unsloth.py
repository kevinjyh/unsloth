"""
Unsloth代碼研究腳本
用於分析和研究unsloth代碼結構，支援真實CUDA環境
"""

import os
import sys
import importlib
import inspect
import torch

def analyze_module(module_name, depth=0, max_depth=2):
    """分析模組結構"""
    indent = "  " * depth
    print(f"{indent}分析模組: {module_name}")

    if depth >= max_depth:
        print(f"{indent}達到最大深度限制，停止探索")
        return

    try:
        # 導入模組
        module = importlib.import_module(module_name)
        
        # 分析模組屬性
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue  # 跳過私有屬性
                
            try:
                attr = getattr(module, attr_name)
                
                # 檢查是否是函數或類
                if inspect.isfunction(attr):
                    sig = inspect.signature(attr)
                    print(f"{indent}- 函數: {attr_name}{sig}")
                elif inspect.isclass(attr):
                    print(f"{indent}- 類: {attr_name}")
                    
                    # 遞歸分析子模組
                    if hasattr(attr, '__module__') and attr.__module__.startswith(module_name):
                        sub_module_name = attr.__module__ + '.' + attr.__name__
                        analyze_module(sub_module_name, depth + 1, max_depth)
            except Exception as e:
                print(f"{indent}  無法分析 {attr_name}: {e}")
    
    except ImportError as e:
        print(f"{indent}無法導入模組 {module_name}: {e}")
    except Exception as e:
        print(f"{indent}分析模組 {module_name} 時出錯: {e}")

def get_models_structure():
    """獲取models目錄下的結構"""
    print("分析 unsloth.models 目錄結構...")
    
    try:
        # 嘗試導入unsloth.models
        from unsloth import models
        
        # 列出所有models下的模組
        print("\nUnsloth模型目錄包含以下文件:")
        if hasattr(models, '__path__'):
            for path in models.__path__:
                if os.path.exists(path):
                    files = os.listdir(path)
                    for file in files:
                        if file.endswith('.py'):
                            print(f"- {file}")
                            
                            # 嘗試分析這個模組
                            module_name = f"unsloth.models.{file[:-3]}"
                            try:
                                analyze_module(module_name, depth=1, max_depth=2)
                            except Exception as e:
                                print(f"  無法分析模組 {module_name}: {e}")
    except ImportError:
        print("無法導入unsloth.models，請確保unsloth已正確安裝")
    except Exception as e:
        print(f"分析models時出錯: {e}")

def print_system_info():
    """打印系統信息"""
    print("\n系統信息:")
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA是否可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"CUDA設備數量: {torch.cuda.device_count()}")
        print(f"當前CUDA設備: {torch.cuda.current_device()}")
        print(f"CUDA設備名稱: {torch.cuda.get_device_name()}")
        print(f"CUDA設備能力: {torch.cuda.get_device_capability()}")
        
        # 嘗試獲取cuDNN版本
        try:
            if torch.backends.cudnn.is_available():
                print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            else:
                print(f"cuDNN: 不可用")
        except:
            print(f"cuDNN: 檢測時發生錯誤")
    else:
        print("CUDA不可用，將使用CPU進行研究")

def main():
    """主程序"""
    print("開始研究Unsloth代碼結構...")
    
    # 打印系統信息
    print_system_info()
    
    try:
        # 導入unsloth
        import unsloth
        print(f"\nUnsloth版本: {unsloth.__version__ if hasattr(unsloth, '__version__') else '未知'}")
        
        # 分析unsloth目錄結構
        print("\n模組路徑:")
        if hasattr(unsloth, '__path__'):
            for path in unsloth.__path__:
                print(f"- {path}")
        
        # 分析models結構
        get_models_structure()
        
    except ImportError:
        print("無法導入unsloth，請確保它已正確安裝")
    except Exception as e:
        print(f"研究過程中發生錯誤: {e}")
    
    print("\n研究完成！")

if __name__ == "__main__":
    main() 