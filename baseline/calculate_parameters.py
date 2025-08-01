import torch
from baseline.networks.omni_vision_transformer_PG_02 import OmniVisionTransformer
from config_PG import get_config  

def count_parameters(model):
    """
    统计模型的总参数数量以及每个模块的参数数量
    Args:
        model: PyTorch 模型
    Returns:
        None
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nParameters by module:")
    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters())
        if module_params > 0:
            print(f"{name}: {module_params:,}")

if __name__ == "__main__":
    # 加载配置
    config = get_config()
    
    # 初始化模型
    model = OmniVisionTransformer(config, prompt=True)
    
    # 打印参数统计
    count_parameters(model)