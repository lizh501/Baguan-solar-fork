import torch
from collections import OrderedDict

def load_state_dict_flexible(model, checkpoint_state_dict):
    """
    灵活加载 state_dict，处理形状不匹配和缺失/多余的键。

    Args:
        model (torch.nn.Module): 你要加载权重的模型。
        checkpoint_state_dict (dict): 从 checkpoint 文件中加载的 state_dict。
    """
    model_state_dict = model.state_dict()
    loaded_keys = []
    mismatched_keys = []
    missing_keys_in_ckpt = []
    
    # 1. 创建一个新的 state_dict，只包含那些名称和形状都匹配的权重
    new_state_dict = OrderedDict()

    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            # 检查形状是否匹配
            if model_state_dict[key].shape == value.shape:
                new_state_dict[key] = value
                loaded_keys.append(key)
            else:
                mismatched_keys.append({
                    "key": key,
                    "model_shape": model_state_dict[key].shape,
                    "ckpt_shape": value.shape
                })
        else:
            # Checkpoint 中有，但当前模型中没有的键（通常可以忽略）
            pass

    # 2. 检查当前模型中有哪些键在 Checkpoint 中缺失
    for key in model_state_dict.keys():
        if key not in checkpoint_state_dict:
            missing_keys_in_ckpt.append(key)

    # 3. 使用严格模式为 False 来加载筛选后的 state_dict
    #    这能处理一些 new_state_dict 中没有，但 model_state_dict 中有的情况
    model.load_state_dict(new_state_dict, strict=False)

    # 4. 打印详细的报告
    print("✅ State_dict loading report:")
    print(f"  - Successfully loaded {len(loaded_keys)} keys.")
    
    if mismatched_keys:
        print("\n⚠️ Mismatched shape keys (skipped):")
        for info in mismatched_keys:
            print(f"  - Key: {info['key']}")
            print(f"    - Model shape: {info['model_shape']}")
            print(f"    - Checkpoint shape: {info['ckpt_shape']}")
            
    if missing_keys_in_ckpt:
        print("\n⚠️ Keys in model but not in checkpoint (initialized from scratch):")
        for key in missing_keys_in_ckpt:
            print(f"  - {key}")