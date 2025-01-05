import torch
import os


def load_state_dict(checkpoint_path):
    """
    加载检查点并返回模型的状态字典。
    支持检查点中包含完整模型或仅包含状态字典。
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"文件未找到: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 根据检查点内容提取状态字典
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'model' in checkpoint:
            return checkpoint['model']
        else:
            # 假设整个字典就是状态字典
            return checkpoint
    else:
        # 如果检查点不是字典，假设它就是状态字典
        return checkpoint


def compare_state_dicts(state_dict1, state_dict2, atol=1e-8, rtol=1e-5):
    """
    比较两个状态字典，逐一对比每个参数的形状和数值。

    参数：
        state_dict1 (dict): 第一个状态字典。
        state_dict2 (dict): 第二个状态字典。
        atol (float): 绝对容差，用于数值比较。
        rtol (float): 相对容差，用于数值比较。

    返回：
        differences (dict): 包含差异信息的字典。
    """
    differences = {
        'missing_in_1': [],
        'missing_in_2': [],
        'shape_mismatch': [],
        'value_mismatch': []
    }

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # 查找缺失的参数
    missing_in_1 = keys2 - keys1
    missing_in_2 = keys1 - keys2

    if missing_in_1:
        differences['missing_in_1'] = list(missing_in_1)
    if missing_in_2:
        differences['missing_in_2'] = list(missing_in_2)

    # 仅比较共有的参数
    common_keys = keys1 & keys2

    for key in common_keys:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            differences['shape_mismatch'].append({
                'parameter': key,
                'shape1': tensor1.shape,
                'shape2': tensor2.shape
            })
            continue  # 形状不同，无需进一步比较数值

        # 检查数值是否相近
        if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            # 计算数值差异
            difference = torch.abs(tensor1 - tensor2)
            max_diff = difference.max().item()
            mean_diff = difference.mean().item()
            num_diffs = (difference > atol + rtol * torch.abs(tensor1)).sum().item()

            differences['value_mismatch'].append({
                'parameter': key,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'num_diffs_exceed': num_diffs
            })

    return differences


def print_differences(differences):
    """
    打印差异信息。
    """
    if differences['missing_in_1']:
        print("参数在第一个检查点中缺失:")
        for param in differences['missing_in_1']:
            print(f"  - {param}")

    if differences['missing_in_2']:
        print("参数在第二个检查点中缺失:")
        for param in differences['missing_in_2']:
            print(f"  - {param}")

    if differences['shape_mismatch']:
        print("\n形状不匹配的参数:")
        for mismatch in differences['shape_mismatch']:
            print(f"  - {mismatch['parameter']}: {mismatch['shape1']} vs {mismatch['shape2']}")

    if differences['value_mismatch']:
        print("\n数值不匹配的参数:")
        for mismatch in differences['value_mismatch']:
            print(
                f"  - {mismatch['parameter']}: 最大差异={mismatch['max_diff']:.6f}, 平均差异={mismatch['mean_diff']:.6f}, 超出容差的元素数量={mismatch['num_diffs_exceed']}")

    if not any(differences.values()):
        print("两个检查点的参数完全一致。")


def main(file1, file2, atol=1e-8, rtol=1e-5):
    """
    主函数，加载两个权重文件并比较它们。

    参数：
        file1 (str): 第一个权重文件路径。
        file2 (str): 第二个权重文件路径。
        atol (float): 绝对容差。
        rtol (float): 相对容差。
    """
    print(f"加载第一个检查点: {file1}")
    state_dict1 = load_state_dict(file1)
    print(f"第一个检查点包含 {len(state_dict1)} 个参数。\n")

    print(f"加载第二个检查点: {file2}")
    state_dict2 = load_state_dict(file2)
    print(f"第二个检查点包含 {len(state_dict2)} 个参数。\n")

    print("开始比较两个状态字典...\n")
    differences = compare_state_dicts(state_dict1, state_dict2, atol=atol, rtol=rtol)
    print_differences(differences)


# 使用示例
if __name__ == "__main__":
    # 替换为您的权重文件路径
    file1 = '/home/hanzz/projects/image_caption/clipText/output/train/checkpoint_0.pth'
    file2 = '/home/hanzz/projects/image_caption/clipText/output/train/checkpoint_1.pth'

    main(file1, file2)
