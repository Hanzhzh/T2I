from torch.optim import AdamW

def create_optimizer(args, model):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)
    print("### lr_mult:", lr_mult)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr},                    # 组0: 有权重衰减，学习率=lr
        {"params": [], "weight_decay": 0.0, "lr": lr},                  # 组1: 无权重衰减，学习率=lr
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult},        # 组2: 有权重衰减，学习率=lr * lr_mult
        {"params": [], "weight_decay": 0.0, "lr": lr * lr_mult}         # 组3: 无权重衰减，学习率=lr * lr_mult
    ]

    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight",
                "norm.bias", "norm.weight",
                "norm1.bias", "norm1.weight",
                "norm2.bias", "norm2.weight"}

    # 统一 large_lr 为集合类型
    if hasattr(model, 'init_params'):
        large_lr = set(model.init_params)
        print("### model has 'init_params', count:", len(large_lr))
    else:
        large_lr = set()

    # 参数分组并记录分组信息
    param_group_info = {0: [], 1: [], 2: [], 3: []}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # 冻结的权重不参与优化

        if any(nd in n for nd in no_decay):
            if n in large_lr:
                optimizer_grouped_parameters[3]['params'].append(p)
                param_group_info[3].append(n)
                print(f"### 添加到组3 (无权重衰减，高学习率): {n}")
            else:
                optimizer_grouped_parameters[1]['params'].append(p)
                param_group_info[1].append(n)
                print(f"### 添加到组1 (无权重衰减): {n}")
        else:  # 有权重衰减的参数
            if n in large_lr:
                optimizer_grouped_parameters[2]['params'].append(p)
                param_group_info[2].append(n)
                print(f"### 添加到组2 (有权重衰减，高学习率): {n}")
            else:
                optimizer_grouped_parameters[0]['params'].append(p)
                param_group_info[0].append(n)
                print(f"### 添加到组0 (有权重衰减): {n}")

    # 打印每个组包含的参数数量
    for group_id, params in param_group_info.items():
        print(f"### 参数组{group_id} 包含 {len(params)} 个参数")

    # 检查是否有参数未被分组
    all_trainable_params = set(name for name, p in model.named_parameters() if p.requires_grad)
    grouped_params = set()
    for group in optimizer_grouped_parameters:
        for p in group['params']:
            for name, param in model.named_parameters():
                if p is param:
                    grouped_params.add(name)
                    break

    missing_params = all_trainable_params - grouped_params
    if missing_params:
        print(f"### 警告: 优化器中缺少以下参数: {missing_params}")

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))

    return optimizer
