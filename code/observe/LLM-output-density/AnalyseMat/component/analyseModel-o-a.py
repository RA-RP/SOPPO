import os
import json
import numpy as np
import torch
from tqdm import tqdm  # 进度条

# ==== 参数设置 ====
modelA_dir = "../autodl-tmp/modelMat/Qwen2.5-1.5B"  # 原始模型
modelB_root = "../autodl-tmp/modelMat"                     # 微调模型根目录
output_root  = "../autodl-tmp/analyseModel/vsOrigin/additionMat"
os.makedirs(output_root, exist_ok=True)

device = torch.device("cpu")
print("Using device:", device)

k_default = 50  # 前 k 个奇异值

# ================= 辅助函数 =================
def entropy(p, eps=1e-10):
    p = p / torch.sum(p)
    return float(-torch.sum(p * torch.log(p + eps)))

def compute_metrics(W, Wp, k=None, full=False):
    """计算单层指标"""
    W = W.cpu()
    Wp = Wp.cpu()
    W = W - Wp
    
    if full:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    else:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        if k is not None:
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]

    result = {
        "mean_Sigma": float(torch.mean(S)),
        "var_Sigma": float(torch.var(S)),
        "entropy_Sigma": entropy(S),
    }
    return result

# ================= 批量处理 =================
# 遍历 modelB_root 下的每个模型目录
for model_name in sorted(os.listdir(modelB_root)):
    model_path = os.path.join(modelB_root, model_name)
    if not os.path.isdir(model_path):
        continue
    
    # 遍历数据量目录
    for data_size in sorted(os.listdir(model_path)):
        data_path = os.path.join(model_path, data_size)
        if not os.path.isdir(data_path):
            continue

        print(f"\nProcessing model={model_name}, data_size={data_size}")

        # 输出目录
        output_dir = os.path.join(output_root, model_name, data_size)
        os.makedirs(output_dir, exist_ok=True)

        # 列出所有层
        layers = sorted([f for f in os.listdir(data_path) if f.endswith(".npy")])
        if not layers:
            print(f"  ⚠ No .npy layers found in {data_path}")
            continue

        for layer_name in tqdm(layers, desc=f"{model_name}/{data_size} layers"):
            # 跳过 embedding 层
            if layer_name == "model_embed_tokens_weight.npy":
                continue

            # 加载原始模型层
            W_path  = os.path.join(modelA_dir, layer_name)
            if not os.path.exists(W_path):
                print(f"  ⚠ 原始层不存在: {W_path}")
                continue
            W  = torch.from_numpy(np.load(W_path))

            # 加载微调模型层
            Wp_path = os.path.join(data_path, layer_name)
            Wp = torch.from_numpy(np.load(Wp_path))

            try:
                # ---- 前 k 个奇异值 ----
                result_k = compute_metrics(W, Wp, k=k_default, full=False)
                out_path_k = os.path.join(output_dir, f"{layer_name}_k.json")
                with open(out_path_k, 'w') as f:
                    json.dump(result_k, f, indent=4)

                # ---- 全量奇异值 ----
                result_full = compute_metrics(W, Wp, full=True)
                out_path_full = os.path.join(output_dir, f"{layer_name}_full.json")
                with open(out_path_full, 'w') as f:
                    json.dump(result_full, f, indent=4)

            except RuntimeError as e:
                print(f"  ⚠ Error processing {layer_name}: {e}")
                continue
