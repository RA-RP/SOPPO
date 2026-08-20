import os
import json
import numpy as np
import torch
from tqdm import tqdm  # 进度条库

# ==== 参数设置 ====
modelA_dir = "../autodl-tmp/modelMat/originalQwen2.5/0k" # 原始模型
modelB_dir = "../autodl-tmp/modelMat/qwen-235B/2k"           # 微调模型
output_dir = "../autodl-tmp/analyseModel/vsOrigin/qwen-235B/2K"       # 保存结果的文件夹
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cpu")  # 全部使用 CPU
print("Using device:", device)

layers = sorted([f for f in os.listdir(modelA_dir) if f.endswith(".npy")])
k_default = 50  # 前 k 个奇异值

# ================= 辅助函数 =================
def kl_divergence(p, q, eps=1e-10):
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    return float(torch.sum(p * torch.log((p + eps) / (q + eps))))

def entropy(p, eps=1e-10):
    p = p / torch.sum(p)
    return float(-torch.sum(p * torch.log(p + eps)))

def subspace_distance(U, Up):
    """子空间差异，避免生成大矩阵"""
    UtUp = U.T @ Up
    k = U.shape[1]
    return float((k - torch.norm(UtUp, 'fro')**2)**0.5)

def compute_metrics(W, Wp, k=None, full=False):
    """计算单层指标"""
    W = W.cpu()
    Wp = Wp.cpu()

    if full:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Up, Sp, Vhp = torch.linalg.svd(Wp, full_matrices=False)
    else:
        # 先计算全量 SVD，再取前 k 个
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Up, Sp, Vhp = torch.linalg.svd(Wp, full_matrices=False)
        if k is not None:
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
            Up, Sp, Vhp = Up[:, :k], Sp[:k], Vhp[:k, :]

    result = {
        "mean_Sigma": float(torch.mean(S)),
        "var_Sigma": float(torch.var(S)),
        "entropy_Sigma": entropy(S),
        "kl_sigma": kl_divergence(S, Sp),
        "dU": subspace_distance(U, Up),
        "dV": subspace_distance(Vh.T, Vhp.T)
    }
    return result

# ================= 分层串行处理 =================
for layer_name in tqdm(layers, desc="Processing layers"):
    # 跳过 embedding 层
    if layer_name == "model_embed_tokens_weight.npy":
        print(f"Skipping embedding layer: {layer_name}")
        continue

    # 读取矩阵
    W  = torch.from_numpy(np.load(os.path.join(modelA_dir, layer_name)))
    Wp = torch.from_numpy(np.load(os.path.join(modelB_dir, layer_name)))

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
        print(f"Error processing {layer_name}: {e}")
        continue
