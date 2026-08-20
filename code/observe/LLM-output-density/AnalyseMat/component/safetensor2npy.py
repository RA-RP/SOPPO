from pathlib import Path
import numpy as np
from safetensors.torch import load_file
import torch
import re

# ----------------------------
# 配置
# ----------------------------
#input_root = Path("../autodl-tmp/mergeOutput/Qwen2.5")  # 输入根目录
input_root = Path("./Qwen2.5")  # 输入根目录
output_root = Path("../autodl-tmp/modelMat")           # 你指定的高 output_root
overwrite = False    # 已存在时是否覆盖
prefix_with_stem = False  # 是否在输出文件名前加 safetensors 文件名（避免不同文件中同名层冲突）
keep_layer_keywords = None  # 若只保存特定层，设置为列表，例如 ["q_proj", "k_proj"]; None 表示保存全部

# ----------------------------
# 辅助函数
# ----------------------------
def safe_filename(name: str) -> str:
    # 把任意非法字符替换成下划线
    return re.sub(r'[^A-Za-z0-9_]', '_', name)

# ----------------------------
# 运行
# ----------------------------
input_root = input_root.resolve()
output_root = output_root.resolve()

if not input_root.exists():
    raise FileNotFoundError(f"输入目录不存在: {input_root}")

print(f"输入根目录: {input_root}")
print(f"输出根目录: {output_root}")

# 遍历所有 .safetensors 文件（递归）
st_paths = sorted(input_root.rglob("*.safetensors"))

if not st_paths:
    print("未找到任何 .safetensors 文件。请检查 input_root。")
else:
    for st_path in st_paths:
        try:
            # 以 input_root 为基准计算相对目录（例如 Qwen-math-plus/0.4k）
            rel_dir = st_path.parent.relative_to(input_root)
        except Exception as e:
            # 若无法相对化（理论上不会发生），则放到 output_root/unknown 下
            rel_dir = Path("unknown")

        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        st_stem = st_path.stem  # safetensors 文件名（无后缀）
        print(f"\nProcessing: {st_path}")
        print(f" -> 输出目录: {out_dir}")

        try:
            tensors = load_file(str(st_path))  # {name: torch.Tensor}
        except Exception as e:
            print(f"  ⚠ 无法加载 {st_path}: {e}")
            continue

        n_saved = 0
        for name, tensor in tensors.items():
            # 若指定只保存特定层关键字，则过滤
            if keep_layer_keywords is not None:
                lk = name.lower()
                if not any(kw.lower() in lk for kw in keep_layer_keywords):
                    continue

            try:
                arr = tensor.cpu().to(torch.float32).numpy()
            except Exception as e:
                print(f"  ⚠ 无法将张量转为 numpy ({name}): {e}")
                continue

            safe_name = safe_filename(name)
            if prefix_with_stem:
                out_fname = f"{safe_filename(st_stem)}_{safe_name}.npy"
            else:
                out_fname = f"{safe_name}.npy"

            out_path = out_dir / out_fname

            if out_path.exists() and not overwrite:
                # 跳过已有文件
                # 若需要覆盖，设置 overwrite=True
                # 若想改为压缩保存，可在此改用 np.savez_compressed
                # print(f"  跳过已存在: {out_path.name}")
                continue

            try:
                np.save(out_path, arr)
                n_saved += 1
            except Exception as e:
                print(f"  ⚠ 保存失败 {out_path}: {e}")

        print(f"  已保存 {n_saved} 个层到 {out_dir}")
