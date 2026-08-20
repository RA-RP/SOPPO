import yaml
from pathlib import Path

from Eval.component.target_metrics import get_target_metrics_csv

def eval2res(density_config):
    """
    Convert split lm-eval JSON files into the unified target metrics CSV.

    Legacy per-task clean/getResult CSV generation has been removed. The single
    supported output is eval.target_metrics_csv, driven by eval.target_metrics.
    """
    eval_cfg = density_config["eval"]
    output_origin_root = Path(eval_cfg["output_origin_root"])
    output_root = Path(eval_cfg["output_root"])
    csv_path = Path(eval_cfg["csv_path"])

    target_metrics = eval_cfg.get("target_metrics") or []
    if not target_metrics:
        print("⚠️ eval.target_metrics 为空，未生成 target metrics CSV。")
        return

    try:
        get_target_metrics_csv(
            origin_root=output_origin_root,
            fix_root=output_root,
            output_root=csv_path,
            metric_specs=target_metrics,
            output_name=eval_cfg.get("target_metrics_csv", "target_metrics_results.csv"),
        )
    except Exception as e:
        print(f"❌ Error during target_metrics conversion: {e}")


if __name__ == "__main__":
    # 从 density.yaml 启动
    config_path = "../../Density.yaml"
    with open(config_path, "r") as f:
        density = yaml.safe_load(f)

    eval2res(density)
