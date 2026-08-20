# coding: utf8
import math
import torch


def _safe_trace_solve(gram_x: torch.Tensor, gram_s: torch.Tensor, epsilon: float) -> float:
    """Compute tr((gram_x + eps I)^-1 gram_s) in float64 with jitter fallback."""
    gram_x = gram_x.to(dtype=torch.float64)
    gram_s = gram_s.to(dtype=torch.float64)

    eye = torch.eye(gram_x.shape[0], dtype=torch.float64)
    jitter = float(epsilon)

    for _ in range(6):
        try:
            solved = torch.linalg.solve(gram_x + jitter * eye, gram_s)
            phi_val = torch.trace(solved).item()
            if phi_val < 0 and abs(phi_val) < 1e-10:
                phi_val = 0.0
            return float(phi_val)
        except RuntimeError:
            jitter *= 10.0

    # Last resort: pseudo-inverse (numerically safer than hard failure).
    inv_like = torch.linalg.pinv(gram_x + jitter * eye)
    phi_val = torch.trace(inv_like @ gram_s).item()
    if phi_val < 0 and abs(phi_val) < 1e-10:
        phi_val = 0.0
    return float(phi_val)


def compute_phi_metrics(profile_x: dict, profile_s: dict, epsilon: float) -> dict:
    """
    Compute per-layer per-linear metrics:
      Phi = tr((XX^T + eps I)^-1 SS^T)
      A   = Phi
      C   = 1/sqrt(Phi)
    """
    gram_x = profile_x["gram_mat"]
    gram_s = profile_s["gram_mat"]

    metrics = {}
    common_layers = sorted(set(gram_x.keys()) & set(gram_s.keys()))

    for layer_idx in common_layers:
        layer_key = f"layer_{layer_idx}"
        metrics[layer_key] = {}

        names_x = set(gram_x[layer_idx].keys())
        names_s = set(gram_s[layer_idx].keys())
        common_names = sorted(names_x & names_s)

        for name in common_names:
            gx = gram_x[layer_idx][name]
            gs = gram_s[layer_idx][name]
            phi = _safe_trace_solve(gx, gs, epsilon=epsilon)
            adaptability = phi
            cost = float("inf") if phi <= 0 else (1.0 / math.sqrt(phi))

            metrics[layer_key][name] = {
                "phi": adaptability,
                "adaptability": adaptability,
                "cost": cost,
                "in_features": int(gx.shape[0]),
            }

    return metrics
