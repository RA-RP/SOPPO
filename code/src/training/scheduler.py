"""
Lambda scheduler for weighting PE or pseudo-target loss.

Three types: Fixed, Linear-Warmup, Exponential-Warmup
"""

from typing import Optional


class LambdaScheduler:
    """Base class for lambda schedulers."""

    def __init__(self, lambda_star: float, total_steps: int):
        """
        Args:
            lambda_star: Target lambda value
            total_steps: Total training steps
        """
        self.lambda_star = lambda_star
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> float:
        """Get current lambda and advance step."""
        lambda_t = self.get_lambda(self.current_step)
        self.current_step += 1
        return lambda_t

    def get_lambda(self, step: int) -> float:
        """Get lambda at specific step (to be overridden)."""
        raise NotImplementedError

    def reset(self):
        """Reset scheduler."""
        self.current_step = 0


class FixedLambdaScheduler(LambdaScheduler):
    """
    Fixed lambda scheduler: λ(t) = λ*
    """

    def get_lambda(self, step: int) -> float:
        return self.lambda_star


class LinearWarmupLambdaScheduler(LambdaScheduler):
    """
    Linear warmup scheduler:
        λ(t) = λ* × min(1, t / t_warmup)

    where t_warmup = warmup_ratio × total_steps
    """

    def __init__(self, lambda_star: float, total_steps: int,
                 warmup_ratio: float = 0.1):
        """
        Args:
            lambda_star: Target lambda value
            total_steps: Total training steps
            warmup_ratio: Warmup period as fraction of total steps (default 0.1)
        """
        super().__init__(lambda_star, total_steps)
        self.warmup_steps = int(warmup_ratio * total_steps)

    def get_lambda(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.lambda_star
        else:
            return self.lambda_star * (step / self.warmup_steps)


class ExponentialWarmupLambdaScheduler(LambdaScheduler):
    """
    Exponential warmup scheduler:
        λ(t) = λ* × (1 - exp(-5 × t / total_steps))
    """

    def __init__(self, lambda_star: float, total_steps: int):
        super().__init__(lambda_star, total_steps)

    def get_lambda(self, step: int) -> float:
        import math
        progress = step / self.total_steps
        return self.lambda_star * (1 - math.exp(-5 * progress))


def create_lambda_scheduler(
    scheduler_type: str,
    lambda_star: float,
    total_steps: int,
    warmup_ratio: float = 0.1
) -> LambdaScheduler:
    """
    Factory function to create lambda scheduler.

    Args:
        scheduler_type: Type of scheduler ('fixed', 'linear_warmup', 'exp_warmup')
        lambda_star: Target lambda value
        total_steps: Total training steps
        warmup_ratio: Warmup ratio (for linear warmup)

    Returns:
        Lambda scheduler instance
    """
    if scheduler_type == 'fixed':
        return FixedLambdaScheduler(lambda_star, total_steps)
    elif scheduler_type == 'linear_warmup':
        return LinearWarmupLambdaScheduler(lambda_star, total_steps, warmup_ratio)
    elif scheduler_type == 'exp_warmup':
        return ExponentialWarmupLambdaScheduler(lambda_star, total_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
