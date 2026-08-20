"""
Training module for preference learning with label encoding.
"""

from .scheduler import (
    LambdaScheduler,
    FixedLambdaScheduler,
    LinearWarmupLambdaScheduler,
    ExponentialWarmupLambdaScheduler,
    create_lambda_scheduler
)
from .diagnostics import (
    DiagnosticsTracker,
    compute_responsibility_quality,
    compute_prediction_distribution_stats,
    check_numerical_stability,
    log_training_step
)
from .trainer import Trainer

__all__ = [
    'LambdaScheduler',
    'FixedLambdaScheduler',
    'LinearWarmupLambdaScheduler',
    'ExponentialWarmupLambdaScheduler',
    'create_lambda_scheduler',
    'DiagnosticsTracker',
    'compute_responsibility_quality',
    'compute_prediction_distribution_stats',
    'check_numerical_stability',
    'log_training_step',
    'Trainer'
]
