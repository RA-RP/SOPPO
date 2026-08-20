"""No-op reward for Cycle 08 OPD (pure distillation, use_task_rewards=False).

verl's V1 RewardLoopWorker always runs and dispatches a task reward by data_source;
for an unregistered data_source ("Math-CoT-20k") it raises NotImplementedError. Since
OPD here is supervised distillation only (the distillation loss ignores task rewards),
we supply this constant-0 scorer so the reward loop is a no-op instead of crashing.
Kwargs-tolerant to survive verl signature variations across versions.
"""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    return 0.0
