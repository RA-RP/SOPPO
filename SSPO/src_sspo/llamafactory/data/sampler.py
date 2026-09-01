"""Sampling utilities for mixed labeled/unlabeled preference training."""

import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class PreferenceTwoStreamSampler(Sampler[int]):
    """Arrange labeled and unlabeled preference rows into deterministic mixed batches.

    Every selected row is used at most once per epoch. The selected totals follow
    the labeled:unlabeled ratio of the dataset up to integer rounding, while each
    full batch contains at least ``min_unlabeled_per_batch`` unlabeled pairs.
    ``min_labeled_per_batch`` may be zero when the source labeled ratio is
    smaller than one divided by the requested physical batch size.
    """

    def __init__(
        self,
        data_types: Sequence[str],
        batch_size: int,
        seed: int,
        min_labeled_per_batch: int = 0,
        min_unlabeled_per_batch: int = 2,
        unlabeled_type: str = "unlabeled_pair",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive for two-stream sampling.")
        if min_labeled_per_batch < 0 or min_unlabeled_per_batch < 1:
            raise ValueError(
                "Two-stream sampling requires a non-negative labeled minimum and at least one unlabeled row per batch."
            )
        if min_labeled_per_batch + min_unlabeled_per_batch > batch_size:
            raise ValueError("The two-stream minima exceed `batch_size`.")

        if unlabeled_type not in {"unlabeled", "unlabeled_pair"}:
            raise ValueError(f"Unsupported unlabeled row type: {unlabeled_type}")

        self.labeled_indices = [index for index, data_type in enumerate(data_types) if data_type == "labeled"]
        self.unlabeled_indices = [index for index, data_type in enumerate(data_types) if data_type == unlabeled_type]
        unsupported = sorted(set(data_types) - {"labeled", unlabeled_type})
        if unsupported:
            raise ValueError(
                f"Two-stream sampling accepts only `labeled` and `{unlabeled_type}` rows, "
                f"but found: {unsupported}. Regenerate the mixed preference dataset."
            )
        if not self.labeled_indices or not self.unlabeled_indices:
            raise ValueError("Two-stream sampling requires both labeled and unlabeled rows.")

        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.min_labeled_per_batch = min_labeled_per_batch
        self.min_unlabeled_per_batch = min_unlabeled_per_batch

        total_examples = len(self.labeled_indices) + len(self.unlabeled_indices)
        self.num_batches = total_examples // batch_size
        if self.num_batches == 0:
            raise ValueError("StaticPE needs at least one full batch.")

        self.num_samples = self.num_batches * batch_size
        lower_labeled = max(
            self.num_batches * min_labeled_per_batch,
            self.num_samples - len(self.unlabeled_indices),
        )
        upper_labeled = min(
            len(self.labeled_indices),
            self.num_samples - self.num_batches * min_unlabeled_per_batch,
        )
        if lower_labeled > upper_labeled:
            raise ValueError(
                "Cannot construct two-stream batches without replacement using the requested "
                "batch size and minimum unlabeled count."
            )

        labeled_ratio = len(self.labeled_indices) / total_examples
        ratio_target = round(self.num_samples * labeled_ratio)
        self.num_labeled = min(max(ratio_target, lower_labeled), upper_labeled)
        self.num_unlabeled = self.num_samples - self.num_labeled

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self.seed + self.epoch)
        labeled = self.labeled_indices.copy()
        unlabeled = self.unlabeled_indices.copy()
        generator.shuffle(labeled)
        generator.shuffle(unlabeled)
        labeled = labeled[: self.num_labeled]
        unlabeled = unlabeled[: self.num_unlabeled]

        base_labeled, remainder = divmod(self.num_labeled, self.num_batches)
        labeled_counts = [base_labeled + int(batch_index < remainder) for batch_index in range(self.num_batches)]
        generator.shuffle(labeled_counts)

        ordered_indices = []
        labeled_offset = 0
        unlabeled_offset = 0
        for labeled_count in labeled_counts:
            unlabeled_count = self.batch_size - labeled_count
            batch = labeled[labeled_offset : labeled_offset + labeled_count]
            batch += unlabeled[unlabeled_offset : unlabeled_offset + unlabeled_count]
            generator.shuffle(batch)
            ordered_indices.extend(batch)
            labeled_offset += labeled_count
            unlabeled_offset += unlabeled_count

        return iter(ordered_indices)


# Backward-compatible name used by the initial StaticPE implementation.
StaticPETwoStreamSampler = PreferenceTwoStreamSampler
