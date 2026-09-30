"""Convert a cache byte budget into residual bit counts after RVQ storage.

One bit-count unit is one bit per scalar in an equal-sized chunk. Shared
codebooks and residual metadata are outside this payload accounting scope.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualBudget:
    total_bytes: int
    chunk_elements: int
    base_bytes_by_step: tuple[int, ...]

    def __post_init__(self):
        values = (self.total_bytes, self.chunk_elements, *self.base_bytes_by_step)
        if any(type(value) is not int for value in values):
            raise ValueError("cache budget sizes must be integer byte/element counts")
        if (
            self.total_bytes < 0
            or self.chunk_elements <= 0
            or not self.base_bytes_by_step
        ):
            raise ValueError("invalid cache budget geometry")
        previous = 0
        for size in self.base_bytes_by_step:
            if not previous <= size <= self.total_bytes:
                raise ValueError(
                    "RVQ Base exceeds total capacity or decreases across steps"
                )
            previous = size

    @property
    def capacities(self) -> tuple[int, ...]:
        # Subtract bytes before converting, and floor only once. Rounding the
        # base cost down first would allow allocations beyond the byte budget.
        return tuple(
            (self.total_bytes - size) * 8 // self.chunk_elements
            for size in self.base_bytes_by_step
        )

    def as_dict(self):
        return dict(
            total_bytes=self.total_bytes,
            chunk_elements=self.chunk_elements,
            base_bytes_by_step=list(self.base_bytes_by_step),
            residual_capacity_by_step=list(self.capacities),
            scope="rvq_base_payload_plus_nominal_residual",
        )

    @classmethod
    def from_dict(cls, raw):
        budget = cls(
            raw["total_bytes"], raw["chunk_elements"], tuple(raw["base_bytes_by_step"])
        )
        if tuple(raw["residual_capacity_by_step"]) != budget.capacities:
            raise ValueError(
                "residual capacities disagree with RVQ Base byte accounting"
            )
        return budget


def rvq_schedule_budget(config, terminal: int) -> ResidualBudget:
    """Conservative byte reservation for the deployed no-norm L16C64 codec.

    Each RVQ stage stores a uint8 index per vector (not a packed six-bit
    index), plus one uint8 relative-stop depth per chunk. Reserving max_depth
    makes the frozen schedule independent of input-dependent early stopping.
    Sizes describe one batch item and one K or V stream in one layer. Larger
    batches amortize the depth byte, so this per-item reservation remains safe.
    """
    calibration, model = config["calibration"], config["model_config"]
    if (
        calibration["input_norm_mode"] != "none"
        or calibration["centroids"] != 64
        or calibration["levels"] != 16
    ):
        raise ValueError("RVQ Base reservation requires the no-norm L16C64 codec")
    if type(terminal) is not int or terminal < 0:
        raise ValueError("terminal residual bit count must be a nonnegative integer")
    vectors = model["heads"] * model["spatial_tokens"] * model["latent_chunk_frames"]
    elements = vectors * model["head_dim"]
    base_bytes = vectors * calibration["early_stop"]["max_depth"] + 1
    horizon = calibration["schedule"]["horizon"]
    return ResidualBudget(
        total_bytes=horizon * base_bytes + (terminal * elements + 7) // 8,
        chunk_elements=elements,
        base_bytes_by_step=tuple(step * base_bytes for step in range(1, horizon + 1)),
    )
