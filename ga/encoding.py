"""Binary encoding of bounded real coordinates, most significant bit first."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import binary_vector, real_array, require_integer
from .contracts import Chromosome, DecodedCandidate


@dataclass(frozen=True)
class BinaryEncoding:
    """Map each real coordinate to a fixed-size block of 1-32 bits.

    Bounds are normalized to independent immutable tuples. Encoding chooses
    the nearest grid index, using round-to-even for an exact halfway tie.
    """

    bounds: tuple[tuple[float, float], ...]
    bits_per_variable: int = 16

    def __post_init__(self) -> None:
        bits = require_integer(
            self.bits_per_variable, name="bits_per_variable", maximum=32
        )
        bounds = real_array(self.bounds, name="Bounds")
        if bounds.ndim != 2 or bounds.shape[0] == 0 or bounds.shape[1] != 2:
            raise ValueError("Bounds must contain one (lower, upper) pair per coordinate.")
        with np.errstate(over="ignore", invalid="ignore"):
            spans = bounds[:, 1] - bounds[:, 0]
        if np.any(spans <= 0) or not np.all(np.isfinite(spans)):
            raise ValueError("Each bound must have lower < upper and a finite span.")
        with np.errstate(under="ignore"):
            resolution = spans / ((1 << bits) - 1)
        if np.any(resolution == 0):
            raise ValueError("Bounds are too narrow for the requested float64 resolution.")
        object.__setattr__(self, "bits_per_variable", bits)
        object.__setattr__(
            self, "bounds", tuple((float(lower), float(upper)) for lower, upper in bounds)
        )

    @property
    def dimensions(self) -> int:
        return len(self.bounds)

    @property
    def chromosome_length(self) -> int:
        return self.dimensions * self.bits_per_variable

    @property
    def resolution(self) -> DecodedCandidate:
        """Return the coordinate grid spacing, as a newly allocated array."""
        bounds = np.asarray(self.bounds, dtype=np.float64)
        with np.errstate(under="ignore"):
            return (bounds[:, 1] - bounds[:, 0]) / ((1 << self.bits_per_variable) - 1)

    def decode(self, chromosome: ArrayLike) -> DecodedCandidate:
        """Decode one binary chromosome to an independent coordinate array."""
        bits = binary_vector(chromosome, length=self.chromosome_length)
        shifts = np.arange(self.bits_per_variable - 1, -1, -1, dtype=np.int64)
        weights = np.left_shift(np.uint64(1), shifts.astype(np.uint64))
        blocks = bits.reshape(self.dimensions, self.bits_per_variable)
        indices = blocks.astype(np.uint64) @ weights
        maximum = (1 << self.bits_per_variable) - 1
        bounds = np.asarray(self.bounds, dtype=np.float64)
        lower, upper = bounds[:, 0], bounds[:, 1]
        coordinates = lower + (indices.astype(np.float64) / maximum) * (upper - lower)
        # Preserve exact endpoints even when lower + span rounds past upper.
        coordinates = np.clip(coordinates, lower, upper)
        coordinates[indices == 0] = lower[indices == 0]
        coordinates[indices == maximum] = upper[indices == maximum]
        return coordinates

    def encode(self, candidate: ArrayLike) -> Chromosome:
        """Encode bounded coordinates; reject out-of-range values rather than clip."""
        coordinates = real_array(candidate, name="Candidate")
        if coordinates.ndim != 1 or coordinates.size != self.dimensions:
            raise ValueError(
                f"Candidate must be one-dimensional with {self.dimensions} coordinates."
            )
        bounds = np.asarray(self.bounds, dtype=np.float64)
        lower, upper = bounds[:, 0], bounds[:, 1]
        if np.any(coordinates < lower) or np.any(coordinates > upper):
            raise ValueError("Candidate coordinates must be within their bounds.")
        maximum = (1 << self.bits_per_variable) - 1
        indices = np.rint(
            (coordinates - lower) / (upper - lower) * maximum
        ).astype(np.uint64)
        shifts = np.arange(self.bits_per_variable - 1, -1, -1, dtype=np.int64)
        bits = np.right_shift(indices[:, None], shifts.astype(np.uint64)) & np.uint64(1)
        return bits.astype(np.uint8).reshape(-1)
