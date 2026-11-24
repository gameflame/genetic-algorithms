"""Ready-to-use problem specifications connecting encodings to objectives."""

import numpy as np

from ._validation import binary_vector, require_integer
from .contracts import Chromosome, DecodedCandidate, ProblemSpec
from .encoding import BinaryEncoding
from .objectives import himmelblau, rosenbrock, sphere, zero_count


def make_benchmark_problem(
    name: str, *, dimensions: int = 2, bits_per_variable: int = 16
) -> ProblemSpec:
    """Build a sphere, rosenbrock, or himmelblau problem with default bounds."""
    benchmarks = {
        "sphere": (sphere, (-5.12, 5.12), 1),
        "rosenbrock": (rosenbrock, (-2.048, 2.048), 2),
        "himmelblau": (himmelblau, (-5.0, 5.0), 2),
    }
    if not isinstance(name, str) or name not in benchmarks:
        raise ValueError("Unknown benchmark; choose sphere, rosenbrock, or himmelblau.")
    objective, bound, minimum_dimensions = benchmarks[name]
    size = require_integer(dimensions, name="dimensions", minimum=minimum_dimensions)
    if name == "himmelblau" and size != 2:
        raise ValueError("Himmelblau requires exactly 2 dimensions.")
    encoding = BinaryEncoding((bound,) * size, bits_per_variable=bits_per_variable)
    return ProblemSpec(
        name=name,
        chromosome_length=encoding.chromosome_length,
        decode=encoding.decode,
        objective=objective,
        metadata={
            "representation": "binary",
            "dimensions": size,
            "bits_per_variable": encoding.bits_per_variable,
            "bounds": [list(pair) for pair in encoding.bounds],
            "resolution": encoding.resolution.tolist(),
            "bit_order": "most_significant_first",
        },
    )


def make_discrete_problem(*, chromosome_length: int = 32) -> ProblemSpec:
    """Build the zero-count problem without converting bits to real ranges."""
    length = require_integer(chromosome_length, name="chromosome_length")

    def decode(chromosome: Chromosome) -> DecodedCandidate:
        return binary_vector(chromosome, length=length).astype(np.float64)

    return ProblemSpec(
        name="zero_count",
        chromosome_length=length,
        decode=decode,
        objective=zero_count,
        metadata={"representation": "binary", "dimensions": length},
    )
