"""Evaluate candidate populations in their existing row order."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import binary_population, real_array, require_integer
from .contracts import ProblemSpec


def evaluate_population(population: ArrayLike, problem: ProblemSpec) -> NDArray[np.float64]:
    """Return one finite minimization cost per chromosome, without reordering.

    Single-row batches are allowed for later offspring evaluation. Each row is
    decoded and evaluated once, and the decoder receives an independent copy.
    Invalid decoded candidates or costs raise ValueError identifying the row.
    """
    if not isinstance(problem, ProblemSpec):
        raise ValueError("problem must be a ProblemSpec.")
    length = require_integer(problem.chromosome_length, name="problem.chromosome_length")
    if not callable(problem.decode) or not callable(problem.objective):
        raise ValueError("Problem decoder and objective must be callable.")
    chromosomes = binary_population(population, chromosome_length=length)
    costs = np.empty(chromosomes.shape[0], dtype=np.float64)
    for index, chromosome in enumerate(chromosomes):
        try:
            candidate = real_array(problem.decode(chromosome.copy()), name="Decoded candidate")
            if candidate.ndim != 1 or candidate.size == 0:
                raise ValueError("Decoded candidate must be a non-empty one-dimensional vector.")
            cost = real_array(problem.objective(candidate), name="Objective cost")
            if cost.ndim != 0:
                raise ValueError("Objective cost must be a scalar.")
            costs[index] = float(cost)
        except ValueError as exc:
            raise ValueError(f"Cannot evaluate population row {index}: {exc}") from exc
    return costs
