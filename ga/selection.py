"""Tournament selection for minimization objectives."""

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike

from ._validation import binary_population, real_array, require_integer
from .contracts import Chromosome


def tournament_select(
    population: ArrayLike,
    costs: ArrayLike,
    tournament_size: int,
    rng: Generator,
) -> Chromosome:
    """Copy the lowest-cost parent from a tournament sampled without replacement.

    Costs must correspond to the current population's row order. For equal
    costs, the first sampled tied contestant wins. Separate calls may choose
    the same parent, although an individual tournament has distinct members.
    """
    chromosomes = binary_population(population, minimum_size=2)
    values = real_array(costs, name="Costs")
    population_size = chromosomes.shape[0]
    if values.ndim != 1 or values.size != population_size:
        raise ValueError("Costs must be one-dimensional with one value per population row.")
    size = require_integer(
        tournament_size, name="tournament_size", maximum=population_size
    )
    if not isinstance(rng, Generator):
        raise ValueError("rng must be a NumPy random Generator.")
    contestants = rng.choice(population_size, size=size, replace=False)
    winner = contestants[np.argmin(values[contestants])]
    return chromosomes[winner].copy()
