"""Initialization of independent random binary candidates."""

import numpy as np
from numpy.random import Generator

from ._validation import require_integer
from .contracts import Population


def initialize_population(
    population_size: int, chromosome_length: int, rng: Generator
) -> Population:
    """Draw an N-by-L uint8 population using only the supplied generator.

    Recreate the generator from the same seed to reproduce its draws. Reusing
    a generator advances its state, as required by a later complete GA run.
    """
    size = require_integer(population_size, name="population_size", minimum=2)
    length = require_integer(chromosome_length, name="chromosome_length")
    if not isinstance(rng, Generator):
        raise ValueError("rng must be a NumPy random Generator.")
    return rng.integers(0, 2, size=(size, length), dtype=np.uint8)
