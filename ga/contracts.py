"""Shared data types and callable contracts; all objectives are minimized.

These declarations establish module boundaries. They do not implement a GA,
validate runtime inputs, or supply crossover and mutation operators.
"""

from dataclasses import dataclass, field
from typing import Mapping, Protocol, TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


# Array shapes and binary-value constraints are documented in CORE_INTERFACES.md.
Chromosome: TypeAlias = NDArray[np.uint8]
Population: TypeAlias = NDArray[np.uint8]
DecodedCandidate: TypeAlias = NDArray[np.float64]


class Decoder(Protocol):
    """Convert one chromosome into the coordinates evaluated by an objective."""

    def __call__(self, chromosome: Chromosome) -> DecodedCandidate: ...


class Objective(Protocol):
    """Return a finite cost for one decoded candidate; smaller is better."""

    def __call__(self, candidate: DecodedCandidate) -> float: ...


class Crossover(Protocol):
    """Produce two independent children without modifying either parent."""

    def __call__(
        self,
        parent_a: Chromosome,
        parent_b: Chromosome,
        rng: Generator,
    ) -> tuple[Chromosome, Chromosome]: ...


class Mutation(Protocol):
    """Return an independent mutated child, interpreting the supplied rate."""

    def __call__(
        self,
        child: Chromosome,
        mutation_rate: float,
        rng: Generator,
    ) -> Chromosome: ...


@dataclass(frozen=True)
class ProblemSpec:
    """Connect a chromosome representation to its decoder and objective.

    Metadata describes the problem, for example bounds and bits per coordinate.
    A decoder for a discrete problem can expose the bits as numeric coordinates.
    """

    name: str
    chromosome_length: int
    decode: Decoder
    objective: Objective
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GAConfig:
    """Settings for one run; mutation rate is always an explicit choice.

    One elite is retained by the planned engine. A generation limit of zero
    means evaluating and returning the initial population only.
    """

    mutation_rate: float
    population_size: int = 100
    generations: int = 300
    tournament_size: int = 3
    crossover_probability: float = 0.8
    seed: int = 42


@dataclass(frozen=True)
class GenerationRecord:
    """Progress after evaluating generation zero or a replacement generation."""

    generation: int
    best_so_far_cost: float
    mean_cost: float
    cumulative_evaluations: int


@dataclass(frozen=True)
class RunResult:
    """Outputs from a completed run, independent of plotting or file export.

    The seed is stored in config. Arrays and metadata must be copied when the
    engine constructs a result; frozen dataclasses do not freeze array contents.
    Runtime versions and rng_algorithm describe the replay environment;
    elapsed_seconds is excluded when comparing otherwise identical seeded runs.
    """

    problem_name: str
    problem_metadata: Mapping[str, object]
    config: GAConfig
    crossover_name: str
    mutation_name: str
    best_chromosome: Chromosome
    best_solution: DecodedCandidate
    best_cost: float
    elapsed_seconds: float
    evaluation_count: int
    history: tuple[GenerationRecord, ...]
    python_version: str
    numpy_version: str
    rng_algorithm: str


class GAEngine(Protocol):
    """Contract implemented by run_ga; operators are required inputs."""

    def __call__(
        self,
        problem: ProblemSpec,
        config: GAConfig,
        *,
        crossover: Crossover,
        mutation: Mutation,
    ) -> RunResult: ...
