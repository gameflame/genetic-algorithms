"""Public core interfaces, representations, and objective functions."""

from .contracts import (
    Chromosome,
    Crossover,
    DecodedCandidate,
    Decoder,
    GAConfig,
    GAEngine,
    GenerationRecord,
    Mutation,
    Objective,
    Population,
    ProblemSpec,
    RunResult,
)
from .encoding import BinaryEncoding
from .engine import run_ga
from .evaluation import evaluate_population
from .objectives import himmelblau, rosenbrock, sphere, zero_count
from .population import initialize_population
from .problems import make_benchmark_problem, make_discrete_problem
from .selection import tournament_select

__all__ = [
    "BinaryEncoding",
    "Chromosome",
    "Crossover",
    "DecodedCandidate",
    "Decoder",
    "GAConfig",
    "GAEngine",
    "GenerationRecord",
    "Mutation",
    "Objective",
    "Population",
    "ProblemSpec",
    "RunResult",
    "evaluate_population",
    "himmelblau",
    "initialize_population",
    "make_benchmark_problem",
    "make_discrete_problem",
    "rosenbrock",
    "run_ga",
    "sphere",
    "tournament_select",
    "zero_count",
]
