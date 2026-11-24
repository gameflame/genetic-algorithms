"""Generational GA orchestration with one elite and supplied operators."""

from copy import deepcopy
from platform import python_version
from time import perf_counter

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ._validation import binary_vector, real_array, require_integer
from .contracts import (
    Crossover,
    GAConfig,
    GenerationRecord,
    Mutation,
    Population,
    ProblemSpec,
    RunResult,
)
from .evaluation import evaluate_population
from .population import initialize_population
from .selection import tournament_select


def _probability(value: float, *, name: str) -> float:
    values = real_array(value, name=name)
    if values.ndim != 0 or not 0.0 <= float(values) <= 1.0:
        raise ValueError(f"{name} must be a scalar probability in [0, 1].")
    return float(values)


def _validated_config(config: GAConfig) -> GAConfig:
    """Validate before starting a run and normalize NumPy scalar settings."""
    if not isinstance(config, GAConfig):
        raise ValueError("config must be a GAConfig.")
    size = require_integer(config.population_size, name="population_size", minimum=2)
    return GAConfig(
        mutation_rate=_probability(config.mutation_rate, name="mutation_rate"),
        population_size=size,
        generations=require_integer(config.generations, name="generations", minimum=0),
        tournament_size=require_integer(
            config.tournament_size, name="tournament_size", maximum=size
        ),
        crossover_probability=_probability(
            config.crossover_probability, name="crossover_probability"
        ),
        seed=require_integer(config.seed, name="seed", minimum=0),
    )


def _evaluate_generation(
    population: Population, problem: ProblemSpec, generation: int
) -> NDArray[np.float64]:
    """Keep the generation alongside the evaluator's row-level failure context."""
    try:
        return evaluate_population(population, problem)
    except ValueError as exc:
        raise ValueError(f"Evaluation at generation {generation}: {exc}") from exc


def _offspring(
    population: Population,
    costs: NDArray[np.float64],
    config: GAConfig,
    crossover: Crossover,
    mutation: Mutation,
    rng: Generator,
    generation: int,
) -> Population:
    """Fill exactly N-1 slots; mutate only the children that will be used."""
    length = population.shape[1]
    children = np.empty((config.population_size - 1, length), dtype=np.uint8)
    filled = 0
    while filled < len(children):
        parent_a = tournament_select(population, costs, config.tournament_size, rng)
        parent_b = tournament_select(population, costs, config.tournament_size, rng)
        if rng.random() < config.crossover_probability:
            try:
                pair = crossover(parent_a, parent_b, rng)
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("Crossover must return exactly two children.")
                # Copy both before mutation, even if an operator returns aliases.
                pair = tuple(binary_vector(child, length=length) for child in pair)
            except ValueError as exc:
                raise ValueError(f"Crossover at generation {generation}: {exc}") from exc
        else:
            pair = (parent_a, parent_b)
        for child in pair:
            if filled == len(children):
                break
            try:
                mutated = mutation(child, config.mutation_rate, rng)
                children[filled] = binary_vector(mutated, length=length)
            except ValueError as exc:
                raise ValueError(
                    f"Mutation at generation {generation}, offspring {filled}: {exc}"
                ) from exc
            filled += 1
    return children


def _record(
    generation: int,
    costs: NDArray[np.float64],
    best_cost: float,
    evaluations: int,
) -> GenerationRecord:
    # Scaling avoids sum overflow when all individual costs are finite but large.
    scale = float(np.max(np.abs(costs)))
    mean_cost = float(np.mean(costs / scale) * scale) if scale else 0.0
    return GenerationRecord(generation, best_cost, mean_cost, evaluations)


def run_ga(
    problem: ProblemSpec,
    config: GAConfig,
    *,
    crossover: Crossover,
    mutation: Mutation,
) -> RunResult:
    """Run G replacement generations, retaining the lowest-cost elite each time.

    Both operators are required callables, including for a zero-generation run.
    The problem's decoder and objective must be deterministic: elite costs are
    cached, and the best chromosome is decoded again when assembling the result.
    No crossover or mutation implementations are supplied by this module.
    """
    config = _validated_config(config)
    if not isinstance(problem, ProblemSpec):
        raise ValueError("problem must be a ProblemSpec.")
    length = require_integer(problem.chromosome_length, name="problem.chromosome_length")
    if not callable(problem.decode) or not callable(problem.objective):
        raise ValueError("Problem decoder and objective must be callable.")
    if not callable(crossover) or not callable(mutation):
        raise ValueError("Crossover and mutation must both be supplied callables.")

    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    population = initialize_population(config.population_size, length, rng)
    costs = _evaluate_generation(population, problem, 0)
    evaluations = config.population_size
    best_index = int(np.argmin(costs))
    best_chromosome = population[best_index].copy()
    best_cost = float(costs[best_index])
    history = [_record(0, costs, best_cost, evaluations)]

    for generation in range(1, config.generations + 1):
        elite_index = int(np.argmin(costs))
        next_population = np.empty_like(population)
        next_costs = np.empty_like(costs)
        next_population[0] = population[elite_index]
        next_costs[0] = costs[elite_index]

        children = _offspring(
            population, costs, config, crossover, mutation, rng, generation
        )
        next_population[1:] = children
        next_costs[1:] = _evaluate_generation(children, problem, generation)
        evaluations += len(children)
        population, costs = next_population, next_costs

        best_index = int(np.argmin(costs))
        if costs[best_index] < best_cost:
            best_cost = float(costs[best_index])
            best_chromosome = population[best_index].copy()
        history.append(_record(generation, costs, best_cost, evaluations))

    best_solution = real_array(
        problem.decode(best_chromosome.copy()), name="Best solution"
    ).copy()
    if best_solution.ndim != 1 or best_solution.size == 0:
        raise ValueError("Best solution must be a non-empty one-dimensional vector.")
    return RunResult(
        problem_name=problem.name,
        problem_metadata=deepcopy(dict(problem.metadata)),
        config=config,
        crossover_name=getattr(crossover, "__name__", type(crossover).__name__),
        mutation_name=getattr(mutation, "__name__", type(mutation).__name__),
        best_chromosome=best_chromosome.copy(),
        best_solution=best_solution,
        best_cost=best_cost,
        elapsed_seconds=perf_counter() - started,
        evaluation_count=evaluations,
        history=tuple(history),
        python_version=python_version(),
        numpy_version=np.__version__,
        rng_algorithm=type(rng.bit_generator).__name__,
    )
