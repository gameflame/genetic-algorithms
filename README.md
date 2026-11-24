# Genetic Algorithms with Diverse Operators

CP468 group project: a modular genetic algorithm framework for comparing
crossover and mutation operators on benchmark optimization functions.

## Current progress

**Steps 1-7: core engine, reproducibility, and integration review.**

The package provides configuration/result dataclasses, typing contracts, four
objective functions, binary encoding/decoding, seeded population initialization,
continuous/discrete problem specifications, population evaluation, tournament
selection, and a generation loop with elite preservation and termination.
`run_ga` requires supplied crossover and mutation functions; their implementations
remain separate group work. Controlled tests verify the engine using test-only
operator doubles. Complete seeded replay, independent runs/results, and result
counts/history are verified. The core review passes 102 tests, including a
100-individual, 300-generation check. Final core documentation and handoff
preparation are next. See [CORE_REVIEW.md](CORE_REVIEW.md) for review evidence
and the remaining integration boundaries.

All objective values use the same convention: **smaller is better**.

## Setup on Windows

Install Python 3.12 or newer, then open PowerShell in your project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

If using a newer Python version, replace `-3.12` with its installed version.
These commands use the virtual environment directly; activation is unnecessary.
NumPy is pinned to 2.3.5 in `pyproject.toml` to keep the initial dependency
consistent across group members.

## Check the package

From the project root, after setup:

```powershell
.\.venv\Scripts\python.exe -c "from ga import GAConfig; print(GAConfig(mutation_rate=1 / 32))"
```

This should print the configuration with population 100, 300 generations,
tournament size 3, crossover probability 0.8, and seed 42. The mutation rate
here is an example suitable for the proposed 32-bit flip-bit operator; it does
not select or implement that operator.

## Structure

```text
ga/
  __init__.py          Public imports
  contracts.py         Data types and callable interfaces
  encoding.py          Bounded real coordinates encoded as binary chromosomes
  engine.py            Generation loop, elite preservation, and termination
  evaluation.py        One finite cost per population row
  objectives.py        Continuous benchmarks and discrete zero-count objective
  population.py        Initialization using a supplied random generator
  problems.py          Connect representations to objective functions
  selection.py         Tournament selection using minimization costs
  _validation.py       Shared representation and size checks
tests/
  __init__.py
  test_objectives.py   Known-value and input-validation checks
  test_encoding.py     Bit order, bounds, round trips, and precision
  test_engine.py       Controlled generation-loop and operator-boundary checks
  test_core_integration.py  Configuration matrix, failure recovery, default-size run
  test_evaluation.py   Population costs and callback validation
  test_population.py   Shape, storage, seeds, and generator use
  test_problems.py     Representation/objective integration
  test_reproducibility.py  Complete seeded replay and result integrity
  test_selection.py    Tournament membership, minimization, and copies
CORE_INTERFACES.md     Array conventions and integration contracts
CORE_REVIEW.md         Step 7 verification evidence and boundaries
pyproject.toml        Package metadata and dependency
```

See [CORE_INTERFACES.md](CORE_INTERFACES.md) for the exact boundaries between the
engine, problem definitions, and future operator functions.

## Objective functions

Each function evaluates one candidate and returns a Python `float`:

| Function | Input size | Objective | Known minimum |
|---|---|---|---|
| `sphere` | One or more coordinates | Sum of squared coordinates | 0 at the origin |
| `rosenbrock` | Two or more coordinates | Sum of adjacent-coordinate Rosenbrock terms | 0 when all coordinates are 1 |
| `himmelblau` | Exactly two coordinates | `(x*x + y - 11)**2 + (x + y*y - 7)**2` | 0 at four points, including `(3, 2)` |
| `zero_count` | One or more binary values | Number of zero bits | 0 when all bits are 1 |

For example:

```python
from ga import himmelblau, rosenbrock, sphere, zero_count

print(sphere([1, 2]))          # 5.0
print(rosenbrock([0, 0]))      # 1.0
print(himmelblau([3, 2]))      # 0.0
print(zero_count([0, 1, 0]))   # 2.0
```

Inputs can be one-dimensional real numeric arrays, lists, or tuples. Empty,
incorrectly shaped, non-numeric, complex, or non-finite inputs raise
`ValueError`. The discrete function also rejects values other than 0 and 1.
An objective calculation that overflows raises `ValueError` instead of
returning infinity or NaN. Inputs are not modified.

These functions calculate costs directly, including outside the proposed
search bounds. Bounds will be handled by the encoding/decoding component.
The continuous benchmark definitions are documented in the
[gek benchmark reference](https://cran.r-project.org/web/packages/gek/refman/gek.html).
The zero-count objective is a small project demonstration of discrete
minimization, rather than a fourth required continuous benchmark.

## Encoding and population initialization

`BinaryEncoding` represents each coordinate with a fixed block of 1-32 bits
(16 by default), most significant bit first. For a block with integer value
`k`, it decodes the coordinate as:

```text
lower + (upper - lower) * k / (2**bits_per_variable - 1)
```

All-zero blocks decode exactly to their lower bounds and all-one blocks to
their upper bounds. Encoding chooses the nearest representable grid point,
using round-to-even for an exact halfway tie. Out-of-bounds coordinates and
invalid chromosomes raise `ValueError`; input arrays are not modified.

```python
from ga import BinaryEncoding

encoding = BinaryEncoding(((0.0, 3.0),), bits_per_variable=2)
print(encoding.encode([2.0]))   # [1 0]
print(encoding.decode([0, 1]))  # [1.]
print(encoding.resolution)     # [1.]
```

The default two-dimensional benchmark problems have 32-bit chromosomes:

| Problem name | Default bounds per coordinate |
|---|---|
| `sphere` | `[-5.12, 5.12]` |
| `rosenbrock` | `[-2.048, 2.048]` |
| `himmelblau` | `[-5, 5]` |

`make_benchmark_problem` accepts these exact lowercase names, with optional
`dimensions` and `bits_per_variable`. Sphere supports one or more dimensions,
Rosenbrock two or more, and Himmelblau exactly two. Metadata records bounds,
precision, and bit order. For custom bounds, construct `BinaryEncoding`
directly and use its `decode` method in a `ProblemSpec`.

```python
import numpy as np
from ga import initialize_population, make_benchmark_problem

problem = make_benchmark_problem("sphere")
rng = np.random.default_rng(42)
population = initialize_population(100, problem.chromosome_length, rng)
print(population.shape)                     # (100, 32)
candidate = problem.decode(population[0])
print(candidate)                           # The first candidate's coordinates
print(problem.objective(candidate))        # Its cost, not an optimized result
```

Initialization requires a population size of at least 2 and a positive
chromosome length. It returns an independent `uint8` array of shape `(N, L)`
containing random 0/1 values. Recreating a NumPy generator with the same seed
reproduces the population in the same environment. Reusing a generator advances
its state; the initializer never reseeds it or uses global random state.

For the discrete example, `make_discrete_problem(chromosome_length=32)` returns
a specification that preserves the binary values as numeric coordinates and
evaluates their zero count. It does not apply continuous-coordinate bounds.

Binary encoding searches a finite grid. The coordinate spacing is
`(upper - lower) / (2**bits_per_variable - 1)`, available as `resolution`.
For Sphere's default bounds this is approximately `0.00015625` per coordinate.
Some mathematical optima, including the origin in this default grid, cannot
be represented exactly. Interpret small remaining costs using the encoding
precision rather than requiring every optimization result to equal zero.

## Population evaluation and tournament selection

`evaluate_population(population, problem)` decodes each row and evaluates its
objective once, returning a one-dimensional `float64` cost array in the same
row order. It accepts any non-empty binary population, including a single-row
offspring batch. Chromosome lengths must match the problem specification.
The decoder receives a copy of each chromosome to protect the population.
Invalid decoded coordinates or non-finite/non-scalar costs raise `ValueError`
with the failing row's zero-based index.

`tournament_select(population, costs, tournament_size, rng)` samples distinct
row indices without replacement and returns a copy of the lowest-cost sampled
parent. With the default tournament size of 3, three candidates compete for
one parent position. It may select the same parent again in another call.
Equal-cost ties go to the first sampled tied candidate.

Selection requires at least two population rows and a tournament size from 1
through the population size. Costs must be finite, with exactly one value per
row. Keep costs aligned with the current population: changing or reordering
the population requires updating the costs too. Both functions preserve their
input arrays; the selected parent can be modified independently.

```python
import numpy as np
from ga import evaluate_population, make_discrete_problem, tournament_select

problem = make_discrete_problem(chromosome_length=4)
population = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
costs = evaluate_population(population, problem)
print(costs)  # [4. 2. 0.]

# All three candidates compete here, so the all-ones chromosome wins.
parent = tournament_select(population, costs, 3, np.random.default_rng(42))
print(parent)  # [1 1 1 1]
```

## Generation loop

The engine's public entry point is:

```text
run_ga(problem, config, *, crossover, mutation) -> RunResult
```

Provide a `ProblemSpec`, a `GAConfig`, and both operator functions matching the
interfaces in `CORE_INTERFACES.md`. There are no default operator implementations.
The engine validates sizes, generation count, probabilities, and seed before
initialization, and creates one random generator shared by all random steps.

Each run:

1. Initializes and evaluates the starting population, recording generation 0.
2. Copies the lowest-cost elite into the next population, retaining its cost.
3. Selects parent pairs and applies the configured crossover probability.
4. Calls the supplied mutation for each child needed to fill the population.
5. Evaluates the `N - 1` new offspring and replaces the population.
6. Records progress and repeats until the generation limit is reached.

Population size remains exactly `N` for odd and even sizes. If only one slot
remains from a pair, its second child is not mutated or evaluated. Both children
returned by crossover must still be valid. A zero-generation run evaluates only
the initial population; it still requires callable operator arguments. The
engine stops at the generation limit rather than stopping early at a threshold.

History has `G + 1` entries, and objective evaluation count is
`N + G * (N - 1)`. Decoding the best chromosome for the final result does not
add an objective evaluation. Best-so-far cost never worsens. Decoder and objective
functions must be deterministic because elite costs are reused.

Evaluation validation failures include both the generation and zero-based row
in the evaluated batch. After generation 0, this batch contains new offspring
only, excluding the cached elite. Invalid data stops the run with the original
error chained; callback programming errors propagate without retries.

Results include the best chromosome, decoded solution, cost, configuration,
problem metadata, operator names, progress history, evaluation count, elapsed
time, Python/NumPy versions, and the actual RNG algorithm. Result arrays and nested problem metadata are
independent copies. File export and plotting remain outside the engine.

The current tests exercise the complete loop without implementing the group's
operators. To run those checks alone:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_engine -v
```

The operator doubles are confined to the tests; they are controlled fixtures
for engine correctness, and their outputs are not experimental GA findings.

## Reproducibility and result verification

Each run creates a fresh generator from `config.seed` (42 by default) and shares
it with initialization, selection, and the supplied operators. No previous run's
generator is reused. Under the same implementation and environment, repeating
the same configuration, problem, and operator behavior reproduces the chromosome,
decoded solution, cost, evaluation count, and complete generation history.
**Elapsed time is measured separately and is excluded from replay comparisons.**

Operators must use only the supplied generator for randomness. Any other
operator state must also start consistently. Different seeds change the random
stream, but can still lead to the same best solution. Changing an operator or
its random draws can change the remaining stream even with the same seed.
Exact replay across different Python/NumPy builds or machines is not promised.

The returned `RunResult` contains:

| Field | Meaning |
|---|---|
| `config` | Normalized run settings, including `config.seed` |
| `problem_name`, `problem_metadata` | Problem label and representation details |
| `crossover_name`, `mutation_name` | Labels for the supplied operator callables |
| `best_chromosome`, `best_solution`, `best_cost` | Best binary candidate, decoded values, and cost |
| `history` | Generation 0 through G, with best-so-far cost, population mean, and cumulative evaluations |
| `evaluation_count` | Actual objective evaluation budget: `N + G * (N - 1)` |
| `elapsed_seconds` | Measured wall-clock duration of the run |
| `python_version`, `numpy_version`, `rng_algorithm` | Runtime versions and actual bit-generator class, such as `PCG64` |

Problem and operator names are labels, not serialized implementations. Replaying
a custom problem/operator also requires keeping its code and any settings held
outside `GAConfig`. Preserve the recorded environment versions for comparisons.

Result arrays and nested metadata are independent snapshots. Editing one result
does not affect the source problem, another result, or a future run. Frozen
dataclasses prevent field reassignment but do not freeze array/dictionary
contents; changing a result can still make that individual snapshot inconsistent.

The reproducibility checks exercise all three benchmarks and the discrete
problem using random-consuming test-only operator fixtures. They compare every
result field except time, check actual objective calls against recorded counts,
verify decoded best solutions and histories, and test independence from global
random state and unrelated runs. To run these checks alone:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest tests.test_reproducibility -v
```

This step verifies the engine's random-state and result contracts. File export,
batch experiments, statistics, and plots remain separate work.

## Tests

Use Python's built-in unittest runner:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite checks known optimum and non-optimum values, higher-dimensional
Sphere/Rosenbrock inputs, all four Himmelblau minima, discrete costs, input
validation, input preservation, and objective overflow. Representation tests
also check bit order, exact bounds, quantization error, small-grid round trips,
seeded initialization, generator state, independent storage, and integration
with all four objective functions. Evaluation/selection checks cover known
population costs, per-row callback calls, callback isolation, invalid results,
tournament membership, negative costs, tie handling, seeded reproducibility,
and independent selected parents. Engine tests cover exact generation and
evaluation counts, zero generations, odd/even population sizes, elite survival,
improved offspring, operator call counts, callback isolation, invalid operator
outputs, configuration checks, and consistent results on all three benchmarks.
Full-run reproducibility checks cover repeated/interleaved seeds, random operator
draws, global-state independence, probability endpoints, runtime/RNG metadata,
actual objective-call counts, measured timing, and independence across results.
Core integration checks also cover representative combinations of population
size, precision, dimensions, tournament size, and probability endpoints; failure
recovery; and the default 100-individual, 300-generation budget. These checks use
test-only operators and do not measure the real operators' optimization quality.
