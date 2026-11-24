# Core interfaces

Status: Steps 1-7 provide these contracts, objective functions, binary encoding,
population initialization, problem factories, population evaluation, tournament
selection, and a reviewed generational engine with verified replay and results.
Crossover and mutation
implementations remain separate work and must be supplied by the caller.

## Data conventions

| Type | Required representation |
|---|---|
| `Chromosome` | One-dimensional NumPy `uint8` array of length `L`, containing only 0 or 1 |
| `Population` | Two-dimensional NumPy `uint8` array with shape `(N, L)` |
| `DecodedCandidate` | One-dimensional NumPy `float64` array of coordinates |
| Cost | Finite numeric value; **lower is better** for every problem |

Python type hints describe dtypes, but do not enforce array shapes or values.
Objective functions and representations already validate their inputs; other
consuming components will perform runtime checks when implemented. Continuous
benchmarks default to 16 bits per coordinate, most significant bit first,
and problem-specific bounds.

## Representation and initialization interfaces

```python
encoding = BinaryEncoding(bounds, bits_per_variable=16)
encoding.encode(candidate) -> chromosome
encoding.decode(chromosome) -> decoded_candidate
encoding.dimensions          # Number of bound pairs
encoding.chromosome_length   # dimensions * bits_per_variable
encoding.resolution          # Per-coordinate grid spacing, independent array

initialize_population(population_size, chromosome_length, rng) -> population
make_benchmark_problem(name, *, dimensions=2, bits_per_variable=16) -> ProblemSpec
make_discrete_problem(*, chromosome_length=32) -> ProblemSpec
```

Bounds contain one finite `(lower, upper)` pair per coordinate, with a positive
finite span. Bit counts must be integers from 1 through 32, and grid spacing
must be representable as a positive float64 value. Encoding rounds to the
nearest grid index (ties to even) and rejects coordinates outside the bounds.
Decoding validates binary values before conversion to uint8, preventing invalid
integers from wrapping into valid bits. Conversions return independent arrays.

Initialization requires integer sizes (`population_size >= 2`, chromosome
length at least 1), and a supplied NumPy `Generator`. It advances that
generator without touching global random state. Validation failures raise
`ValueError`.

Continuous factory names are exactly `sphere`, `rosenbrock`, and `himmelblau`.
Their metadata includes dimensions, bounds, bits per variable, resolution, and
bit order. The discrete factory preserves bits as float64 coordinates for the
zero-count objective. Problem metadata is independent of the decoder's bounds.

## Evaluation and selection interfaces

```python
evaluate_population(population, problem) -> costs
tournament_select(population, costs, tournament_size, rng) -> copied_parent
```

`evaluate_population` accepts a binary matrix `(N, L)` with `N >= 1`, validates
`L` against `problem.chromosome_length`, and returns a new `float64` array of
shape `(N,)`. Each row is decoded and evaluated exactly once in row order.
Decoder inputs are independent copies. Decoded values must be finite, non-empty
one-dimensional real numeric vectors, and each objective result a finite real
scalar. A `ValueError` for a decoded candidate or objective result identifies
the zero-based population row; callback validation errors are chained.

When invoked through `run_ga`, evaluation validation errors also include the
generation. The row index refers to the evaluated batch: initial population
for generation 0, new offspring thereafter (excluding the cached elite).
Programming errors outside `ValueError` propagate without retries.

`tournament_select` requires `N >= 2`, a finite one-dimensional cost array with
one value per row, an integer tournament size from 1 through N, and a NumPy
`Generator`. It samples without replacement within the tournament, chooses the
lowest cost, and returns an independent `uint8` chromosome. Negative costs are
valid. Equal-cost ties go to the first sampled tied contestant. Separate calls
can return the same individual, allowing independent parent selection.

Both functions validate binary values before converting to uint8 and preserve
their input arrays. Costs must describe the exact current population order and
contents; the caller is responsible for keeping them aligned. Selection uses
only the supplied generator and does not reseed or use global random state.

## Callable boundaries

```python
decode(chromosome) -> decoded_candidate
objective(decoded_candidate) -> float_cost
crossover(parent_a, parent_b, rng) -> (child_a, child_b)
mutation(child, mutation_rate, rng) -> mutated_child
run_ga(problem, config, *, crossover, mutation) -> result
```

Available objectives are `sphere`, `rosenbrock`, `himmelblau`, and `zero_count`,
exported from `ga` and defined in `ga.objectives`. They accept a single numeric
vector, return a finite Python float, and do not modify inputs. Sphere requires
at least one coordinate, Rosenbrock at least two, and Himmelblau exactly two.
Zero-count requires a non-empty binary vector and returns its number of zeros.
Invalid inputs and overflowing objective values raise `ValueError`.

`GAEngine` is the typing protocol implemented by `ga.run_ga`, defined in
`ga.engine`. `Decoder` and `Objective` allow each
`ProblemSpec` to supply its own interpretation of a binary chromosome.

## Ownership at the operator boundary

- The engine selects parents and owns the crossover probability check.
- The engine passes independent parent copies to crossover. If crossover is
  skipped, it makes independent child copies instead.
- Crossover returns two independent binary arrays of the same length as the
  parents and does not modify its inputs.
- The engine calls mutation once for each child it needs. Mutation returns an
  independent binary array without modifying its input.
- The mutation function owns the rate's meaning and its probability checks.
  Flip-bit uses a per-bit probability; inversion uses a per-child probability.
- All randomness uses the supplied NumPy `Generator`. Operators must not seed
  their own generators or use global random state.
- The engine validates operator outputs before adding offspring to a population.

Runtime crossover output must be a tuple or list containing exactly two valid
chromosomes. Both are validated and copied before either is passed to mutation,
even if there is space for only one offspring. Mutation is called only for
children that will enter the population, with the configured rate unchanged.
It is called even at rate zero; the mutation function owns that rate's behavior.
Value errors from operator calls/output validation include the generation and,
for mutation, the zero-based offspring index, retaining the original cause.

These are proposed integration contracts for the group to review. Operator
implementations and their experiments remain separate work.

## Configuration and results

`GAConfig` requires a mutation rate. The other defaults are population 100,
300 generations, tournament size 3, crossover probability 0.8, and seed 42.
The engine retains one elite each generation. Rates must be in `[0, 1]`,
population size at least 2, tournament size from 1 through population size,
generation count non-negative, and seed a non-negative integer. `run_ga` checks
these settings before initialization and stores a normalized configuration
with Python integer/float values in the result. Both operators must be callable,
including for zero generations or crossover probability zero.

`RunResult` retains the configuration, problem metadata, operator names, best
chromosome, decoded solution, cost, elapsed time, evaluation count, progress
history, Python/NumPy versions, and `rng_algorithm` (the actual bit-generator
class name). Metadata should use JSON-compatible values
such as strings, numbers, and lists to make later export straightforward.
File export is outside the core interface.

History starts at generation 0, after initial evaluation. A run with `G`
replacement generations has `G + 1` records. `best_so_far_cost` is the lowest
cost observed so far; `mean_cost` describes the current population. Evaluating
the initial `N` candidates and `N - 1` new offspring per generation gives
`N + G * (N - 1)` evaluations when the elite cost is reused.

The engine places the elite in row zero of the new population, then appends
offspring in pair order until it has exactly N rows. The first lowest-cost row
is retained when elite costs tie. A strictly improved best-so-far cost replaces
the saved best chromosome; equal costs preserve the first best solution.
Termination is solely the configured generation limit. Decoder/objective
functions must be deterministic so cached elite costs remain valid.

The run creates one `numpy.random.default_rng(config.seed)` and passes it to
initialization, selection, crossover, and mutation. Result fields capture
history, counts, timing, seed (inside config), operator names, runtime versions,
and the actual algorithm from `type(rng.bit_generator).__name__`.

Frozen dataclasses prevent field reassignment; they do not make contained
arrays or dictionaries immutable. The engine must return independent copies
of its solution arrays and metadata. Independent runs return independent
solution arrays and nested metadata. Reproducibility compares every result
field except elapsed time in the same implementation and environment. Callers
must preserve the problem/operator code, any settings outside GAConfig, and
consistent initial operator state; names alone cannot reconstruct callables.
Operators using global randomness or their own unseeded generators do not meet
the reproducibility contract. Different seeds need not produce distinct optima.

Step 6 verifies complete replay (including random-consuming operator calls),
independence from other runs/global random state, actual objective-call counts,
best solution/cost agreement, histories, measured elapsed time, and result
storage independence. File export and experiment aggregation are not added by
this interface.
