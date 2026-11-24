# Step 7: core correctness and integration review

Review date: November 19, 2026.

## Outcome

The reviewed core passes **102 tests**, including a default-size generation-loop
check. One diagnostic gap was fixed: evaluation failures now identify both the
generation and the evaluator's row, while preserving the original exception.

The review used Python 3.12.14, NumPy 2.3.5, and the PCG64 generator. The suite
ran with warnings treated as errors:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

## Checks completed

| Area | Evidence |
|---|---|
| Mathematical objectives | Known optimum/non-optimum values, dimensions, finite values, and overflow checks |
| Representation | Exact endpoints, most-significant-bit order, quantization, round trips, and independent storage |
| Evaluation and selection | Ordered costs, actual callback counts, correct minimization, ties, and copied parents |
| Configuration combinations | Nine integration cases covering minimum/odd/even population sizes, tournament sizes 1 and N, dimensions up to 5, 1/8/16/32 bits per coordinate, and probability endpoints |
| Default-size loop | Sphere with N=100 and G=300: 301 history records and exactly 29,800 objective calls |
| Replacement | One preserved elite, fixed population sizes, improvements, and non-worsening best-so-far cost |
| Failure handling | Initial and offspring evaluation context, invalid outputs, no retries of callback programming errors, and reproducible recovery after a failed run |
| Reproducibility | Seeded replay, global-state independence, runtime metadata, independent results, and honest timing/evaluation counts |
| Package portability | Built a wheel from a temporary source-only copy; installed it to a temporary target; imported the installed package and ran an initialization-only engine check from outside the project directory |

The package check used the existing NumPy installation, disabled dependency
downloads, and removed its temporary staging directory. It verifies packaging
and imports, rather than a fresh dependency download on another computer.

The default-size run and integration cases use **test-only operator doubles**.
They establish engine correctness and stability; they are not experimental
results or evidence about the group's crossover/mutation quality. No runtime
performance threshold is imposed by these tests.

## Failure behavior clarified

- Invalid evaluation data now raises a message such as
  `Evaluation at generation 1: Cannot evaluate population row 0: ...`.
- The row is relative to the evaluated batch: the initial population for
  generation 0, or new offspring for later generations. The cached elite is
  not part of that offspring batch.
- Existing operator validation errors retain generation/offspring context.
- Other callback programming errors propagate normally; the engine does not
  hide them, retry, or return a misleading successful result.
- A failed run leaves no shared engine RNG state that changes a later run.

## Remaining work

Step 8 is final core documentation and handoff preparation. Integration with
the group's actual operators still requires agreement with the other members
and another end-to-end check after their implementations arrive. Schema
analysis, experiment batches, statistics, plots, GUI, and submission artifacts
remain separate work.

All changes and verification were local. No repository remote, commits, or
pushes were created during this review.
