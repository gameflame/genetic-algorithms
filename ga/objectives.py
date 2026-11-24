"""Benchmark objective functions and a small discrete minimization problem.

Each function evaluates one candidate and returns a finite Python float.
Inputs may be one-dimensional real numeric arrays or sequences. Search bounds
belong to the representation/decoder; these functions do not enforce bounds.
"""

import numpy as np
from numpy.typing import ArrayLike

from .contracts import DecodedCandidate


def _coordinates(
    candidate: ArrayLike,
    *,
    name: str,
    minimum_dimensions: int = 1,
    exact_dimensions: int | None = None,
) -> DecodedCandidate:
    """Validate one candidate without modifying or flattening its input."""
    try:
        values = np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requires a real numeric vector.") from exc

    if values.ndim != 1:
        raise ValueError(f"{name} requires a one-dimensional coordinate vector.")
    if exact_dimensions is not None and values.size != exact_dimensions:
        raise ValueError(f"{name} requires exactly {exact_dimensions} coordinates.")
    if values.size < minimum_dimensions:
        raise ValueError(f"{name} requires at least {minimum_dimensions} coordinates.")
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{name} coordinates must be real numeric values.")

    # Conversion can overflow for a finite value outside float64's range.
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} coordinates must be finite float64 values.")
    return coordinates


def _finite_cost(cost: np.float64 | float, *, name: str) -> float:
    """Fail clearly if finite coordinates overflow an objective calculation."""
    result = float(cost)
    if not np.isfinite(result):
        raise ValueError(f"{name} cost is not finite; coordinate magnitudes are too large.")
    return result


def sphere(candidate: ArrayLike) -> float:
    """Return sum(x_i**2) for one or more coordinates.

    De Jong's Sphere has its minimum cost of zero at the origin.
    """
    coordinates = _coordinates(candidate, name="Sphere")
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.sum(coordinates**2)
    return _finite_cost(cost, name="Sphere")


def rosenbrock(candidate: ArrayLike) -> float:
    """Return sum(100*(x_(i+1) - x_i**2)**2 + (1 - x_i)**2).

    Requires at least two coordinates. The minimum cost is zero when every
    coordinate equals one.
    """
    coordinates = _coordinates(candidate, name="Rosenbrock", minimum_dimensions=2)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.sum(
            100.0 * (coordinates[1:] - coordinates[:-1] ** 2) ** 2
            + (1.0 - coordinates[:-1]) ** 2
        )
    return _finite_cost(cost, name="Rosenbrock")


def himmelblau(candidate: ArrayLike) -> float:
    """Return (x**2 + y - 11)**2 + (x + y**2 - 7)**2.

    Requires exactly two coordinates. There are four global minima with cost
    zero, including the exact point (3, 2).
    """
    coordinates = _coordinates(candidate, name="Himmelblau", exact_dimensions=2)
    x, y = coordinates
    with np.errstate(over="ignore", invalid="ignore"):
        cost = (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2
    return _finite_cost(cost, name="Himmelblau")


def zero_count(candidate: ArrayLike) -> float:
    """Count zero bits in a non-empty binary vector, as a minimization cost.

    Accepts integer, floating-point, or boolean binary values. Any all-ones
    vector has the minimum cost of zero. This is the project's small discrete
    demonstration, separate from the three continuous benchmark functions.
    """
    coordinates = _coordinates(candidate, name="Zero-count")
    if not np.all((coordinates == 0.0) | (coordinates == 1.0)):
        raise ValueError("Zero-count coordinates must contain only binary values 0 or 1.")
    return float(np.count_nonzero(coordinates == 0.0))
