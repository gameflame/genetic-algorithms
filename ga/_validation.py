"""Small shared validators for representations and initialization."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contracts import Chromosome, Population


def require_integer(
    value: int,
    *,
    name: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Accept Python/NumPy integers without treating booleans as sizes."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        limit = (
            f" between {minimum} and {maximum}"
            if maximum is not None else f" at least {minimum}"
        )
        raise ValueError(f"{name} must be{limit}.")
    return result


def real_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert real numeric input to finite float64 values without mutation."""
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite float64 values.")
    return result


def binary_vector(value: ArrayLike, *, length: int) -> Chromosome:
    """Validate before uint8 conversion so invalid values cannot wrap to bits."""
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chromosome must be a one-dimensional binary vector.") from exc
    if values.ndim != 1 or values.size != length:
        raise ValueError(f"Chromosome must be one-dimensional with length {length}.")
    if values.dtype.kind not in "biuf" or not np.all((values == 0) | (values == 1)):
        raise ValueError("Chromosome must contain only binary values 0 or 1.")
    return values.astype(np.uint8, copy=True)


def binary_population(
    value: ArrayLike, *, chromosome_length: int | None = None, minimum_size: int = 1
) -> Population:
    """Validate a binary matrix; callers copy rows before exposing them to callbacks."""
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Population must be a two-dimensional binary matrix.") from exc
    if values.ndim != 2 or values.shape[0] < minimum_size or values.shape[1] < 1:
        raise ValueError(
            f"Population must be two-dimensional with at least {minimum_size} rows "
            "and a positive chromosome length."
        )
    if chromosome_length is not None and values.shape[1] != chromosome_length:
        raise ValueError(f"Population chromosomes must have length {chromosome_length}.")
    if values.dtype.kind not in "biuf" or not np.all((values == 0) | (values == 1)):
        raise ValueError("Population must contain only binary values 0 or 1.")
    return values.astype(np.uint8, copy=False)
