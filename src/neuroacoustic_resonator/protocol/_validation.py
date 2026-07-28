from math import isfinite


def require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)


def require_unit_interval(name: str, value: float) -> None:
    require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], got {value!r}"
        raise ValueError(msg)


def require_non_negative(name: str, value: float) -> None:
    require_finite(name, value)
    if value < 0.0:
        msg = f"{name} must be non-negative, got {value!r}"
        raise ValueError(msg)


def require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        msg = f"{name} must be non-negative, got {value!r}"
        raise ValueError(msg)


def require_positive_int(name: str, value: int) -> None:
    if value < 1:
        msg = f"{name} must be positive, got {value!r}"
        raise ValueError(msg)


def require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        msg = f"{name} must not be empty"
        raise ValueError(msg)


def require_optional_non_empty(name: str, value: str | None) -> None:
    if value is not None:
        require_non_empty(name, value)


def require_not_greater(name: str, value: float, upper_name: str, upper: float) -> None:
    if value > upper:
        msg = f"{name} must not exceed {upper_name}"
        raise ValueError(msg)
