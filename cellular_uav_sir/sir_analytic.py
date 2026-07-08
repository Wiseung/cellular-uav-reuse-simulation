from __future__ import annotations

import math


def sir_linear(reuse_factor: int, pathloss_exponent: float) -> float:
    return (3.0 * reuse_factor) ** (pathloss_exponent / 2.0) / 6.0


def sir_db(reuse_factor: int, pathloss_exponent: float) -> float:
    return 10.0 * math.log10(sir_linear(reuse_factor, pathloss_exponent))


def area_spectral_efficiency(
    sir_linear_value: float,
    reuse_factor: int,
) -> float:
    return math.log2(1.0 + sir_linear_value) / reuse_factor
