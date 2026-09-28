"""`kurt` of whole numbers far from zero, against exact arithmetic.

Near two to the sixty two neighbouring floats are 1024 apart, so a column cast
to float before its moments are taken has moved each value by up to 512. A
kurtosis does not change when a constant is added, so the column is measured
from its least value as whole numbers first, and the answer is then right to
the last few bits.
"""

from __future__ import annotations

from fractions import Fraction
from types import ModuleType

import pytest


def exact(values: list[int]) -> float:
    """The unbiased excess kurtosis in rational arithmetic."""
    n = len(values)
    mean = Fraction(sum(values), n)
    m2 = sum((v - mean) ** 2 for v in values)
    m4 = sum((v - mean) ** 4 for v in values)
    ratio = Fraction(n * (n + 1) * (n - 1)) * m4 / ((n - 2) * (n - 3) * m2 * m2)
    return float(ratio - Fraction(3 * (n - 1) ** 2, (n - 2) * (n - 3)))


BASE = -(2**62)
SPREADS = [0, 4_000_000_000, 17, 3_999_999_999, 123_456_789, 2_000_000_000, 7, 3_000_000_001]


@pytest.mark.parametrize("gap", [False, True])
def test_kurt_far_from_zero_is_exact(firepanda: ModuleType, gap: bool) -> None:
    """Right to within a few bits of the exact answer, with or without a gap."""
    values = [BASE + spread for spread in SPREADS]
    column = firepanda.Series([*values, None] if gap else values)
    assert column.kurt() == pytest.approx(exact(values), rel=1e-12)


def test_kurt_of_a_column_spanning_every_whole_number_still_answers(
    firepanda: ModuleType,
) -> None:
    """A column too wide to shift is cast as it is, as before."""
    column = firepanda.Series([-(2**63), 2**63 - 1, 0, 5])
    assert column.kurt() == pytest.approx(1.5)
