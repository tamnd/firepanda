"""The floor for the compile budget.

This program links the package and touches nothing that is generic over a dtype,
so its compile time and binary size are what the Mojo runtime and the standard
library cost before firepanda's monomorphization adds anything. Every other probe
is reported as a delta against this one.
"""

from firepanda.version import version


def main():
    print(version())
