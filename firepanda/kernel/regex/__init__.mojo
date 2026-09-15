"""The two regular expression engines pandas answers out of.

Document 76 is the argument for why there are two rather than one. The short
version is that pandas parses every pattern with Python's own parser, routes a
pattern holding a lookaround or a backreference to Python's `re` and everything
else to Arrow's RE2, and the two engines disagree about what `\\d` means, about
what `$` matches and about several other things that are visible in answers
rather than only in refusals.

This package is built front to back. `parse.mojo` is the parser, which is
Python's grammar because it is Python's parser making the routing decision
upstream. `route.mojo` is that decision.
"""
