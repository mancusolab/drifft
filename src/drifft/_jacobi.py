"""Compatibility imports for existing private Jacobi research scripts.

The implementation lives in :mod:`drifft.backends.jacobi`.
"""

# pattern: Functional Core
from .backends.jacobi import (
    Jacobi as Jacobi,
)
from .backends.jacobi import (
    _assemble as _assemble,
)
from .backends.jacobi import (
    _choose_rank as _choose_rank,
)
from .backends.jacobi import (
    _JacobiGeometry as _JacobiGeometry,
)
from .backends.jacobi import (
    _padded_rank as _padded_rank,
)
