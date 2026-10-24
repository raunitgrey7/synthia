"""Synthia: privacy-preserving synthetic tabular data.

The package is organised around one contract: a ``TableSchema`` describes the columns, a generator learns a
table and samples new rows, the ``privacy`` package supplies the differential-privacy mechanisms and the
accountant, and the ``evaluation`` package measures fidelity, utility and privacy with bootstrap confidence
intervals. Everything is deterministic for a fixed seed and runs on a CPU with no network access.
"""

from synthia.schema import ColumnKind, ColumnSpec, TableSchema

__version__ = "0.1.0"

__all__ = ["ColumnKind", "ColumnSpec", "TableSchema", "__version__"]
