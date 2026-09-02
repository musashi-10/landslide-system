"""
CSV loader for landslide and environmental datasets.

Reads a CSV file into a pandas DataFrame while:
  * validating that required columns are present
  * preserving the original source name in a ``source`` column
  * not inferring types that would silently coerce bad values
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class IngestionError(ValueError):
    """Raised when an input file cannot be ingested."""


def load_csv(
    path: str | Path,
    *,
    required_columns: Optional[list[str]] = None,
    source_name: Optional[str] = None,
    encoding: str = "utf-8",
    **read_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    required_columns : list[str] | None
        Column names that must be present.  Raises :class:`IngestionError`
        if any are missing.
    source_name : str | None
        Human-readable source identifier stored in the ``source`` column.
        Defaults to the file name.
    encoding : str
        File encoding (default ``"utf-8"``).
    **read_kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a ``source`` column populated.

    Raises
    ------
    IngestionError
        When the file is missing, empty, or required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")

    logger.info("Loading CSV: %s", path)

    try:
        df = pd.read_csv(path, encoding=encoding, **read_kwargs)
    except Exception as exc:
        raise IngestionError(f"Failed to read CSV {path}: {exc}") from exc

    if df.empty:
        raise IngestionError(f"CSV file is empty: {path}")

    # Column validation
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise IngestionError(
                f"CSV {path.name} is missing required columns: {missing}"
            )

    # Source tracking
    df["source"] = source_name or path.name

    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df
