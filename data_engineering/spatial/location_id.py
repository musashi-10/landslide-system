"""
location_id generation utilities.

``location_id`` is the primary key used across all modules to link data
from different engineers.  It must be deterministic and reproducible
given the same inputs.

Format::

    LOC_{lat_str}_{lon_str}

where lat/lon are formatted to 4 decimal places with sign, e.g.::

    LOC_27.1230_88.4560
    LOC_-34.5678_-58.1234

This format was chosen because it is:
  * Human-readable and debuggable
  * Stable across runs (deterministic)
  * Unique at ~11 m spatial precision (4 d.p.)
  * Compatible with the shared contract ``location_id`` field
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LAT_FMT = "{:+.4f}"
_LON_FMT = "{:+.4f}"


def generate_location_id(latitude: float, longitude: float) -> str:
    """
    Return a deterministic ``location_id`` string for a coordinate pair.

    Parameters
    ----------
    latitude : float
        WGS-84 latitude in decimal degrees.
    longitude : float
        WGS-84 longitude in decimal degrees.

    Returns
    -------
    str
        Location identifier, e.g. ``"LOC_+27.1230_+088.4560"``.

    Examples
    --------
    >>> generate_location_id(27.123, 88.456)
    'LOC_+27.1230_+088.4560'
    """
    lat_str = _LAT_FMT.format(latitude)
    lon_str = "{:+08.4f}".format(longitude)  # zero-pad to maintain sort order
    return f"LOC_{lat_str}_{lon_str}"
