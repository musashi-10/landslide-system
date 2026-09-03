from .base import (
    WeatherProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderRateLimitError,
    ProviderInvalidResponseError,
    ProviderMissingDataError,
)
from .mock_provider import MockWeatherProvider
from .open_meteo_provider import OpenMeteoProvider

__all__ = [
    "WeatherProvider",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderRateLimitError",
    "ProviderInvalidResponseError",
    "ProviderMissingDataError",
    "MockWeatherProvider",
    "OpenMeteoProvider",
]


def get_provider(name: str, **kwargs) -> WeatherProvider:
    """
    Factory function: return a provider instance by name.

    Supported names
    ---------------
    'mock'       -> MockWeatherProvider
    'open_meteo' -> OpenMeteoProvider

    Parameters
    ----------
    name : str
        Provider identifier.
    **kwargs
        Passed to the provider constructor.
    """
    name = name.lower().strip()
    if name == "mock":
        return MockWeatherProvider(**kwargs)
    if name in ("open_meteo", "openmeteo"):
        return OpenMeteoProvider(**kwargs)
    raise ValueError(
        f"Unknown provider {name!r}. Supported: 'mock', 'open_meteo'."
    )
