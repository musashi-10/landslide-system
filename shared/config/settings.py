from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "landslide-system"
    environment: str = "development"
    log_level: str = "INFO"

    # Backend
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    postgres_db: str = "landslide"
    postgres_user: str = "landslide_user"
    postgres_password: str = "change_me"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Weather / rainfall
    weather_api_key: str = ""
    weather_api_base_url: str = ""

    # Satellite / Earth Engine
    google_cloud_project: str = ""
    earth_engine_project: str = ""

    # ML
    model_path: str = ""
    model_version: str = "v1"

    # Alerts
    alert_enabled: bool = False
    alert_provider: str = ""
    alert_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
