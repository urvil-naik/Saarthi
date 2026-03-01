# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL:            str
    MODEL_DIR:               str = "./models"
    PERSONAL_MODEL_MIN_ROWS: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",          # look for .env in working directory
        env_file_encoding="utf-8",
        extra="ignore",           # silently ignore unknown env vars
    )


settings = Settings()
