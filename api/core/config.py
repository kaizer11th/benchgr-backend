from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://benchgr:benchgr_dev@localhost:5432/benchgr"
    REDIS_URL: str = "redis://localhost:6379"
    SECRET_KEY: str = "dev_secret_change_in_production"
    ENVIRONMENT: str = "development"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
