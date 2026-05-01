import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # VK токены — берутся из переменных окружения на Render
    vk_group_token: str
    vk_user_token: str
    vk_confirmation_token: str
    vk_group_id: int
    vk_admin_user_id: int
    vk_api_version: str = "5.199"

    # База данных PostgreSQL — строка подключения из Render
    database_url: str

    # ML пороги
    spam_threshold: float = 0.7
    negative_threshold: float = 0.65
    ml_model_path: str = "rf_model.joblib"

    # Сервер
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", 8000))
    debug: bool = False  # На проде всегда False

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
