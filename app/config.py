import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model_quantized.onnx")
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))
    MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
    ALLOWED_CONTENT_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"]

    class Config:
        env_file = ".env"


settings = Settings()
