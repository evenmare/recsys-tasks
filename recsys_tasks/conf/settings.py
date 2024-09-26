from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ReadFileSettings(BaseSettings):
    """Общая конфигурация для чтения файлов."""

    skip_header: bool = False
    delimiter: Literal[',', ';', ' '] = ';'


class Settings(BaseSettings):
    """Общая конфигурация для приложения."""

    BASE_DIR: Path = Path(__file__).cwd().parent
    INPUT_DATA_DIR: Path = BASE_DIR / 'input_data'


@lru_cache
def get_settings() -> Settings:
    """Получение настроек для запуска решения задач."""
    return Settings()


@lru_cache
def get_read_file_settings() -> ReadFileSettings:
    """Получение настроек для чтения из файла."""
    return ReadFileSettings()
