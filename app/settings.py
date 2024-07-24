from functools import lru_cache
import logging
import pathlib

from pydantic_settings import BaseSettings  # pylint: disable=no-name-in-module

LOG = logging.getLogger(__name__)

class ApplicationSettings(BaseSettings):

    MIDDLELAYER_PATH: str = "http://0.0.0.0:5050"
    PINECONE_API_KEY: str
    WHISPER_PATH: str
    OPENAI_API_KEY: str
    class Config:
        case_sensitive = True
        frozen = True

    @staticmethod
    def create():
        env_file = pathlib.Path(f"app.env")
        if not env_file.exists():
            raise ValueError(f"{env_file} does not exist.")

        LOG.info(f"CONSTRUCTING: Settings using {env_file}")
        app_settings = ApplicationSettings(_env_file=env_file)
        return app_settings


@lru_cache()
def inject_settings() -> ApplicationSettings:
    return ApplicationSettings.create()
