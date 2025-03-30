from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_username: str
    mongo_password: str
    mongo_host: str
    mongo_db: str
    mongo_coll: str
    aws_server_public_key: str
    aws_server_secret_key: str
    temp_dir: str
    train_dir: str
    test_dir: str
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


app_settings = Settings()
