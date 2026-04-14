from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"

    database_url: str = "postgresql://postgres@signsafe-db:5432/signsafe_db"
    rabbitmq_url: str = "amqp://guest:guest@signsafe-queue:5672"

    qdrant_url: str = "http://signsafe-vector:6333"

    s3_endpoint: str = "http://signsafe-store:8333"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "contracts"

    openai_api_key: str = ""
    law_api_oc: str = ""


settings = Settings()
