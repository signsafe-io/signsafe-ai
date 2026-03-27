from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"

    database_url: str = "postgresql://postgres:postgres@signsafe-db:5432/signsafe"
    rabbitmq_url: str = "amqp://guest:guest@signsafe-queue:5672"

    s3_endpoint: str = "http://signsafe-store:8333"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "contracts"

    qdrant_host: str = "signsafe-vectors"
    qdrant_port: int = 6333

    anthropic_api_key: str = ""
    openai_api_key: str = ""


settings = Settings()
