"""
Configuration management for FinBERT API
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # Hugging Face Configuration
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    
    # Finnhub Configuration
    finnhub_api_key: Optional[str] = Field(default=None, env="FINNHUB_API_KEY")
    
    # Model Configuration
    model_name: str = Field(default="ProsusAI/finbert", env="MODEL_NAME")
    max_sequence_length: int = Field(default=512, env="MAX_SEQUENCE_LENGTH")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    
    # Cache Configuration
    transformers_cache: Optional[str] = Field(default=None, env="TRANSFORMERS_CACHE")
    hf_home: Optional[str] = Field(default=None, env="HF_HOME")
    
    # Logging Configuration
    log_level: str = Field(default="info", env="LOG_LEVEL")
    log_format: str = Field(default="standard", env="LOG_FORMAT")
    
    # CORS Configuration
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    allowed_methods: str = Field(default="GET,POST", env="ALLOWED_METHODS")
    allowed_headers: str = Field(default="*", env="ALLOWED_HEADERS")
    
    # Performance Configuration
    pytorch_cuda_alloc_conf: Optional[str] = Field(default=None, env="PYTORCH_CUDA_ALLOC_CONF")
    
    # Health Check Configuration
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT")
    
    # Security Configuration
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    rate_limit: Optional[int] = Field(default=None, env="RATE_LIMIT")
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from string"""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def cors_methods(self) -> List[str]:
        """Parse CORS methods from string"""
        return [method.strip() for method in self.allowed_methods.split(",")]
    
    @property
    def cors_headers(self) -> List[str]:
        """Parse CORS headers from string"""
        if self.allowed_headers == "*":
            return ["*"]
        return [header.strip() for header in self.allowed_headers.split(",")]
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",)
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Set environment variables for ML libraries
settings = get_settings()

if settings.transformers_cache:
    os.environ["TRANSFORMERS_CACHE"] = settings.transformers_cache

if settings.hf_home:
    os.environ["HF_HOME"] = settings.hf_home

if settings.pytorch_cuda_alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.pytorch_cuda_alloc_conf