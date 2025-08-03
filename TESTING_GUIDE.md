# FinBERT API Testing Guide

## Overview
This guide provides comprehensive instructions for running tests, understanding test results, and maintaining the test suite for the FinBERT API.

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (for integration tests)
docker run -d -p 6379:6379 redis:alpine

# Set environment variables
export ENVIRONMENT=test
export REDIS_URL=redis://localhost:6379/15
```

### Running Tests

#### All Tests
```bash
# Run complete test suite
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run in parallel (faster)
pytest -n auto
```

#### By Test Type
```bash
# Unit tests only (fast)
pytest tests/unit/ -m unit

# Integration tests only
pytest tests/integration/ -m integration

# End-to-end tests only
pytest tests/e2e/ -m e2e

# Security tests only
pytest tests/integration/test_security.py -m security
```

#### By Performance
```bash
# Exclude slow tests
pytest -m "not slow"

# Run only benchmark tests
pytest -m benchmark

# Run load tests (requires running API)
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## Test Structure

### Directory Organization
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_model_service.py
│   └── test_cache_service.py
├── integration/             # Integration tests (slower)
│   ├── test_api_endpoints.py
│   └── test_security.py
├── e2e/                     # End-to-end tests (slowest)
│   └── test_full_workflow.py
├── load/                    # Load testing
│   └── locustfile.py
├── fixtures/                # Test data
│   └── financial_texts.py
└── mocks/                   # Mock objects and responses
```

### Test Categories

#### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: Very fast (< 1s per test)
- **Scope**: Single function/method
- **Dependencies**: Mocked
- **Examples**: Model inference logic, cache operations, data validation

#### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions
- **Speed**: Moderate (1-10s per test)
- **Scope**: Multiple components together
- **Dependencies**: Real external services (Redis, etc.)
- **Examples**: API endpoints, database operations, external API calls

#### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Speed**: Slow (10-60s per test)
- **Scope**: Full system
- **Dependencies**: All real services
- **Examples**: Complete sentiment analysis workflow, error handling

#### Load Tests (`tests/load/`)
- **Purpose**: Test performance and scalability
- **Speed**: Very slow (minutes)
- **Scope**: System under load
- **Dependencies**: Running API instance
- **Examples**: Concurrent requests, throughput testing

## Test Configuration

### Environment Variables
```bash
# Required for all tests
ENVIRONMENT=test
API_HOST=127.0.0.1
API_PORT=8001
LOG_LEVEL=warning

# Required for integration tests
REDIS_URL=redis://localhost:6379/15

# Required for load tests
MODEL_NAME=ProsusAI/finbert
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=1
```

### Pytest Configuration (`pytest.ini`)
- **Coverage**: Minimum 80% required
- **Markers**: Organized by test type and characteristics
- **Timeouts**: 5-minute default timeout
- **Logging**: Detailed logs for debugging

### Fixtures (`conftest.py`)
- **Mock Models**: Avoid downloading large models
- **Test Data**: Realistic financial text samples
- **Redis**: Clean test database
- **HTTP Client**: Async test client for API testing

## Running Specific Test Scenarios

### Development Workflow
```bash
# Quick feedback loop (unit tests only)
pytest tests/unit/ -x -v

# Pre-commit checks
pytest tests/unit/ tests/integration/ -m "not slow"

# Full validation
pytest --cov=src --cov-report=term-missing
```

### CI/CD Pipeline
```bash
# Stage 1: Fast tests
pytest tests/unit/ -m unit --junitxml=unit-results.xml

# Stage 2: Integration tests
pytest tests/integration/ -m integration --junitxml=integration-results.xml

# Stage 3: Security tests
pytest tests/integration/test_security.py -m security

# Stage 4: Performance benchmarks
pytest -m benchmark --benchmark-json=benchmark-results.json
```

### Production Validation
```bash
# Health checks
pytest tests/e2e/test_full_workflow.py::TestFullWorkflow::test_health_monitoring_workflow

# Load testing
locust -f tests/load/locustfile.py FinBERTAPIUser --users=50 --spawn-rate=10 --run-time=5m
```

## Test Data and Fixtures

### Financial Text Samples
Located in `tests/fixtures/financial_texts.py`:
- **Positive Sentiment**: 25+ samples of bullish financial news
- **Negative Sentiment**: 25+ samples of bearish financial news  
- **Neutral Sentiment**: 25+ samples of factual announcements
- **Edge Cases**: Special characters, unicode, very long/short texts
- **Sector-Specific**: Technology, healthcare, finance, energy, retail

### Expected Results
- **Sentiment Mappings**: Known text → sentiment label pairs
- **Performance Benchmarks**: Response time and throughput targets
- **Security Test Data**: Injection attempts, malformed requests

### Mock Objects
- **Model Responses**: Simulated FinBERT inference results
- **API Responses**: Standard response formats
- **Error Conditions**: Various failure scenarios

## Performance Testing

### Locust Load Testing

#### Basic Load Test
```bash
# Start API server
uvicorn src.main:app --host=0.0.0.0 --port=8000

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

#### Load Test Scenarios
```bash
# Normal load (10 users, 5 minutes)
locust -f tests/load/locustfile.py FinBERTAPIUser --users=10 --spawn-rate=2 --run-time=5m --host=http://localhost:8000

# Stress test (100 users, 3 minutes)
locust -f tests/load/locustfile.py HighLoadUser --users=100 --spawn-rate=20 --run-time=3m --host=http://localhost:8000

# Error handling test (20 users, 3 minutes)
locust -f tests/load/locustfile.py ErrorTestUser --users=20 --spawn-rate=5 --run-time=3m --host=http://localhost:8000
```

#### Performance Benchmarks
- **Response Time**: < 500ms (95th percentile)
- **Throughput**: > 100 requests/second
- **Error Rate**: < 0.1% under normal load
- **Memory Usage**: < 2GB per worker

### Benchmark Tests
```bash
# Run performance benchmarks
pytest -m benchmark --benchmark-sort=mean

# Generate benchmark report
pytest -m benchmark --benchmark-json=benchmark.json
pytest-benchmark compare benchmark.json
```

## Security Testing

### Automated Security Tests
```bash
# Run all security tests
pytest tests/integration/test_security.py -m security -v

# Specific security categories
pytest tests/integration/test_security.py::TestSecurity::test_input_validation_sql_injection
pytest tests/integration/test_security.py::TestSecurity::test_denial_of_service_protection
```

### Security Test Categories
- **Input Validation**: SQL injection, XSS, path traversal
- **DoS Protection**: Rate limiting, request size limits
- **Error Handling**: Information disclosure prevention
- **Headers**: Security header validation
- **Authentication**: Bypass attempt detection

### Manual Security Testing
```bash
# Test with security scanner
bandit -r src/

# Dependency vulnerability check
safety check

# Static code analysis
flake8 src/ --select=E9,F63,F7,F82
```

## Test Maintenance

### Adding New Tests

#### Unit Test Template
```python
class TestNewFeature:
    @pytest.fixture
    def feature_service(self):
        return NewFeatureService()
    
    def test_feature_functionality(self, feature_service):
        # Arrange
        input_data = "test input"
        
        # Act
        result = feature_service.process(input_data)
        
        # Assert
        assert result is not None
        assert result.status == "success"
```

#### Integration Test Template
```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_new_endpoint(test_client):
    payload = {"data": "test"}
    
    response = await test_client.post("/new-endpoint", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
```

### Updating Test Data
1. **Add new samples** to `tests/fixtures/financial_texts.py`
2. **Update expected results** in fixture mappings
3. **Regenerate fixture files** if needed
4. **Update documentation** for new test scenarios

### Test Performance Optimization
- **Parallel Execution**: Use `pytest-xdist` for faster test runs
- **Mock Heavy Operations**: Mock model loading and inference
- **Efficient Fixtures**: Use session-scoped fixtures for expensive setup
- **Selective Testing**: Use markers to run relevant test subsets

## Troubleshooting

### Common Issues

#### Tests Timeout
```bash
# Increase timeout
pytest --timeout=600

# Check for hanging processes
ps aux | grep pytest
```

#### Redis Connection Failed
```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Check Redis connectivity
redis-cli ping
```

#### Mock Model Loading Issues
```bash
# Clear transformers cache
rm -rf ~/.cache/huggingface/transformers/

# Check mock configuration in conftest.py
pytest tests/unit/test_model_service.py -v -s
```

#### Memory Issues During Testing
```bash
# Run tests in smaller batches
pytest tests/unit/ --maxfail=1

# Monitor memory usage
pytest --memray tests/unit/
```

### Debug Mode
```bash
# Verbose output
pytest -v -s

# Drop into debugger on failure
pytest --pdb

# Print debug information
pytest --capture=no --log-cli-level=DEBUG
```

### Test Coverage Issues
```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html --cov-branch

# View coverage report
open htmlcov/index.html

# Find untested code
pytest --cov=src --cov-report=term-missing
```

## Best Practices

### Writing Good Tests
1. **Descriptive Names**: Test names should explain what is being tested
2. **Arrange-Act-Assert**: Clear test structure
3. **Independent Tests**: Tests should not depend on each other
4. **Realistic Data**: Use representative test data
5. **Edge Cases**: Test boundary conditions and error cases

### Test Organization
1. **Group Related Tests**: Use test classes for related functionality
2. **Use Fixtures**: Share setup code with fixtures
3. **Mark Tests**: Use pytest markers for test categorization
4. **Document Complex Tests**: Add docstrings for complex test logic

### Performance Considerations
1. **Mock External Dependencies**: Avoid real API calls in unit tests
2. **Use Fast Fixtures**: Prefer function-scoped over session-scoped when possible
3. **Parallel Execution**: Design tests to run in parallel
4. **Resource Cleanup**: Properly clean up resources after tests

### Security Testing
1. **Input Validation**: Test all input validation thoroughly
2. **Error Messages**: Ensure error messages don't leak sensitive information
3. **Authentication**: Test authentication and authorization thoroughly
4. **Rate Limiting**: Verify rate limiting works correctly

## Continuous Integration

### GitHub Actions Example
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:alpine
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run unit tests
      run: pytest tests/unit/ -m unit --junitxml=unit-results.xml
    
    - name: Run integration tests
      run: pytest tests/integration/ -m integration --junitxml=integration-results.xml
      env:
        REDIS_URL: redis://localhost:6379/15
    
    - name: Generate coverage report
      run: pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
```

### Quality Gates
- **Unit Test Coverage**: > 80%
- **Integration Test Coverage**: > 70%
- **Performance Benchmarks**: Meet or exceed targets
- **Security Tests**: All pass
- **Code Quality**: Pass linting and static analysis

## Metrics and Reporting

### Test Metrics
- **Test Count**: Track number of tests over time
- **Coverage**: Monitor code coverage trends
- **Performance**: Track test execution time
- **Flakiness**: Identify and fix flaky tests

### Performance Metrics
- **Response Time**: Track API response times
- **Throughput**: Monitor requests per second
- **Error Rate**: Track error rates under load
- **Resource Usage**: Monitor CPU and memory usage

### Reporting Tools
- **Coverage**: htmlcov reports, Codecov integration
- **Performance**: pytest-benchmark reports
- **Load Testing**: Locust web UI and reports
- **Security**: Bandit security reports

This comprehensive testing guide ensures the FinBERT API maintains high quality, performance, and security standards throughout its development lifecycle.