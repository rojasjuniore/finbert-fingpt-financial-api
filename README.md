# FinBERT API - Financial Sentiment Analysis

A high-performance REST API for financial sentiment analysis using the FinBERT model from Hugging Face. This API provides fast, accurate sentiment classification for financial text with support for batch processing and real-time analysis.

## Features

- **Financial Sentiment Analysis**: Uses the pre-trained FinBERT model for domain-specific sentiment analysis
- **Batch Processing**: Efficient processing of multiple texts with configurable batch sizes
- **Real-time Analysis**: Fast inference for single text analysis
- **Probability Scores**: Optional probability scores for all sentiment classes
- **Health Monitoring**: Comprehensive health checks and performance metrics
- **Docker Support**: Ready-to-deploy Docker configuration
- **Production Ready**: Proper logging, error handling, and monitoring
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## Quick Start

### Using Docker (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd FINBERT
   cp .env.example .env
   # Edit .env with your Hugging Face token if needed
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Test the API**:
   ```bash
   curl http://localhost:8000/health
   ```

### Local Development

1. **Setup environment**:
   ```bash
   ./start.sh
   ```

2. **Or manual setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   cp .env.example .env
   python main.py
   ```

## API Usage

### Base URL
- Local development: `http://localhost:8000`
- API endpoints: `http://localhost:8000/api/v1`

### Authentication
Currently no authentication required. For production, configure `API_KEY` in environment variables.

### Endpoints

#### 1. Analyze Single/Multiple Texts
```http
POST /api/v1/analyze
```

**Request**:
```json
{
  "text": "The company reported strong quarterly earnings with significant growth.",
  "return_probabilities": false,
  "batch_size": 32
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "text": "The company reported strong quarterly earnings with significant growth.",
    "sentiment": "positive",
    "confidence": 0.89,
    "processing_time": 0.12
  },
  "metadata": {
    "model_name": "ProsusAI/finbert",
    "batch_size_used": 32,
    "return_probabilities": false
  },
  "total_processing_time": 0.12
}
```

#### 2. Bulk Analysis
```http
POST /api/v1/analyze/bulk
```

**Request**:
```json
{
  "texts": [
    "Strong quarterly results exceeded expectations.",
    "Market volatility caused concerns among investors.",
    "The company maintains a stable outlook."
  ],
  "return_probabilities": true,
  "batch_size": 32
}
```

#### 3. Health Check
```http
GET /api/v1/health?deep_check=true
```

#### 4. Model Information
```http
GET /api/v1/model/info
```

### Example with Python

```python
import requests

# Single text analysis
response = requests.post('http://localhost:8000/api/v1/analyze', json={
    'text': 'The stock market showed strong performance today.',
    'return_probabilities': True
})

result = response.json()
print(f"Sentiment: {result['data']['sentiment']}")
print(f"Confidence: {result['data']['confidence']:.2f}")

# Multiple texts
response = requests.post('http://localhost:8000/api/v1/analyze', json={
    'text': [
        'Earnings exceeded expectations significantly.',
        'The market crash led to substantial losses.',
        'Trading volume remained stable today.'
    ]
})

results = response.json()
for item in results['data']:
    print(f"'{item['text'][:50]}...' -> {item['sentiment']} ({item['confidence']:.2f})")
```

### Example with cURL

```bash
# Single text analysis
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The company announced record profits this quarter.",
    "return_probabilities": true
  }'

# Health check
curl http://localhost:8000/api/v1/health

# Model info
curl http://localhost:8000/api/v1/model/info
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model Configuration
MODEL_NAME=ProsusAI/finbert
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=32

# Hugging Face Token (for private models)
HF_TOKEN=your_token_here

# Performance
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Security (production)
API_KEY=your-secret-key
RATE_LIMIT=100
```

### Model Options

The API supports any FinBERT-compatible model from Hugging Face:

- `ProsusAI/finbert` (default) - Original FinBERT
- `yiyanghkust/finbert-tone` - Alternative FinBERT variant
- Custom fine-tuned models

## Deployment

### Docker Production Deployment

1. **Build and deploy**:
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

2. **With Nginx proxy**:
   ```bash
   docker-compose --profile nginx up -d
   ```

3. **Scale workers**:
   ```bash
   docker-compose up -d --scale finbert-api=3
   ```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finbert-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: finbert-api
  template:
    metadata:
      labels:
        app: finbert-api
    spec:
      containers:
      - name: finbert-api
        image: finbert-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Performance

### Benchmarks

- **Single text**: ~50-100ms per request
- **Batch processing**: ~2-5ms per text (batch of 32)
- **Memory usage**: ~2-4GB (model loaded)
- **CPU usage**: Moderate (GPU recommended for high throughput)

### Optimization Tips

1. **Use batch processing** for multiple texts
2. **Enable GPU** if available (`CUDA_VISIBLE_DEVICES=0`)
3. **Adjust batch size** based on available memory
4. **Use multiple workers** for concurrent requests
5. **Cache responses** for repeated texts

## Monitoring

### Health Checks

- Basic: `GET /health`
- Detailed: `GET /api/v1/health?deep_check=true`
- Metrics: Check response headers for processing times

### Logging

Structured JSON logging in production:

```bash
# View logs
docker-compose logs -f finbert-api

# Log format
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "uuid-here",
  "status_code": 200,
  "process_time": 0.123
}
```

## Testing

Run the test suite:

```bash
# All tests
python -m pytest

# Unit tests only
python -m pytest -m unit

# Integration tests
python -m pytest -m integration

# With coverage
python -m pytest --cov=app tests/
```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Error Handling

The API returns structured error responses:

```json
{
  "success": false,
  "error": "ValidationError",
  "message": "Request validation failed",
  "details": {
    "validation_errors": "...",
    "request_id": "uuid-here"
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation error)
- `422`: Unprocessable Entity (validation error)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues, questions, or contributions:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review existing issues
3. Create a new issue with detailed information

## Changelog

### v1.0.0
- Initial release
- FinBERT model integration
- REST API endpoints
- Docker support
- Comprehensive testing
- Production-ready features