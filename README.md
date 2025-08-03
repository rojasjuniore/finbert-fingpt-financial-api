# 🚀 FinBERT + FinGPT Financial LLM API

A comprehensive, production-ready REST API that combines **FinBERT** for sentiment analysis and **FinGPT** for financial text generation and analysis. Built with FastAPI, Docker, and optimized for high-performance financial text processing.

## 🎯 What This API Does

### **Dual LLM Architecture**
- **🎭 FinBERT**: Specialized sentiment analysis for financial texts (positive/negative/neutral)
- **🤖 FinGPT**: Advanced financial text generation, forecasting, and risk analysis
- **🔗 Combined Analysis**: Leverage both models for comprehensive financial insights

### **Core Capabilities**
- **Financial Sentiment Analysis**: Domain-specific sentiment classification
- **Financial Text Generation**: AI-powered financial content creation
- **Market Forecasting**: Generate predictions and market analysis
- **Risk Assessment**: Identify and analyze financial risks
- **Batch Processing**: Efficient processing of multiple texts
- **Real-time Analysis**: Fast inference for live applications

## 🌟 Features

- **🏗️ Production Ready**: Docker, health checks, monitoring, structured logging
- **⚡ High Performance**: Async processing, batch optimization, GPU support
- **🔒 Secure**: Environment-based configuration, token management
- **📚 Well Documented**: Auto-generated OpenAPI/Swagger documentation
- **🧪 Fully Tested**: Comprehensive test suite with 90%+ coverage
- **🐳 Docker First**: Ready-to-deploy containerization

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/rojasjuniore/finbert-fingpt-financial-api.git
cd finbert-fingpt-financial-api

# 2. Setup environment
cp .env.example .env
# Edit .env with your HuggingFace token (optional)

# 3. Start with Docker
docker-compose up -d

# 4. Test the API
curl http://localhost:8000/health
```

### Local Development

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Run the API
python main.py
```

---

## 📖 API Documentation

### Base URL
- **Local**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs` (Swagger UI)

---

## 🎭 FinBERT Sentiment Analysis

### Single Text Analysis

**Endpoint**: `POST /analyze`

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple stock is performing exceptionally well this quarter with record-breaking profits.",
    "return_probabilities": true
  }'
```

**Response**:
```json
{
  "data": {
    "text": "Apple stock is performing exceptionally well this quarter with record-breaking profits.",
    "sentiment": "positive",
    "confidence": 0.9234,
    "probabilities": {
      "positive": 0.9234,
      "negative": 0.0123,
      "neutral": 0.0643
    }
  },
  "metadata": {
    "model_name": "ProsusAI/finbert",
    "batch_size_used": 32,
    "return_probabilities": true
  },
  "total_processing_time": 0.145
}
```

### Bulk Analysis

**Endpoint**: `POST /analyze/bulk`

```bash
curl -X POST "http://localhost:8000/analyze/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Tesla stock surged 15% after announcing record deliveries",
      "Market volatility increased due to inflation concerns",
      "Microsoft maintained steady performance throughout the quarter"
    ],
    "return_probabilities": true,
    "batch_size": 32
  }'
```

**Response**:
```json
{
  "data": [
    {
      "text": "Tesla stock surged 15% after announcing record deliveries",
      "sentiment": "positive",
      "confidence": 0.8967,
      "probabilities": {
        "positive": 0.8967,
        "negative": 0.0234,
        "neutral": 0.0799
      }
    },
    {
      "text": "Market volatility increased due to inflation concerns",
      "sentiment": "negative",
      "confidence": 0.8543,
      "probabilities": {
        "positive": 0.0456,
        "negative": 0.8543,
        "neutral": 0.1001
      }
    },
    {
      "text": "Microsoft maintained steady performance throughout the quarter",
      "sentiment": "neutral",
      "confidence": 0.7234,
      "probabilities": {
        "positive": 0.1234,
        "negative": 0.1532,
        "neutral": 0.7234
      }
    }
  ],
  "statistics": {
    "sentiment_distribution": {
      "positive": 1,
      "negative": 1,
      "neutral": 1
    },
    "average_confidence": 0.8248,
    "min_confidence": 0.7234,
    "max_confidence": 0.8967,
    "processing_rate": 20.5,
    "texts_per_second": 20.5
  },
  "total_processing_time": 0.146,
  "processed_count": 3
}
```

---

## 🤖 FinGPT Text Generation & Analysis

### Financial Text Generation

**Endpoint**: `POST /fingpt/generate`

```bash
curl -X POST "http://localhost:8000/fingpt/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Based on current market trends, the outlook for tech stocks in Q4 2024",
    "max_length": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_return_sequences": 1
  }'
```

**Response**:
```json
{
  "generated_text": [
    "Based on current market trends, the outlook for tech stocks in Q4 2024 appears cautiously optimistic. Several factors support this perspective: strong earnings from major tech companies, continued AI investment driving innovation, and improving supply chain conditions. However, investors should monitor interest rate policies and global economic indicators that could impact valuations."
  ],
  "metadata": {
    "model_name": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
    "prompt_length": 73,
    "generation_config": {
      "max_length": 200,
      "temperature": 0.7,
      "top_p": 0.9,
      "do_sample": true
    }
  },
  "total_processing_time": 2.34
}
```

### Financial Text Analysis

**Endpoint**: `POST /fingpt/analyze`

```bash
curl -X POST "http://localhost:8000/fingpt/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Federal Reserve announced a 0.25% interest rate hike, citing persistent inflation concerns and strong labor market data.",
    "analysis_type": "risk"
  }'
```

**Response**:
```json
{
  "analysis": "This Federal Reserve interest rate increase presents several risk factors for market participants:\n\n1. **Credit Risk**: Higher borrowing costs may strain leveraged companies, potentially increasing default rates\n2. **Market Risk**: Rate hikes typically pressure equity valuations, particularly growth stocks\n3. **Liquidity Risk**: Tighter monetary policy could reduce market liquidity\n4. **Sector Risk**: Interest-sensitive sectors like real estate and utilities may underperform\n\nMitigation strategies should focus on quality investments, diversification, and monitoring duration risk in fixed-income portfolios.",
  "analysis_type": "risk",
  "key_insights": [
    "Interest rate sensitivity analysis",
    "Credit quality deterioration potential",
    "Sector rotation implications",
    "Liquidity conditions assessment"
  ],
  "confidence_score": 0.87,
  "total_processing_time": 3.12
}
```

**Analysis Types Available**:
- `general`: Comprehensive financial analysis
- `sentiment`: Sentiment analysis with detailed reasoning
- `forecast`: Market predictions and trend analysis
- `risk`: Risk identification and assessment

---

## 🔗 Combined Analysis (FinBERT + FinGPT)

**Endpoint**: `POST /combined/analyze`

```bash
curl -X POST "http://localhost:8000/combined/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Amazon reported stronger than expected Q3 earnings with AWS revenue growth of 12%, beating analyst estimates.",
    "include_sentiment": true,
    "include_generation": true,
    "analysis_type": "forecast",
    "return_probabilities": true
  }'
```

**Response**:
```json
{
  "sentiment_analysis": {
    "text": "Amazon reported stronger than expected Q3 earnings with AWS revenue growth of 12%, beating analyst estimates.",
    "sentiment": "positive",
    "confidence": 0.9156,
    "probabilities": {
      "positive": 0.9156,
      "negative": 0.0234,
      "neutral": 0.0610
    }
  },
  "text_generation": {
    "analysis": "Amazon's Q3 results demonstrate strong fundamental performance with AWS leading growth momentum. The 12% AWS revenue growth, while moderating from previous quarters, remains robust and above market expectations.\n\nForecast Outlook:\n• AWS market leadership position strengthens competitive moat\n• Cloud infrastructure demand remains structurally sound\n• E-commerce profitability improvements likely to continue\n• Capital allocation efficiency supporting margin expansion\n\nTarget price adjustment upward based on improved earnings visibility and cloud market dynamics.",
    "analysis_type": "forecast",
    "key_insights": [
      "AWS growth sustainability",
      "Margin expansion trajectory",
      "Competitive positioning strength",
      "Earnings visibility improvement"
    ],
    "confidence_score": 0.89
  },
  "combined_insights": {
    "sentiment_alignment": {
      "finbert_sentiment": "positive",
      "finbert_confidence": 0.9156,
      "fingpt_analysis_type": "forecast",
      "analysis_length": 447
    },
    "key_insights": {
      "sentiment_detected": "positive",
      "analysis_focus": "forecast",
      "combined_recommendation": "Strong positive sentiment detected with high confidence. Detailed forecast analysis provides additional context for decision-making."
    }
  },
  "total_processing_time": 4.23
}
```

---

## 🛠️ Utility Endpoints

### Health Check

```bash
# Basic health check
curl http://localhost:8000/health

# Response:
{"status":"healthy"}

# Detailed health check
curl "http://localhost:8000/health?deep_check=true"
```

**Detailed Health Response**:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "system_info": {
    "platform": "Darwin",
    "python_version": "3.11.0",
    "cpu_count": 8,
    "memory_total": 17179869184,
    "memory_available": 8589934592,
    "memory_percent": 50.0
  },
  "model_info": {
    "finbert_loaded": true,
    "fingpt_loaded": true,
    "device": "cpu",
    "load_time": 12.34
  },
  "performance_metrics": {
    "test_inference_successful": true,
    "inference_time": 0.123,
    "memory_usage": "2.1GB"
  }
}
```

### Model Information

```bash
curl http://localhost:8000/model/info
```

**Response**:
```json
{
  "model_name": "ProsusAI/finbert",
  "model_loaded": true,
  "config": {
    "model_name": "ProsusAI/finbert",
    "max_sequence_length": 512,
    "batch_size": 32,
    "device": "cpu",
    "load_time": 12.34
  },
  "capabilities": [
    "financial_sentiment_analysis",
    "batch_processing",
    "probability_scores",
    "text_classification",
    "financial_text_generation",
    "market_forecasting",
    "risk_analysis"
  ],
  "performance_stats": {
    "total_inferences": 1547,
    "total_inference_time": 189.23,
    "average_inference_time": 0.122
  }
}
```

### API Root Information

```bash
curl http://localhost:8000/
```

**Response**:
```json
{
  "name": "FinBERT + FinGPT API",
  "version": "1.0.0",
  "description": "Financial sentiment analysis and text generation using FinBERT and FinGPT models",
  "model": "ProsusAI/finbert + FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
  "model_loaded": true,
  "endpoints": {
    "analyze": "/analyze",
    "bulk_analyze": "/analyze/bulk",
    "fingpt_generate": "/fingpt/generate",
    "fingpt_analyze": "/fingpt/analyze",
    "combined_analyze": "/combined/analyze",
    "health": "/health",
    "model_info": "/model/info",
    "docs": "/docs",
    "openapi": "/openapi.json"
  }
}
```

---

## 💻 Python SDK Examples

### Complete Python Example

```python
import requests
import json
from typing import List, Dict, Any

class FinancialLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_sentiment(self, text: str, return_probabilities: bool = True) -> Dict[str, Any]:
        """Analyze sentiment of financial text using FinBERT"""
        response = requests.post(f"{self.base_url}/analyze", json={
            "text": text,
            "return_probabilities": return_probabilities
        })
        return response.json()
    
    def bulk_analyze(self, texts: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """Analyze multiple texts efficiently"""
        response = requests.post(f"{self.base_url}/analyze/bulk", json={
            "texts": texts,
            "return_probabilities": True,
            "batch_size": batch_size
        })
        return response.json()
    
    def generate_financial_text(self, prompt: str, max_length: int = 200) -> Dict[str, Any]:
        """Generate financial content using FinGPT"""
        response = requests.post(f"{self.base_url}/fingpt/generate", json={
            "prompt": prompt,
            "max_length": max_length,
            "temperature": 0.7,
            "top_p": 0.9
        })
        return response.json()
    
    def analyze_with_fingpt(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze text using FinGPT"""
        response = requests.post(f"{self.base_url}/fingpt/analyze", json={
            "text": text,
            "analysis_type": analysis_type
        })
        return response.json()
    
    def combined_analysis(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Get combined analysis using both models"""
        response = requests.post(f"{self.base_url}/combined/analyze", json={
            "text": text,
            "include_sentiment": True,
            "include_generation": True,
            "analysis_type": analysis_type,
            "return_probabilities": True
        })
        return response.json()

# Usage Examples
client = FinancialLLMClient()

# 1. Single sentiment analysis
result = client.analyze_sentiment(
    "Netflix stock jumped 8% after subscriber growth exceeded expectations"
)
print(f"Sentiment: {result['data']['sentiment']}")
print(f"Confidence: {result['data']['confidence']:.3f}")

# 2. Bulk analysis
news_headlines = [
    "Federal Reserve signals potential rate cuts in 2024",
    "Tech stocks rally on strong AI earnings reports", 
    "Oil prices decline amid supply increase concerns",
    "Bitcoin reaches new all-time high above $50,000"
]

bulk_results = client.bulk_analyze(news_headlines)
print(f"\nBulk Analysis Results:")
for item in bulk_results['data']:
    print(f"• {item['sentiment'].upper()}: {item['text'][:60]}...")

# 3. Generate financial forecast
forecast = client.generate_financial_text(
    "The outlook for renewable energy stocks in 2024 suggests"
)
print(f"\nGenerated Forecast:\n{forecast['generated_text'][0]}")

# 4. Risk analysis
risk_analysis = client.analyze_with_fingpt(
    "The company has increased its debt-to-equity ratio from 0.3 to 1.2 over the past year",
    analysis_type="risk"
)
print(f"\nRisk Analysis:\n{risk_analysis['analysis']}")

# 5. Combined analysis
combined = client.combined_analysis(
    "Apple announced a $110 billion share buyback program, the largest in company history",
    analysis_type="forecast"
)
print(f"\nCombined Analysis:")
print(f"Sentiment: {combined['sentiment_analysis']['sentiment']}")
print(f"Forecast: {combined['text_generation']['analysis'][:200]}...")
```

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=false

# Hugging Face Configuration (optional for public models)
HF_TOKEN=your_huggingface_token_here

# Model Configuration
MODEL_NAME=ProsusAI/finbert
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=32

# Cache Configuration
TRANSFORMERS_CACHE=/app/.cache/transformers
HF_HOME=/app/.cache/huggingface

# Logging Configuration
LOG_LEVEL=info
LOG_FORMAT=json

# CORS Configuration
ALLOWED_ORIGINS=["*"]
ALLOWED_METHODS=["GET", "POST"]
ALLOWED_HEADERS=["*"]

# Performance Configuration
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Security (Production)
# API_KEY=your-secret-api-key
# RATE_LIMIT=100

# Environment
ENVIRONMENT=development
```

---

## 🚀 Deployment

### Docker Production

```bash
# Build and deploy
docker-compose up -d

# Scale for high availability
docker-compose up -d --scale finbert-api=3

# With Nginx reverse proxy
docker-compose --profile nginx up -d
```

### Performance Optimization

- **GPU Support**: Set `CUDA_VISIBLE_DEVICES=0` for GPU acceleration
- **Memory**: Adjust `BATCH_SIZE` based on available RAM
- **Workers**: Scale `API_WORKERS` for concurrent processing
- **Caching**: Enable model and response caching for repeated queries

---

## 📊 Performance Benchmarks

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Single sentiment analysis | ~50-100ms | ~20 req/sec |
| Batch processing (32 texts) | ~500ms | ~60 texts/sec |
| Text generation | ~2-4s | ~5 req/sec |
| Combined analysis | ~3-6s | ~3 req/sec |

**Memory Requirements**:
- Base: ~2GB (models loaded)
- Peak: ~4GB (during processing)
- GPU: Recommended for >100 req/min

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app tests/

# Performance tests
python -m pytest -m performance

# Load testing with Locust
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## 📈 Monitoring & Observability

### Health Checks
- **Basic**: `GET /health`
- **Deep**: `GET /health?deep_check=true`
- **Metrics**: Processing times in response headers

### Structured Logging
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "uuid-here",
  "endpoint": "/analyze",
  "status_code": 200,
  "process_time": 0.123,
  "model_inference_time": 0.089
}
```

---

## 🛡️ Security & Best Practices

### Production Checklist
- [ ] Set `ENVIRONMENT=production`
- [ ] Configure `API_KEY` for authentication
- [ ] Set specific `ALLOWED_ORIGINS` for CORS
- [ ] Enable `RATE_LIMIT`
- [ ] Use HTTPS in production
- [ ] Monitor resource usage
- [ ] Set up log aggregation
- [ ] Configure health check endpoints

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **API Docs**: http://localhost:8000/docs
- **Issues**: [GitHub Issues](https://github.com/rojasjuniore/finbert-fingpt-financial-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rojasjuniore/finbert-fingpt-financial-api/discussions)

---

## 🏷️ Tags

`finbert` `fingpt` `financial-ai` `sentiment-analysis` `llm` `fastapi` `docker` `nlp` `machine-learning` `production-ready`