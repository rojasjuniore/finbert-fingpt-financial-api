"""
Tests for FinBERT API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_health_check(self):
        """Test root health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["service"] == "FinBERT API"
    
    def test_simple_health_check(self):
        """Test simple health check endpoint"""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # Depends on model loading
        data = response.json()
        assert "status" in data
    
    def test_detailed_health_check(self):
        """Test detailed health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "api_version" in data
        assert "model_loaded" in data


class TestModelInfo:
    """Test model info endpoint"""
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_loaded" in data
        assert "config" in data
        assert "capabilities" in data


class TestSentimentAnalysis:
    """Test sentiment analysis endpoints"""
    
    def test_analyze_single_text(self):
        """Test single text sentiment analysis"""
        test_data = {
            "text": "The company reported strong quarterly earnings with significant growth.",
            "return_probabilities": False
        }
        
        response = client.post("/api/v1/analyze", json=test_data)
        
        # If model is not loaded, expect 503
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "data" in data
        assert data["success"] is True
        
        # Check sentiment result structure
        result = data["data"]
        assert "text" in result
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
    
    def test_analyze_multiple_texts(self):
        """Test multiple texts sentiment analysis"""
        test_data = {
            "text": [
                "The stock market showed strong performance today.",
                "The company's earnings disappointed investors.",
                "Market conditions remain stable."
            ],
            "return_probabilities": True
        }
        
        response = client.post("/api/v1/analyze", json=test_data)
        
        # If model is not loaded, expect 503
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "data" in data
        assert data["success"] is True
        
        # Check results structure
        results = data["data"]
        assert isinstance(results, list)
        assert len(results) == 3
        
        for result in results:
            assert "text" in result
            assert "sentiment" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_bulk_analyze(self):
        """Test bulk sentiment analysis"""
        test_data = {
            "texts": [
                "Strong quarterly results exceeded expectations.",
                "Market volatility caused concerns among investors.",
                "The company maintains a stable outlook."
            ],
            "return_probabilities": False,
            "batch_size": 2
        }
        
        response = client.post("/api/v1/analyze/bulk", json=test_data)
        
        # If model is not loaded, expect 503
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "data" in data
        assert "statistics" in data
        assert "processed_count" in data
        assert data["success"] is True
        assert data["processed_count"] == 3


class TestValidation:
    """Test input validation"""
    
    def test_empty_text_validation(self):
        """Test validation of empty text"""
        test_data = {"text": ""}
        response = client.post("/api/v1/analyze", json=test_data)
        assert response.status_code == 422
    
    def test_long_text_validation(self):
        """Test validation of very long text"""
        test_data = {"text": "a" * 20000}  # Very long text
        response = client.post("/api/v1/analyze", json=test_data)
        assert response.status_code == 422
    
    def test_empty_text_list_validation(self):
        """Test validation of empty text list"""
        test_data = {"text": []}
        response = client.post("/api/v1/analyze", json=test_data)
        assert response.status_code == 422
    
    def test_too_many_texts_validation(self):
        """Test validation of too many texts"""
        test_data = {"text": ["text"] * 200}  # Too many texts
        response = client.post("/api/v1/analyze", json=test_data)
        assert response.status_code == 422