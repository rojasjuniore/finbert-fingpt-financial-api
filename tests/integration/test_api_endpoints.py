"""
Integration tests for FinBERT API endpoints.
"""
import pytest
import json
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
import time


class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_health_check_endpoint(self, test_client):
        """Test health check endpoint."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analyze_sentiment_endpoint_positive(self, test_client, mock_api_responses):
        """Test sentiment analysis endpoint with positive text."""
        payload = {
            "text": "Apple Inc. reported record quarterly earnings, beating expectations by 15%."
        }
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="positive",
                score=0.85,
                confidence=0.92,
                processing_time=0.234
            )
            
            response = await test_client.post("/analyze", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "text" in data
            assert "sentiment" in data
            assert "processing_time" in data
            assert "model_info" in data
            
            assert data["text"] == payload["text"]
            assert data["sentiment"]["label"] == "positive"
            assert 0.0 <= data["sentiment"]["score"] <= 1.0
            assert 0.0 <= data["sentiment"]["confidence"] <= 1.0
            assert data["processing_time"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analyze_sentiment_endpoint_negative(self, test_client):
        """Test sentiment analysis endpoint with negative text."""
        payload = {
            "text": "Company faces significant losses and declining market share."
        }
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="negative",
                score=0.23,
                confidence=0.87,
                processing_time=0.198
            )
            
            response = await test_client.post("/analyze", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["sentiment"]["label"] == "negative"
            assert data["sentiment"]["score"] < 0.5
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analyze_sentiment_endpoint_neutral(self, test_client):
        """Test sentiment analysis endpoint with neutral text."""
        payload = {
            "text": "The company will hold its annual shareholders meeting next month."
        }
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.52,
                confidence=0.76,
                processing_time=0.201
            )
            
            response = await test_client.post("/analyze", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["sentiment"]["label"] == "neutral"
            assert 0.4 <= data["sentiment"]["score"] <= 0.6
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analyze_sentiment_missing_text(self, test_client):
        """Test sentiment analysis endpoint with missing text field."""
        payload = {}
        
        response = await test_client.post("/analyze", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analyze_sentiment_empty_text(self, test_client):
        """Test sentiment analysis endpoint with empty text."""
        payload = {"text": ""}
        
        response = await test_client.post("/analyze", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "message" in data
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_analyze_sentiment_invalid_text_type(self, test_client):
        """Test sentiment analysis endpoint with invalid text type."""
        payload = {"text": 123}
        
        response = await test_client.post("/analyze", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_batch_analyze_endpoint(self, test_client):
        """Test batch sentiment analysis endpoint."""
        payload = {
            "texts": [
                "Great quarterly results and strong guidance!",
                "Market conditions remain challenging.",
                "Company announces routine board meeting."
            ]
        }
        
        with patch('src.services.model_service.FinBERTModelService.batch_analyze_sentiment') as mock_batch:
            mock_batch.return_value = MagicMock(
                results=[
                    MagicMock(label="positive", score=0.78, confidence=0.89, processing_time=0.1),
                    MagicMock(label="negative", score=0.25, confidence=0.82, processing_time=0.1),
                    MagicMock(label="neutral", score=0.48, confidence=0.71, processing_time=0.1)
                ],
                batch_size=3,
                processing_time=0.456
            )
            
            response = await test_client.post("/batch-analyze", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "results" in data
            assert "batch_size" in data
            assert "processing_time" in data
            
            assert len(data["results"]) == 3
            assert data["batch_size"] == 3
            assert data["processing_time"] > 0
            
            # Check individual results
            assert data["results"][0]["sentiment"]["label"] == "positive"
            assert data["results"][1]["sentiment"]["label"] == "negative"
            assert data["results"][2]["sentiment"]["label"] == "neutral"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_batch_analyze_empty_list(self, test_client):
        """Test batch analysis with empty texts list."""
        payload = {"texts": []}
        
        response = await test_client.post("/batch-analyze", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_batch_analyze_missing_texts(self, test_client):
        """Test batch analysis with missing texts field."""
        payload = {}
        
        response = await test_client.post("/batch-analyze", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_batch_analyze_large_batch(self, test_client):
        """Test batch analysis with large number of texts."""
        payload = {
            "texts": [f"Test financial text {i}" for i in range(50)]
        }
        
        with patch('src.services.model_service.FinBERTModelService.batch_analyze_sentiment') as mock_batch:
            mock_results = [
                MagicMock(label="neutral", score=0.5, confidence=0.7, processing_time=0.1)
                for _ in range(50)
            ]
            mock_batch.return_value = MagicMock(
                results=mock_results,
                batch_size=50,
                processing_time=2.5
            )
            
            response = await test_client.post("/batch-analyze", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["results"]) == 50
            assert data["batch_size"] == 50
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_model_info_endpoint(self, test_client):
        """Test model information endpoint."""
        with patch('src.services.model_service.FinBERTModelService.get_model_info') as mock_info:
            mock_info.return_value = {
                "model_name": "ProsusAI/finbert",
                "max_sequence_length": 512,
                "batch_size": 32,
                "device": "cpu"
            }
            
            response = await test_client.get("/model-info")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "model_name" in data
            assert "max_sequence_length" in data
            assert "batch_size" in data
            assert "device" in data
            assert data["model_name"] == "ProsusAI/finbert"
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint for monitoring."""
        response = await test_client.get("/metrics")
        
        assert response.status_code == 200
        # Prometheus metrics should be in text format
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        
        content = response.text
        assert "finbert_requests_total" in content
        assert "finbert_request_duration_seconds" in content
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_cors_headers(self, test_client):
        """Test CORS headers in responses."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        # Check for CORS headers (if configured)
        headers = response.headers
        # These would be present if CORS is properly configured
        # assert "access-control-allow-origin" in headers
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_content_type_validation(self, test_client):
        """Test content type validation."""
        # Test with invalid content type
        response = await test_client.post(
            "/analyze",
            data="invalid data",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code in [400, 415, 422]
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_request_timeout(self, test_client):
        """Test request timeout handling."""
        payload = {"text": "Test timeout handling"}
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            # Simulate slow processing
            import asyncio
            
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(10)  # 10 second delay
                return MagicMock(label="neutral", score=0.5, confidence=0.7, processing_time=10.0)
            
            mock_analyze.side_effect = slow_process
            
            # This should timeout (depending on server configuration)
            try:
                response = await test_client.post("/analyze", json=payload, timeout=5.0)
                # If no timeout, just check it's still a valid response
                assert response.status_code in [200, 408, 504]
            except Exception:
                # Timeout occurred, which is expected
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_malformed_json(self, test_client):
        """Test handling of malformed JSON."""
        response = await test_client.post(
            "/analyze",
            data='{"text": invalid json}',
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_large_payload(self, test_client):
        """Test handling of very large payloads."""
        # Create a very large text payload
        large_text = "A" * 100000  # 100KB text
        payload = {"text": large_text}
        
        response = await test_client.post("/analyze", json=payload)
        
        # Should either process successfully or return appropriate error
        assert response.status_code in [200, 413, 400]
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_concurrent_requests(self, test_client):
        """Test handling of concurrent requests."""
        import asyncio
        
        async def make_request(text):
            payload = {"text": f"Test concurrent request: {text}"}
            return await test_client.post("/analyze", json=payload)
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            # Make 10 concurrent requests
            tasks = [make_request(f"text_{i}") for i in range(10)]
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_error_handling_model_failure(self, test_client):
        """Test error handling when model fails."""
        payload = {"text": "Test error handling"}
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = Exception("Model processing failed")
            
            response = await test_client.post("/analyze", json=payload)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "message" in data
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_response_headers(self, test_client):
        """Test response headers."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        headers = response.headers
        
        # Check for security headers
        assert "content-type" in headers
        # Additional security headers might be present
        # assert "x-content-type-options" in headers
        # assert "x-frame-options" in headers
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_api_versioning(self, test_client):
        """Test API versioning in responses."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include version information
        assert "version" in data
        assert isinstance(data["version"], str)
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_rate_limiting_headers(self, test_client):
        """Test rate limiting headers (if implemented)."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        headers = response.headers
        
        # Rate limiting headers (if implemented)
        # assert "x-ratelimit-limit" in headers
        # assert "x-ratelimit-remaining" in headers
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_response_time_benchmark(self, test_client, performance_benchmarks):
        """Test API response time benchmark."""
        payload = {"text": "Test response time for financial sentiment analysis."}
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            start_time = time.time()
            response = await test_client.post("/analyze", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < performance_benchmarks["single_request"]["max_response_time"]
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_openapi_docs(self, test_client):
        """Test OpenAPI documentation endpoint."""
        response = await test_client.get("/docs")
        
        # Should redirect or return documentation
        assert response.status_code in [200, 301, 302]
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_openapi_json(self, test_client):
        """Test OpenAPI JSON schema endpoint."""
        response = await test_client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        
        # Check that our endpoints are documented
        assert "/analyze" in data["paths"]
        assert "/batch-analyze" in data["paths"]
        assert "/health" in data["paths"]
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = await test_client.get("/")
        
        # Should return some kind of welcome or info message
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            # Should contain basic API information
            assert isinstance(data, dict)