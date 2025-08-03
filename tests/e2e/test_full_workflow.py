"""
End-to-end tests for FinBERT API full workflow.
"""
import pytest
import asyncio
import time
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
import json


class TestFullWorkflow:
    """End-to-end tests for complete FinBERT API workflow."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_sentiment_analysis_workflow(self, test_client, sample_financial_texts):
        """Test complete sentiment analysis workflow from request to response."""
        # Test positive sentiment workflow
        positive_text = sample_financial_texts["positive"][0]
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="positive",
                score=0.87,
                confidence=0.93,
                processing_time=0.245
            )
            
            # Make request
            response = await test_client.post("/analyze", json={"text": positive_text})
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Verify complete response structure
            assert "text" in data
            assert "sentiment" in data
            assert "processing_time" in data
            assert "model_info" in data
            assert "timestamp" in data
            
            # Verify sentiment analysis
            sentiment = data["sentiment"]
            assert sentiment["label"] == "positive"
            assert 0.0 <= sentiment["score"] <= 1.0
            assert 0.0 <= sentiment["confidence"] <= 1.0
            
            # Verify model info
            model_info = data["model_info"]
            assert "name" in model_info
            assert "version" in model_info
            
            # Verify processing time is reasonable
            assert data["processing_time"] > 0
            assert data["processing_time"] < 5.0  # Should be under 5 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_batch_processing_workflow(self, test_client, sample_financial_texts):
        """Test complete batch processing workflow."""
        texts = [
            sample_financial_texts["positive"][0],
            sample_financial_texts["negative"][0],
            sample_financial_texts["neutral"][0]
        ]
        
        with patch('src.services.model_service.FinBERTModelService.batch_analyze_sentiment') as mock_batch:
            mock_batch.return_value = MagicMock(
                results=[
                    MagicMock(label="positive", score=0.85, confidence=0.91, processing_time=0.1),
                    MagicMock(label="negative", score=0.22, confidence=0.88, processing_time=0.1),
                    MagicMock(label="neutral", score=0.48, confidence=0.75, processing_time=0.1)
                ],
                batch_size=3,
                processing_time=0.567
            )
            
            # Make batch request
            response = await test_client.post("/batch-analyze", json={"texts": texts})
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            # Verify batch response structure
            assert "results" in data
            assert "batch_size" in data
            assert "processing_time" in data
            assert "timestamp" in data
            
            # Verify batch processing
            assert data["batch_size"] == 3
            assert len(data["results"]) == 3
            
            # Verify individual results
            results = data["results"]
            assert results[0]["sentiment"]["label"] == "positive"
            assert results[1]["sentiment"]["label"] == "negative"
            assert results[2]["sentiment"]["label"] == "neutral"
            
            # Verify all results have required fields
            for result in results:
                assert "text" in result
                assert "sentiment" in result
                assert "processing_time" in result
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_caching_workflow(self, test_client, clean_redis):
        """Test caching workflow with cache hits and misses."""
        text = "Apple Inc. reports strong quarterly earnings with 15% growth."
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="positive",
                score=0.82,
                confidence=0.89,
                processing_time=0.234
            )
            
            # First request - should be cache miss
            response1 = await test_client.post("/analyze", json={"text": text})
            assert response1.status_code == 200
            
            # Second request - should be cache hit (faster)
            start_time = time.time()
            response2 = await test_client.post("/analyze", json={"text": text})
            end_time = time.time()
            
            assert response2.status_code == 200
            
            # Cache hit should be faster
            cache_response_time = end_time - start_time
            assert cache_response_time < 0.1  # Should be very fast for cache hit
            
            # Results should be identical
            data1 = response1.json()
            data2 = response2.json()
            assert data1["sentiment"]["label"] == data2["sentiment"]["label"]
            assert data1["sentiment"]["score"] == data2["sentiment"]["score"]
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_error_handling_workflow(self, test_client):
        """Test complete error handling workflow."""
        # Test invalid input
        response = await test_client.post("/analyze", json={"text": ""})
        assert response.status_code == 400
        
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data
        assert "code" in error_data
        
        # Test malformed JSON
        response = await test_client.post(
            "/analyze",
            data='{"text": invalid}',
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing field
        response = await test_client.post("/analyze", json={})
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_health_monitoring_workflow(self, test_client):
        """Test health monitoring and status workflow."""
        # Check health endpoint
        response = await test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "version" in health_data
        assert "model_loaded" in health_data
        
        # Check metrics endpoint
        response = await test_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        
        # Check model info endpoint
        with patch('src.services.model_service.FinBERTModelService.get_model_info') as mock_info:
            mock_info.return_value = {
                "model_name": "ProsusAI/finbert",
                "max_sequence_length": 512,
                "batch_size": 32,
                "device": "cpu"
            }
            
            response = await test_client.get("/model-info")
            assert response.status_code == 200
            
            model_data = response.json()
            assert model_data["model_name"] == "ProsusAI/finbert"
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_concurrent_requests_workflow(self, test_client, sample_financial_texts):
        """Test concurrent requests handling workflow."""
        texts = sample_financial_texts["positive"][:5]  # Use 5 different texts
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="positive",
                score=0.8,
                confidence=0.85,
                processing_time=0.2
            )
            
            async def make_request(text):
                return await test_client.post("/analyze", json={"text": text})
            
            # Make 5 concurrent requests
            start_time = time.time()
            tasks = [make_request(text) for text in texts]
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["sentiment"]["label"] == "positive"
            
            # Total time should be reasonable (not 5x single request time)
            total_time = end_time - start_time
            assert total_time < 2.0  # Concurrent processing should be faster
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_edge_cases_workflow(self, test_client, sample_financial_texts):
        """Test edge cases workflow."""
        edge_cases = sample_financial_texts["edge_cases"]
        
        for i, text in enumerate(edge_cases):
            with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
                if text == "":  # Empty string
                    # Should return error for empty text
                    response = await test_client.post("/analyze", json={"text": text})
                    assert response.status_code == 400
                    continue
                
                mock_analyze.return_value = MagicMock(
                    label="neutral",
                    score=0.5,
                    confidence=0.6,
                    processing_time=0.15
                )
                
                response = await test_client.post("/analyze", json={"text": text})
                
                if len(text.strip()) > 0:  # Non-empty text should work
                    assert response.status_code == 200
                    data = response.json()
                    assert "sentiment" in data
                    assert data["sentiment"]["label"] in ["positive", "negative", "neutral"]
                else:
                    assert response.status_code == 400
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_load_testing_workflow(self, test_client):
        """Test basic load testing workflow."""
        # Simple load test with multiple requests
        num_requests = 20
        test_text = "Load testing text for FinBERT API performance validation."
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1
            )
            
            async def make_request():
                return await test_client.post("/analyze", json={"text": test_text})
            
            # Measure performance
            start_time = time.time()
            tasks = [make_request() for _ in range(num_requests)]
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Verify all requests succeeded
            success_count = sum(1 for r in responses if r.status_code == 200)
            assert success_count == num_requests
            
            # Calculate performance metrics
            total_time = end_time - start_time
            requests_per_second = num_requests / total_time
            
            # Performance should be reasonable
            assert requests_per_second > 10  # At least 10 req/sec
            assert total_time < 10  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_api_documentation_workflow(self, test_client):
        """Test API documentation workflow."""
        # Test OpenAPI JSON
        response = await test_client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Verify our endpoints are documented
        paths = openapi_spec["paths"]
        assert "/analyze" in paths
        assert "/batch-analyze" in paths
        assert "/health" in paths
        assert "/model-info" in paths
        
        # Verify endpoint documentation
        analyze_endpoint = paths["/analyze"]
        assert "post" in analyze_endpoint
        assert "summary" in analyze_endpoint["post"]
        assert "requestBody" in analyze_endpoint["post"]
        assert "responses" in analyze_endpoint["post"]
        
        # Test docs UI (if available)
        response = await test_client.get("/docs")
        assert response.status_code in [200, 301, 302]  # Might redirect
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_model_warm_up_workflow(self, test_client):
        """Test model warm-up workflow."""
        # First request might be slower (model loading)
        warm_up_text = "Warm up text for model initialization."
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.8  # Simulated slower first request
            )
            
            # First request (warm-up)
            start_time = time.time()
            response1 = await test_client.post("/analyze", json={"text": warm_up_text})
            first_time = time.time() - start_time
            
            assert response1.status_code == 200
            
            # Mock faster subsequent requests
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.1  # Much faster
            )
            
            # Second request (should be faster)
            start_time = time.time()
            response2 = await test_client.post("/analyze", json={"text": warm_up_text})
            second_time = time.time() - start_time
            
            assert response2.status_code == 200
            
            # Both should work, timing will depend on actual implementation
            data1 = response1.json()
            data2 = response2.json()
            assert "sentiment" in data1
            assert "sentiment" in data2
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_integration_with_news_api(self, test_client):
        """Test integration workflow with news trading API format."""
        # Test format compatible with the main news trading API
        financial_news = {
            "title": "Apple Inc. Reports Record Q4 Earnings",
            "content": "Apple Inc. reported record quarterly earnings, beating analyst expectations by 15%. Revenue grew 8% year-over-year to $94.9 billion.",
            "source": "financial_news_test",
            "symbols": ["AAPL"],
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        # Extract text for sentiment analysis
        analysis_text = f"{financial_news['title']} {financial_news['content']}"
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="positive",
                score=0.87,
                confidence=0.92,
                processing_time=0.234
            )
            
            response = await test_client.post("/analyze", json={"text": analysis_text})
            
            assert response.status_code == 200
            data = response.json()
            
            # Format response for news API integration
            sentiment_data = {
                "finbert_score": data["sentiment"]["score"],
                "finbert_label": data["sentiment"]["label"],
                "confidence": data["sentiment"]["confidence"],
                "processing_time": data["processing_time"]
            }
            
            # Verify integration format
            assert "finbert_score" in sentiment_data
            assert "finbert_label" in sentiment_data
            assert "confidence" in sentiment_data
            assert sentiment_data["finbert_label"] == "positive"
            assert 0.0 <= sentiment_data["finbert_score"] <= 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_graceful_degradation_workflow(self, test_client):
        """Test graceful degradation workflow when services fail."""
        # Test when model service fails
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = Exception("Model processing failed")
            
            response = await test_client.post("/analyze", json={"text": "Test graceful failure"})
            
            # Should return error but not crash
            assert response.status_code == 500
            error_data = response.json()
            assert "error" in error_data
            assert "message" in error_data
        
        # Health check should still work
        response = await test_client.get("/health")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_monitoring_metrics_workflow(self, test_client):
        """Test monitoring and metrics collection workflow."""
        # Make some requests to generate metrics
        test_requests = [
            {"text": "Positive earnings report"},
            {"text": "Market decline continues"},
            {"text": "Neutral company announcement"}
        ]
        
        with patch('src.services.model_service.FinBERTModelService.analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = MagicMock(
                label="neutral",
                score=0.5,
                confidence=0.7,
                processing_time=0.15
            )
            
            # Make test requests
            for request_data in test_requests:
                response = await test_client.post("/analyze", json=request_data)
                assert response.status_code == 200
        
        # Check metrics endpoint
        response = await test_client.get("/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        # Should contain Prometheus metrics
        assert "finbert_requests_total" in metrics_text
        assert "finbert_request_duration_seconds" in metrics_text
        
        # Check health endpoint for system status
        response = await test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"