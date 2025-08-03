"""
Unit tests for FinBERT model service.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any
import torch

from src.services.model_service import (
    FinBERTModelService,
    SentimentResult,
    BatchSentimentResult,
    ModelConfig
)
from src.exceptions import ModelError, ValidationError


class TestFinBERTModelService:
    """Test cases for FinBERT model service."""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration for testing."""
        return ModelConfig(
            model_name="ProsusAI/finbert",
            max_sequence_length=512,
            batch_size=32,
            cache_dir="/tmp/test_cache"
        )
    
    @pytest.fixture
    def mock_model_outputs(self):
        """Mock model outputs for testing."""
        return {
            "positive": torch.tensor([[0.1, 0.2, 0.7]]),  # [negative, neutral, positive]
            "negative": torch.tensor([[0.8, 0.15, 0.05]]),
            "neutral": torch.tensor([[0.2, 0.6, 0.2]]),
            "batch": torch.tensor([
                [0.1, 0.2, 0.7],  # positive
                [0.8, 0.15, 0.05],  # negative
                [0.2, 0.6, 0.2]    # neutral
            ])
        }
    
    @pytest.fixture
    def model_service(self, model_config, mock_transformers):
        """Create model service instance for testing."""
        with patch('src.services.model_service.AutoModelForSequenceClassification') as mock_model_cls, \
             patch('src.services.model_service.AutoTokenizer') as mock_tokenizer_cls:
            
            mock_model_cls.from_pretrained.return_value = mock_transformers["model"]
            mock_tokenizer_cls.from_pretrained.return_value = mock_transformers["tokenizer"]
            
            service = FinBERTModelService(model_config)
            return service
    
    def test_model_initialization(self, model_config, mock_transformers):
        """Test model service initialization."""
        with patch('src.services.model_service.AutoModelForSequenceClassification') as mock_model_cls, \
             patch('src.services.model_service.AutoTokenizer') as mock_tokenizer_cls:
            
            mock_model_cls.from_pretrained.return_value = mock_transformers["model"]
            mock_tokenizer_cls.from_pretrained.return_value = mock_transformers["tokenizer"]
            
            service = FinBERTModelService(model_config)
            
            assert service.config == model_config
            assert service.model is not None
            assert service.tokenizer is not None
            mock_model_cls.from_pretrained.assert_called_once()
            mock_tokenizer_cls.from_pretrained.assert_called_once()
    
    def test_model_initialization_failure(self, model_config):
        """Test model initialization failure handling."""
        with patch('src.services.model_service.AutoModelForSequenceClassification') as mock_model_cls:
            mock_model_cls.from_pretrained.side_effect = Exception("Model load failed")
            
            with pytest.raises(ModelError, match="Failed to load model"):
                FinBERTModelService(model_config)
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive(self, model_service, mock_model_outputs):
        """Test sentiment analysis for positive text."""
        text = "Apple Inc. reported record quarterly earnings, beating expectations."
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=mock_model_outputs["positive"]
            )
            
            result = await model_service.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.label == "positive"
            assert 0.6 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative(self, model_service, mock_model_outputs):
        """Test sentiment analysis for negative text."""
        text = "Company faces significant losses and declining market share."
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=mock_model_outputs["negative"]
            )
            
            result = await model_service.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.label == "negative"
            assert 0.0 <= result.score <= 0.4
            assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral(self, model_service, mock_model_outputs):
        """Test sentiment analysis for neutral text."""
        text = "The company will hold its annual meeting next month."
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=mock_model_outputs["neutral"]
            )
            
            result = await model_service.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.label == "neutral"
            assert 0.4 <= result.score <= 0.6
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_empty_text(self, model_service):
        """Test sentiment analysis with empty text."""
        with pytest.raises(ValidationError, match="Text cannot be empty"):
            await model_service.analyze_sentiment("")
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_none_text(self, model_service):
        """Test sentiment analysis with None text."""
        with pytest.raises(ValidationError, match="Text cannot be None"):
            await model_service.analyze_sentiment(None)
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_long_text(self, model_service):
        """Test sentiment analysis with very long text."""
        long_text = "This is a test. " * 1000  # Very long text
        
        with patch.object(model_service.tokenizer, '__call__') as mock_tokenizer:
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[101] + [1] * 510 + [102]]),  # Max length tokens
                "attention_mask": torch.tensor([[1] * 512])
            }
            
            with patch.object(model_service.model, '__call__') as mock_model_call:
                mock_model_call.return_value = MagicMock(
                    logits=torch.tensor([[0.2, 0.6, 0.2]])
                )
                
                result = await model_service.analyze_sentiment(long_text)
                
                assert isinstance(result, SentimentResult)
                # Verify text was truncated
                mock_tokenizer.assert_called_once()
                call_args = mock_tokenizer.call_args
                assert call_args[1]["max_length"] == 512
                assert call_args[1]["truncation"] is True
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_special_characters(self, model_service, mock_model_outputs):
        """Test sentiment analysis with special characters."""
        text = "Company's Q3 earnings: +15% YoY! 📈 $AAPL #bullish"
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=mock_model_outputs["positive"]
            )
            
            result = await model_service.analyze_sentiment(text)
            
            assert isinstance(result, SentimentResult)
            assert result.label in ["positive", "negative", "neutral"]
    
    @pytest.mark.asyncio
    async def test_batch_analyze_sentiment(self, model_service, mock_model_outputs):
        """Test batch sentiment analysis."""
        texts = [
            "Great quarterly results!",
            "Declining market conditions.",
            "Routine board meeting scheduled."
        ]
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=mock_model_outputs["batch"]
            )
            
            result = await model_service.batch_analyze_sentiment(texts)
            
            assert isinstance(result, BatchSentimentResult)
            assert len(result.results) == 3
            assert result.batch_size == 3
            assert result.processing_time > 0
            
            # Check individual results
            assert result.results[0].label == "positive"
            assert result.results[1].label == "negative"
            assert result.results[2].label == "neutral"
    
    @pytest.mark.asyncio
    async def test_batch_analyze_empty_list(self, model_service):
        """Test batch analysis with empty list."""
        with pytest.raises(ValidationError, match="Texts list cannot be empty"):
            await model_service.batch_analyze_sentiment([])
    
    @pytest.mark.asyncio
    async def test_batch_analyze_large_batch(self, model_service):
        """Test batch analysis with large batch size."""
        texts = [f"Test text {i}" for i in range(100)]
        
        with patch.object(model_service, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = SentimentResult(
                label="neutral",
                score=0.5,
                confidence=0.8,
                processing_time=0.1
            )
            
            result = await model_service.batch_analyze_sentiment(texts)
            
            assert isinstance(result, BatchSentimentResult)
            assert len(result.results) == 100
            assert result.batch_size == 100
            # Should be called in batches
            assert mock_analyze.call_count == 100
    
    @pytest.mark.asyncio
    async def test_model_inference_error(self, model_service):
        """Test handling of model inference errors."""
        text = "Test text"
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.side_effect = Exception("CUDA out of memory")
            
            with pytest.raises(ModelError, match="Model inference failed"):
                await model_service.analyze_sentiment(text)
    
    @pytest.mark.asyncio
    async def test_tokenization_error(self, model_service):
        """Test handling of tokenization errors."""
        text = "Test text"
        
        with patch.object(model_service.tokenizer, '__call__') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Tokenization failed")
            
            with pytest.raises(ModelError, match="Tokenization failed"):
                await model_service.analyze_sentiment(text)
    
    def test_preprocess_text(self, model_service):
        """Test text preprocessing."""
        # Test normal text
        text = "  Apple Inc. reported strong earnings!  "
        processed = model_service._preprocess_text(text)
        assert processed == "Apple Inc. reported strong earnings!"
        
        # Test text with multiple whitespaces
        text = "Multiple   spaces    here"
        processed = model_service._preprocess_text(text)
        assert processed == "Multiple spaces here"
        
        # Test text with newlines and tabs
        text = "Line 1\nLine 2\tTabbed"
        processed = model_service._preprocess_text(text)
        assert processed == "Line 1 Line 2 Tabbed"
    
    def test_postprocess_logits(self, model_service):
        """Test logits postprocessing."""
        # Test positive sentiment
        logits = torch.tensor([[0.1, 0.2, 0.7]])
        label, score, confidence = model_service._postprocess_logits(logits)
        assert label == "positive"
        assert 0.6 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        
        # Test negative sentiment
        logits = torch.tensor([[0.8, 0.15, 0.05]])
        label, score, confidence = model_service._postprocess_logits(logits)
        assert label == "negative"
        assert 0.0 <= score <= 0.4
        
        # Test neutral sentiment
        logits = torch.tensor([[0.2, 0.6, 0.2]])
        label, score, confidence = model_service._postprocess_logits(logits)
        assert label == "neutral"
        assert 0.4 <= score <= 0.6
    
    def test_calculate_confidence(self, model_service):
        """Test confidence calculation."""
        # High confidence case
        probabilities = torch.tensor([0.1, 0.05, 0.85])
        confidence = model_service._calculate_confidence(probabilities)
        assert confidence > 0.8
        
        # Low confidence case (similar probabilities)
        probabilities = torch.tensor([0.35, 0.33, 0.32])
        confidence = model_service._calculate_confidence(probabilities)
        assert confidence < 0.5
        
        # Medium confidence case
        probabilities = torch.tensor([0.2, 0.1, 0.7])
        confidence = model_service._calculate_confidence(probabilities)
        assert 0.5 <= confidence <= 0.8
    
    @pytest.mark.benchmark
    def test_single_analysis_performance(self, model_service, benchmark):
        """Benchmark single text analysis performance."""
        text = "Apple Inc. reported strong quarterly earnings beating analyst expectations."
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=torch.tensor([[0.1, 0.2, 0.7]])
            )
            
            async def analyze():
                return await model_service.analyze_sentiment(text)
            
            import asyncio
            result = benchmark(asyncio.run, analyze())
            assert isinstance(result, SentimentResult)
    
    @pytest.mark.benchmark
    def test_batch_analysis_performance(self, model_service, benchmark):
        """Benchmark batch analysis performance."""
        texts = [f"Test financial text {i} with earnings data." for i in range(10)]
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=torch.tensor([[0.1, 0.2, 0.7]] * 10)
            )
            
            async def batch_analyze():
                return await model_service.batch_analyze_sentiment(texts)
            
            import asyncio
            result = benchmark(asyncio.run, batch_analyze())
            assert isinstance(result, BatchSentimentResult)
            assert len(result.results) == 10
    
    def test_memory_usage(self, model_service):
        """Test memory usage during analysis."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        text = "Test text for memory usage analysis."
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=torch.tensor([[0.1, 0.2, 0.7]])
            )
            
            import asyncio
            result = asyncio.run(model_service.analyze_sentiment(text))
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB for single analysis)
            assert memory_increase < 100
            assert isinstance(result, SentimentResult)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, model_service):
        """Test handling of concurrent sentiment analysis requests."""
        texts = [f"Concurrent test text {i}" for i in range(10)]
        
        with patch.object(model_service.model, '__call__') as mock_model_call:
            mock_model_call.return_value = MagicMock(
                logits=torch.tensor([[0.1, 0.2, 0.7]])
            )
            
            import asyncio
            
            # Create concurrent tasks
            tasks = [model_service.analyze_sentiment(text) for text in texts]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            for result in results:
                assert isinstance(result, SentimentResult)
                assert result.label in ["positive", "negative", "neutral"]
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid config
        config = ModelConfig(
            model_name="ProsusAI/finbert",
            max_sequence_length=512,
            batch_size=32
        )
        assert config.model_name == "ProsusAI/finbert"
        assert config.max_sequence_length == 512
        assert config.batch_size == 32
        
        # Invalid sequence length
        with pytest.raises(ValueError):
            ModelConfig(
                model_name="ProsusAI/finbert",
                max_sequence_length=0,
                batch_size=32
            )
        
        # Invalid batch size
        with pytest.raises(ValueError):
            ModelConfig(
                model_name="ProsusAI/finbert",
                max_sequence_length=512,
                batch_size=0
            )
    
    @pytest.mark.asyncio
    async def test_model_warm_up(self, model_service):
        """Test model warm-up functionality."""
        with patch.object(model_service, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = SentimentResult(
                label="neutral",
                score=0.5,
                confidence=0.8,
                processing_time=0.1
            )
            
            await model_service.warm_up()
            
            # Should call analyze_sentiment with warm-up text
            mock_analyze.assert_called_once()
            args = mock_analyze.call_args[0]
            assert len(args[0]) > 0  # Should have warm-up text
    
    def test_get_model_info(self, model_service):
        """Test getting model information."""
        info = model_service.get_model_info()
        
        assert "model_name" in info
        assert "max_sequence_length" in info
        assert "batch_size" in info
        assert "device" in info
        assert info["model_name"] == "ProsusAI/finbert"
        assert info["max_sequence_length"] == 512
    
    @pytest.mark.asyncio
    async def test_health_check(self, model_service):
        """Test model service health check."""
        with patch.object(model_service, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = SentimentResult(
                label="neutral",
                score=0.5,
                confidence=0.8,
                processing_time=0.1
            )
            
            is_healthy = await model_service.health_check()
            
            assert is_healthy is True
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, model_service):
        """Test model service health check failure."""
        with patch.object(model_service, 'analyze_sentiment') as mock_analyze:
            mock_analyze.side_effect = Exception("Model failed")
            
            is_healthy = await model_service.health_check()
            
            assert is_healthy is False