"""
Tests for FinBERT service
"""

import pytest
from unittest.mock import Mock, patch
from app.services.finbert_service import FinBERTService


class TestFinBERTService:
    """Test FinBERT service functionality"""
    
    @pytest.fixture
    def service(self):
        """Create a FinBERT service instance"""
        return FinBERTService()
    
    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service.model is None
        assert service.tokenizer is None
        assert not service.model_loaded
        assert service.inference_count == 0
        assert service.total_inference_time == 0.0
    
    def test_is_loaded_false_initially(self, service):
        """Test is_loaded returns False initially"""
        assert not service.is_loaded()
    
    def test_get_model_info_not_loaded(self, service):
        """Test get_model_info when model is not loaded"""
        info = service.get_model_info()
        assert not info["model_loaded"]
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_not_loaded(self, service):
        """Test analyze_sentiment raises error when model not loaded"""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await service.analyze_sentiment("test text")
    
    @pytest.mark.asyncio
    async def test_health_check_not_loaded(self, service):
        """Test health check when model not loaded"""
        health_info = await service.health_check()
        assert not health_info["model_loaded"]
        assert health_info["device"] is None
        assert health_info["load_time"] is None
    
    @pytest.mark.asyncio
    @patch('app.services.finbert_service.AutoTokenizer')
    @patch('app.services.finbert_service.AutoModelForSequenceClassification')
    @patch('app.services.finbert_service.pipeline')
    @patch('app.services.finbert_service.torch')
    async def test_load_model_success(self, mock_torch, mock_pipeline, mock_model, mock_tokenizer, service):
        """Test successful model loading"""
        # Mock dependencies
        mock_torch.device.return_value = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Test loading
        result = await service.load_model()
        
        assert result is True
        assert service.model_loaded is True
        assert service.load_time is not None
    
    @pytest.mark.asyncio
    @patch('app.services.finbert_service.AutoTokenizer')
    async def test_load_model_failure(self, mock_tokenizer, service):
        """Test model loading failure"""
        # Mock tokenizer to raise exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Loading failed")
        
        # Test loading
        result = await service.load_model()
        
        assert result is False
        assert service.model_loaded is False