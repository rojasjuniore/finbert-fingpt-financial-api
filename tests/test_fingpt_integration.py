"""
Test cases for FinGPT integration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.fingpt_service import FinGPTService
from app.models.responses import FinGPTResponse, FinGPTAnalysisResponse


class TestFinGPTService:
    """Test FinGPT service functionality"""
    
    @pytest.fixture
    def fingpt_service(self):
        """Create FinGPT service instance for testing"""
        return FinGPTService()
    
    def test_fingpt_service_initialization(self, fingpt_service):
        """Test FinGPT service initialization"""
        assert fingpt_service.model is None
        assert fingpt_service.tokenizer is None
        assert fingpt_service.pipeline is None
        assert not fingpt_service.model_loaded
        assert fingpt_service.inference_count == 0
        assert fingpt_service.total_inference_time == 0.0
    
    def test_is_loaded_when_not_loaded(self, fingpt_service):
        """Test is_loaded returns False when model not loaded"""
        assert not fingpt_service.is_loaded()
    
    @patch('app.services.fingpt_service.AutoTokenizer')
    @patch('app.services.fingpt_service.AutoModelForCausalLM')
    @patch('app.services.fingpt_service.pipeline')
    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_pipeline, mock_model, mock_tokenizer, fingpt_service):
        """Test successful model loading"""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Load model
        result = await fingpt_service.load_model()
        
        # Assertions
        assert result is True
        assert fingpt_service.is_loaded()
        assert fingpt_service.tokenizer is not None
        assert fingpt_service.model is not None
        assert fingpt_service.pipeline is not None
        assert fingpt_service.load_time is not None
    
    @patch('app.services.fingpt_service.AutoTokenizer')
    @patch('app.services.fingpt_service.AutoModelForCausalLM')
    @pytest.mark.asyncio
    async def test_load_model_failure(self, mock_model, mock_tokenizer, fingpt_service):
        """Test model loading failure"""
        # Mock tokenizer to raise exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
        
        # Load model
        result = await fingpt_service.load_model()
        
        # Assertions
        assert result is False
        assert not fingpt_service.is_loaded()
    
    @pytest.mark.asyncio
    async def test_generate_text_not_loaded(self, fingpt_service):
        """Test text generation when model not loaded"""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await fingpt_service.generate_text("Test prompt")
    
    @patch('app.services.fingpt_service.AutoTokenizer')
    @patch('app.services.fingpt_service.AutoModelForCausalLM')
    @patch('app.services.fingpt_service.pipeline')
    @pytest.mark.asyncio
    async def test_generate_text_success(self, mock_pipeline, mock_model, mock_tokenizer, fingpt_service):
        """Test successful text generation"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{"generated_text": "Generated financial text"}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Load model first
        await fingpt_service.load_model()
        
        # Generate text
        result = await fingpt_service.generate_text("Test prompt")
        
        # Assertions
        assert isinstance(result, FinGPTResponse)
        assert result.prompt == "Test prompt"
        assert result.generated_text == "Generated financial text"
        assert result.processing_time > 0
        assert fingpt_service.inference_count == 1
    
    @pytest.mark.asyncio
    async def test_analyze_financial_text_not_loaded(self, fingpt_service):
        """Test financial analysis when model not loaded"""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await fingpt_service.analyze_financial_text("Test text")
    
    @patch('app.services.fingpt_service.AutoTokenizer')
    @patch('app.services.fingpt_service.AutoModelForCausalLM')
    @patch('app.services.fingpt_service.pipeline')
    @pytest.mark.asyncio
    async def test_analyze_financial_text_success(self, mock_pipeline, mock_model, mock_tokenizer, fingpt_service):
        """Test successful financial text analysis"""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{"generated_text": "Detailed financial analysis"}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Load model first
        await fingpt_service.load_model()
        
        # Analyze text
        result = await fingpt_service.analyze_financial_text("Test financial text", "general")
        
        # Assertions
        assert isinstance(result, dict)
        assert result["analysis_type"] == "general"
        assert result["original_text"] == "Test financial text"
        assert result["analysis"] == "Detailed financial analysis"
        assert result["processing_time"] > 0
    
    def test_get_model_info_not_loaded(self, fingpt_service):
        """Test get_model_info when model not loaded"""
        info = fingpt_service.get_model_info()
        assert info["model_loaded"] is False
    
    @patch('app.services.fingpt_service.AutoTokenizer')
    @patch('app.services.fingpt_service.AutoModelForCausalLM')
    @patch('app.services.fingpt_service.pipeline')
    @pytest.mark.asyncio
    async def test_get_model_info_loaded(self, mock_pipeline, mock_model, mock_tokenizer, fingpt_service):
        """Test get_model_info when model loaded"""
        # Setup mocks and load model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        await fingpt_service.load_model()
        
        # Get model info
        info = fingpt_service.get_model_info()
        
        # Assertions
        assert info["model_loaded"] is True
        assert "model_name" in info
        assert "device" in info
        assert "load_time" in info
        assert info["inference_count"] == 0
    
    @patch('app.services.fingpt_service.AutoTokenizer')
    @patch('app.services.fingpt_service.AutoModelForCausalLM')
    @patch('app.services.fingpt_service.pipeline')
    @pytest.mark.asyncio
    async def test_health_check_loaded(self, mock_pipeline, mock_model, mock_tokenizer, fingpt_service):
        """Test health check when model loaded"""
        # Setup mocks and load model
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token_id = 0
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{"generated_text": "Test output"}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        await fingpt_service.load_model()
        
        # Health check
        health = await fingpt_service.health_check(deep_check=True)
        
        # Assertions
        assert health["model_loaded"] is True
        assert "device" in health
        assert "load_time" in health
        assert health["test_generation_successful"] is True
        assert "test_generation_time" in health
    
    def test_health_check_not_loaded(self, fingpt_service):
        """Test health check when model not loaded"""
        health = asyncio.run(fingpt_service.health_check())
        assert health["model_loaded"] is False
        assert health["device"] is None


class TestFinGPTParameterValidation:
    """Test parameter validation for FinGPT requests"""
    
    def test_valid_analysis_types(self):
        """Test valid analysis types"""
        from app.models.requests import FinGPTAnalysisRequest
        
        valid_types = ["general", "sentiment", "forecast", "risk"]
        for analysis_type in valid_types:
            request = FinGPTAnalysisRequest(
                text="Test text",
                analysis_type=analysis_type
            )
            assert request.analysis_type == analysis_type
    
    def test_invalid_analysis_type(self):
        """Test invalid analysis type"""
        from app.models.requests import FinGPTAnalysisRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            FinGPTAnalysisRequest(
                text="Test text",
                analysis_type="invalid_type"
            )
    
    def test_generation_parameter_validation(self):
        """Test generation parameter validation"""
        from app.models.requests import FinGPTGenerationRequest
        from pydantic import ValidationError
        
        # Valid request
        request = FinGPTGenerationRequest(
            prompt="Test prompt",
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        assert request.max_length == 100
        assert request.temperature == 0.7
        
        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            FinGPTGenerationRequest(
                prompt="Test prompt",
                temperature=3.0  # > 2.0
            )
        
        # Invalid max_length (too small)
        with pytest.raises(ValidationError):
            FinGPTGenerationRequest(
                prompt="Test prompt",
                max_length=10  # < 50
            )