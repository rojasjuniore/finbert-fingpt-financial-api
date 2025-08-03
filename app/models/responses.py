"""
Response models for the FinBERT + FinGPT API
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class SentimentResult(BaseModel):
    """Individual sentiment analysis result"""
    text: str = Field(..., description="Original text that was analyzed")
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative/neutral)")
    confidence: float = Field(..., description="Confidence score for the prediction")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probability scores for all sentiment classes"
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    data: Union[SentimentResult, List[SentimentResult]] = Field(
        ...,
        description="Sentiment analysis results"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadata about the analysis"
    )
    total_processing_time: float = Field(
        ...,
        description="Total time taken for processing (seconds)"
    )


class BulkSentimentResponse(BaseModel):
    """Response model for bulk sentiment analysis"""
    data: List[SentimentResult] = Field(
        ...,
        description="List of sentiment analysis results"
    )
    statistics: Dict[str, Any] = Field(
        ...,
        description="Statistics about the bulk analysis"
    )
    total_processing_time: float = Field(
        ...,
        description="Total time taken for processing (seconds)"
    )
    processed_count: int = Field(
        ...,
        description="Number of texts processed"
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Health status (healthy/unhealthy/degraded)")
    api_version: Optional[str] = Field(default=None, description="API version")
    model_loaded: Optional[bool] = Field(default=None, description="Whether models are loaded")
    system_info: Optional[Dict[str, Any]] = Field(default=None, description="System information")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Model information")
    performance_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance metrics from deep check"
    )


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="Name of the loaded model")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    config: Dict[str, Any] = Field(..., description="Model configuration")
    capabilities: List[str] = Field(..., description="Model capabilities")
    performance_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance statistics"
    )


class FinGPTResponse(BaseModel):
    """Response model for FinGPT text generation"""
    generated_text: List[str] = Field(
        ...,
        description="Generated text sequences"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Generation metadata"
    )
    total_processing_time: float = Field(
        ...,
        description="Total time taken for generation (seconds)"
    )


class FinGPTAnalysisResponse(BaseModel):
    """Response model for FinGPT text analysis"""
    analysis: str = Field(
        ...,
        description="Generated analysis text"
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis performed"
    )
    key_insights: Optional[List[str]] = Field(
        default=None,
        description="Key insights extracted from the analysis"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the analysis"
    )
    total_processing_time: float = Field(
        ...,
        description="Total time taken for analysis (seconds)"
    )


class CombinedAnalysisResponse(BaseModel):
    """Response model for combined FinBERT + FinGPT analysis"""
    sentiment_analysis: Optional[SentimentResult] = Field(
        default=None,
        description="FinBERT sentiment analysis result"
    )
    text_generation: Optional[FinGPTAnalysisResponse] = Field(
        default=None,
        description="FinGPT analysis result"
    )
    combined_insights: Dict[str, Any] = Field(
        ...,
        description="Combined insights from both models"
    )
    total_processing_time: float = Field(
        ...,
        description="Total time taken for combined analysis (seconds)"
    )


class ErrorResponse(BaseModel):
    """Response model for errors"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp")


class SentimentScore(BaseModel):
    """Individual sentiment score"""
    sentiment: str = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Class probabilities")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")

