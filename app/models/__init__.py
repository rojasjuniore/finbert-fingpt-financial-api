"""
Models package for FinBERT + FinGPT API
"""

from .requests import *
from .responses import *

__all__ = [
    # Request models
    "SentimentRequest",
    "BulkSentimentRequest", 
    "HealthCheckRequest",
    "FinGPTGenerationRequest",
    "FinGPTAnalysisRequest",
    "CombinedAnalysisRequest",
    "AnalysisType",
    
    # Response models
    "SentimentResult",
    "SentimentResponse",
    "BulkSentimentResponse",
    "HealthCheckResponse",
    "ModelInfoResponse",
    "FinGPTResponse",
    "FinGPTAnalysisResponse",
    "CombinedAnalysisResponse",
    "ErrorResponse"
]