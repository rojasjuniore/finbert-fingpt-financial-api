"""
Request models for the FinBERT + FinGPT API
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class AnalysisType(str, Enum):
    """Types of analysis available for FinGPT"""
    general = "general"
    sentiment = "sentiment"
    forecast = "forecast"
    risk = "risk"


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: Union[str, List[str]] = Field(
        ...,
        description="Text or list of texts to analyze",
        example="Apple stock is performing well this quarter"
    )
    return_probabilities: Optional[bool] = Field(
        default=False,
        description="Whether to return probability scores for all classes"
    )
    batch_size: Optional[int] = Field(
        default=32,
        ge=1,
        le=100,
        description="Batch size for processing multiple texts"
    )

    @validator("text")
    def validate_text(cls, v):
        if isinstance(v, str):
            if len(v.strip()) == 0:
                raise ValueError("Text cannot be empty")
            if len(v) > 10000:
                raise ValueError("Text too long (max 10000 characters)")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Text list cannot be empty")
            if len(v) > 1000:
                raise ValueError("Too many texts (max 1000)")
            for item in v:
                if not isinstance(item, str):
                    raise ValueError("All items in text list must be strings")
                if len(item.strip()) == 0:
                    raise ValueError("Text items cannot be empty")
                if len(item) > 10000:
                    raise ValueError("Text item too long (max 10000 characters)")
        return v


class BulkSentimentRequest(BaseModel):
    """Request model for bulk sentiment analysis"""
    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        example=["Stock market is up", "Economic uncertainty rises"]
    )
    return_probabilities: Optional[bool] = Field(
        default=False,
        description="Whether to return probability scores for all classes"
    )
    batch_size: Optional[int] = Field(
        default=32,
        ge=1,
        le=100,
        description="Batch size for processing"
    )

    @validator("texts")
    def validate_texts(cls, v):
        if len(v) == 0:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Too many texts (max 1000)")
        for text in v:
            if not isinstance(text, str):
                raise ValueError("All items must be strings")
            if len(text.strip()) == 0:
                raise ValueError("Text cannot be empty")
            if len(text) > 10000:
                raise ValueError("Text too long (max 10000 characters)")
        return v


class HealthCheckRequest(BaseModel):
    """Request model for health check"""
    deep_check: Optional[bool] = Field(
        default=False,
        description="Whether to perform deep health check with model inference"
    )


class FinGPTGenerationRequest(BaseModel):
    """Request model for FinGPT text generation"""
    prompt: str = Field(
        ...,
        description="Input prompt for text generation",
        example="The outlook for tech stocks in Q4 2024"
    )
    max_length: Optional[int] = Field(
        default=200,
        ge=50,
        le=1000,
        description="Maximum length of generated text"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Randomness in generation (0.0 = deterministic, 2.0 = very random)"
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling parameter"
    )
    do_sample: Optional[bool] = Field(
        default=True,
        description="Whether to use sampling vs greedy decoding"
    )
    num_return_sequences: Optional[int] = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of sequences to generate"
    )

    @validator("prompt")
    def validate_prompt(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 5000:
            raise ValueError("Prompt too long (max 5000 characters)")
        return v


class FinGPTAnalysisRequest(BaseModel):
    """Request model for FinGPT text analysis"""
    text: str = Field(
        ...,
        description="Text to analyze",
        example="The Federal Reserve announced interest rate changes"
    )
    analysis_type: Optional[AnalysisType] = Field(
        default=AnalysisType.general,
        description="Type of analysis to perform"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Stock symbol for enhanced analysis with real-time data (e.g., AAPL, TSLA)",
        example="AAPL"
    )

    @validator("text")
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(v) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        return v
    
    @validator("symbol")
    def validate_symbol(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError("Symbol cannot be empty if provided")
            if not v.isalpha():
                raise ValueError("Symbol must contain only letters")
            if len(v) > 10:
                raise ValueError("Symbol too long (max 10 characters)")
        return v.upper() if v else None


class CombinedAnalysisRequest(BaseModel):
    """Request model for combined FinBERT + FinGPT analysis"""
    text: str = Field(
        ...,
        description="Text to analyze with both models",
        example="Amazon reported strong quarterly earnings"
    )
    include_sentiment: Optional[bool] = Field(
        default=True,
        description="Whether to include FinBERT sentiment analysis"
    )
    include_generation: Optional[bool] = Field(
        default=True,
        description="Whether to include FinGPT text analysis"
    )
    analysis_type: Optional[AnalysisType] = Field(
        default=AnalysisType.general,
        description="Type of FinGPT analysis to perform"
    )
    return_probabilities: Optional[bool] = Field(
        default=False,
        description="Whether to return sentiment probability scores"
    )

    @validator("text")
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(v) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        return v
