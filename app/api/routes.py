"""
API routes for FinBERT sentiment analysis
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from loguru import logger

from ..models.requests import (
    SentimentRequest, BulkSentimentRequest, HealthCheckRequest,
    FinGPTGenerationRequest, FinGPTAnalysisRequest, CombinedAnalysisRequest
)
from ..models.responses import (
    SentimentResponse, BulkSentimentResponse, ErrorResponse, 
    HealthCheckResponse, ModelInfoResponse, FinGPTResponse,
    FinGPTAnalysisResponse, CombinedAnalysisResponse
)
from ..services.finbert_service import finbert_service
from ..services.fingpt_service import fingpt_service
from ..core.config import get_settings
from .. import __version__

settings = get_settings()
router = APIRouter()


@router.post(
    "/analyze",
    response_model=SentimentResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze Financial Sentiment",
    description="Analyze the sentiment of financial text using FinBERT model. "
                "Supports single text or multiple texts in a single request."
)
async def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    Analyze sentiment of financial text(s)
    
    - **text**: Single text string or list of text strings to analyze
    - **return_probabilities**: Include probability scores for all sentiment classes
    - **batch_size**: Batch size for processing multiple texts (1-100)
    
    Returns sentiment analysis results with confidence scores.
    """
    try:
        start_time = time.time()
        
        if not finbert_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Please try again later."
            )
        
        # Perform sentiment analysis
        results = await finbert_service.analyze_sentiment(
            text=request.text,
            return_probabilities=request.return_probabilities,
            batch_size=request.batch_size
        )
        
        processing_time = time.time() - start_time
        
        # Prepare metadata
        metadata = {
            "model_name": settings.model_name,
            "batch_size_used": request.batch_size or settings.batch_size,
            "return_probabilities": request.return_probabilities
        }
        
        if isinstance(request.text, list):
            metadata["text_count"] = len(request.text)
        
        return SentimentResponse(
            data=results,
            metadata=metadata,
            total_processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/analyze/bulk",
    response_model=BulkSentimentResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Bulk Financial Sentiment Analysis",
    description="Analyze sentiment of multiple financial texts efficiently in batches. "
                "Optimized for processing large numbers of texts."
)
async def bulk_analyze_sentiment(request: BulkSentimentRequest) -> BulkSentimentResponse:
    """
    Perform bulk sentiment analysis on multiple texts
    
    - **texts**: List of text strings to analyze (up to 1000 texts)
    - **return_probabilities**: Include probability scores for all sentiment classes
    - **batch_size**: Batch size for processing (1-100, default: 32)
    
    Returns bulk sentiment analysis results with statistics.
    """
    try:
        start_time = time.time()
        
        if not finbert_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Please try again later."
            )
        
        # Perform bulk sentiment analysis
        results = await finbert_service.analyze_sentiment(
            text=request.texts,
            return_probabilities=request.return_probabilities,
            batch_size=request.batch_size
        )
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        sentiment_counts = {}
        confidence_scores = []
        
        for result in results:
            sentiment = result.sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidence_scores.append(result.confidence)
        
        statistics = {
            "sentiment_distribution": sentiment_counts,
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores),
            "processing_rate": len(request.texts) / processing_time,
            "texts_per_second": len(request.texts) / processing_time
        }
        
        return BulkSentimentResponse(
            data=results,
            statistics=statistics,
            total_processing_time=processing_time,
            processed_count=len(request.texts)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk_analyze_sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the FinBERT API service and model."
)
async def health_check(deep_check: bool = False) -> HealthCheckResponse:
    """
    Perform health check of the API service
    
    - **deep_check**: Perform deep health check including model inference test
    
    Returns detailed health status information.
    """
    try:
        import psutil
        import platform
        
        # Basic health info
        model_info = finbert_service.get_model_info()
        
        # System information
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent
        }
        
        # Determine health status
        if not finbert_service.is_loaded():
            status_value = "unhealthy"
        else:
            status_value = "healthy"
        
        response = HealthCheckResponse(
            status=status_value,
            api_version=__version__,
            model_loaded=finbert_service.is_loaded(),
            system_info=system_info,
            model_info=model_info
        )
        
        # Perform deep check if requested
        if deep_check:
            health_check_result = await finbert_service.health_check(deep_check=True)
            response.performance_metrics = health_check_result
            
            # Update status based on deep check
            if not health_check_result.get("test_inference_successful", False):
                response.status = "degraded"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            api_version=__version__,
            model_loaded=False,
            system_info={"error": str(e)}
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get detailed information about the loaded FinBERT model."
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about the loaded model
    
    Returns detailed model information including configuration and performance stats.
    """
    try:
        model_info = finbert_service.get_model_info()
        
        config = {
            "model_name": settings.model_name,
            "max_sequence_length": settings.max_sequence_length,
            "batch_size": settings.batch_size,
            "device": model_info.get("device"),
            "load_time": model_info.get("load_time")
        }
        
        capabilities = [
            "financial_sentiment_analysis",
            "batch_processing",
            "probability_scores",
            "text_classification"
        ]
        
        performance_stats = None
        if model_info.get("inference_count", 0) > 0:
            performance_stats = {
                "total_inferences": model_info.get("inference_count"),
                "total_inference_time": model_info.get("total_inference_time"),
                "average_inference_time": model_info.get("average_inference_time")
            }
        
        return ModelInfoResponse(
            model_name=settings.model_name,
            model_loaded=model_info.get("model_loaded", False),
            config=config,
            capabilities=capabilities,
            performance_stats=performance_stats
        )
        
    except Exception as e:
        logger.error(f"Error in get_model_info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get(
    "/",
    summary="API Root",
    description="Basic API information and available endpoints."
)
async def root() -> Dict[str, Any]:
    """
    API root endpoint with basic information
    """
    return {
        "name": "FinBERT API",
        "version": __version__,
        "description": "Financial sentiment analysis using FinBERT model",
        "model": settings.model_name,
        "model_loaded": finbert_service.is_loaded(),
        "endpoints": {
            "analyze": "/analyze",
            "bulk_analyze": "/analyze/bulk", 
            "health": "/health",
            "model_info": "/model/info",
            "fingpt_generate": "/fingpt/generate",
            "fingpt_analyze": "/fingpt/analyze",
            "combined_analyze": "/combined/analyze",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@router.post(
    "/fingpt/generate",
    response_model=FinGPTResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate Financial Text",
    description="Generate financial text using FinGPT model. "
                "Supports customizable generation parameters for different use cases."
)
async def generate_financial_text(request: FinGPTGenerationRequest) -> FinGPTResponse:
    """
    Generate financial text using FinGPT model
    
    - **prompt**: Input text prompt for generation
    - **max_length**: Maximum length of generated text (50-1000)
    - **temperature**: Randomness in generation (0.0-2.0)
    - **top_p**: Nucleus sampling parameter (0.0-1.0)
    - **top_k**: Top-k sampling parameter (1-100)
    - **do_sample**: Whether to use sampling vs greedy decoding
    - **num_return_sequences**: Number of sequences to generate (1-5)
    
    Returns generated financial text with metadata.
    """
    try:
        start_time = time.time()
        
        if not fingpt_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FinGPT model is not loaded. Please try again later."
            )
        
        # Generate text
        result = await fingpt_service.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            num_return_sequences=request.num_return_sequences
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_financial_text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/fingpt/analyze",
    response_model=FinGPTAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze Financial Text with FinGPT",
    description="Analyze financial text using FinGPT model for various types of analysis. "
                "Supports general analysis, sentiment analysis, forecasting, and risk analysis."
)
async def analyze_financial_text_fingpt(request: FinGPTAnalysisRequest) -> FinGPTAnalysisResponse:
    """
    Analyze financial text using FinGPT model
    
    - **text**: Text to analyze (up to 5000 characters)
    - **analysis_type**: Type of analysis to perform
      - general: General financial analysis
      - sentiment: Sentiment analysis with reasoning
      - forecast: Market forecasting based on the text
      - risk: Risk analysis and identification
    
    Returns detailed financial analysis with insights.
    """
    try:
        if not fingpt_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FinGPT model is not loaded. Please try again later."
            )
        
        # Perform analysis
        result = await fingpt_service.analyze_financial_text(
            text=request.text,
            analysis_type=request.analysis_type
        )
        
        return FinGPTAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_financial_text_fingpt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/combined/analyze",
    response_model=CombinedAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Combined FinBERT + FinGPT Analysis",
    description="Perform comprehensive financial analysis using both FinBERT for sentiment "
                "and FinGPT for detailed text analysis. Provides combined insights."
)
async def combined_financial_analysis(request: CombinedAnalysisRequest) -> CombinedAnalysisResponse:
    """
    Perform combined analysis using both FinBERT and FinGPT models
    
    - **text**: Text to analyze (up to 5000 characters)
    - **include_sentiment**: Include FinBERT sentiment analysis
    - **include_generation**: Include FinGPT text analysis
    - **analysis_type**: Type of FinGPT analysis to perform
    - **return_probabilities**: Return sentiment probability scores
    
    Returns comprehensive analysis combining both models' insights.
    """
    try:
        start_time = time.time()
        
        # Check model availability
        if request.include_sentiment and not finbert_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FinBERT model is not loaded. Please try again later."
            )
        
        if request.include_generation and not fingpt_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FinGPT model is not loaded. Please try again later."
            )
        
        sentiment_result = None
        generation_result = None
        
        # Perform FinBERT sentiment analysis if requested
        if request.include_sentiment:
            sentiment_result = await finbert_service.analyze_sentiment(
                text=request.text,
                return_probabilities=request.return_probabilities
            )
        
        # Perform FinGPT analysis if requested
        if request.include_generation:
            fingpt_result = await fingpt_service.analyze_financial_text(
                text=request.text,
                analysis_type=request.analysis_type
            )
            generation_result = FinGPTAnalysisResponse(**fingpt_result)
        
        # Generate combined insights
        combined_insights = {}
        
        if sentiment_result and generation_result:
            combined_insights = {
                "sentiment_alignment": {
                    "finbert_sentiment": sentiment_result.sentiment,
                    "finbert_confidence": sentiment_result.confidence,
                    "fingpt_analysis_type": request.analysis_type,
                    "analysis_length": len(generation_result.analysis)
                },
                "key_insights": {
                    "sentiment_detected": sentiment_result.sentiment,
                    "analysis_focus": request.analysis_type,
                    "combined_recommendation": _generate_combined_recommendation(
                        sentiment_result, generation_result
                    )
                }
            }
        elif sentiment_result:
            combined_insights = {
                "sentiment_only": True,
                "sentiment": sentiment_result.sentiment,
                "confidence": sentiment_result.confidence
            }
        elif generation_result:
            combined_insights = {
                "analysis_only": True,
                "analysis_type": request.analysis_type,
                "analysis_length": len(generation_result.analysis)
            }
        
        total_time = time.time() - start_time
        
        return CombinedAnalysisResponse(
            sentiment_analysis=sentiment_result,
            text_generation=generation_result,
            combined_insights=combined_insights,
            total_processing_time=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in combined_financial_analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


def _generate_combined_recommendation(sentiment_result, generation_result) -> str:
    """Generate a combined recommendation based on both analyses"""
    sentiment = sentiment_result.sentiment
    confidence = sentiment_result.confidence
    
    base_recommendation = ""
    
    if sentiment == "positive" and confidence > 0.8:
        base_recommendation = "Strong positive sentiment detected with high confidence. "
    elif sentiment == "positive":
        base_recommendation = "Positive sentiment detected. "
    elif sentiment == "negative" and confidence > 0.8:
        base_recommendation = "Strong negative sentiment detected with high confidence. "
    elif sentiment == "negative":
        base_recommendation = "Negative sentiment detected. "
    else:
        base_recommendation = "Neutral sentiment detected. "
    
    base_recommendation += f"Detailed {generation_result.analysis_type} analysis provides additional context for decision-making."
    
    return base_recommendation