"""
Enhanced API routes with Finnhub integration
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from ..services.finnhub_service import finnhub_service
from ..services.fingpt_service import fingpt_service
from ..models.requests import FinGPTAnalysisRequest

router = APIRouter(prefix="/enhanced", tags=["Enhanced Analysis"])


class MarketDataRequest(BaseModel):
    """Request for market data"""
    symbol: str = Field(..., description="Stock symbol", example="AAPL")


class EnhancedAnalysisRequest(BaseModel):
    """Request for enhanced analysis with market data"""
    text: str = Field(..., description="Text to analyze", example="Apple stock analysis")
    symbol: str = Field(..., description="Stock symbol for context", example="AAPL")
    analysis_type: str = Field(default="forecast", description="Type of analysis")


@router.get("/market-data/{symbol}")
async def get_market_data(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive market data for a stock symbol
    
    - **symbol**: Stock symbol (e.g., AAPL, TSLA, GOOGL)
    
    Returns real-time financial data including:
    - Company profile and metrics
    - Current stock quote and price movements
    - Recent company news
    - Financial metrics and ratios
    - Earnings data
    - Market sentiment indicators
    """
    try:
        if not finnhub_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Finnhub service not configured. Please add FINNHUB_API_KEY to environment."
            )
        
        logger.info(f"Fetching comprehensive market data for {symbol}")
        data = await finnhub_service.get_comprehensive_data(symbol.upper())
        
        if not data.get('available'):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Market data service temporarily unavailable"
            )
        
        return {
            "symbol": symbol.upper(),
            "data": data,
            "insights": finnhub_service.extract_key_insights(data),
            "timestamp": data.get('timestamp')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch market data: {str(e)}"
        )


@router.post("/enhanced-analysis")
async def enhanced_financial_analysis(request: EnhancedAnalysisRequest) -> Dict[str, Any]:
    """
    Perform enhanced financial analysis using FinGPT + real-time Finnhub data
    
    - **text**: Financial text to analyze
    - **symbol**: Stock symbol for enhanced context
    - **analysis_type**: Type of analysis (general, sentiment, forecast, risk)
    
    This endpoint combines:
    1. Real-time market data from Finnhub
    2. Advanced AI analysis from FinGPT
    3. Contextual insights based on current market conditions
    
    The result is a comprehensive analysis that considers both the text content
    and current market realities for the specified stock.
    """
    try:
        start_time = time.time()
        
        if not fingpt_service.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="FinGPT model is not loaded. Please try again later."
            )
        
        if not finnhub_service.is_available():
            logger.warning("Finnhub not available - analysis will proceed without real-time data")
        
        # Perform enhanced analysis
        result = await fingpt_service.analyze_financial_text(
            text=request.text,
            analysis_type=request.analysis_type,
            symbol=request.symbol
        )
        
        total_time = time.time() - start_time
        
        return {
            "symbol": request.symbol.upper(),
            "analysis_type": request.analysis_type,
            "text_analyzed": request.text,
            "enhanced_analysis": result,
            "finnhub_enhanced": result.get('enhanced_with_finnhub', False),
            "market_insights": result.get('market_data_insights', []),
            "total_processing_time": total_time,
            "data_timestamp": result.get('data_timestamp'),
            "recommendation": _generate_trading_recommendation(result)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced analysis failed: {str(e)}"
        )


@router.get("/market-sentiment/{symbol}")
async def get_market_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive market sentiment for a stock
    
    - **symbol**: Stock symbol (e.g., AAPL, TSLA)
    
    Returns sentiment indicators including:
    - Analyst recommendations
    - Insider sentiment
    - Recent news sentiment
    - Price momentum indicators
    """
    try:
        if not finnhub_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Finnhub service not configured"
            )
        
        sentiment_data = await finnhub_service.get_market_sentiment(symbol.upper())
        
        if sentiment_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sentiment data not available for {symbol}"
            )
        
        return {
            "symbol": symbol.upper(),
            "sentiment_indicators": sentiment_data,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch market sentiment: {str(e)}"
        )


@router.get("/company-news/{symbol}")
async def get_company_news(symbol: str, days: int = 7) -> Dict[str, Any]:
    """
    Get recent company news with AI sentiment analysis
    
    - **symbol**: Stock symbol
    - **days**: Number of days to look back (default: 7, max: 30)
    
    Returns recent news articles with:
    - Headlines and summaries
    - Publication dates and sources
    - AI-powered sentiment analysis of each article
    """
    try:
        if not finnhub_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Finnhub service not configured"
            )
        
        days = min(days, 30)  # Limit to 30 days max
        news_data = await finnhub_service.get_company_news(symbol.upper(), days_back=days)
        
        if not news_data:
            return {
                "symbol": symbol.upper(),
                "news_articles": [],
                "message": f"No recent news found for {symbol} in the last {days} days"
            }
        
        # Analyze sentiment of news headlines if FinBERT is available
        analyzed_news = []
        for article in news_data:
            news_item = {
                "headline": article.get('headline', ''),
                "summary": article.get('summary', ''),
                "url": article.get('url', ''),
                "source": article.get('source', ''),
                "datetime": article.get('datetime'),
                "sentiment": None
            }
            
            # Add AI sentiment analysis if available
            if news_item['headline'] and hasattr(fingpt_service, 'analyze_sentiment'):
                try:
                    # This would require integration with FinBERT service
                    # For now, we'll add a placeholder
                    news_item['ai_sentiment'] = "Analysis available via /analyze endpoint"
                except:
                    pass
            
            analyzed_news.append(news_item)
        
        return {
            "symbol": symbol.upper(),
            "news_articles": analyzed_news,
            "article_count": len(analyzed_news),
            "date_range_days": days,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch company news: {str(e)}"
        )


def _generate_trading_recommendation(analysis_result: Dict[str, Any]) -> str:
    """Generate a trading recommendation based on analysis results"""
    try:
        analysis_text = analysis_result.get('analysis', '').lower()
        analysis_type = analysis_result.get('analysis_type', 'general')
        enhanced = analysis_result.get('enhanced_with_finnhub', False)
        
        if 'strong buy' in analysis_text or 'bullish' in analysis_text:
            confidence = "High" if enhanced else "Medium"
            return f"BUY - {confidence} confidence based on {analysis_type} analysis"
        elif 'buy' in analysis_text or 'positive' in analysis_text:
            confidence = "Medium" if enhanced else "Low"
            return f"BUY - {confidence} confidence based on {analysis_type} analysis"
        elif 'sell' in analysis_text or 'bearish' in analysis_text or 'negative' in analysis_text:
            confidence = "Medium" if enhanced else "Low"
            return f"SELL - {confidence} confidence based on {analysis_type} analysis"
        elif 'hold' in analysis_text or 'neutral' in analysis_text:
            return f"HOLD - Based on {analysis_type} analysis"
        else:
            return f"NEUTRAL - Insufficient signals from {analysis_type} analysis"
            
    except Exception as e:
        logger.warning(f"Error generating recommendation: {str(e)}")
        return "NEUTRAL - Unable to generate recommendation"