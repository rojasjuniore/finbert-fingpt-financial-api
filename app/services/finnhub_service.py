"""
Finnhub API service for real-time financial data integration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import finnhub
from loguru import logger

from ..core.config import get_settings

settings = get_settings()


class FinnhubService:
    """Service for accessing real-time financial data from Finnhub"""
    
    def __init__(self):
        self.client = None
        self.is_configured = False
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Finnhub client if API key is available"""
        try:
            if settings.finnhub_api_key:
                self.client = finnhub.Client(api_key=settings.finnhub_api_key)
                self.is_configured = True
                logger.info("Finnhub client initialized successfully")
            else:
                logger.warning("Finnhub API key not configured - real-time data features disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Finnhub client: {str(e)}")
            self.is_configured = False
    
    def is_available(self) -> bool:
        """Check if Finnhub service is available"""
        return self.is_configured and self.client is not None
    
    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information"""
        if not self.is_available():
            return None
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            profile = await loop.run_in_executor(
                None, 
                lambda: self.client.company_profile2(symbol=symbol.upper())
            )
            return profile
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {str(e)}")
            return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote data"""
        if not self.is_available():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(
                None,
                lambda: self.client.quote(symbol.upper())
            )
            return quote
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return None
    
    async def get_company_news(self, symbol: str, days_back: int = 7) -> Optional[List[Dict[str, Any]]]:
        """Get recent company news"""
        if not self.is_available():
            return None
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            loop = asyncio.get_event_loop()
            news = await loop.run_in_executor(
                None,
                lambda: self.client.company_news(
                    symbol.upper(),
                    _from=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d')
                )
            )
            
            # Limit to most recent 10 articles
            return news[:10] if news else []
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return None
    
    async def get_financial_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get key financial metrics"""
        if not self.is_available():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                None,
                lambda: self.client.company_basic_financials(symbol.upper(), 'all')
            )
            return metrics
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
            return None
    
    async def get_earnings(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get earnings data"""
        if not self.is_available():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            earnings = await loop.run_in_executor(
                None,
                lambda: self.client.company_earnings(symbol.upper(), limit=4)
            )
            return earnings
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {str(e)}")
            return None
    
    async def get_market_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market sentiment indicators"""
        if not self.is_available():
            return None
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get recommendation trends
            recommendations = await loop.run_in_executor(
                None,
                lambda: self.client.recommendation_trends(symbol.upper())
            )
            
            # Get insider sentiment
            insider_sentiment = await loop.run_in_executor(
                None,
                lambda: self.client.stock_insider_sentiment(symbol.upper(), '2023-01-01', '2024-12-31')
            )
            
            return {
                'recommendations': recommendations[:1] if recommendations else None,  # Most recent
                'insider_sentiment': insider_sentiment.get('data', [])[:3] if insider_sentiment else []  # Last 3 months
            }
        except Exception as e:
            logger.error(f"Error fetching market sentiment for {symbol}: {str(e)}")
            return None
    
    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive financial data for enhanced analysis"""
        if not self.is_available():
            return {
                'available': False,
                'message': 'Finnhub service not configured'
            }
        
        logger.info(f"Fetching comprehensive data for {symbol}")
        
        # Fetch all data concurrently
        tasks = {
            'profile': self.get_company_profile(symbol),
            'quote': self.get_quote(symbol),
            'news': self.get_company_news(symbol, days_back=7),
            'metrics': self.get_financial_metrics(symbol),
            'earnings': self.get_earnings(symbol),
            'sentiment': self.get_market_sentiment(symbol)
        }
        
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"Error fetching {key} for {symbol}: {str(e)}")
                results[key] = None
        
        # Add metadata  
        results['symbol'] = symbol.upper()
        results['timestamp'] = time.time()  # Use Unix timestamp instead of datetime
        results['available'] = True
        
        return results
    
    def extract_key_insights(self, data: Dict[str, Any]) -> List[str]:
        """Extract key insights from Finnhub data for FinGPT enhancement"""
        insights = []
        
        if not data.get('available'):
            return insights
        
        try:
            # Company profile insights
            if profile := data.get('profile'):
                if name := profile.get('name'):
                    insights.append(f"Company: {name}")
                if industry := profile.get('finnhubIndustry'):
                    insights.append(f"Industry: {industry}")
                if market_cap := profile.get('marketCapitalization'):
                    insights.append(f"Market Cap: ${market_cap:,.0f}M")
            
            # Quote insights
            if quote := data.get('quote'):
                if current_price := quote.get('c'):
                    previous_close = quote.get('pc', current_price)
                    change = current_price - previous_close
                    change_pct = (change / previous_close) * 100 if previous_close else 0
                    insights.append(f"Current Price: ${current_price:.2f} ({change_pct:+.2f}%)")
                
                if high := quote.get('h'):
                    low = quote.get('l')
                    insights.append(f"Day Range: ${low:.2f} - ${high:.2f}")
            
            # Financial metrics insights
            if metrics := data.get('metrics'):
                metric_data = metrics.get('metric', {})
                if pe_ratio := metric_data.get('peBasicExclExtraTTM'):
                    insights.append(f"P/E Ratio: {pe_ratio:.2f}")
                if roe := metric_data.get('roe'):
                    insights.append(f"ROE: {roe:.2f}%")
                if debt_equity := metric_data.get('totalDebt/totalEquityQuarterly'):
                    insights.append(f"Debt/Equity: {debt_equity:.2f}")
            
            # News insights
            if news := data.get('news'):
                if news and len(news) > 0:
                    insights.append(f"Recent News: {len(news)} articles in past week")
                    # Add most recent headline
                    if news[0].get('headline'):
                        insights.append(f"Latest: {news[0]['headline'][:100]}...")
            
            # Earnings insights
            if earnings := data.get('earnings'):
                if earnings and len(earnings) > 0:
                    latest_earnings = earnings[0]
                    if actual := latest_earnings.get('actual'):
                        estimate = latest_earnings.get('estimate', actual)
                        surprise = actual - estimate
                        insights.append(f"Latest EPS: ${actual:.2f} (surprise: ${surprise:+.2f})")
            
            # Recommendation insights
            if sentiment := data.get('sentiment'):
                if recs := sentiment.get('recommendations'):
                    if recs and len(recs) > 0:
                        rec_data = recs[0]
                        buy_count = rec_data.get('buy', 0)
                        hold_count = rec_data.get('hold', 0)
                        sell_count = rec_data.get('sell', 0)
                        total = buy_count + hold_count + sell_count
                        if total > 0:
                            buy_pct = (buy_count / total) * 100
                            insights.append(f"Analyst Sentiment: {buy_pct:.0f}% Buy recommendations")
        
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
        
        return insights


# Global service instance
finnhub_service = FinnhubService()