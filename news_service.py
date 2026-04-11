"""
News service for fetching and analyzing financial news with Groq LLM.
Handles news fetching, sentiment analysis, and trade signal extraction.
"""
import logging
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# Try to import groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq library not available. Install with: pip install groq")


class NewsService:
    """Service for analyzing financial news with Groq LLM."""
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 600):
        """
        Initialize NewsService.
        
        Args:
            api_key: Groq API key. If None, will attempt to load from environment/file
            cache_ttl: Cache time-to-live in seconds (10 min default)
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_times = {}
        self.client = None
        
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Groq client: {str(e)}")
        elif not GROQ_AVAILABLE:
            logger.warning("Groq library not available - news analysis will return defaults")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self.cache_times:
            return False
        age = time.time() - self.cache_times[key]
        return age < self.cache_ttl
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached news analysis."""
        if self._is_cache_valid(key):
            return self.cache[key]
        return None
    
    def _set_cache(self, key: str, value: Dict):
        """Store news analysis in cache."""
        self.cache[key] = value
        self.cache_times[key] = time.time()
    
    def get_safe_default_response(self) -> Dict:
        """Return safe default response when analysis is unavailable."""
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "signals": [],
            "summary": "News analysis unavailable",
            "timestamp": datetime.now().isoformat(),
            "status": "unavailable"
        }
    
    def analyze_news(self, sector: str, news_items: Optional[List[str]] = None) -> Dict:
        """
        Analyze financial news sentiment and extract trade signals.
        
        Args:
            sector: Sector name (e.g., 'tech', 'banks', 'mining')
            news_items: Optional list of news headlines/summaries to analyze
            
        Returns:
            Dictionary with keys:
                - sentiment: 'bullish', 'neutral', or 'bearish'
                - sentiment_score: Float between -1 (bearish) and 1 (bullish)
                - signals: List of identified trade signals
                - summary: Text summary of analysis
                - timestamp: ISO timestamp
                - status: 'success' or error description
        """
        cache_key = f"news_{sector}"
        
        # Check cache first
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for news analysis: {sector}")
            return cached
        
        # If no client or no news items, return safe default
        if not self.client or not news_items:
            logger.warning(f"No Groq client or news items for {sector}")
            result = self.get_safe_default_response()
            result["status"] = "default"
            # Cache the default response too
            self._set_cache(cache_key, result)
            return result
        
        try:
            # Prepare news text for analysis
            news_text = "\n".join([f"- {item}" for item in news_items[:5]])  # Top 5
            
            # Build prompt for Groq
            prompt = f"""
Analyze the following financial news for the {sector} sector. 
Provide sentiment (bullish/neutral/bearish) and any trade signals.

News:
{news_text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "sentiment": "bullish|neutral|bearish",
    "sentiment_score": <float -1 to 1>,
    "signals": [<list of identified signals>],
    "summary": "<brief analysis>"
}}
"""
            
            # Call Groq API
            message = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            response_text = message.choices[0].message.content.strip()
            
            # Remove markdown code block if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            analysis = json.loads(response_text)
            
            # Validate response structure
            result = {
                "sentiment": analysis.get("sentiment", "neutral").lower(),
                "sentiment_score": float(analysis.get("sentiment_score", 0.0)),
                "signals": analysis.get("signals", []),
                "summary": analysis.get("summary", ""),
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            # Cache result
            self._set_cache(cache_key, result)
            logger.info(f"News analysis successful for {sector}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in news analysis: {str(e)}")
            result = self.get_safe_default_response()
            result["status"] = "parse_error"
            return result
        except Exception as e:
            logger.error(f"Error analyzing news for {sector}: {str(e)}")
            result = self.get_safe_default_response()
            result["status"] = str(type(e).__name__)
            return result
    
    def extract_signals(self, sector: str, news_text: Optional[str] = None) -> List[str]:
        """
        Extract specific trade signals from news.
        
        Args:
            sector: Sector for context
            news_text: Concatenated news text
            
        Returns:
            List of identified signals
        """
        if not news_text:
            return []
        
        # Simple keyword-based signal extraction (fallback if Groq unavailable)
        signals = []
        
        bullish_keywords = ['surge', 'rally', 'jump', 'soar', 'beat', 'upgrade', 'buy', 'strong']
        bearish_keywords = ['crash', 'plunge', 'fall', 'downgrade', 'sell', 'weak', 'miss']
        
        text_lower = news_text.lower()
        
        for keyword in bullish_keywords:
            if keyword in text_lower:
                signals.append(f"Bullish signal: {keyword.upper()}")
        
        for keyword in bearish_keywords:
            if keyword in text_lower:
                signals.append(f"Bearish signal: {keyword.upper()}")
        
        return signals[:5]  # Top 5 signals
    
    def get_news_status(self) -> Dict:
        """Get status of news service."""
        return {
            "groq_available": GROQ_AVAILABLE,
            "client_ready": self.client is not None,
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat()
        }


def create_news_service(api_key: Optional[str] = None, cache_ttl: int = 600) -> NewsService:
    """Factory function to create NewsService instance."""
    return NewsService(api_key=api_key, cache_ttl=cache_ttl)
