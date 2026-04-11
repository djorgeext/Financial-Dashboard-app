"""
Financial data fetcher module.
Handles fetching price data, calculating indicators, and caching results.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# In-memory cache with TTL
_cache = {}
_cache_times = {}
CACHE_TTL = 300  # 5 minutes


class DataFetcher:
    """Service for fetching and processing financial data."""
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize DataFetcher.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_times = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self.cache_times:
            return False
        age = time.time() - self.cache_times[key]
        return age < self.cache_ttl
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get value from cache if valid."""
        if self._is_cache_valid(key):
            return self.cache[key]
        return None
    
    def _set_cache(self, key: str, value: any):
        """Store value in cache with timestamp."""
        self.cache[key] = value
        self.cache_times[key] = time.time()
    
    def fetch_price_data(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Data period ('1mo', '3mo', '1y', etc.)
            interval: Candle interval ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with OHLCV data or None on error
        """
        cache_key = f"price_{ticker}_{interval}"
        
        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {ticker}")
            return cached
        
        try:
            logger.info(f"Fetching data for {ticker} (period={period}, interval={interval})")
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                prepost=False
            )
            
            if data is None or data.empty:
                logger.warning(f"No data fetched for {ticker}")
                return None
            
            # Ensure column names are lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Cache the result
            self._set_cache(cache_key, data)
            logger.info(f"Data fetched successfully for {ticker}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get the most recent close price for a ticker."""
        try:
            data = self.fetch_price_data(ticker, period="1d", interval="1d")
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
        return None
    
    def get_price_stats(self, ticker: str, days: int = 5) -> Optional[Dict]:
        """
        Get price statistics for a ticker.
        
        Args:
            ticker: Stock ticker
            days: Number of days to look back
            
        Returns:
            Dictionary with price stats or None on error
        """
        try:
            data = self.fetch_price_data(ticker, period=f"{days}d", interval="1d")
            if data is None or data.empty or len(data) < 2:
                return None
            
            close_prices = data['close']
            current = close_prices.iloc[-1]
            previous = close_prices.iloc[-2]
            high_24h = data['high'].iloc[-1]
            low_24h = data['low'].iloc[-1]
            
            change = current - previous
            change_pct = (change / previous) * 100 if previous != 0 else 0
            
            return {
                "current_price": float(current),
                "previous_close": float(previous),
                "change": float(change),
                "change_pct": float(change_pct),
                "high_24h": float(high_24h),
                "low_24h": float(low_24h),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating price stats for {ticker}: {str(e)}")
            return None
    
    def get_sector_summary(self, tickers: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Get price summary for multiple tickers in a sector.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to their stats
        """
        summary = {}
        for ticker in tickers:
            try:
                summary[ticker] = self.get_price_stats(ticker)
            except Exception as e:
                logger.error(f"Error in sector summary for {ticker}: {str(e)}")
                summary[ticker] = None
        
        return summary
    
    def prepare_features_for_model(
        self,
        ticker: str,
        sequence_length: int = 40,
        interval: str = "1d"
    ) -> Optional[np.ndarray]:
        """
        Fetch and prepare features for model inference.
        
        This is a simplified version that fetches OHLCV data.
        A full implementation would apply all feature engineering from utils.py.
        
        Args:
            ticker: Stock ticker
            sequence_length: Number of time steps to use
            interval: Candle interval
            
        Returns:
            Feature array shape (sequence_length, num_features) or None
        """
        try:
            # Fetch more data than needed to account for technical indicators
            data = self.fetch_price_data(
                ticker,
                period="3mo" if interval == "1d" else "1mo",
                interval=interval
            )
            
            if data is None or len(data) < sequence_length:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Get latest sequence_length rows
            data = data.tail(sequence_length)
            
            # Simple feature engineering (OHLCV-based)
            # Full version would use utils.py functions
            features = []
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    # Normalize by close price
                    normalized = (data[col] / data['close']) - 1
                    features.append(normalized.values)
            
            if 'volume' in data.columns:
                # Normalize volume (log scale)
                volume = np.log1p(data['volume'].values)
                features.append(volume)
            
            # Stack features: (sequence_length, num_features)
            if features:
                features_array = np.column_stack(features)
                logger.info(f"Features prepared for {ticker}: shape {features_array.shape}")
                return features_array
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing features for {ticker}: {str(e)}")
            return None


def create_data_fetcher(cache_ttl: int = 300) -> DataFetcher:
    """Factory function to create a DataFetcher instance."""
    return DataFetcher(cache_ttl=cache_ttl)
