"""
Configuration management for the financial dashboard.
Handles environment variables, API keys, and app settings.
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Base configuration class."""
    
    # Environment & Debug
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENV = os.getenv("ENV", "development")
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
    
    # API & Ports
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Data Configuration
    DATA_CACHE_TTL = int(os.getenv("DATA_CACHE_TTL", "300"))  # 5 minutes default
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # Model Configuration
    SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "40"))
    LSTM_HIDDEN = int(os.getenv("LSTM_HIDDEN", "128"))
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "3"))
    
    # News Configuration
    NEWS_CACHE_TTL = int(os.getenv("NEWS_CACHE_TTL", "600"))  # 10 minutes
    MAX_NEWS_PER_SECTOR = int(os.getenv("MAX_NEWS_PER_SECTOR", "10"))
    
    @classmethod
    def get_groq_api_key(cls) -> Optional[str]:
        """
        Retrieve Groq API key from environment or local file.
        
        Priority:
        1. GROQ_API_KEY environment variable (production)
        2. groq_api_key.txt file in BASE_DIR (development only)
        
        Returns:
            API key string or None if not found
            
        Raises:
            RuntimeError: If in production (ENV != 'development') and key not in env var
        """
        # Try environment variable first
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and api_key.strip():
            return api_key.strip()
        
        # Try local file (development only)
        if cls.ENV == "development":
            key_file = cls.BASE_DIR / "groq_api_key.txt"
            if key_file.exists():
                with open(key_file, "r") as f:
                    api_key = f.read().strip()
                    if api_key:
                        return api_key
        
        # Production check
        if cls.ENV != "development":
            raise RuntimeError(
                "GROQ_API_KEY environment variable is required in production mode"
            )
        
        return None
    
    # Sector configuration with model mappings
    SECTORS = {
        "tech": {
            "display_name": "Technology",
            "model_daily": "tech_us_model.pth",
            "model_hourly": "tech_us_model_hourly.pth",
            "tickers": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
        },
        "banks": {
            "display_name": "Banks",
            "model_daily": "banks_model.pth",
            "model_hourly": "banks_model_hourly.pth",
            "tickers": ["JPM", "BAC", "WFC", "C"],
        },
        "mining": {
            "display_name": "Mining",
            "model_daily": "mining_model.pth",
            "model_hourly": "mining_model_hourly.pth",
            "tickers": ["RIO", "BHP", "VALE", "FCX"],
        },
    }


def get_config() -> Config:
    """Get the active configuration object."""
    return Config()
