"""
Inference service for running hourly forecasts based on INFERENCE_v3.ipynb pipeline.
Handles model loading, feature engineering, and OHLC forecast generation.
"""
import logging
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import yfinance as yf

from utils_2 import (
    LSTMMixedModel,
    engineer_features,
    apply_robust_normalization,
    get_drift,
    yang_zhang_volatility,
)

logger = logging.getLogger(__name__)

# Sector configuration extracted from INFERENCE_v3.ipynb
SECTORS_CONFIG = {
    "tech": {
        "display_name": "Technology",
        "model_files": {
            "lstm": "tech_us_model.pth",
            "scaler": "scalers_tech_us.pkl",
            "meta": "meta_model_xgb_tech.json",
            "lstm_hourly": "tech_us_model_hourly.pth",
            "scaler_hourly": "scalers_tech_us_hourly.pkl",
            "meta_hourly": "meta_model_xgb_tech_hourly.json"
        },
        "bayesian": {
            "p_call": 0.3965, "p_put": 0.3918,
            "sensitivity": 0.2, "specificity": 0.92
        },
        "bayesian_hourly": {
            "p_call": 0.4305, "p_put": 0.4031,
            "sensitivity": 0.42, "specificity": 0.84
        },
        "tickers": ["QQQ", "META", "AAPL", "AMZN", "NFLX", "TSLA", "NVDA", "PLTR", "MSFT", "GOOGL", "INTC", "AMD"],
    },
    "banks": {
        "display_name": "Banks",
        "model_files": {
            "lstm": "banks_model.pth",
            "scaler": "scalers_banks.pkl",
            "meta": "meta_model_xgb_banks.json",
            "lstm_hourly": "banks_model_hourly.pth",
            "scaler_hourly": "scalers_banks_hourly.pkl",
            "meta_hourly": "meta_model_xgb_banks_hourly.json"
        },
        "bayesian": {
            "p_call": 0.4199, "p_put": 0.394,
            "sensitivity": 0.32, "specificity": 0.89
        },
        "bayesian_hourly": {
            "p_call": 0.5163, "p_put": 0.4631,
            "sensitivity": 0.61, "specificity": 0.75
        },
        "tickers": ["BAC", "JPM", "WFC", "C", "XLF", "TNA"],
    },
    "mining": {
        "display_name": "Mining",
        "model_files": {
            "lstm": "mining_model.pth",
            "scaler": "scalers_mining.pkl",
            "meta": "meta_model_xgb_mining.json",
            "lstm_hourly": "mining_model_hourly.pth",
            "scaler_hourly": "scalers_mining_hourly.pkl",
            "meta_hourly": "meta_model_xgb_mining_hourly.json"
        },
        "bayesian": {
            "p_call": 0.3896, "p_put": 0.3783,
            "sensitivity": 0.16, "specificity": 0.92
        },
        "bayesian_hourly": {
            "p_call": 0.5261, "p_put": 0.52,
            "sensitivity": 0.65, "specificity": 0.68
        },
        "tickers": ["GLD", "SLV", "NEM", "HL", "PAAS", "NUE", "CLF"],
    }
}


class InferenceService:
    """Service for running inference with pre-trained models."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize InferenceService with model paths.
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.meta_models = {}
        
        logger.info(f"Using device: {self.device}")
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all pre-trained models for all sectors."""
        for sector, config in SECTORS_CONFIG.items():
            try:
                self._load_sector_models(sector)
                logger.info(f"✅ Models loaded for sector: {sector}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading models for {sector}: {e}")
    
    def _load_sector_models(self, sector: str):
        """Load LSTM and meta-models for a specific sector."""
        config = SECTORS_CONFIG[sector]
        model_files = config["model_files"]
        
        # Load hourly LSTM model
        lstm_path = os.path.join(self.model_dir, model_files["lstm_hourly"])
        if os.path.exists(lstm_path):
            model = LSTMMixedModel(num_features=22, lstm_hidden=64, dropout=0.3)
            model.load_state_dict(torch.load(lstm_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[f"{sector}_lstm"] = model
            logger.debug(f"Loaded LSTM model for {sector}")
        
        # Load hourly scaler
        scaler_path = os.path.join(self.model_dir, model_files["scaler_hourly"])
        if os.path.exists(scaler_path):
            try:
                scalers = joblib.load(scaler_path)
                self.scalers[sector] = scalers
                logger.debug(f"Loaded scalers for {sector}")
            except Exception as e:
                logger.warning(f"Error loading scalers for {sector}: {e}")
        
        # Load meta-model (XGBoost)
        meta_path = os.path.join(self.model_dir, model_files["meta_hourly"])
        if os.path.exists(meta_path):
            try:
                meta_model = xgb.XGBClassifier()
                meta_model.load_model(meta_path)
                self.meta_models[sector] = meta_model
                logger.debug(f"Loaded meta-model for {sector}")
            except Exception as e:
                logger.warning(f"Error loading meta-model for {sector}: {e}")
    
    def get_tickers_by_sector(self, sector: str) -> List[Dict]:
        """
        Get tickers and their display names for a sector.
        
        Args:
            sector: Sector key (tech, banks, mining)
            
        Returns:
            List of dicts with 'symbol' and 'name'
        """
        sector_lower = sector.lower()
        if sector_lower not in SECTORS_CONFIG:
            return []
        
        config = SECTORS_CONFIG[sector_lower]
        tickers = []
        
        for symbol in config["tickers"]:
            try:
                ticker_info = yf.Ticker(symbol)
                name = ticker_info.info.get("longName", symbol)
            except:
                name = symbol
            
            tickers.append({
                "symbol": symbol,
                "name": name
            })
        
        return tickers
    
    def get_sector_ticker_mapping(self) -> Dict[str, Dict]:
        """
        Get complete mapping of sectors to tickers.
        
        Returns:
            Dict with sector info and available tickers with models
        """
        result = {"sectors": {}}
        
        for sector, config in SECTORS_CONFIG.items():
            result["sectors"][sector] = {
                "display_name": config["display_name"],
                "tickers": config["tickers"],
                "model_file": config["model_files"].get("lstm_hourly"),
                "count": len(config["tickers"])
            }
        
        return result
    
    def forecast_hourly(self, ticker: str, sector: str = None) -> Dict:
        """
        Generate 10-hour ahead forecast with historical OHLC data.
        
        Args:
            ticker: Stock ticker symbol
            sector: Optional sector (will be inferred if not provided)
            
        Returns:
            Dict with structure:
            {
                "ticker": "JPM",
                "sector": "banks",
                "status": "success" | "error",
                "error": str (if error),
                "last_40_hours": [
                    {
                        "timestamp": ISO string,
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "volume": int,
                        "is_historical": true
                    },
                    ...
                ],
                "forecast_10_hours": [
                    {
                        "timestamp": ISO string,
                        "forecast": float,
                        "confidence": float,
                        "is_forecast": true
                    },
                    ...
                ],
                "median_price": float,
                "forecast_high": float,
                "forecast_low": float,
                "current_price": float,
                "last_update": ISO string
            }
        """
        try:
            # Find sector if not provided
            if sector is None:
                sector = self._find_sector_for_ticker(ticker)
                if sector is None:
                    return {
                        "ticker": ticker,
                        "status": "error",
                        "error": f"Ticker {ticker} not found in any sector configuration"
                    }
            
            sector_lower = sector.lower()
            
            # Fetch historical hourly data (last 40 hours + some buffer)
            try:
                hist = yf.Ticker(ticker).history(period="7d", interval="1h")
            except Exception as e:
                return {
                    "ticker": ticker,
                    "sector": sector_lower,
                    "status": "error",
                    "error": f"Failed to fetch data: {str(e)}"
                }
            
            if len(hist) < 40:
                return {
                    "ticker": ticker,
                    "sector": sector_lower,
                    "status": "error",
                    "error": f"Insufficient historical data. Got {len(hist)} hours, need 40"
                }
            
            # Take last 40 hours
            hist_40 = hist.iloc[-40:].copy()
            hist_40.index = pd.to_datetime(hist_40.index, utc=True)
            
            # Prepare features for model
            try:
                df_features, full_features = engineer_features(hist_40, hist_40['Close'])
            except Exception as e:
                logger.warning(f"Feature engineering error: {e}. Using simpler fallback.")
                return self._fallback_forecast(ticker, sector_lower, hist_40)
            
            if full_features is None or len(full_features) < 40:
                return self._fallback_forecast(ticker, sector_lower, hist_40)
            
            # Get scaler for this ticker
            if sector_lower not in self.scalers:
                logger.warning(f"No scaler found for sector {sector_lower}")
                return self._fallback_forecast(ticker, sector_lower, hist_40)
            
            scalers = self.scalers[sector_lower]
            if ticker not in scalers:
                logger.warning(f"No scaler found for ticker {ticker}")
                return self._fallback_forecast(ticker, sector_lower, hist_40)
            
            # Normalize features
            X_input = full_features.values[-40:].reshape(1, 40, -1)
            X_normalized = apply_robust_normalization(X_input, scalers[ticker])
            X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(self.device)
            
            # Get model predictions
            if f"{sector_lower}_lstm" not in self.models:
                logger.warning(f"LSTM model not loaded for sector {sector_lower}")
                return self._fallback_forecast(ticker, sector_lower, hist_40)
            
            model = self.models[f"{sector_lower}_lstm"]
            
            with torch.no_grad():
                logits, _ = model(X_tensor)
                probs = F.softmax(logits, dim=1)
                confidence = float(torch.max(probs[0]).cpu().numpy())
                pred_class = int(torch.argmax(logits, dim=1)[0].cpu().numpy())
            
            # Generate forecast points
            current_price = float(hist_40['Close'].iloc[-1])
            forecast_prices = self._generate_forecast_prices(
                current_price, 
                hist_40,
                pred_class,
                confidence,
                horizon_hours=10
            )
            
            # Build response
            last_40_ohlc = []
            for idx, row in hist_40.iterrows():
                last_40_ohlc.append({
                    "timestamp": idx.isoformat(),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']) if 'Volume' in row else 0,
                    "is_historical": True
                })
            
            forecast_10h = []
            for i, price in enumerate(forecast_prices, 1):
                timestamp = hist_40.index[-1] + timedelta(hours=i)
                forecast_10h.append({
                    "timestamp": timestamp.isoformat(),
                    "forecast": float(price),
                    "confidence": float(confidence * (1.0 - i * 0.03)),  # Confidence decreases with horizon
                    "is_forecast": True
                })
            
            median_price = float(hist_40['Close'].median())
            forecast_arr = np.array(forecast_prices)
            
            return {
                "ticker": ticker,
                "sector": sector_lower,
                "status": "success",
                "last_40_hours": last_40_ohlc,
                "forecast_10_hours": forecast_10h,
                "median_price": float(median_price),
                "forecast_high": float(np.max(forecast_arr)),
                "forecast_low": float(np.min(forecast_arr)),
                "current_price": float(current_price),
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in forecast_hourly: {e}", exc_info=True)
            return {
                "ticker": ticker,
                "sector": sector or "unknown",
                "status": "error",
                "error": str(e)
            }
    
    def _find_sector_for_ticker(self, ticker: str) -> Optional[str]:
        """Find which sector a ticker belongs to."""
        for sector, config in SECTORS_CONFIG.items():
            if ticker in config["tickers"]:
                return sector
        return None
    
    def _generate_forecast_prices(
        self,
        current_price: float,
        historical_data: pd.DataFrame,
        pred_class: int,
        confidence: float,
        horizon_hours: int = 10
    ) -> List[float]:
        """
        Generate forecast prices for next N hours.
        
        pred_class: 0=down, 1=neutral, 2=up
        """
        try:
            # Calculate volatility and drift from historical data
            returns = historical_data['Close'].pct_change().dropna()
            volatility = float(returns.std())
            drift = float(returns.mean())
            
            if np.isnan(volatility) or volatility == 0:
                volatility = 0.01
            if np.isnan(drift):
                drift = 0.0
        except:
            volatility = 0.01
            drift = 0.0
        
        forecast_prices = []
        
        for h in range(1, horizon_hours + 1):
            # Bias forecast direction based on pred_class
            if pred_class == 2:  # Up
                direction_bias = confidence * 0.02 * h / horizon_hours
            elif pred_class == 0:  # Down
                direction_bias = -confidence * 0.02 * h / horizon_hours
            else:  # Neutral
                direction_bias = 0.0
            
            # Brownian motion with direction bias
            random_shock = np.random.normal(0, volatility * np.sqrt(h / horizon_hours))
            forecast_price = current_price * np.exp(
                (drift + direction_bias) * (h / horizon_hours) + random_shock
            )
            
            forecast_prices.append(float(forecast_price))
        
        return forecast_prices
    
    def _fallback_forecast(
        self,
        ticker: str,
        sector: str,
        hist_data: pd.DataFrame
    ) -> Dict:
        """Generate simple fallback forecast when model inference fails."""
        current_price = float(hist_data['Close'].iloc[-1])
        median_price = float(hist_data['Close'].median())
        volatility = float(hist_data['Close'].pct_change().std())
        
        # Simple extrapolation
        forecast_prices = []
        for h in range(1, 11):
            noise = np.random.normal(0, volatility * 0.5)
            price = current_price * (1 + noise * h / 10)
            forecast_prices.append(float(max(0.01, price)))
        
        last_40_ohlc = []
        for idx, row in hist_data.iterrows():
            last_40_ohlc.append({
                "timestamp": idx.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']) if 'Volume' in row else 0,
                "is_historical": True
            })
        
        forecast_10h = []
        last_time = hist_data.index[-1]
        for i, price in enumerate(forecast_prices, 1):
            timestamp = last_time + timedelta(hours=i)
            forecast_10h.append({
                "timestamp": timestamp.isoformat(),
                "forecast": float(price),
                "confidence": 0.5,
                "is_forecast": True
            })
        
        return {
            "ticker": ticker,
            "sector": sector,
            "status": "success_fallback",
            "last_40_hours": last_40_ohlc,
            "forecast_10_hours": forecast_10h,
            "median_price": float(median_price),
            "forecast_high": float(np.max(forecast_prices)),
            "forecast_low": float(np.min(forecast_prices)),
            "current_price": float(current_price),
            "last_update": datetime.now(timezone.utc).isoformat()
        }


def create_inference_service(model_dir: str = "models") -> InferenceService:
    """Factory function to create inference service."""
    return InferenceService(model_dir=model_dir)
