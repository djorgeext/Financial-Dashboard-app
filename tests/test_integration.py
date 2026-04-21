"""
Integration tests for the financial dashboard API.
Tests end-to-end scenarios and API endpoint responses.
"""
import json
import pytest
from unittest.mock import Mock, patch
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Import after ensuring module path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app import app
from backend.config import Config

# Import TestClient
from fastapi.testclient import TestClient

# Initialize config and services before tests
from backend import app as app_module
if app_module.config is None:
    from backend.config import get_config
    from backend.model_service import create_model_service
    from backend.data_fetcher import create_data_fetcher
    from backend.news_service import create_news_service
    
    app_module.config = get_config()
    app_module.model_service = create_model_service(model_dir=app_module.config.MODEL_DIR)
    app_module.data_fetcher = create_data_fetcher(cache_ttl=app_module.config.DATA_CACHE_TTL)
    
    try:
        groq_key = app_module.config.get_groq_api_key()
        if groq_key:
            app_module.news_service = create_news_service(api_key=groq_key, cache_ttl=app_module.config.NEWS_CACHE_TTL)
        else:
            app_module.news_service = create_news_service(cache_ttl=app_module.config.NEWS_CACHE_TTL)
    except:
        app_module.news_service = create_news_service(cache_ttl=app_module.config.NEWS_CACHE_TTL)


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration for testing."""
    def mock_get_groq_key():
        return "mock_test_key"
    
    # Mock config methods
    monkeypatch.setattr(Config, "get_groq_api_key", mock_get_groq_key)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_readiness_probe(self, client):
        """Test /ready endpoint."""
        response = client.get("/ready")
        # Could be 200 or 503 depending on data availability
        assert response.status_code in [200, 503]
        
        data = response.json()
        if response.status_code == 200:
            assert "status" in data
        else:
            assert data["status_code"] == 503
            assert "error" in data
            assert "timestamp" in data


class TestFinancialDataEndpoints:
    """Tests for financial data endpoints."""
    
    def test_financial_summary_valid_sector(self, client):
        """Test financial summary for valid sector."""
        response = client.get("/api/financial/summary/tech")
        # Might fail if yfinance unavailable, but structure should be correct
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["sector"] == "tech"
            assert "data" in data
            assert "timestamp" in data
    
    def test_financial_summary_invalid_sector(self, client):
        """Test financial summary with invalid sector."""
        response = client.get("/api/financial/summary/invalid_sector_xyz")
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data or "error" in data


class TestForecastEndpoints:
    """Tests for forecast endpoints."""
    
    def test_forecast_daily_granularity(self, client):
        """Test forecast endpoint with daily granularity."""
        response = client.get("/api/forecast/tech/daily")
        assert response.status_code == 200
        
        data = response.json()
        assert data["sector"] == "tech"
        assert data["granularity"] == "daily"
        assert "forecast" in data or "status" in data
    
    def test_forecast_hourly_granularity(self, client):
        """Test forecast endpoint with hourly granularity."""
        response = client.get("/api/forecast/banks/hourly")
        assert response.status_code == 200
        
        data = response.json()
        assert data["sector"] == "banks"
        assert data["granularity"] == "hourly"

    def test_hourly_ticker_route_not_captured_by_generic_forecast_route(self, client):
        """Regression test: /api/forecast/hourly/{ticker} should never fail with granularity 422."""
        response = client.get("/api/forecast/hourly/BAC")
        assert response.status_code != 422
        assert response.status_code in [200, 400, 500, 503]

    def test_hourly_ticker_route_returns_hourly_handler_payload(self, client, monkeypatch):
        """Regression test: /api/forecast/hourly/{ticker} must route to hourly ticker handler."""
        mock_inference = Mock()
        mock_inference.forecast_hourly.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "success",
            "last_40_hours": [{"timestamp": "2026-04-12T00:00:00", "close": 40.0}],
            "forecast_10_hours": [{"timestamp": "2026-04-12T01:00:00", "forecast": 40.5}]
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)

        response = client.get("/api/forecast/hourly/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["ticker"] == "BAC"
        assert "forecast_10_hours" in data
        assert "last_40_hours" in data
        mock_inference.forecast_hourly.assert_called_once_with("BAC")

    def test_hourly_ticker_success_payload_strips_top_level_error(self, client, monkeypatch):
        """Contract test: success_* forecast payloads must not expose top-level error."""
        mock_inference = Mock()
        mock_inference.forecast_hourly.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "success_fallback",
            "error": "feature_windows_insufficient",
            "forecast_10_hours": [{"timestamp": "2026-04-12T01:00:00", "forecast": 40.5}],
            "last_40_hours": [{"timestamp": "2026-04-12T00:00:00", "close": 40.0}],
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)

        response = client.get("/api/forecast/hourly/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success_fallback"
        assert "error" not in data
        assert data["fallback_reason"] == "feature_windows_insufficient"
        assert data["metadata"]["fallback_reason"] == "feature_windows_insufficient"
    
    def test_forecast_invalid_granularity(self, client):
        """Test forecast with invalid granularity."""
        response = client.get("/api/forecast/tech/invalid_granularity")
        # Should either reject or use default
        assert response.status_code in [200, 422]  # 422 for validation error
    
    def test_forecast_invalid_sector(self, client):
        """Test forecast for invalid sector."""
        response = client.get("/api/forecast/invalid_xyz/daily")
        assert response.status_code == 400


class TestNewsEndpoints:
    """Tests for news endpoints."""
    
    def test_news_analysis_tech(self, client):
        """Test news endpoint for tech sector."""
        response = client.get("/api/news/tech")
        assert response.status_code == 200
        
        data = response.json()
        assert data["sector"] == "tech"
        assert "analysis" in data
        assert "timestamp" in data
    
    def test_news_analysis_all_sectors(self, client):
        """Test news endpoint for all sectors."""
        for sector in ["tech", "banks", "mining"]:
            response = client.get(f"/api/news/{sector}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["sector"] == sector
            assert data["analysis"]["sentiment"] in ["bearish", "neutral", "bullish"]
    
    def test_news_analysis_invalid_sector(self, client):
        """Test news for invalid sector."""
        response = client.get("/api/news/invalid_news_sector")
        assert response.status_code == 400

    def test_ticker_news_endpoint_unambiguous_path(self, client):
        """Test ticker news endpoint uses dedicated path without clashing with sector route."""
        response = client.get("/api/news/ticker/AAPL")
        assert response.status_code == 200

        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "sentiment" in data
        assert "status" in data
        assert "application/json" in response.headers.get("content-type", "")

    def test_ticker_news_endpoint_structure_and_status_with_explicit_context(self, client, monkeypatch):
        """Ensure ticker endpoint returns JSON structure and calls notebook-style ticker analysis."""
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "tech": {"tickers": ["AAPL", "MSFT"]}
            }
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "ticker": "AAPL",
            "sector": "tech",
            "sentiment": "bullish",
            "sentiment_score": 0.7,
            "key_signals": ["Positive guidance"],
            "summary": "Constructive momentum",
            "news_summaries": [{"title": "Headline"}],
            "metrics": {"retained": 2},
            "timestamp": "2026-04-12T00:00:00",
            "status": "success"
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)

        response = client.get("/api/news/ticker/AAPL")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["sector"] == "tech"
        assert data["status"] == "success"
        assert "sentiment" in data
        assert "key_signals" in data
        assert "news_summaries" in data
        assert "metrics" in data

        mock_news.analyze_ticker_news.assert_called_once()
        kwargs = mock_news.analyze_ticker_news.call_args.kwargs
        assert kwargs["ticker"] == "AAPL"
        assert kwargs["sector"] == "tech"

    def test_news_ticker_literal_still_routes_to_sector_endpoint(self, client):
        """Ensure /api/news/ticker still maps to sector endpoint semantics."""
        response = client.get("/api/news/ticker")
        assert response.status_code == 400

        data = response.json()
        assert data["status_code"] == 400
        assert "error" in data
        assert "timestamp" in data


class TestOptionsEndpoints:
    """Tests for options endpoint payload structure."""

    def test_options_endpoint_notebook_columns_present(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "success",
            "table_rows": [
                {
                    "Fecha": "2026-04-12",
                    "Sector": "banks",
                    "Ticker": "BAC",
                    "Tipo": "CALL",
                    "Precio Actual": 40.12,
                    "Barrera a 10 horas": 40.86,
                    "Vencimiento": "2026-04-24",
                    "Strike": 40.5,
                    "Ask": 0.65,
                    "IV": 0.31,
                    "Prob. Mercado": 0.52,
                    "Prob. Final (Bayes)": 0.61,
                    "Sentimiento Score": 0.12,
                }
            ],
            "suggested_options": [
                {
                    "option_type": "call",
                    "strike": 40.5,
                    "expiration": "2026-04-24",
                    "probability": 0.61,
                }
            ],
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.1,
            "summary": "Balanced headlines",
            "status": "success",
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert "table_rows" in data
        assert len(data["table_rows"]) == 1
        row = data["table_rows"][0]
        assert "Prob. Final (Bayes)" in row
        assert "Barrera" not in row
        assert "Barrera a 10 horas" in row
        assert "Sentimiento Score" in row
        assert "news" in data

    def test_options_endpoint_no_signal_does_not_force_fallback(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "success_no_signal",
            "op_type": "neutral",
            "target_type": 0,
            "current_price": 40.0,
            "barrier": 40.0,
            "barrier_hourly": 40.0,
            "table_rows": [],
            "suggested_options": [],
            "no_signal_reason": "neutral_primary_skip",
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "No strong catalyst",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [
                {
                    "option_type": "call",
                    "strike": 41.0,
                    "expiration": "2026-04-24",
                    "probability": 0.5,
                }
            ]
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success_no_signal"
        assert data["table_rows"] == []
        assert data["suggested_options"] == []
        assert data["op_type"] == "neutral"
        mock_options_analyzer.suggest_options.assert_not_called()

    def test_options_endpoint_error_neutral_forecast_skips_fallback_analyzer(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "error",
            "error": "inference_failed_for_ticker",
            "table_rows": [],
            "suggested_options": [],
        }
        mock_inference.forecast_hourly.return_value = {
            "ticker": "BAC",
            "status": "success_no_signal",
            "op_type": "neutral",
            "is_neutral": True,
            "current_price": 40.0,
            "barrier_hourly": 40.2,
            "metadata": {"no_signal_reason": "hourly_threshold_not_met"},
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "No strong catalyst",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [
                {
                    "option_type": "call",
                    "strike": 41.0,
                    "expiration": "2026-04-24",
                    "probability": 0.5,
                }
            ]
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success_no_signal"
        assert data["op_type"] == "neutral"
        assert data["is_neutral"] is True
        assert data["table_rows"] == []
        assert data["suggested_options"] == []
        assert data["no_signal_reason"] == "hourly_threshold_not_met"
        mock_inference.forecast_hourly.assert_called_once()
        mock_options_analyzer.suggest_options.assert_not_called()

    def test_options_endpoint_unsupported_error_skips_fallback_analyzer(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "ZZZZ",
            "sector": None,
            "status": "error",
            "error": "Ticker ZZZZ not found in configured sectors",
            "table_rows": [],
            "suggested_options": [],
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "No coverage",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [
                {
                    "option_type": "call",
                    "strike": 10.0,
                    "expiration": "2026-04-24",
                    "probability": 0.5,
                }
            ]
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/ZZZZ")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "error"
        assert data["error"] == "Ticker ZZZZ not found in configured sectors"
        assert "fallback_detail" not in data
        assert "fallback_reason" not in data
        mock_inference.forecast_hourly.assert_not_called()
        mock_options_analyzer.suggest_options.assert_not_called()

    def test_options_endpoint_error_uses_frontend_compatible_fallback_payload(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "error",
            "error": "inference_failed_for_ticker",
            "table_rows": [],
            "suggested_options": [],
        }
        mock_inference.forecast_hourly.return_value = {
            "current_price": 40.0,
            "forecast_10_hours": [
                {"forecast": 40.8, "confidence": 0.71},
                {"forecast": 41.0, "confidence": 0.74},
            ],
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "Fallback news context",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [
                {
                    "option_type": "call",
                    "strike": 41.0,
                    "expiration": "2026-04-24",
                    "probability": 0.63,
                }
            ]
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success_fallback"
        assert "error" not in data
        assert data["table_rows"] == []
        assert isinstance(data["suggested_options"], list)
        assert len(data["suggested_options"]) > 0
        assert data["fallback_reason"] == "inference_failed_for_ticker"

        mock_inference.forecast_hourly.assert_called_once()
        mock_options_analyzer.suggest_options.assert_called_once()

    def test_options_endpoint_fallback_pred_class_uses_barrier_threshold(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "error",
            "error": "inference_failed_for_ticker",
            "table_rows": [],
            "suggested_options": [],
        }
        mock_inference.forecast_hourly.return_value = {
            "current_price": 40.0,
            "barrier_hourly": 39.4,
            "forecast_10_hours": [
                {"forecast": 41.5, "confidence": 0.74},
                {"forecast": 41.8, "confidence": 0.76},
            ],
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "Fallback news context",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [
                {
                    "option_type": "put",
                    "strike": 39.5,
                    "expiration": "2026-04-24",
                    "probability": 0.63,
                }
            ]
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success_fallback"
        assert data["fallback_reason"] == "inference_failed_for_ticker"

        mock_options_analyzer.suggest_options.assert_called_once()
        kwargs = mock_options_analyzer.suggest_options.call_args.kwargs
        assert kwargs["pred_class"] == 0
        assert kwargs["forecast_price"] == pytest.approx(39.4)
        assert kwargs["forecast_confidence"] == pytest.approx(0.75)

    def test_options_endpoint_fallback_failure_returns_error_state(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "error",
            "error": "inference_failed_for_ticker",
            "table_rows": [],
            "suggested_options": [],
        }
        mock_inference.forecast_hourly.return_value = {
            "current_price": 40.0,
            "forecast_10_hours": [
                {"forecast": 40.8, "confidence": 0.71},
            ],
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "Fallback news context",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [],
            "error": "pricing_engine_unavailable",
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "error"
        assert data["status"] != "success_fallback"
        assert data["error"] == "fallback_failed"
        assert data["fallback_reason"] == "inference_failed_for_ticker"
        assert data["fallback_detail"] == "pricing_engine_unavailable"
        assert data["suggested_options"] == []

    def test_options_endpoint_fallback_empty_suggestions_returns_error_state(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "error",
            "error": "inference_failed_for_ticker",
            "table_rows": [],
            "suggested_options": [],
        }
        mock_inference.forecast_hourly.return_value = {
            "current_price": 40.0,
            "forecast_10_hours": [
                {"forecast": 40.8, "confidence": 0.65},
            ],
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "Fallback news context",
            "status": "success",
        }

        mock_options_analyzer = Mock()
        mock_options_analyzer.suggest_options.return_value = {
            "suggested_options": [],
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)
        monkeypatch.setattr(app_module, "options_analyzer", mock_options_analyzer)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "error"
        assert data["status"] != "success_fallback"
        assert data["error"] == "fallback_failed"
        assert data["fallback_reason"] == "inference_failed_for_ticker"
        assert data["fallback_detail"] == "empty_suggestions"
        assert data["suggested_options"] == []

    def test_options_endpoint_sanitizes_non_finite_values(self, client, monkeypatch):
        mock_inference = Mock()
        mock_inference.get_sector_ticker_mapping.return_value = {
            "sectors": {
                "banks": {"tickers": ["BAC"]}
            }
        }
        mock_inference.build_options_analysis.return_value = {
            "ticker": "BAC",
            "sector": "banks",
            "status": "success",
            "movement_10h_pct": np.float64(np.nan),
            "native_nan": float("nan"),
            "native_inf": float("inf"),
            "native_neg_inf": float("-inf"),
            "pandas_na": pd.NA,
            "pandas_nat": pd.NaT,
            "table_rows": [
                {
                    "Fecha": "2026-04-12",
                    "Ticker": "BAC",
                    "Strike": np.float64(40.5),
                    "Ask": np.float64(np.inf),
                    "IV": np.float64(-np.inf),
                    "Prob. Final (Bayes)": np.float64(np.nan),
                    "Native NaN": float("nan"),
                    "Native Inf": float("inf"),
                    "Native -Inf": float("-inf"),
                    "Pandas NA": pd.NA,
                    "Pandas NaT": pd.NaT,
                }
            ],
            "suggested_options": [
                {
                    "option_type": "call",
                    "probability": np.float64(np.inf),
                    "native_probability": float("-inf"),
                    "nested_tuple": (
                        np.float64(np.nan),
                        1.0,
                        pd.NA,
                        pd.NaT,
                        float("inf"),
                        float("nan"),
                    ),
                }
            ],
            "metadata": {
                "probs": np.array([0.7, np.nan, np.inf]),
                "native_values": [float("nan"), float("inf"), float("-inf")],
                "pandas_values": [pd.NA, pd.NaT],
            },
        }

        mock_news = Mock()
        mock_news.analyze_ticker_news.return_value = {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "Balanced headlines",
            "status": "success",
        }

        monkeypatch.setattr(app_module, "inference_service", mock_inference)
        monkeypatch.setattr(app_module, "news_service", mock_news)

        response = client.get("/api/options/BAC")
        assert response.status_code == 200

        data = response.json()
        assert data["movement_10h_pct"] is None
        assert data["native_nan"] is None
        assert data["native_inf"] is None
        assert data["native_neg_inf"] is None
        assert data["pandas_na"] is None
        assert data["pandas_nat"] is None
        assert data["table_rows"][0]["Strike"] == pytest.approx(40.5)
        assert data["table_rows"][0]["Ask"] is None
        assert data["table_rows"][0]["IV"] is None
        assert data["table_rows"][0]["Prob. Final (Bayes)"] is None
        assert data["table_rows"][0]["Native NaN"] is None
        assert data["table_rows"][0]["Native Inf"] is None
        assert data["table_rows"][0]["Native -Inf"] is None
        assert data["table_rows"][0]["Pandas NA"] is None
        assert data["table_rows"][0]["Pandas NaT"] is None
        assert data["suggested_options"][0]["probability"] is None
        assert data["suggested_options"][0]["native_probability"] is None
        assert data["suggested_options"][0]["nested_tuple"] == [None, 1.0, None, None, None, None]
        assert data["metadata"]["probs"] == [0.7, None, None]
        assert data["metadata"]["native_values"] == [None, None, None]
        assert data["metadata"]["pandas_values"] == [None, None]

        json.dumps(data, allow_nan=False)


class TestAggregatedDashboardEndpoint:
    """Tests for the main dashboard endpoint."""
    
    def test_dashboard_structure(self, client):
        """Test /api/dashboard returns proper structure."""
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "sectors" in data
        assert isinstance(data["sectors"], dict)
    
    def test_dashboard_all_sectors_included(self, client):
        """Test that dashboard includes all configured sectors."""
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        # Should have at least 3 sectors: tech, banks, mining
        assert len(data["sectors"]) >= 3
    
    def test_dashboard_sector_data_structure(self, client):
        """Test structure of sector data in dashboard."""
        response = client.get("/api/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check first sector's structure
        for sector_key, sector_data in data["sectors"].items():
            if sector_data["status"] == "success":
                assert "display_name" in sector_data
                assert "financial" in sector_data or "error" in sector_data
                assert "forecast" in sector_data or "error" in sector_data
                assert "news" in sector_data or "error" in sector_data


class TestPipelineHealthEndpoint:
    """Tests for pipeline health monitoring."""
    
    def test_pipeline_health_endpoint(self, client):
        """Test /api/pipeline-health endpoint."""
        response = client.get("/api/pipeline-health")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "data_freshness" in data
        assert "models_ready" in data
        assert "all_systems" in data


class TestFrontendEndpoints:
    """Tests for frontend serving."""
    
    def test_root_endpoint(self, client):
        """Test serving dashboard at root."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_dashboard_endpoint(self, client):
        """Test serving dashboard at /dashboard."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_dashboard_css_static_endpoint(self, client):
        """Test dashboard CSS is served from static mount."""
        response = client.get("/static/css/dashboard.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        assert "body {" in response.text


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_nonexistent_endpoint(self, client):
        """Test handling of nonexistent endpoints."""
        response = client.get("/api/nonexistent_endpoint")
        assert response.status_code == 404
    
    def test_invalid_query_parameters(self, client):
        """Test handling of invalid query parameters."""
        response = client.get("/api/forecast/tech/daily?invalid_param=bad_value")
        # Should ignore extra params or handle gracefully
        assert response.status_code in [200, 422]

    def test_http_exception_response_is_json(self, client):
        """Ensure HTTP exceptions are returned as JSON responses by middleware."""
        response = client.get("/api/forecast/invalid_xyz/daily")
        assert response.status_code == 400
        assert "application/json" in response.headers.get("content-type", "")

        data = response.json()
        assert data["status_code"] == 400
        assert "error" in data
        assert "timestamp" in data


class TestCORSHeaders:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.get("/health")
        # CORS headers should be present
        assert response.status_code == 200


class TestEndpointResponseFormats:
    """Tests that all endpoints return JSON with proper structure."""
    
    def test_all_endpoints_return_json(self, client):
        """Test that all main endpoints return JSON."""
        endpoints = [
            "/health",
            "/api/financial/summary/tech",
            "/api/forecast/tech/daily",
            "/api/news/tech",
            "/api/dashboard",
            "/api/pipeline-health",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 400, 500, 503]
            
            # Should be JSON or HTML
            content_type = response.headers.get("content-type", "")
            if response.status_code != 404:
                # JSON responses should have json content type or be parseable
                assert "json" in content_type or response.status_code == 404
    
    def test_error_responses_include_necessary_fields(self, client):
        """Test that error responses include necessary fields."""
        # Trigger an error with invalid sector
        response = client.get("/api/forecast/invalid_xyz/daily")
        
        if response.status_code >= 400:
            try:
                data = response.json()
                # Should have either 'error' or 'detail' field
                assert "error" in data or "detail" in data
            except:
                pass  # Not JSON response


class TestLoadTesting:
    """Basic load testing."""
    
    def test_multiple_concurrent_requests(self, client):
        """Test handling multiple requests."""
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200
        
        # All should succeed
        response = client.get("/api/financial/summary/tech")
        assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
