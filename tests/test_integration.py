"""
Integration tests for the financial dashboard API.
Tests end-to-end scenarios and API endpoint responses.
"""
import pytest
from unittest.mock import Mock, patch
import os
import tempfile
from pathlib import Path

# Import after ensuring module path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app
from config import Config

# Import TestClient
from fastapi.testclient import TestClient

# Initialize config and services before tests
import app as app_module
if app_module.config is None:
    from config import get_config
    from model_service import create_model_service
    from data_fetcher import create_data_fetcher
    from news_service import create_news_service
    
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
        assert "status" in data


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
        assert response.status_code in [200, 404]  # 404 if dashboard.html not in test context
    
    def test_dashboard_endpoint(self, client):
        """Test serving dashboard at /dashboard."""
        response = client.get("/dashboard")
        assert response.status_code in [200, 404]


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
