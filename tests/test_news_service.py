"""
Unit tests for news_service module.
Tests news analysis, sentiment classification, and error handling.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from news_service import NewsService, create_news_service


@pytest.fixture
def news_service_no_groq():
    """Create NewsService without Groq API (offline mode)."""
    return NewsService(api_key=None)


@pytest.fixture
def news_service_with_mock_groq():
    """Create NewsService with mocked Groq client."""
    service = NewsService(api_key="mock_key")
    # Mock the client
    service.client = Mock()
    return service


class TestNewsServiceInitialization:
    """Tests for NewsService initialization."""
    
    def test_init_without_api_key(self):
        """Test initialization without API key."""
        service = NewsService(api_key=None)
        assert service.api_key is None
        assert service.client is None
    
    def test_init_with_api_key(self):
        """Test initialization with mock API key."""
        service = NewsService(api_key="mock_key")
        # Note: Groq client might not be available in test environment
        # So we just verify the service was created
        assert service.api_key == "mock_key"
    
    def test_cache_ttl_configuration(self):
        """Test cache TTL configuration."""
        service = NewsService(api_key=None, cache_ttl=1200)
        assert service.cache_ttl == 1200


class TestCaching:
    """Tests for caching functionality."""
    
    def test_cache_valid_check_no_entry(self, news_service_no_groq):
        """Test cache validation for missing entry."""
        assert news_service_no_groq._is_cache_valid("nonexistent_key") is False
    
    def test_cache_store_and_retrieve(self, news_service_no_groq):
        """Test storing and retrieving from cache."""
        test_data = {"sentiment": "bullish", "score": 0.8}
        news_service_no_groq._set_cache("test_key", test_data)
        
        retrieved = news_service_no_groq._get_cached("test_key")
        assert retrieved == test_data
    
    def test_cache_expiry(self, news_service_no_groq):
        """Test cache expiration."""
        import time
        
        service = NewsService(api_key=None, cache_ttl=1)  # 1 second TTL
        test_data = {"sentiment": "bearish"}
        service._set_cache("expiring_key", test_data)
        
        # Should be valid immediately
        assert service._get_cached("expiring_key") is not None
        
        # Wait for expiry
        time.sleep(1.5)
        assert service._get_cached("expiring_key") is None


class TestSafeDefaults:
    """Tests for safe default responses."""
    
    def test_get_safe_default_response(self, news_service_no_groq):
        """Test safe default response structure."""
        response = news_service_no_groq.get_safe_default_response()
        
        assert response["sentiment"] == "neutral"
        assert response["sentiment_score"] == 0.0
        assert response["signals"] == []
        assert "timestamp" in response
        assert response["status"] == "unavailable"
    
    def test_analyze_news_without_groq(self, news_service_no_groq):
        """Test news analysis returns default when Groq unavailable."""
        result = news_service_no_groq.analyze_news("tech")
        
        assert result["sentiment"] == "neutral"
        assert result["status"] == "default"
    
    def test_analyze_news_without_items(self, news_service_no_groq):
        """Test news analysis without news items."""
        result = news_service_no_groq.analyze_news("tech", news_items=None)
        
        assert result["sentiment"] == "neutral"
        assert result["status"] == "default"


class TestSignalExtraction:
    """Tests for signal extraction."""
    
    def test_extract_signals_empty_text(self, news_service_no_groq):
        """Test signal extraction with empty text."""
        signals = news_service_no_groq.extract_signals("tech", "")
        assert signals == []
    
    def test_extract_signals_bullish_keywords(self, news_service_no_groq):
        """Test extraction of bullish signals."""
        text = "Stock prices surge and rally to new highs"
        signals = news_service_no_groq.extract_signals("tech", text)
        
        assert len(signals) > 0
        # Should contain bullish signals
        assert any("surge" in sig.lower() for sig in signals)
    
    def test_extract_signals_bearish_keywords(self, news_service_no_groq):
        """Test extraction of bearish signals."""
        text = "Market crash and downgrade by analysts"
        signals = news_service_no_groq.extract_signals("tech", text)
        
        assert len(signals) > 0
        # Should contain bearish signals
        assert any("crash" in sig.lower() or "downgrade" in sig.lower() for sig in signals)
    
    def test_extract_signals_limit(self, news_service_no_groq):
        """Test that signal extraction limits results."""
        text = "surge rally jump soar beat upgrade buy strong robust powerful excellent fantastic amazing wonderful"
        signals = news_service_no_groq.extract_signals("tech", text)
        
        assert len(signals) <= 5  # Should not exceed max


class TestNewsStatus:
    """Tests for news service status."""
    
    def test_get_news_status_structure(self, news_service_no_groq):
        """Test news status response structure."""
        status = news_service_no_groq.get_news_status()
        
        assert "groq_available" in status
        assert "client_ready" in status
        assert "cache_size" in status
        assert "timestamp" in status
    
    def test_get_news_status_client_ready_false(self, news_service_no_groq):
        """Test news status when client not ready."""
        status = news_service_no_groq.get_news_status()
        assert status["client_ready"] is False
    
    def test_get_news_status_client_ready_true(self, news_service_with_mock_groq):
        """Test news status when client is mocked/ready."""
        status = news_service_with_mock_groq.get_news_status()
        assert status["client_ready"] is True


class TestAnalysisErrorHandling:
    """Tests for error handling in analysis."""
    
    @patch('news_service.GROQ_AVAILABLE', True)
    def test_analyze_news_json_parse_error(self):
        """Test handling of JSON parsing errors."""
        service = NewsService(api_key="mock_key")
        service.client = Mock()
        
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json"
        service.client.chat.completions.create.return_value = mock_response
        
        result = service.analyze_news("tech", ["test news"])
        
        # Should return error status
        assert result["status"] == "parse_error"
        assert result["sentiment"] == "neutral"


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_news_service(self):
        """Test creating NewsService via factory."""
        service = create_news_service(api_key=None)
        assert service is not None
        assert isinstance(service, NewsService)
    
    def test_create_news_service_with_cache_ttl(self):
        """Test factory function with custom cache TTL."""
        service = create_news_service(cache_ttl=1800)
        assert service.cache_ttl == 1800


class TestCacheKeyGeneration:
    """Tests for cache key generation and usage."""
    
    def test_analyze_news_caches_result(self, news_service_no_groq):
        """Test that analyze_news caches results."""
        # First call - should create cache entry even with default response
        result1 = news_service_no_groq.analyze_news("tech", ["news1"])
        # Note: Cache is set even for default responses
        cached = news_service_no_groq._get_cached("news_tech")
        assert cached is not None  # Cache should be populated
        
        # Second call should return cached result
        result2 = news_service_no_groq.analyze_news("tech", ["news2"])
        assert result2 == result1  # Same cached result (ignores new items)
