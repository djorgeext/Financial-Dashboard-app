"""
Unit tests for news_service module.
Tests news analysis, sentiment classification, and error handling.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from backend.news_service import NewsService, create_news_service


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
    
    @patch('backend.news_service.GROQ_AVAILABLE', True)
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


class TestTickerNewsFlow:
    """Tests for notebook-style ticker news flow."""

    def test_relevance_score_for_one_letter_ticker_requires_token_boundary(self, news_service_no_groq):
        unrelated_title = "Cloud computing demand remains stable"
        unrelated_summary = "Macro data points to mixed growth and calmer volatility."
        explicit_title = "C jumps after earnings beat"
        explicit_summary = "Investors cited C as a beneficiary of stronger capital returns."

        low_score = news_service_no_groq._relevance_score("C", "banks", unrelated_title, unrelated_summary)
        high_score = news_service_no_groq._relevance_score("C", "banks", explicit_title, explicit_summary)

        assert low_score < 0.2
        assert high_score >= 0.55
        assert high_score > low_score

    def test_analyze_ticker_news_fallback_structure(self, news_service_no_groq, monkeypatch):
        sample = [
            {
                "ticker": "BAC",
                "title": "Bank of America beats earnings estimates",
                "summary": "Bank of America reported stronger net interest income and upbeat guidance.",
                "url": "https://example.com/news1",
                "url_canonical": "https://example.com/news1",
                "publisher": "Example",
                "published_at": "2026-04-12T00:00:00+00:00",
            }
        ]

        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", lambda ticker, limit=30: sample)

        result = news_service_no_groq.analyze_ticker_news("BAC", sector="banks", bank_name="BAC", max_items=5)

        assert result["ticker"] == "BAC"
        assert "news_summaries" in result
        assert "bank_summary_en" in result
        assert "sentiment" in result
        assert "sentiment_score" in result
        assert result["status"] in ["success", "success_fallback"]

    def test_analyze_ticker_news_uses_cache(self, news_service_no_groq, monkeypatch):
        sample = [
            {
                "ticker": "AAPL",
                "title": "Apple sees strong iPhone demand",
                "summary": "Demand data remained firm across key regions.",
                "url": "https://example.com/news2",
                "url_canonical": "https://example.com/news2",
                "publisher": "Example",
                "published_at": "2026-04-12T00:00:00+00:00",
            }
        ]

        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", lambda ticker, limit=30: sample)

        first = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech")
        second = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech")

        assert first == second

    def test_analyze_ticker_news_exception_payload_is_not_cached(self, news_service_no_groq, monkeypatch):
        call_counter = {"count": 0}

        def _fail_fetch(ticker, limit=30):
            call_counter["count"] += 1
            raise RuntimeError("provider_temporarily_unavailable")

        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", _fail_fetch)

        cache_key = news_service_no_groq._ticker_news_cache_key(
            ticker="AAPL",
            sector="tech",
            max_items=10,
            bank_name=None,
        )

        first = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech")
        second = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech")

        assert first["status"] == "error"
        assert second["status"] == "error"
        assert cache_key not in news_service_no_groq.cache
        assert cache_key not in news_service_no_groq.cache_times
        assert call_counter["count"] == 2

    def test_analyze_ticker_news_cache_key_differs_by_max_items(self, news_service_no_groq, monkeypatch):
        call_counter = {"count": 0}

        def fake_fetch(ticker, limit=30):
            call_counter["count"] += 1
            return [
                {
                    "ticker": ticker,
                    "title": f"{ticker} reports strong growth {i}",
                    "summary": f"{ticker} sees continued momentum in core business line {i}.",
                    "url": f"https://example.com/{i}",
                    "url_canonical": f"https://example.com/{i}",
                    "publisher": "Example",
                    "published_at": "2026-04-12T00:00:00+00:00",
                }
                for i in range(5)
            ]

        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", fake_fetch)

        first = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech", max_items=1)
        second = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech", max_items=3)

        assert first["metrics"]["retained"] == 1
        assert second["metrics"]["retained"] == 3
        assert call_counter["count"] == 2

    def test_analyze_ticker_news_cache_key_differs_by_sector(self, news_service_no_groq, monkeypatch):
        call_counter = {"count": 0}

        def fake_fetch(ticker, limit=30):
            call_counter["count"] += 1
            return [
                {
                    "ticker": ticker,
                    "title": f"{ticker} reports stable demand",
                    "summary": f"{ticker} remains in focus for investors.",
                    "url": "https://example.com/sector",
                    "url_canonical": "https://example.com/sector",
                    "publisher": "Example",
                    "published_at": "2026-04-12T00:00:00+00:00",
                }
            ]

        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", fake_fetch)

        first = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech", max_items=2)
        second = news_service_no_groq.analyze_ticker_news("AAPL", sector="banks", max_items=2)

        assert first["sector"] == "tech"
        assert second["sector"] == "banks"
        assert call_counter["count"] == 2

    def test_clean_llm_output_removes_think_leaks_and_unclosed_blocks(self, news_service_no_groq):
        raw = "```json\n<think>internal chain\nstill internal"
        cleaned = news_service_no_groq._clean_llm_output(raw)

        assert cleaned == ""
        assert "<think" not in cleaned.lower()
        assert "```" not in cleaned

    def test_normalize_item_summary_rejects_reasoning_without_think_tags(self, news_service_no_groq):
        source_text = "Apple reported stronger revenue and raised guidance for next quarter."
        llm_output = (
            "Let's break this down before finalizing.\n"
            "First, revenue was stronger than expected.\n"
            "Tone: positive."
        )

        normalized = news_service_no_groq._normalize_item_summary_output(llm_output, source_text)

        assert normalized == news_service_no_groq._heuristic_item_summary(source_text)

    def test_normalize_item_summary_accepts_three_lines_and_enforces_labels(self, news_service_no_groq):
        source_text = "Provider text used only for fallback if needed."
        llm_output = (
            "Revenue beat consensus and margins expanded.\n"
            "Could support improved forward guidance and sentiment.\n"
            "positive."
        )

        normalized = news_service_no_groq._normalize_item_summary_output(llm_output, source_text)
        lines = normalized.splitlines()

        assert len(lines) == 3
        assert lines[0].startswith("- Key facts:")
        assert lines[1].startswith("- Business/risk impact:")
        assert lines[2].startswith("- Tone:")

    def test_normalize_ticker_summary_rejects_reasoning_like_labeled_lines(self, news_service_no_groq):
        item_summaries = [
            "- Key facts: Revenue beat estimates.\n- Business/risk impact: Supports guidance confidence.\n- Tone: positive.",
        ]
        ticker = "AAPL"
        llm_output = (
            "Overview: Let's break this down before the final signal.\n"
            "Risks: Margin pressure from costs.\n"
            "Catalysts: Product cycle momentum.\n"
            "Final signal: Constructive/Bullish."
        )

        normalized = news_service_no_groq._normalize_ticker_summary_output(llm_output, item_summaries, ticker)

        assert normalized == news_service_no_groq._heuristic_ticker_summary(item_summaries, ticker)
        assert "Let's break this down" not in normalized

    def test_normalize_ticker_summary_accepts_clean_labeled_output_in_order(self, news_service_no_groq):
        item_summaries = [
            "- Key facts: Demand remained stable.\n- Business/risk impact: Supports earnings visibility.\n- Tone: neutral.",
        ]
        ticker = "MSFT"
        llm_output = (
            "Overview: Headline flow is balanced with moderate upside drivers.\n"
            "Risks: Slower enterprise spending could delay expansion.\n"
            "Catalysts: Upcoming earnings and cloud bookings updates may improve visibility.\n"
            "Final signal: Neutral."
        )

        normalized = news_service_no_groq._normalize_ticker_summary_output(llm_output, item_summaries, ticker)

        assert normalized == llm_output

    def test_ticker_news_graceful_without_chromadb_collection(self, news_service_no_groq, monkeypatch):
        sample = [
            {
                "ticker": "AAPL",
                "title": "Apple sees resilient services growth",
                "summary": "Services and wearables showed stable momentum.",
                "url": "https://example.com/chroma-fallback",
                "url_canonical": "https://example.com/chroma-fallback",
                "publisher": "Example",
                "published_at": "2026-04-12T00:00:00+00:00",
            }
        ]

        news_service_no_groq._chroma_collection = None
        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", lambda ticker, limit=30: sample)

        result = news_service_no_groq.analyze_ticker_news("AAPL", sector="tech", max_items=3)

        assert result["status"] in ["success", "success_fallback"]
        assert result["metrics"]["retained"] == 1

    def test_ticker_news_reuses_persisted_item_summary_when_available(self, news_service_no_groq, monkeypatch):
        sample = [
            {
                "ticker": "BAC",
                "title": "Bank of America expands digital onboarding",
                "summary": "Management highlighted lower service costs and customer adoption.",
                "url": "https://example.com/persisted",
                "url_canonical": "https://example.com/persisted",
                "publisher": "Example",
                "published_at": "2026-04-12T00:00:00+00:00",
            }
        ]

        monkeypatch.setattr(news_service_no_groq, "_fetch_news_from_yf", lambda ticker, limit=30: sample)
        monkeypatch.setattr(news_service_no_groq, "_get_persisted_news_summary", lambda item_id: "- Key facts: Cached summary.\n- Business/risk impact: Cached impact.\n- Tone: neutral.")
        monkeypatch.setattr(news_service_no_groq, "_persist_news_summary", lambda item_id, summary, metadata: False)

        def _must_not_run(*args, **kwargs):
            raise AssertionError("_summarize_news_item should not be called when persisted summary is available")

        monkeypatch.setattr(news_service_no_groq, "_summarize_news_item", _must_not_run)

        result = news_service_no_groq.analyze_ticker_news("BAC", sector="banks", max_items=2)

        assert result["metrics"]["persisted_reused"] == 1
        assert result["metrics"]["summaries_generated"] == 0
        assert result["news_summaries"][0]["summary_en"].startswith("- Key facts:")
