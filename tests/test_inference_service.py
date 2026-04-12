"""
Unit tests for inference_service gating parity with INFERENCE_v3 notebook.
"""
import numpy as np
import pandas as pd
import pytest

import inference_service as inference_module



def _make_hist(length: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=length, freq=freq, tz="UTC")
    close = np.linspace(100.0, 120.0, length)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(length, 1000, dtype=int),
        },
        index=idx,
    )


@pytest.fixture
def bare_inference_service(monkeypatch):
    monkeypatch.setattr(inference_module.InferenceService, "_load_all_models", lambda self: None)
    service = inference_module.InferenceService(model_dir="models")
    service.daily_models = {"tech": object()}
    service.hourly_models = {"tech": object()}
    service.daily_scalers = {"tech": {"AAPL": {"dummy": True}}}
    service.hourly_scalers = {"tech": {"AAPL": {"dummy": True}}}
    service.daily_meta_models = {"tech": None}
    service.hourly_meta_models = {"tech": None}
    return service


def test_pipeline_context_requires_seq_len_plus_20(bare_inference_service, monkeypatch):
    daily_hist = _make_hist(500, "D")
    hourly_hist = _make_hist(1000, "h")

    class FakeTicker:
        def history(self, period="max", interval=None):
            return hourly_hist.copy() if interval == "1h" else daily_hist.copy()

    monkeypatch.setattr(inference_module.yf, "Ticker", lambda ticker: FakeTicker())

    full_daily = pd.DataFrame(
        np.zeros((inference_module.SEQ_LEN_DAILY + inference_module.MIN_INFERENCE_BUFFER - 1, 4))
    )
    full_hourly = pd.DataFrame(
        np.zeros((inference_module.SEQ_LEN_HOURLY + inference_module.MIN_INFERENCE_BUFFER + 5, 4))
    )

    responses = iter([(daily_hist, full_daily), (hourly_hist, full_hourly)])
    monkeypatch.setattr(inference_module, "engineer_features", lambda hist, close: next(responses))

    result = bare_inference_service._pipeline_context("AAPL", sector="tech")

    assert result["status"] == "error"
    assert "seq_len + 20" in result["error"]


def test_pipeline_context_neutral_neutral_returns_no_signal(bare_inference_service, monkeypatch):
    daily_hist = _make_hist(500, "D")
    hourly_hist = _make_hist(1000, "h")

    class FakeTicker:
        def history(self, period="max", interval=None):
            return hourly_hist.copy() if interval == "1h" else daily_hist.copy()

    monkeypatch.setattr(inference_module.yf, "Ticker", lambda ticker: FakeTicker())

    full_daily = pd.DataFrame(
        np.zeros((inference_module.SEQ_LEN_DAILY + inference_module.MIN_INFERENCE_BUFFER + 2, 4))
    )
    full_hourly = pd.DataFrame(
        np.zeros((inference_module.SEQ_LEN_HOURLY + inference_module.MIN_INFERENCE_BUFFER + 2, 4))
    )

    responses = iter([(daily_hist, full_daily), (hourly_hist, full_hourly)])
    monkeypatch.setattr(inference_module, "engineer_features", lambda hist, close: next(responses))
    monkeypatch.setattr(inference_module, "apply_robust_normalization", lambda x, scaler: x)

    preds = iter(
        [
            {
                "pred_primary": 0,
                "meta_pred": 0,
                "confidence": 0.61,
                "probs": [0.7, 0.2, 0.1],
                "x_tensor": None,
            },
            {
                "pred_primary": 0,
                "meta_pred": 0,
                "confidence": 0.58,
                "probs": [0.68, 0.22, 0.1],
                "x_tensor": None,
            },
        ]
    )
    monkeypatch.setattr(bare_inference_service, "_run_primary_and_meta", lambda *args, **kwargs: next(preds))

    def _must_not_reach_barriers(*args, **kwargs):
        raise AssertionError("_compute_barriers should not run on neutral-neutral early skip")

    monkeypatch.setattr(bare_inference_service, "_compute_barriers", _must_not_reach_barriers)

    result = bare_inference_service._pipeline_context("AAPL", sector="tech")

    assert result["status"] == "no_signal"
    assert result["has_opportunity"] is False
    assert result["no_signal_reason"] == "neutral_primary_skip"


def test_pipeline_context_hourly_signal_dominates_daily_for_option_type(bare_inference_service, monkeypatch):
    daily_hist = _make_hist(500, "D")
    hourly_hist = _make_hist(1000, "h")

    class FakeTicker:
        def history(self, period="max", interval=None):
            return hourly_hist.copy() if interval == "1h" else daily_hist.copy()

    monkeypatch.setattr(inference_module.yf, "Ticker", lambda ticker: FakeTicker())

    full_daily = pd.DataFrame(
        np.zeros((inference_module.SEQ_LEN_DAILY + inference_module.MIN_INFERENCE_BUFFER + 2, 4))
    )
    full_hourly = pd.DataFrame(
        np.zeros((inference_module.SEQ_LEN_HOURLY + inference_module.MIN_INFERENCE_BUFFER + 2, 4))
    )

    responses = iter([(daily_hist, full_daily), (hourly_hist, full_hourly)])
    monkeypatch.setattr(inference_module, "engineer_features", lambda hist, close: next(responses))
    monkeypatch.setattr(inference_module, "apply_robust_normalization", lambda x, scaler: x)

    preds = iter(
        [
            {
                "pred_primary": 1,
                "meta_pred": 0,
                "confidence": 0.61,
                "probs": [0.1, 0.8, 0.1],
                "x_tensor": None,
            },
            {
                "pred_primary": 2,
                "meta_pred": 0,
                "confidence": 0.58,
                "probs": [0.1, 0.2, 0.7],
                "x_tensor": None,
            },
        ]
    )
    monkeypatch.setattr(bare_inference_service, "_run_primary_and_meta", lambda *args, **kwargs: next(preds))

    barrier_values = iter(
        [
            {
                "p_t": 120.0,
                "vol_t": 0.02,
                "drift_val": 0.001,
                "upper_barrier": 125.0,
                "lower_barrier": 115.0,
                "is_call": True,
                "is_put": False,
            },
            {
                "p_t": 120.0,
                "vol_t": 0.03,
                "drift_val": -0.001,
                "upper_barrier": 123.0,
                "lower_barrier": 112.0,
                "is_call": False,
                "is_put": True,
            },
        ]
    )
    monkeypatch.setattr(bare_inference_service, "_compute_barriers", lambda *args, **kwargs: next(barrier_values))

    result = bare_inference_service._pipeline_context("AAPL", sector="tech")

    assert result["status"] == "success"
    assert result["op_type"] == "put"
    assert result["target_type"] == 2
    assert result["barrier"] == 112.0
    assert result["barrier_hourly"] == 112.0


def test_to_market_timezone_localizes_naive_index_as_market_timezone(bare_inference_service):
    idx = pd.date_range("2024-01-02 09:30:00", periods=3, freq="h")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

    converted = bare_inference_service._to_market_timezone(df)

    assert str(converted.index.tz) == inference_module.MARKET_TIMEZONE
    assert converted.index[0] == pd.Timestamp("2024-01-02 09:30:00", tz=inference_module.MARKET_TIMEZONE)


def test_fallback_forecast_omits_top_level_error_and_preserves_reason(bare_inference_service):
    hist_hourly = _make_hist(72, "h")

    result = bare_inference_service._fallback_forecast(
        ticker="AAPL",
        sector="tech",
        hist_data=hist_hourly,
        error="pipeline_feature_window_missing",
    )

    assert result["status"] == "success_fallback"
    assert "error" not in result
    assert result["fallback_reason"] == "pipeline_feature_window_missing"
    assert result["metadata"]["fallback_reason"] == "pipeline_feature_window_missing"


def test_forecast_hourly_sanitizes_success_payload_top_level_error(bare_inference_service, monkeypatch):
    monkeypatch.setattr(
        bare_inference_service,
        "_pipeline_context",
        lambda ticker, sector=None: {"status": "success", "ticker": ticker, "sector": sector or "tech"},
    )

    monkeypatch.setattr(
        bare_inference_service,
        "_build_forecast_payload",
        lambda context: {
            "ticker": context["ticker"],
            "sector": context["sector"],
            "status": "success",
            "error": "legacy_contract_reason",
            "metadata": {"has_opportunity": True},
        },
    )

    result = bare_inference_service.forecast_hourly("AAPL", sector="tech")

    assert result["status"] == "success"
    assert "error" not in result
    assert result["fallback_reason"] == "legacy_contract_reason"
    assert result["metadata"]["fallback_reason"] == "legacy_contract_reason"


def test_forecast_hourly_unsupported_ticker_returns_error_without_fallback(bare_inference_service, monkeypatch):
    monkeypatch.setattr(
        bare_inference_service,
        "_pipeline_context",
        lambda ticker, sector=None: {
            "status": "error",
            "ticker": ticker.upper().strip(),
            "error": f"Ticker {ticker.upper().strip()} not found in configured sectors",
        },
    )

    def _unexpected_ticker(*args, **kwargs):
        raise AssertionError("forecast fallback should not call yfinance for unsupported ticker")

    monkeypatch.setattr(inference_module.yf, "Ticker", _unexpected_ticker)

    result = bare_inference_service.forecast_hourly("zzzz")

    assert result["status"] == "error"
    assert result["ticker"] == "ZZZZ"
    assert "not found in configured sectors" in result["error"]
    assert not str(result.get("status", "")).startswith("success")
