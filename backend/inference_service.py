"""
Inference service for single-ticker forecasting and options analysis.
Implements the INFERENCE_v3 notebook pipeline with robust API-friendly fallbacks.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
import yfinance as yf

from .utils_2 import (
    LSTMMixedModel,
    apply_robust_normalization,
    calculate_bayesian_final_probability,
    engineer_features,
    get_barrier_probabilities,
)

logger = logging.getLogger(__name__)

SEQ_LEN_DAILY = 20
SEQ_LEN_HOURLY = 40
TBM_HORIZON_DAILY = 2
TBM_HORIZON_HOURLY = 10
TBM_K = 1
SIGNAL_THRESHOLD = 0.01
MIN_INFERENCE_BUFFER = 20
MARKET_TIMEZONE = "America/New_York"

SECTORS_CONFIG = {
    "tech": {
        "display_name": "Technology",
        "model_files": {
            "lstm": "tech_us_model.pth",
            "scaler": "scalers_tech_us.pkl",
            "meta": "meta_model_xgb_tech.json",
            "lstm_hourly": "tech_us_model_hourly.pth",
            "scaler_hourly": "scalers_tech_us_hourly.pkl",
            "meta_hourly": "meta_model_xgb_tech_hourly.json",
        },
        "bayesian": {
            "p_call": 0.3965,
            "p_put": 0.3918,
            "sensitivity": 0.2,
            "specificity": 0.92,
        },
        "bayesian_hourly": {
            "p_call": 0.4305,
            "p_put": 0.4031,
            "sensitivity": 0.42,
            "specificity": 0.84,
        },
        "tickers": [
            "QQQ",
            "META",
            "AAPL",
            "AMZN",
            "NFLX",
            "TSLA",
            "NVDA",
            "PLTR",
            "MSFT",
            "GOOGL",
            "INTC",
            "AMD",
        ],
    },
    "banks": {
        "display_name": "Banks",
        "model_files": {
            "lstm": "banks_model.pth",
            "scaler": "scalers_banks.pkl",
            "meta": "meta_model_xgb_banks.json",
            "lstm_hourly": "banks_model_hourly.pth",
            "scaler_hourly": "scalers_banks_hourly.pkl",
            "meta_hourly": "meta_model_xgb_banks_hourly.json",
        },
        "bayesian": {
            "p_call": 0.4199,
            "p_put": 0.394,
            "sensitivity": 0.32,
            "specificity": 0.89,
        },
        "bayesian_hourly": {
            "p_call": 0.5163,
            "p_put": 0.4631,
            "sensitivity": 0.61,
            "specificity": 0.75,
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
            "meta_hourly": "meta_model_xgb_mining_hourly.json",
        },
        "bayesian": {
            "p_call": 0.3896,
            "p_put": 0.3783,
            "sensitivity": 0.16,
            "specificity": 0.92,
        },
        "bayesian_hourly": {
            "p_call": 0.5261,
            "p_put": 0.52,
            "sensitivity": 0.65,
            "specificity": 0.68,
        },
        "tickers": ["GLD", "SLV", "NEM", "HL", "PAAS", "NUE", "CLF"],
    },
}


class InferenceService:
    """Service for notebook-aligned inference and options analysis."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.market_tz = ZoneInfo(MARKET_TIMEZONE)

        self.daily_models: Dict[str, LSTMMixedModel] = {}
        self.hourly_models: Dict[str, LSTMMixedModel] = {}
        self.daily_scalers: Dict[str, Dict[str, Any]] = {}
        self.hourly_scalers: Dict[str, Dict[str, Any]] = {}
        self.daily_meta_models: Dict[str, Any] = {}
        self.hourly_meta_models: Dict[str, Any] = {}

        logger.info("Using device: %s", self.device)
        self._load_all_models()

    def _to_market_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        converted = df.copy()
        try:
            idx = pd.to_datetime(converted.index)
            if not isinstance(idx, pd.DatetimeIndex):
                return converted
            if idx.tz is None:
                converted.index = idx.tz_localize(self.market_tz)
            else:
                converted.index = idx.tz_convert(self.market_tz)
            return converted
        except Exception:
            return converted

    def _now_market_iso(self) -> str:
        return datetime.now(timezone.utc).astimezone(self.market_tz).isoformat()

    def _load_all_models(self):
        for sector in SECTORS_CONFIG:
            try:
                self._load_sector_models(sector)
                logger.info("Models loaded for sector: %s", sector)
            except Exception as exc:
                logger.warning("Error loading models for %s: %s", sector, exc)

    def _load_lstm_model(self, model_path: str) -> LSTMMixedModel:
        state_dict = torch.load(model_path, map_location=self.device)

        try:
            model = LSTMMixedModel(num_features=23, lstm_hidden=64, dropout=0.3)
            model.load_state_dict(state_dict)
        except RuntimeError:
            in_channels = int(state_dict["cnn_block.0.weight"].shape[1])
            inferred_num_features = in_channels + 1
            model = LSTMMixedModel(
                num_features=inferred_num_features,
                lstm_hidden=64,
                dropout=0.3,
            )
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    def _load_xgb_model(self, model_path: str) -> Any:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    def _load_sector_models(self, sector: str):
        cfg = SECTORS_CONFIG[sector]
        files = cfg["model_files"]

        daily_lstm_path = os.path.join(self.model_dir, files["lstm"])
        hourly_lstm_path = os.path.join(self.model_dir, files["lstm_hourly"])
        daily_scaler_path = os.path.join(self.model_dir, files["scaler"])
        hourly_scaler_path = os.path.join(self.model_dir, files["scaler_hourly"])
        daily_meta_path = os.path.join(self.model_dir, files["meta"])
        hourly_meta_path = os.path.join(self.model_dir, files["meta_hourly"])

        if os.path.exists(daily_lstm_path):
            self.daily_models[sector] = self._load_lstm_model(daily_lstm_path)
        if os.path.exists(hourly_lstm_path):
            self.hourly_models[sector] = self._load_lstm_model(hourly_lstm_path)

        if os.path.exists(daily_scaler_path):
            self.daily_scalers[sector] = joblib.load(daily_scaler_path)
        if os.path.exists(hourly_scaler_path):
            self.hourly_scalers[sector] = joblib.load(hourly_scaler_path)

        if os.path.exists(daily_meta_path):
            self.daily_meta_models[sector] = self._load_xgb_model(daily_meta_path)
        if os.path.exists(hourly_meta_path):
            self.hourly_meta_models[sector] = self._load_xgb_model(hourly_meta_path)

    def get_tickers_by_sector(self, sector: str) -> List[Dict[str, str]]:
        sector_lower = sector.lower()
        if sector_lower not in SECTORS_CONFIG:
            return []

        tickers = []
        for symbol in SECTORS_CONFIG[sector_lower]["tickers"]:
            try:
                ticker_info = yf.Ticker(symbol)
                name = ticker_info.info.get("longName", symbol)
            except Exception:
                name = symbol
            tickers.append({"symbol": symbol, "name": name})
        return tickers

    def get_sector_ticker_mapping(self) -> Dict[str, Dict[str, Any]]:
        result = {"sectors": {}}
        for sector, cfg in SECTORS_CONFIG.items():
            result["sectors"][sector] = {
                "display_name": cfg["display_name"],
                "tickers": cfg["tickers"],
                "model_file": cfg["model_files"].get("lstm_hourly"),
                "count": len(cfg["tickers"]),
            }
        return result

    def _find_sector_for_ticker(self, ticker: str) -> Optional[str]:
        ticker_upper = ticker.upper()
        for sector, cfg in SECTORS_CONFIG.items():
            if ticker_upper in cfg["tickers"]:
                return sector
        return None

    def _compute_barriers(
        self,
        df: pd.DataFrame,
        pred_primary: int,
        horizon: int,
        k: int,
    ) -> Dict[str, float]:
        p_t = float(df["Close"].values[-1])

        vol_t = float(df["vol_20"].values[-1])
        if not np.isfinite(vol_t) or vol_t <= 0:
            vol_t = float(df["Close"].pct_change().dropna().std())
        if not np.isfinite(vol_t) or vol_t <= 0:
            vol_t = 0.01

        drift_val = float(df["drift"].values[-1])
        if not np.isfinite(drift_val):
            drift_val = 0.0

        step_sqrt = np.sqrt(horizon)
        deviation = (drift_val - 0.5 * vol_t**2) * horizon
        upper_barrier = p_t * np.exp(deviation + k * vol_t * step_sqrt)
        lower_barrier = p_t * np.exp(deviation - k * vol_t * step_sqrt)

        is_call = (pred_primary == 1) and (upper_barrier > (p_t * 1.01))
        is_put = (pred_primary == 2) and (lower_barrier < (p_t * 0.99))

        return {
            "p_t": p_t,
            "vol_t": float(vol_t),
            "drift_val": float(drift_val),
            "upper_barrier": float(upper_barrier),
            "lower_barrier": float(lower_barrier),
            "is_call": bool(is_call),
            "is_put": bool(is_put),
        }

    def _run_primary_and_meta(
        self,
        model: LSTMMixedModel,
        meta_model: Optional[Any],
        x_input: np.ndarray,
    ) -> Dict[str, Any]:
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = model(x_tensor)
            probs = F.softmax(logits, dim=1)
            pred_primary = int(torch.argmax(logits, dim=1)[0].item())
            confidence = float(torch.max(probs[0]).item())

            x_raw_feat = x_tensor[0, -10:, :].flatten()
            x_meta_model = (
                torch.cat((probs[0], x_raw_feat)).detach().cpu().numpy().reshape(1, -1)
            )

        if meta_model is None:
            meta_pred = 0
        else:
            try:
                meta_pred = int(meta_model.predict(x_meta_model)[0])
            except Exception:
                meta_pred = 0

        return {
            "pred_primary": pred_primary,
            "meta_pred": meta_pred,
            "confidence": confidence,
            "probs": probs[0].detach().cpu().numpy().astype(float).tolist(),
            "x_tensor": x_tensor,
        }

    def _pipeline_context(self, ticker: str, sector: Optional[str] = None) -> Dict[str, Any]:
        ticker_upper = ticker.upper().strip()

        if sector is None:
            sector = self._find_sector_for_ticker(ticker_upper)
            if sector is None:
                return {
                    "status": "error",
                    "ticker": ticker_upper,
                    "error": f"Ticker {ticker_upper} not found in configured sectors",
                }

        sector_lower = sector.lower()
        if sector_lower not in SECTORS_CONFIG:
            return {
                "status": "error",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "error": f"Unknown sector: {sector_lower}",
            }

        assets_ready = (
            sector_lower in self.daily_models
            and sector_lower in self.hourly_models
            and sector_lower in self.daily_scalers
            and sector_lower in self.hourly_scalers
        )
        if not assets_ready:
            return {
                "status": "error",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "error": f"Missing model or scaler assets for sector {sector_lower}",
            }

        y_ticker = yf.Ticker(ticker_upper)
        hist_daily = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False)
        hist_daily.columns= hist_daily.columns.get_level_values(0)
        del hist_daily['Adj Close']
        
        hist_hourly = yf.download(ticker, period="730d", interval="1h", auto_adjust=False, progress=False)
        hist_hourly.columns= hist_hourly.columns.get_level_values(0)
        del hist_hourly['Adj Close']

        if hist_daily.empty or hist_hourly.empty:
            return {
                "status": "error",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "error": "Historical data unavailable",
            }

        hist_daily = self._to_market_timezone(hist_daily)
        hist_hourly = self._to_market_timezone(hist_hourly)

        df_daily, full_daily = engineer_features(hist_daily.copy(), hist_daily["Close"])
        df_hourly, full_hourly = engineer_features(hist_hourly.copy(), hist_hourly["Close"])

        if full_daily is None or full_hourly is None:
            return {
                "status": "error",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "error": "Feature engineering did not produce sufficient rows",
            }

        if len(full_daily) < (SEQ_LEN_DAILY + MIN_INFERENCE_BUFFER) or len(full_hourly) < (SEQ_LEN_HOURLY + MIN_INFERENCE_BUFFER):
            return {
                "status": "error",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "error": "Insufficient processed feature windows for inference (requires seq_len + 20)",
            }

        daily_scalers = self.daily_scalers[sector_lower]
        hourly_scalers = self.hourly_scalers[sector_lower]

        if ticker_upper not in daily_scalers or ticker_upper not in hourly_scalers:
            return {
                "status": "error",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "error": f"Ticker scaler not available for {ticker_upper}",
            }

        x_daily = full_daily.values[-SEQ_LEN_DAILY:].reshape(1, SEQ_LEN_DAILY, -1)
        x_hourly = full_hourly.values[-SEQ_LEN_HOURLY:].reshape(1, SEQ_LEN_HOURLY, -1)

        x_daily = apply_robust_normalization(x_daily, daily_scalers[ticker_upper])
        x_hourly = apply_robust_normalization(x_hourly, hourly_scalers[ticker_upper])

        daily_pred = self._run_primary_and_meta(
            self.daily_models[sector_lower],
            self.daily_meta_models.get(sector_lower),
            x_daily,
        )
        hourly_pred = self._run_primary_and_meta(
            self.hourly_models[sector_lower],
            self.hourly_meta_models.get(sector_lower),
            x_hourly,
        )

        # Notebook parity: skip ticker early when both primary models are neutral.
        if daily_pred["pred_primary"] == 0 and hourly_pred["pred_primary"] == 0:
            return {
                "status": "no_signal",
                "ticker": ticker_upper,
                "sector": sector_lower,
                "hist_hourly": hist_hourly,
                "daily_pred": daily_pred,
                "hourly_pred": hourly_pred,
                "has_opportunity": False,
                "no_signal_reason": "neutral_primary_skip",
            }

        daily_barriers = self._compute_barriers(df_daily, daily_pred["pred_primary"], TBM_HORIZON_DAILY, TBM_K)
        hourly_barriers = self._compute_barriers(df_hourly, hourly_pred["pred_primary"], TBM_HORIZON_HOURLY, TBM_K)

        # Enforce notebook-style 10-hour gating for recommendations.
        # CALL only if hourly barrier >= current * (1 + threshold)
        # PUT only if hourly barrier <= current * (1 - threshold)
        # Otherwise neutral.
        hourly_current = float(hourly_barriers["p_t"])
        pred_hourly = int(hourly_pred.get("pred_primary", 0))
        if pred_hourly == 1:
            barrier_hourly_signal = float(hourly_barriers["upper_barrier"])
        elif pred_hourly == 2:
            barrier_hourly_signal = float(hourly_barriers["lower_barrier"])
        else:
            barrier_hourly_signal = hourly_current

        upper_threshold = hourly_current * (1.0 + SIGNAL_THRESHOLD)
        lower_threshold = hourly_current * (1.0 - SIGNAL_THRESHOLD)

        if pred_hourly == 1 and barrier_hourly_signal >= upper_threshold:
            op_type = "call"
            target_type = 1
            no_signal_reason = ""
        elif pred_hourly == 2 and barrier_hourly_signal <= lower_threshold:
            op_type = "put"
            target_type = 2
            no_signal_reason = ""
        else:
            op_type = "neutral"
            target_type = 0
            no_signal_reason = "hourly_threshold_not_met" if pred_hourly in {1, 2} else "hourly_primary_neutral"

        hourly_move_pct = (barrier_hourly_signal - hourly_current) / hourly_current if hourly_current > 0 else 0.0
        has_opportunity = target_type in {1, 2}
        barrier = barrier_hourly_signal
        barrier_hourly = barrier_hourly_signal

        return {
            "status": "success",
            "ticker": ticker_upper,
            "sector": sector_lower,
            "y_ticker": y_ticker,
            "hist_hourly": hist_hourly,
            "daily_pred": daily_pred,
            "hourly_pred": hourly_pred,
            "daily_barriers": daily_barriers,
            "hourly_barriers": hourly_barriers,
            "op_type": op_type,
            "target_type": target_type,
            "barrier": float(barrier),
            "barrier_hourly": float(barrier_hourly),
            "hourly_move_pct": float(hourly_move_pct),
            "is_neutral": bool(target_type == 0),
            "has_opportunity": has_opportunity,
            "no_signal_reason": no_signal_reason,
        }

    def _generate_deterministic_forecast_prices(
        self,
        current_price: float,
        barrier_hourly: float,
        confidence_hourly: float,
        op_type: str,
        historical_data: pd.DataFrame,
        horizon_hours: int = 10,
    ) -> List[float]:
        returns = historical_data["Close"].pct_change().dropna()
        trend = float(returns.tail(12).mean()) if not returns.empty else 0.0
        confidence = float(np.clip(confidence_hourly, 0.2, 0.95))
        curvature = float(1.0 + (1.0 - confidence) * 0.6)

        forecast = []
        for step in range(1, horizon_hours + 1):
            alpha = step / float(horizon_hours)
            weighted_alpha = alpha**curvature
            base_target = current_price + (barrier_hourly - current_price) * weighted_alpha
            trend_adj = current_price * trend * step * 0.35
            mean_revert = (current_price - base_target) * (1.0 - confidence) * (1.0 - alpha) * 0.12

            price = max(base_target + trend_adj + mean_revert, 0.01)
            if op_type == "call":
                price = max(price, current_price * 0.96)
            elif op_type == "put":
                price = min(price, current_price * 1.04)

            forecast.append(float(price))

        return forecast

    def _format_last_40_ohlc(self, hist_hourly: pd.DataFrame) -> List[Dict[str, Any]]:
        hist_40 = hist_hourly.iloc[-40:].copy()
        rows = []
        for idx, row in hist_40.iterrows():
            rows.append(
                {
                    "timestamp": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if "Volume" in row else 0,
                    "is_historical": True,
                }
            )
        return rows

    def _build_forecast_payload(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hist_hourly = context["hist_hourly"]
        hist_40 = hist_hourly.iloc[-40:].copy()

        current_price = float(hist_40["Close"].iloc[-1])
        confidence_hourly = float(context["hourly_pred"]["confidence"])

        forecast_prices = self._generate_deterministic_forecast_prices(
            current_price=current_price,
            barrier_hourly=float(context["barrier_hourly"]),
            confidence_hourly=confidence_hourly,
            op_type=context["op_type"],
            historical_data=hist_40,
            horizon_hours=10,
        )

        forecast_rows = []
        base_ts = hist_40.index[-1]
        for i, price in enumerate(forecast_prices, 1):
            step_conf = float(max(0.25, confidence_hourly * (1.0 - i * 0.03)))
            forecast_rows.append(
                {
                    "timestamp": (base_ts + timedelta(hours=i)).isoformat(),
                    "forecast": float(price),
                    "confidence": step_conf,
                    "is_forecast": True,
                }
            )

        forecast_arr = np.array(forecast_prices)

        return {
            "ticker": context["ticker"],
            "sector": context["sector"],
            "status": "success" if context["has_opportunity"] else "success_no_signal",
            "last_40_hours": self._format_last_40_ohlc(hist_hourly),
            "forecast_10_hours": forecast_rows,
            "median_price": float(hist_40["Close"].median()),
            "forecast_high": float(np.max(forecast_arr)),
            "forecast_low": float(np.min(forecast_arr)),
            "current_price": current_price,
            "op_type": context["op_type"],
            "is_neutral": bool(context.get("is_neutral", context["target_type"] == 0)),
            "movement_10h_pct": float(context.get("hourly_move_pct", 0.0)),
            "barrier": float(context["barrier"]),
            "barrier_hourly": float(context["barrier_hourly"]),
            "confidence_daily": float(context["daily_pred"]["confidence"]),
            "confidence_hourly": confidence_hourly,
            "metadata": {
                "has_opportunity": bool(context["has_opportunity"]),
                "target_type": int(context["target_type"]),
                "is_neutral": bool(context.get("is_neutral", context["target_type"] == 0)),
                "hourly_move_pct": float(context.get("hourly_move_pct", 0.0)),
                "pred_primary_daily": int(context["daily_pred"]["pred_primary"]),
                "pred_primary_hourly": int(context["hourly_pred"]["pred_primary"]),
                "meta_pred_daily": int(context["daily_pred"]["meta_pred"]),
                "meta_pred_hourly": int(context["hourly_pred"]["meta_pred"]),
                "daily_probs": context["daily_pred"]["probs"],
                "hourly_probs": context["hourly_pred"]["probs"],
                "daily_barriers": {
                    "upper": float(context["daily_barriers"]["upper_barrier"]),
                    "lower": float(context["daily_barriers"]["lower_barrier"]),
                },
                "hourly_barriers": {
                    "upper": float(context["hourly_barriers"]["upper_barrier"]),
                    "lower": float(context["hourly_barriers"]["lower_barrier"]),
                },
            },
            "last_update": self._now_market_iso(),
        }

    def _build_no_signal_forecast_payload(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hist_hourly = context["hist_hourly"]
        hist_40 = hist_hourly.iloc[-40:].copy()
        current_price = float(hist_40["Close"].iloc[-1])
        confidence_hourly = float(context["hourly_pred"]["confidence"])

        forecast_prices = self._generate_deterministic_forecast_prices(
            current_price=current_price,
            barrier_hourly=current_price,
            confidence_hourly=confidence_hourly,
            op_type="neutral",
            historical_data=hist_40,
            horizon_hours=10,
        )

        forecast_rows = []
        base_ts = hist_40.index[-1]
        for i, price in enumerate(forecast_prices, 1):
            step_conf = float(max(0.25, confidence_hourly * (1.0 - i * 0.03)))
            forecast_rows.append(
                {
                    "timestamp": (base_ts + timedelta(hours=i)).isoformat(),
                    "forecast": float(price),
                    "confidence": step_conf,
                    "is_forecast": True,
                }
            )

        forecast_arr = np.array(forecast_prices)

        return {
            "ticker": context["ticker"],
            "sector": context["sector"],
            "status": "success_no_signal",
            "last_40_hours": self._format_last_40_ohlc(hist_hourly),
            "forecast_10_hours": forecast_rows,
            "median_price": float(hist_40["Close"].median()),
            "forecast_high": float(np.max(forecast_arr)),
            "forecast_low": float(np.min(forecast_arr)),
            "current_price": current_price,
            "op_type": "neutral",
            "is_neutral": True,
            "movement_10h_pct": 0.0,
            "barrier": current_price,
            "barrier_hourly": current_price,
            "confidence_daily": float(context["daily_pred"]["confidence"]),
            "confidence_hourly": confidence_hourly,
            "metadata": {
                "has_opportunity": False,
                "target_type": 0,
                "is_neutral": True,
                "hourly_move_pct": 0.0,
                "pred_primary_daily": int(context["daily_pred"]["pred_primary"]),
                "pred_primary_hourly": int(context["hourly_pred"]["pred_primary"]),
                "meta_pred_daily": int(context["daily_pred"]["meta_pred"]),
                "meta_pred_hourly": int(context["hourly_pred"]["meta_pred"]),
                "daily_probs": context["daily_pred"]["probs"],
                "hourly_probs": context["hourly_pred"]["probs"],
                "no_signal_reason": context.get("no_signal_reason", "neutral_primary_skip"),
            },
            "last_update": self._now_market_iso(),
        }

    def _sanitize_success_forecast_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        status = str(payload.get("status", ""))
        if not status.startswith("success"):
            return payload

        error_detail = payload.pop("error", None)
        if error_detail is None:
            return payload

        fallback_reason = str(error_detail)
        payload.setdefault("fallback_reason", fallback_reason)

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata
        metadata.setdefault("fallback_reason", fallback_reason)

        return payload

    def _is_unsupported_context_error(self, context: Dict[str, Any]) -> bool:
        if context.get("status") != "error":
            return False

        error_text = str(context.get("error", "")).lower()
        return (
            "not found in configured sectors" in error_text
            or "unknown sector:" in error_text
        )

    def _fallback_forecast(
        self,
        ticker: str,
        sector: str,
        hist_data: pd.DataFrame,
        error: str,
    ) -> Dict[str, Any]:
        hist_40 = hist_data.iloc[-40:].copy() if len(hist_data) >= 40 else hist_data.copy()
        current_price = float(hist_40["Close"].iloc[-1])
        returns = hist_40["Close"].pct_change().dropna()
        drift = float(returns.mean()) if not returns.empty else 0.0

        forecast_prices = [max(current_price * float(np.exp(drift * h)), 0.01) for h in range(1, 11)]

        forecast_rows = []
        last_time = hist_40.index[-1]
        for i, price in enumerate(forecast_prices, 1):
            forecast_rows.append(
                {
                    "timestamp": (last_time + timedelta(hours=i)).isoformat(),
                    "forecast": float(price),
                    "confidence": 0.4,
                    "is_forecast": True,
                }
            )

        return {
            "ticker": ticker,
            "sector": sector,
            "status": "success_fallback",
            "fallback_reason": error,
            "last_40_hours": self._format_last_40_ohlc(hist_40),
            "forecast_10_hours": forecast_rows,
            "median_price": float(hist_40["Close"].median()),
            "forecast_high": float(np.max(forecast_prices)),
            "forecast_low": float(np.min(forecast_prices)),
            "current_price": current_price,
            "op_type": "neutral",
            "barrier": current_price,
            "barrier_hourly": current_price,
            "confidence_daily": 0.4,
            "confidence_hourly": 0.4,
            "metadata": {
                "has_opportunity": False,
                "target_type": 0,
                "fallback_reason": error,
            },
            "last_update": self._now_market_iso(),
        }

    def forecast_hourly(self, ticker: str, sector: str = None) -> Dict[str, Any]:
        try:
            context = self._pipeline_context(ticker=ticker, sector=sector)
            if context.get("status") == "error":
                if self._is_unsupported_context_error(context):
                    return context

                ticker_upper = ticker.upper().strip()
                sector_resolved = sector or self._find_sector_for_ticker(ticker_upper) or "unknown"
                try:
                    hist_hourly = yf.Ticker(ticker_upper).history(period="7d", interval="1h")
                    if hist_hourly.empty:
                        return context
                    hist_hourly = self._to_market_timezone(hist_hourly)
                    return self._sanitize_success_forecast_payload(
                        self._fallback_forecast(
                            ticker=ticker_upper,
                            sector=sector_resolved,
                            hist_data=hist_hourly,
                            error=context.get("error", "pipeline_error"),
                        )
                    )
                except Exception:
                    return context

            if context.get("status") == "no_signal":
                return self._sanitize_success_forecast_payload(
                    self._build_no_signal_forecast_payload(context)
                )

            return self._sanitize_success_forecast_payload(self._build_forecast_payload(context))
        except Exception as exc:
            logger.error("Error in forecast_hourly: %s", exc, exc_info=True)
            return {
                "ticker": ticker,
                "sector": sector or "unknown",
                "status": "error",
                "error": str(exc),
            }

    def build_options_analysis(
        self,
        ticker: str,
        sentiment_probs: Optional[np.ndarray] = None,
        sentiment_score: float = 0.0,
        sector: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            context = self._pipeline_context(ticker=ticker, sector=sector)
            if context.get("status") == "no_signal":
                hist_hourly = context.get("hist_hourly")
                current_price = (
                    float(hist_hourly["Close"].iloc[-1])
                    if isinstance(hist_hourly, pd.DataFrame) and not hist_hourly.empty
                    else 0.0
                )
                return {
                    "ticker": context.get("ticker", ticker.upper().strip()),
                    "sector": context.get("sector", sector),
                    "status": "success_no_signal",
                    "op_type": "neutral",
                    "is_neutral": True,
                    "target_type": 0,
                    "current_price": current_price,
                    "barrier": current_price,
                    "barrier_hourly": current_price,
                    "movement_10h_pct": 0.0,
                    "confidence_daily": float(context.get("daily_pred", {}).get("confidence", 0.0)),
                    "confidence_hourly": float(context.get("hourly_pred", {}).get("confidence", 0.0)),
                    "table_rows": [],
                    "suggested_options": [],
                    "sentiment_score": float(sentiment_score),
                    "no_signal_reason": context.get("no_signal_reason", "neutral_primary_skip"),
                    "timestamp": self._now_market_iso(),
                }

            if context.get("status") == "error":
                return {
                    "ticker": ticker.upper().strip(),
                    "sector": sector,
                    "status": "error",
                    "error": context.get("error", "inference_error"),
                    "table_rows": [],
                    "suggested_options": [],
                }

            ticker_upper = context["ticker"]
            sector_lower = context["sector"]
            has_opportunity = bool(context["has_opportunity"])

            if sentiment_probs is None:
                sentiment_probs = np.array([])

            table_rows: List[Dict[str, Any]] = []
            suggested_options: List[Dict[str, Any]] = []

            if has_opportunity and context["op_type"] in {"call", "put"}:
                barrier_df = get_barrier_probabilities(
                    context["y_ticker"],
                    context["barrier"],
                    context["op_type"],
                )

                if not barrier_df.empty:
                    b_params = SECTORS_CONFIG[sector_lower]["bayesian"]
                    b_params_hourly = SECTORS_CONFIG[sector_lower]["bayesian_hourly"]

                    precision_base = b_params["p_call"] if context["target_type"] == 1 else b_params["p_put"]
                    precision_base_hourly = b_params_hourly["p_call"] if context["target_type"] == 1 else b_params_hourly["p_put"]

                    for _, row in barrier_df.iterrows():
                        p_final, score = calculate_bayesian_final_probability(
                            target_type=context["target_type"],
                            pred_primary=context["daily_pred"]["pred_primary"],
                            meta_pred=context["daily_pred"]["meta_pred"],
                            precision_base=precision_base,
                            recall_meta=b_params["sensitivity"],
                            spec_meta=b_params["specificity"],
                            pred_primary_hourly=context["hourly_pred"]["pred_primary"],
                            meta_pred_hourly=context["hourly_pred"]["meta_pred"],
                            precision_base_hourly=precision_base_hourly,
                            recall_meta_hourly=b_params_hourly["sensitivity"],
                            spec_meta_hourly=b_params_hourly["specificity"],
                            probs_news=sentiment_probs,
                            p_market=float(row["Prob. Toca Strike"]),
                            w_market=0.3,
                        )

                        row_dict = {
                            "Fecha": datetime.now().strftime("%Y-%m-%d"),
                            "Sector": sector_lower,
                            "Ticker": ticker_upper,
                            "Tipo": context["op_type"].upper(),
                            "Precio Actual": round(float(context["daily_barriers"]["p_t"]), 2),
                            "Barrera a 10 horas": round(float(context["barrier_hourly"]), 2),
                            "Vencimiento": str(row["Vencimiento"]),
                            "Strike": float(row["Strike Seleccionado"]),
                            "Ask": float(row["Ask"]),
                            "IV": float(row["IV"]),
                            "Prob. Mercado": float(row["Prob. Toca Strike"]),
                            "Prob. Final (Bayes)": round(float(p_final), 4),
                            "Sentimiento Score": round(float(score if np.isfinite(score) else sentiment_score), 4),
                        }
                        table_rows.append(row_dict)

                        suggested_options.append(
                            {
                                "option_type": context["op_type"],
                                "strike": row_dict["Strike"],
                                "expiration": row_dict["Vencimiento"],
                                "ask": row_dict["Ask"],
                                "iv": row_dict["IV"],
                                "market_probability": row_dict["Prob. Mercado"],
                                "probability": row_dict["Prob. Final (Bayes)"],
                                "recommendation": "Buy" if row_dict["Prob. Final (Bayes)"] >= 0.6 else "Consider",
                            }
                        )

            table_rows.sort(key=lambda item: item.get("Prob. Final (Bayes)", 0), reverse=True)
            suggested_options.sort(key=lambda item: item.get("probability", 0), reverse=True)

            status = "success" if table_rows else ("success_no_signal" if not has_opportunity else "success_no_options")

            payload = {
                "ticker": ticker_upper,
                "sector": sector_lower,
                "status": status,
                "op_type": context["op_type"],
                "is_neutral": bool(context.get("is_neutral", context["target_type"] == 0)),
                "target_type": int(context["target_type"]),
                "current_price": float(context["daily_barriers"]["p_t"]),
                "barrier": float(context["barrier"]),
                "barrier_hourly": float(context["barrier_hourly"]),
                "movement_10h_pct": float(context.get("hourly_move_pct", 0.0)),
                "confidence_daily": float(context["daily_pred"]["confidence"]),
                "confidence_hourly": float(context["hourly_pred"]["confidence"]),
                "table_rows": table_rows,
                "suggested_options": suggested_options,
                "sentiment_score": float(sentiment_score),
                "timestamp": self._now_market_iso(),
            }

            if status == "success_no_signal":
                payload["no_signal_reason"] = context.get("no_signal_reason", "barrier_criteria_not_met")

            return payload
        except Exception as exc:
            logger.error("Error building options analysis for %s: %s", ticker, exc, exc_info=True)
            return {
                "ticker": ticker.upper().strip(),
                "status": "error",
                "error": str(exc),
                "table_rows": [],
                "suggested_options": [],
                "timestamp": self._now_market_iso(),
            }


def create_inference_service(model_dir: str = "models") -> InferenceService:
    """Factory function to create inference service."""
    return InferenceService(model_dir=model_dir)
