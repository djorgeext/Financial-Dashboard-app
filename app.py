"""
Financial Dashboard Backend API
Main FastAPI application with endpoints for price data, forecasts, and sentiment.
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import Config, get_config
from model_service import create_model_service, ModelService
from data_fetcher import create_data_fetcher, DataFetcher
from news_service import create_news_service, NewsService
from inference_service import create_inference_service, InferenceService
from options_analyzer import create_options_analyzer, OptionsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
config: Config = None
model_service: ModelService = None
data_fetcher: DataFetcher = None
news_service: NewsService = None
inference_service: InferenceService = None
options_analyzer: OptionsAnalyzer = None


def _score_to_news_probs(score: float) -> np.ndarray:
    """Convert scalar sentiment score [-1, 1] to pseudo class probabilities [neg, neu, pos]."""
    clipped = float(np.clip(score, -1.0, 1.0))
    pos = max(0.0, clipped)
    neg = max(0.0, -clipped)
    neu = max(0.0, 1.0 - abs(clipped))
    probs = np.array([neg, neu, pos], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.array([0.0, 1.0, 0.0], dtype=float)
    return probs.reshape(1, -1)


def _is_unsupported_ticker_or_sector_error(error_detail: Optional[str]) -> bool:
    message = str(error_detail or "").lower()
    return (
        "not found in configured sectors" in message
        or "unknown sector:" in message
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown.
    Initializes services on startup, cleans up on shutdown.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Financial Dashboard API Starting Up")
    logger.info("=" * 60)
    
    global config, model_service, data_fetcher, news_service, inference_service, options_analyzer
    
    config = get_config()
    logger.info(f"Environment: {config.ENV}")
    logger.info(f"Debug Mode: {config.DEBUG}")
    logger.info(f"Model Directory: {config.MODEL_DIR}")
    
    # Initialize services
    model_service = create_model_service(model_dir=config.MODEL_DIR)
    data_fetcher = create_data_fetcher(cache_ttl=config.DATA_CACHE_TTL)
    inference_service = create_inference_service(model_dir=config.MODEL_DIR)
    options_analyzer = create_options_analyzer()
    
    # Initialize Groq news service
    try:
        groq_key = config.get_groq_api_key()
        if groq_key:
            logger.info("Using local groq key")
            news_service = create_news_service(api_key=groq_key, cache_ttl=config.NEWS_CACHE_TTL)
        else:
            logger.warning("No Groq API key available - news analysis disabled")
            news_service = create_news_service(cache_ttl=config.NEWS_CACHE_TTL)
    except Exception as e:
        logger.error(f"Error initializing news service: {str(e)}")
        news_service = create_news_service(cache_ttl=config.NEWS_CACHE_TTL)
    
    logger.info("Services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Financial Dashboard API Shutting Down")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Financial Dashboard API",
    description="Real-time financial data and ML forecasts",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== STATIC ENDPOINTS ====================

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint - returns 204 No Content."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/api/debug/status")
async def debug_status() -> Dict:
    """Debug endpoint to check service status."""
    return {
        "config_ready": config is not None,
        "model_service_ready": model_service is not None,
        "data_fetcher_ready": data_fetcher is not None,
        "news_service_ready": news_service is not None,
        "inference_service_ready": inference_service is not None,
        "options_analyzer_ready": options_analyzer is not None,
        "timestamp": datetime.now().isoformat()
    }


# ==================== HEALTH & STATUS ENDPOINTS ====================

@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "financial-dashboard"
    }


@app.get("/ready")
async def readiness_probe() -> Dict:
    """
    Readiness probe - validates data pipeline and model loading.
    Returns 200 if ready, 503 if not.
    """
    try:
        # Check data fetcher
        if data_fetcher is None:
            raise Exception("Data fetcher not initialized")
        
        # Try fetching recent data for a major index
        test_data = data_fetcher.get_price_stats("AAPL", days=1)
        if test_data is None:
            raise Exception("Failed to fetch test data")
        
        # Check model service
        if model_service is None:
            raise Exception("Model service not initialized")
        
        return {
            "status": "ready",
            "data_pipeline": "operational",
            "models_available": model_service.get_model_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/pipeline-health")
async def pipeline_health() -> Dict:
    """Get detailed pipeline health status."""
    try:
        # Data freshness
        price_data = data_fetcher.get_price_stats("AAPL", days=1)
        data_fresh = price_data is not None
        
        # Model status
        model_status = model_service.get_model_status()
        models_ready = len(model_status) > 0 if model_status else False
        
        # News service status
        news_status = news_service.get_news_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_freshness": data_fresh,
            "models_ready": models_ready,
            "news_analysis": news_status.get("client_ready", False),
            "groq_available": news_status.get("groq_available", False),
            "all_systems": data_fresh and models_ready
        }
    except Exception as e:
        logger.error(f"Error checking pipeline health: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "all_systems": False
        }


# ==================== FINANCIAL DATA ENDPOINTS ====================

@app.get("/api/financial/summary/{sector}")
async def get_financial_summary(sector: str) -> Dict:
    """
    Get current financial data for a sector.
    
    Args:
        sector: Sector name ('tech', 'banks', 'mining')
    """
    sector_lower = sector.lower()
    
    if sector_lower not in config.SECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sector: {sector_lower}. Available: {list(config.SECTORS.keys())}"
        )
    
    try:
        sector_config = config.SECTORS[sector_lower]
        tickers = sector_config.get("tickers", [])
        
        summary = data_fetcher.get_sector_summary(tickers)
        
        return {
            "sector": sector_lower,
            "display_name": sector_config.get("display_name"),
            "timestamp": datetime.now().isoformat(),
            "data": summary,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting financial summary for {sector}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FORECAST ENDPOINTS ====================

@app.get("/api/forecast/hourly/{ticker}")
async def get_hourly_forecast(ticker: str) -> Dict:
    """
    Get hourly forecast with OHLC data for next 10 hours.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Forecast data with last 40 hours historical + 10 hour forecast
    """
    if not ticker or not isinstance(ticker, str):
        raise HTTPException(status_code=400, detail="Invalid ticker parameter")
    
    if inference_service is None:
        logger.error("Inference service not initialized")
        raise HTTPException(status_code=503, detail="Service not ready")
    
    ticker_upper = ticker.upper().strip()
    
    try:
        logger.info(f"Generating forecast for {ticker_upper}")
        forecast = inference_service.forecast_hourly(ticker_upper)
        
        if forecast is None:
            raise HTTPException(status_code=500, detail="Forecast returned None")
        
        if forecast.get("status") == "error":
            logger.warning(f"Forecast error for {ticker_upper}: {forecast.get('error')}")
            raise HTTPException(
                status_code=400,
                detail=forecast.get("error", "Forecast generation failed")
            )

        status = str(forecast.get("status", ""))
        if status.startswith("success") and "error" in forecast:
            fallback_reason = str(forecast.pop("error"))
            forecast.setdefault("fallback_reason", fallback_reason)
            metadata = forecast.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                forecast["metadata"] = metadata
            metadata.setdefault("fallback_reason", fallback_reason)
        
        logger.info(f"Successfully generated forecast for {ticker_upper}")
        return forecast
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting hourly forecast for {ticker}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.get("/api/forecast/{sector}/{granularity}")
async def get_forecast(
    sector: str,
    granularity: str = Path(..., pattern="^(daily|hourly)$")
) -> Dict:
    """
    Get model forecast for a sector.
    
    Args:
        sector: Sector name ('tech', 'banks', 'mining')
        granularity: 'daily' or 'hourly'
    """
    sector_lower = sector.lower()
    granularity_lower = granularity.lower()
    
    if sector_lower not in config.SECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sector: {sector_lower}"
        )
    
    try:
        sector_config = config.SECTORS[sector_lower]
        
        # Select model based on granularity
        if granularity_lower == "hourly":
            model_name = sector_config.get("model_hourly", "").replace(".pth", "")
        else:
            model_name = sector_config.get("model_daily", "").replace(".pth", "")
        
        if not model_name or not model_service.model_ready(model_name):
            logger.warning(f"Model not loaded: {model_name}. Returning placeholder.")
            return {
                "sector": sector_lower,
                "granularity": granularity_lower,
                "status": "model_not_loaded",
                "forecast": {
                    "trend": "neutral",
                    "confidence": 0.33,
                    "classes": {"down": 0.33, "neutral": 0.33, "up": 0.33}
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Get tickers and fetch data
        tickers = sector_config.get("tickers", [])
        if not tickers:
            raise Exception(f"No tickers configured for {sector_lower}")
        
        # Use first ticker for forecast demo
        ticker = tickers[0]
        features = data_fetcher.prepare_features_for_model(
            ticker,
            sequence_length=config.SEQUENCE_LENGTH,
            interval="1h" if granularity_lower == "hourly" else "1d"
        )
        
        if features is None:
            raise Exception(f"Failed to prepare features for {ticker}")
        
        # Run prediction
        result = model_service.predict(model_name, features)
        
        if "error" in result:
            return {
                "sector": sector_lower,
                "granularity": granularity_lower,
                "status": "inference_error",
                "error": result.get("error"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Format forecast response
        probs = result["probabilities"][0] if result["probabilities"] else [0.33, 0.33, 0.33]
        predicted_class = result["predicted_class"][0] if result["predicted_class"] else 1
        confidence = result["confidence"][0] if result["confidence"] else 0.33
        
        class_names = ["down", "neutral", "up"]
        
        return {
            "sector": sector_lower,
            "granularity": granularity_lower,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "forecast": {
                "trend": class_names[int(predicted_class)],
                "confidence": float(confidence),
                "classes": {
                    "down": float(probs[0]),
                    "neutral": float(probs[1]),
                    "up": float(probs[2])
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting forecast for {sector}/{granularity}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NEWS ENDPOINTS ====================

@app.get("/api/news/{sector}")
async def get_news(sector: str) -> Dict:
    """
    Get news sentiment and signals for a sector.
    
    Args:
        sector: Sector name ('tech', 'banks', 'mining')
    """
    sector_lower = sector.lower()
    
    if sector_lower not in config.SECTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sector: {sector_lower}"
        )
    
    try:
        tickers = config.SECTORS[sector_lower].get("tickers", [])[:3]
        ticker_results = []
        for ticker_symbol in tickers:
            ticker_results.append(
                news_service.analyze_ticker_news(
                    ticker=ticker_symbol,
                    sector=sector_lower,
                    bank_name=ticker_symbol,
                    max_items=4,
                )
            )

        if ticker_results:
            scores = [float(item.get("sentiment_score", 0.0)) for item in ticker_results]
            avg_score = float(np.mean(scores)) if scores else 0.0
            if avg_score > 0.15:
                sector_sentiment = "bullish"
            elif avg_score < -0.15:
                sector_sentiment = "bearish"
            else:
                sector_sentiment = "neutral"

            merged_signals = []
            for item in ticker_results:
                for sig in item.get("key_signals", []):
                    if sig not in merged_signals:
                        merged_signals.append(sig)

            analysis = {
                "sentiment": sector_sentiment,
                "sentiment_score": avg_score,
                "signals": merged_signals[:8],
                "summary": " ".join(
                    [
                        str(item.get("bank_summary_en", "")).strip()
                        for item in ticker_results
                        if item.get("bank_summary_en")
                    ][:2]
                )
                or "No summarizable news.",
                "ticker_summaries": [
                    {
                        "ticker": item.get("ticker"),
                        "sentiment": item.get("sentiment", "neutral"),
                        "sentiment_score": item.get("sentiment_score", 0.0),
                        "status": item.get("status", "unknown"),
                    }
                    for item in ticker_results
                ],
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }
            news_count = int(sum(item.get("metrics", {}).get("retained", 0) for item in ticker_results))
        else:
            analysis = news_service.analyze_news(sector_lower, [])
            news_count = 0

        return {
            "sector": sector_lower,
            "timestamp": datetime.now().isoformat(),
            "news_count": news_count,
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting news for {sector}: {str(e)}")
        return {
            "sector": sector_lower,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "analysis": news_service.get_safe_default_response()
        }


# ==================== AGGREGATED DASHBOARD ENDPOINT ====================

@app.get("/api/dashboard")
async def get_dashboard() -> Dict:
    """
    Get aggregated dashboard data combining prices, forecasts, and sentiment.
    Ready for UI rendering.
    """
    try:
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "sectors": {}
        }
        
        for sector_key, sector_config in config.SECTORS.items():
            try:
                # Get financial summary
                financial = await get_financial_summary(sector_key)
                
                # Get daily forecast
                forecast = await get_forecast(sector_key, "daily")
                
                # Get news analysis
                news = await get_news(sector_key)
                
                dashboard_data["sectors"][sector_key] = {
                    "display_name": sector_config.get("display_name"),
                    "financial": financial.get("data", {}),
                    "forecast": forecast.get("forecast", {}),
                    "news": news.get("analysis", {}),
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error loading dashboard data for {sector_key}: {str(e)}")
                dashboard_data["sectors"][sector_key] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return dashboard_data
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ENHANCED FORECAST ENDPOINTS ====================

@app.get("/api/tickers/{sector}")
async def get_tickers_by_sector(sector: str) -> Dict:
    """
    Get list of available tickers for a sector.
    
    Args:
        sector: Sector name (tech, banks, mining)
        
    Returns:
        List of tickers with names
    """
    sector_lower = sector.lower()
    
    try:
        tickers = inference_service.get_tickers_by_sector(sector_lower)
        
        if not tickers:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sector: {sector_lower}. Available: tech, banks, mining"
            )
        
        return {
            "sector": sector_lower,
            "tickers": tickers,
            "count": len(tickers),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tickers for sector {sector}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/options/{ticker}")
async def get_options_suggestions(ticker: str) -> Dict:
    """
    Get suggested options trading strategies based on forecast.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of recommended options with probabilities
    """
    if not ticker or not isinstance(ticker, str):
        raise HTTPException(status_code=400, detail="Invalid ticker parameter")
    
    if inference_service is None or news_service is None:
        logger.error("Required services not initialized")
        raise HTTPException(status_code=503, detail="Service not ready")
    
    ticker_upper = ticker.upper().strip()
    
    try:
        logger.info(f"Generating options for {ticker_upper}")
        sector = None
        ticker_map = inference_service.get_sector_ticker_mapping()
        for sec, data in ticker_map.get("sectors", {}).items():
            if ticker_upper in data.get("tickers", []):
                sector = sec
                break

        news_result = news_service.analyze_ticker_news(
            ticker=ticker_upper,
            sector=sector,
            bank_name=ticker_upper,
            max_items=8,
        )
        sentiment_score = float(news_result.get("sentiment_score", 0.0))
        sentiment_probs = _score_to_news_probs(sentiment_score)

        options = inference_service.build_options_analysis(
            ticker=ticker_upper,
            sector=sector,
            sentiment_probs=sentiment_probs,
            sentiment_score=sentiment_score,
        )

        options["news"] = {
            "sentiment": news_result.get("sentiment", "neutral"),
            "sentiment_score": sentiment_score,
            "summary": news_result.get("summary", ""),
            "status": news_result.get("status", "unknown"),
        }

        if (
            not options.get("table_rows")
            and options_analyzer is not None
            and options.get("status") == "error"
            and not _is_unsupported_ticker_or_sector_error(options.get("error"))
        ):
            fallback_reason = options.get("error")
            fallback_failure_detail = None

            try:
                forecast = inference_service.forecast_hourly(ticker_upper, sector=sector)
                current_price = float(forecast.get("current_price", 0.0))
                forecast_price = float(
                    (forecast.get("forecast_10_hours", [{}])[-1] or {}).get("forecast", current_price)
                )
                price_change = (forecast_price - current_price) / current_price if current_price > 0 else 0.0
                pred_class = 2 if price_change > 0.005 else 0 if price_change < -0.005 else 1
                confidences = [
                    float(point.get("confidence", 0.5))
                    for point in forecast.get("forecast_10_hours", [])
                    if isinstance(point, dict)
                ]
                avg_confidence = float(np.mean(confidences)) if confidences else 0.5

                fallback = options_analyzer.suggest_options(
                    ticker=ticker_upper,
                    current_price=current_price,
                    forecast_price=forecast_price,
                    forecast_confidence=avg_confidence,
                    pred_class=pred_class,
                )

                suggested_options = fallback.get("suggested_options", []) if isinstance(fallback, dict) else []
                fallback_error = fallback.get("error") if isinstance(fallback, dict) else "invalid_fallback_response"

                if fallback_error:
                    fallback_failure_detail = str(fallback_error)
                elif not isinstance(suggested_options, list) or not suggested_options:
                    fallback_failure_detail = "empty_suggestions"
                else:
                    options["status"] = "success_fallback"
                    options.pop("error", None)
                    options["table_rows"] = options.get("table_rows") or []
                    options["suggested_options"] = suggested_options
                    if fallback_reason:
                        options["fallback_reason"] = fallback_reason
            except Exception as fallback_exc:
                fallback_failure_detail = str(fallback_exc)

            if fallback_failure_detail:
                options["status"] = "error"
                options["error"] = "fallback_failed"
                options["table_rows"] = options.get("table_rows") or []
                options["suggested_options"] = options.get("suggested_options") or []
                if fallback_reason:
                    options["fallback_reason"] = fallback_reason
                options["fallback_detail"] = fallback_failure_detail

        return options
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting options for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/tickers-with-models")
async def get_available_tickers() -> Dict:
    """
    Get all tickers configured with trained models grouped by sector.
    
    Returns:
        Mapping of sectors to tickers and model info
    """
    try:
        return inference_service.get_sector_ticker_mapping()
    except Exception as e:
        logger.error(f"Error getting ticker mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NEWS ENDPOINTS ====================

@app.get("/api/news/ticker/{ticker}")
async def get_ticker_news(ticker: str) -> Dict:
    """
    Get news sentiment analysis for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Sentiment analysis and key news signals
    """
    ticker_upper = ticker.upper()
    
    try:
        sector = None
        ticker_map = inference_service.get_sector_ticker_mapping()
        for sec, data in ticker_map.get("sectors", {}).items():
            if ticker_upper in data.get("tickers", []):
                sector = sec
                break

        analysis = news_service.analyze_ticker_news(
            ticker=ticker_upper,
            sector=sector,
            bank_name=ticker_upper,
            max_items=10,
        )

        return {
            "ticker": ticker_upper,
            "sector": sector,
            "sentiment": analysis.get("sentiment", "neutral"),
            "sentiment_score": float(analysis.get("sentiment_score", 0.0)),
            "summary": analysis.get("summary", analysis.get("bank_summary_en", "")),
            "key_signals": analysis.get("key_signals", []),
            "news_summaries": analysis.get("news_summaries", []),
            "metrics": analysis.get("metrics", {}),
            "last_updated": analysis.get("timestamp", datetime.now().isoformat()),
            "status": analysis.get("status", "success"),
        }
    except Exception as e:
        logger.error(f"Error getting news for {ticker}: {str(e)}")
        return {
            "ticker": ticker_upper,
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "summary": "News analysis unavailable",
            "key_signals": [],
            "last_updated": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }


# ==================== FRONTEND ENDPOINTS ====================

@app.get("/")
async def serve_root() -> HTMLResponse:
    """Serve the dashboard HTML."""
    return FileResponse("dashboard.html", media_type="text/html")


@app.get("/dashboard")
async def serve_dashboard() -> HTMLResponse:
    """Serve the dashboard HTML at /dashboard."""
    return FileResponse("dashboard.html", media_type="text/html")


# Static files (CSS, JS, images)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files directory not found: {str(e)}")


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    config = get_config()
    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        reload=config.DEBUG
    )
