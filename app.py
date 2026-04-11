"""
Financial Dashboard Backend API
Main FastAPI application with endpoints for price data, forecasts, and sentiment.
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import Config, get_config
from model_service import create_model_service, ModelService
from data_fetcher import create_data_fetcher, DataFetcher
from news_service import create_news_service, NewsService

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
    
    global config, model_service, data_fetcher, news_service
    
    config = get_config()
    logger.info(f"Environment: {config.ENV}")
    logger.info(f"Debug Mode: {config.DEBUG}")
    logger.info(f"Model Directory: {config.MODEL_DIR}")
    
    # Initialize services
    model_service = create_model_service(model_dir=config.MODEL_DIR)
    data_fetcher = create_data_fetcher(cache_ttl=config.DATA_CACHE_TTL)
    
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
        # Mock news items for demo (in production, would fetch from news API)
        news_items = [
            f"{sector_lower.capitalize()} sector shows strong growth",
            f"New regulations impact {sector_lower} companies",
            f"Investment surge in {sector_lower} industry",
            f"Analyst upgrade for {sector_lower.capitalize()} stocks",
            f"Earnings beat from major {sector_lower.capitalize()} player"
        ]
        
        # Analyze with Groq
        analysis = news_service.analyze_news(sector_lower, news_items)
        
        return {
            "sector": sector_lower,
            "timestamp": datetime.now().isoformat(),
            "news_count": len(news_items),
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
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }


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
