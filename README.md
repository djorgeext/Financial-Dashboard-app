# Financial Dashboard API & Web UI

A production-ready financial dashboard system that combines real-time market data, ML-powered forecasts, and AI-driven news sentiment analysis.

## Features

- **Real-time Financial Data**: Live price data for tech, banks, and mining sectors
- **ML Forecasting**: LSTM+CNN+Attention models for price trend prediction (hourly/daily)
- **News Sentiment Analysis**: Groq LLM integration for automatic trade signal extraction
- **Professional Dashboard UI**: Real-time visualization with responsive design
- **RESTful API**: Complete API for programmatic access
- **Caching & Performance**: Intelligent caching to minimize API calls
- **Error Resilience**: Graceful degradation when external services unavailable

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│           Financial Dashboard System                 │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Dashboard   │  │   FastAPI    │  │  Services  │ │
│  │     HTML     │──│  Backend     │──│  (Groq,    │ │
│  │     UI       │  │   (app.py)   │  │  yfinance) │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                           │                           │
│        ┌──────────────────┼──────────────────┐       │
│        │                  │                  │        │
│   ┌────▼────┐      ┌─────▼─────┐    ┌──────▼──┐    │
│   │  Model  │      │   Data    │    │  News   │    │
│   │ Service │      │ Fetcher   │    │ Service │    │
│   │         │      │           │    │         │    │
│   └────┬────┘      └─────┬─────┘    └──────┬──┘    │
│        │                 │                 │        │
│        │            ┌────▼─────┐           │        │
│        │            │ yfinance  │           │        │
│        │            └───────────┘      ┌────▼────┐  │
│        │                               │  Groq   │  │
│        │                               │  LLM    │  │
│        │                               └─────────┘  │
│        │                                             │
│   ┌────▼──────────────────────────────────────────┐ │
│   │         Pre-trained Models (PyTorch)          │ │
│   │  - tech_us_model.pth (daily/hourly)          │ │
│   │  - banks_model.pth (daily/hourly)            │ │
│   │  - mining_model.pth (daily/hourly)           │ │
│   └─────────────────────────────────────────────── │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.9+
- PyTorch (GPU or CPU)
- Pre-trained model files in `models/` directory

### Setup Steps

1. **Create virtual environment:**
   ```bash
   cd /home/david/Documents/trade
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API credentials (development):**
   - For development, use `groq_api_key.txt`:
     ```bash
     echo "your_groq_api_key_here" > groq_api_key.txt
     ```
   
   - For production, set environment variable:
     ```bash
     export GROQ_API_KEY="your_groq_api_key_here"
     export ENV="production"
     ```

4. **Verify models are present:**
   ```bash
   ls -la models/
   # Should show: banks_model.pth, mining_model.pth, tech_us_model.pth
   # And hourly variants: *_hourly.pth
   ```

## Configuration

Configure via environment variables (create `.env` file in project root):

```bash
# Environment
ENV=development          # or 'production'
DEBUG=False             # Set to True for debug mode

# Server
HOST=127.0.0.1
PORT=8000

# Paths
MODEL_DIR=/home/david/Documents/trade/models

# Cache Configuration
DATA_CACHE_TTL=300      # 5 minutes
NEWS_CACHE_TTL=600      # 10 minutes

# API Configuration
MAX_RETRIES=3
MAX_NEWS_PER_SECTOR=10

# Model Architecture (usually don't change)
SEQUENCE_LENGTH=40
LSTM_HIDDEN=128
NUM_CLASSES=3
```

## Running the Application

### Development Server

```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The dashboard will be available at: `http://127.0.0.1:8000/`

### Production Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health & Status

- **GET `/health`** - Health check
  ```json
  {
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00",
    "service": "financial-dashboard"
  }
  ```

- **GET `/ready`** - Readiness probe (validates pipeline)
- **GET `/api/pipeline-health`** - Detailed pipeline status

### Financial Data

- **GET `/api/financial/summary/{sector}`** - Current prices
  - Sectors: `tech`, `banks`, `mining`
  - Returns: current price, 24h change, high/low

### Forecasts

- **GET `/api/forecast/{sector}/{granularity}`** - ML predictions
  - Granularity: `daily` or `hourly`
  - Returns: trend (up/neutral/down), confidence, class probabilities

- **GET `/api/news/{sector}`** - News sentiment analysis
  - Returns: sentiment (bullish/neutral/bearish), score, signals

### Dashboard

- **GET `/api/dashboard`** - Aggregated view
  - Returns: Combined financial, forecast, and sentiment data for all sectors
  - Primary endpoint for UI

### Frontend

- **GET `/`** - Serve dashboard HTML
- **GET `/dashboard`** - Dashboard page

## Usage Examples

### Fetch Current Dashboard Data

```bash
curl http://127.0.0.1:8000/api/dashboard
```

### Get Forecast for Tech Sector

```bash
curl http://127.0.0.1:8000/api/forecast/tech/daily
```

### Get News Sentiment

```bash
curl http://127.0.0.1:8000/api/news/banks
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# Model service tests
pytest tests/test_model_service.py -v

# News service tests
pytest tests/test_news_service.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

## Key Components

### `config.py`
- Environment configuration
- API key management (secure Groq key handling)
- Sector definitions and model mappings

### `model_service.py`
- PyTorch model loading
- Batch inference
- Attention weight extraction
- Error handling for predictions

### `data_fetcher.py`
- yfinance integration
- Price data fetching & normalization
- Feature engineering for models
- In-memory caching with TTL

### `news_service.py`
- Groq LLM integration
- News sentiment classification
- Trade signal extraction
- Safe defaults on API failures

### `app.py`
- FastAPI backend
- All RESTful endpoints
- CORS configuration
- Error handling & logging

### `dashboard.html`
- Single-page application (SPA)
- Real-time data visualization
- Responsive design
- Auto-refresh every 5 minutes

## Security Considerations

### API Key Management
✅ **Development**: Reads from `groq_api_key.txt` (local only)
✅ **Production**: Requires `GROQ_API_KEY` environment variable
✅ **Never Exposed**: Keys not logged, printed, or included in responses

### Sensitive Data
- No API keys in error messages
- No credentials in logs
- `groq_api_key.txt` excluded from git (.gitignore)

### Best Practices
1. Always use environment variables in production
2. Rotate API keys regularly
3. Use HTTPS in production
4. Implement rate limiting for public deployment

## Troubleshooting

### "Model not found" Error
```
Solution: Ensure model files exist in models/ directory
ls -la models/ | grep .pth
```

### "Groq API Error"
```
Solution: Check API key configuration
- Development: echo $GROQ_API_KEY (use groq_api_key.txt)
- Production: export GROQ_API_KEY="your_key"
```

### "No data fetched" from yfinance
```
Solution: Check internet connection and ticker symbols
curl -X GET "http://127.0.0.1:8000/api/financial/summary/tech"
```

### Port Already in Use
```bash
# Change port
export PORT=8001
python app.py
```

## Performance Optimization

### Caching Strategy
- **Financial Data**: 5-minute cache (configurable)
- **News Analysis**: 10-minute cache
- **In-memory**: Fast retrieval, no database overhead

### Model Optimization
- Models run on GPU if available (automatic detection)
- Batch predictions supported for efficiency
- <500ms inference time per request

### Frontend
- CSS-in-HTML for minimal requests
- Auto-refresh every 5 minutes (configurable)
- Responsive design for mobile

## Known Limitations

1. **Demo Mode**: News items are mocked (would need news API in production)
2. **Single Ticker**: Forecasts show first ticker per sector (expandable)
3. **Real-time**: Updates every 5 minutes (can be increased)
4. **PyTorch Models**: Require significant memory (GPU recommended)

## Future Enhancements

- [ ] Real news API integration (NewsAPI, etc.)
- [ ] Database for historical predictions
- [ ] WebSocket for true real-time updates
- [ ] Multi-model ensemble
- [ ] Portfolio optimization
- [ ] Alert system for price targets
- [ ] Mobile app (React Native)
- [ ] Docker containerization

## File Structure

```
/home/david/Documents/trade/
├── app.py                           # FastAPI main application
├── config.py                        # Configuration & secrets
├── model_service.py                 # PyTorch inference wrapper
├── data_fetcher.py                  # Financial data pipeline
├── news_service.py                  # Groq LLM integration
├── dashboard.html                   # Frontend UI
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusions
├── tests/
│   ├── __init__.py
│   ├── test_model_service.py        # Model service unit tests
│   ├── test_news_service.py         # News service unit tests
│   └── test_integration.py          # End-to-end API tests
├── models/
│   ├── banks_model.pth
│   ├── banks_model_hourly.pth
│   ├── mining_model.pth
│   ├── mining_model_hourly.pth
│   ├── tech_us_model.pth
│   └── tech_us_model_hourly.pth
├── utils.py                         # LSTM+CNN model definition
├── utils_2.py                       # Additional utilities
└── README.md                        # This file
```

## Support & Documentation

- **API Docs (Swagger)**: http://127.0.0.1:8000/docs
- **API Docs (ReDoc)**: http://127.0.0.1:8000/redoc
- **Configuration**: See `config.py` for all settings
- **Model Architecture**: See `utils.py` for LSTMMixedModel

## License

Internal project - Use at your discretion.

## Contact

For issues or questions, refer to the inline code documentation and docstrings.
