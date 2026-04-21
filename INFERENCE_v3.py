# %%
!pip install git+https://github.com/ranaroussi/yfinance.git

# %%
# --- SETUP & IMPORTS ---
from google.colab import drive
import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn.functional as F
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import warnings
from datetime import datetime

# 1. Montar Drive
drive.mount('/content/drive', force_remount=True)

# 2. Configurar Rutas
BASE_PATH = '/content/drive/MyDrive/Challenges_ML-DL'
MODELS_PATH = os.path.join(BASE_PATH, 'models')

# Agregar al path para importar utils
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

# Importar funciones personalizadas
try:
    from backend.utils_2 import (
        LSTMMixedModel,
        find_optimal_d,
        frac_diff_ffd,
        yang_zhang_volatility,
        get_drift,
        calculate_amihud_illiquidity,
        calculate_vp_divergence_robust,
        apply_robust_normalization,
        get_sentiment_logits,
        calculate_bayesian_final_probability,
        get_barrier_probabilities,
        engineer_features
    )
    print("✅ backend.utils_2 importado correctamente.")
except ImportError as backend_import_error:
    try:
        from utils_2 import (
            LSTMMixedModel,
            find_optimal_d,
            frac_diff_ffd,
            yang_zhang_volatility,
            get_drift,
            calculate_amihud_illiquidity,
            calculate_vp_divergence_robust,
            apply_robust_normalization,
            get_sentiment_logits,
            calculate_bayesian_final_probability,
            get_barrier_probabilities,
            engineer_features
        )
        print("✅ utils_2 importado correctamente (fallback legacy).")
    except ImportError as legacy_import_error:
        import_error_message = (
            "❌ Error importando utilidades desde backend.utils_2 y fallback utils_2: "
            f"{backend_import_error} | {legacy_import_error}"
        )
        print(import_error_message)
        raise ImportError(import_error_message) from legacy_import_error

# Configuración Global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
warnings.filterwarnings('ignore')

# %%
# --- CONFIGURACIÓN DE SECTORES ---
# Aquí centralizamos toda la información específica de cada sector
SECTORS_CONFIG = {
    "TECH": {
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
        "keywords": {
            "QQQ": ["Invesco QQQ Trust", "Nasdaq 100", "NDX", "Technology Sector ETF", "Tech stocks", "Growth stocks"],
            "META": ["META", "Facebook", "Meta Platforms", "Instagram", "WhatsApp", "Mark Zuckerberg", "Metaverse", "Reality Labs", "Llama", "Reels"],
            "AAPL": ["AAPL", "Apple", "iPhone", "Mac", "Tim Cook", "App Store", "Services", "Apple Intelligence", "Vision Pro"],
            "AMZN": ["AMZN", "Amazon", "AWS", "Cloud Computing", "E-commerce", "Jeff Bezos", "Andy Jassy", "Prime", "Logistics"],
            "NFLX": ["NFLX", "Netflix", "Streaming service", "Subscriber growth", "Ted Sarandos", "Ad-supported", "Password sharing"],
            "TSLA": ["TSLA", "Tesla", "Elon Musk", "Electric Vehicles", "Cybertruck", "FSD", "Robotaxi", "Energy storage", "Optimus"],
            "NVDA": ["NVDA", "Nvidia", "Jensen Huang", "GPU", "Data Center", "AI chips", "Blackwell", "H100", "CUDA"],
            "PLTR": ["PLTR", "Palantir", "Alex Karp", "Big Data Analytics", "Gotham", "Foundry", "AIP", "Bootcamps", "Commercial revenue"],
            "MSFT": ["MSFT", "Microsoft", "Satya Nadella", "Azure", "Windows", "Copilot", "OpenAI", "Activision", "Office 365"],
            "GOOGL": ["GOOGL", "Alphabet", "Google", "Sundar Pichai", "Google Cloud", "Gemini", "YouTube", "Waymo", "Search", "DeepMind"],
            "INTC": ["INTC", "Intel", "Pat Gelsinger", "Xeon", "Intel Foundry", "Gaudi", "CHIPS Act", "18A"],
            "AMD": ["AMD", "Advanced Micro Devices", "Lisa Su", "Ryzen", "EPYC", "Radeon", "MI300", "Instinct", "Data Center"]
        }
    },
    "BANKS": {
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
        "keywords": {
            "BAC": ["BAC", "Bank of America", "BofA", "Brian Moynihan", "Merrill Lynch", "Merrill", "Federal Reserve"],
            "JPM": ["JPM", "JPMorgan", "JP Morgan", "Chase", "Jamie Dimon", "First Republic", "Investment Banking", "Federal Reserve"],
            "WFC": ["WFC", "Wells Fargo", "Charlie Scharf", "Commercial Banking", "Mortgage", "Asset Cap", "Federal Reserve"],
            "C": ["Citigroup", "Citi", "Jane Fraser", "Wealth Management", "Banamex", "Federal Reserve"],
            "XLF": ["XLF", "Financial Select Sector SPDR", "Financials ETF", "S&P 500 Financials", "Banking Sector", "Interest Rates", "Federal Reserve", "Berkshire Hathaway"],
            "TNA": ["TNA", "Direxion Daily Small Cap Bull", "Russell 2000", "Small Caps", "Leveraged ETF", "Risk-on", "IWM", "Regional Banks", "Federal Reserve"]
        }
    },
    # "OIL": {
    #     "model_files": {
    #         "lstm": "petroleo_model.pth",
    #         "scaler": "scalers_petroleo.pkl",
    #         "meta": "meta_model_xgb_petroleo.json",
    #         "lstm_hourly": "petroleo_model_hourly.pth",
    #         "scaler_hourly": "scalers_petroleo_hourly.pkl",
    #         "meta_hourly": "meta_model_xgb_petroleo_hourly.json"
    #     },
    #     "bayesian": {
    #         "p_call": 0.41, "p_put": 0.42,
    #         "sensitivity": 0.29, "specificity": 0.83
    #     },
    #     "bayesian_hourly": {
    #         "p_call": 0.4748, "p_put": 0.4375,
    #         "sensitivity": 0.54, "specificity": 0.75
    #     },
    #     "tickers": ["CVX", "XOM", "USO"],
    #     "keywords": {
    #         "CVX": ["Chevron", "Mike Wirth", "Permian Basin", "Tengiz", "Oil & Gas"],
    #         "XOM": ["ExxonMobil", "Exxon", "Darren Woods", "Upstream", "Energy"],
    #         "USO": ["United States Oil Fund", "WTI Crude", "Oil prices", "Petroleum"]
    #     }
    # },
    "MINING": {
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
        "keywords": {
            "GLD": ["GLD", "SPDR Gold Shares", "Gold", "Gold Futures", "Bullion", "Precious Metals", "Safe haven", "Inflation hedge", "Central Banks", "Geopolitical risk", "Real rates"],
            "SLV": ["SLV", "iShares Silver Trust", "Silver", "Silver Futures", "Industrial metals", "Solar energy", "Photovoltaic", "Electronics", "Gold/Silver ratio", "Precious Metals"],
            "NEM": ["NEM", "Newmont", "Gold mining", "Tom Palmer", "Newcrest", "Precious metals", "Gold miners"],
            "HL": ["HL", "Hecla", "Hecla Mining", "Silver mining", "Greens Creek", "Lucky Friday", "Primary silver"],
            "PAAS": ["PAAS", "Pan American Silver", "Silver mining", "Yamana Gold", "Latin America mining", "Precious metals"],
            "NUE": ["NUE", "Nucor", "Steel", "Steelmaker", "Electric Arc Furnace", "EAF", "Scrap metal", "Infrastructure"],
            "CLF": ["CLF", "Cleveland-Cliffs", "Lourenco Goncalves", "Steel", "Iron ore", "Blast furnace", "Automotive steel", "Infrastructure"]
        }
    }
}

# %%
# --- CARGA DE MODELOS COMPARTIDOS (NLP) ---
print("Cargando modelos de NLP...")
model_name = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_deberta = AutoModelForSequenceClassification.from_pretrained(model_name)
model_deberta.to(device)
model_deberta.eval()

relevance_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Modelos de NLP cargados.")

# %%
# --- INFERENCIA UNIFICADA ---
SEQ_LEN_2d = 20
SEQ_LEN_hourly = 40
TBM_HORIZON = 2
k = 1
all_inference_results = []

TBM_HORIZON_hourly = 10

for sector_name, config in SECTORS_CONFIG.items():
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO INFERENCIA SECTOR: {sector_name}")
    print(f"{'='*50}")

    # 1. Cargar Modelos Específicos del Sector
    try:
        # LSTM
        lstm_path = os.path.join(MODELS_PATH, config['model_files']['lstm'])
        model = LSTMMixedModel(num_features=23, lstm_hidden=64, dropout=0.3).to(device)
        model.load_state_dict(torch.load(lstm_path, map_location=device))
        model.eval()

        # LSTM_hourly
        lstm_hourly_path = os.path.join(MODELS_PATH, config['model_files']['lstm_hourly'])
        model_hourly = LSTMMixedModel(num_features=23, lstm_hidden=64, dropout=0.3).to(device)
        model_hourly.load_state_dict(torch.load(lstm_hourly_path, map_location=device))
        model_hourly.eval()

        # Meta-Model (XGB)
        meta_path = os.path.join(MODELS_PATH, config['model_files']['meta'])
        meta_model = xgb.XGBClassifier()
        meta_model.load_model(meta_path)

        # Meta-Model_hourly (XGB)
        meta_hourly_path = os.path.join(MODELS_PATH, config['model_files']['meta_hourly'])
        meta_model_hourly = xgb.XGBClassifier()
        meta_model_hourly.load_model(meta_hourly_path)

        # Scalers
        scaler_path = os.path.join(MODELS_PATH, config['model_files']['scaler'])
        scalers = joblib.load(scaler_path)

        # Scalers_hourly
        scaler_hourly_path = os.path.join(MODELS_PATH, config['model_files']['scaler_hourly'])
        scalers_hourly = joblib.load(scaler_hourly_path)


        print(f"✅ Modelos cargados para {sector_name}")

    except FileNotFoundError as e:
        print(f"❌ Error cargando archivos para {sector_name}: {e}")
        continue

    # 2. Configurar Probabilidades Bayesianas
    b_params = config['bayesian']
    b_params_hourly = config['bayesian_hourly']

    # 3. Iterar Tickers
    for ticker in config['tickers']:
            print(f"🔹 Procesando: {ticker} ...")
            y_ticker = yf.Ticker(ticker)
            hist = y_ticker.history(period="max")
            hist_hourly = y_ticker.history(period="730d", interval="1h")

            if hist.empty: continue

            hist.index = pd.to_datetime(hist.index, utc=True)
            df_raw = hist.copy()
            df_raw_hourly = hist_hourly.copy()

            # --- FEATURE ENGINEERING ---
            df, full_df = engineer_features(df_raw, hist['Close'])
            df_hourly, full_df_hourly = engineer_features(df_raw_hourly, hist_hourly['Close'])

            if full_df is None or len(full_df) < SEQ_LEN_2d + 20:
                continue

            # --- PREDICCIÓN ---
            X_inference = full_df.values[-SEQ_LEN_2d:].reshape(1, SEQ_LEN_2d, -1)
            X_inference_hourly = full_df_hourly.values[-SEQ_LEN_hourly:].reshape(1, SEQ_LEN_hourly, -1)

            if ticker not in scalers:
                print(f"⚠️ No hay scaler para {ticker}")
                continue

            X_inference = apply_robust_normalization(X_inference, scalers[ticker])
            X_inference_hourly = apply_robust_normalization(X_inference_hourly, scalers_hourly[ticker])
            X_inference_tensor = torch.tensor(X_inference, dtype=torch.float32).to(device)
            X_inference_tensor_hourly = torch.tensor(X_inference_hourly, dtype=torch.float32).to(device)

            logits, _ = model(X_inference_tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            pred_primary = pred[0].item()

            logits_hourly, _ = model_hourly(X_inference_tensor_hourly)
            probs_hourly = F.softmax(logits_hourly, dim=1)
            pred_hourly = torch.argmax(logits_hourly, dim=1)
            pred_primary_hourly = pred_hourly[0].item()

            if pred_primary == 0 and pred_primary_hourly == 0: continue

            # --- META-MODEL VALIDATION ---
            with torch.no_grad():
                x_raw_feat = X_inference_tensor[0, -10:, :].flatten()
                x_meta_model = torch.cat((probs[0], x_raw_feat)).cpu().numpy().reshape(1, -1)
                meta_pred = meta_model.predict(x_meta_model)

                x_raw_feat_hourly = X_inference_tensor_hourly[0, -10:, :].flatten()
                x_meta_model_hourly = torch.cat((probs_hourly[0], x_raw_feat_hourly)).cpu().numpy().reshape(1, -1)
                meta_pred_hourly = meta_model_hourly.predict(x_meta_model_hourly)

            # Barreras
            p_t = df['Close'].values[-1]
            vol_t = df['vol_20'].values[-1]
            drift_val = df['drift'].values[-1]
            step_sqrt = np.sqrt(TBM_HORIZON)
            deviation = (drift_val - 0.5 * vol_t**2) * TBM_HORIZON

            upper_barrier = p_t * np.exp(deviation + k * vol_t * step_sqrt)
            lower_barrier = p_t * np.exp(deviation - k * vol_t * step_sqrt)

            is_call = (pred_primary == 1) and (upper_barrier > (p_t * 1.01))
            is_put = (pred_primary == 2) and (lower_barrier < (p_t * 0.99))

            # Barreras_hourly
            p_t_hourly = df_hourly['Close'].values[-1]
            vol_t_hourly = df_hourly['vol_20'].values[-1]
            drift_val_hourly = df_hourly['drift'].values[-1]
            step_sqrt_hourly = np.sqrt(TBM_HORIZON_hourly)
            deviation_hourly = (drift_val_hourly - 0.5 * vol_t_hourly**2) * TBM_HORIZON_hourly

            upper_barrier_hourly = p_t_hourly * np.exp(deviation_hourly + k * vol_t_hourly * step_sqrt_hourly)
            lower_barrier_hourly = p_t_hourly * np.exp(deviation_hourly - k * vol_t_hourly * step_sqrt_hourly)

            is_call_hourly = (pred_primary_hourly == 1) and (upper_barrier_hourly > (p_t_hourly * 1.01))
            is_put_hourly = (pred_primary_hourly == 2) and (lower_barrier_hourly < (p_t_hourly * 0.99))

            if is_call or is_put or is_call_hourly or is_put_hourly:
                # Determine main op_type
                if is_call or is_call_hourly:
                    op_type = "call"
                    target_type = 1
                    barrier = upper_barrier if is_call else upper_barrier_hourly
                    barrier_hourly = upper_barrier_hourly
                else:
                    op_type = "put"
                    target_type = 2
                    barrier = lower_barrier if is_put else lower_barrier_hourly
                    barrier_hourly = lower_barrier_hourly

                mov_type = "alcista" if op_type == "call" else "bajista"
                print(f"⚡ OPORTUNIDAD: {ticker} ({op_type.upper()}) | Barrera: {barrier:.2f}")
                print(f"⚡ Mov. en hrs: {ticker} ({mov_type.upper()}) | Barrera: {barrier_hourly:.2f}")

                # Sentiment Analysis
                news = y_ticker.news
                logits_sent = get_sentiment_logits(news, ticker, config['keywords'], relevance_model, tokenizer, model_deberta, device)

                probs_news = np.array([])
                if logits_sent is not None:
                    probs_news = F.softmax(logits_sent, dim=1).detach().cpu().numpy()

                # Datos Mercado
                barrier_prob = get_barrier_probabilities(y_ticker, barrier, op_type)

                if not barrier_prob.empty:
                    precision_base = b_params['p_call'] if target_type == 1 else b_params['p_put']
                    recall_meta = b_params['sensitivity']
                    spec_meta = b_params['specificity']

                    precision_base_hourly = b_params_hourly['p_call'] if target_type == 1 else b_params_hourly['p_put']
                    recall_meta_hourly = b_params_hourly['sensitivity']
                    spec_meta_hourly = b_params_hourly['specificity']

                    for idx, row in barrier_prob.iterrows():
                        p_final, score = calculate_bayesian_final_probability(
                            target_type=target_type,
                            pred_primary=pred_primary,
                            meta_pred=meta_pred[0],
                            precision_base=precision_base,
                            recall_meta=recall_meta,
                            spec_meta=spec_meta,
                            pred_primary_hourly=pred_primary_hourly,
                            meta_pred_hourly=meta_pred_hourly[0],
                            precision_base_hourly=precision_base_hourly,
                            recall_meta_hourly=recall_meta_hourly,
                            spec_meta_hourly=spec_meta_hourly,
                            probs_news=probs_news,
                            p_market=row['Prob. Toca Strike'],
                            w_market=0.3
                        )

                        result_data = {
                            "Fecha": datetime.now().strftime("%Y-%m-%d"),
                            "Sector": sector_name,
                            "Ticker": ticker,
                            "Tipo": op_type.upper(),
                            "Precio Actual": round(p_t, 2),
                            "Barrera": round(barrier, 2),
                            "Barrera a 10 hrs": round(barrier_hourly, 2),
                            "Vencimiento": row['Vencimiento'],
                            "Strike": row['Strike Seleccionado'],
                            "Ask": row['Ask'],
                            "IV": row['IV'],
                            "Prob. Mercado": row['Prob. Toca Strike'],
                            "Prob. Final (Bayes)": round(p_final, 4),
                            "Sentimiento Score": round(score, 4)
                        }
                        all_inference_results.append(result_data)
                        pd.set_option('display.max_columns', None)
                        display(pd.DataFrame([result_data]))


# --- GUARDAR RESULTADOS ---
if all_inference_results:
    df_results = pd.DataFrame(all_inference_results)
    excel_path = 'Inference_Results.xlsx'
    df_results.to_excel(excel_path, index=False)
    print(f"\n✅ Resultados procesados y guardados en: {excel_path}")
    display(df_results)
else:
    print("\n⚠️ No se encontraron oportunidades que cumplan los criterios.")


