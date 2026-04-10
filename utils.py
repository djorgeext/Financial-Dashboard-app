import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm, linregress
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import h5py
from tqdm.auto import tqdm
import joblib
import os
import random

# ==========================================
# MODELOS DE REDES NEURONALES
# ==========================================

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.hidden_dim_half = hidden_dim // 2

        # Definimos las capas por separado para manejar la transposición en forward
        self.query_layer = nn.Linear(hidden_dim, self.hidden_dim_half)
        self.bn = nn.BatchNorm1d(self.hidden_dim_half)
        self.tanh = nn.Tanh()
        self.scoring_layer = nn.Linear(self.hidden_dim_half, 1)

    def forward(self, x):
        # x shape: [Batch, Seq, Hidden_dim] -> [B, 10, 64]

        # 1. Proyección lineal
        scores = self.query_layer(x) # [B, 10, 32]

        # 2. BatchNorm3D: Permutar a [B, 32, 10] para que BN actúe sobre los canales
        scores = scores.transpose(1, 2)
        scores = self.bn(scores)
        scores = scores.transpose(1, 2) # Volver a [B, 10, 32]

        # 3. Cálculo de Pesos
        scores = self.scoring_layer(self.tanh(scores)) # [B, 10, 1]
        weights = F.softmax(scores, dim=1)

        # 4. Vector de Contexto
        context_vector = torch.sum(x * weights, dim=1) # [B, 64]
        return context_vector, weights

class LSTMMixedModel(nn.Module):
    def __init__(self, num_features, lstm_hidden=128, num_classes=3, dropout=0.3):
        super(LSTMMixedModel, self).__init__()

        self.num_indep = 1
        self.num_process = num_features - self.num_indep

        # --- 1. FRONT-END CONVOLUCIONAL ---
        # El flujo aquí ya es [B, Channels, Seq], por lo que BN1d entra directo
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=self.num_process, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Seq 40 -> 20

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Seq 20 -> 10
        )

        # --- 2. BACKBONE TEMPORAL ---
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.attention = AttentionBlock(lstm_hidden)

        # --- 3. CLASIFICADOR (Input Total: 64 + 17 = 81) ---
        input_total = lstm_hidden + num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_total), # BN en 2D: [B, 81]
            nn.Linear(input_total, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.BatchNorm1d(lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, num_classes)
        )

    def forward(self, x):
        # x shape: [B, Seq, Features]
        # x_to_process: [B, Seq, Features-1]
        x_to_process = x[:, :, :self.num_process] 
        ind_data = x[:, -1, self.num_process:]    
        x_last_main = x[:, -1, :self.num_process] 

        # 1. PASO CONVOLUCIONAL
        x_cnn = x_to_process.permute(0, 2, 1) # [B, Features-1, Seq]
        x_cnn = self.cnn_block(x_cnn)         # [B, 32, Seq_pool]

        # 2. PASO RECURRENTE
        x_lstm_in = x_cnn.permute(0, 2, 1)    # [B, Seq_pool, 32]
        lstm_out, _ = self.lstm(x_lstm_in)    # [B, Seq_pool, Hidden]

        # Atención
        context_vector, weights = self.attention(lstm_out) 

        # 3. CONCATENACIÓN Y CLASIFICACIÓN
        combined = torch.cat((context_vector, x_last_main, ind_data), dim=1)
        logits = self.classifier(combined)

        return logits, weights


# ==========================================
# FUNCIONES DE PREPROCESAMIENTO E INDICADORES
# ==========================================

def calculate_rolling_hurst_numpy(series, window_size=100, min_win=20):
    """
    Versión optimizada con alineación de índice garantizada.
    """
    data = series.values
    
    # sliding_window_view genera n - window_size + 1 ventanas
    windows = sliding_window_view(data, window_shape=window_size)

    def get_h_wrapper(window):
        try:
            # kind='change' porque pasamos rendimientos logarítmicos normalmente
            H, _, _ = compute_Hc(window, kind='change', simplified=False, min_window=min_win)
            return H
        except:
            return np.nan

    hurst_results = np.apply_along_axis(get_h_wrapper, axis=1, arr=windows)

    # RESOLUCIÓN DE LA ALINEACIÓN:
    hurst_series = pd.Series(
        hurst_results,
        index=series.index[window_size - 1:]
    )

    return hurst_series.reindex(series.index)

def get_weights_ffd(d, thres, lim):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres: break
        w.append(w_)
        k += 1
        if k >= lim: break
    return np.array(w[::-1])

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Aplica diferenciación fraccionaria manteniendo la memoria.
    """
    series = series.ffill().bfill()
    w = get_weights_ffd(d, thres, len(series))
    width = len(w) - 1
    
    if width / len(series) > 0.3:
        while width / len(series) > 0.3:
            thres *= 5
            w = get_weights_ffd(d, thres, len(series))
            width = len(w) - 1
            
    if width >= len(series): return pd.Series(0, index=series.index)

    vals = np.log(series.values + 1e-9)  
    res = np.convolve(vals, w, mode='valid')
    return pd.Series(res, index=series.index[width:])

def find_optimal_d(series, step=0.05):
    """Busca el mínimo d que pasa el test ADF (p < 0.05)"""
    best_d = 1.0
    for d in np.arange(0.1, 1.1, step):
        try:
            diff_series = frac_diff_ffd(series, d)
            # Necesitamos suficientes datos para ADF
            if len(diff_series) < 50: continue
            p_val = adfuller(diff_series.dropna())[1]
            if p_val < 0.05:
                best_d = d
                break
        except:
            continue
    return best_d

def yang_zhang_volatility(df, window=20):
    log_oc = np.log(df['Open'] / df['Close'].shift(1))
    log_co = np.log(df['Close'] / df['Open'])
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])

    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_rs_sq = rs_var.rolling(window=window).mean()

    sigma_o_sq = log_oc.rolling(window=window).var()
    sigma_c_sq = log_co.rolling(window=window).var()

    k = 0.34 / (1 + (window + 1) / (window - 1))
    sigma_yz_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq
    
    anual_vol = np.sqrt(sigma_yz_sq * 252)
    last_20_days = np.sqrt(sigma_yz_sq)

    return anual_vol, last_20_days

def get_drift(df, window=40):
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    return log_ret.rolling(window=window).mean()

def calculate_amihud_illiquidity(returns, close, volume, window=10):
    dollar_volume = close * volume
    illiquidity = np.where(dollar_volume > 0, np.abs(returns) / dollar_volume, np.nan)
    return pd.Series(illiquidity, index=returns.index).rolling(window, min_periods=window).mean()

def calculate_vp_divergence_robust(volume, high, low, window=40):
    log_v = np.log(volume + 1e-9)
    log_r = np.log(high / low + 1e-9)
    return log_v.rolling(window, min_periods=window).corr(log_r)

def apply_robust_normalization(X_test, fitted_scalers):
    """
    Aplica los RobustScalers aprendidos a un nuevo conjunto de datos.
    X_test shape: (N, T, F)
    fitted_scalers: dict of fitted scaler objects
    """
    N, T, F = X_test.shape
    X_test_norm = np.zeros_like(X_test)

    for i in range(F):
        # scaler key usually 'feature_0', 'feature_1', etc.
        scaler = fitted_scalers[f'feature_{i}']
        feature_data = X_test[:, :, i].reshape(-1, 1)
        normalized_data = scaler.transform(feature_data)
        X_test_norm[:, :, i] = normalized_data.reshape(N, T)

    return X_test_norm


# ==========================================
# FUNCIONES DE ANÁLISIS DE SENTIMIENTO Y PROBABILIDAD (BAYESIANAS)
# ==========================================

def get_sentiment_logits(news_list, ticker, ticker_keywords, relevance_model, tokenizer, model_deberta, device):
    """
    Obtiene los logits de sentimiento.
    """
    if not news_list:
        return None

    keywords = ticker_keywords.get(ticker, [ticker])
    relevant_texts = []

    print(f"--- FIltrando noticias para {ticker} ---")

    for item in news_list:
        # Check structure of item, assuming item['content']['summary'] exists based on previous code
        try:
            text = item.get('content', {}).get('summary', '') or item.get('title', '')
        except:
            continue
            
        if not text: continue

        pairs = [[f"News regarding {kw}", text] for kw in keywords]
        scores = relevance_model.predict(pairs)
        max_score = max(scores)

        if max_score > -5: # Threshold from original code
            relevant_texts.append(text)
            print(text)

    if not relevant_texts:
        print(f"⚠️ Ninguna noticia pasó el filtro de relevancia para {ticker}.")
        return None

    inputs = tokenizer(
        relevant_texts,
        padding=True,
        truncation=True,
        max_length=1000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model_deberta(**inputs)
        logits = outputs.logits

    return logits

def calculate_bayesian_final_probability(pred_type, p_prior, probs_news, p_market, w_market=0.5):
    EPSILON = 1e-9
    prior_odds = np.log(p_prior / max(1 - p_prior, EPSILON))

    if len(probs_news) > 0:
        log_lrs = probs_news[:, 2] - probs_news[:, 0]
        avg_sentiment_score = np.mean(log_lrs)
    else:
        avg_sentiment_score = 0.0

    evidence_sentiment = avg_sentiment_score if pred_type == 1 else -avg_sentiment_score

    p_market = np.clip(p_market, EPSILON, 1 - EPSILON)
    market_log_odds = np.log(p_market / (1 - p_market))
    evidence_market = market_log_odds * w_market

    final_log_odds = prior_odds + evidence_sentiment + evidence_market
    p_final = 1 / (1 + np.exp(-final_log_odds))

    return p_final, avg_sentiment_score

def calcular_probabilidad_teorica(S, K, T_dias, r, sigma, tipo='call'):
    if T_dias <= 0 or sigma <= 0: return 0.0
    T = T_dias / 365.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo.lower() == 'call':
        return norm.cdf(d2)
    else:
        return norm.cdf(-d2)

def get_barrier_probabilities(y_ticker, barrier, option_type='call', r=0.044):
    try:
        hist = y_ticker.history(period="1d")
        if hist.empty: return pd.DataFrame()
        S_spot = hist['Close'].iloc[-1]
    except: return pd.DataFrame()

    target_strike = (S_spot + barrier) / 2
    hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fecha_min = hoy + timedelta(days=6)
    fecha_max = hoy + timedelta(days=15)

    try: expiraciones = y_ticker.options
    except: return pd.DataFrame()

    fechas_validas = [d for d in expiraciones if fecha_min <= datetime.strptime(d, '%Y-%m-%d') <= fecha_max]

    if not fechas_validas:
        return pd.DataFrame()

    lista_resultados = []

    for fecha in fechas_validas:
        try:
            dias_al_vencimiento = (datetime.strptime(fecha, '%Y-%m-%d') - hoy).days
            chain = y_ticker.option_chain(fecha)
            df_options = chain.calls if option_type.lower() == 'call' else chain.puts

            if df_options.empty: continue

            if option_type.lower() == 'call':
                df_filtrado = df_options[(df_options['strike'] > S_spot) & (df_options['strike'] <= barrier)].copy()
            else:
                df_filtrado = df_options[(df_options['strike'] >= barrier) & (df_options['strike'] < S_spot)].copy()

            if df_filtrado.empty: continue

            idx_mas_cercano = (df_filtrado['strike'] - target_strike).abs().idxmin()
            mejor_opcion = df_filtrado.loc[idx_mas_cercano]

            strike = mejor_opcion['strike']
            iv = mejor_opcion['impliedVolatility']
            ask = mejor_opcion['ask']

            prob_expire_itm = calcular_probabilidad_teorica(S_spot, strike, dias_al_vencimiento, r, iv, option_type)
            prob_touch_strike = min(prob_expire_itm * 2, 0.999)

            lista_resultados.append({
                "Vencimiento": fecha,
                "Días": dias_al_vencimiento,
                "Tipo": option_type.upper(),
                "Strike Seleccionado": strike,
                "Ask": ask,
                "IV": round(iv, 3),
                "Prob. Expira ITM": round(prob_expire_itm, 4),
                "Prob. Toca Strike": round(prob_touch_strike, 4)
            })

        except Exception:
            continue

    df_res = pd.DataFrame(lista_resultados)
    if not df_res.empty:
        df_res = df_res.sort_values(by='Vencimiento')

    return df_res

def engineer_features(df, hist_close):
    """
    Genera features técnicos, indicadores de volatilidad y análisis fractal.
    Retorna el DataFrame completo con nuevas columnas y el DataFrame filtrado para inferencia.
    """
    # 1. Indicadores Básicos
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/10, min_periods=10, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/10, min_periods=10, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    df['rsi_slope_40'] = df['RSI'].rolling(40).apply(
        lambda x: linregress(np.arange(len(x)), x).slope, raw=True
    )

    vol_anual, vol_20 = yang_zhang_volatility(df)
    df['vol_20'] = vol_20
    df['vol_anual'] = vol_anual
    df['drift'] = get_drift(df)
    df['kaufman_efficiency'] = abs(delta) / vol_20
    df['overnight_gap'] = np.log(df['Open'] / df['Close'].shift(1))
    
    period = 20
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    df['price_range_pos'] = (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    
    range_len = df['High'] - df['Low'] + 1e-9
    body_len = (df['Close'] - df['Open']).abs()
    upper_shadow = df['High'] - np.maximum(df['Close'], df['Open'])
    df['candle_body_ratio'] = body_len / range_len
    df['upper_shadow_ratio'] = upper_shadow / range_len
    
    # 2. Fractalidad y Métricas Complejas
    price_avg = (df['High'] + df['Low'] + df['Open'] + df['Close']) / 4
    log_ret = np.log(hist_close).diff()
    # df['hurst'] = calculate_rolling_hurst_numpy(log_ret.dropna(), window_size=100, min_win=20)
    
    df['skew'] = df['Close'].pct_change().rolling(40).skew()
    df['kurt'] = df['Close'].pct_change().rolling(40).kurt()
    df['rel_vol_z'] = (df['Volume'] - df['Volume'].rolling(252).mean()) / df['Volume'].rolling(252).std()
    df['rel_spread'] = (df['High'] - df['Low']) / df['Close']
    df['day_of_week'] = df.index.dayofweek
    
    df['ma_20_hours'] = price_avg.rolling(3).mean() 
    df['ma_40_hours'] = price_avg.rolling(6).mean() 
    df['ma_100_hours'] = price_avg.rolling(15).mean() 
    df['ma_200_hours'] = price_avg.rolling(30).mean() 
    df['Dist_SMA40'] = (df['Close'] - df['ma_40_hours']) / df['ma_40_hours']
    
    df['amihud'] = calculate_amihud_illiquidity(log_ret, df['Close'], df['Volume'])
    df['vp_div'] = calculate_vp_divergence_robust(df['Volume'], df['High'], df['Low'])

    # 3. Limpieza preliminar
    if len(df.dropna()) < 350:
        return df, None

    # 4. Diferenciación Fraccionaria (FFD)
    d_opt = find_optimal_d(price_avg, step=0.05)
    price_fr = frac_diff_ffd(price_avg, d=d_opt)
    df['ma_20_hours'] = frac_diff_ffd(df['ma_20_hours'], d=d_opt)
    df['ma_40_hours'] = frac_diff_ffd(df['ma_40_hours'], d=d_opt)
    df['ma_100_hours'] = frac_diff_ffd(df['ma_100_hours'], d=d_opt)
    df['ma_200_hours'] = frac_diff_ffd(df['ma_200_hours'], d=d_opt)

    # 5. Dataset de Inferencia
    full_df = pd.concat([
        df['vol_anual'], df['rel_spread'], df['kurt'],
        df['RSI'].rename('rsi'), df['rsi_slope_40'].rename('rsi_slope'), 
        df['Dist_SMA40'].rename('dist_sma40'), df['skew'],
        df['amihud'], df['rel_vol_z'], np.log(df['Volume'] + 1e-9), 
        df['vp_div'].rename('v_p_divergence'),
        df['kaufman_efficiency'], df['overnight_gap'], df['price_range_pos'], 
        df['candle_body_ratio'], df['upper_shadow_ratio'],
        # df['hurst'], 
        price_fr.rename('price_ffd'), 
        df['ma_20_hours'], df['ma_40_hours'], df['ma_100_hours'], df['ma_200_hours'],
        df['day_of_week']
    ], axis=1)

    full_df = full_df.dropna()
    return df, full_df


# ==========================================
# CONFIGURACIÓN DE TICKERS
# ==========================================

tech_us_tickers = {
    "Apple Inc.": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Alphabet Inc. (Class A)": "GOOGL",
    "Alphabet Inc. (Class C)": "GOOG", "Amazon": "AMZN", "Meta Platforms": "META", "Tesla, Inc.": "TSLA",
    "Broadcom Inc.": "AVGO", "Oracle Corporation": "ORCL", "Salesforce": "CRM", "Adobe Inc.": "ADBE",
    "ServiceNow": "NOW", "Intuit Inc.": "INTU", "IBM": "IBM", "Palo Alto Networks": "PANW",
    "Snowflake Inc.": "SNOW", "Workday, Inc.": "WDAY", "Autodesk": "ADSK", "Synopsys": "SNPS",
    "Cadence Design Systems": "CDNS", "Palantir Technologies": "PLTR", "Cloudflare": "NET", "Fortinet": "FTNT",
    "CrowdStrike": "CRWD", "Atlassian": "TEAM", "HubSpot": "HUBS", "Advanced Micro Devices (AMD)": "AMD",
    "Intel Corporation": "INTC", "Texas Instruments": "TXN", "Qualcomm": "QCOM", "Applied Materials": "AMAT",
    "Micron Technology": "MU", "Analog Devices": "ADI", "Lam Research": "LRCX", "KLA Corporation": "KLAC",
    "Microchip Technology": "MCHP", "NXP Semiconductors": "NXPI", "ON Semiconductor": "ON",
    "Monolithic Power Systems": "MPWR", "Teradyne": "TER", "Skyworks Solutions": "SWKS", "Qorvo": "QRVO",
    "Cisco Systems": "CSCO", "Arista Networks": "ANET", "Dell Technologies": "DELL", "Hewlett Packard Enterprise": "HPE",
    "HP Inc.": "HPQ", "NetApp": "NTAP", "Seagate Technology": "STX", "Western Digital": "WDC",
    "Super Micro Computer": "SMCI", "F5, Inc.": "FFIV", "Accenture": "ACN", "Cognizant": "CTSH",
    "Infosys (ADR)": "INFY", "Gartner": "IT", "EPAM Systems": "EPAM", "DXC Technology": "DXC",
    "Visa Inc.": "V", "Mastercard": "MA", "PayPal": "PYPL", "Fidelity National Information Services": "FIS",
    "Global Payments": "GPN", "Amphenol Corporation": "APH", "TE Connectivity": "TEL", "Corning Inc.": "GLW",
    "Keysight Technologies": "KEYS", "Teledyne Technologies": "TDY", "Zebra Technologies": "ZBRA",
    "Trimble Inc.": "TRMB", "Netflix": "NFLX", "Spotify": "SPOT", "Electronic Arts": "EA",
    "Take-Two Interactive": "TTWO", "Roblox Corporation": "RBLX", "Airbnb": "ABNB", "Uber Technologies": "UBER",
    "Booking Holdings": "BKNG", "Equinix": "EQIX", "Digital Realty": "DLR", "VRSN (Verisign)": "VRSN",
    "Akamai Technologies": "AKAM", "Iron Mountain": "IRM", "First Solar": "FSLR", "Enphase Energy": "ENPH",
    "DoorDash": "DASH", "Unity Software": "U", "Pinterest": "PINS", "Snap Inc.": "SNAP",
    "CoStar Group": "CSGP", "MongoDB": "MDB", "Technology Select Sector SPDR": "XLK",
    "Invesco QQQ Trust (Nasdaq 100)": "QQQ", "Vanguard Information Technology": "VGT",
    "iShares Expanded Tech-Software": "IGV", "VanEck Semiconductor ETF": "SMH", "iShares Semiconductor ETF": "SOXX",
    "Communication Services Select Sector": "XLC", "First Trust NASDAQ Cybersecurity": "CIBR",
    "Global X Cybersecurity ETF": "BUG", "First Trust Cloud Computing": "SKYY", "Global X Cloud Computing": "CLOU",
    "ARK Fintech Innovation": "ARKF", "Global X Fintech ETF": "FINX", "First Trust Dow Jones Internet": "FDN",
    "Invesco NASDAQ Internet": "PNQI", "Global X Robotics & Artificial Intelligence": "BOTZ",
    "Global X Artificial Intelligence & Tech": "AIQ", "Invesco Solar ETF": "TAN", "iShares Global Clean Energy": "ICLN"
}

petroleo_us_tickers = {
    "ExxonMobil": "XOM", "Chevron Corporation": "CVX", "Shell PLC (ADR)": "SHEL", "TotalEnergies SE (ADR)": "TTE",
    "BP PLC (ADR)": "BP", "ConocoPhillips": "COP", "Eni S.p.A. (ADR)": "E", "Occidental Petroleum": "OXY",
    "EOG Resources": "EOG", "Devon Energy": "DVN", "Diamondback Energy": "FANG", "Coterra Energy": "CTRA",
    "Apache Corporation (APA)": "APA", "Schlumberger (SLB)": "SLB", "Halliburton": "HAL", "Baker Hughes": "BKR",
    "Weatherford International": "WFRD", "TechnipFMC": "FTI", "Marathon Petroleum": "MPC", "Valero Energy": "VLO",
    "Phillips 66": "PSX", "PBF Energy": "PBF", "HF Sinclair": "DINO", "Enterprise Products Partners": "EPD",
    "Kinder Morgan": "KMI", "Williams Companies": "WMB", "ONEOK, Inc.": "OKE", "Targa Resources": "TRGP",
    "Cheniere Energy (Gas Natural Licuado)": "LNG", "Enbridge Inc.": "ENB", "Energy Select Sector SPDR": "XLE",
    "Vanguard Energy ETF": "VDE", "SPDR S&P Oil & Gas Exploration": "XOP", "VanEck Oil Services ETF": "OIH",
    "United States Oil Fund": "USO", "United States Natural Gas": "UNG"
}

banks_us_tickers = {
    "JPMorgan Chase": "JPM",
    "Bank of America": "BAC",
    "Citigroup": "C",
    "Wells Fargo": "WFC",
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "U.S. Bancorp": "USB",
    "PNC Financial Services": "PNC",
    "Truist Financial": "TFC",
    "TD Bank, N.A.": "TD",
    "Capital One": "COF",
    "Charles Schwab Corporation": "SCHW",
    "The Bank of New York Mellon": "BK",
    "State Street Corporation": "STT",
    "BMO USA": "BMO",
    "American Express": "AXP",
    "HSBC Bank USA": "HSBC",
    "First Citizens BancShares": "FCNCA",
    "Citizens Financial Group": "CFG",
    "Fifth Third Bank": "FITB",
    "UBS": "UBS",
    "M&T Bank": "MTB",
    "Huntington Bancshares": "HBAN",
    "Barclays": "BCS",
    "Ally Financial": "ALLY",
    "KeyCorp": "KEY",
    "RBC Bank": "RY",
    "Ameriprise": "AMP",
    "Santander Bank": "SAN",
    "Northern Trust": "NTRS",
    "Regions Financial Corporation": "RF",
    "Discover Financial": "DFS",
    "Synchrony Financial": "SYF",
    "Deutsche Bank": "DB",
    "Flagstar Financial": "FLG",
    "Raymond James Financial": "RJF",
    "Western Alliance Bancorporation": "WAL",
    "Mizuho Americas": "MFG",
    "First Horizon National Corporation": "FHN",
    "Webster Bank": "WBS",
    "Comerica": "CMA",
    "East West Bank": "EWBC",
    "CIBC Bank USA": "CM",
    "Popular, Inc.": "BPOP",
    "UMB Financial Corporation": "UMBF",
    "BNP Paribas": "BNPQY",
    "Wintrust Financial": "WTFC",
    "South State Bank": "SSB",
    "Valley Bank": "VLY",
    "Synovus": "SNV",
    "John Deere Bank": "DE",
    "Pinnacle Financial Partners": "PNFP",
    "Old National Bank": "ONB",
    "Cullen/Frost Bankers, Inc.": "CFR",
    "Columbia Bank": "COLB",
    "BOK Financial Corporation": "BOKF",
    "FNB Corporation": "FNB",
    "Associated Banc-Corp": "ASB",
    "Stifel": "SF",
    "Prosperity Bancshares": "PB",
    "SoFi": "SOFI",
    "BankUnited": "BKU",
    "Hancock Whitney": "HWC",
    "Banc of California": "BANC",
    "United Bank (West Virginia)": "UBSI",
    "Commerce Bancshares": "CBSH",
    "First National of Nebraska": "FINN",
    "Fulton Financial Corporation": "FULT",
    "Texas Capital Bank": "TCBI",
    "First Interstate BancSystem": "FIBK",
    "United Community Bank": "UCB",
    "Glacier Bancorp": "GBCI",
    "WaFd Bank": "WAFD",
    "WesBanco": "WSBC",
    "Simmons Bank": "SFNC",
    "Ameris Bancorp": "ABCB",
    "Eastern Bank": "EBC",
    "Atlantic Union Bank": "AUB",
    "Provident Bank of New Jersey": "PFS",
    "Axos Financial": "AX",
    "Bank of Hawaii": "BOH",
    "First Hawaiian Bank": "FHB",
    "Cathay Bank": "CATY",
    "Home BancShares": "HOMB",
    "Customers Bancorp": "CUBI",
    "WSFS Bank": "WSFS",
    "Independent Bank Corp": "INDB",
    "Busey Bank": "BUSE",
    "First BanCorp": "FBP",
    "ServisFirst": "SFBS",

    "Financial Select Sector SPDR Fund": "XLF",
    "Vanguard Financials ETF": "VFH",

    "SPDR S&P Bank ETF": "KBE",
    "Invesco KBW Bank ETF": "KBWB",
    "SPDR S&P Regional Banking ETF": "KRE",
    "iShares U.S. Regional Banks ETF": "IAT",
    "iShares U.S. Broker-Dealers & Securities Exchanges ETF": "IAI",
    "Direxion Daily Small Cap Bull 3x Shares": "TNA"
}

mining_us_tickers = {
    "Newmont Corporation": "NEM", "Barrick Gold": "GOLD", "Agnico Eagle Mines": "AEM", "AngloGold Ashanti": "AU",
    "Gold Fields Limited": "GFI", "Kinross Gold": "KGC", "Harmony Gold Mining": "HMY", "Sibanye Stillwater": "SBSW",
    "Alamos Gold": "AGI", "B2Gold Corp.": "BTG", "Eldorado Gold": "EGO", "Iamgold Corp.": "IAG",
    "DRDGOLD Limited": "DRD", "Equinox Gold": "EQX", "Pan American Silver": "PAAS", "First Majestic Silver": "AG",
    "Hecla Mining": "HL", "Fortuna Silver Mines": "FSM", "Endeavour Silver": "EXK", "Rio Tinto": "RIO",
    "BHP Group": "BHP", "Vale S.A.": "VALE", "Freeport-McMoRan": "FCX", "Southern Copper": "SCCO",
    "Teck Resources": "TECK", "Ero Copper": "ERO", "Alcoa Corporation": "AA", "ArcelorMittal": "MT",
    "Gerdau S.A.": "GGB", "Companhia Siderúrgica Nacional": "SID", "Albemarle Corporation": "ALB",
    "SQM (Soc. Química y Minera de Chile)": "SQM", "Lithium Americas": "LAC", "Sigma Lithium": "SGML",
    "MP Materials": "MP", "Franco-Nevada": "FNV", "Wheaton Precious Metals": "WPM", "Royal Gold": "RGLD",
    "Osisko Gold Royalties": "OR", "VanEck Gold Miners ETF": "GDX", "VanEck Junior Gold Miners ETF": "GDXJ",
    "Global X Silver Miners ETF": "SIL", "Global X Copper Miners ETF": "COPX", "Global X Lithium & Battery Tech": "LIT",
    "SPDR S&P Metals & Mining ETF": "XME", "VanEck Rare Earth/Strategic Metals": "REMX", "iShares Silver Trust": "SLV",
    "SPDR Gold Shares": "GLD", "Cameco Corporation": "CCJ", "NexGen Energy": "NXE", "Uranium Energy Corp": "UEC",
    "Energy Fuels": "UUUU", "Denison Mines": "DNN", "Centrus Energy": "LEU", "Nucor Corporation": "NUE",
    "Cleveland-Cliffs": "CLF", "Steel Dynamics": "STLD", "POSCO Holdings": "PKX", "Ternium S.A.": "TX",
    "Coeur Mining": "CDE", "SSR Mining": "SSRM", "NovaGold Resources": "NG", "Centerra Gold": "CGAU",
    "Seabridge Gold": "SA", "Hudbay Minerals": "HBM", "Taseko Mines": "TGB", "Ivanhoe Electric": "IE",
    "Triple Flag Precious Metals": "TFPM", "Global X Uranium ETF": "URA", "Sprott Uranium Miners ETF": "URNM",
    "VanEck Steel ETF": "SLX", "Sprott Physical Uranium Trust": "SRUUF"
}

# ==========================================
# FUNCIONES AUXILIARES DE ENTRENAMIENTO Y DATOS
# ==========================================

def get_triple_barrier_labels(df, t_final=10, k=1):
    """
    Implementa la lógica "First Touch" con un margen de seguridad del 1%.
    0: Neutral (Time Expiration o movimiento < 1%)
    1: Call (Toca barrera superior > 1% primero)
    2: Put (Toca barrera inferior < -1% primero)
    """
    n_samples = len(df)
    limit = n_samples - t_final

    if limit <= 0:
        return pd.Series(0, index=df.index[:0])

    p_t = df['Close'].values[:limit]
    vol_t = df['vol_20'].values[:limit]
    drift = df['drift'].values[:limit]

    step_sqrt = np.sqrt(t_final)
    deviation = (drift - 0.5 * vol_t**2) * t_final

    upper_barrier = p_t * np.exp(deviation + k * vol_t * step_sqrt)
    lower_barrier = p_t * np.exp(deviation - k * vol_t * step_sqrt)

    high_vals = df['High'].values
    low_vals = df['Low'].values
    windows_high = sliding_window_view(high_vals[1:], window_shape=t_final)
    windows_low = sliding_window_view(low_vals[1:], window_shape=t_final)

    touch_upper = windows_high >= upper_barrier[:, None]
    touch_lower = windows_low <= lower_barrier[:, None]

    idx_u = np.argmax(touch_upper, axis=1)
    idx_l = np.argmax(touch_lower, axis=1)
    has_u = np.any(touch_upper, axis=1)
    has_l = np.any(touch_lower, axis=1)

    first_u = np.where(has_u, idx_u, t_final + 1)
    first_l = np.where(has_l, idx_l, t_final + 1)

    labels_vals = np.zeros(limit, dtype=int)
    min_dist_call = p_t * 1.01
    min_dist_put = p_t * 0.99

    mask_call = (first_u < first_l) & (first_u < t_final) & (upper_barrier > min_dist_call)
    mask_put = (first_l < first_u) & (first_l < t_final) & (lower_barrier < min_dist_put)

    labels_vals[mask_call] = 1
    labels_vals[mask_put] = 2

    invalid_mask = (np.isnan(vol_t)) | (vol_t == 0)
    labels_vals[invalid_mask] = 0

    return pd.Series(labels_vals, index=df.index[:limit])

def create_sequences(features, targets, seq_len):
    common_idx = features.index.intersection(targets.index)
    f_data = features.loc[common_idx].values
    t_data = targets.loc[common_idx].values
    N = len(f_data)

    if N <= seq_len:
        f_dim = f_data.shape[1] if f_data.ndim > 1 else 1
        return np.empty((0, seq_len, f_dim)), np.array([])

    X_view = sliding_window_view(f_data, window_shape=seq_len, axis=0)
    X_view = X_view.transpose(0, 2, 1)
    X = X_view[:-1]
    y = t_data[seq_len - 1 : -1]
    return X, y

def fit_robust_normalization(X_train):
    N, T, F = X_train.shape
    X_train_norm = np.zeros_like(X_train)
    fitted_scalers = {}

    for i in range(F):
        feature_data = X_train[:, :, i].reshape(-1, 1)
        scaler = RobustScaler(with_centering=True, with_scaling=True)
        normalized_data = scaler.fit_transform(feature_data)
        X_train_norm[:, :, i] = normalized_data.reshape(N, T)
        fitted_scalers[f'feature_{i}'] = scaler

    return X_train_norm, fitted_scalers

def save_array_in_chunks(hf, name, array, batch_size=10000):
    shape = array.shape
    dtype = array.dtype
    dset = hf.create_dataset(name, shape=shape, dtype=dtype, compression="gzip", compression_opts=4, chunks=True)
    print(f"  -> Guardando '{name}' ({shape}) in chunks...")
    for i in tqdm(range(0, shape[0], batch_size), desc=f"Saving {name}"):
        end_i = min(i + batch_size, shape[0])
        dset[i:end_i] = array[i:end_i]

def save_processed_data_optimized(X_train, y_train, X_test, y_test, file_path):
    print(f"💾 Iniciando guardado seguro en {file_path}...")
    with h5py.File(file_path, 'w') as hf:
        save_array_in_chunks(hf, 'X_train', X_train)
        save_array_in_chunks(hf, 'y_train', y_train)
        save_array_in_chunks(hf, 'X_test', X_test)
        save_array_in_chunks(hf, 'y_test', y_test)
    print("✅ ¡Guardado exitoso sin colapsar la RAM!")

def load_processed_data(file_path):
    print(f"Cargando datos desde {file_path}...")
    with h5py.File(file_path, 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
    return X_train, y_train, X_test, y_test

def calculate_phi_metric(preds, targets, lambda_ret=0.02, delta=20):
    preds = np.array(preds)
    targets = np.array(targets)
    dcc = ((preds != 0) & (preds == targets)).sum()
    dic = ((preds == 1) & (targets == 2)).sum() + ((preds == 2) & (targets == 1)).sum()
    tec = ((preds != 0) & (targets == 0)).sum()
    log_phi = (dcc * np.log(1 + lambda_ret) +
               dic * np.log(1 - lambda_ret) +
               tec * np.log(1 - (lambda_ret / delta)))
    return log_phi, dcc, dic, tec

# ==========================================
# META MODELO
# ==========================================

import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_meta_dataset(model, test_loader, device):
    model.eval()
    meta_features = []
    meta_labels = []

    print("Generando datos para el Meta-modelo...")

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            for i in range(len(y_batch)):
                p_pred = preds[i].item()
                p_true = y_batch[i].item()

                if p_pred != 0:
                    x_raw_feat = X_batch[i, -10:, :].flatten()
                    combined_feats = torch.cat((probs[i], x_raw_feat)).cpu().numpy()
                    m_label = 1 if p_pred == p_true else 0 # 1 si acierto, 0 si fallo
                    meta_features.append(combined_feats)
                    meta_labels.append(m_label)

    return np.array(meta_features), np.array(meta_labels)

def train_xgboost_meta_model(X_train, y_train, X_val, y_val):
    print("Iniciando entrenamiento con estrategia anti-overfitting...")
    meta_model = xgb.XGBClassifier(
        n_estimators=200,
        num_parallel_tree=50,
        learning_rate=0.1,
        max_depth=6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.2,
        subsample=0.7,
        colsample_bytree=0.4,
        objective='binary:logistic',
        random_state=111,
        eval_metric=['logloss', 'auc'],
        early_stopping_rounds=20
    )

    meta_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    print(f"✅ Entrenamiento detenido en la iteración: {meta_model.best_iteration}")
    return meta_model

def evaluate_meta_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\\n" + "="*50)
    print("REPORTE DEL META-MODELO (Filtro de Confianza)")
    print("="*50)
    print(classification_report(y_test, preds, target_names=['Fallo (0)', 'Acierto (1)']))

    auc = roc_auc_score(y_test, probs)
    f1_macro = f1_score(y_test, preds, average='macro')

    print(f"ROC-AUC:  {auc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")

    return probs

def plot_learning_curves(model):
    results = model.evals_result()
    plt.figure(figsize=(10, 5))
    plt.plot(results['validation_0']['logloss'], label='Train Loss')
    plt.plot(results['validation_1']['logloss'], label='Val Loss')
    plt.axvline(model.best_iteration, color='r', linestyle='--', label='Best Iteration')
    plt.title('Curvas de Aprendizaje (LogLoss)')
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()
