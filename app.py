# =============================================================================
# Der-AI | Institutional Market Analysis — UPGRADED BUILD
#
# Changes vs. the previous version (see chat for full root-cause analysis):
#   1. call_groq() now ALWAYS tries GROQ_MODELS in the exact configured order.
#      The old get_groq_models() discovery call used to silently re-filter/drop
#      the primary model before it was ever tried if Groq's /models endpoint
#      didn't happen to list it — that was the actual cause of "it skips model
#      1 and uses model 3". Discovery is now diagnostic-only (Settings tab).
#   2. Each model gets its own max_completion_tokens + reasoning_effort
#      (openai/gpt-oss-120b: 4000 tokens, reasoning_effort='low') instead of a
#      single 950-token cap, which was starving the reasoning model of room to
#      finish its JSON output.
#   3. Every signal is now a genuine resting LIMIT order inside a real
#      min/max "entry gap" band (get_entry_gap_bounds / compute_default_limit_
#      entry). The old code capped that band so tight (~0.25% / 0.9 ATR) that
#      any real pullback level got rejected and silently replaced with a
#      market-price entry + order_type='MARKET'. That fallback is gone.
#   4. No more WAIT signals anywhere in the resolved output — every dead end
#      (missing AI entry, failed structural plan, failed final validation)
#      now resolves through build_atr_fallback_plan() to a deterministic,
#      risk-defined BUY/SELL LIMIT instead of surrendering.
#   5. Added next_level_watch (AI-populated, with a Python-computed fallback)
#      so each signal also states the next structural level price may
#      approach beyond TP.
#   6. Settings tab now shows per-model call diagnostics and a live Groq
#      /models check, so you can see exactly what happened on any given run.
# =============================================================================
import os, json, requests, time, re, random, traceback, uuid, html, base64, io
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    import yfinance as yf
except Exception:
    yf = None

# ── Page Config & UI Styling ───────────────────────────────────────────────
st.set_page_config(page_title="Der-AI | Institutional Market Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stButton>button { background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); color: white; border: none; padding: 10px 24px; border-radius: 8px; font-weight: bold; font-size: 16px; width: 100%; }
    .stButton>button:hover { background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%); }
    .signal-card { background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 6px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .buy-signal { border-left-color: #10b981; }
    .sell-signal { border-left-color: #ef4444; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .debug-box { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ── Core Helpers & Session State ───────────────────────────────────────────
def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, None)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)

TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "")
SYMBOLS = ['XAUUSD', 'EURUSD', 'BTCUSD', 'US30']
YFINANCE_MAP = {'XAUUSD': 'GC=F', 'EURUSD': 'EURUSD=X', 'BTCUSD': 'BTC-USD', 'US30': '^DJI', 'DXY': 'DX-Y.NYB'}
MINIMUM_CONFLUENCE_SCORE = 72
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MIN_REQUEST_INTERVAL = 3
GROQ_TOKEN_LIMIT_PER_MINUTE = 1000000
GROQ_REQUEST_TIMEOUT = 90
GROQ_MAX_RETRIES_PER_MODEL = 2          # retries on transient errors (timeout/5xx) before moving to the next model
GROQ_RETRY_BACKOFF_SECONDS = 2.5

# GPT-OSS models on Groq are *reasoning* models: a large share of the completion-token
# budget is consumed by hidden chain-of-thought before the final JSON is emitted. The old
# fixed 950-token cap starved openai/gpt-oss-120b of room to finish, so it returned
# truncated/invalid JSON, which made the loop silently fall through to qwen3-32b — this is
# why the app looked like it was "skipping" the first model. Each model now gets its own
# budget and reasoning_effort tuned so the primary model actually completes.
# Ordering here is authoritative — models are ALWAYS attempted in this exact order.
GROQ_MODELS = [
    'openai/gpt-oss-120b',  # Primary: Groq's high-reasoning flagship (131K ctx)
    'openai/gpt-oss-20b',   # Secondary: Lighter/faster GPT-OSS fallback
    'qwen/qwen3-32b'        # Tertiary: Qwen fallback
]
GROQ_MODEL_CONFIG = {
    'openai/gpt-oss-120b': {'max_completion_tokens': 4000, 'reasoning_effort': 'low', 'supports_reasoning_effort': True},
    'openai/gpt-oss-20b':  {'max_completion_tokens': 3200, 'reasoning_effort': 'low', 'supports_reasoning_effort': True},
    'qwen/qwen3-32b':      {'max_completion_tokens': 2200, 'reasoning_effort': 'none', 'supports_reasoning_effort': True},
}
GROQ_DEFAULT_MODEL_CONFIG = {'max_completion_tokens': 2000, 'reasoning_effort': None, 'supports_reasoning_effort': False}
# Kept only as a legacy default for token-budget estimation; actual requests use the
# per-model 'max_completion_tokens' above.
GROQ_MAX_OUTPUT_TOKENS = 4000
GROQ_ESTIMATED_RESPONSE_TOKENS = GROQ_MAX_OUTPUT_TOKENS

PYTHON_FALLBACK_MODEL = 'Python fallback (rule-based MTF confluence)'

if 'signal_history' not in st.session_state: st.session_state.signal_history = []
if 'notifications' not in st.session_state: st.session_state.notifications = []
if 'cached_market_data' not in st.session_state: st.session_state.cached_market_data = {}
if 'last_market_fetch_time' not in st.session_state: st.session_state.last_market_fetch_time = None
if 'active_signals' not in st.session_state: st.session_state.active_signals = {}
if 'directional_bias' not in st.session_state: st.session_state.directional_bias = {}
if 'signal_ledger' not in st.session_state: st.session_state.signal_ledger = []
if 'learning_stats' not in st.session_state: st.session_state.learning_stats = {}
if 'market_state' not in st.session_state: st.session_state.market_state = 'coiling'
if 'state_history' not in st.session_state: st.session_state.state_history = []
if 'groq_tokens_used' not in st.session_state: st.session_state.groq_tokens_used = 0
if 'groq_token_window_start' not in st.session_state: st.session_state.groq_token_window_start = datetime.now()
if 'last_groq_request_time' not in st.session_state: st.session_state.last_groq_request_time = None
if 'groq_rate_limit_until' not in st.session_state: st.session_state.groq_rate_limit_until = None
if 'groq_rate_limit_reason' not in st.session_state: st.session_state.groq_rate_limit_reason = ''
if 'cached_analysis' not in st.session_state: st.session_state.cached_analysis = {}
if 'last_model_attempts' not in st.session_state: st.session_state.last_model_attempts = {}

def add_notification(note_type, message, symbol=None, signal=None, score=None):
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    signal = normalize_ai_signal(signal) if signal is not None else signal
    notification = {
        'id': str(uuid.uuid4()),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': note_type,
        'message': message,
        'symbol': symbol,
        'signal': signal,
        'score': score,
        'read': False
    }
    st.session_state.notifications.append(notification)
    if len(st.session_state.notifications) > 200:
        st.session_state.notifications = st.session_state.notifications[-200:]
    return notification

def get_notifications():
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    return st.session_state.notifications

def clear_notifications():
    st.session_state.notifications = []

def _escape_telegram_html(text):
    if text is None:
        return ''
    return html.escape(str(text), quote=False)

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        try:
            data = resp.json()
        except Exception:
            data = None
        if resp.status_code == 200 and (data is None or data.get('ok', True)):
            return True
        else:
            print(f"Telegram send failed: status={resp.status_code} body={resp.text}")
            return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def build_telegram_signal_message(symbol, result):
    tp_values = result.get('take_profit', [])
    tp_value = tp_values[0] if tp_values else 'N/A'
    score = result.get('confluence_score', 0)
    signal = normalize_ai_signal(result.get('signal'))
    signal = _escape_telegram_html(signal)
    reasoning = _escape_telegram_html(result.get('reasoning'))
    order_type = _escape_telegram_html(result.get('order_type', 'LIMIT'))
    model = _escape_telegram_html(result.get('model_used', 'Unknown'))
    tokens = result.get('total_tokens', 'N/A')
    return (
        f"🌍 <b>DER-AI MARKET SIGNAL</b>\n"
        f"📊 <b>{_escape_telegram_html(symbol)}</b> - {signal}\n"
        f"🤖 Model: {model} | 📈 Score: {score}/100 | 🔋 Tokens: {tokens}\n"
        f"🧾 Order: {order_type}\n"
        f"💰 Entry: {_escape_telegram_html(result.get('entry'))} | 🛑 SL: {_escape_telegram_html(result.get('stop_loss'))} | 🎯 TP: {_escape_telegram_html(tp_value)}\n"
        f"📈 DXY: {_escape_telegram_html(result.get('dxy_correlation'))}\n"
        f"🧠 {reasoning}"
    )

# ── Data Fetching & SMC Engines (Complete Original Logic) ──────────────────
def _build_dataframe_from_records(records):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['datetime'], utc=True)
    else:
        return pd.DataFrame()
    df = df.set_index('Date').sort_index()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

def fetch_candles_from_bitfinex(symbol, interval, limit=200):
    pair_map = {'BTCUSD': 'tBTCUSD', 'XAUUSD': 'tXAUT:USD', 'EURUSD': 'tEURUSD', 'DXY': None}
    bitfinex_symbol = pair_map.get(symbol)
    if not bitfinex_symbol:
        return pd.DataFrame()
    interval_map = {'15m': '15m', '30m': '30m', '60m': '1h', '1h': '1h', '4h': '4h'}
    interval_code = interval_map.get(interval)
    if not interval_code:
        return pd.DataFrame()
    try:
        url = f'https://api-pub.bitfinex.com/v2/candles/trade:{interval_code}:{bitfinex_symbol}/hist?limit={limit}'
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return pd.DataFrame()
        records = []
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                continue
            ts, open_price, close_price, high_price, low_price, volume = item[:6]
            records.append({
                'timestamp': ts,
                'Open': float(open_price),
                'High': float(high_price),
                'Low': float(low_price),
                'Close': float(close_price),
                'Volume': float(volume)
            })
        return _build_dataframe_from_records(records)
    except Exception as exc:
        print(f"⚠️ Bitfinex fetch failed for {symbol} [{interval}]: {exc}")
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(yf_symbol, interval, period):
    if yf is None:
        return pd.DataFrame()
    for symbol in ['BTCUSD', 'XAUUSD', 'EURUSD', 'DXY']:
        if yf_symbol in {symbol, YFINANCE_MAP.get(symbol, symbol)}:
            direct_df = fetch_candles_from_bitfinex(symbol, interval, limit=250)
            if not direct_df.empty:
                return direct_df
            break
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False, timeout=30)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if df.empty:
            return pd.DataFrame()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        if 'Date' in df.columns:
            df = df.set_index('Date')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        return df if len(df) >= 5 else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Yahoo fetch failed for {yf_symbol} [{interval}/{period}]: {e}")
        return pd.DataFrame()

def fetch_symbol_data(symbol, yf_symbol):
    df_m10 = pd.DataFrame()
    for interval, period in [('10m', '5d'), ('15m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m10 = df_candidate
            break
    if df_m10.empty:
        for interval, period in [('15m', '5d'), ('30m', '5d'), ('60m', '5d')]:
            df_candidate = fetch_ohlcv(yf_symbol, interval, period)
            if not df_candidate.empty:
                df_m10 = df_candidate
                break
    df_m15 = pd.DataFrame()
    for interval, period in [('15m', '5d'), ('30m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m15 = df_candidate
            break
    df_m30 = pd.DataFrame()
    for interval, period in [('30m', '5d'), ('60m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m30 = df_candidate
            break
    df_h1 = fetch_ohlcv(yf_symbol, '1h', '1mo')
    if df_h1.empty:
        df_h1 = fetch_ohlcv(yf_symbol, '60m', '1mo')
    df_h4 = fetch_ohlcv(yf_symbol, '4h', '3mo')
    if df_h4.empty:
        df_h4 = fetch_ohlcv(yf_symbol, '1d', '6mo')
    if df_m10.empty and not df_m15.empty:
        df_m10 = df_m15
    if df_m30.empty and not df_h1.empty:
        df_m30 = df_h1
    return {'M10': df_m10, 'M15': df_m15, 'M30': df_m30, 'H1': df_h1, 'H4': df_h4}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_all_data():
    data = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=min(5, len(YFINANCE_MAP))) as executor:
        for symbol, yf_symbol in YFINANCE_MAP.items():
            futures[executor.submit(fetch_symbol_data, symbol, yf_symbol)] = symbol
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data[symbol] = future.result()
            except Exception as e:
                print(f"❌ Exception fetching {symbol}: {str(e)}")
                data[symbol] = {'M10': pd.DataFrame(), 'H1': pd.DataFrame(), 'H4': pd.DataFrame()}
    return data

def calculate_microstructure(df):
    if df is None or len(df) < 2:
        return {}
    try:
        typical_price = (pd.to_numeric(df['High'], errors='coerce') + pd.to_numeric(df['Low'], errors='coerce') + pd.to_numeric(df['Close'], errors='coerce')) / 3
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
        current_vwap = float(vwap.iloc[-1])
        current_price = float(pd.to_numeric(df['Close'], errors='coerce').iloc[-1])
        avg_volume = volume.rolling(window=min(20, len(volume))).mean().iloc[-1]
        current_volume = float(volume.iloc[-1])
        rvol = current_volume / avg_volume if avg_volume > 0 else 1.0
        anchor_index = -min(5, len(df))
        price_change = current_price - float(pd.to_numeric(df['Close'], errors='coerce').iloc[anchor_index])
        return {
            "vwap": round(current_vwap, 2),
            "price_vs_vwap": "ABOVE" if current_price > current_vwap else "BELOW",
            "rvol": round(rvol, 2),
            "volume_anomaly": "HIGH_INSTITUTIONAL" if rvol > 2.0 else "NORMAL",
            "momentum": "BULLISH" if price_change > 0 else "BEARISH"
        }
    except Exception:
        return {}

def detect_bos_choch(df):
    if df is None or len(df) < 2:
        return None, None
    try:
        highs = pd.to_numeric(df['High'], errors='coerce').dropna()
        lows = pd.to_numeric(df['Low'], errors='coerce').dropna()
        if highs.empty or lows.empty or len(df) < 4:
            return None, None
        recent_high = float(highs.iloc[-1])
        prev_high = float(highs.iloc[-2]) if len(highs) >= 2 else recent_high
        recent_low = float(lows.iloc[-1])
        prev_low = float(lows.iloc[-2]) if len(lows) >= 2 else recent_low
        bos, choch = None, None
        if recent_high > prev_high * 1.001:
            bos = "BULLISH_BOS"
        elif recent_low < prev_low * 0.999:
            bos = "BEARISH_BOS"
        if bos == "BULLISH_BOS" and recent_low > prev_low:
            choch = "BULLISH_CHOCH"
        elif bos == "BEARISH_BOS" and recent_high < prev_high:
            choch = "BEARISH_CHOCH"
        return bos, choch
    except Exception:
        return None, None

def find_swings(df, window=5):
    highs = df['High'].rolling(window * 2 + 1, center=True).max()
    lows = df['Low'].rolling(window * 2 + 1, center=True).min()
    return {
        "recent_swing_highs": [round(p, 5) for p in df['High'][df['High'] == highs].tail(4).tolist()],
        "recent_swing_lows": [round(p, 5) for p in df['Low'][df['Low'] == lows].tail(4).tolist()]
    }

def detect_order_blocks(df):
    if len(df) < 5:
        return []
    order_blocks = []
    for i in range(len(df)-3, len(df)):
        if i < 2:
            continue
        candle, prev_candle = df.iloc[i], df.iloc[i-1]
        if (candle['Close'] > candle['Open'] and
            (candle['Close'] - candle['Open']) > (candle['High'] - candle['Low']) * 0.6 and
            prev_candle['Close'] < prev_candle['Open']):
            order_blocks.append({
                'type': 'BULLISH_OB',
                'price': candle['Low'],
                'strength': 'STRONG' if (candle['Close'] - candle['Open']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
        if (candle['Close'] < candle['Open'] and
            (candle['Open'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.6 and
            prev_candle['Close'] > prev_candle['Open']):
            order_blocks.append({
                'type': 'BEARISH_OB',
                'price': candle['High'],
                'strength': 'STRONG' if (candle['Open'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
    return order_blocks[-3:]

def detect_fvg(df):
    if len(df) < 3:
        return []
    fvgs = []
    for i in range(len(df)-2, len(df)):
        if i < 2:
            continue
        curr, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        if prev['Low'] > prev2['High'] and curr['Low'] > prev['High']:
            fvgs.append({'type': 'BULLISH_FVG', 'top': prev['Low'], 'bottom': prev2['High']})
        if prev['High'] < prev2['Low'] and curr['High'] < prev['Low']:
            fvgs.append({'type': 'BEARISH_FVG', 'top': prev2['Low'], 'bottom': prev['High']})
    return fvgs[-2:]

def detect_liquidity_sweeps(df):
    if len(df) < 10:
        return []
    sweeps = []
    recent = df.tail(10)
    for i in range(1, len(recent)):
        candle, prev = recent.iloc[i], recent.iloc[i-1]
        if (candle['Low'] < prev['Low'] * 0.999 and
            candle['Close'] > candle['Open'] and
            (candle['Close'] - candle['Low']) > (candle['High'] - candle['Low']) * 0.6):
            sweeps.append({
                'type': 'BULLISH_SWEEP',
                'price': candle['Low'],
                'strength': 'STRONG' if (candle['Close'] - candle['Low']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
        if (candle['High'] > prev['High'] * 1.001 and
            candle['Close'] < candle['Open'] and
            (candle['High'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.6):
            sweeps.append({
                'type': 'BEARISH_SWEEP',
                'price': candle['High'],
                'strength': 'STRONG' if (candle['High'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'
            })
    return sweeps[-2:]

def calculate_atr(df, period=14):
    if len(df) < period + 2:
        return None
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None

def calculate_rsi(series, period=14):
    if series is None:
        return pd.Series(dtype=float)
    original = pd.Series(series)
    if len(original) < 2:
        return pd.Series([np.nan] * len(original), index=original.index, dtype=float)
    series = pd.to_numeric(original, errors='coerce').dropna()
    if series.empty:
        return pd.Series([np.nan] * len(original), index=original.index, dtype=float)
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    window = min(period, len(series))
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(original.index, fill_value=50).astype(float)

def detect_rsi_divergence(df, period=14):
    if df is None or df.empty:
        return None
    closes = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if closes.empty or len(closes) < 4:
        return None
    rsi = calculate_rsi(closes, period=period)
    if rsi.empty:
        return None
    pivot_highs = []
    pivot_lows = []
    for i in range(2, len(closes) - 2):
        prev2 = closes.iloc[i - 2]
        prev1 = closes.iloc[i - 1]
        curr = closes.iloc[i]
        next1 = closes.iloc[i + 1]
        next2 = closes.iloc[i + 2]
        if curr >= prev1 and curr >= next1 and curr >= prev2 and curr >= next2:
            pivot_highs.append((i, float(curr), float(rsi.iloc[i])))
        if curr <= prev1 and curr <= next1 and curr <= prev2 and curr <= next2:
            pivot_lows.append((i, float(curr), float(rsi.iloc[i])))
    if len(pivot_highs) >= 2:
        prev_high = pivot_highs[-2]
        curr_high = pivot_highs[-1]
        if curr_high[1] < prev_high[1] and curr_high[2] > prev_high[2]:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is printing a lower high while RSI is holding a higher high, suggesting bullish divergence.'}
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is printing a higher high while RSI is failing, suggesting bearish divergence.'}
    if len(pivot_lows) >= 2:
        prev_low = pivot_lows[-2]
        curr_low = pivot_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is printing a lower low while RSI is holding a higher low, suggesting bullish divergence.'}
        if curr_low[1] > prev_low[1] and curr_low[2] < prev_low[2]:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is printing a higher low while RSI is weakening, suggesting bearish divergence.'}
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    last_rsi = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2])
    if last_close < prev_close and last_rsi > prev_rsi:
        return {'type': 'BULLISH_DIV', 'reason': 'The latest candles show a bullish RSI divergence against the recent price decline.'}
    if last_close > prev_close and last_rsi < prev_rsi:
        return {'type': 'BEARISH_DIV', 'reason': 'The latest candles show a bearish RSI divergence against the recent price advance.'}
    if len(closes) >= 6:
        lookback = min(5, len(closes) - 1)
        price_change = float(closes.iloc[-1] - closes.iloc[-lookback - 1])
        rsi_change = float(rsi.iloc[-1] - rsi.iloc[-lookback - 1])
        if price_change <= 0 and rsi_change > 0:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is failing to continue lower while RSI is rising, which is bullish divergence.'}
        if price_change >= 0 and rsi_change < 0:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is pushing higher while RSI is weakening, which is bearish divergence.'}
    return None

def detect_reversal(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    body = abs(last['Close'] - last['Open'])
    total_range = last['High'] - last['Low']
    if total_range == 0:
        return None
    wick_ratio = max(last['High'] - max(last['Open'], last['Close']), min(last['Open'], last['Close']) - last['Low']) / total_range
    body_ratio = body / total_range
    bullish_reversal = last['Close'] > last['Open'] and last['Close'] >= prev['Close'] and wick_ratio > 0.45 and body_ratio < 0.45 and last['Low'] <= min(prev['Low'], prev2['Low'])
    bearish_reversal = last['Close'] < last['Open'] and last['Close'] <= prev['Close'] and wick_ratio > 0.45 and body_ratio < 0.45 and last['High'] >= max(prev['High'], prev2['High'])
    if bullish_reversal or bearish_reversal:
        return {
            'type': 'BULLISH_REVERSAL' if bullish_reversal else 'BEARISH_REVERSAL',
            'direction': 'BUY' if bullish_reversal else 'SELL',
            'strength': 'STRONG' if wick_ratio > 0.6 else 'MODERATE',
            'reason': 'The market is rejecting a previous extreme and showing a clean reversal candle.'
        }
    return None

def detect_continuation(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent_high = max(df['High'].tail(4).iloc[:-1])
    recent_low = min(df['Low'].tail(4).iloc[:-1])
    bullish_continuation = last['Close'] > prev['Close'] and last['Close'] > recent_high and last['Low'] > recent_low
    bearish_continuation = last['Close'] < prev['Close'] and last['Close'] < recent_low and last['High'] < recent_high
    if bullish_continuation or bearish_continuation:
        return {
            'type': 'BULLISH_CONTINUATION' if bullish_continuation else 'BEARISH_CONTINUATION',
            'direction': 'BUY' if bullish_continuation else 'SELL',
            'strength': 'STRONG' if abs(last['Close'] - prev['Close']) > (last['High'] - last['Low']) * 0.5 else 'MODERATE',
            'reason': 'Price is extending beyond the recent range with follow-through, supporting continuation.'
        }
    return None

def detect_exhaustion(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    total_range = last['High'] - last['Low']
    if total_range == 0:
        return None
    wick = max(last['High'] - max(last['Open'], last['Close']), min(last['Open'], last['Close']) - last['Low'])
    wick_ratio = wick / total_range
    body_ratio = body / total_range
    vol = float(last['Volume']) if 'Volume' in df.columns else 0.0
    avg_vol = float(df['Volume'].tail(10).mean()) if 'Volume' in df.columns else 0.0
    volume_spike = vol / avg_vol if avg_vol > 0 else 1.0
    bullish_exhaustion = last['Close'] > prev['Close'] and wick_ratio > 0.55 and body_ratio < 0.35 and volume_spike > 1.1
    bearish_exhaustion = last['Close'] < prev['Close'] and wick_ratio > 0.55 and body_ratio < 0.35 and volume_spike > 1.1
    if bullish_exhaustion or bearish_exhaustion:
        return {
            'type': 'BULLISH_EXHAUSTION' if bullish_exhaustion else 'BEARISH_EXHAUSTION',
            'direction': None,
            'strength': 'STRONG' if volume_spike > 1.3 else 'MODERATE',
            'reason': 'Price made a stretched wick into the current bar and closed with a weak body, signaling exhaustion.'
        }
    return None

def assess_entry_quality(df, swings, current_price, signal):
    if not swings:
        return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
    if signal == 'BUY':
        swing_lows = [level for level in swings.get('recent_swing_lows', []) if level < current_price]
        zone = swing_lows[0] if swing_lows else None
        if zone is None:
            return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
        distance_pct = abs(current_price - zone) / current_price
        quality = 'early' if distance_pct < 0.003 else 'acceptable' if distance_pct < 0.008 else 'late'
        return {'entry_quality': quality, 'entry_zone': zone, 'distance_pct': round(distance_pct, 6)}
    swing_highs = [level for level in swings.get('recent_swing_highs', []) if level > current_price]
    zone = swing_highs[0] if swing_highs else None
    if zone is None:
        return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
    distance_pct = abs(zone - current_price) / current_price
    quality = 'early' if distance_pct < 0.003 else 'acceptable' if distance_pct < 0.008 else 'late'
    return {'entry_quality': quality, 'entry_zone': zone, 'distance_pct': round(distance_pct, 6)}

def detect_market_phase(df, swings=None):
    micro = calculate_microstructure(df)
    continuation = detect_continuation(df)
    reversal = detect_reversal(df)
    exhaustion = detect_exhaustion(df)
    divergence = detect_rsi_divergence(df)
    entry_quality = assess_entry_quality(df, swings, df['Close'].iloc[-1], 'BUY' if micro.get('momentum') == 'BULLISH' else 'SELL') if swings is not None else {'entry_quality': 'unknown'}
    direction = None
    if continuation:
        phase = 'continuation'
        reason = continuation['reason']
        direction = continuation.get('direction')
    elif reversal:
        phase = 'reversal'
        reason = reversal['reason']
        direction = reversal.get('direction')
    elif exhaustion:
        phase = 'exhaustion'
        reason = exhaustion['reason']
    elif micro.get('momentum') == 'BULLISH' and micro.get('price_vs_vwap') == 'ABOVE':
        phase = 'trend'
        reason = 'Price is holding above VWAP with positive momentum and is still in a directional trend.'
        direction = 'BUY'
    elif micro.get('momentum') == 'BEARISH' and micro.get('price_vs_vwap') == 'BELOW':
        phase = 'trend'
        reason = 'Price is holding below VWAP with negative momentum and is still in a directional trend.'
        direction = 'SELL'
    else:
        phase = 'coiling'
        reason = 'The market is coiling and lacks a clear continuation or reversal edge yet.'
    if divergence:
        direction = 'BUY' if divergence['type'] == 'BULLISH_DIV' else 'SELL'
        reason += f" RSI divergence hint: {divergence['reason']}"
    return {
        'phase': phase,
        'reason': reason,
        'direction': direction,
        'entry_quality': entry_quality.get('entry_quality', 'unknown'),
        'entry_zone': entry_quality.get('entry_zone'),
        'distance_pct': entry_quality.get('distance_pct')
    }

def range_position(df, current_price):
    if df is None or df.empty:
        return None
    look = df.tail(60)
    hi = float(look['High'].max())
    lo = float(look['Low'].min())
    if hi <= lo:
        return None
    return (float(current_price) - lo) / (hi - lo), lo, hi

def build_premium_discount_context(m10, current_price):
    pos_info = range_position(m10, current_price)
    if not pos_info:
        return 'Range position unavailable.'
    pos, lo, hi = pos_info
    zone = 'PREMIUM (upper half of range - SMC favors sells/shorts here)' if pos >= 0.5 else 'DISCOUNT (lower half of range - SMC favors buys/longs here)'
    return f"Recent 60-bar range {lo:.2f}-{hi:.2f}; price at {pos * 100:.0f}% of the range => {zone}."

def build_volatility_context(m10):
    atr = calculate_atr(m10)
    if atr is None or m10 is None or m10.empty:
        return 'ATR unavailable.'
    last = float(m10['Close'].iloc[-1])
    atr_pct = atr / last * 100 if last else 0.0
    return f"M10 ATR {atr:.2f} ({atr_pct:.2f}% of price). Judge whether volatility is expanded or compressed versus recent sessions."

def build_historical_context(m10):
    if m10 is None or m10.empty:
        return 'Historical context unavailable.'
    closes = [round(float(v), 4) for v in m10['Close'].tail(20).tolist()]
    if len(closes) < 2:
        return 'Historical context unavailable.'
    latest = float(m10['Close'].iloc[-1])
    prev = float(m10['Close'].iloc[-2])
    change_pct = round(((latest - prev) / prev) * 100, 2) if prev else 0.0
    high = float(m10['High'].tail(20).max())
    low = float(m10['Low'].tail(20).min())
    range_pct = round(((high - low) / latest) * 100, 2) if latest else 0.0
    return f"Last 20 closes: {closes}; latest 1-bar change: {change_pct}%; recent 20-bar range: {range_pct}%."

def compute_rsi_last(series, period=14):
    rsi = calculate_rsi(series, period)
    if rsi is None or rsi.empty:
        return None
    val = rsi.dropna()
    if val.empty:
        return None
    return round(float(val.iloc[-1]), 1)

def build_rsi_values_context(all_data, symbol):
    data = all_data.get(symbol, {}) or {}
    parts = []
    for label, key in [('M10', 'M10'), ('M15', 'M15'), ('M30', 'M30'), ('H1', 'H1'), ('H4', 'H4')]:
        df = data.get(key)
        if df is None or getattr(df, 'empty', True):
            continue
        v = compute_rsi_last(pd.to_numeric(df['Close'], errors='coerce'))
        if v is not None:
            state = 'OVERBOUGHT' if v >= 70 else 'OVERSOLD' if v <= 30 else 'neutral'
            parts.append(f"{label} RSI {v} ({state})")
    return ' | '.join(parts) if parts else 'RSI values unavailable.'

def build_market_structure_summary(df, current_price=None, swings=None, order_blocks=None, fvgs=None, sweeps=None, bos=None, choch=None, symbol=None):
    if df is None or df.empty:
        return 'Market structure unavailable.'
    current_price = float(current_price) if current_price is not None else float(df['Close'].iloc[-1])
    swings = swings or find_swings(df)
    recent_swing_highs = [float(v) for v in swings.get('recent_swing_highs', []) if v is not None]
    recent_swing_lows = [float(v) for v in swings.get('recent_swing_lows', []) if v is not None]
    zone_buffer = max(abs(current_price) * 0.0015, 0.5 if current_price >= 100 else 0.01)
    support_zones = [f"{max(level - zone_buffer, 0):.2f}-{level + zone_buffer:.2f}" for level in recent_swing_lows[:3]]
    resistance_zones = [f"{level - zone_buffer:.2f}-{level + zone_buffer:.2f}" for level in recent_swing_highs[:3]]
    demand_zones = []
    supply_zones = []
    for ob in (order_blocks or [])[:3]:
        if ob['type'].startswith('BULLISH'):
            demand_zones.append(f"{ob['price'] - zone_buffer:.2f}-{ob['price'] + zone_buffer:.2f}")
        else:
            supply_zones.append(f"{ob['price'] - zone_buffer:.2f}-{ob['price'] + zone_buffer:.2f}")
    for fvg in (fvgs or [])[:2]:
        if fvg['type'].startswith('BULLISH'):
            demand_zones.append(f"{fvg['bottom']:.2f}-{fvg['top']:.2f}")
        else:
            supply_zones.append(f"{fvg['bottom']:.2f}-{fvg['top']:.2f}")
    for sweep in (sweeps or [])[:2]:
        if sweep['type'].startswith('BULLISH'):
            demand_zones.append(f"{sweep['price'] - zone_buffer:.2f}-{sweep['price'] + zone_buffer:.2f}")
        else:
            supply_zones.append(f"{sweep['price'] - zone_buffer:.2f}-{sweep['price'] + zone_buffer:.2f}")
    structure_markers = []
    if bos:
        structure_markers.append(bos)
    if choch:
        structure_markers.append(choch)
    structure_markers = structure_markers[:2]
    support_text = ', '.join(support_zones) if support_zones else 'none'
    resistance_text = ', '.join(resistance_zones) if resistance_zones else 'none'
    demand_text = ', '.join(demand_zones) if demand_zones else 'none'
    supply_text = ', '.join(supply_zones) if supply_zones else 'none'
    marker_text = ', '.join(structure_markers) if structure_markers else 'balanced'
    return (
        f"Market structure zones: support={support_text}; resistance={resistance_text}; "
        f"demand={demand_text}; supply={supply_text}; structure markers={marker_text}; "
        f"current price={current_price:.2f}. The AI must identify whether price is approaching a demand/supply or support/resistance zone, "
        f"and whether the next move is a continuation or a reversal into the next clear invalidation zone."
    )

def build_multitimeframe_context(all_data, symbol):
    if not all_data or not symbol:
        return 'Multi-timeframe context unavailable.'
    data = all_data.get(symbol, {})
    if not isinstance(data, dict):
        return 'Multi-timeframe context unavailable.'
    frames = [('10M', data.get('M10')), ('15M', data.get('M15')), ('30M', data.get('M30')), ('1H', data.get('H1')), ('4H', data.get('H4'))]
    parts = []
    for label, frame in frames:
        if frame is None or getattr(frame, 'empty', True):
            continue
        micro = calculate_microstructure(frame)
        if not micro:
            micro = {'price_vs_vwap': 'NEUTRAL', 'momentum': 'NEUTRAL', 'rvol': 1.0}
        bos, choch = detect_bos_choch(frame)
        order_blocks = detect_order_blocks(frame)
        fvgs = detect_fvg(frame)
        sweeps = detect_liquidity_sweeps(frame)
        divergence = detect_rsi_divergence(frame)
        structure_bits = []
        if bos:
            structure_bits.append(bos)
        if choch:
            structure_bits.append(choch)
        if order_blocks:
            structure_bits.append(f"OB:{order_blocks[-1]['type']}")
        if fvgs:
            structure_bits.append(f"FVG:{fvgs[-1]['type']}")
        if sweeps:
            structure_bits.append(f"SWP:{sweeps[-1]['type']}")
        if divergence:
            structure_bits.append(divergence['type'])
        structure_text = ', '.join(structure_bits) if structure_bits else 'balanced structure'
        parts.append(f"{label}: price is {micro['price_vs_vwap']} VWAP, momentum is {micro['momentum']}, RVOL is {micro['rvol']:.2f}, and the current read is {structure_text}.")
    if not parts:
        return 'No usable higher-timeframe or lower-timeframe context is available.'
    return ' '.join(parts)

def detect_directional_confluence(df, swings=None, htf_context=None, dxy_context=None, symbol=None):
    if df is None or df.empty:
        return {'direction': None, 'bullish_evidence': [], 'bearish_evidence': [], 'bull_count': 0, 'bear_count': 0}
    bullish_evidence = []
    bearish_evidence = []
    micro = calculate_microstructure(df)
    reversal = detect_reversal(df)
    continuation = detect_continuation(df)
    sweeps = detect_liquidity_sweeps(df)
    divergence = detect_rsi_divergence(df)
    if reversal:
        if reversal.get('direction') == 'BUY':
            bullish_evidence.append(f"reversal candle rejecting lows ({reversal['type']})")
        else:
            bearish_evidence.append(f"reversal candle rejecting highs ({reversal['type']})")
    if continuation:
        if continuation.get('direction') == 'BUY':
            bullish_evidence.append(f"continuation break to the upside ({continuation['type']})")
        else:
            bearish_evidence.append(f"continuation break to the downside ({continuation['type']})")
    for sweep in sweeps[-2:]:
        if sweep['type'] == 'BULLISH_SWEEP':
            bullish_evidence.append('liquidity sweep of lows reclaimed')
        else:
            bearish_evidence.append('liquidity sweep of highs rejected')
    if divergence:
        if divergence['type'] == 'BULLISH_DIV':
            bullish_evidence.append('bullish RSI divergence')
        else:
            bearish_evidence.append('bearish RSI divergence')
    if micro.get('momentum') == 'BULLISH' and micro.get('price_vs_vwap') == 'ABOVE':
        bullish_evidence.append('price holding above VWAP with bullish momentum')
    elif micro.get('momentum') == 'BEARISH' and micro.get('price_vs_vwap') == 'BELOW':
        bearish_evidence.append('price holding below VWAP with bearish momentum')
    pos_info = range_position(df, df['Close'].iloc[-1])
    if pos_info:
        pos, lo, hi = pos_info
        if pos <= 0.35:
            bullish_evidence.append('price in discount zone of the recent range (favors longs)')
        elif pos >= 0.65:
            bearish_evidence.append('price in premium zone of the recent range (favors shorts)')
    if isinstance(htf_context, dict):
        htf_trend = str(htf_context.get('trend') or htf_context.get('bias') or '').upper()
        if htf_trend == 'BULLISH':
            bullish_evidence.append('higher-timeframe trend bullish')
        elif htf_trend == 'BEARISH':
            bearish_evidence.append('higher-timeframe trend bearish')
    if dxy_context and symbol in ['XAUUSD', 'EURUSD', 'BTCUSD']:
        trend = dxy_context.get('trend')
        pos = dxy_context.get('price_vs_vwap')
        if trend == 'BEARISH' and pos == 'BELOW':
            bullish_evidence.append('DXY weakness supporting longs')
        elif trend == 'BULLISH' and pos == 'ABOVE':
            bearish_evidence.append('DXY strength supporting shorts')
    bull = len(bullish_evidence)
    bear = len(bearish_evidence)
    direction = None
    if bull >= 2 and bull > bear:
        direction = 'BUY'
    elif bear >= 2 and bear > bull:
        direction = 'SELL'
    return {'direction': direction, 'bullish_evidence': bullish_evidence, 'bearish_evidence': bearish_evidence, 'bull_count': bull, 'bear_count': bear}

def build_mtf_picture(all_data, symbol):
    data = all_data.get(symbol, {}) or {}
    specs = [('M10', 1.0), ('M15', 1.0), ('M30', 1.5), ('H1', 2.0), ('H4', 3.0)]
    snaps = []
    for label, weight in specs:
        df = data.get(label)
        if df is None or getattr(df, 'empty', True):
            continue
        micro = calculate_microstructure(df)
        bos, choch = detect_bos_choch(df)
        sweeps = detect_liquidity_sweeps(df)
        divergence = detect_rsi_divergence(df)
        reversal = detect_reversal(df)
        continuation = detect_continuation(df)
        snaps.append({
            'label': label,
            'weight': weight,
            'micro': micro,
            'bos': bos,
            'choch': choch,
            'sweeps': sweeps,
            'divergence': divergence,
            'reversal': reversal,
            'continuation': continuation
        })
    score = 0.0
    bull_items, bear_items = [], []
    for s in snaps:
        w = s['weight']
        m = s['micro']
        if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
            score += w
            bull_items.append(f"{s['label']} trend bullish (above VWAP)")
        elif m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
            score -= w
            bear_items.append(f"{s['label']} trend bearish (below VWAP)")
        if s['bos'] == 'BULLISH_BOS' or s['choch'] == 'BULLISH_CHOCH':
            score += w
            bull_items.append(f"{s['label']} {s['bos'] or s['choch']}")
        elif s['bos'] == 'BEARISH_BOS' or s['choch'] == 'BEARISH_CHOCH':
            score -= w
            bear_items.append(f"{s['label']} {s['bos'] or s['choch']}")
        for sw in s['sweeps'][-1:]:
            if sw['type'] == 'BULLISH_SWEEP':
                score += 0.5 * w
                bull_items.append(f"{s['label']} swept lows reclaimed")
            else:
                score -= 0.5 * w
                bear_items.append(f"{s['label']} swept highs rejected")
        if s['divergence']:
            if s['divergence']['type'] == 'BULLISH_DIV':
                score += 0.5 * w
                bull_items.append(f"{s['label']} bullish RSI divergence")
            else:
                score -= 0.5 * w
                bear_items.append(f"{s['label']} bearish RSI divergence")
        if s['reversal']:
            if s['reversal']['direction'] == 'BUY':
                score += w
                bull_items.append(f"{s['label']} reversal candle rejecting lows")
            else:
                score -= w
                bear_items.append(f"{s['label']} reversal candle rejecting highs")
        if s['continuation']:
            if s['continuation']['direction'] == 'BUY':
                score += w
                bull_items.append(f"{s['label']} continuation break upside")
            else:
                score -= w
                bear_items.append(f"{s['label']} continuation break downside")
    htf_reads = []
    for s in snaps:
        if s['label'] in ('H1', 'H4'):
            m = s['micro']
            if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
                htf_reads.append('BULLISH')
            elif m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
                htf_reads.append('BEARISH')
            else:
                htf_reads.append('NEUTRAL')
    htf_bias = 'BULLISH' if htf_reads and all(r == 'BULLISH' for r in htf_reads) else ('BEARISH' if htf_reads and all(r == 'BEARISH' for r in htf_reads) else 'NEUTRAL')
    return {
        'snaps': snaps,
        'score': score,
        'bull_items': bull_items,
        'bear_items': bear_items,
        'htf_bias': htf_bias
    }

BIAS_MIN_HOLD_MINUTES = 45

def resolve_firm_direction(symbol, picture):
    score = picture.get('score', 0)
    htf = picture.get('htf_bias', 'NEUTRAL')
    notes = []
    base = 'BUY' if score >= 2.0 else ('SELL' if score <= -2.0 else None)
    if base and htf != 'NEUTRAL' and base != htf:
        if abs(score) >= 6.0:
            notes.append(f"lower-timeframe evidence is overwhelming ({score:+.1f}), overriding the {htf} HTF bias")
        else:
            base = htf
            notes.append(f"HTF bias {htf} overrides conflicting lower-timeframe noise")
    firm = base if base else (htf if htf != 'NEUTRAL' and abs(score) >= 2.0 else None)
    now = datetime.now()
    stored = st.session_state.directional_bias.get(symbol)
    if stored and (now - stored['since']).total_seconds() < BIAS_MIN_HOLD_MINUTES * 60:
        standing = stored['direction']
        if firm and firm != standing:
            counter = picture['bull_items'] if firm == 'BUY' else picture['bear_items']
            has_htf_break = any(('H1' in it or 'H4' in it) and ('BOS' in it or 'CHOCH' in it) for it in counter)
            if has_htf_break and abs(score) >= 6.0:
                notes.append(f"standing {standing} bias overridden by an H1/H4 structural break with strong evidence ({score:+.1f})")
            elif abs(score) >= 4.0:
                firm = None
                notes.append(f"tape contradicts the standing {standing} bias but lacks an H1/H4 structural break - standing aside (WAIT) instead of flipping")
            else:
                firm = standing
                notes.append(f"maintaining the standing {standing} bias set at {stored['since'].strftime('%H:%M')} - insufficient proof to flip")
        elif firm is None and abs(score) < 4.0:
            firm = standing
            notes.append(f"tape is quiet - maintaining the standing {standing} bias set at {stored['since'].strftime('%H:%M')}")
        elif firm is None:
            notes.append(f"tape strongly contradicts the standing {standing} bias - standing aside (WAIT) until structure resolves")
    if firm:
        if not stored or stored['direction'] != firm:
            st.session_state.directional_bias[symbol] = {'direction': firm, 'since': now}
    elif stored and (now - stored['since']).total_seconds() >= BIAS_MIN_HOLD_MINUTES * 60:
        st.session_state.directional_bias.pop(symbol, None)
    return firm, notes

def get_pair_config(symbol):
    # NOTE ON THE LIMIT ZONE: 'min_entry_gap_pct'/'min_entry_gap_atr' set a FLOOR on how far
    # the entry must sit from live price (so a signal can never be a disguised market order),
    # while 'max_entry_gap_pct'/'max_entry_points'/'limit_zone_atr' set the CEILING (so the
    # limit stays realistic and likely to be tagged rather than sitting in no-man's land).
    # The old ceiling values here were tight enough (0.20-0.30% / ~0.9 ATR) that any genuine
    # structural pullback (an order block, FVG or swing level a reasonable distance away) got
    # rejected and silently replaced with a market-price entry. Both bounds are now wider and
    # symbol-appropriate, and a floor was added that did not exist before.
    base = {
        'digits': 2,
        'tick_size': 0.01,
        'min_dist_pct': 0.0015,
        'max_risk_pct': 0.008,
        'min_rr': 1.3,
        'target_rr': 2.0,
        'max_rr': 3.0,
        'score_floor': MINIMUM_CONFLUENCE_SCORE,
        'candidate_score': MINIMUM_CONFLUENCE_SCORE,
        'cooldown_minutes': 30,
        'min_entry_gap_pct': 0.0009,
        'min_entry_gap_atr': 0.12,
        'max_entry_gap_pct': 0.0085,
        'max_entry_points': 35,
        'min_stop_atr': 1.0,
        'max_stop_atr': 3.0,
        'stop_buffer_atr': 0.25,
        'tp_buffer_atr': 0.12,
        'limit_zone_atr': 2.2,
        'default_pullback_atr': 0.55,
        'spread_multiplier': 1.5,
    }
    overrides = {
        'XAUUSD': {
            'digits': 2,
            'tick_size': 0.01,
            'min_dist_pct': 0.0028,
            'max_risk_pct': 0.010,
            'min_entry_gap_pct': 0.0012,
            'min_entry_gap_atr': 0.12,
            'max_entry_gap_pct': 0.009,
            'max_entry_points': 35,
            'min_stop_atr': 1.2,
            'target_rr': 2.0,
            'max_rr': 3.0,
            'limit_zone_atr': 2.2,
            'default_pullback_atr': 0.55,
        },
        'EURUSD': {
            'digits': 5,
            'tick_size': 0.00001,
            'min_dist_pct': 0.0012,
            'max_risk_pct': 0.004,
            'min_entry_gap_pct': 0.0007,
            'min_entry_gap_atr': 0.12,
            'max_entry_gap_pct': 0.006,
            'max_entry_points': 0.0075,
            'min_stop_atr': 1.0,
            'target_rr': 2.0,
            'max_rr': 3.0,
            'max_stop_atr': 2.5,
            'limit_zone_atr': 2.2,
            'default_pullback_atr': 0.55,
        },
        'BTCUSD': {
            'digits': 2,
            'tick_size': 0.01,
            'min_dist_pct': 0.006,
            'max_risk_pct': 0.015,
            'min_entry_gap_pct': 0.0025,
            'min_entry_gap_atr': 0.12,
            'max_entry_gap_pct': 0.014,
            'max_entry_points': 450,
            'min_stop_atr': 1.2,
            'target_rr': 2.0,
            'max_rr': 3.0,
            'max_stop_atr': 3.5,
            'limit_zone_atr': 2.2,
            'default_pullback_atr': 0.55,
        },
        'US30': {
            'digits': 1,
            'tick_size': 0.1,
            'min_dist_pct': 0.004,
            'max_risk_pct': 0.010,
            'min_entry_gap_pct': 0.0015,
            'min_entry_gap_atr': 0.12,
            'max_entry_gap_pct': 0.009,
            'max_entry_points': 350,
            'min_stop_atr': 1.0,
            'target_rr': 2.0,
            'max_rr': 3.0,
            'limit_zone_atr': 2.2,
            'default_pullback_atr': 0.55,
        },
    }
    return {**base, **overrides.get(symbol, {})}

def calculate_structural_score(df, symbol, dxy_context=None, phase_context=None):
    if df.empty or len(df) < 10:
        return {'structural_score': 0, 'score_reason': 'Insufficient data', 'candidate_direction': None, 'market_phase': 'coiling', 'phase_reason': 'Not enough data to assess structure.'}
    micro = calculate_microstructure(df)
    bos, choch = detect_bos_choch(df)
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    sweeps = detect_liquidity_sweeps(df)
    phase_context = phase_context or detect_market_phase(df)
    score = 42
    reasons = []
    if micro.get('price_vs_vwap') == 'ABOVE':
        score += 8
        reasons.append('price holding above VWAP')
    else:
        score += 4
        reasons.append('price trading near VWAP')
    if micro.get('momentum') == 'BULLISH':
        score += 6
        reasons.append('short-term momentum bullish')
    else:
        score += 4
        reasons.append('short-term momentum bearish')
    if micro.get('rvol', 0) > 2.0:
        score += 10
        reasons.append('strong institutional volume')
    elif micro.get('rvol', 0) < 0.5:
        score -= 8
        reasons.append('low volume / exhaustion risk')
    if bos == 'BULLISH_BOS' or choch == 'BULLISH_CHOCH':
        score += 10
        reasons.append('bullish BOS/CHOCH')
    elif bos == 'BEARISH_BOS' or choch == 'BEARISH_CHOCH':
        score += 10
        reasons.append('bearish BOS/CHOCH')
    if order_blocks:
        score += 6
        reasons.append('order block present')
    if fvgs:
        score += 6
        reasons.append('fair value gap present')
    if sweeps:
        score += 6
        reasons.append('liquidity sweep detected')
    phase = phase_context.get('phase')
    if phase == 'continuation':
        score += 8
        reasons.append('continuation structure is present')
    elif phase == 'reversal':
        score += 6
        reasons.append('reversal structure is forming')
    elif phase == 'exhaustion':
        score -= 5
        reasons.append('exhaustion is present and needs caution')
    entry_quality = phase_context.get('entry_quality')
    if entry_quality == 'early':
        score += 5
        reasons.append('entry zone is still early and actionable')
    elif entry_quality == 'late':
        score -= 4
        reasons.append('entry zone is late and may be chasing price')
    if dxy_context and symbol in ['XAUUSD', 'EURUSD', 'BTCUSD']:
        if dxy_context['trend'] == 'BULLISH' and dxy_context['price_vs_vwap'] == 'ABOVE':
            score -= 6
            reasons.append('DXY is suppressing the setup')
        elif dxy_context['trend'] == 'BEARISH' and dxy_context['price_vs_vwap'] == 'BELOW':
            score += 6
            reasons.append('DXY is supporting the setup')
    score = max(0, min(100, int(score)))
    phase_dir = phase_context.get('direction')
    candidate_direction = None
    if phase_dir and score >= 70:
        candidate_direction = phase_dir
    elif score >= 75 and micro.get('momentum') == 'BULLISH':
        candidate_direction = 'BUY'
    elif score >= 75 and micro.get('momentum') == 'BEARISH':
        candidate_direction = 'SELL'
    return {
        'structural_score': score,
        'score_reason': '; '.join(reasons[-4:]),
        'candidate_direction': candidate_direction,
        'market_phase': phase,
        'phase_reason': phase_context.get('reason', 'Structure is being assessed.'),
        'entry_quality': entry_quality,
        'entry_zone': phase_context.get('entry_zone')
    }

def analyze_candle_structure(df):
    if len(df) < 3:
        return []
    analysis = []
    for i in range(max(0, len(df)-10), len(df)):
        candle = df.iloc[i]
        body = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        if total_range == 0:
            continue
        upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        body_ratio = body / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        candle_type = "BULLISH" if candle['Close'] > candle['Open'] else "BEARISH"
        pattern = "NORMAL"
        if body_ratio > 0.7:
            pattern = "STRONG_" + candle_type
        elif body_ratio < 0.3:
            pattern = "DOJI"
        elif upper_wick_ratio > 0.6:
            pattern = "REJECTION_HIGH"
        elif lower_wick_ratio > 0.6:
            pattern = "REJECTION_LOW"
        elif upper_wick_ratio > 0.4 and body_ratio < 0.4:
            pattern = "SHOOTING_STAR" if candle_type == "BEARISH" else "HANGING_MAN"
        elif lower_wick_ratio > 0.4 and body_ratio < 0.4:
            pattern = "HAMMER" if candle_type == "BULLISH" else "INVERTED_HAMMER"
        analysis.append({
            'time': df.index[i],
            'candle_type': candle_type,
            'pattern': pattern,
            'body_ratio': body_ratio,
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'price': candle['Close'],
            'volume': candle['Volume']
        })
    return analysis[-5:]

def build_setup_context(df, swings, current_price, symbol, dxy_context=None):
    micro = calculate_microstructure(df)
    phase_context = detect_market_phase(df, swings=swings)
    continuation = detect_continuation(df)
    reversal = detect_reversal(df)
    exhaustion = detect_exhaustion(df)
    atr = calculate_atr(df)
    if continuation:
        setup_type = 'continuation'
        setup_bias = continuation.get('direction') or ('BUY' if micro.get('momentum') == 'BULLISH' else 'SELL')
    elif reversal:
        setup_type = 'reversal'
        setup_bias = reversal.get('direction') or ('BUY' if micro.get('momentum') == 'BULLISH' else 'SELL')
    elif exhaustion:
        setup_type = 'exhaustion'
        setup_bias = 'WAIT'
    else:
        setup_type = 'coiling'
        setup_bias = 'WAIT'
    entry_quality = phase_context.get('entry_quality', 'unknown')
    timing_state = 'ready' if entry_quality == 'early' else 'watch' if entry_quality == 'acceptable' else 'late'
    if setup_type == 'exhaustion' or entry_quality == 'late':
        timing_state = 'late'
    return {
        'setup_type': setup_type,
        'setup_bias': setup_bias,
        'phase': phase_context.get('phase', 'coiling'),
        'phase_reason': phase_context.get('reason', 'Structure is forming.'),
        'entry_quality': entry_quality,
        'entry_timing': timing_state,
        'atr': atr,
        'micro': micro
    }

def _adx_value(df, period=14):
    try:
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
        if len(df) < period + 3:
            return None
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
        vals = adx.dropna()
        return float(vals.iloc[-1]) if not vals.empty else None
    except Exception:
        return None

def classify_market_regime(df, adx_trend=25, adx_range=18):
    adx = _adx_value(df)
    if adx is None:
        return {"regime": "UNKNOWN", "adx": None, "trend_direction": None, "tradable": True}
    micro = calculate_microstructure(df) or {}
    mom = micro.get("momentum")
    if adx >= adx_trend:
        return {"regime": "TRENDING", "adx": round(adx, 1), "trend_direction": mom, "tradable": True}
    if adx >= adx_range:
        return {"regime": "TRANSITIONAL", "adx": round(adx, 1), "trend_direction": mom, "tradable": True}
    return {"regime": "RANGING", "adx": round(adx, 1), "trend_direction": None, "tradable": False}

def strategy_htf_trend(symbol, all_data):
    data = all_data.get(symbol, {}) or {}
    votes = []
    for key in ("H1", "H4"):
        df = data.get(key)
        if df is None or getattr(df, "empty", True):
            continue
        m = calculate_microstructure(df) or {}
        if m.get("momentum") == "BULLISH" and m.get("price_vs_vwap") == "ABOVE":
            votes.append("BUY")
        elif m.get("momentum") == "BEARISH" and m.get("price_vs_vwap") == "BELOW":
            votes.append("SELL")
    if not votes:
        return None
    if all(v == "BUY" for v in votes):
        return "BUY"
    if all(v == "SELL" for v in votes):
        return "SELL"
    return None

def strategy_zone_reversion(m10, current_price, swings, order_blocks, fvgs):
    try:
        atr = calculate_atr(m10) or (float(current_price) * 0.002)
        near = float(atr) * 0.6
        demand, supply = [], []
        for ob in (order_blocks or []):
            p = float(ob.get("price", 0) or 0)
            if ob.get("type") == "BULLISH_OB" and p <= current_price:
                demand.append(p)
            if ob.get("type") == "BEARISH_OB" and p >= current_price:
                supply.append(p)
        for fvg in (fvgs or []):
            if fvg.get("type") == "BULLISH_FVG" and float(fvg.get("bottom", 0)) <= current_price:
                demand.append(float(fvg.get("bottom")))
            if fvg.get("type") == "BEARISH_FVG" and float(fvg.get("top", 0)) >= current_price:
                supply.append(float(fvg.get("top")))
        for sl in (swings or {}).get("recent_swing_lows", []):
            if float(sl) <= current_price:
                demand.append(float(sl))
        for sh in (swings or {}).get("recent_swing_highs", []):
            if float(sh) >= current_price:
                supply.append(float(sh))
        near_demand = bool(demand) and (current_price - max(demand)) <= near
        near_supply = bool(supply) and (min(supply) - current_price) <= near
        if near_demand and not near_supply:
            return "BUY"
        if near_supply and not near_demand:
            return "SELL"
    except Exception:
        pass
    return None

def strategy_momentum_breakout(m10):
    micro = calculate_microstructure(m10) or {}
    bos, choch = detect_bos_choch(m10)
    if micro.get("rvol", 0) >= 1.5:
        if bos == "BULLISH_BOS" or choch == "BULLISH_CHOCH":
            return "BUY"
        if bos == "BEARISH_BOS" or choch == "BEARISH_CHOCH":
            return "SELL"
    return None

def strategy_liquidity_rejection(m10):
    rev = detect_reversal(m10)
    if rev and rev.get("direction") == "BUY":
        return "BUY"
    if rev and rev.get("direction") == "SELL":
        return "SELL"
    sweeps = detect_liquidity_sweeps(m10)
    if sweeps:
        s = sweeps[-1]
        if s.get("type") == "BULLISH_SWEEP":
            return "BUY"
        if s.get("type") == "BEARISH_SWEEP":
            return "SELL"
    return None

def multi_strategy_vote(symbol, all_data, m10, current_price, swings, order_blocks, fvgs):
    votes = {
        "htf_trend": strategy_htf_trend(symbol, all_data),
        "zone_reversion": strategy_zone_reversion(m10, current_price, swings, order_blocks, fvgs),
        "momentum_breakout": strategy_momentum_breakout(m10),
        "liquidity_rejection": strategy_liquidity_rejection(m10),
    }
    buy = [k for k, v in votes.items() if v == "BUY"]
    sell = [k for k, v in votes.items() if v == "SELL"]
    direction = None
    if len(buy) >= 2 and (len(buy) - len(sell)) >= 2:
        direction = "BUY"
    elif len(sell) >= 2 and (len(sell) - len(buy)) >= 2:
        direction = "SELL"
    return {"direction": direction, "votes": votes, "buy_strategies": buy, "sell_strategies": sell}

def htf_direction_gate(symbol, all_data):
    return strategy_htf_trend(symbol, all_data)

def _desk_position_lock(symbol, proposed, current_price):
    try:
        active = st.session_state.active_signals.get(symbol)
        if not active:
            return None
        prior_dir = active.get("direction")
        prior_entry = float(active.get("entry") or 0)
        ts = active.get("timestamp")
        if prior_dir == proposed or prior_entry <= 0:
            return None
        age_min = (datetime.now() - ts).total_seconds() / 60.0 if ts else 999
        risk_est = abs(prior_entry) * 0.004
        if prior_dir == "BUY" and proposed == "SELL":
            if current_price is not None and current_price <= prior_entry - risk_est:
                return None
            if age_min < 90:
                return "A BUY from {:.2f} is still live and its invalidation has not been taken out. Blocking a premature SELL to prevent whipsaw.".format(prior_entry)
        if prior_dir == "SELL" and proposed == "BUY":
            if current_price is not None and current_price >= prior_entry + risk_est:
                return None
            if age_min < 90:
                return "A SELL from {:.2f} is still live and its invalidation has not been taken out. Blocking a premature BUY to prevent whipsaw.".format(prior_entry)
        return None
    except Exception:
        return None

def build_candidate_levels(symbol, current_price, swings, order_blocks, fvgs, atr, pair_config, vwap=None):
    """Build genuine LIMIT-order candidate plans the AI can choose from or refine. These are
    pullback entries anchored to real structure (order block / FVG / swing / VWAP) inside the
    institutional limit-order band — never a plan sitting at the live price, which previously
    biased the AI toward proposing market-style entries."""
    plans = []
    try:
        current_price = float(current_price)
    except Exception:
        return plans
    for signal in ('BUY', 'SELL'):
        candidate_entry = compute_default_limit_entry(signal, current_price, swings, order_blocks, fvgs, atr, pair_config, vwap=vwap)
        if not candidate_entry:
            continue
        market_plan = build_structural_plan_v2(
            signal=signal,
            entry=candidate_entry,
            current_price=current_price,
            swings=swings,
            order_blocks=order_blocks,
            fvgs=fvgs,
            atr=atr,
            pair_config=pair_config,
            market_df=None
        )
        if market_plan:
            market_plan['plan'] = 'LIMIT'
            market_plan['signal'] = signal
            plans.append(market_plan)
    return plans

def build_exhaustion_target(signal, entry, df, swings, order_blocks, fvgs, atr, pair_config):
    try:
        entry = float(entry)
        atr = float(atr or entry * float(pair_config.get('min_dist_pct', 0.0015)))
    except Exception:
        return None
    candidates = []
    if df is not None and not df.empty:
        look = df.tail(60)
        if signal == 'BUY':
            candidates.append(float(look['High'].max()))
        else:
            candidates.append(float(look['Low'].min()))
    highs = [float(x) for x in (swings or {}).get('recent_swing_highs', []) if x]
    lows = [float(x) for x in (swings or {}).get('recent_swing_lows', []) if x]
    if signal == 'BUY':
        candidates.extend(x for x in highs if x > entry)
        candidates.extend(float(ob.get('price')) for ob in (order_blocks or [])
                          if ob.get('type') == 'BEARISH_OB' and float(ob.get('price', 0) or 0) > entry)
        candidates.extend(float(fvg.get('top')) for fvg in (fvgs or [])
                          if fvg.get('type') == 'BEARISH_FVG' and float(fvg.get('top', 0) or 0) > entry)
        candidates = [x for x in candidates if x > entry + atr * 0.35]
        return min(candidates) if candidates else entry + atr * float(pair_config.get('target_rr', 2.0))
    candidates.extend(x for x in lows if x < entry)
    candidates.extend(float(ob.get('price')) for ob in (order_blocks or [])
                      if ob.get('type') == 'BULLISH_OB' and float(ob.get('price', 0) or 0) < entry)
    candidates.extend(float(fvg.get('bottom')) for fvg in (fvgs or [])
                      if fvg.get('type') == 'BULLISH_FVG' and float(fvg.get('bottom', 0) or 0) < entry)
    candidates = [x for x in candidates if x < entry - atr * 0.35]
    return max(candidates) if candidates else entry - atr * float(pair_config.get('target_rr', 2.0))

def build_structural_plan_v2(signal, entry, current_price, swings, order_blocks, fvgs, atr, pair_config, market_df=None):
    try:
        entry = float(entry)
        current_price = float(current_price)
    except Exception:
        return None
    if signal not in ('BUY', 'SELL'):
        return None
    tick = float(pair_config.get('tick_size', 0.01) or 0.01)
    if atr is None or float(atr) <= 0:
        atr = abs(entry) * float(pair_config.get('min_dist_pct', 0.0015))
        atr = float(atr)
    stop_buffer = max(
        atr * float(pair_config.get('stop_buffer_atr', 0.25)),
        tick * 3.0
    )
    tp_buffer = max(
        atr * float(pair_config.get('tp_buffer_atr', 0.12)),
        tick * 2.0
    )
    min_stop_distance = max(
        abs(entry) * float(pair_config.get('min_dist_pct', 0.0015)),
        atr * float(pair_config.get('min_stop_atr', 1.0))
    )
    max_stop_price = abs(entry) * float(pair_config.get('max_risk_pct', 0.008))
    max_stop_distance = min(max_stop_price, atr * float(pair_config.get('max_stop_atr', 3.0)))
    sl_anchor, tp_anchor = get_structural_anchors(signal, entry, swings, order_blocks, fvgs)
    exhaustion_target = build_exhaustion_target(signal, entry, market_df, swings, order_blocks, fvgs, atr, pair_config)
    target_rr = float(pair_config.get('target_rr', 2.0))
    min_rr = float(pair_config.get('min_rr', 1.3))
    max_rr = float(pair_config.get('max_rr', 3.0))
    if signal == 'BUY':
        if sl_anchor is not None:
            sl = sl_anchor - stop_buffer
        else:
            sl = entry - min_stop_distance
        risk = entry - sl
        if risk < min_stop_distance:
            sl = entry - min_stop_distance
            risk = min_stop_distance
        if risk > max_stop_distance:
            sl = entry - max_stop_distance
            risk = max_stop_distance
        if risk <= 0:
            return None
        structure_target = exhaustion_target - tp_buffer if exhaustion_target is not None else None
        tp = None
        if structure_target is not None and structure_target > entry:
            srr = (structure_target - entry) / risk
            if srr >= min_rr:
                tp = min(structure_target, entry + risk * max_rr)
        if tp is None:
            tp = min(entry + risk * target_rr, entry + risk * max_rr)
        if (tp - entry) / risk < min_rr:
            tp = entry + risk * min_rr
        order_type = infer_order_type(signal, entry, current_price, pair_config, atr)
        rr = round((tp - entry) / risk, 2) if risk > 0 else 0
        return {
            'entry': round_price(entry, pair_config),
            'stop_loss': round_price(sl, pair_config),
            'take_profit': [round_price(tp, pair_config)],
            'exhaustion_target': round_price(tp, pair_config),
            'rr_ratio': rr,
            'risk_band': round_price(risk, pair_config),
            'order_type': order_type,
            'levels_source': 'PYTHON',
        }
    if signal == 'SELL':
        if sl_anchor is not None:
            sl = sl_anchor + stop_buffer
        else:
            sl = entry + min_stop_distance
        risk = sl - entry
        if risk < min_stop_distance:
            sl = entry + min_stop_distance
            risk = min_stop_distance
        if risk > max_stop_distance:
            sl = entry + max_stop_distance
            risk = max_stop_distance
        if risk <= 0:
            return None
        structure_target = exhaustion_target + tp_buffer if exhaustion_target is not None else None
        tp = None
        if structure_target is not None and structure_target < entry:
            srr = (entry - structure_target) / risk
            if srr >= min_rr:
                tp = max(structure_target, entry - risk * max_rr)
        if tp is None:
            tp = max(entry - risk * target_rr, entry - risk * max_rr)
        if (entry - tp) / risk < min_rr:
            tp = entry - risk * min_rr
        order_type = infer_order_type(signal, entry, current_price, pair_config, atr)
        rr = round((entry - tp) / risk, 2) if risk > 0 else 0
        return {
            'entry': round_price(entry, pair_config),
            'stop_loss': round_price(sl, pair_config),
            'take_profit': [round_price(tp, pair_config)],
            'exhaustion_target': round_price(tp, pair_config),
            'rr_ratio': rr,
            'risk_band': round_price(risk, pair_config),
            'order_type': order_type,
            'levels_source': 'PYTHON',
        }
    return None

def get_structural_anchors(signal, entry, swings, order_blocks, fvgs):
    try:
        entry = float(entry)
    except Exception:
        return None, None
    swing_highs = []
    swing_lows = []
    try:
        swing_highs = [float(x) for x in (swings or {}).get('recent_swing_highs', []) if x]
        swing_lows = [float(x) for x in (swings or {}).get('recent_swing_lows', []) if x]
    except Exception:
        pass
    order_blocks = order_blocks or []
    fvgs = fvgs or []
    try:
        if signal == 'BUY':
            sl_candidates = []
            sl_candidates.extend([x for x in swing_lows if x < entry])
            sl_candidates.extend([
                float(ob.get('price'))
                for ob in order_blocks
                if ob.get('type') == 'BULLISH_OB' and float(ob.get('price', 0) or 0) < entry
            ])
            sl_candidates.extend([
                float(fvg.get('bottom'))
                for fvg in fvgs
                if fvg.get('type') == 'BULLISH_FVG' and float(fvg.get('bottom', 0) or 0) < entry
            ])
            tp_candidates = []
            tp_candidates.extend([x for x in swing_highs if x > entry])
            tp_candidates.extend([
                float(ob.get('price'))
                for ob in order_blocks
                if ob.get('type') == 'BEARISH_OB' and float(ob.get('price', 0) or 0) > entry
            ])
            tp_candidates.extend([
                float(fvg.get('top'))
                for fvg in fvgs
                if fvg.get('type') == 'BEARISH_FVG' and float(fvg.get('top', 0) or 0) > entry
            ])
            sl_anchor = max(sl_candidates) if sl_candidates else None
            tp_anchor = min(tp_candidates) if tp_candidates else None
            return sl_anchor, tp_anchor
        if signal == 'SELL':
            sl_candidates = []
            sl_candidates.extend([x for x in swing_highs if x > entry])
            sl_candidates.extend([
                float(ob.get('price'))
                for ob in order_blocks
                if ob.get('type') == 'BEARISH_OB' and float(ob.get('price', 0) or 0) > entry
            ])
            sl_candidates.extend([
                float(fvg.get('top'))
                for fvg in fvgs
                if fvg.get('type') == 'BEARISH_FVG' and float(fvg.get('top', 0) or 0) > entry
            ])
            tp_candidates = []
            tp_candidates.extend([x for x in swing_lows if x < entry])
            tp_candidates.extend([
                float(ob.get('price'))
                for ob in order_blocks
                if ob.get('type') == 'BULLISH_OB' and float(ob.get('price', 0) or 0) < entry
            ])
            tp_candidates.extend([
                float(fvg.get('bottom'))
                for fvg in fvgs
                if fvg.get('type') == 'BULLISH_FVG' and float(fvg.get('bottom', 0) or 0) < entry
            ])
            sl_anchor = min(sl_candidates) if sl_candidates else None
            tp_anchor = max(tp_candidates) if tp_candidates else None
            return sl_anchor, tp_anchor
    except Exception:
        return None, None
    return None, None

def round_price(price, pair_config):
    try:
        if price is None:
            return None
        return round(float(price), int(pair_config.get('digits', 2)))
    except Exception:
        return None

def get_entry_gap_bounds(current_price, atr, pair_config):
    """Return (min_gap, max_gap): the institutional 'limit zone' band an entry must sit in.

    min_gap guarantees a signal can never be a disguised market order — the entry must be a
    meaningful distance from live price. max_gap keeps the limit realistic (still likely to be
    tagged) rather than sitting so far away it may never fill. Both scale with the symbol's ATR
    so the band automatically widens in high-volatility conditions and tightens when the market
    is quiet.
    """
    try:
        current_price = float(current_price)
    except Exception:
        current_price = 0.0
    atr = float(atr) if atr else 0.0
    min_gap = max(
        float(pair_config.get('tick_size', 0.01) or 0.01) * 5.0,
        current_price * float(pair_config.get('min_entry_gap_pct', 0.0009)),
        atr * float(pair_config.get('min_entry_gap_atr', 0.12))
    )
    max_gap = min(
        float(pair_config.get('max_entry_points', 35) or 35),
        current_price * float(pair_config.get('max_entry_gap_pct', 0.0085) or 0.0085)
    )
    if atr:
        max_gap = min(max_gap, atr * float(pair_config.get('limit_zone_atr', 2.2)))
    if max_gap < min_gap:
        max_gap = min_gap * 1.5
    return min_gap, max_gap

def infer_order_type(signal, entry=None, current_price=None, pair_config=None, atr=None):
    # The desk only ever works resting LIMIT orders — never chases price at market and never
    # uses breakout STOP entries. This mirrors "buy limit / sell limit" institutional execution:
    # the AI must wait for price to retrace to its stated level before the order can fill.
    return 'LIMIT'

def compute_default_limit_entry(signal, current_price, swings, order_blocks, fvgs, atr, pair_config, vwap=None):
    """Fallback limit-entry generator used whenever the AI's proposed entry is missing, invalid,
    or outside the reasonable limit band. Instead of snapping to market price (the old
    behaviour), this looks for the nearest genuine structural pullback level — an order block,
    a fair value gap, a recent swing, or VWAP — that sits inside [min_gap, max_gap] from live
    price. If nothing structural qualifies, it anchors a fixed ATR-based pullback distance so
    the entry is still always a real limit, never the current price.
    """
    try:
        current_price = float(current_price)
    except Exception:
        return None
    if signal not in ('BUY', 'SELL'):
        return None
    atr = float(atr) if atr else current_price * float(pair_config.get('min_dist_pct', 0.0015))
    min_gap, max_gap = get_entry_gap_bounds(current_price, atr, pair_config)
    order_blocks = order_blocks or []
    fvgs = fvgs or []
    swings = swings or {}
    candidates = []
    if signal == 'BUY':
        candidates.extend(float(ob.get('price')) for ob in order_blocks
                           if ob.get('type') == 'BULLISH_OB' and ob.get('price') is not None and float(ob.get('price')) < current_price)
        candidates.extend(
            (float(fvg.get('top')) + float(fvg.get('bottom'))) / 2.0
            for fvg in fvgs
            if fvg.get('type') == 'BULLISH_FVG' and fvg.get('top') is not None and fvg.get('bottom') is not None
            and (float(fvg.get('top')) + float(fvg.get('bottom'))) / 2.0 < current_price
        )
        candidates.extend(float(x) for x in swings.get('recent_swing_lows', []) if x is not None and float(x) < current_price)
        if vwap is not None and float(vwap) < current_price:
            candidates.append(float(vwap))
        in_band = [c for c in candidates if min_gap <= (current_price - c) <= max_gap]
        if in_band:
            return max(in_band)  # closest qualifying pullback level to live price
        default_gap = min(max(atr * float(pair_config.get('default_pullback_atr', 0.55)), min_gap), max_gap)
        return current_price - default_gap
    else:
        candidates.extend(float(ob.get('price')) for ob in order_blocks
                           if ob.get('type') == 'BEARISH_OB' and ob.get('price') is not None and float(ob.get('price')) > current_price)
        candidates.extend(
            (float(fvg.get('top')) + float(fvg.get('bottom'))) / 2.0
            for fvg in fvgs
            if fvg.get('type') == 'BEARISH_FVG' and fvg.get('top') is not None and fvg.get('bottom') is not None
            and (float(fvg.get('top')) + float(fvg.get('bottom'))) / 2.0 > current_price
        )
        candidates.extend(float(x) for x in swings.get('recent_swing_highs', []) if x is not None and float(x) > current_price)
        if vwap is not None and float(vwap) > current_price:
            candidates.append(float(vwap))
        in_band = [c for c in candidates if min_gap <= (c - current_price) <= max_gap]
        if in_band:
            return min(in_band)  # closest qualifying pullback level to live price
        default_gap = min(max(atr * float(pair_config.get('default_pullback_atr', 0.55)), min_gap), max_gap)
        return current_price + default_gap

def check_level_math(signal, order_type, entry, sl, tp, current_price, atr, pair_config):
    try:
        entry = float(entry)
        sl = float(sl)
        tp = float(tp)
        current_price = float(current_price)
    except Exception:
        return False, "Missing or non-numeric entry/SL/TP."
    if entry <= 0 or sl <= 0 or tp <= 0 or current_price <= 0:
        return False, "Entry, SL, TP, and current price must be positive."
    min_entry_gap, max_entry_gap = get_entry_gap_bounds(current_price, atr, pair_config)
    gap = abs(entry - current_price)
    if gap > max_entry_gap:
        return False, f"Entry too far from live price. Gap={gap:.6f}, max={max_entry_gap:.6f}."
    if gap < min_entry_gap:
        return False, f"Entry too close to live price to qualify as a genuine limit order. Gap={gap:.6f}, min={min_entry_gap:.6f}."
    order_type = str(order_type or '').upper()
    if order_type != 'LIMIT':
        return False, f"Only LIMIT orders are permitted; got {order_type or 'UNSET'}."
    if signal == 'BUY' and entry >= current_price:
        return False, "BUY LIMIT must be below current price."
    if signal == 'SELL' and entry <= current_price:
        return False, "SELL LIMIT must be above current price."
    min_stop_distance = max(
        abs(entry) * float(pair_config.get('min_dist_pct', 0.0015)),
        float(atr or 0.0) * float(pair_config.get('min_stop_atr', 1.0))
    )
    max_stop_price = abs(entry) * float(pair_config.get('max_risk_pct', 0.008))
    if atr:
        max_stop_distance = min(max_stop_price, float(atr) * float(pair_config.get('max_stop_atr', 3.0)))
    else:
        max_stop_distance = max_stop_price
    if signal == 'BUY':
        if sl >= entry:
            return False, f"BUY SL must be below entry. SL={sl}, entry={entry}."
        if tp <= entry:
            return False, f"BUY TP must be above entry. TP={tp}, entry={entry}."
        risk = entry - sl
        reward = tp - entry
    elif signal == 'SELL':
        if sl <= entry:
            return False, f"SELL SL must be above entry. SL={sl}, entry={entry}."
        if tp >= entry:
            return False, f"SELL TP must be below entry. TP={tp}, entry={entry}."
        risk = sl - entry
        reward = entry - tp
    else:
        return False, "Invalid signal."
    if risk <= 0:
        return False, "Risk distance must be positive."
    if risk < min_stop_distance * 0.95:
        return False, f"Stop too tight. Risk={risk:.6f}, min={min_stop_distance:.6f}."
    if risk > max_stop_distance * 1.05:
        return False, f"Stop too wide. Risk={risk:.6f}, max={max_stop_distance:.6f}."
    rr = reward / risk if risk > 0 else 0
    min_rr = float(pair_config.get('min_rr', pair_config.get('target_rr', 1.3)))
    max_rr = float(pair_config.get('max_rr', 3.0))
    if rr + 0.01 < min_rr:
        return False, f"RR too low. RR={rr:.2f}, min={min_rr:.2f}."
    if rr > max_rr * 1.05:
        return False, f"RR too high / TP too far from entry. RR={rr:.2f}, max={max_rr:.2f}."
    return True, "Valid"

def build_atr_fallback_plan(signal, entry, pair_config, atr):
    """Deterministic, always-valid SL/TP builder used only when no structural anchor (order
    block, FVG, swing) is usable. Sizes risk purely from ATR/min-distance rules so a signal
    is always produced — this desk never returns WAIT for lack of a plan."""
    entry = float(entry)
    atr = float(atr) if atr else abs(entry) * float(pair_config.get('min_dist_pct', 0.0015))
    min_stop_distance = max(
        abs(entry) * float(pair_config.get('min_dist_pct', 0.0015)),
        atr * float(pair_config.get('min_stop_atr', 1.0))
    )
    max_stop_distance = min(
        abs(entry) * float(pair_config.get('max_risk_pct', 0.008)),
        atr * float(pair_config.get('max_stop_atr', 3.0))
    )
    if max_stop_distance < min_stop_distance:
        max_stop_distance = min_stop_distance * 1.5
    risk = min(max(min_stop_distance * 1.15, min_stop_distance), max_stop_distance)
    target_rr = max(float(pair_config.get('target_rr', 2.0)), float(pair_config.get('min_rr', 1.3)))
    target_rr = min(target_rr, float(pair_config.get('max_rr', 3.0)))
    reward = risk * target_rr
    if signal == 'BUY':
        sl = entry - risk
        tp = entry + reward
    else:
        sl = entry + risk
        tp = entry - reward
    rr = round(reward / risk, 2) if risk > 0 else 0
    return {
        'entry': round_price(entry, pair_config),
        'stop_loss': round_price(sl, pair_config),
        'take_profit': [round_price(tp, pair_config)],
        'exhaustion_target': round_price(tp, pair_config),
        'rr_ratio': rr,
        'risk_band': round_price(risk, pair_config),
        'order_type': 'LIMIT',
        'levels_source': 'PYTHON_ATR_FALLBACK',
    }

def finalize_trade_plan(analysis, symbol, current_price, swings, order_blocks, fvgs, atr, pair_config, market_df=None):
    if not isinstance(analysis, dict):
        return analysis
    signal = analysis.get('signal')
    if signal not in ('BUY', 'SELL'):
        return analysis
    try:
        current_price = float(current_price)
    except Exception:
        return analysis
    entry = analysis.get('entry')
    try:
        entry = float(entry)
    except Exception:
        entry = None
    min_entry_gap, max_entry_gap = get_entry_gap_bounds(current_price, atr, pair_config)
    gap = abs(entry - current_price) if entry else None
    needs_recompute = entry is None or entry <= 0 or gap is None or gap > max_entry_gap or gap < min_entry_gap
    # Directional sanity: a BUY entry must sit below live price and a SELL entry above it,
    # otherwise the AI proposed a market/stop-style level, which is not permitted here.
    if entry is not None and entry > 0:
        if signal == 'BUY' and entry >= current_price:
            needs_recompute = True
        if signal == 'SELL' and entry <= current_price:
            needs_recompute = True
    if needs_recompute:
        vwap = None
        try:
            if market_df is not None and not market_df.empty:
                micro = calculate_microstructure(market_df)
                vwap = micro.get('vwap')
        except Exception:
            vwap = None
        computed_entry = compute_default_limit_entry(signal, current_price, swings, order_blocks, fvgs, atr, pair_config, vwap=vwap)
        entry = computed_entry if computed_entry else (
            current_price - max(min_entry_gap, (atr or 0) * float(pair_config.get('default_pullback_atr', 0.55)))
            if signal == 'BUY' else
            current_price + max(min_entry_gap, (atr or 0) * float(pair_config.get('default_pullback_atr', 0.55)))
        )
        analysis['order_type'] = 'LIMIT'
        analysis['levels_source'] = 'PYTHON'
        add_python_validation_note(analysis, "Entry was re-anchored to the nearest qualifying structural pullback level (order block / FVG / swing / VWAP) because the proposed entry was missing, too close to a market order, or outside the reasonable limit-order band.")
    analysis['order_type'] = 'LIMIT'
    canonical_plan = build_structural_plan_v2(
        signal=signal,
        entry=entry,
        current_price=current_price,
        swings=swings,
        order_blocks=order_blocks,
        fvgs=fvgs,
        atr=atr,
        pair_config=pair_config,
        market_df=market_df
    )
    if canonical_plan:
        analysis['stop_loss'] = canonical_plan['stop_loss']
        analysis['take_profit'] = canonical_plan['take_profit']
        analysis['exhaustion_target'] = canonical_plan['exhaustion_target']
        analysis['levels_source'] = 'PYTHON_EXHAUSTION_TARGET'
        add_python_validation_note(analysis, "TP is the nearest executable exhaustion/liquidity target; SL is beyond the structural invalidation with ATR buffering.")
    analysis['entry'] = round_price(entry, pair_config)
    sl = analysis.get('stop_loss')
    tp_list = analysis.get('take_profit', [])
    tp = tp_list[0] if tp_list else None
    ok, reason = check_level_math(
        signal=signal,
        order_type=analysis.get('order_type'),
        entry=entry,
        sl=sl,
        tp=tp,
        current_price=current_price,
        atr=atr,
        pair_config=pair_config
    )
    if not ok:
        plan = build_structural_plan_v2(
            signal=signal,
            entry=entry,
            current_price=current_price,
            swings=swings,
            order_blocks=order_blocks,
            fvgs=fvgs,
            atr=atr,
            pair_config=pair_config,
            market_df=market_df
        )
        if not plan:
            # Last-resort deterministic ATR-based plan. No WAIT signals are produced by this
            # desk — if no structural anchor is usable, a clean risk-defined limit trade is
            # built directly from volatility so the analysis always resolves to an executable
            # BUY/SELL limit.
            plan = build_atr_fallback_plan(signal, entry, pair_config, atr)
            add_python_validation_note(analysis, f"No structural SL/TP anchor was usable ({reason}); levels were built directly from ATR risk sizing instead.")
        analysis.update(plan)
    analysis['entry'] = round_price(analysis.get('entry'), pair_config)
    analysis['stop_loss'] = round_price(analysis.get('stop_loss'), pair_config)
    if analysis.get('take_profit'):
        analysis['take_profit'] = [
            round_price(x, pair_config)
            for x in analysis.get('take_profit')
            if x is not None
        ]
    final_entry = analysis.get('entry')
    final_sl = analysis.get('stop_loss')
    final_tp = (analysis.get('take_profit') or [None])[0]
    ok_final, reason_final = check_level_math(
        signal=analysis.get('signal'),
        order_type=analysis.get('order_type'),
        entry=final_entry,
        sl=final_sl,
        tp=final_tp,
        current_price=current_price,
        atr=atr,
        pair_config=pair_config
    )
    if not ok_final:
        # Deterministic safety net: rebuild from ATR risk sizing directly off the (already
        # band-validated) entry rather than surrendering to a WAIT signal.
        plan = build_atr_fallback_plan(analysis.get('signal'), final_entry, pair_config, atr)
        analysis.update(plan)
        analysis['entry'] = round_price(analysis.get('entry'), pair_config)
        analysis['stop_loss'] = round_price(analysis.get('stop_loss'), pair_config)
        analysis['take_profit'] = [round_price(x, pair_config) for x in analysis.get('take_profit', []) if x is not None]
        add_python_validation_note(analysis, f"Final level validation rebuilt the plan from ATR risk sizing (previous levels failed: {reason_final}).")
    entry_f = float(analysis['entry'])
    sl_f = float(analysis['stop_loss'])
    tp_f = float(analysis['take_profit'][0])
    if analysis['signal'] == 'BUY':
        risk = entry_f - sl_f
        reward = tp_f - entry_f
    else:
        risk = sl_f - entry_f
        reward = entry_f - tp_f
    analysis['rr_ratio'] = round(reward / risk, 2) if risk > 0 else 0
    analysis['exhaustion_target'] = analysis.get('exhaustion_target', analysis.get('take_profit', [None])[0])
    analysis['risk_band'] = round_price(risk, pair_config)
    analysis.setdefault('levels_source', 'HYBRID')
    return analysis

def normalize_ai_signal(signal):
    if not isinstance(signal, str):
        return signal
    normalized = signal.strip().upper()
    if normalized in {'BULLISH', 'LONG', 'BUY'}:
        return 'BUY'
    if normalized in {'BEARISH', 'SHORT', 'SELL'}:
        return 'SELL'
    if normalized in {'WAIT', 'NO_TRADE', 'NONE'}:
        return 'WAIT'
    return signal

def normalize_analysis_signals(analysis):
    if not isinstance(analysis, dict):
        return analysis
    if 'signal' in analysis:
        analysis['signal'] = normalize_ai_signal(analysis['signal'])
    if 'candidate_direction' in analysis:
        analysis['candidate_direction'] = normalize_ai_signal(analysis['candidate_direction'])
    return analysis

def add_python_validation_note(analysis, note):
    if not note:
        return analysis
    notes = analysis.setdefault('python_validation_notes', [])
    if note not in notes:
        notes.append(note)
    return analysis

def validate_ai_logic(analysis):
    signal = analysis.get('signal')
    reasoning = (analysis.get('reasoning') or '').lower()
    micro_read = (analysis.get('microstructure_read') or '').lower()
    buy_bad = ['invalidates the buy', 'invalidates the long', 'invalid buy', 'invalid long',
               'buy setup is invalid', 'long setup is invalid', 'do not buy', "don't buy",
               'avoid buying', 'no buy setup', 'buy is invalidated']
    sell_bad = ['invalidates the sell', 'invalidates the short', 'invalid sell', 'invalid short',
                'sell setup is invalid', 'short setup is invalid', 'do not sell', "don't sell",
                'avoid selling', 'no sell setup', 'sell is invalidated']
    if signal == 'BUY':
        if any(p in reasoning for p in buy_bad):
            return False, 'The reasoning explicitly invalidates the BUY setup.'
        if any(t in micro_read for t in ['exhaustion', 'trap']):
            if not any(t in reasoning for t in ['pullback', 'retest', 'reclaim', 'confirmation', 'liquidity', 'sweep', 'zone', 'order block', 'fvg']):
                return False, 'Microstructure suggests a trap or exhaustion and the reasoning lacks a clear continuation or invalidation framework.'
    elif signal == 'SELL':
        if any(p in reasoning for p in sell_bad):
            return False, 'The reasoning explicitly invalidates the SELL setup.'
        if any(t in micro_read for t in ['exhaustion', 'trap']):
            if not any(t in reasoning for t in ['pullback', 'retest', 'reclaim', 'confirmation', 'liquidity', 'sweep', 'zone', 'order block', 'fvg']):
                return False, 'Microstructure suggests a trap or exhaustion and the reasoning lacks a clear continuation or invalidation framework.'
    return True, 'Valid'

def validate_signal_math(analysis, pair_config=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return False, 'Invalid signal direction.'
    pair_config = pair_config or {}
    entry = analysis.get('entry')
    sl = analysis.get('stop_loss')
    tp_list = analysis.get('take_profit', [])
    tp1 = tp_list[0] if tp_list else None
    if entry is None or sl is None or tp1 is None:
        return False, 'Missing entry, SL, or TP values after finalization.'
    try:
        entry = float(entry)
        sl = float(sl)
        tp1 = float(tp1)
    except Exception:
        return False, 'Entry, SL, and TP must be numeric.'
    if signal == 'BUY':
        if tp1 <= entry:
            return False, f'Invalid Math: For BUY, TP1 ({tp1}) MUST be > Entry ({entry}).'
        if sl >= entry:
            return False, f'Invalid Math: For BUY, SL ({sl}) MUST be < Entry ({entry}).'
    elif signal == 'SELL':
        if tp1 >= entry:
            return False, f'Invalid Math: For SELL, TP1 ({tp1}) MUST be < Entry ({entry}).'
        if sl <= entry:
            return False, f'Invalid Math: For SELL, SL ({sl}) MUST be > Entry ({entry}).'
    risk = abs(entry - sl)
    reward = abs(entry - tp1)
    if risk <= 0:
        return False, 'Risk distance must be positive.'
    min_rr = float(pair_config.get('min_rr', pair_config.get('target_rr', 1.3)))
    if (reward / risk) + 0.01 < min_rr:
        return False, f'Invalid Math: R:R is too low ({(reward / risk):.2f}). Minimum required is 1:{min_rr:.2f}.'
    return True, 'Valid'

def apply_dxy_guardrails(analysis, symbol, dxy_context):
    if symbol not in ['XAUUSD', 'EURUSD', 'BTCUSD'] or not dxy_context:
        return analysis
    trend = dxy_context.get('trend')
    price_vs_vwap = dxy_context.get('price_vs_vwap')
    reasoning = (analysis.get('reasoning') or '').lower()
    signal = analysis.get('signal')
    if trend == 'BULLISH' and price_vs_vwap == 'ABOVE':
        expected_bias = 'SELL'
    elif trend == 'BEARISH' and price_vs_vwap == 'BELOW':
        expected_bias = 'BUY'
    else:
        expected_bias = None
    if expected_bias is None:
        analysis['dxy_correlation'] = 'NEUTRAL'
        return analysis
    if signal == expected_bias:
        analysis['dxy_correlation'] = 'CONFIRMING'
        if 'dxy' not in reasoning and 'dollar' not in reasoning:
            add_python_validation_note(analysis, f"DXY confirms the bias: the dollar index is {trend.lower()} and price is {price_vs_vwap.lower()} VWAP.")
        return analysis
    analysis['dxy_correlation'] = 'CONTRADICTING'
    if analysis.get('confidence') == 'HIGH':
        analysis['confidence'] = 'MEDIUM'
        analysis['confluence_score'] = max(0, analysis.get('confluence_score', 0) - 4)
    if 'dxy' not in reasoning and 'dollar' not in reasoning:
        add_python_validation_note(analysis, "The setup is contrarian versus the DXY bias and therefore needs stronger macro confirmation.")
    return analysis

def apply_htf_trend_guard(analysis, symbol, htf_context):
    if analysis.get('signal') not in ['BUY', 'SELL'] or not isinstance(htf_context, dict):
        return analysis
    trend = str(htf_context.get('trend') or '').upper()
    bias = str(htf_context.get('bias') or '').upper()
    if trend not in {'BULLISH', 'BEARISH'} and bias not in {'BULLISH', 'BEARISH'}:
        return analysis
    expected_bias = trend or bias
    signal = analysis.get('signal')
    if (signal == 'BUY' and expected_bias == 'BEARISH') or (signal == 'SELL' and expected_bias == 'BULLISH'):
        analysis['confidence'] = 'MEDIUM' if analysis.get('confidence') == 'HIGH' else analysis.get('confidence')
        analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 78)
        add_python_validation_note(analysis, f"Higher-timeframe trend is {expected_bias.lower()}, so this countertrend idea has reduced conviction and needs strong structural confirmation.")
    return analysis

def cross_check_ai_evidence(analysis):
    ev = analysis.get('directional_evidence')
    if not isinstance(ev, dict):
        return analysis
    bull = ev.get('bullish') or []
    bear = ev.get('bearish') or []
    if not isinstance(bull, list) or not isinstance(bear, list):
        return analysis
    signal = analysis.get('signal')
    if signal == 'BUY' and len(bear) - len(bull) >= 2:
        analysis['confidence'] = 'MEDIUM' if analysis.get('confidence') == 'HIGH' else analysis.get('confidence')
        analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 74)
        add_python_validation_note(analysis, "The AI evidence ledger was bearish-heavy, so bullish conviction was reduced.")
    elif signal == 'SELL' and len(bull) - len(bear) >= 2:
        analysis['confidence'] = 'MEDIUM' if analysis.get('confidence') == 'HIGH' else analysis.get('confidence')
        analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 74)
        add_python_validation_note(analysis, "The AI evidence ledger was bullish-heavy, so bearish conviction was reduced.")
    return analysis

def apply_direction_correction_guard(analysis, confluence, symbol):
    signal = analysis.get('signal')
    if signal not in ('BUY', 'SELL') or not confluence:
        return analysis
    direction = confluence.get('direction')
    bull = confluence.get('bull_count', 0)
    bear = confluence.get('bear_count', 0)
    lead = abs(bull - bear)
    if direction and direction != signal and max(bull, bear) >= 3 and lead >= 2:
        analysis['signal'] = direction
        analysis['confidence'] = 'MEDIUM'
        analysis['confluence_score'] = max(MINIMUM_CONFLUENCE_SCORE, min(analysis.get('confluence_score', 0), 82))
        ev = confluence.get('bullish_evidence') if direction == 'BUY' else confluence.get('bearish_evidence')
        add_python_validation_note(analysis, f"Direction corrected to {direction} by the structural evidence audit: {'; '.join(ev[:4])}.")
        analysis['rejection_reason'] = None
    elif direction == signal:
        ev = confluence.get('bullish_evidence') if signal == 'BUY' else confluence.get('bearish_evidence')
        analysis['confluence_score'] = min(100, analysis.get('confluence_score', 0) + 2)
        add_python_validation_note(analysis, f"Directional evidence audit confirms the {signal} side: {'; '.join(ev[:4])}.")
    return analysis

def apply_conservative_signal_filter(analysis, structural_context, candles, dxy_context, current_price, swings, symbol, pair_config=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return analysis
    structural_score = (structural_context or {}).get('structural_score', 0)
    reasoning = (analysis.get('reasoning') or '').lower()
    recent_patterns = [c.get('pattern') for c in (candles or []) if c.get('pattern')]
    strong_recent = any(pattern in {'STRONG_BULLISH', 'STRONG_BEARISH', 'HAMMER', 'INVERTED_HAMMER', 'REJECTION_LOW', 'REJECTION_HIGH'} for pattern in recent_patterns)
    structural_markers = any(term in reasoning for term in ['order block', 'fvg', 'liquidity', 'retest', 'reclaim', 'zone', 'bos', 'choch', 'sweep'])
    if current_price is not None and swings:
        valid_swing_lows = [l for l in swings.get('recent_swing_lows', []) if l < current_price]
        valid_swing_highs = [h for h in swings.get('recent_swing_highs', []) if h > current_price]
    else:
        valid_swing_lows = []
        valid_swing_highs = []
    has_clear_anchor = bool((signal == 'BUY' and valid_swing_lows) or (signal == 'SELL' and valid_swing_highs))
    has_structure_support = structural_score >= 60 or strong_recent or structural_markers or (has_clear_anchor and structural_score >= 55)
    if not has_structure_support:
        analysis['confidence'] = 'LOW'
        analysis['confluence_score'] = max(analysis.get('confluence_score', 0), MINIMUM_CONFLUENCE_SCORE)
        analysis['rejection_reason'] = 'Structure is still forming, so the setup remains an early candidate rather than a hard no-trade.'
    return analysis

def build_display_reason(analysis, symbol, current_price=None, phase_context=None, structural_context=None, dxy_context=None):
    reasoning = (analysis.get('reasoning') or '').strip()
    rejection = (analysis.get('rejection_reason') or '').strip()
    setup_context = analysis.get('setup_context') or {}
    phase = (phase_context or {}).get('phase') or setup_context.get('phase') or analysis.get('market_state') or 'unknown'
    setup_type = setup_context.get('setup_type') or analysis.get('market_state') or 'unknown'
    timing = setup_context.get('entry_timing') or (phase_context or {}).get('entry_quality') or 'unknown'
    signal = analysis.get('signal')
    score = analysis.get('confluence_score')
    confidence = analysis.get('confidence')
    dxy_status = analysis.get('dxy_correlation') or ('CONFIRMING' if dxy_context else '')
    entry = analysis.get('entry')
    current = current_price if current_price is not None else entry
    parts = []
    if reasoning:
        parts.append(reasoning)
    if analysis.get('confluence_breakdown'):
        parts.append(f"Confluence breakdown: {analysis.get('confluence_breakdown')}")
    else:
        breakdown_parts = []
        dxy = analysis.get('dxy_correlation') or 'N/A'
        micro = analysis.get('microstructure_read') or ''
        rsi_ctx = analysis.get('rsi_context') or ''
        struct_score = analysis.get('structural_score') if structural_context is None else structural_context.get('structural_score')
        live_price = analysis.get('live_price') or current_price
        if dxy:
            breakdown_parts.append(f"DXY: {dxy}")
        if micro:
            breakdown_parts.append(f"VWAP/RVOL: {micro}")
        if rsi_ctx:
            breakdown_parts.append(f"RSI: {rsi_ctx}")
        if struct_score is not None:
            breakdown_parts.append(f"Structure score: {struct_score}/100")
        if live_price is not None:
            try:
                breakdown_parts.append(f"Live price: {float(live_price):.2f}")
            except Exception:
                breakdown_parts.append(f"Live price: {live_price}")
        if breakdown_parts:
            parts.append('Confluence breakdown: ' + ' | '.join(breakdown_parts))
        else:
            parts.append(f"{symbol} is being assessed from the current market and execution context.")
    if phase and phase != 'unknown':
        parts.append(f"Market state is {phase}.")
    if setup_type and setup_type != 'unknown':
        parts.append(f"Setup type is {setup_type}.")
    if timing and timing != 'unknown':
        parts.append(f"Entry timing is {timing}.")
    if current is not None and entry is not None and current not in [0, None]:
        gap_pct = abs(entry - current) / current * 100 if current else 0.0
        parts.append(f"The proposed entry is about {gap_pct:.2f}% from the live price.")
    if dxy_status:
        parts.append(f"DXY correlation is {dxy_status.lower()}.")
    if score is not None:
        parts.append(f"Confluence score is {score}/100 with {confidence.lower() if confidence else 'unknown'} confidence.")
    if rejection and signal == 'WAIT':
        parts.append(f"Decision: {rejection}")
    elif rejection:
        parts.append(f"Decision: {rejection}")
    return ' '.join(parts)

def build_validation_detail(analysis, swings, current_price, symbol, pair_config=None, structural_context=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return 'No trade signal was produced because the setup did not meet the required structural or risk criteria.'
    pair_config = pair_config or {}
    min_dist = current_price * pair_config.get('min_dist_pct', 0.001)
    target_rr = pair_config.get('min_rr', pair_config.get('target_rr', 1.3))
    entry = analysis.get('entry', current_price)
    sl = analysis.get('stop_loss')
    tp_list = analysis.get('take_profit', [])
    tp1 = tp_list[0] if tp_list else None
    reasons = []
    if signal == 'BUY':
        if sl is None or sl >= entry:
            reasons.append(f'SL is not below entry ({sl} >= {entry}).')
        else:
            risk = entry - sl
            if risk + 1e-6 < min_dist:
                reasons.append(f'SL is too close to entry; risk is {risk:.4f}, below the minimum {min_dist:.4f} for {symbol}.')
        if tp1 is None or tp1 <= entry:
            reasons.append(f'TP is not above entry ({tp1} <= {entry}).')
        else:
            reward = tp1 - entry
            risk = entry - sl if sl is not None else 0
            if risk > 0 and (reward / risk) + 0.01 < target_rr:
                reasons.append(f'The proposed risk/reward is too low ({reward / risk:.2f} vs required {target_rr:.2f}).')
    else:
        if sl is None or sl <= entry:
            reasons.append(f'SL is not above entry ({sl} <= {entry}).')
        else:
            risk = sl - entry
            if risk + 1e-6 < min_dist:
                reasons.append(f'SL is too close to entry; risk is {risk:.4f}, below the minimum {min_dist:.4f} for {symbol}.')
        if tp1 is None or tp1 >= entry:
            reasons.append(f'TP is not below entry ({tp1} >= {entry}).')
        else:
            reward = entry - tp1
            risk = sl - entry if sl is not None else 0
            if risk > 0 and (reward / risk) + 0.01 < target_rr:
                reasons.append(f'The proposed risk/reward is too low ({reward / risk:.2f} vs required {target_rr:.2f}).')
    if structural_context and structural_context.get('structural_score', 0) < 70:
        reasons.append('The structural score is too weak for a high-quality setup.')
    return ' '.join(reasons) if reasons else 'The setup did not meet the structural and risk requirements for execution.'

def get_live_market_snapshot(symbol, yf_symbol, fallback_df=None):
    fallback_price = None
    if fallback_df is not None and not fallback_df.empty:
        try:
            fallback_price = float(pd.to_numeric(fallback_df['Close'], errors='coerce').dropna().iloc[-1])
        except Exception:
            fallback_price = None
    bitfinex_symbols = {'XAUUSD': 'tXAUT:USD'}
    bitfinex_symbol = bitfinex_symbols.get(symbol)
    if bitfinex_symbol:
        try:
            response = requests.get(
                f'https://api-pub.bitfinex.com/v2/ticker/{bitfinex_symbol}',
                timeout=10
            )
            response.raise_for_status()
            payload = response.json()
            price = float(payload[6])
            if price > 0:
                return {
                    'symbol': symbol,
                    'price': price,
                    'source': 'bitfinex_spot_quote',
                    'quote_time': datetime.now(timezone.utc).isoformat()
                }
        except Exception as exc:
            print(f"⚠️ Bitfinex spot quote failed for {symbol}: {exc}")
    try:
        response = requests.get(
            f'https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}',
            params={'range': '1d', 'interval': '1m', 'includePrePost': 'true'},
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        response.raise_for_status()
        chart = response.json().get('chart', {})
        result = (chart.get('result') or [{}])[0]
        meta = result.get('meta') or {}
        quote_candidates = [
            meta.get('regularMarketPrice'),
            meta.get('postMarketPrice'),
            meta.get('preMarketPrice')
        ]
        for candidate in quote_candidates:
            try:
                price = float(candidate)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return {
                    'symbol': symbol,
                    'price': price,
                    'source': 'yahoo_chart_quote',
                    'quote_time': meta.get('regularMarketTime')
                }
    except Exception as exc:
        print(f"⚠️ Live quote fetch failed for {symbol}: {exc}")
    return {'symbol': symbol, 'price': fallback_price, 'source': 'candle_close_fallback'}

def refresh_symbol_data_if_stale(symbol, yf_symbol, market_snapshot, data):
    m10 = data.get('M10') if isinstance(data, dict) else None
    live_price = market_snapshot.get('price') if isinstance(market_snapshot, dict) else None
    if m10 is None or m10.empty or live_price is None:
        return data
    try:
        candle_price = float(pd.to_numeric(m10['Close'], errors='coerce').dropna().iloc[-1])
        live_price = float(live_price)
        discrepancy = abs(live_price - candle_price) / live_price
        atr = calculate_atr(m10) or live_price * 0.002
        materially_stale = discrepancy > 0.005 or abs(live_price - candle_price) > atr * 2.5
        if not materially_stale:
            return data
        fetch_ohlcv.clear()
        refreshed = fetch_symbol_data(symbol, yf_symbol)
        refreshed_m10 = refreshed.get('M10') if isinstance(refreshed, dict) else None
        if refreshed_m10 is not None and not refreshed_m10.empty:
            print(f"🔄 Refreshed stale {symbol} candles: close={candle_price:.5f}, live={live_price:.5f}")
            return refreshed
    except Exception as exc:
        print(f"⚠️ Stale-data refresh failed for {symbol}: {exc}")
    return data

def update_market_state(new_state):
    if not new_state:
        return
    previous = st.session_state.get('market_state', 'coiling')
    if previous != new_state:
        st.session_state.state_history.append({'state': new_state, 'time': datetime.now().strftime('%H:%M:%S')})
        if len(st.session_state.state_history) > 20:
            st.session_state.state_history = st.session_state.state_history[-20:]
        st.session_state.market_state = new_state

def estimate_tokens_for_text(text):
    return max(1, int(len(text) / 4))

def estimate_analysis_tokens(system_prompt, user_content):
    prompt_text = system_prompt + ' ' + ' '.join([item.get('text', '') for item in user_content if isinstance(item, dict)])
    return estimate_tokens_for_text(prompt_text) + GROQ_ESTIMATED_RESPONSE_TOKENS

def reserve_groq_tokens(estimated_tokens):
    now = datetime.now()
    window_start = st.session_state.groq_token_window_start
    if (now - window_start).total_seconds() >= 60:
        st.session_state.groq_token_window_start = now
        st.session_state.groq_tokens_used = 0
    if estimated_tokens is None:
        estimated_tokens = 0
    if st.session_state.groq_tokens_used + estimated_tokens > GROQ_TOKEN_LIMIT_PER_MINUTE:
        next_reset = st.session_state.groq_token_window_start + timedelta(minutes=1)
        st.session_state.groq_rate_limit_until = next_reset
        st.session_state.groq_rate_limit_reason = f"Token budget exceeded: {st.session_state.groq_tokens_used}/{GROQ_TOKEN_LIMIT_PER_MINUTE} used. Needs {estimated_tokens} more tokens and resets at {next_reset.strftime('%H:%M:%S')}."
        return False
    st.session_state.groq_rate_limit_reason = ''
    # Don't add estimated tokens here, we will add ACTUAL tokens after the API call succeeds
    return True

def is_groq_rate_limited():
    retry_until = st.session_state.get('groq_rate_limit_until')
    return retry_until is not None and datetime.now() < retry_until

def get_groq_models(api_key):
    """Diagnostic helper only. This used to also RE-ORDER/FILTER the models actually called,
    which is what let the primary model silently disappear: if Groq's /models discovery
    endpoint didn't happen to list 'openai/gpt-oss-120b' for this account at that moment
    (propagation lag, transient discovery error, preview-model visibility, etc.), the model
    was quietly dropped from the list before it was ever tried. call_groq() no longer uses
    this to decide which models to call — GROQ_MODELS is always tried in its literal order.
    This function is kept only so the Settings tab can show what Groq currently reports as
    available, for debugging."""
    try:
        response = requests.get(
            f"{GROQ_API_URL.rsplit('/chat/completions', 1)[0]}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20
        )
        if response.status_code != 200:
            return {'ok': False, 'reason': f"HTTP {response.status_code}", 'available': []}
        payload = response.json()
        available = [item.get('id') for item in payload.get('data', []) if item.get('id')]
        return {'ok': True, 'available': available}
    except Exception as exc:
        return {'ok': False, 'reason': str(exc), 'available': []}

def is_model_cooling_down(model):
    cooldowns = st.session_state.setdefault('groq_model_cooldowns', {})
    until = cooldowns.get(model)
    return until is not None and datetime.now() < until

def set_model_cooldown(model, seconds):
    cooldowns = st.session_state.setdefault('groq_model_cooldowns', {})
    cooldowns[model] = datetime.now() + timedelta(seconds=max(1, int(seconds)))

# HTTP statuses that mean "this model id is genuinely unusable right now" (deprecated,
# decommissioned, not entitled, bad request shape) — retrying the SAME model won't help, so we
# move straight to the next model in the list without burning retries on it.
_GROQ_NON_RETRYABLE_STATUSES = {400, 401, 403, 404, 422}

def call_groq(system_prompt, user_content, max_tokens=None, retry_count=0, estimated_tokens=None, image_b64=None, image_mime_type='image/png'):
    api_key = get_secret("GROQ_API_KEY", "").strip()
    if not api_key:
        print("❌ GROQ_API_KEY is missing from st.secrets!")
        return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                "rejection_reason": "Missing Groq API Key.",
                "model_used": "Groq unavailable", "estimated_tokens": 0,
                "api_status": "MISSING_KEY"}

    if estimated_tokens is None:
        estimated_tokens = estimate_analysis_tokens(system_prompt, user_content)

    parts = [{"type": "text", "text": system_prompt}]
    for item in user_content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif isinstance(item, str):
            parts.append({"type": "text", "text": item})

    # Add image if provided (but compress to save tokens)
    if image_b64:
        # Only add image if it's under 500KB base64 (roughly 375KB original)
        if len(image_b64) < 500000:
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{image_b64}"
                }
            })
            print(f"📸 Image attached ({len(image_b64)} chars base64)")
        else:
            print(f"⚠️ Image too large ({len(image_b64)} chars), skipping to save tokens")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    model_errors = []
    attempts_log = []
    # GROQ_MODELS is tried in this exact order, every time. No discovery-based filtering.
    for model in GROQ_MODELS:
        model_cfg = GROQ_MODEL_CONFIG.get(model, GROQ_DEFAULT_MODEL_CONFIG)
        model_max_tokens = int(max_tokens) if max_tokens else int(model_cfg['max_completion_tokens'])

        if is_model_cooling_down(model):
            reason = f"{model}: still cooling down from a recent rate limit"
            model_errors.append(reason)
            attempts_log.append({'model': model, 'status': 'SKIPPED_COOLDOWN'})
            continue

        succeeded = False
        for attempt in range(GROQ_MAX_RETRIES_PER_MODEL + 1):
            try:
                time_since_last = (datetime.now() - st.session_state.last_groq_request_time).total_seconds() if st.session_state.last_groq_request_time else None
                if time_since_last is not None and time_since_last < GROQ_MIN_REQUEST_INTERVAL:
                    time.sleep(max(0.0, GROQ_MIN_REQUEST_INTERVAL - time_since_last))

                if not reserve_groq_tokens(estimated_tokens):
                    attempts_log.append({'model': model, 'status': 'TOKEN_BUDGET'})
                    return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
                            "rejection_reason": "RATE_LIMIT", "model_used": model,
                            "estimated_tokens": estimated_tokens, "api_status": "TOKEN_BUDGET",
                            "model_attempts": attempts_log}

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": parts}],
                    "temperature": 0.2,
                    "max_completion_tokens": model_max_tokens,
                    "response_format": {"type": "json_object"}
                }
                if model_cfg.get('supports_reasoning_effort') and model_cfg.get('reasoning_effort'):
                    payload["reasoning_effort"] = model_cfg['reasoning_effort']
                    payload["include_reasoning"] = False

                print(f"🚀 Calling {model} (attempt {attempt + 1}/{GROQ_MAX_RETRIES_PER_MODEL + 1}, max_completion_tokens={model_max_tokens})...")

                res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=GROQ_REQUEST_TIMEOUT)
                st.session_state.last_groq_request_time = datetime.now()

                if res.status_code == 429:
                    retry_after = int(res.headers.get('Retry-After', '20')) if res.headers.get('Retry-After') else 20
                    error_text = res.text[:500]
                    print(f"⏳ 429 on {model}. Cooling down {retry_after}s and trying the next model.")
                    model_errors.append(f"{model}: HTTP 429 {error_text}")
                    attempts_log.append({'model': model, 'status': 'RATE_LIMIT_429'})
                    set_model_cooldown(model, retry_after)
                    break  # try the next model in the list rather than aborting entirely

                if res.status_code in _GROQ_NON_RETRYABLE_STATUSES:
                    error_text = res.text[:500]
                    print(f"❌ {model} unusable: HTTP {res.status_code}: {error_text}")
                    model_errors.append(f"{model}: HTTP {res.status_code} {error_text}")
                    attempts_log.append({'model': model, 'status': f'HTTP_{res.status_code}'})
                    break  # this model id itself is the problem; retrying won't help

                if res.status_code != 200:
                    error_text = res.text[:500]
                    print(f"⚠️ {model} transient error HTTP {res.status_code}: {error_text}")
                    model_errors.append(f"{model}: HTTP {res.status_code} {error_text}")
                    attempts_log.append({'model': model, 'status': f'HTTP_{res.status_code}_RETRY'})
                    if attempt < GROQ_MAX_RETRIES_PER_MODEL:
                        time.sleep(GROQ_RETRY_BACKOFF_SECONDS * (attempt + 1))
                        continue
                    break

                res_data = res.json()
                usage = res_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                choices = res_data.get("choices", [])
                if not choices:
                    print(f"❌ No choices from {model}: {str(res_data)[:300]}")
                    model_errors.append(f"{model}: no choices returned")
                    attempts_log.append({'model': model, 'status': 'NO_CANDIDATES'})
                    break

                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
                content = str(content).strip()
                finish_reason = choices[0].get("finish_reason")
                print(f"✅ Got response from {model} | Tokens: {total_tokens} (completion {completion_tokens}/{model_max_tokens}) | finish_reason={finish_reason}")

                content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE).strip()
                content = re.sub(r'\s*```$', '', content).strip()

                if not content:
                    reason = "Empty content"
                    if finish_reason == 'length':
                        reason = "Response was cut off before any JSON was produced (max_completion_tokens too low for this model/prompt)."
                    print(f"❌ {model}: {reason}")
                    model_errors.append(f"{model}: {reason}")
                    attempts_log.append({'model': model, 'status': 'EMPTY_CONTENT'})
                    break

                try:
                    cleaned = re.sub(r',\s*([}\]])', r'\1', content)
                    result = json.loads(cleaned)
                except json.JSONDecodeError:
                    first = content.find('{')
                    last = content.rfind('}')
                    result = None
                    if first != -1 and last != -1 and last > first:
                        substring = content[first:last + 1]
                        try:
                            substring_clean = re.sub(r',\s*([}\]])', r'\1', substring)
                            result = json.loads(substring_clean)
                        except Exception:
                            result = None
                    if result is None:
                        print(f"❌ JSON parse failed on {model}. Raw: {content[:200]}")
                        model_errors.append(f"{model}: PARSE_ERROR")
                        attempts_log.append({'model': model, 'status': 'PARSE_ERROR', 'finish_reason': finish_reason})
                        # Try the next model instead of giving up entirely.
                        break

                result['model_used'] = model
                result['estimated_tokens'] = estimated_tokens
                result['total_tokens'] = total_tokens
                result['prompt_tokens'] = prompt_tokens
                result['completion_tokens'] = completion_tokens
                result['api_status'] = 'SUCCESS'
                result['model_attempts'] = attempts_log + [{'model': model, 'status': 'SUCCESS'}]
                st.session_state.groq_tokens_used += total_tokens
                st.session_state.groq_rate_limit_reason = ''
                succeeded = True
                return result

            except requests.exceptions.Timeout:
                print(f"⏰ Timeout calling {model} (attempt {attempt + 1})")
                model_errors.append(f"{model}: request timed out")
                attempts_log.append({'model': model, 'status': 'TIMEOUT'})
                if attempt < GROQ_MAX_RETRIES_PER_MODEL:
                    time.sleep(GROQ_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                break
            except Exception as e:
                print(f"❌ Exception calling {model}: {str(e)}")
                model_errors.append(f"{model}: {str(e)}")
                attempts_log.append({'model': model, 'status': f'EXCEPTION: {e}'})
                break
        if succeeded:
            break

    return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW",
            "rejection_reason": "Error: all Groq models failed. " + " | ".join(model_errors[-4:]),
            "model_used": "None", "api_status": "ALL_MODELS_FAILED",
            "estimated_tokens": estimated_tokens,
            "model_attempts": attempts_log,
            "raw_output": "\n".join(model_errors[-4:])}

def build_market_fallback_analysis(symbol, m10, swings, pair_config, dxy_context, candles=None, phase_context=None, live_price=None, htf_context=None, picture=None, firm=None, firm_notes=None, learning=None, historical_context=None):
    if not st.session_state.get("_upgrade_fallback_warned"):
        try:
            add_notification("warning", "Groq AI is unavailable (missing API key or rate-limited). Signals are coming from the Python fallback model. Verify GROQ_API_KEY in Streamlit Secrets for full-quality institutional analysis.")
        except Exception:
            pass
        st.session_state._upgrade_fallback_warned = True
        
    micro = calculate_microstructure(m10) or {}
    regime = classify_market_regime(m10)
    try:
        current_price = float(live_price) if live_price is not None else float(m10["Close"].iloc[-1])
    except Exception:
        current_price = None
        
    if current_price is None:
        return {
            "bias": "RANGING", "signal": "WAIT", "confluence_score": 40, "confidence": "LOW",
            "dxy_correlation": "NEUTRAL", "microstructure_read": "VWAP {} | RVOL {} | Momentum {} | ADX {}".format(
                micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL"), regime.get("adx")),
            "reasoning": "No trade: No reliable current price available.", "rejection_reason": "No reliable current price available.",
            "structural_score": 40, "score_reason": "Upgraded model declined the setup.", "candidate_direction": None,
            "levels_source": "PYTHON", "historical_pattern": historical_context or "", "api_status": "FALLBACK", "model_used": PYTHON_FALLBACK_MODEL
        }
        
    order_blocks = detect_order_blocks(m10)
    fvgs = detect_fvg(m10)
    all_data_ref = st.session_state.get("cached_market_data", {}) or {}
    vote = multi_strategy_vote(symbol, all_data_ref, m10, current_price, swings, order_blocks, fvgs)
    htf_dir = htf_direction_gate(symbol, all_data_ref)
    reversal = detect_reversal(m10)
    direction = vote.get("direction")
    
    if direction is None:
        return {
            "bias": "RANGING", "signal": "WAIT", "confluence_score": 40, "confidence": "LOW",
            "dxy_correlation": "NEUTRAL", "microstructure_read": "VWAP {} | RVOL {} | Momentum {} | ADX {}".format(
                micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL"), regime.get("adx")),
            "reasoning": "No trade: Multi-strategy confluence found no clean edge (buy={}, sell={}). Standing aside instead of guessing on momentum.".format(
                vote.get("buy_strategies"), vote.get("sell_strategies")),
            "rejection_reason": "Multi-strategy confluence found no clean edge.",
            "structural_score": 40, "score_reason": "Upgraded model declined the setup.", "candidate_direction": None,
            "levels_source": "PYTHON", "historical_pattern": historical_context or "", "api_status": "FALLBACK", "model_used": PYTHON_FALLBACK_MODEL
        }
        
    if regime.get("regime") == "RANGING" and reversal is None:
        return {
            "bias": "RANGING", "signal": "WAIT", "confluence_score": 40, "confidence": "LOW",
            "dxy_correlation": "NEUTRAL", "microstructure_read": "VWAP {} | RVOL {} | Momentum {} | ADX {}".format(
                micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL"), regime.get("adx")),
            "reasoning": "No trade: Market is ranging/choppy (ADX {}) with no reversal trigger. Momentum entries here have negative expectancy.".format(regime.get("adx")),
            "rejection_reason": "Market is ranging/choppy with no reversal trigger.",
            "structural_score": 40, "score_reason": "Upgraded model declined the setup.", "candidate_direction": None,
            "levels_source": "PYTHON", "historical_pattern": historical_context or "", "api_status": "FALLBACK", "model_used": PYTHON_FALLBACK_MODEL
        }
        
    if htf_dir and direction != htf_dir and reversal is None:
        return {
            "bias": "RANGING", "signal": "WAIT", "confluence_score": 40, "confidence": "LOW",
            "dxy_correlation": "NEUTRAL", "microstructure_read": "VWAP {} | RVOL {} | Momentum {} | ADX {}".format(
                micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL"), regime.get("adx")),
            "reasoning": "No trade: Proposed {} is counter to the higher-timeframe {} trend with no reversal confirmation. Declining countertrend chop.".format(direction, htf_dir),
            "rejection_reason": "Counter-trend without reversal confirmation.",
            "structural_score": 40, "score_reason": "Upgraded model declined the setup.", "candidate_direction": None,
            "levels_source": "PYTHON", "historical_pattern": historical_context or "", "api_status": "FALLBACK", "model_used": PYTHON_FALLBACK_MODEL
        }
        
    lock = _desk_position_lock(symbol, direction, current_price)
    if lock:
        return {
            "bias": "RANGING", "signal": "WAIT", "confluence_score": 40, "confidence": "LOW",
            "dxy_correlation": "NEUTRAL", "microstructure_read": "VWAP {} | RVOL {} | Momentum {} | ADX {}".format(
                micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL"), regime.get("adx")),
            "reasoning": "No trade: " + lock, "rejection_reason": lock,
            "structural_score": 40, "score_reason": "Upgraded model declined the setup.", "candidate_direction": None,
            "levels_source": "PYTHON", "historical_pattern": historical_context or "", "api_status": "FALLBACK", "model_used": PYTHON_FALLBACK_MODEL
        }
        
    bias = "BULLISH" if direction == "BUY" else "BEARISH"
    supporting = vote.get("buy_strategies") if direction == "BUY" else vote.get("sell_strategies")
    parts = []
    parts.append("Upgraded desk model: {} via multi-strategy confluence ({}).".format(
        direction, ", ".join(supporting) if supporting else "structure"))
    parts.append("Regime {} (ADX {}).".format(regime.get("regime"), regime.get("adx")))
    if htf_dir:
        parts.append("Higher-timeframe trend is {} and aligned.".format(htf_dir))
    if reversal:
        parts.append("Reversal candle {} confirms zone rejection.".format(reversal.get("type")))
    parts.append("Microstructure: VWAP {}, RVOL {}, momentum {}.".format(
        micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL")))
    if historical_context:
        parts.append("Price context: {}".format(historical_context))
        
    return {
        "bias": bias, "signal": normalize_ai_signal(direction), "confluence_score": 76, "confidence": "MEDIUM",
        "dxy_correlation": "CONFIRMING" if dxy_context else "NEUTRAL",
        "microstructure_read": "VWAP {} | RVOL {} | Momentum {} | ADX {}".format(
            micro.get("price_vs_vwap", "NEUTRAL"), micro.get("rvol", 0), micro.get("momentum", "NEUTRAL"), regime.get("adx")),
        "reasoning": " ".join(parts), "rejection_reason": None,
        "structural_score": 72, "score_reason": "Upgraded regime + multi-strategy confluence model.",
        "candidate_direction": direction, "levels_source": "PYTHON",
        "historical_pattern": historical_context or "", "api_status": "FALLBACK", "model_used": PYTHON_FALLBACK_MODEL
    }

def build_market_analysis_prompt():
    return """You are an elite institutional trading desk AI. You have FULL access to all data below including an attached chart screenshot. Use ALL concepts — miss nothing.

DATA PROVIDED:
{data_summary}
MICROSTRUCTURE (M10):
{microstructure_data}
STRUCTURE CONTEXT:
{structure_context}
MARKET STRUCTURE ZONES:
{market_structure_summary}
MULTI-TIMEFRAME CONTEXT (10M/15M/30M/1H/4H):
{multitimeframe_context}
RSI VALUES (MULTI-TIMEFRAME):
{rsi_values}
RSI / DIVERGENCE CONTEXT:
{rsi_context}
PREMIUM/DISCOUNT POSITION:
{premium_discount}
VOLATILITY (ATR):
{volatility_context}
HTF CONTEXT (H1/H4):
{htf_context}
DXY (US Dollar Index) TREND:
{dxy_data}
HISTORICAL CONTEXT:
{historical_context}
STRUCTURAL SCORE (PYTHON):
{structural_score_context}
PYTHON DIRECTIONAL LEDGER:
{directional_ledger}
FIRM DESK BIAS (HTF-FIRST WITH HYSTERESIS):
{firm_bias}
ENTRY GAP RULE (institutional limit-order band, hard-enforced in Python after you answer):
{max_entry_distance}
PYTHON CANDIDATE LIMIT-ENTRY PLANS (pullback levels you may use as-is or refine):
{candidate_levels}

DECISION RULES:
1. Use every supplied input: live quote, all timeframes, RSI/divergence, VWAP, RVOL, DXY, SMC structure, zones, volatility, candles and screenshot.
2. Set market_state to continuation, reversal, exhaustion, trend or coiling. Choose BUY or SELL only; never output WAIT — this desk always resolves to an executable trade.
3. Decide direction in this order: H4/H1 trend, sweep plus divergence, premium/discount, then VWAP/momentum. Explain any override.
4. This desk trades ONLY resting limit orders. The live quote is context for where price is now, never the entry. You must select or build a pullback level the market has to retrace to before the order fills — an order block, FVG, VWAP, premium/discount boundary, swing retracement or session level — and it must sit inside the ENTRY GAP RULE band above (far enough from live price to be a genuine limit, close enough to realistically get tagged). Prefer the supplied PYTHON CANDIDATE LIMIT-ENTRY PLANS when one fits; refine the level only when structure clearly supports a better one, still inside the band.
5. Prefer BUY in discount and SELL in premium, but follow a clearly confirmed sweep/reversal when stronger.
6. Describe bullish and bearish evidence separately; signal must match the stronger side.
7. Use DXY as a macro filter for XAUUSD, EURUSD and BTCUSD. Contradiction reduces confidence but does not automatically reverse direction.
8. order_type is always "LIMIT": BUY LIMIT must sit below the live quote, SELL LIMIT must sit above it. Never propose an entry at or within a hair of the live price — that is a market order in disguise and will be rejected and rebuilt by the execution layer.
9. Set an exhaustion_target at the most realistic opposing range/liquidity extreme that the move can reach before stalling, measured FROM THE LIMIT ENTRY, not from the live price. Use it as TP. Put SL beyond clear invalidation with ATR room; never guarantee a target, and reject levels that violate minimum 1.5 R:R or maximum risk.
10. Keep all numbers identical across entry, stop_loss, take_profit and order_description — and make sure the written reasoning explains the SAME entry/SL/TP numbers you output, not a different price you considered along the way.
11. If the screenshot is usable, identify visible support, resistance, liquidity, trendlines and patterns in visual_levels; otherwise say unavailable.
12. Also populate "next_level_watch": the next key structural level beyond your take_profit in the trade direction (the level price is likely to approach next if the setup keeps running), with a one-line reason it matters.
13. Be concise but specific: reasoning must mention HTF bias, key evidence, macro, momentum/volume, entry anchor, invalidation, target, risk, and why the entry is a genuine pullback rather than a chase. Complete valid JSON before adding detail.

ENTRY EXECUTION RULES:
- Choose from PYTHON CANDIDATE LIMIT-ENTRY PLANS when one is well-supported.
- Any modified level must stay anchored to a swing, OB, FVG, session level, VWAP, premium/discount boundary, or liquidity pool — never a round-number guess and never the live price.
- Use ENTRY price for SL/TP math, not live price.
- BUY LIMIT: entry below live price, inside the band. SELL LIMIT: entry above live price, inside the band.
- SL beyond stop_anchor. TP respects tp_anchor.

OUTPUT STRICT JSON ONLY (NO MARKDOWN, NO CODE FENCES):
{{
"market_state": "continuation|reversal|exhaustion|trend|coiling",
"bias": "BULLISH|BEARISH|RANGING",
"signal": "BUY|SELL",
"confluence_score": 0,
"confidence": "HIGH|MEDIUM|LOW",
"dxy_correlation": "CONFIRMING|CONTRADICTING|NEUTRAL",
"microstructure_read": "Brief VWAP/RVOL status and intrabar read",
"directional_evidence": {{"bullish": ["item1","item2"], "bearish": ["item1","item2"]}},
"visual_levels": "Describe key levels seen on the chart screenshot: support, resistance, liquidity pools, trendlines",
"entry": 0.00,
"stop_loss": 0.00,
"take_profit": [0.00, 0.00],
"exhaustion_target": 0.00,
"rr_ratio": 0.00,
"order_type": "LIMIT",
"entry_anchor": "demand zone / swing low / FVG / VWAP / session low",
"stop_anchor": "swing low / OB low / FVG bottom / invalidation level",
"tp_anchor": "exhaustion range extreme / swing high / supply zone / FVG top / session high",
"next_level_watch": "Next key structural level beyond TP and why price may approach it next",
"order_expiry": "until next H1 close / until structure invalidates / GTC",
"order_description": "Execution plan using SAME numbers as entry/stop_loss/take_profit.",
"confluence_breakdown": "Weighting behind score: DXY, RSI, VWAP, RVOL, structure, premium/discount, market phase.",
"reasoning": "Concise pair-specific analysis covering HTF structure, strongest evidence, RSI/divergence, DXY, VWAP/RVOL, entry anchor, invalidation, exhaustion target and risk.",
"rejection_reason": ""
}}"""

def analyze_symbol_premium(symbol, all_data, image_b64=None, image_mime_type='image/png'):
    try:
        data = all_data.get(symbol, {})
        m10 = data.get('M10', pd.DataFrame())
        h1 = data.get('H1', pd.DataFrame())
        h4 = data.get('H4', pd.DataFrame())
        live_snapshot = get_live_market_snapshot(symbol, YFINANCE_MAP.get(symbol, symbol), fallback_df=m10)
        if m10.empty:
            return {"error": f"Failed to fetch market data for {symbol}. Yahoo Finance may be temporarily rate-limiting your IP. Please wait a few minutes and try again."}

        refreshed_data = refresh_symbol_data_if_stale(
            symbol, YFINANCE_MAP.get(symbol, symbol), live_snapshot, data
        )
        if refreshed_data is not data:
            data = refreshed_data
            all_data[symbol] = data
            m10 = data.get('M10', pd.DataFrame())
            h1 = data.get('H1', pd.DataFrame())
            h4 = data.get('H4', pd.DataFrame())
            if m10.empty:
                return {"error": f"Fresh market data was unavailable for {symbol}. Please try again."}
            
        micro = calculate_microstructure(m10)
        current_price = live_snapshot.get('price') or float(m10['Close'].iloc[-1])
        swings = find_swings(m10)
        pair_config = get_pair_config(symbol)
        
        dxy_data = all_data.get('DXY', {}).get('H1', pd.DataFrame())
        dxy_summary = "DXY Data Unavailable"
        dxy_context = None
        if not dxy_data.empty:
            dxy_price = dxy_data['Close'].iloc[-1]
            dxy_micro = calculate_microstructure(dxy_data)
            dxy_context = {'trend': dxy_micro['momentum'], 'price_vs_vwap': dxy_micro['price_vs_vwap']}
            dxy_summary = f"Current: {dxy_price} | VWAP Position: {dxy_micro['price_vs_vwap']} | Momentum: {dxy_micro['momentum']} | RVOL: {dxy_micro['rvol']}"
            
        htf_context = None
        if not h4.empty:
            h4_micro = calculate_microstructure(h4)
            htf_context = {'trend': h4_micro['momentum'], 'bias': h4_micro['momentum'], 'price_vs_vwap': h4_micro['price_vs_vwap']}
            
        picture = build_mtf_picture(all_data, symbol)
        firm = None
        firm_notes = []
        if picture:
            firm, firm_notes = resolve_firm_direction(symbol, picture)
            
        phase_context = detect_market_phase(m10, swings=swings)
        setup_context = build_setup_context(m10, swings, current_price, symbol, dxy_context=dxy_context)
        structural_context = calculate_structural_score(m10, symbol, dxy_context=dxy_context, phase_context=phase_context)
        candles = analyze_candle_structure(m10)
        
        bos, choch = detect_bos_choch(m10)
        order_blocks = detect_order_blocks(m10)
        fvgs = detect_fvg(m10)
        sweeps = detect_liquidity_sweeps(m10)
        
        structure_parts = []
        if bos or choch: structure_parts.append(f"BOS/CHOCH: {bos or choch}")
        if order_blocks: structure_parts.append("Order blocks: " + ", ".join([f"{ob['type']}@{ob['price']:.2f}" for ob in order_blocks]))
        if fvgs: structure_parts.append("FVGs: " + ", ".join([f"{fvg['type']}({fvg['top']:.2f}->{fvg['bottom']:.2f})" for fvg in fvgs]))
        if sweeps: structure_parts.append("Sweeps: " + ", ".join([f"{s['type']}@{s['price']:.2f}" for s in sweeps]))
        if candles: structure_parts.append("Recent candles: " + "; ".join([f"{c['time'].strftime('%H:%M')} {c['pattern']} ({c['candle_type']})" for c in candles]))
        structure_context = " | ".join(structure_parts) if structure_parts else "No strong structural clues detected."
        
        market_structure_summary = build_market_structure_summary(m10, current_price=current_price, swings=swings, order_blocks=order_blocks, fvgs=fvgs, sweeps=sweeps, bos=bos, choch=choch, symbol=symbol)
        multitimeframe_context = build_multitimeframe_context(all_data, symbol)
        rsi_values = build_rsi_values_context(all_data, symbol)
        premium_discount = build_premium_discount_context(m10, current_price)
        volatility_context = build_volatility_context(m10)
        
        ledger = detect_directional_confluence(m10, swings=swings, htf_context=htf_context, dxy_context=dxy_context, symbol=symbol)
        directional_ledger = f"Bullish ({ledger['bull_count']}): {'; '.join(ledger['bullish_evidence']) or 'none'} | Bearish ({ledger['bear_count']}): {'; '.join(ledger['bearish_evidence']) or 'none'} | Ledger direction: {ledger['direction'] or 'none'}"
        
        rsi_context = ''
        if not m10.empty:
            divergence = detect_rsi_divergence(m10)
            rsi_context = f"M10 RSI context: {divergence['type']} - {divergence['reason']}" if divergence else 'M10 RSI context: no clear divergence detected.'
            
        m15_data = data.get('M15', pd.DataFrame())
        m30_data = data.get('M30', pd.DataFrame())
        for label, frame in [('M15', m15_data), ('M30', m30_data), ('H1', h1)]:
            if frame is not None and not getattr(frame, 'empty', True):
                d = detect_rsi_divergence(frame)
                if d:
                    rsi_context += f" | {label} RSI context: {d['type']} - {d['reason']}"
        if not rsi_context:
            rsi_context = 'RSI context unavailable.'
            
        h1_summary = f"Latest H1 close: {h1['Close'].iloc[-1]:.2f}" if not h1.empty else "H1 data unavailable"
        h4_summary = f"Latest H4 close: {h4['Close'].iloc[-1]:.2f}" if not h4.empty else "H4 data unavailable"
        htf_summary = f"H1: {h1_summary} | H4: {h4_summary}"
        
        prompt_data = f"Symbol: {symbol} | Live Price: {current_price} | Live Price Source: {live_snapshot.get('source')} | Quote Time: {live_snapshot.get('quote_time', 'N/A')} | Swing Highs: {swings['recent_swing_highs']} | Swing Lows: {swings['recent_swing_lows']} | Market Phase: {phase_context['phase']} | Phase Reason: {phase_context['reason']} | Setup Type: {setup_context['setup_type']} | Entry Timing: {setup_context['entry_timing']} | Entry Quality: {phase_context['entry_quality']} | Entry Rule: propose a resting LIMIT order at a genuine pullback level inside the entry-gap band; never at or near the live price."
        prompt_micro = f"VWAP: {micro.get('vwap', 'N/A')} | Price vs VWAP: {micro.get('price_vs_vwap', 'N/A')} | RVOL: {micro.get('rvol', 'N/A')} ({micro.get('volume_anomaly', 'N/A')})"
        structural_score_context = f"Python structural score: {structural_context['structural_score']}/100 | Basis: {structural_context['score_reason']}"
        historical_context = build_historical_context(m10)
        firm_bias_text = f"{firm} (standing desk bias; weighted MTF evidence {picture.get('score', 0):+.1f})" if firm else "NONE - evidence tied; stand aside unless a clear edge emerges."
        atr_for_entry = setup_context.get('atr')
        min_gap, max_gap = get_entry_gap_bounds(current_price, atr_for_entry, pair_config)
        max_entry_distance = (
            f"Entry must be between {min_gap:.5f} and {max_gap:.5f} price-units away from the live quote "
            f"({symbol}) — below live price for BUY LIMIT, above it for SELL LIMIT. Closer than "
            f"{min_gap:.5f} is a disguised market order and will be rejected and rebuilt; farther than "
            f"{max_gap:.5f} is unrealistic and will be rejected and rebuilt."
        )
        candidate_levels = build_candidate_levels(symbol, current_price, swings, order_blocks, fvgs, atr_for_entry, pair_config, vwap=micro.get('vwap'))
        
        all_format_kwargs = {
            'data_summary': prompt_data, 'microstructure_data': prompt_micro, 'structure_context': structure_context,
            'market_structure_summary': market_structure_summary, 'multitimeframe_context': multitimeframe_context,
            'rsi_values': rsi_values, 'rsi_context': rsi_context, 'premium_discount': premium_discount,
            'volatility_context': volatility_context, 'htf_context': htf_summary, 'dxy_data': dxy_summary,
            'historical_context': historical_context, 'structural_score_context': structural_score_context,
            'directional_ledger': directional_ledger, 'firm_bias': firm_bias_text,
            'max_entry_distance': max_entry_distance, 'candidate_levels': json.dumps(candidate_levels, indent=2, default=str),
        }
        
        try:
            prompt_text = build_market_analysis_prompt().format(**all_format_kwargs)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            prompt_text = build_market_analysis_prompt().format(**{**all_format_kwargs, missing_key: f"[missing:{missing_key}]"})
            
        user_content = [{"type": "text", "text": prompt_text}]
        estimated_tokens = estimate_analysis_tokens(prompt_text, [])
        
        # 🚀 CALL AI FIRST — max_tokens intentionally omitted so call_groq applies each
        # model's own tuned max_completion_tokens/reasoning_effort instead of one shared cap.
        analysis = call_groq(
            prompt_text, [],
            estimated_tokens=estimated_tokens,
            image_b64=image_b64, image_mime_type=image_mime_type
        )
        st.session_state.last_model_attempts[symbol] = analysis.get('model_attempts', [])
        
        _post_ai_snapshot = get_live_market_snapshot(symbol, YFINANCE_MAP.get(symbol, symbol), fallback_df=m10)
        if _post_ai_snapshot.get("price"):
            current_price = _post_ai_snapshot.get("price")
            
        analysis = normalize_analysis_signals(analysis)
        analysis.setdefault('model_used', PYTHON_FALLBACK_MODEL)
        analysis.setdefault('microstructure_read', prompt_micro)
        analysis.setdefault('rsi_context', rsi_context)
        analysis.setdefault('dxy_summary', dxy_summary)
        analysis.setdefault('live_price', current_price)
        analysis['setup_context'] = setup_context
        analysis['market_state'] = analysis.get('market_state') or setup_context['setup_type']
        update_market_state(analysis.get('market_state') or setup_context['setup_type'])
        
        # 🚨 STRICT FALLBACK TRIGGER: Only fallback if API completely failed
        if analysis.get('api_status') not in ['SUCCESS', 'SUCCESS_EXTRACTED']:
            groq_failure = analysis.get('rejection_reason', 'Unknown API Error')
            groq_status = analysis.get('api_status', 'UNKNOWN')
            groq_model = analysis.get('model_used', 'None')
            groq_raw_output = analysis.get('raw_output', '')
            groq_tokens = {
                key: analysis.get(key, 0)
                for key in ('total_tokens', 'prompt_tokens', 'completion_tokens')
            }
            analysis = build_market_fallback_analysis(symbol, m10, swings, pair_config, dxy_context, candles=candles, phase_context=phase_context, live_price=current_price, htf_context=htf_context, picture=picture, firm=firm, firm_notes=firm_notes, learning=None, historical_context=historical_context)
            analysis = normalize_analysis_signals(analysis)
            analysis['groq_failure'] = groq_failure
            analysis['groq_api_status'] = groq_status
            analysis['groq_model'] = groq_model
            analysis.update(groq_tokens)
            if groq_raw_output:
                analysis['groq_raw_output'] = groq_raw_output
            analysis['estimated_tokens'] = analysis.get('estimated_tokens', estimated_tokens)
            
        # 🚨 FORCE BUY/SELL (No WAIT allowed from AI)
        if analysis.get('signal') == 'WAIT':
            firm_norm = normalize_ai_signal(firm) if firm else None
            if firm_norm:
                analysis['signal'] = firm_norm
                add_python_validation_note(analysis, f"AI attempted to WAIT; direction forced to {firm_norm} based on standing desk bias.")
            else:
                analysis['signal'] = 'BUY' if micro.get('momentum') == 'BULLISH' else 'SELL'
                add_python_validation_note(analysis, f"AI attempted to WAIT; direction forced to {analysis['signal']} based on microstructure momentum.")
            analysis['confidence'] = 'LOW'
            
        analysis = apply_htf_trend_guard(analysis, symbol, htf_context)
        
        ai_score = int(round(analysis.get('confluence_score', 0)))
        analysis['confluence_score'] = min(100, max(0, ai_score))
        if analysis.get('confidence') == 'LOW' and analysis['confluence_score'] >= 75:
            analysis['confidence'] = 'MEDIUM'
        elif analysis.get('confidence') == 'MEDIUM' and analysis['confluence_score'] >= 85:
            analysis['confidence'] = 'HIGH'
            
        analysis['structural_score'] = structural_context['structural_score']
        analysis['atr'] = setup_context.get('atr')
        analysis['score_reason'] = structural_context['score_reason']
        analysis['candidate_direction'] = structural_context['candidate_direction']
        
        analysis = apply_dxy_guardrails(analysis, symbol, dxy_context)
        analysis = cross_check_ai_evidence(analysis)
        analysis = apply_direction_correction_guard(analysis, ledger, symbol)
        analysis = cross_check_ai_evidence(analysis)
        
        firm_norm = normalize_ai_signal(firm) if firm else None
        if firm_norm and analysis.get('signal') in ('BUY', 'SELL') and analysis['signal'] != firm_norm:
            ev = analysis.get('directional_evidence') or {}
            counter = len(ev.get('bullish', [])) if firm_norm == 'BUY' else len(ev.get('bearish', []))
            if counter >= 3:
                add_python_validation_note(analysis, f"AI overrode the standing {firm_norm} desk bias with {counter} counter-evidences.")
            else:
                analysis['signal'] = firm_norm
                add_python_validation_note(analysis, f"Standing {firm_norm} desk bias was maintained and the AI direction was aligned to it.")
                
        analysis = apply_conservative_signal_filter(analysis, structural_context, candles, dxy_context, current_price, swings, symbol, pair_config=pair_config)
        
        analysis = finalize_trade_plan(
            analysis=analysis, symbol=symbol, current_price=current_price, swings=swings,
            order_blocks=order_blocks, fvgs=fvgs, atr=setup_context.get('atr'),
            pair_config=pair_config, market_df=m10
        )
        
        if analysis.get('signal') in ('BUY', 'SELL') and analysis.get('take_profit'):
            final_note = f"Final levels: Entry {analysis['entry']} | SL {analysis['stop_loss']} | TP {analysis['take_profit'][0]}."
            analysis['order_description'] = f"{final_note} {analysis.get('order_description') or ''}".strip()
            if not analysis.get('next_level_watch'):
                # The AI didn't populate this (or Python built the plan) — derive the next
                # structural level beyond TP so the user always sees where price is likely
                # headed next if the trade keeps running, per "next level approaching".
                try:
                    tp1 = analysis['take_profit'][0]
                    next_level = build_exhaustion_target(
                        analysis['signal'], tp1, m10, swings, order_blocks, fvgs,
                        setup_context.get('atr'), pair_config
                    )
                    if next_level and abs(next_level - tp1) > (setup_context.get('atr') or 0) * 0.1:
                        analysis['next_level_watch'] = (
                            f"Beyond TP1, the next level in play is ~{round_price(next_level, pair_config)} "
                            f"— watch for price to approach it if {symbol} continues {analysis['signal'].lower()}ing "
                            f"through the current {analysis.get('market_state', 'phase')}."
                        )
                except Exception:
                    pass
            
        analysis['validation_detail'] = build_validation_detail(analysis, swings, current_price, symbol, pair_config=pair_config, structural_context=structural_context)
        analysis['display_reasoning'] = build_display_reason(analysis, symbol, current_price=current_price, phase_context=phase_context, structural_context=structural_context, dxy_context=dxy_context)
        analysis['symbol'] = symbol
        analysis = normalize_analysis_signals(analysis)
        analysis['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return analysis
        
    except Exception as e:
        return {"error": str(e), "api_status": "PYTHON_EXCEPTION"}

# ── UI Layout ──────────────────────────────────────────────────────────────
st.title("📊 Der-AI | Institutional Market Analysis")
st.markdown("**Multi-Timeframe Structure | SMC | DXY Correlation | Multimodal Chart Analysis**")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Analysis", "📜 Signal History", "🔔 Notifications", "⚙️ Settings"])

with tab1:
    st.header("🚀 Run AI Market Analysis")
    selected_symbols = st.multiselect("Select Symbols to Analyse", SYMBOLS, default=['XAUUSD', 'EURUSD', 'BTCUSD'])
    uploaded_file = st.file_uploader("📸 Attach Market Chart Screenshot (Optional - AI will analyze price action)", type=["png", "jpg", "jpeg"])
    
    image_b64 = None
    image_mime_type = 'image/png'
    if uploaded_file is not None:
        image_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
        image_mime_type = uploaded_file.type or image_mime_type
        st.image(uploaded_file, caption="Uploaded Chart Snapshot", width=400)

    if st.button("🧠 Analyse Market Now", type="primary"):
        if not get_secret("GROQ_API_KEY"):
            st.error("⚠️ Please set your GROQ_API_KEY in Streamlit Secrets.")
        else:
            with st.spinner("Fetching market data and running institutional analysis..."):
                all_data = fetch_all_data()
                st.session_state.cached_market_data = all_data
                
                for symbol in selected_symbols:
                    st.info(f"Analysing {symbol}...")
                    result = analyze_symbol_premium(symbol, all_data, image_b64=image_b64, image_mime_type=image_mime_type)
                    
                    if 'error' in result:
                        st.error(f"❌ {symbol}: {result['error']}")
                        add_notification('warning', f"❌ {symbol}: {result['error']}", symbol=symbol)
                    else:
                        # 🚨 DEBUGGING UI: Show API Status and Tokens
                        api_status = result.get('api_status', 'UNKNOWN')
                        model_used = result.get('model_used', 'Unknown')
                        total_tokens = result.get('total_tokens', 0)
                        prompt_tokens = result.get('prompt_tokens', 0)
                        completion_tokens = result.get('completion_tokens', 0)
                        rejection = result.get('rejection_reason', '')
                        
                        status_color = "green" if api_status in ['SUCCESS', 'SUCCESS_EXTRACTED', 'FALLBACK'] else "red"
                        primary_ok = model_used == GROQ_MODELS[0] and api_status in ['SUCCESS', 'SUCCESS_EXTRACTED']
                        model_badge = "🥇 primary" if primary_ok else ("⬇️ fallback model" if model_used in GROQ_MODELS else "")
                        st.markdown(f"**🤖 AI Model:** `{model_used}` {model_badge} | **🔋 Tokens Used:** `{total_tokens}` (Prompt: {prompt_tokens}, Completion: {completion_tokens}) | **📡 Status:** <span style='color:{status_color}; font-weight:bold;'>{api_status}</span>", unsafe_allow_html=True)

                        attempts = st.session_state.last_model_attempts.get(symbol) or result.get('model_attempts')
                        if attempts and (not primary_ok or any(a.get('status') not in ('SUCCESS',) for a in attempts)):
                            with st.expander(f"🔎 Why {GROQ_MODELS[0]} wasn't used" if not primary_ok else "🔎 Model attempt log"):
                                for a in attempts:
                                    st.caption(f"`{a.get('model')}` → {a.get('status')}")

                        if api_status == 'FALLBACK' and result.get('groq_failure'):
                            st.warning(
                                f"Groq unavailable ({result.get('groq_api_status', 'UNKNOWN')}): "
                                f"{result['groq_failure']}"
                            )
                        
                        if api_status not in ['SUCCESS', 'SUCCESS_EXTRACTED', 'FALLBACK']:
                            with st.expander("🐛 Debug AI Response (Why it failed)"):
                                st.code(result.get('raw_output', 'No raw output captured.'), language='json')
                                st.error(f"Rejection Reason: {result.get('rejection_reason', 'Unknown')}")
                        
                        is_valid_logic, logic_reason = validate_ai_logic(result)
                        if not is_valid_logic:
                            st.info(f"⚪ {symbol}: Signal Rejected. AI Logic Flaw: {logic_reason}")
                            add_notification('warning', f"⚪ {symbol}: Signal Rejected. AI Logic Flaw: {logic_reason}", symbol=symbol, signal=result.get('signal'))
                            continue
                        
                        if result.get('signal') == 'WAIT':
                            ai_reason = result.get('display_reasoning') or result.get('reasoning') or result.get('rejection_reason', 'Market conditions do not meet high-confidence criteria.')
                            st.info(f"⚪ {symbol}: No Trade (WAIT). AI Reason: {ai_reason}")
                            add_notification('info', f"⚪ {symbol}: No Trade (WAIT). AI Reason: {ai_reason}", symbol=symbol, signal='WAIT', score=result.get('confluence_score'))
                            continue
                        
                        pair_config = get_pair_config(symbol)
                        is_valid_math, math_reason = validate_signal_math(result, pair_config=pair_config)
                        if not is_valid_math:
                            st.info(f"⚪ {symbol}: Signal Rejected. AI Reason: {math_reason}")
                            add_notification('warning', f"⚪ {symbol}: Signal Rejected. AI Reason: {math_reason}", symbol=symbol, signal=result.get('signal'))
                            continue
                        
                        combined_score = result.get('confluence_score', 0)
                        if result.get('signal') in ('BUY', 'SELL') and result.get('confidence') in ['HIGH', 'MEDIUM', 'LOW']:
                            sig_color = "🟢" if result.get('signal') == "BUY" else "🔴"
                            st.markdown(f"### {sig_color} **NEW SIGNAL:** {result.get('symbol', symbol)} - {result.get('signal')}")
                            st.write(f"**DXY Correlation:** {result.get('dxy_correlation', 'N/A')}")
                            st.write(f"**Microstructure:** {result.get('microstructure_read', 'N/A')}")
                            st.write(f"**Visual Levels (Chart):** {result.get('visual_levels', 'N/A')}")
                            st.info(f"**Entry (LIMIT):** {result.get('entry')} | **SL:** {result.get('stop_loss')} | **TP:** {result.get('take_profit')} | **R:R:** {result.get('rr_ratio', 'N/A')}")
                            st.write(f"**Order Type:** {result.get('order_type', 'LIMIT')}")
                            if result.get('order_description'):
                                st.write(f"**Execution Plan:** {result.get('order_description')}")
                            if result.get('next_level_watch'):
                                st.write(f"**🔭 Next Level Watch:** {result.get('next_level_watch')}")
                            st.write(f"**AI Reasoning:** {result.get('reasoning')}")
                            validation_notes = result.get('python_validation_notes') or []
                            if validation_notes:
                                st.write(f"**Python Validation Notes:** {' '.join(validation_notes)}")
                            st.markdown("---")
                            
                            st.session_state.active_signals[symbol] = {'direction': result.get('signal'), 'entry': result.get('entry', 0), 'timestamp': datetime.now(), 'score': combined_score}
                            result['analyzed_at'] = datetime.now()
                            st.session_state.signal_history.append(result)
                            
                            msg = build_telegram_signal_message(symbol, result)
                            if send_telegram_message(msg):
                                st.success("✅ Signal sent to Telegram!")
                            add_notification('success', f"✅ {symbol}: New {result.get('signal')} signal accepted via {result.get('model_used', PYTHON_FALLBACK_MODEL)}. Score: {combined_score}/100. Entry: {result.get('entry')} | SL: {result.get('stop_loss')} | TP: {result.get('take_profit', ['N/A'])[0] if result.get('take_profit') else 'N/A'}.", symbol=symbol, signal=result.get('signal'), score=combined_score)
                        else:
                            ai_reason = result.get('display_reasoning') or result.get('reasoning') or result.get('rejection_reason', 'Low confidence or DXY contradiction')
                            st.info(f"⚪ {symbol}: Signal Rejected. Score: {result.get('confluence_score', 0)}/100, Confidence: {result.get('confidence', 'N/A')}. AI Reason: {ai_reason}")
                            add_notification('warning', f"⚪ {symbol}: Signal Rejected. Score: {result.get('confluence_score', 0)}/100, Confidence: {result.get('confidence', 'N/A')}. AI Reason: {ai_reason}", symbol=symbol, signal=result.get('signal'), score=result.get('confluence_score'))

with tab2:
    st.header("📜 Premium Signal History")
    if not st.session_state.signal_history:
        st.info("📭 No signals generated yet. Run an analysis in the Market Analysis tab.")
    else:
        premium_signals = [s for s in st.session_state.signal_history if s.get('confidence') == 'HIGH' and s.get('confluence_score', 0) >= 80]
        st.metric("Total Premium Signals Logged", len(premium_signals))
        for i, signal in enumerate(reversed(premium_signals)):
            with st.expander(f"{'🟢' if signal.get('signal') == 'BUY' else '🔴'} {signal.get('symbol', 'N/A')} - {signal.get('signal')} | Score: {signal.get('confluence_score')}/100 | {signal.get('timestamp', 'N/A')}", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Entry", signal.get('entry', 'N/A'))
                col_b.metric("Stop Loss", signal.get('stop_loss', 'N/A'))
                col_c.metric("Take Profit", signal.get('take_profit', ['N/A'])[0] if signal.get('take_profit') else 'N/A')
                st.write(f"**Analysis model:** {signal.get('model_used', PYTHON_FALLBACK_MODEL)}")
                st.write(f"**Tokens Used:** {signal.get('total_tokens', 'N/A')}")
                st.write(f"**Bias:** {signal.get('bias')} | **Confidence:** {signal.get('confidence')}")
                st.write(f"**DXY Correlation:** {signal.get('dxy_correlation', 'N/A')}")
                st.write(f"**Reasoning:** {signal.get('reasoning')}")
                st.markdown("---")

with tab3:
    st.header("🔔 Notifications")
    if not st.session_state.notifications:
        st.info("📭 No notifications yet.")
    else:
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            filter_type = st.selectbox("Filter by type", ["All", "Signals Only", "Warnings", "Info", "Success"], key="notif_filter")
        with ctrl_col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_notifications()
                st.rerun()
        
        notifications = get_notifications()
        filtered_notifications = list(reversed(notifications))
        if filter_type == "Signals Only":
            filtered_notifications = [n for n in filtered_notifications if n.get('signal') in ('BUY', 'SELL')]
        elif filter_type == "Warnings":
            filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'warning']
        elif filter_type == "Info":
            filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'info']
        elif filter_type == "Success":
            filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'success']
        
        if not filtered_notifications:
            st.info("📭 No notifications match the current filter.")
        else:
            st.caption(f"Showing {len(filtered_notifications)} notification(s)")
            for note in filtered_notifications:
                note_type = note.get('type', 'info')
                badges = []
                if note.get('symbol'):
                    badges.append(f"`{note['symbol']}`")
                if note.get('signal'):
                    sig_emoji = "🟢" if note['signal'] == "BUY" else "🔴" if note['signal'] == "SELL" else "⚪"
                    badges.append(f"{sig_emoji} {note['signal']}")
                if note.get('score') is not None:
                    badges.append(f"📈 {note['score']}/100")
                header = f"**[{note.get('time', '')}]** {' '.join(badges)}" if badges else f"**[{note.get('time', '')}]**"
                if note_type == 'success':
                    st.success(f"{header}\n{note.get('message', '')}")
                elif note_type == 'warning':
                    st.warning(f"{header}\n{note.get('message', '')}")
                else:
                    st.info(f"{header}\n{note.get('message', '')}")

with tab4:
    st.header("⚙️ System Settings")
    st.info("Ensure `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` are set in your Streamlit Secrets.")
    st.markdown(f"- **AI Model priority:** {' → '.join(GROQ_MODELS)} (always attempted in this order via Groq, multimodal)")
    st.markdown("- **Execution:** All signals are resting LIMIT orders at a structural pullback level — no market or stop entries, no WAIT signals")
    st.markdown("- **Execution:** Manual trigger only (No auto-loop)")
    st.markdown("- **Features:** SMC, BOS/CHOCH, FVG, Order Blocks, Liquidity Sweeps, DXY Correlation, Regime Filter (ADX), Multi-Strategy Confluence")
    st.markdown(f"- **Minimum Confluence Score:** {MINIMUM_CONFLUENCE_SCORE}/100")
    st.markdown("- **Chart Screenshot:** Upload market charts for AI to analyze alongside data")

    st.markdown("---")
    st.subheader("🔧 Groq Model Diagnostics")
    st.caption(
        "GROQ_MODELS is always tried in the order below — this app no longer lets Groq's "
        "model-discovery endpoint silently reorder or drop the primary model. Use this panel "
        "if the primary model still isn't being used, to see what your account/key actually "
        "reports as available."
    )
    for m in GROQ_MODELS:
        cfg = GROQ_MODEL_CONFIG.get(m, GROQ_DEFAULT_MODEL_CONFIG)
        st.write(f"**{m}** — max_completion_tokens={cfg['max_completion_tokens']}, reasoning_effort={cfg.get('reasoning_effort')}")
    if st.button("Check Groq /models endpoint now"):
        api_key = get_secret("GROQ_API_KEY", "").strip()
        if not api_key:
            st.error("GROQ_API_KEY is not set.")
        else:
            discovery = get_groq_models(api_key)
            if discovery.get('ok'):
                available = discovery.get('available', [])
                for m in GROQ_MODELS:
                    if m in available:
                        st.success(f"✅ {m} — reported available by Groq")
                    else:
                        st.warning(f"⚠️ {m} — NOT reported by Groq's /models endpoint right now (the app will still try it; a live HTTP error at call time is the authoritative signal, not this list)")
            else:
                st.error(f"Could not reach Groq /models endpoint: {discovery.get('reason')}")
    last_attempts = st.session_state.get('last_model_attempts') or {}
    if last_attempts:
        st.markdown("**Last analysis run — per-symbol model attempts:**")
        for sym, attempts in last_attempts.items():
            if not attempts:
                continue
            st.write(f"`{sym}`: " + " → ".join(f"{a.get('model')}:{a.get('status')}" for a in attempts))
