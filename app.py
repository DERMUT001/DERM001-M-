# =============================================================================
# Der-AI | Institutional Market Analysis — V3 "STRUCTURE-LOCKED" BUILD
#
# Why V2 flipped BUY -> SELL within ~3 hours (root causes found in the previous build):
#   1. Direction was a weighted vote in which M10/M15/M30 (combined weight 3.5) nearly
#      out-voted H1/H4 (5.0), so ten-minute noise could flip the call.
#   2. "Order blocks", "FVGs" and "swings" were computed from only the last 2-3 candles (or a
#      4-swing window) and INCLUDING the still-forming candle, so every level moved with
#      every new high/low — the entry price was rebuilt from scratch on every run.
#   3. The bias-hold rule (45 min) lived only in st.session_state, so it was lost whenever the
#      session restarted, and the AI was free to override the firm bias anyway.
#   4. The entry band was so tight that genuine HTF zones were rejected and replaced by
#      whatever LTF level was nearest to live price.
#
# What V3 does instead:
#   * Direction comes from H4 + H1 ONLY (confirmed pivots, HH/HL/LH/LL, close-based BOS/CHOCH,
#     EMA trend, RSI regime, DXY filter) and is protected by PERSISTENT hysteresis.
#   * Structure uses CLOSED candles only; order blocks require displacement + a structure break
#     and are mitigation-aware; FVGs track how much is still unfilled; swept swings are retired;
#     equal highs/lows and PDH/PDL/PWH/PWL are tracked as liquidity.
#   * The limit entry is chosen from ranked HTF zones (H4/H1 OB, FVG, OTE, swings, liquidity),
#     then refined by M30/M15 structure INSIDE the zone. SL/TP are structural (TP1/TP2 = next
#     opposing HTF levels paying >= 1.5R).
#   * PLAN LOCK: one plan per symbol is created and managed (pending/filled/TP/SL/missed/expired/
#     stale/cancelled) — re-running returns the same plan; it is replaced only when it resolves
#     or the HTF bias genuinely flips. Results are tracked in a ledger with win rate and net R.
#   * The AI is now an AUDITOR: it cannot change direction or levels; it flags risks, reads chart
#     screenshots and adjusts the transparent 100-point Python score by at most +/-6.
#   * Keeps everything from V2: Groq model order, per-model token budgets, Telegram, LIMIT-only.
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


PYTHON_FALLBACK_MODEL = 'Python structure engine'

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

if 'persist_state' not in st.session_state: st.session_state.persist_state = None
if 'min_send_score' not in st.session_state: st.session_state.min_send_score = MINIMUM_CONFLUENCE_SCORE
if 'send_low_conviction' not in st.session_state: st.session_state.send_low_conviction = False
if 'notify_plan_events' not in st.session_state: st.session_state.notify_plan_events = True

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


def fetch_candles_from_bitfinex(symbol, interval, limit=None):
    pair_map = {'BTCUSD': 'tBTCUSD', 'XAUUSD': 'tXAUT:USD', 'EURUSD': 'tEURUSD', 'DXY': None}
    bitfinex_symbol = pair_map.get(symbol)
    if not bitfinex_symbol:
        return pd.DataFrame()
    interval_map = {'5m': '5m', '15m': '15m', '30m': '30m', '60m': '1h', '1h': '1h', '4h': '4h'}
    default_limits = {'5m': 1000, '15m': 600, '30m': 500, '1h': 700, '4h': 400}
    interval_code = interval_map.get(interval)
    if not interval_code:
        return pd.DataFrame()
    limit = int(limit or default_limits.get(interval_code, 300))
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
                'timestamp': ts, 'Open': float(open_price), 'High': float(high_price),
                'Low': float(low_price), 'Close': float(close_price), 'Volume': float(volume)
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
            direct_df = fetch_candles_from_bitfinex(symbol, interval)
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
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if 'Volume' not in df.columns:
            df['Volume'] = 0.0
        df['Volume'] = df['Volume'].fillna(0.0)
        # Normalise every index to tz-aware UTC so closed-candle logic, day anchoring and plan
        # tracking behave identically across data sources.
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
        df = df.sort_index()
        return df if len(df) >= 5 else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Yahoo fetch failed for {yf_symbol} [{interval}/{period}]: {e}")
        return pd.DataFrame()

def fetch_symbol_data(symbol, yf_symbol):
    df_m10 = pd.DataFrame()
    # True M10 = two 5m bars (the old code asked Yahoo for '10m', which doesn't exist, and
    # silently fell back to M15 while still calling it M10).
    df5 = fetch_ohlcv(yf_symbol, '5m', '5d')
    if df5 is not None and not df5.empty and len(df5) >= 60:
        df_m10 = resample_ohlcv(df5, '10min')
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
    df_h4 = ensure_h4(df_h1, df_h4)
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

def round_price(price, pair_config):
    try:
        if price is None:
            return None
        return round(float(price), int(pair_config.get('digits', 2)))
    except Exception:
        return None

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

# =============================================================================
# ── V3 CORE: PERSISTENT STATE ───────────────────────────────────────────────
# Bias and trade plans now survive Streamlit reruns/restarts (JSON file), which
# is what makes hysteresis and "plan lock" real. Previously the bias lived only
# in st.session_state and was lost on every new session.
# =============================================================================
STATE_PATH = os.environ.get('DERAI_STATE_PATH') or 'der_ai_state.json'
BIAS_DIRECTIONAL_THRESHOLD = 25.0   # |bias score| needed for a directional HTF read
BIAS_FLIP_THRESHOLD = 45.0          # |bias score| needed (plus a structure break) to flip a standing bias
BIAS_MIN_HOLD_HOURS = 3.0           # a bias can't flip sooner than this
MAX_LEDGER = 400
LIVE_PLAN_STATUSES = ('PENDING', 'FILLED')
CLOSED_PLAN_STATUSES = ('TP_HIT', 'SL_HIT', 'MISSED', 'EXPIRED', 'CANCELLED', 'STALE')

def utc_now():
    return datetime.now(timezone.utc)

def parse_ts(value):
    try:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    except Exception:
        return None

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj

def _default_state():
    return {'version': 3, 'bias': {}, 'plans': {}, 'ledger': []}

def load_state(force=False):
    """Session cache first; `force=True` re-reads the file (used at the start of every run so
    two browser sessions can't drift apart)."""
    cached = st.session_state.get('persist_state')
    if cached is not None and not force:
        return cached
    state = cached if cached is not None else _default_state()
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r', encoding='utf-8') as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                fresh = _default_state()
                for key in fresh:
                    if isinstance(loaded.get(key), type(fresh[key])):
                        fresh[key] = loaded[key]
                state = fresh
    except Exception as exc:
        print(f"⚠️ State load failed ({STATE_PATH}): {exc}")
    st.session_state.persist_state = state
    return state

def save_state():
    state = load_state()
    try:
        tmp_path = STATE_PATH + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            json.dump(to_jsonable(state), fh)
        os.replace(tmp_path, STATE_PATH)
        return True
    except Exception as exc:
        print(f"⚠️ State save failed ({STATE_PATH}): {exc}")
        return False

def reset_state():
    st.session_state.persist_state = _default_state()
    save_state()

# =============================================================================
# ── V3 CORE: CANDLE HYGIENE & INDICATORS ────────────────────────────────────
# =============================================================================
def infer_bar_minutes(df):
    try:
        if df is None or len(df) < 3:
            return None
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            return None
        return max(1, int(round(diffs.median().total_seconds() / 60.0)))
    except Exception:
        return None

def closed_candles(df):
    """Drop the still-forming last candle. Structure (pivots, order blocks, FVGs, breaks) must
    only ever be computed from CLOSED candles — computing it on the forming bar is exactly why
    entries used to jump every time a fresh high/low printed."""
    if df is None or df.empty or len(df) < 3:
        return df
    mins = infer_bar_minutes(df)
    if not mins:
        return df
    try:
        last_open = df.index[-1]
        last_open = last_open.tz_localize('UTC') if last_open.tzinfo is None else last_open.tz_convert('UTC')
        if utc_now() < last_open.to_pydatetime() + timedelta(minutes=mins):
            return df.iloc[:-1]
    except Exception:
        pass
    return df

def resample_ohlcv(df, rule, origin='start_day'):
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'Volume' in df.columns:
            agg['Volume'] = 'sum'
        out = df.resample(rule, label='left', closed='left', origin=origin).agg(agg)
        return out.dropna(subset=['Open', 'High', 'Low', 'Close'])
    except Exception as exc:
        print(f"⚠️ Resample failed ({rule}): {exc}")
        return pd.DataFrame()

def ensure_h4(h1, h4):
    """Yahoo has no 4h interval and silently falls back to DAILY candles, which makes 'H4'
    structure meaningless. If the H4 frame isn't really ~4h, build it from H1."""
    try:
        mins = infer_bar_minutes(h4) if h4 is not None and not h4.empty else None
        if mins and mins <= 300 and len(h4) >= 30:
            return h4
        if h1 is not None and len(h1) >= 40:
            rebuilt = resample_ohlcv(h1, '4h')
            if len(rebuilt) >= 20:
                return rebuilt
    except Exception:
        pass
    return h4

def _true_range(df):
    high = pd.to_numeric(df['High'], errors='coerce')
    low = pd.to_numeric(df['Low'], errors='coerce')
    close = pd.to_numeric(df['Close'], errors='coerce')
    return pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

def calculate_atr(df, period=14):
    """Wilder ATR (the old version was a simple rolling mean, which reacts differently to spikes)."""
    if df is None or len(df) < period + 2:
        return None
    atr = _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None

def calculate_rsi(series, period=14):
    """Wilder RSI (old version used a simple rolling average and produced different values from
    every charting platform, and had a divergence detector that always returned *something*)."""
    if series is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(pd.Series(series), errors='coerce')
    if len(s) < 2:
        return pd.Series([50.0] * len(s), index=s.index, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    avg_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_dn = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_dn == 0) & (avg_up > 0)), 100.0)
    return rsi.fillna(50.0).astype(float)

def calculate_microstructure(df):
    """VWAP is now anchored to the current UTC day for intraday frames (the old one was
    cumulative over the entire download, i.e. meaningless after a few days)."""
    if df is None or len(df) < 2:
        return {}
    try:
        d = df.copy()
        for col in ['High', 'Low', 'Close']:
            d[col] = pd.to_numeric(d[col], errors='coerce')
        vol = pd.to_numeric(d['Volume'], errors='coerce').fillna(0) if 'Volume' in d.columns else pd.Series(0.0, index=d.index)
        tp = (d['High'] + d['Low'] + d['Close']) / 3
        mins = infer_bar_minutes(d)
        if mins and mins < 240:
            day_start = d.index[-1].normalize()
            sess = d[d.index >= day_start]
            if len(sess) < 6:
                sess = d.tail(48)
        else:
            sess = d.tail(50)
        tps, vs = tp.loc[sess.index], vol.loc[sess.index]
        vwap = float((tps * vs).sum() / vs.sum()) if vs.sum() > 0 else float(tps.mean())
        price = float(d['Close'].iloc[-1])
        avg_vol = float(vol.tail(21).iloc[:-1].mean()) if len(vol) > 2 else 0.0
        rvol = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
        anchor = -min(5, len(d))
        price_change = price - float(d['Close'].iloc[anchor])
        return {
            "vwap": round(vwap, 5 if price < 10 else 2),
            "price_vs_vwap": "ABOVE" if price > vwap else "BELOW",
            "rvol": round(rvol, 2),
            "volume_anomaly": "HIGH_INSTITUTIONAL" if rvol > 2.0 else "NORMAL",
            "momentum": "BULLISH" if price_change > 0 else "BEARISH",
        }
    except Exception:
        return {}

def ema_trend(d, atr):
    try:
        close = pd.to_numeric(d['Close'], errors='coerce')
        e21 = close.ewm(span=21, adjust=False).mean()
        e50 = close.ewm(span=50, adjust=False).mean()
        last = float(close.iloc[-1])
        slope = float(e50.iloc[-1] - e50.iloc[-6]) / atr if len(e50) > 56 and atr else 0.0
        if last > e21.iloc[-1] > e50.iloc[-1] and slope > 0:
            state = 'BULLISH'
        elif last < e21.iloc[-1] < e50.iloc[-1] and slope < 0:
            state = 'BEARISH'
        else:
            state = 'NEUTRAL'
        return {'state': state, 'e21': float(e21.iloc[-1]), 'e50': float(e50.iloc[-1]), 'slope_atr': round(slope, 2)}
    except Exception:
        return {'state': 'NEUTRAL', 'e21': None, 'e50': None, 'slope_atr': 0.0}

# =============================================================================
# ── V3 CORE: MARKET STRUCTURE ENGINE ────────────────────────────────────────
# Confirmed pivots -> ATR-filtered zig-zag -> HH/HL/LH/LL classification ->
# close-based BOS/CHOCH -> mitigation-aware order blocks & FVGs -> liquidity
# pools -> dealing range. Everything works on CLOSED candles only.
# =============================================================================
def find_pivots(df, left=3, right=3):
    highs = df['High'].to_numpy(dtype=float)
    lows = df['Low'].to_numpy(dtype=float)
    idx = df.index
    out = []
    for i in range(left, len(df) - right):
        h = highs[i]
        if h > highs[i - left:i].max() and h >= highs[i + 1:i + right + 1].max():
            out.append({'i': i, 't': idx[i], 'price': float(h), 'type': 'H'})
        l = lows[i]
        if l < lows[i - left:i].min() and l <= lows[i + 1:i + right + 1].min():
            out.append({'i': i, 't': idx[i], 'price': float(l), 'type': 'L'})
    out.sort(key=lambda p: (p['i'], p['type']))
    return out

def _zigzag(pivots, min_leg):
    """Force H/L alternation and drop legs smaller than `min_leg` (ATR-based significance)."""
    out = []
    for p in pivots:
        if not out:
            out.append(p)
            continue
        last = out[-1]
        if p['type'] == last['type']:
            if (p['type'] == 'H' and p['price'] >= last['price']) or (p['type'] == 'L' and p['price'] <= last['price']):
                out[-1] = p
            continue
        if abs(p['price'] - last['price']) < min_leg:
            continue
        out.append(p)
    return out

def _mark_swept(d, pivots):
    """A pivot high that later price has traded above is no longer live liquidity/resistance
    (same for lows). Only un-swept pivots are valid SL anchors or TP targets."""
    highs = d['High'].to_numpy(dtype=float)
    lows = d['Low'].to_numpy(dtype=float)
    for p in pivots:
        after = slice(p['i'] + 1, len(d))
        if p['type'] == 'H':
            p['swept'] = bool(len(highs[after]) and highs[after].max() > p['price'])
        else:
            p['swept'] = bool(len(lows[after]) and lows[after].min() < p['price'])
    return pivots

def classify_structure(pivots):
    highs = [p for p in pivots if p['type'] == 'H'][-3:]
    lows = [p for p in pivots if p['type'] == 'L'][-3:]
    if len(highs) < 2 or len(lows) < 2:
        return 'RANGE', 'insufficient swings'
    hh, lh = highs[-1]['price'] > highs[-2]['price'], highs[-1]['price'] < highs[-2]['price']
    hl, ll = lows[-1]['price'] > lows[-2]['price'], lows[-1]['price'] < lows[-2]['price']
    if hh and hl:
        return 'BULLISH', 'HH + HL'
    if lh and ll:
        return 'BEARISH', 'LH + LL'
    if hh and ll:
        return 'RANGE', 'expanding range (HH + LL)'
    if lh and hl:
        return 'RANGE', 'contracting range (LH + HL)'
    return 'RANGE', 'mixed swings'

def detect_structure_breaks(d, pivots, right):
    """Close-based BOS/CHOCH. A break only counts when a candle CLOSES beyond the last
    confirmed swing (a wick through a level is a liquidity sweep, not a structure break).
    BOS = break in the direction of the standing structure; CHOCH = break against it."""
    closes = d['Close'].to_numpy(dtype=float)
    n = len(d)
    ordered = sorted(pivots, key=lambda p: p['i'])
    pi = 0
    swing_high = swing_low = None
    state = None
    events = []
    for i in range(n):
        while pi < len(ordered) and ordered[pi]['i'] + right <= i:
            p = ordered[pi]
            pi += 1
            if p['type'] == 'H':
                swing_high = {'price': p['price'], 'broken': False}
            else:
                swing_low = {'price': p['price'], 'broken': False}
        if swing_high and not swing_high['broken'] and closes[i] > swing_high['price']:
            kind = 'CHOCH' if state == 'BEARISH' else 'BOS'
            events.append({'i': i, 't': d.index[i], 'dir': 'BULLISH', 'kind': kind, 'level': swing_high['price'], 'age': n - 1 - i})
            state = 'BULLISH'
            swing_high['broken'] = True
        if swing_low and not swing_low['broken'] and closes[i] < swing_low['price']:
            kind = 'CHOCH' if state == 'BULLISH' else 'BOS'
            events.append({'i': i, 't': d.index[i], 'dir': 'BEARISH', 'kind': kind, 'level': swing_low['price'], 'age': n - 1 - i})
            state = 'BEARISH'
            swing_low['broken'] = True
    return {'events': events[-5:], 'last': events[-1] if events else None, 'state': state}

def _overlap_ratio(a_top, a_bottom, b_top, b_bottom):
    inter = min(a_top, b_top) - max(a_bottom, b_bottom)
    if inter <= 0:
        return 0.0
    return inter / max(1e-12, min(a_top - a_bottom, b_top - b_bottom))

def find_order_blocks(d, atr, disp_atr=1.0, impulse_lookback=5, max_age=200, max_keep=6):
    """Institutional order block: the last opposite-colour candle before a DISPLACEMENT candle
    (body >= disp_atr * ATR) that closes beyond the prior `impulse_lookback` candles' extreme.
    Mitigation-aware: a zone whose distal edge has been CLOSED through is dead and dropped; the
    number of times price has re-entered it (`touches`) is tracked because tested zones are weaker.
    The old detector only looked at the last 3 candles, so its 'order blocks' were just the most
    recent candles — they moved every bar."""
    o = d['Open'].to_numpy(dtype=float)
    h = d['High'].to_numpy(dtype=float)
    l = d['Low'].to_numpy(dtype=float)
    c = d['Close'].to_numpy(dtype=float)
    n = len(d)
    zones = []
    for i in range(max(impulse_lookback + 1, n - max_age), n):
        body = abs(c[i] - o[i])
        if body < disp_atr * atr:
            continue
        prev_hi = h[i - impulse_lookback:i].max()
        prev_lo = l[i - impulse_lookback:i].min()
        zone = None
        if c[i] > o[i] and c[i] > prev_hi:
            for k in range(i - 1, i - impulse_lookback - 1, -1):
                if c[k] < o[k]:
                    zone = {'type': 'BULLISH_OB', 'top': h[k], 'bottom': l[k], 'j': k}
                    break
        elif c[i] < o[i] and c[i] < prev_lo:
            for k in range(i - 1, i - impulse_lookback - 1, -1):
                if c[k] > o[k]:
                    zone = {'type': 'BEARISH_OB', 'top': h[k], 'bottom': l[k], 'j': k}
                    break
        if zone is None:
            continue
        j = zone['j']
        if (zone['top'] - zone['bottom']) > 2.0 * atr:  # oversized candle -> use its body only
            zone['top'], zone['bottom'] = max(o[j], c[j]), min(o[j], c[j])
        touches, inside, mitigated = 0, False, False
        for k in range(i + 1, n):
            if zone['type'] == 'BULLISH_OB':
                if c[k] < zone['bottom']:
                    mitigated = True
                    break
                if l[k] <= zone['top']:
                    if not inside:
                        touches += 1
                    inside = True
                else:
                    inside = False
            else:
                if c[k] > zone['top']:
                    mitigated = True
                    break
                if h[k] >= zone['bottom']:
                    if not inside:
                        touches += 1
                    inside = True
                else:
                    inside = False
        if mitigated or zone['top'] <= zone['bottom']:
            continue
        disp = body / atr
        zones.append({
            'type': zone['type'], 'top': float(zone['top']), 'bottom': float(zone['bottom']),
            'mid': float((zone['top'] + zone['bottom']) / 2.0), 'touches': touches,
            'strength': 'STRONG' if disp >= 2.0 else 'MODERATE', 'disp_atr': round(disp, 2),
            'j': j, 't': d.index[j], 'age': n - 1 - j,
        })
    zones.sort(key=lambda z: z['j'], reverse=True)
    kept = []
    for z in zones:
        if any(k['type'] == z['type'] and _overlap_ratio(z['top'], z['bottom'], k['top'], k['bottom']) > 0.5 for k in kept):
            continue
        kept.append(z)
    return kept[:max_keep * 2]

def find_fvgs(d, atr, min_gap_atr=0.25, max_age=200, max_keep=6):
    """Three-candle imbalances that are still (at least partly) UNFILLED. The remaining
    unfilled slice is what's returned as the zone."""
    h = d['High'].to_numpy(dtype=float)
    l = d['Low'].to_numpy(dtype=float)
    c = d['Close'].to_numpy(dtype=float)
    n = len(d)
    out = []
    for i in range(max(2, n - max_age), n):
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap_atr * atr:
            bottom, top = float(h[i - 2]), float(l[i])
            lowest = top
            for k in range(i + 1, n):
                lowest = min(lowest, l[k])
                if lowest <= bottom:
                    break
            if lowest <= bottom:
                continue
            remaining_top = min(top, lowest)
            if remaining_top - bottom < 0.1 * atr:
                continue
            out.append({'type': 'BULLISH_FVG', 'top': remaining_top, 'bottom': bottom, 'mid': (remaining_top + bottom) / 2.0,
                        'touches': 1 if lowest < top else 0, 'i': i, 't': d.index[i], 'age': n - 1 - i})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap_atr * atr:
            top, bottom = float(l[i - 2]), float(h[i])
            highest = bottom
            for k in range(i + 1, n):
                highest = max(highest, h[k])
                if highest >= top:
                    break
            if highest >= top:
                continue
            remaining_bottom = max(bottom, highest)
            if top - remaining_bottom < 0.1 * atr:
                continue
            out.append({'type': 'BEARISH_FVG', 'top': top, 'bottom': remaining_bottom, 'mid': (top + remaining_bottom) / 2.0,
                        'touches': 1 if highest > bottom else 0, 'i': i, 't': d.index[i], 'age': n - 1 - i})
    out.sort(key=lambda z: z['i'], reverse=True)
    return out[:max_keep * 2]

def find_liquidity_pools(pivots, atr, tol_atr=0.2, max_pivots=10):
    """Equal highs / equal lows (2+ un-swept pivots within tolerance) = resting stop liquidity."""
    result = {'eqh': [], 'eql': []}
    for ptype, key in (('H', 'eqh'), ('L', 'eql')):
        pts = [p for p in pivots if p['type'] == ptype and not p.get('swept')][-max_pivots:]
        used = set()
        for a in range(len(pts)):
            if a in used:
                continue
            group = [a]
            for b in range(a + 1, len(pts)):
                if b not in used and abs(pts[b]['price'] - pts[a]['price']) <= tol_atr * atr:
                    group.append(b)
            if len(group) >= 2:
                used.update(group)
                prices = [pts[g]['price'] for g in group]
                result[key].append({'level': max(prices) if ptype == 'H' else min(prices), 'count': len(group)})
    return result

def dealing_range(d, pivots, atr):
    highs = [p['price'] for p in pivots if p['type'] == 'H'][-2:]
    lows = [p['price'] for p in pivots if p['type'] == 'L'][-2:]
    if highs and lows:
        hi, lo = max(highs), min(lows)
    else:
        hi, lo = float(d['High'].tail(100).max()), float(d['Low'].tail(100).min())
    hi = max(hi, float(d['High'].tail(3).max()))
    lo = min(lo, float(d['Low'].tail(3).min()))
    if hi - lo < 2.0 * atr:
        hi, lo = float(d['High'].tail(100).max()), float(d['Low'].tail(100).min())
    rng = max(hi - lo, 1e-9)
    return {
        'high': float(hi), 'low': float(lo), 'eq': float((hi + lo) / 2.0),
        'ote_buy': (float(hi - 0.79 * rng), float(hi - 0.62 * rng)),    # (bottom, top) retracement 62-79% from the high
        'ote_sell': (float(lo + 0.62 * rng), float(lo + 0.79 * rng)),   # (bottom, top) retracement 62-79% from the low
    }

def range_pos(rng, price):
    try:
        span = rng['high'] - rng['low']
        return (float(price) - rng['low']) / span if span > 0 else 0.5
    except Exception:
        return 0.5

def classify_candles(d, lookback=10):
    """Per-candle pattern read (STRONG_BULLISH/BEARISH, DOJI, REJECTION_HIGH/LOW, HAMMER/
    INVERTED_HAMMER, SHOOTING_STAR/HANGING_MAN) over the last `lookback` CLOSED candles. Used
    only as an LTF confirmation input (is a pullback actually reversing right now) — it has no
    vote on HTF direction."""
    if d is None or len(d) < 3:
        return []
    o = d['Open'].to_numpy(dtype=float)
    h = d['High'].to_numpy(dtype=float)
    l = d['Low'].to_numpy(dtype=float)
    c = d['Close'].to_numpy(dtype=float)
    n = len(d)
    out = []
    for i in range(max(0, n - lookback), n):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        upper = h[i] - max(o[i], c[i])
        lower = min(o[i], c[i]) - l[i]
        body_ratio, upper_ratio, lower_ratio = body / rng, upper / rng, lower / rng
        ctype = 'BULLISH' if c[i] > o[i] else 'BEARISH' if c[i] < o[i] else 'DOJI'
        if body_ratio > 0.7 and ctype != 'DOJI':
            pattern = 'STRONG_' + ctype
        elif body_ratio < 0.3:
            pattern = 'DOJI'
        elif upper_ratio > 0.6:
            pattern = 'REJECTION_HIGH'
        elif lower_ratio > 0.6:
            pattern = 'REJECTION_LOW'
        elif upper_ratio > 0.4 and body_ratio < 0.4:
            pattern = 'SHOOTING_STAR' if ctype == 'BEARISH' else 'HANGING_MAN'
        elif lower_ratio > 0.4 and body_ratio < 0.4:
            pattern = 'HAMMER' if ctype == 'BULLISH' else 'INVERTED_HAMMER'
        else:
            pattern = 'NORMAL'
        out.append({'t': d.index[i], 'type': ctype, 'pattern': pattern,
                    'body_ratio': round(body_ratio, 2), 'upper_wick_ratio': round(upper_ratio, 2),
                    'lower_wick_ratio': round(lower_ratio, 2), 'close': float(c[i])})
    return out

def candle_pattern_signal(direction, candles, max_pts=2.0):
    """Does the most recent CLOSED candle actually support this direction right now? Wick-ratio
    based (robust) with the named pattern kept for the human-readable explanation. Returns a
    score in [0, max_pts] plus a one-line detail string for transparency in the reasoning/notes."""
    if not candles:
        return {'score': max_pts * 0.4, 'detail': None}
    last = candles[-1]
    buy = direction == 'BUY'
    wick_support = last['lower_wick_ratio'] if buy else last['upper_wick_ratio']
    wick_oppose = last['upper_wick_ratio'] if buy else last['lower_wick_ratio']
    strong_support = last['pattern'] == ('STRONG_BULLISH' if buy else 'STRONG_BEARISH')
    strong_oppose = last['pattern'] == ('STRONG_BEARISH' if buy else 'STRONG_BULLISH')
    if strong_support or wick_support >= 0.5:
        return {'score': max_pts, 'detail': f"last candle {last['pattern']} supports {direction} (wick {wick_support:.0%})"}
    if strong_oppose or wick_oppose >= 0.5:
        return {'score': 0.0, 'detail': f"last candle {last['pattern']} contradicts {direction} (wick {wick_oppose:.0%})"}
    return {'score': max_pts * 0.4, 'detail': f"last candle {last['pattern']} is neutral"}

def analyze_structure(df, label, left=3, right=3, leg_atr=1.0, ob_disp_atr=1.0, fvg_min_atr=0.25, liq_tol_atr=0.2):
    """One consistent structural snapshot for a timeframe, built from CLOSED candles only."""
    d = closed_candles(df)
    if d is None or len(d) < 40:
        return None
    try:
        d = d.tail(600).copy()
        atr = calculate_atr(d)
        if not atr or atr <= 0:
            return None
        piv = _zigzag(find_pivots(d, left, right), atr * leg_atr)
        piv = _mark_swept(d, piv)
        trend, pattern = classify_structure(piv)
        breaks = detect_structure_breaks(d, piv, right)
        return {
            'label': label, 'df': d, 'atr': atr, 'close': float(d['Close'].iloc[-1]),
            'pivots': piv, 'trend': trend, 'pattern': pattern, 'breaks': breaks,
            'order_blocks': find_order_blocks(d, atr, disp_atr=ob_disp_atr),
            'fvgs': find_fvgs(d, atr, min_gap_atr=fvg_min_atr),
            'liquidity': find_liquidity_pools(piv, atr, tol_atr=liq_tol_atr),
            'ema': ema_trend(d, atr),
            'rsi': compute_rsi_last(d['Close']),
            'range': dealing_range(d, piv, atr),
            'micro': calculate_microstructure(d),
            'candles': classify_candles(d),
        }
    except Exception as exc:
        print(f"⚠️ analyze_structure({label}) failed: {exc}")
        traceback.print_exc()
        return None

def prior_period_levels(h1_df):
    """Previous-day and previous-week high/low from H1 — the levels the market reliably reacts to."""
    levels = {}
    try:
        d = closed_candles(h1_df)
        if d is None or len(d) < 30:
            return levels
        idx = d.index.tz_convert('UTC') if d.index.tz is not None else d.index.tz_localize('UTC')
        days = d.groupby(idx.normalize())
        keys = sorted(days.groups.keys())
        if len(keys) >= 2:
            prev = days.get_group(keys[-2])
            levels['PDH'], levels['PDL'] = float(prev['High'].max()), float(prev['Low'].min())
        weeks = d.groupby(idx.tz_localize(None).to_period('W'))
        wkeys = sorted(weeks.groups.keys())
        if len(wkeys) >= 2:
            prev = weeks.get_group(wkeys[-2])
            levels['PWH'], levels['PWL'] = float(prev['High'].max()), float(prev['Low'].min())
    except Exception as exc:
        print(f"⚠️ prior_period_levels failed: {exc}")
    return levels

def compute_dxy_context(all_data):
    """DXY direction from H1 + H4 EMA trend (replaces the cumulative-VWAP read)."""
    try:
        dxy = all_data.get('DXY', {}) or {}
        h1 = closed_candles(dxy.get('H1'))
        h4 = closed_candles(ensure_h4(dxy.get('H1'), dxy.get('H4')))
        states = []
        for frame in (h1, h4):
            if frame is not None and not frame.empty and len(frame) >= 60:
                states.append(ema_trend(frame, calculate_atr(frame) or 1.0)['state'])
        if not states:
            return {'direction': 'NEUTRAL', 'detail': 'DXY data unavailable'}
        if all(s == 'BULLISH' for s in states):
            direction = 'BULLISH'
        elif all(s == 'BEARISH' for s in states):
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        last = float(h1['Close'].iloc[-1]) if h1 is not None and not h1.empty else None
        return {'direction': direction, 'detail': f"DXY last {last} | H1/H4 EMA trend: {', '.join(states)}"}
    except Exception:
        return {'direction': 'NEUTRAL', 'detail': 'DXY data unavailable'}

def dxy_relation(symbol, direction, dxy):
    if symbol not in ('XAUUSD', 'EURUSD', 'BTCUSD') or not dxy:
        return 'NEUTRAL'
    d = dxy.get('direction')
    if d == 'NEUTRAL':
        return 'NEUTRAL'
    supportive = (direction == 'BUY' and d == 'BEARISH') or (direction == 'SELL' and d == 'BULLISH')
    return 'CONFIRMING' if supportive else 'CONTRADICTING'


# =============================================================================
# ── V3 CORE: PAIR CONFIG ────────────────────────────────────────────────────
# =============================================================================
def get_pair_config(symbol):
    """Entry band, stop and target rules are now sized from the H1 ATR (the timeframe the thesis
    lives on). The old band (max ~0.85% / 450pts on BTC, 0.9 ATR of M15) rejected almost every
    genuine HTF zone and pushed entries onto whatever M15 candle had just printed."""
    base = {
        'digits': 2, 'tick_size': 0.01,
        'min_rr': 1.5, 'max_rr': 4.0,
        'htf_min_gap_atr': 0.15, 'htf_max_gap_atr': 3.0, 'htf_max_gap_pct': 0.012,
        'min_stop_atr': 0.8, 'max_stop_atr': 3.0, 'max_risk_pct': 0.012,
        'stop_buffer_atr': 0.30, 'tp_buffer_atr': 0.15,
        'plan_expiry_hours': 18,
        'ob_disp_atr': 1.0, 'fvg_min_atr': 0.25, 'liq_tol_atr': 0.2,
        'default_pullback_atr': 0.6, 'min_dist_pct': 0.0015,
        'score_floor': MINIMUM_CONFLUENCE_SCORE,
    }
    overrides = {
        'XAUUSD': {'htf_max_gap_pct': 0.012, 'max_risk_pct': 0.012},
        'EURUSD': {'digits': 5, 'tick_size': 0.00001, 'htf_max_gap_pct': 0.007, 'max_risk_pct': 0.006, 'min_dist_pct': 0.0008},
        'BTCUSD': {'htf_max_gap_pct': 0.025, 'max_risk_pct': 0.03},
        'US30': {'digits': 1, 'tick_size': 0.1, 'htf_max_gap_pct': 0.012, 'max_risk_pct': 0.012},
    }
    return {**base, **overrides.get(symbol, {})}

def htf_entry_gap_bounds(price, atr_h1, cfg):
    min_gap = max(float(cfg['tick_size']) * 5.0, float(atr_h1) * float(cfg['htf_min_gap_atr']), float(price) * 0.0002)
    max_gap = min(float(atr_h1) * float(cfg['htf_max_gap_atr']), float(price) * float(cfg['htf_max_gap_pct']))
    if max_gap <= min_gap:
        max_gap = min_gap * 3.0
    return min_gap, max_gap

def compute_fallback_atr_plan_levels(direction, price, atr_h1, cfg):
    """Only used when NO structural plan is valid. Always low-scored, never pushed to Telegram."""
    min_gap, max_gap = htf_entry_gap_bounds(price, atr_h1, cfg)
    gap = min(max(atr_h1 * float(cfg['default_pullback_atr']), min_gap), max_gap)
    entry = price - gap if direction == 'BUY' else price + gap
    return entry

# =============================================================================
# ── V3 CORE: HTF BIAS (H4 + H1 ONLY) WITH PERSISTENT HYSTERESIS ─────────────
# The LTFs (M10/M15/M30) have NO vote in direction any more. The old MTF score gave M10/M15/M30
# a combined weight of 3.5 vs 5.0 for H1/H4, so ten-minute noise regularly out-voted the
# structure that actually matters — that is how a BUY at 20:53 became a SELL at 00:03.
# =============================================================================
def _snap_bias_score(snap):
    """-100..+100 directional score for one HTF snapshot, plus the evidence behind it."""
    if not snap:
        return 0.0, []
    label = snap['label']
    score, notes = 0.0, []
    if snap['trend'] == 'BULLISH':
        score += 20
        notes.append(f"{label} structure {snap['pattern']}")
    elif snap['trend'] == 'BEARISH':
        score -= 20
        notes.append(f"{label} structure {snap['pattern']}")
    else:
        notes.append(f"{label} structure ranging ({snap['pattern']})")
    state = snap['breaks']['state']
    if state == 'BULLISH':
        score += 14
    elif state == 'BEARISH':
        score -= 14
    last = snap['breaks']['last']
    if last:
        notes.append(f"{label} last close-based break: {last['kind']} {last['dir'].lower()} at {last['level']:.5g} ({last['age']} bars ago)")
        if last['kind'] == 'CHOCH' and last['age'] <= 24:
            score += 6 if last['dir'] == 'BULLISH' else -6
    ema_state = snap['ema']['state']
    if ema_state == 'BULLISH':
        score += 12
        notes.append(f"{label} price>EMA21>EMA50")
    elif ema_state == 'BEARISH':
        score -= 12
        notes.append(f"{label} price<EMA21<EMA50")
    rsi = snap.get('rsi')
    if rsi is not None:
        if rsi > 55:
            score += 6
        elif rsi < 45:
            score -= 6
        notes.append(f"{label} RSI {rsi}")
    return score / 58.0 * 100.0, notes

def _opposing_break_since(snaps, dir_word, since):
    for label in ('H1', 'H4'):
        snap = snaps.get(label)
        last = ((snap or {}).get('breaks') or {}).get('last')
        if last and last['dir'] == dir_word:
            ts = parse_ts(last['t'])
            if ts is not None and since is not None and ts > since:
                return f"{label} {last['kind']} ({dir_word.lower()})"
    return None

def resolve_htf_bias(symbol, snaps, dxy, persist=True):
    h4, h1 = snaps.get('H4'), snaps.get('H1')
    s4, n4 = _snap_bias_score(h4)
    s1, n1 = _snap_bias_score(h1)
    if h4 and h1:
        agg = (1.5 * s4 + 1.0 * s1) / 2.5
    elif h1:
        agg = s1
    elif h4:
        agg = s4
    else:
        agg = 0.0
    dxy_adj = 0.0
    if symbol in ('XAUUSD', 'EURUSD', 'BTCUSD') and dxy:
        dxy_adj = -6.0 if dxy.get('direction') == 'BULLISH' else (6.0 if dxy.get('direction') == 'BEARISH' else 0.0)
    agg = max(-100.0, min(100.0, agg + dxy_adj))
    raw = 'BUY' if agg >= BIAS_DIRECTIONAL_THRESHOLD else ('SELL' if agg <= -BIAS_DIRECTIONAL_THRESHOLD else None)
    aligned = bool(h4 and h1 and s4 * s1 > 0 and abs(s4) >= 20 and abs(s1) >= 20)

    state = load_state()
    prev = state['bias'].get(symbol)
    now = utc_now()
    notes = []
    held = False
    low_conviction = False
    direction = None
    since = now
    if prev and prev.get('direction') in ('BUY', 'SELL'):
        prev_since = parse_ts(prev.get('since')) or now
        hours = (now - prev_since).total_seconds() / 3600.0
        since = prev_since
        if raw is None:
            direction, held = prev['direction'], True
            notes.append(f"HTF read is mixed ({agg:+.0f}); holding the standing {prev['direction']} bias set {prev_since.strftime('%d %b %H:%M')} UTC")
        elif raw == prev['direction']:
            direction = raw
        else:
            dir_word = 'BULLISH' if raw == 'BUY' else 'BEARISH'
            evidence = _opposing_break_since(snaps, dir_word, prev_since)
            if abs(agg) >= BIAS_FLIP_THRESHOLD and evidence and hours >= BIAS_MIN_HOLD_HOURS:
                direction, since = raw, now
                notes.append(f"BIAS FLIPPED {prev['direction']} -> {raw}: score {agg:+.0f} and {evidence}")
            else:
                direction, held = prev['direction'], True
                why = []
                if abs(agg) < BIAS_FLIP_THRESHOLD:
                    why.append(f"score {agg:+.0f} is short of the ±{BIAS_FLIP_THRESHOLD:.0f} flip threshold")
                if not evidence:
                    why.append('no opposing H1/H4 close-based structure break yet')
                if hours < BIAS_MIN_HOLD_HOURS:
                    why.append(f"bias is only {hours:.1f}h old (minimum {BIAS_MIN_HOLD_HOURS:.0f}h)")
                notes.append(f"Opposing {raw} read rejected as noise ({'; '.join(why)}) - standing {direction} bias retained")
    else:
        if raw:
            direction = raw
        else:
            low_conviction = True
            if abs(s4) > 1e-9:
                direction = 'BUY' if s4 > 0 else 'SELL'
                notes.append('HTF mixed; tie-broken by the H4 macro read (low conviction)')
            elif abs(s1) > 1e-9:
                direction = 'BUY' if s1 > 0 else 'SELL'
                notes.append('HTF mixed; tie-broken by the H1 read (low conviction)')
            else:
                ref = (h4 or h1 or {}).get('range')
                pos = range_pos(ref, (h1 or h4)['close']) if ref else 0.5
                direction = 'SELL' if pos >= 0.5 else 'BUY'
                notes.append('HTF flat; tie-broken by premium/discount location (low conviction)')
    if direction is None:
        direction = 'BUY'
    if persist and direction:
        state['bias'][symbol] = {'direction': direction, 'since': since.isoformat(), 'score': round(agg, 1), 'updated': now.isoformat()}
    return {
        'direction': direction, 'raw_direction': raw, 'score': round(agg, 1),
        'h4_score': round(s4, 1), 'h1_score': round(s1, 1), 'aligned': aligned, 'held': held,
        'low_conviction': low_conviction, 'since': since.isoformat(), 'dxy_adj': dxy_adj,
        'evidence': n4 + n1, 'notes': notes,
        'word': 'BULLISH' if direction == 'BUY' else 'BEARISH',
    }

# =============================================================================
# ── V3 CORE: HTF ENTRY ZONES -> BEST LIMIT PRICE ────────────────────────────
# =============================================================================
def _zone(kind, tf, top, bottom, weight, touches=0):
    top, bottom = float(max(top, bottom)), float(min(top, bottom))
    return {'kind': kind, 'tf': tf, 'top': top, 'bottom': bottom, 'mid': (top + bottom) / 2.0,
            'weight': float(weight), 'touches': int(touches)}

def collect_entry_zones(direction, snaps, levels, atr_h1, cfg):
    """Every HTF point of interest on the correct side of price that a limit order could rest in."""
    buy = direction == 'BUY'
    want_ob = 'BULLISH_OB' if buy else 'BEARISH_OB'
    want_fvg = 'BULLISH_FVG' if buy else 'BEARISH_FVG'
    word = 'demand' if buy else 'supply'
    zones = []
    for label, tfw in (('H4', 3.0), ('H1', 2.0)):
        snap = snaps.get(label)
        if not snap:
            continue
        for ob in snap['order_blocks']:
            if ob['type'] == want_ob:
                w = tfw * (1.25 if ob['strength'] == 'STRONG' else 1.0)
                zones.append(_zone(f"{label} {word} OB", label, ob['top'], ob['bottom'], w, ob['touches']))
        for fvg in snap['fvgs']:
            if fvg['type'] == want_fvg:
                zones.append(_zone(f"{label} FVG", label, fvg['top'], fvg['bottom'], tfw * 0.75, fvg['touches']))
        lo, hi = snap['range']['ote_buy'] if buy else snap['range']['ote_sell']
        zones.append(_zone(f"{label} OTE 62-79%", label, hi, lo, tfw * 0.7, 0))
        want_type = 'L' if buy else 'H'
        live = [p for p in snap['pivots'] if p['type'] == want_type and not p.get('swept')][-3:]
        for p in live:
            zones.append(_zone(f"{label} swing {'low' if buy else 'high'}", label, p['price'] + 0.15 * atr_h1, p['price'] - 0.15 * atr_h1, tfw * 0.5, 0))
        pool_key = 'eql' if buy else 'eqh'
        for pool in snap['liquidity'][pool_key][-2:]:
            zones.append(_zone(f"{label} equal {'lows' if buy else 'highs'} x{pool['count']}", label,
                               pool['level'] + 0.10 * atr_h1, pool['level'] - 0.25 * atr_h1, tfw * 0.5 + 0.4 * pool['count'], 0))
    for key, w in (('PDL' if buy else 'PDH', 1.0), ('PWL' if buy else 'PWH', 1.3)):
        if key in levels:
            zones.append(_zone(f"{key} liquidity", 'H1', levels[key] + 0.10 * atr_h1, levels[key] - 0.25 * atr_h1, w, 0))
    return zones

def _entry_point_in_zone(direction, zone, price, min_gap):
    """Mean-threshold (50%) of the zone, pulled inside the zone/gap rules when price is close."""
    entry = zone['mid']
    if direction == 'BUY':
        if entry > price - min_gap:
            entry = min(zone['top'], price - min_gap)
            if entry < zone['bottom']:
                return None
    else:
        if entry < price + min_gap:
            entry = max(zone['bottom'], price + min_gap)
            if entry > zone['top']:
                return None
    return float(entry)

def rank_entry_zones(direction, price, zones, snaps, atr_h1, cfg):
    min_gap, max_gap = htf_entry_gap_bounds(price, atr_h1, cfg)
    buy = direction == 'BUY'
    ref = (snaps.get('H4') or snaps.get('H1'))['range']
    ranked = []
    for z in zones:
        entry = _entry_point_in_zone(direction, z, price, min_gap)
        if entry is None:
            continue
        dist = (price - entry) if buy else (entry - price)
        if dist < min_gap * 0.999 or dist > max_gap:
            continue
        overlaps = [o for o in zones if o is not z and _overlap_or_near(z, o, 0.25 * atr_h1)]
        conf_w = sum(o['weight'] for o in overlaps) * 0.5
        fresh = 1.0 if z['touches'] == 0 else (0.8 if z['touches'] == 1 else 0.55)
        pos = range_pos(ref, entry)
        location_bonus = 0.0
        if (buy and pos <= 0.5) or ((not buy) and pos >= 0.5):
            location_bonus += 1.5
        lo, hi = ref['ote_buy'] if buy else ref['ote_sell']
        if lo - 0.1 * atr_h1 <= entry <= hi + 0.1 * atr_h1:
            location_bonus += 1.0
        dist_atr = dist / atr_h1 if atr_h1 else 0.0
        penalty = 0.6 * max(0.0, dist_atr - 1.0)
        total = (z['weight'] + conf_w) * fresh + location_bonus - penalty
        ranked.append({**z, 'entry': float(entry), 'dist': float(dist), 'dist_atr': round(dist_atr, 2),
                       'confluence': sorted({o['kind'] for o in overlaps}), 'zone_score': round(total, 2),
                       'range_pos': round(pos, 2)})
    ranked.sort(key=lambda r: r['zone_score'], reverse=True)
    dedup = []
    for r in ranked:
        if any(abs(r['entry'] - k['entry']) < 0.2 * atr_h1 for k in dedup):
            continue
        dedup.append(r)
    return dedup

def _overlap_or_near(a, b, tol):
    return (a['bottom'] - tol) <= b['top'] and (b['bottom'] - tol) <= a['top']

def refine_entry_with_ltf(direction, zone, ltf_snaps, atr_h1, price, min_gap):
    """Inside the chosen HTF zone, snap the limit to a genuine M30/M15 level (OB/FVG/swing) if one
    exists — the 'compare HTF entry with LTF' step. LTF can only refine WITHIN the zone; it can't
    move the order outside it."""
    lo, hi = zone['bottom'], zone['top']
    tol = 0.1 * atr_h1
    buy = direction == 'BUY'
    want_ob = 'BULLISH_OB' if buy else 'BEARISH_OB'
    want_fvg = 'BULLISH_FVG' if buy else 'BEARISH_FVG'
    want_piv = 'L' if buy else 'H'
    cands = []
    for label, snap in ltf_snaps:
        if not snap:
            continue
        for ob in snap['order_blocks']:
            if ob['type'] == want_ob and ob['bottom'] <= hi + tol and ob['top'] >= lo - tol:
                cands.append((min(max(ob['mid'], lo), hi), f"{label} OB"))
        for fvg in snap['fvgs']:
            if fvg['type'] == want_fvg and fvg['bottom'] <= hi + tol and fvg['top'] >= lo - tol:
                cands.append((min(max(fvg['mid'], lo), hi), f"{label} FVG"))
        for p in snap['pivots']:
            if p['type'] == want_piv and not p.get('swept') and lo - tol <= p['price'] <= hi + tol:
                cands.append((min(max(p['price'], lo), hi), f"{label} swing"))
    if not cands:
        return None, []
    ref_price, _ = min(cands, key=lambda c: abs(c[0] - zone['mid']))
    dist = (price - ref_price) if buy else (ref_price - price)
    if dist < min_gap:
        return None, []
    near = sorted({name for p, name in cands if abs(p - ref_price) <= 0.15 * atr_h1})
    return float(ref_price), near

def collect_targets(direction, entry, snaps, levels, atr_h1):
    """Opposing HTF liquidity/zones ahead of entry, nearest first."""
    buy = direction == 'BUY'
    out = []
    for label in ('H1', 'H4'):
        snap = snaps.get(label)
        if not snap:
            continue
        for ob in snap['order_blocks']:
            if buy and ob['type'] == 'BEARISH_OB' and ob['bottom'] > entry:
                out.append({'price': ob['bottom'], 'kind': f"{label} supply OB"})
            if (not buy) and ob['type'] == 'BULLISH_OB' and ob['top'] < entry:
                out.append({'price': ob['top'], 'kind': f"{label} demand OB"})
        for fvg in snap['fvgs']:
            if buy and fvg['type'] == 'BEARISH_FVG' and fvg['bottom'] > entry:
                out.append({'price': fvg['bottom'], 'kind': f"{label} bearish FVG"})
            if (not buy) and fvg['type'] == 'BULLISH_FVG' and fvg['top'] < entry:
                out.append({'price': fvg['top'], 'kind': f"{label} bullish FVG"})
        for p in snap['pivots']:
            if p.get('swept'):
                continue
            if buy and p['type'] == 'H' and p['price'] > entry:
                out.append({'price': p['price'], 'kind': f"{label} swing high"})
            if (not buy) and p['type'] == 'L' and p['price'] < entry:
                out.append({'price': p['price'], 'kind': f"{label} swing low"})
        pool = snap['liquidity']['eqh' if buy else 'eql']
        for item in pool:
            if (buy and item['level'] > entry) or ((not buy) and item['level'] < entry):
                out.append({'price': item['level'], 'kind': f"{label} equal {'highs' if buy else 'lows'}"})
        rng = snap['range']
        edge = rng['high'] if buy else rng['low']
        if (buy and edge > entry) or ((not buy) and edge < entry):
            out.append({'price': edge, 'kind': f"{label} range {'high' if buy else 'low'}"})
    for key in (('PDH', 'PWH') if buy else ('PDL', 'PWL')):
        if key in levels and ((buy and levels[key] > entry) or ((not buy) and levels[key] < entry)):
            out.append({'price': levels[key], 'kind': key})
    out.sort(key=lambda t: abs(t['price'] - entry))
    merged = []
    for t in out:
        if merged and abs(t['price'] - merged[-1]['price']) < 0.2 * atr_h1:
            if t['kind'] not in merged[-1]['kind']:
                merged[-1]['kind'] += f" + {t['kind']}"
            continue
        merged.append(dict(t))
    return merged

def build_plan_for_entry(direction, entry, zone, snaps, levels, atr_h1, cfg, ltf_names=None):
    """SL beyond the zone's distal edge (and any live H1/H4 swing just beyond it) + ATR buffer;
    TP1 = first opposing HTF level that pays >= min_rr; TP2 = the next one."""
    buy = direction == 'BUY'
    tick = float(cfg['tick_size'])
    buffer = max(atr_h1 * float(cfg['stop_buffer_atr']), tick * 3.0)
    tp_buffer = max(atr_h1 * float(cfg['tp_buffer_atr']), tick * 2.0)
    min_risk = max(entry * float(cfg['min_dist_pct']), atr_h1 * float(cfg['min_stop_atr']))
    max_risk = min(entry * float(cfg['max_risk_pct']), atr_h1 * float(cfg['max_stop_atr']))
    if max_risk < min_risk:
        max_risk = min_risk * 1.5
    anchor = min(zone['bottom'], entry) if buy else max(zone['top'], entry)
    anchor_note = f"{zone['kind']} distal edge"
    want_type = 'L' if buy else 'H'
    near = []
    for label in ('H1', 'H4'):
        snap = snaps.get(label)
        for p in ((snap or {}).get('pivots') or []):
            if p['type'] == want_type and not p.get('swept'):
                if buy and anchor - 0.6 * atr_h1 <= p['price'] < anchor:
                    near.append(p['price'])
                if (not buy) and anchor < p['price'] <= anchor + 0.6 * atr_h1:
                    near.append(p['price'])
    if near:
        anchor = min(near) if buy else max(near)
        anchor_note = f"live H1/H4 swing {'low' if buy else 'high'} just beyond the zone"
    sl = anchor - buffer if buy else anchor + buffer
    risk = (entry - sl) if buy else (sl - entry)
    if risk < min_risk:
        sl = entry - min_risk if buy else entry + min_risk
        risk = min_risk
        anchor_note += ' (widened to the minimum HTF-ATR stop)'
    if risk > max_risk:
        return None, f"stop needs {risk:.5g} risk, above the {max_risk:.5g} cap"
    targets = collect_targets(direction, entry, snaps, levels, atr_h1)
    min_rr, max_rr = float(cfg['min_rr']), float(cfg['max_rr'])
    tp1 = tp1_kind = tp2 = tp2_kind = None
    idx_used = None
    for n, t in enumerate(targets):
        price = t['price'] - tp_buffer if buy else t['price'] + tp_buffer
        reward = (price - entry) if buy else (entry - price)
        if reward / risk >= min_rr:
            tp1, tp1_kind, idx_used = price, t['kind'], n
            break
    tp_structural = True
    if tp1 is None:
        reward = risk * min_rr * 1.15
        tp1 = entry + reward if buy else entry - reward
        tp1_kind, tp_structural = 'RR projection (no HTF level pays the minimum R:R)', False
    else:
        rr1 = abs(tp1 - entry) / risk
        if rr1 > max_rr:
            tp1 = entry + risk * max_rr if buy else entry - risk * max_rr
            tp1_kind += ' (capped at max R:R)'
    tp2_structural = False
    if idx_used is not None:
        for t in targets[idx_used + 1:]:
            price = t['price'] - tp_buffer if buy else t['price'] + tp_buffer
            if (buy and price > tp1 + 0.2 * atr_h1) or ((not buy) and price < tp1 - 0.2 * atr_h1):
                tp2, tp2_kind, tp2_structural = price, t['kind'], True
                break
    if tp2 is None:
        extra = risk * max(abs(tp1 - entry) / risk + 1.0, 3.0)
        extra = min(extra, risk * (max_rr + 1.0))
        tp2 = entry + extra if buy else entry - extra
        tp2_kind = 'R-multiple projection'
    rr = abs(tp1 - entry) / risk
    plan = {
        'direction': direction,
        'entry': round_price(entry, cfg), 'stop_loss': round_price(sl, cfg),
        'take_profit': [round_price(tp1, cfg), round_price(tp2, cfg)],
        'rr_ratio': round(rr, 2), 'risk': float(risk),
        'sl_anchor': anchor_note, 'tp1_kind': tp1_kind, 'tp2_kind': tp2_kind,
        'tp1_structural': tp_structural, 'tp2_structural': tp2_structural,
        'ltf_refs': ltf_names or [],
        'order_type': 'LIMIT',
    }
    return plan, 'ok'

def score_plan(direction, plan, zone, bias, snaps, dxy_rel, regime, ltf_snaps, price, atr_h1):
    """100-point transparent score. Every component is reported so a low score is explainable."""
    buy = direction == 'BUY'
    b = {}
    dir_norm = bias['score'] if buy else -bias['score']
    align = 25.0 * max(0.0, min(1.0, dir_norm / 60.0))
    if not bias['aligned']:
        align *= 0.7
    if bias['held']:
        align *= 0.6
    if bias['low_conviction']:
        align *= 0.4
    b['HTF alignment (H4+H1)'] = round(align, 1)
    b['Zone quality & confluence'] = round(min(25.0, zone.get('zone_score', 0) * 4.0), 1)
    ref = (snaps.get('H4') or snaps.get('H1'))['range']
    pos = range_pos(ref, plan['entry'])
    depth = pos if buy else 1.0 - pos
    b['Premium/discount location'] = 15.0 if depth <= 0.35 else 11.0 if depth <= 0.5 else 6.0 if depth <= 0.62 else 2.0
    ltf = 0.0
    candle_notes = []
    if plan.get('ltf_refs'):
        ltf += 4.0
    m15 = next((s for lbl, s in ltf_snaps if lbl == 'M15' and s), None)
    if m15:
        e21 = m15['ema'].get('e21')
        if e21 is not None:
            pulling = (m15['close'] < e21) if buy else (m15['close'] > e21)
            if pulling and abs(price - plan['entry']) <= 2.0 * atr_h1:
                ltf += 2.0
        rv = (m15.get('micro') or {}).get('rvol', 1.0)
        if rv <= 2.5:
            ltf += 1.0
        m15_cp = candle_pattern_signal(direction, m15.get('candles'), max_pts=2.0)
        ltf += m15_cp['score']
        if m15_cp['detail']:
            candle_notes.append(f"M15 {m15_cp['detail']}")
    m10_candles = snaps.get('_candles_m10')
    if m10_candles:
        m10_cp = candle_pattern_signal(direction, m10_candles, max_pts=1.0)
        ltf += m10_cp['score']
        if m10_cp['detail']:
            candle_notes.append(f"M10 {m10_cp['detail']}")
    b['LTF confirmation'] = round(min(10.0, ltf), 1)
    b['Risk:reward'] = round(min(10.0, 4.0 + max(0.0, plan['rr_ratio'] - 1.5) * 4.0), 1)
    macro = {'CONFIRMING': 5.0, 'NEUTRAL': 3.0, 'CONTRADICTING': 0.0}.get(dxy_rel, 3.0)
    reg = (regime or {}).get('regime')
    want_mom = 'BULLISH' if buy else 'BEARISH'
    if reg == 'TRENDING':
        macro += 5.0 if (regime or {}).get('trend_direction') == want_mom else 1.0
    elif reg == 'TRANSITIONAL':
        macro += 3.0
    else:
        macro += 2.0
    b['Macro (DXY) & regime'] = round(macro, 1)
    b['Structural targets'] = (3.0 if plan.get('tp1_structural') else 0.0) + (2.0 if plan.get('tp2_structural') else 0.0)
    total = int(round(sum(b.values())))
    return max(0, min(100, total)), b, candle_notes

def grade_for(score):
    return 'A' if score >= 85 else 'B' if score >= 75 else 'C' if score >= 65 else 'D'

def confidence_for(score):
    return 'HIGH' if score >= 82 else 'MEDIUM' if score >= 72 else 'LOW'

def select_best_plan(direction, price, snaps, levels, atr_h1, cfg, bias, dxy_rel, regime, max_eval=8):
    """Rank HTF zones, refine each with LTF, build SL/TP, score, and return the best VALID plan
    plus a full audit trail of every candidate (so you can see why one zone beat the others)."""
    zones = collect_entry_zones(direction, snaps, levels, atr_h1, cfg)
    ranked = rank_entry_zones(direction, price, zones, snaps, atr_h1, cfg)
    min_gap, _ = htf_entry_gap_bounds(price, atr_h1, cfg)
    ltf_snaps = [('M30', snaps.get('M30')), ('M15', snaps.get('M15'))]
    audit, valid = [], []
    for z in ranked[:max_eval]:
        entry, names = z['entry'], []
        refined, ref_names = refine_entry_with_ltf(direction, z, ltf_snaps, atr_h1, price, min_gap)
        if refined is not None:
            entry, names = refined, ref_names
        plan, reason = build_plan_for_entry(direction, entry, z, snaps, levels, atr_h1, cfg, ltf_names=names)
        row = {'kind': z['kind'], 'tf': z['tf'], 'bottom': z['bottom'], 'top': z['top'], 'entry': round_price(entry, cfg),
               'zone_score': z['zone_score'], 'dist_atr': z['dist_atr'], 'touches': z['touches'], 'confluence': z['confluence'],
               'valid': plan is not None, 'reason': reason}
        if plan is not None:
            score, breakdown, candle_notes = score_plan(direction, plan, z, bias, snaps, dxy_rel, regime, ltf_snaps, price, atr_h1)
            plan['zone'] = {'kind': z['kind'], 'tf': z['tf'], 'top': z['top'], 'bottom': z['bottom'],
                            'touches': z['touches'], 'confluence': z['confluence'], 'zone_score': z['zone_score']}
            plan['score'], plan['score_breakdown'], plan['candle_notes'] = score, breakdown, candle_notes
            row['score'] = score
            valid.append(plan)
        audit.append(row)
    if valid:
        best = max(valid, key=lambda p: (p['score'], p['zone']['zone_score']))
        best['source'] = 'HTF_STRUCTURE'
        return best, audit
    entry = compute_fallback_atr_plan_levels(direction, price, atr_h1, cfg)
    fake_zone = _zone('ATR pullback (no valid HTF zone in reach)', 'H1', entry + 0.2 * atr_h1, entry - 0.2 * atr_h1, 0.0, 0)
    fake_zone['zone_score'] = 0.0
    plan, reason = build_plan_for_entry(direction, entry, fake_zone, snaps, levels, atr_h1, cfg)
    if plan is None:
        risk = max(entry * float(cfg['min_dist_pct']), atr_h1 * float(cfg['min_stop_atr']))
        sl = entry - risk if direction == 'BUY' else entry + risk
        tp = entry + risk * 1.7 if direction == 'BUY' else entry - risk * 1.7
        plan = {'direction': direction, 'entry': round_price(entry, cfg), 'stop_loss': round_price(sl, cfg),
                'take_profit': [round_price(tp, cfg), round_price(entry + risk * 3 if direction == 'BUY' else entry - risk * 3, cfg)],
                'rr_ratio': 1.7, 'risk': float(risk), 'sl_anchor': 'ATR stop', 'tp1_kind': 'R-multiple projection',
                'tp2_kind': 'R-multiple projection', 'tp1_structural': False, 'tp2_structural': False, 'ltf_refs': [], 'order_type': 'LIMIT'}
    plan['zone'] = {'kind': fake_zone['kind'], 'tf': 'H1', 'top': fake_zone['top'], 'bottom': fake_zone['bottom'],
                    'touches': 0, 'confluence': [], 'zone_score': 0.0}
    score, breakdown, candle_notes = score_plan(direction, plan, fake_zone, bias, snaps, dxy_rel, regime, ltf_snaps, price, atr_h1)
    plan['score'], plan['score_breakdown'], plan['candle_notes'] = min(score, 60), breakdown, candle_notes
    plan['source'] = 'ATR_FALLBACK'
    return plan, audit


# =============================================================================
# ── V3 CORE: PLAN LIFECYCLE & PLAN LOCK ─────────────────────────────────────
# A plan is created ONCE and then managed until it fills+resolves, is missed, expires, goes
# stale, or the HTF bias genuinely flips. Re-running the analysis returns the SAME plan instead
# of inventing a new one — this is what ends the "BUY now, SELL 30 minutes later" behaviour.
# =============================================================================
def _close_plan(plan, outcome, ts, events, ledger=True):
    plan['status'] = outcome
    plan['closed_at'] = (ts if isinstance(ts, str) else pd.Timestamp(ts).isoformat())
    plan['outcome'] = outcome
    if outcome == 'TP_HIT':
        r = float(plan.get('rr_ratio') or 0.0)
    elif outcome == 'SL_HIT':
        r = -1.0
    else:
        r = 0.0
    plan['r_multiple'] = r
    events.append({'type': outcome, 'time': plan['closed_at']})
    if not ledger:
        return
    state = load_state()
    state['ledger'].append({
        'symbol': plan.get('symbol'), 'direction': plan.get('direction'), 'entry': plan.get('entry'),
        'stop_loss': plan.get('stop_loss'), 'tp1': (plan.get('take_profit') or [None])[0], 'rr': plan.get('rr_ratio'),
        'score': plan.get('score'), 'grade': plan.get('grade'), 'created_at': plan.get('created_at'),
        'filled_at': plan.get('filled_at'), 'closed_at': plan['closed_at'], 'outcome': outcome, 'r': r,
        'zone': (plan.get('zone') or {}).get('kind'),
    })
    if len(state['ledger']) > MAX_LEDGER:
        state['ledger'] = state['ledger'][-MAX_LEDGER:]

def evaluate_plan_lifecycle(plan, price_df, live_price=None, now=None):
    """Walk every candle since the plan was created (pessimistic on same-candle SL/TP ambiguity)."""
    events = []
    if not plan or plan.get('status') not in LIVE_PLAN_STATUSES:
        return events
    now = now or utc_now()
    created = parse_ts(plan.get('created_at'))
    if created is None:
        return events
    buy = plan['direction'] == 'BUY'
    entry, sl = float(plan['entry']), float(plan['stop_loss'])
    tp1 = float(plan['take_profit'][0])
    path = []
    if price_df is not None and not price_df.empty:
        window = price_df[price_df.index > created]
        for ts, row in window.iterrows():
            path.append((ts, float(row['High']), float(row['Low'])))
    if live_price:
        path.append((pd.Timestamp(now), float(live_price), float(live_price)))
    for ts, hi, lo in path:
        if plan['status'] == 'PENDING':
            filled = (lo <= entry) if buy else (hi >= entry)
            ran = ((hi >= tp1) if buy else (lo <= tp1)) and not filled
            if filled:
                plan['status'] = 'FILLED'
                plan['filled_at'] = pd.Timestamp(ts).isoformat()
                events.append({'type': 'FILLED', 'time': plan['filled_at']})
            elif ran:
                _close_plan(plan, 'MISSED', ts, events)
                break
            else:
                continue
        if plan['status'] == 'FILLED':
            sl_hit = (lo <= sl) if buy else (hi >= sl)
            tp_hit = (hi >= tp1) if buy else (lo <= tp1)
            if sl_hit:
                _close_plan(plan, 'SL_HIT', ts, events)
                break
            if tp_hit:
                _close_plan(plan, 'TP_HIT', ts, events)
                break
    if plan.get('status') == 'PENDING':
        age_h = (now - created).total_seconds() / 3600.0
        if age_h > float(plan.get('expiry_hours', 18)):
            _close_plan(plan, 'EXPIRED', now.isoformat(), events)
    return events

def describe_plan_event(plan, ev):
    sym, side = plan.get('symbol'), plan.get('direction')
    entry, sl = plan.get('entry'), plan.get('stop_loss')
    tp1 = (plan.get('take_profit') or [None])[0]
    kind = ev['type']
    if kind == 'FILLED':
        return 'success', f"✅ {sym} {side} LIMIT FILLED at {entry}. SL {sl} | TP1 {tp1}. The trade is now live."
    if kind == 'TP_HIT':
        return 'success', f"🎯 {sym} {side} hit TP1 {tp1} (+{plan.get('r_multiple')}R)."
    if kind == 'SL_HIT':
        return 'warning', f"🛑 {sym} {side} stopped out at {sl} (-1R)."
    if kind == 'MISSED':
        return 'info', f"⏭ {sym} {side}: price reached TP1 {tp1} without filling {entry}. Plan closed (missed) — a new plan will be built on the next run."
    if kind == 'EXPIRED':
        return 'info', f"⌛ {sym} {side} LIMIT {entry} expired unfilled after {plan.get('expiry_hours', 18)}h."
    if kind == 'CANCELLED':
        return 'warning', f"⛔ {sym} {side} LIMIT {entry} cancelled: {plan.get('cancel_reason', 'HTF bias flipped')}."
    if kind == 'STALE':
        return 'info', f"↔ {sym} {side} LIMIT {entry} went stale (price ran too far from the level). Re-planning."
    return 'info', f"{sym} {side}: {kind}"

def push_plan_events(symbol, plan, events):
    for ev in events:
        note_type, text = describe_plan_event(plan, ev)
        add_notification(note_type, text, symbol=symbol, signal=plan.get('direction'))
        if st.session_state.get('notify_plan_events', True) and ev['type'] in ('FILLED', 'TP_HIT', 'SL_HIT', 'CANCELLED', 'MISSED', 'EXPIRED'):
            send_telegram_message(f"📌 <b>DER-AI PLAN UPDATE</b>\n{_escape_telegram_html(text)}")

def get_plan_stats():
    ledger = load_state().get('ledger', [])
    tp = [x for x in ledger if x.get('outcome') == 'TP_HIT']
    sl = [x for x in ledger if x.get('outcome') == 'SL_HIT']
    resolved = len(tp) + len(sl)
    total_r = sum(float(x.get('r') or 0) for x in tp + sl)
    return {
        'total_closed': len(ledger), 'wins': len(tp), 'losses': len(sl),
        'missed': sum(1 for x in ledger if x.get('outcome') == 'MISSED'),
        'expired': sum(1 for x in ledger if x.get('outcome') == 'EXPIRED'),
        'cancelled': sum(1 for x in ledger if x.get('outcome') in ('CANCELLED', 'STALE')),
        'win_rate': (len(tp) / resolved * 100.0) if resolved else None,
        'total_r': round(total_r, 2), 'avg_r': round(total_r / resolved, 2) if resolved else None,
    }

def refresh_live_plans(all_data, symbols=None):
    """Re-check every live plan against the latest candles + live quote (no new signals)."""
    state = load_state(force=True)
    changed = []
    for symbol, plan in list(state['plans'].items()):
        if symbols and symbol not in symbols:
            continue
        if plan.get('status') not in LIVE_PLAN_STATUSES:
            continue
        data = all_data.get(symbol, {}) or {}
        finest = data.get('M10') if data.get('M10') is not None and not data.get('M10').empty else data.get('M15')
        snap = get_live_market_snapshot(symbol, YFINANCE_MAP.get(symbol, symbol), fallback_df=finest)
        events = evaluate_plan_lifecycle(plan, finest, live_price=snap.get('price'))
        if events:
            push_plan_events(symbol, plan, events)
            changed.append(symbol)
    save_state()
    return changed

def mark_plan_sent(symbol):
    state = load_state()
    plan = state['plans'].get(symbol)
    if plan:
        plan['sent'] = True
        save_state()

# =============================================================================
# ── V3 CORE: TEXT BUILDERS ──────────────────────────────────────────────────
# =============================================================================
def _fmt(v, cfg=None):
    if v is None:
        return 'n/a'
    try:
        digits = int((cfg or {}).get('digits', 2))
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)

def describe_snapshot(snap, cfg):
    if not snap:
        return 'data unavailable'
    last = snap['breaks']['last']
    brk = f"{last['kind']} {last['dir'].lower()} ({last['age']} bars ago)" if last else 'no break'
    obs = '; '.join(f"{o['type'].replace('_OB', '')} OB {_fmt(o['bottom'], cfg)}-{_fmt(o['top'], cfg)}" for o in snap['order_blocks'][:3]) or 'none'
    fvg = '; '.join(f"{f['type'].replace('_FVG', '')} FVG {_fmt(f['bottom'], cfg)}-{_fmt(f['top'], cfg)}" for f in snap['fvgs'][:2]) or 'none'
    hs = [p['price'] for p in snap['pivots'] if p['type'] == 'H' and not p.get('swept')][-2:]
    ls = [p['price'] for p in snap['pivots'] if p['type'] == 'L' and not p.get('swept')][-2:]
    rng = snap['range']
    last_candle = snap.get('candles') or []
    cndl = last_candle[-1]['pattern'] if last_candle else 'n/a'
    return (f"structure={snap['trend']} ({snap['pattern']}), last break={brk}, EMA={snap['ema']['state']}, RSI={snap['rsi']}, last candle={cndl}, "
            f"unmitigated OBs=[{obs}], open FVGs=[{fvg}], live swing highs={[_fmt(x, cfg) for x in hs]}, live swing lows={[_fmt(x, cfg) for x in ls]}, "
            f"dealing range {_fmt(rng['low'], cfg)}-{_fmt(rng['high'], cfg)}")

def build_plan_reasoning(symbol, plan, bias, cfg, live_price):
    z = plan['zone']
    side = plan['direction']
    tps = plan['take_profit']
    verb = 'retrace UP into' if side == 'SELL' else 'retrace DOWN into'
    move = 'sell-off' if side == 'SELL' else 'rally'
    bias_txt = f"HTF bias is {bias['word']} (score {bias['score']:+.0f}; H4 {bias['h4_score']:+.0f}, H1 {bias['h1_score']:+.0f}; {'H4 and H1 aligned' if bias['aligned'] else 'H4/H1 not fully aligned'})"
    if bias['held']:
        bias_txt += ' - standing bias retained'
    conf = f", stacked with {', '.join(z['confluence'][:3])}" if z.get('confluence') else ''
    ltf = f" Refined on {', '.join(plan['ltf_refs'])} inside the zone." if plan.get('ltf_refs') else ' No sharper M30/M15 level inside the zone, so the zone mean-threshold is used.'
    return (
        f"{bias_txt}. Price {_fmt(live_price, cfg)} is expected to {verb} the {z['kind']} ({_fmt(z['bottom'], cfg)}-{_fmt(z['top'], cfg)}{conf}) before resuming the {move}."
        f"{ltf} Entry {_fmt(plan['entry'], cfg)}; SL {_fmt(plan['stop_loss'], cfg)} beyond {plan['sl_anchor']}; "
        f"TP1 {_fmt(tps[0], cfg)} ({plan['tp1_kind']}, {plan['rr_ratio']}R), TP2 {_fmt(tps[1], cfg)} ({plan['tp2_kind']}). "
        f"The idea is invalidated by an H1 close beyond {_fmt(plan['stop_loss'], cfg)} or an opposing H1/H4 structure break."
    )

def build_ai_audit_prompt(symbol, plan, bias, audit, snaps, cfg, live_price, dxy, dxy_rel, levels):
    cand_lines = []
    for n, row in enumerate(audit[:5], 1):
        cand_lines.append(f"{n}. {row['kind']} {_fmt(row['bottom'], cfg)}-{_fmt(row['top'], cfg)} -> entry {_fmt(row['entry'], cfg)}, zone score {row['zone_score']}, "
                          f"{row['dist_atr']} ATR away, {'VALID' if row['valid'] else 'rejected: ' + row['reason']}")
    lv = ', '.join(f"{k} {_fmt(v, cfg)}" for k, v in levels.items()) or 'n/a'
    m10 = snaps.get('_micro_m10') or {}
    m10_candles = snaps.get('_candles_m10') or []
    m10_pattern = m10_candles[-1]['pattern'] if m10_candles else 'n/a'
    body = f"""You are the risk auditor of an institutional trading desk. A deterministic Python engine has ALREADY decided direction and exact levels using strict top-down analysis (H4/H1 decide direction; M30/M15 only refine the limit price inside the HTF zone). You may NOT change direction, entry, stop or target. Your job is to sanity-check the plan, use the chart screenshot if one is attached, flag concrete risks, and write a concise rationale.

SYMBOL: {symbol} | LIVE PRICE: {_fmt(live_price, cfg)}
LOCKED PLAN: {plan['direction']} LIMIT | entry {_fmt(plan['entry'], cfg)} | SL {_fmt(plan['stop_loss'], cfg)} | TP1 {_fmt(plan['take_profit'][0], cfg)} ({plan['tp1_kind']}) | TP2 {_fmt(plan['take_profit'][1], cfg)} | R:R {plan['rr_ratio']}
CHOSEN ZONE: {plan['zone']['kind']} {_fmt(plan['zone']['bottom'], cfg)}-{_fmt(plan['zone']['top'], cfg)}; confluence: {', '.join(plan['zone'].get('confluence') or []) or 'none'}; LTF refinement: {', '.join(plan.get('ltf_refs') or []) or 'none'}
PYTHON SCORE: {plan['score']}/100 -> {json.dumps(plan['score_breakdown'])}
HTF BIAS ({bias['word']}, score {bias['score']:+.0f}): {' | '.join(bias['evidence'][:8])}. {' '.join(bias['notes'])}
H4: {describe_snapshot(snaps.get('H4'), cfg)}
H1: {describe_snapshot(snaps.get('H1'), cfg)}
M30: {describe_snapshot(snaps.get('M30'), cfg)}
M15: {describe_snapshot(snaps.get('M15'), cfg)}
M10 MICRO: VWAP {m10.get('vwap', 'n/a')} | price vs VWAP {m10.get('price_vs_vwap', 'n/a')} | RVOL {m10.get('rvol', 'n/a')} | momentum {m10.get('momentum', 'n/a')} | last candle {m10_pattern}
KEY LEVELS: {lv}
DXY: {dxy.get('detail')} -> relation to this trade: {dxy_rel}
RANKED HTF ENTRY CANDIDATES:
{chr(10).join(cand_lines) or 'none'}

TASKS:
1. Say whether you AGREE with the plan (agree=true/false). Disagree ONLY for a concrete reason visible in the data or screenshot (e.g. entry sits under an obvious untested opposing level, target is beyond a major barrier, macro headline risk). Disagreement reduces the score; it cannot flip the trade.
2. Give score_adjustment between -6 and +6.
3. List up to 3 specific risk_flags (empty list if none).
4. next_level_watch: the next key HTF level beyond TP2 and why price may approach it.
5. If a screenshot is attached, summarise visible support/resistance/liquidity in visual_levels; otherwise write "unavailable".
Keep everything concise. Output complete valid JSON only (no markdown, no code fences):
{{"agree": true, "score_adjustment": 0, "htf_bias_basis": "1-2 sentences", "ltf_entry_basis": "1-2 sentences", "risk_flags": [], "next_level_watch": "...", "visual_levels": "...", "microstructure_read": "...", "reasoning": "3-4 sentences: HTF thesis, why this zone, invalidation, target"}}"""
    return body

def build_telegram_signal_message(symbol, result):
    cfg = get_pair_config(symbol)
    tps = result.get('take_profit') or []
    tp1 = _fmt(tps[0], cfg) if tps else 'N/A'
    tp2 = _fmt(tps[1], cfg) if len(tps) > 1 else None
    signal = _escape_telegram_html(normalize_ai_signal(result.get('signal')))
    zone = result.get('zone') or {}
    bias = result.get('htf_bias') or {}
    reasoning = str(result.get('reasoning') or '')
    if len(reasoning) > 1100:
        reasoning = reasoning[:1097] + '...'
    lines = [
        "🌍 <b>DER-AI MARKET SIGNAL</b>",
        f"📊 <b>{_escape_telegram_html(symbol)}</b> - {signal} LIMIT | Grade {_escape_telegram_html(result.get('grade', '-'))}",
        f"🧭 HTF Bias: {_escape_telegram_html(bias.get('word', 'n/a'))} (score {bias.get('score', 0):+.0f}{', H4+H1 aligned' if bias.get('aligned') else ''})",
    ]
    if zone:
        lines.append(f"📍 Zone: {_escape_telegram_html(zone.get('kind'))} {_fmt(zone.get('bottom'), cfg)}-{_fmt(zone.get('top'), cfg)}")
    lines.append(f"🤖 Model: {_escape_telegram_html(result.get('model_used', 'Unknown'))} | 📈 Score: {result.get('confluence_score', 0)}/100 | 🔋 Tokens: {result.get('total_tokens', 'N/A')}")
    lines.append(f"🧾 Order: {_escape_telegram_html(result.get('order_type', 'LIMIT'))}")
    tp_txt = f"🎯 TP1: {tp1}" + (f" | TP2: {tp2}" if tp2 else '')
    lines.append(f"💰 Entry: {_fmt(result.get('entry'), cfg)} | 🛑 SL: {_fmt(result.get('stop_loss'), cfg)} | {tp_txt}")
    lines.append(f"📐 R:R {result.get('rr_ratio')} | ⏳ Valid ~{result.get('expiry_hours', 18)}h (cancel if HTF bias flips)")
    lines.append(f"📈 DXY: {_escape_telegram_html(result.get('dxy_correlation'))}")
    if result.get('risk_flags'):
        lines.append("⚠️ " + _escape_telegram_html('; '.join(str(x) for x in result['risk_flags'][:3])))
    lines.append(f"🧠 {_escape_telegram_html(reasoning)}")
    return '\n'.join(lines)

# =============================================================================
# ── V3 CORE: ORCHESTRATOR ───────────────────────────────────────────────────
# =============================================================================
def _plan_side_ok(direction, entry, price, min_gap):
    return (price - entry) >= min_gap * 0.999 if direction == 'BUY' else (entry - price) >= min_gap * 0.999

def detect_market_closed(symbol, df):
    """Non-crypto symbols close on weekends/holidays; a plan is still built, but not pushed."""
    if symbol == 'BTCUSD':
        return False
    try:
        mins = infer_bar_minutes(df) or 15
        last_open = df.index[-1]
        last_open = last_open.tz_localize('UTC') if last_open.tzinfo is None else last_open.tz_convert('UTC')
        idle = (utc_now() - last_open.to_pydatetime()).total_seconds() / 60.0
        return idle > max(3 * mins, 90)
    except Exception:
        return False

def _finalize_result(symbol, plan, bias, audit, cfg, live_price, ai, dxy, dxy_rel, snaps, regime, market_closed, atr_h1):
    score = int(plan['score'])
    ai_used = bool(ai and ai.get('api_status') in ('SUCCESS', 'SUCCESS_EXTRACTED'))
    notes = []
    if plan.get('source') == 'ATR_FALLBACK':
        notes.append('No valid HTF zone was within reach, so this is a low-conviction ATR pullback plan (capped at 60 and not pushed to Telegram).')
    notes.extend(bias['notes'])
    notes.extend(plan.get('candle_notes') or [])
    ai_text = ''
    risk_flags = []
    if ai_used:
        try:
            adj = int(max(-6, min(6, round(float(ai.get('score_adjustment', 0) or 0)))))
        except Exception:
            adj = 0
        agree = ai.get('agree')
        if agree is False:
            adj = min(adj, -4)
            notes.append('AI auditor DISAGREED with the plan (direction and levels unchanged; conviction reduced).')
        score = max(0, min(100, score + adj))
        if plan.get('source') == 'ATR_FALLBACK':
            score = min(score, 60)
        ai_text = str(ai.get('reasoning') or '').strip()
        risk_flags = [str(x) for x in (ai.get('risk_flags') or []) if x][:3]
    py_reason = build_plan_reasoning(symbol, plan, bias, cfg, live_price)
    reasoning = py_reason + (f" Auditor: {ai_text}" if ai_text else '')
    result = {
        'symbol': symbol, 'signal': plan['direction'], 'bias': bias['word'],
        'confluence_score': score, 'confidence': confidence_for(score), 'grade': grade_for(score),
        'dxy_correlation': dxy_rel, 'reasoning': reasoning, 'python_reasoning': py_reason,
        'entry': plan['entry'], 'stop_loss': plan['stop_loss'], 'take_profit': plan['take_profit'],
        'rr_ratio': plan['rr_ratio'], 'order_type': 'LIMIT', 'exhaustion_target': plan['take_profit'][0],
        'expiry_hours': cfg['plan_expiry_hours'], 'order_expiry': f"~{cfg['plan_expiry_hours']}h or until the HTF bias flips",
        'invalidation': f"H1 close beyond {_fmt(plan['stop_loss'], cfg)} or an opposing H1/H4 structure break",
        'order_description': (f"{plan['direction']} LIMIT {_fmt(plan['entry'], cfg)} | SL {_fmt(plan['stop_loss'], cfg)} | "
                              f"TP1 {_fmt(plan['take_profit'][0], cfg)} | TP2 {_fmt(plan['take_profit'][1], cfg)} | valid ~{cfg['plan_expiry_hours']}h."),
        'zone': plan['zone'], 'htf_bias': bias, 'score_breakdown': plan['score_breakdown'],
        'python_score': int(plan['score']), 'candidates': audit[:6], 'levels_source': plan.get('source', 'HTF_STRUCTURE'),
        'sl_anchor': plan['sl_anchor'], 'tp1_kind': plan['tp1_kind'], 'tp2_kind': plan['tp2_kind'], 'ltf_refs': plan.get('ltf_refs', []),
        'python_validation_notes': notes, 'risk_flags': risk_flags, 'market_closed': market_closed,
        'regime': (regime or {}).get('regime'), 'atr': atr_h1, 'live_price': live_price,
        'dxy_summary': dxy.get('detail'), 'ai_used': ai_used,
        'microstructure_read': (ai.get('microstructure_read') if ai_used and ai.get('microstructure_read') else
                                'VWAP {} | RVOL {} | momentum {} | last candle {}'.format(
                                    *[(snaps.get('_micro_m10') or {}).get(k, 'n/a') for k in ('price_vs_vwap', 'rvol', 'momentum')],
                                    ((snaps.get('_candles_m10') or [{}])[-1] or {}).get('pattern', 'n/a'))),
        'visual_levels': (ai.get('visual_levels') if ai_used else 'unavailable'),
        'next_level_watch': (ai.get('next_level_watch') if ai_used and ai.get('next_level_watch') else
                             f"Beyond TP2 the next liquidity to watch is the {'H4 range high' if plan['direction'] == 'BUY' else 'H4 range low'} at "
                             f"{_fmt((snaps.get('H4') or snaps.get('H1'))['range']['high' if plan['direction'] == 'BUY' else 'low'], cfg)}."),
        'htf_bias_basis': (ai.get('htf_bias_basis') if ai_used else None), 'ltf_entry_basis': (ai.get('ltf_entry_basis') if ai_used else None),
        'model_used': (ai.get('model_used') if ai_used else 'Python structure engine (AI auditor unavailable)'),
        'api_status': (ai.get('api_status') if ai_used else 'ENGINE_ONLY'),
        'total_tokens': (ai.get('total_tokens', 0) if ai else 0), 'prompt_tokens': (ai.get('prompt_tokens', 0) if ai else 0),
        'completion_tokens': (ai.get('completion_tokens', 0) if ai else 0),
        'model_attempts': (ai.get('model_attempts') if ai else None),
        'market_state': 'continuation' if bias['aligned'] else 'coiling',
    }
    if ai and not ai_used:
        result['groq_failure'] = ai.get('rejection_reason')
        result['groq_api_status'] = ai.get('api_status')
    return result

def analyze_symbol_premium(symbol, all_data, image_b64=None, image_mime_type='image/png', force_new=False, use_ai=True):
    try:
        state = load_state()
        data = all_data.get(symbol, {}) or {}
        m10 = data.get('M10', pd.DataFrame())
        yf_symbol = YFINANCE_MAP.get(symbol, symbol)
        live_snapshot = get_live_market_snapshot(symbol, yf_symbol, fallback_df=m10)
        if m10 is None or m10.empty:
            return {"error": f"Failed to fetch market data for {symbol}. The data source may be rate-limiting your IP. Wait a few minutes and try again."}
        refreshed = refresh_symbol_data_if_stale(symbol, yf_symbol, live_snapshot, data)
        if refreshed is not data:
            data = refreshed
            all_data[symbol] = data
            m10 = data.get('M10', pd.DataFrame())
            if m10 is None or m10.empty:
                return {"error": f"Fresh market data was unavailable for {symbol}. Please try again."}
        current_price = float(live_snapshot.get('price') or m10['Close'].iloc[-1])
        cfg = get_pair_config(symbol)
        market_closed = detect_market_closed(symbol, m10)

        # ── 1. Structural snapshots (CLOSED candles only) ────────────────────────────
        h1_raw = data.get('H1', pd.DataFrame())
        h4_raw = ensure_h4(h1_raw, data.get('H4', pd.DataFrame()))
        snaps = {
            'H4': analyze_structure(h4_raw, 'H4', left=2, right=2, leg_atr=1.0, ob_disp_atr=cfg['ob_disp_atr'], fvg_min_atr=cfg['fvg_min_atr'], liq_tol_atr=cfg['liq_tol_atr']),
            'H1': analyze_structure(h1_raw, 'H1', left=3, right=3, leg_atr=1.0, ob_disp_atr=cfg['ob_disp_atr'], fvg_min_atr=cfg['fvg_min_atr'], liq_tol_atr=cfg['liq_tol_atr']),
            'M30': analyze_structure(data.get('M30'), 'M30', left=3, right=3, leg_atr=0.8, ob_disp_atr=cfg['ob_disp_atr'], fvg_min_atr=cfg['fvg_min_atr'], liq_tol_atr=cfg['liq_tol_atr']),
            'M15': analyze_structure(data.get('M15'), 'M15', left=3, right=3, leg_atr=0.8, ob_disp_atr=cfg['ob_disp_atr'], fvg_min_atr=cfg['fvg_min_atr'], liq_tol_atr=cfg['liq_tol_atr']),
        }
        snaps['_micro_m10'] = calculate_microstructure(m10)
        snaps['_candles_m10'] = classify_candles(closed_candles(m10))
        if not snaps['H1']:
            return {"error": f"Not enough closed H1 history for {symbol} to build a reliable higher-timeframe structure. Try again shortly."}
        atr_h1 = snaps['H1']['atr']
        levels = prior_period_levels(h1_raw)
        dxy = compute_dxy_context(all_data)
        regime = classify_market_regime(snaps['H1']['df'])

        # ── 2. Manage any existing plan BEFORE deciding anything new ─────────────────
        events = []
        existing = state['plans'].get(symbol)
        if existing and existing.get('status') in LIVE_PLAN_STATUSES:
            events += evaluate_plan_lifecycle(existing, m10, live_price=current_price)
            if existing.get('status') == 'PENDING':
                _, max_gap = htf_entry_gap_bounds(current_price, atr_h1, cfg)
                if abs(current_price - float(existing['entry'])) > 1.25 * max_gap:
                    _close_plan(existing, 'STALE', utc_now().isoformat(), events)
        if events and existing:
            push_plan_events(symbol, existing, events)

        # ── 3. HTF bias (H4 + H1 only, persistent hysteresis) ────────────────────────
        bias = resolve_htf_bias(symbol, snaps, dxy, persist=True)
        direction = bias['direction']
        dxy_rel = dxy_relation(symbol, direction, dxy)

        live_existing = existing if (existing and existing.get('status') in LIVE_PLAN_STATUSES) else None
        plan_state = 'NEW'
        if live_existing and not force_new:
            if live_existing['direction'] == direction:
                provisional = live_existing.get('status') == 'PENDING' and not live_existing.get('sent')
                if not provisional:
                    plan_state = 'REUSED'
                else:
                    plan_state = 'PROVISIONAL'
            else:
                if live_existing['status'] == 'PENDING':
                    live_existing['cancel_reason'] = f"HTF bias flipped to {direction} ({bias['score']:+.0f})"
                    ev = []
                    _close_plan(live_existing, 'CANCELLED', utc_now().isoformat(), ev)
                    push_plan_events(symbol, live_existing, ev)
                    live_existing = None
                    plan_state = 'NEW'
                else:
                    plan_state = 'OPEN_TRADE_CONFLICT'
        elif live_existing and force_new:
            if live_existing['status'] == 'PENDING':
                live_existing['cancel_reason'] = 'manual "force fresh plan" override'
                _close_plan(live_existing, 'CANCELLED', utc_now().isoformat(), [], ledger=False)
                live_existing = None
            plan_state = 'NEW'

        def _reuse_result(label):
            result = dict(live_existing.get('result') or {})
            result['plan_state'] = label
            result['plan_status'] = live_existing['status']
            result['live_price'] = current_price
            result['htf_bias'] = bias
            result['analyzed_at_live'] = utc_now().isoformat()
            result['gap_to_entry'] = round(abs(current_price - float(live_existing['entry'])), cfg['digits'])
            if label == 'OPEN_TRADE_CONFLICT':
                result.setdefault('python_validation_notes', [])
                msg = f"HTF bias is now {direction} but the {live_existing['direction']} trade is already FILLED and open — manage it (SL/TP unchanged). A new plan will be built after it closes."
                if msg not in result['python_validation_notes']:
                    result['python_validation_notes'].append(msg)
                if not live_existing.get('flip_warned'):
                    live_existing['flip_warned'] = True
                    add_notification('warning', f"⚠️ {symbol}: {msg}", symbol=symbol)
                    if st.session_state.get('notify_plan_events', True):
                        send_telegram_message(f"⚠️ <b>DER-AI</b>\n{_escape_telegram_html(symbol)}: {_escape_telegram_html(msg)}")
            result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result['events'] = events
            save_state()
            return result

        if plan_state in ('REUSED', 'OPEN_TRADE_CONFLICT'):
            return _reuse_result(plan_state)

        # ── 4. Build the best HTF-anchored limit plan ────────────────────────────────
        plan, audit = select_best_plan(direction, current_price, snaps, levels, atr_h1, cfg, bias, dxy_rel, regime)
        if plan_state == 'PROVISIONAL':
            # An unsent (low-conviction) plan is only replaced if the fresh candidate would actually
            # clear the send threshold — otherwise keep the old plan so levels don't drift run to run.
            thr = int(st.session_state.get('min_send_score', MINIMUM_CONFLUENCE_SCORE))
            if plan['score'] < thr or plan.get('source') == 'ATR_FALLBACK':
                return _reuse_result('REUSED')

        # ── 5. AI auditor (cannot change direction/levels) ───────────────────────────
        ai = None
        if use_ai and get_secret('GROQ_API_KEY', '').strip():
            prompt_text = build_ai_audit_prompt(symbol, plan, bias, audit, snaps, cfg, current_price, dxy, dxy_rel, levels)
            ai = call_groq(prompt_text, [], estimated_tokens=estimate_analysis_tokens(prompt_text, []), image_b64=image_b64, image_mime_type=image_mime_type)
            st.session_state.last_model_attempts[symbol] = ai.get('model_attempts', [])
            post = get_live_market_snapshot(symbol, yf_symbol, fallback_df=m10)
            if post.get('price'):
                new_price = float(post['price'])
                min_gap, _ = htf_entry_gap_bounds(new_price, atr_h1, cfg)
                if not _plan_side_ok(direction, float(plan['entry']), new_price, min_gap):
                    current_price = new_price
                    plan, audit = select_best_plan(direction, current_price, snaps, levels, atr_h1, cfg, bias, dxy_rel=dxy_rel, regime=regime)
                    plan.setdefault('notes', []).append('Rebuilt after the AI call because live price moved through the original entry.')
                else:
                    current_price = new_price
        result = _finalize_result(symbol, plan, bias, audit, cfg, current_price, ai, dxy, dxy_rel, snaps, regime, market_closed, atr_h1)
        result['plan_state'] = 'REPLANNED' if plan_state == 'PROVISIONAL' else 'NEW'
        result['plan_status'] = 'PENDING'
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['events'] = events

        ok, reason = validate_signal_math(result, pair_config=cfg)
        if not ok:
            result['python_validation_notes'].append(f"Math check: {reason}")

        # ── 6. Lock the plan ─────────────────────────────────────────────────────────
        if live_existing and plan_state == 'PROVISIONAL':
            live_existing['cancel_reason'] = 'replaced by a higher-quality plan in the same direction'
            _close_plan(live_existing, 'CANCELLED', utc_now().isoformat(), [], ledger=False)
        lock_result = to_jsonable({k: v for k, v in result.items() if k not in ('events', 'model_attempts')})
        state['plans'][symbol] = {
            'plan_id': uuid.uuid4().hex[:8], 'symbol': symbol, 'direction': direction, 'status': 'PENDING',
            'created_at': utc_now().isoformat(), 'entry': plan['entry'], 'stop_loss': plan['stop_loss'],
            'take_profit': plan['take_profit'], 'rr_ratio': plan['rr_ratio'], 'expiry_hours': cfg['plan_expiry_hours'],
            'score': result['confluence_score'], 'grade': result['grade'], 'zone': to_jsonable(plan['zone']),
            'sent': False, 'result': lock_result,
        }
        save_state()
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "api_status": "PYTHON_EXCEPTION"}

def should_send_to_telegram(result):
    if result.get('plan_state') not in ('NEW', 'REPLANNED'):
        return False, 'existing plan unchanged - no new signal to push'
    if result.get('market_closed'):
        return False, 'market appears closed - plan stored but not pushed'
    if result.get('levels_source') == 'ATR_FALLBACK' and not st.session_state.get('send_low_conviction', False):
        return False, 'no valid HTF zone in reach (low-conviction fallback plan)'
    threshold = int(st.session_state.get('min_send_score', MINIMUM_CONFLUENCE_SCORE))
    if result.get('confluence_score', 0) < threshold and not st.session_state.get('send_low_conviction', False):
        return False, f"score {result.get('confluence_score', 0)} is below the {threshold} send threshold"
    return True, ''

# =============================================================================
# ── V3 CORE: CHART ──────────────────────────────────────────────────────────
# =============================================================================
def plot_plan_chart(h1_df, result, bars=110):
    d = closed_candles(h1_df)
    if d is None or d.empty:
        return None
    d = d.tail(bars)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    o, h, l, c = (d[k].to_numpy(dtype=float) for k in ('Open', 'High', 'Low', 'Close'))
    span = max(float(h.max() - l.min()), 1e-9)
    for i in range(len(d)):
        color = '#10b981' if c[i] >= o[i] else '#ef4444'
        ax.vlines(i, l[i], h[i], color=color, linewidth=0.8)
        ax.add_patch(Rectangle((i - 0.3, min(o[i], c[i])), 0.6, max(abs(c[i] - o[i]), span * 0.0008), color=color))
    zone = result.get('zone') or {}
    if zone.get('top') is not None:
        ax.axhspan(zone['bottom'], zone['top'], color='#3b82f6', alpha=0.18, label='HTF entry zone')
    tps = result.get('take_profit') or []
    lines = [('Entry', result.get('entry'), '#2563eb', '-'), ('SL', result.get('stop_loss'), '#dc2626', '--')]
    if tps:
        lines.append(('TP1', tps[0], '#059669', '--'))
        if len(tps) > 1:
            lines.append(('TP2', tps[1], '#34d399', ':'))
    lines.append(('Live', result.get('live_price'), '#6b7280', ':'))
    lo_v, hi_v = float(l.min()), float(h.max())
    for name, val, color, style in lines:
        if val is None:
            continue
        val = float(val)
        if name == 'TP2' and (val > hi_v + span * 0.25 or val < lo_v - span * 0.25):
            continue
        ax.axhline(val, color=color, linestyle=style, linewidth=1.1)
        ax.text(len(d) + 0.5, val, f" {name} {val:,.5g}", color=color, fontsize=8, va='center')
        lo_v, hi_v = min(lo_v, val), max(hi_v, val)
    pad = (hi_v - lo_v) * 0.06
    ax.set_ylim(lo_v - pad, hi_v + pad)
    ax.set_xlim(-1, len(d) + 14)
    ax.set_title(f"{result.get('symbol', '')} H1 — {result.get('signal')} LIMIT plan", fontsize=10)
    ax.grid(alpha=0.15)
    ax.set_xticks([])
    fig.tight_layout()
    return fig


# ── UI Layout ──────────────────────────────────────────────────────────────
st.title("📊 Der-AI | Institutional Market Analysis")
st.markdown("**Top-down structure engine | HTF bias with hysteresis | Plan lock | SMC | DXY | AI risk auditor**")

STATE_LABELS = {
    'NEW': '🆕 New plan',
    'REPLANNED': '🔄 Upgraded plan (replaced an unsent low-conviction one)',
    'REUSED': '🔒 Existing plan still valid — locked, no new signal',
    'OPEN_TRADE_CONFLICT': '⚠️ Open trade in progress — HTF bias has since changed',
}

def render_signal_card(symbol, result, all_data):
    cfg = get_pair_config(symbol)
    sig = result.get('signal')
    icon = '🟢' if sig == 'BUY' else '🔴'
    state = result.get('plan_state', 'NEW')
    st.markdown(f"### {icon} {symbol} — {sig} LIMIT · Grade {result.get('grade', '-')} ({result.get('confluence_score', 0)}/100)")
    st.caption(f"{STATE_LABELS.get(state, state)} · plan status: {result.get('plan_status', 'PENDING')} · live price {_fmt(result.get('live_price'), cfg)}")
    if state == 'REUSED':
        st.info("This plan was created earlier and is still valid (not filled, not invalidated, HTF bias unchanged). "
                "The engine deliberately does NOT invent a fresh signal from a new candle. Tick “Force fresh plan” to override.")
    if result.get('market_closed'):
        st.warning("Market appears closed for this symbol — the plan is stored but won't be pushed to Telegram.")
    tps = result.get('take_profit') or [None, None]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Entry (LIMIT)', _fmt(result.get('entry'), cfg))
    c2.metric('Stop loss', _fmt(result.get('stop_loss'), cfg))
    c3.metric('TP1', _fmt(tps[0], cfg))
    c4.metric('TP2', _fmt(tps[1] if len(tps) > 1 else None, cfg))
    c5.metric('R:R (TP1)', result.get('rr_ratio', 'N/A'))
    bias = result.get('htf_bias') or {}
    if bias:
        st.write(f"**🧭 HTF bias:** {bias.get('word')} · score {bias.get('score', 0):+.0f} (H4 {bias.get('h4_score', 0):+.0f} / H1 {bias.get('h1_score', 0):+.0f}) · "
                 f"{'H4+H1 aligned' if bias.get('aligned') else 'H4/H1 not fully aligned'} · since {str(bias.get('since', ''))[:16].replace('T', ' ')} UTC")
    zone = result.get('zone') or {}
    if zone:
        conf = f" — confluence: {', '.join(zone['confluence'])}" if zone.get('confluence') else ''
        st.write(f"**📍 Entry zone:** {zone.get('kind')} {_fmt(zone.get('bottom'), cfg)}–{_fmt(zone.get('top'), cfg)}{conf}")
    st.write(f"**Stop anchor:** {result.get('sl_anchor')} · **TP1:** {result.get('tp1_kind')} · **TP2:** {result.get('tp2_kind')}")
    st.write(f"**DXY correlation:** {result.get('dxy_correlation', 'N/A')} · **Regime (H1):** {result.get('regime', 'n/a')} · **Microstructure:** {result.get('microstructure_read', 'N/A')}")
    if result.get('order_description'):
        st.write(f"**Execution plan:** {result.get('order_description')}")
    if result.get('invalidation'):
        st.write(f"**Invalidation:** {result.get('invalidation')}")
    if result.get('next_level_watch'):
        st.write(f"**🔭 Next level watch:** {result.get('next_level_watch')}")
    if result.get('risk_flags'):
        st.warning('⚠️ AI risk flags: ' + '; '.join(result['risk_flags']))
    st.write(f"**Reasoning:** {result.get('reasoning')}")
    notes = result.get('python_validation_notes') or []
    if notes:
        st.write('**Engine notes:** ' + ' '.join(notes))
    with st.expander('📊 Score breakdown (why this score)'):
        breakdown = result.get('score_breakdown') or {}
        if breakdown:
            df_bd = pd.DataFrame({'Component': list(breakdown.keys()), 'Points': list(breakdown.values())})
            st.dataframe(df_bd, hide_index=True, use_container_width=True)
        st.caption(f"Python score {result.get('python_score', '-')} → final {result.get('confluence_score')} (AI auditor may adjust ±6; it can never change direction or levels).")
    with st.expander('🔎 Entry candidates the engine compared'):
        rows = result.get('candidates') or []
        if rows:
            st.dataframe(pd.DataFrame([{
                'Zone': r.get('kind'), 'Range': f"{_fmt(r.get('bottom'), cfg)}–{_fmt(r.get('top'), cfg)}", 'Limit entry': _fmt(r.get('entry'), cfg),
                'Zone score': r.get('zone_score'), 'ATR away': r.get('dist_atr'), 'Tested': r.get('touches'),
                'Plan score': r.get('score', '—'), 'Result': 'chosen/valid' if r.get('valid') else r.get('reason'),
            } for r in rows]), hide_index=True, use_container_width=True)
        else:
            st.caption('No HTF zone was in reach on the correct side of price.')
    try:
        h1_df = (all_data.get(symbol, {}) or {}).get('H1')
        if h1_df is not None and not h1_df.empty:
            fig = plot_plan_chart(h1_df, result)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
    except Exception as exc:
        st.caption(f"Chart unavailable: {exc}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Market Analysis", "📜 Signal History", "🧭 Plan Tracker", "🔔 Notifications", "⚙️ Settings"])

with tab1:
    st.header("🚀 Run AI Market Analysis")
    selected_symbols = st.multiselect("Select Symbols to Analyse", SYMBOLS, default=['XAUUSD', 'EURUSD', 'BTCUSD'])
    opt1, opt2 = st.columns(2)
    force_new = opt1.checkbox("Force fresh plan (ignore plan lock)", value=False,
                              help="Normally an existing valid plan is kept. Tick this only if you want the engine to rebuild from scratch.")
    use_ai = opt2.checkbox("Use AI risk auditor (Groq)", value=True,
                           help="The Python engine decides direction and levels. The AI only audits, flags risks, reads your screenshot and adjusts the score by at most ±6.")
    uploaded_file = st.file_uploader("📸 Attach Market Chart Screenshot (Optional - the AI auditor will read it)", type=["png", "jpg", "jpeg"])

    image_b64 = None
    image_mime_type = 'image/png'
    if uploaded_file is not None:
        image_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
        image_mime_type = uploaded_file.type or image_mime_type
        st.image(uploaded_file, caption="Uploaded Chart Snapshot", width=400)

    if st.button("🧠 Analyse Market Now", type="primary"):
        if use_ai and not get_secret("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set — running on the Python structure engine alone (direction and levels are unaffected; only the AI audit is skipped).")
        load_state(force=True)
        with st.spinner("Fetching market data and running top-down structure analysis..."):
            all_data = fetch_all_data()
            st.session_state.cached_market_data = all_data
            for symbol in selected_symbols:
                st.info(f"Analysing {symbol}...")
                result = analyze_symbol_premium(symbol, all_data, image_b64=image_b64, image_mime_type=image_mime_type, force_new=force_new, use_ai=use_ai)
                if 'error' in result:
                    st.error(f"❌ {symbol}: {result['error']}")
                    add_notification('warning', f"❌ {symbol}: {result['error']}", symbol=symbol)
                    continue

                api_status = result.get('api_status', 'UNKNOWN')
                model_used = result.get('model_used', 'Unknown')
                if api_status == 'ENGINE_ONLY':
                    if result.get('plan_state') in ('REUSED', 'OPEN_TRADE_CONFLICT'):
                        st.markdown("**🤖 Analysis:** existing locked plan re-checked against live price (no new AI call needed).")
                    else:
                        st.markdown(f"**🤖 Analysis:** `{model_used}`" + (f" — Groq unavailable ({result.get('groq_api_status')}): {result.get('groq_failure')}" if result.get('groq_failure') else ''))
                else:
                    primary_ok = model_used == GROQ_MODELS[0]
                    badge = "🥇 primary" if primary_ok else ("⬇️ fallback model" if model_used in GROQ_MODELS else "")
                    st.markdown(f"**🤖 AI auditor:** `{model_used}` {badge} | **🔋 Tokens:** `{result.get('total_tokens', 0)}` (Prompt: {result.get('prompt_tokens', 0)}, Completion: {result.get('completion_tokens', 0)})")
                    attempts = st.session_state.last_model_attempts.get(symbol) or result.get('model_attempts')
                    if attempts and (not primary_ok or any(a.get('status') != 'SUCCESS' for a in attempts)):
                        with st.expander(f"🔎 Why {GROQ_MODELS[0]} wasn't the only model used"):
                            for a in attempts:
                                st.caption(f"`{a.get('model')}` → {a.get('status')}")

                render_signal_card(symbol, result, all_data)

                if result.get('plan_state') in ('NEW', 'REPLANNED'):
                    ok_send, why = should_send_to_telegram(result)
                    result['sent'] = False
                    if ok_send:
                        msg = build_telegram_signal_message(symbol, result)
                        if send_telegram_message(msg):
                            st.success("✅ Signal sent to Telegram!")
                            result['sent'] = True
                            mark_plan_sent(symbol)
                        else:
                            st.warning("Telegram send failed (check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). The plan is still locked in the tracker.")
                    else:
                        st.info(f"📭 Not pushed to Telegram: {why}. The plan is stored and tracked.")
                    result['analyzed_at'] = datetime.now()
                    st.session_state.signal_history.append(result)
                    add_notification('success' if result['sent'] else 'info',
                                     f"{symbol}: {result.get('plan_state')} {result.get('signal')} LIMIT {result.get('entry')} | SL {result.get('stop_loss')} | TP1 {(result.get('take_profit') or ['N/A'])[0]} | grade {result.get('grade')} ({result.get('confluence_score')}/100){'' if result['sent'] else ' — not pushed: ' + why}",
                                     symbol=symbol, signal=result.get('signal'), score=result.get('confluence_score'))
                st.markdown("---")
        st.session_state.signal_history = st.session_state.signal_history[-200:]

with tab2:
    st.header("📜 Signal History (this session)")
    history = st.session_state.signal_history
    if not history:
        st.info("📭 No new plans generated in this session yet. Run an analysis in the Market Analysis tab. (All plans, including from earlier sessions, are in the Plan Tracker.)")
    else:
        sent_count = sum(1 for s in history if s.get('sent'))
        h1c, h2c = st.columns(2)
        h1c.metric("Plans generated", len(history))
        h2c.metric("Pushed to Telegram", sent_count)
        for signal in reversed(history):
            cfg_h = get_pair_config(signal.get('symbol', 'XAUUSD'))
            tp_list = signal.get('take_profit') or []
            with st.expander(f"{'🟢' if signal.get('signal') == 'BUY' else '🔴'} {signal.get('symbol', 'N/A')} - {signal.get('signal')} | Grade {signal.get('grade', '-')} {signal.get('confluence_score')}/100 | {'sent' if signal.get('sent') else 'not sent'} | {signal.get('timestamp', 'N/A')}", expanded=False):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Entry", _fmt(signal.get('entry'), cfg_h))
                col_b.metric("Stop Loss", _fmt(signal.get('stop_loss'), cfg_h))
                col_c.metric("TP1", _fmt(tp_list[0], cfg_h) if tp_list else 'N/A')
                col_d.metric("TP2", _fmt(tp_list[1], cfg_h) if len(tp_list) > 1 else 'N/A')
                st.write(f"**Zone:** {(signal.get('zone') or {}).get('kind')} · **Bias:** {signal.get('bias')} · **Confidence:** {signal.get('confidence')} · **DXY:** {signal.get('dxy_correlation', 'N/A')}")
                st.write(f"**Analysis:** {signal.get('model_used', 'N/A')} · **Tokens:** {signal.get('total_tokens', 'N/A')}")
                st.write(f"**Reasoning:** {signal.get('reasoning')}")

with tab3:
    st.header("🧭 Plan Tracker")
    st.caption("Every plan is created once and then tracked until it fills and resolves, is missed, expires, goes stale, or the HTF bias genuinely flips. State is saved to disk so it survives restarts of this session.")
    tstate = load_state()
    rcol1, rcol2 = st.columns([1, 1])
    if rcol1.button("🔄 Refresh live plan statuses"):
        with st.spinner("Checking live plans against the latest candles and quote..."):
            _all = fetch_all_data()
            changed = refresh_live_plans(_all)
        st.success(f"Updated: {', '.join(changed)}" if changed else "No status changes.")
        tstate = load_state()
    stats = get_plan_stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Closed plans", stats['total_closed'])
    m2.metric("Wins / Losses", f"{stats['wins']} / {stats['losses']}")
    m3.metric("Win rate", f"{stats['win_rate']:.0f}%" if stats['win_rate'] is not None else "—")
    m4.metric("Net R", stats['total_r'])
    m5.metric("Missed / Expired", f"{stats['missed']} / {stats['expired']}")
    st.subheader("Live plans")
    live_rows = []
    for sym, plan in tstate['plans'].items():
        if plan.get('status') in LIVE_PLAN_STATUSES:
            cfg_t = get_pair_config(sym)
            live_rows.append({'Symbol': sym, 'Side': plan.get('direction'), 'Status': plan.get('status'), 'Entry': _fmt(plan.get('entry'), cfg_t),
                              'SL': _fmt(plan.get('stop_loss'), cfg_t), 'TP1': _fmt((plan.get('take_profit') or [None])[0], cfg_t),
                              'Grade': plan.get('grade'), 'Sent': 'yes' if plan.get('sent') else 'no', 'Created (UTC)': str(plan.get('created_at', ''))[:16].replace('T', ' '),
                              'Zone': (plan.get('zone') or {}).get('kind')})
    if live_rows:
        st.dataframe(pd.DataFrame(live_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No live plans.")
    st.subheader("Standing HTF bias per symbol")
    bias_rows = [{'Symbol': s, 'Bias': b.get('direction'), 'Score': b.get('score'), 'Since (UTC)': str(b.get('since', ''))[:16].replace('T', ' ')} for s, b in tstate['bias'].items()]
    if bias_rows:
        st.dataframe(pd.DataFrame(bias_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No bias stored yet.")
    st.subheader("Closed plan ledger")
    ledger = tstate.get('ledger', [])
    if ledger:
        st.dataframe(pd.DataFrame(list(reversed(ledger)))[['symbol', 'direction', 'outcome', 'r', 'entry', 'stop_loss', 'tp1', 'rr', 'grade', 'score', 'zone', 'created_at', 'closed_at']], hide_index=True, use_container_width=True)
    else:
        st.caption("No closed plans yet.")

with tab4:
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

with tab5:
    st.header("⚙️ System Settings")
    st.info("Ensure `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` are set in your Streamlit Secrets.")
    st.markdown("**How a signal is decided now**")
    st.markdown(
        "1. **Direction = H4 + H1 only.** Confirmed pivots, HH/HL/LH/LL, close-based BOS/CHOCH, EMA trend and RSI regime. M10/M15/M30 have no vote.\n"
        f"2. **Hysteresis.** A standing bias only flips with a score beyond ±{BIAS_FLIP_THRESHOLD:.0f}, an opposing H1/H4 close-based structure break, and at least {BIAS_MIN_HOLD_HOURS:.0f}h since it was set.\n"
        "3. **Entry = best HTF zone.** Unmitigated H4/H1 order blocks, open FVGs, 62–79% OTE, live swings, equal highs/lows and PDH/PDL/PWH/PWL are ranked by timeframe, confluence, freshness, location and distance; M30/M15 then refine the limit *inside* the zone.\n"
        "4. **SL/TP are structural.** SL beyond the zone's distal edge (and any live swing just beyond) with an H1-ATR buffer; TP1/TP2 are the next opposing HTF levels that pay at least 1.5R.\n"
        "5. **Plan lock.** One plan per symbol is created and managed until it resolves — reruns return the same plan.\n"
        "6. **AI is an auditor.** It cannot change direction or levels; it flags risks, reads screenshots and adjusts the score by ±6."
    )
    st.markdown(f"- **AI model priority:** {' → '.join(GROQ_MODELS)} (always attempted in this order via Groq, multimodal)")
    st.markdown("- **Execution:** resting LIMIT orders only · manual trigger only (no auto-loop)")
    st.markdown(f"- **State file:** `{STATE_PATH}` (set `DERAI_STATE_PATH` to change it; on Streamlit Community Cloud the disk resets when the app is rebuilt/restarted)")
    st.markdown("---")
    st.subheader("Telegram & scoring controls")
    st.slider("Minimum score to push a signal to Telegram", 60, 90, MINIMUM_CONFLUENCE_SCORE, key="min_send_score")
    st.checkbox("Also push low-conviction plans (below the threshold / ATR fallback)", value=False, key="send_low_conviction")
    st.checkbox("Send plan events (filled / TP / SL / cancelled / missed / expired) to Telegram", value=True, key="notify_plan_events")
    if st.button("🧹 Reset stored bias, plans and ledger"):
        reset_state()
        st.success("Stored bias, plans and ledger cleared.")
    st.markdown("---")
    st.subheader("🔧 Groq Model Diagnostics")
    st.caption("GROQ_MODELS is always tried in the order below. Use this panel to see what your key reports as available.")
    for m in GROQ_MODELS:
        cfg_m = GROQ_MODEL_CONFIG.get(m, GROQ_DEFAULT_MODEL_CONFIG)
        st.write(f"**{m}** — max_completion_tokens={cfg_m['max_completion_tokens']}, reasoning_effort={cfg_m.get('reasoning_effort')}")
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
                        st.warning(f"⚠️ {m} — NOT reported by Groq's /models endpoint right now (the app will still try it)")
            else:
                st.error(f"Could not reach Groq /models endpoint: {discovery.get('reason')}")
    last_attempts = st.session_state.get('last_model_attempts') or {}
    if last_attempts:
        st.markdown("**Last analysis run — per-symbol model attempts:**")
        for sym, attempts in last_attempts.items():
            if attempts:
                st.write(f"`{sym}`: " + " → ".join(f"{a.get('model')}:{a.get('status')}" for a in attempts))
