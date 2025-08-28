# backtest_from_dropbox.py

import streamlit as st
st.set_page_config(page_title="LRC + EMA Backtest (5m giriş + 1h onay)", layout="wide")

import sys, pkgutil, importlib.metadata as imd
import pandas as pd
import numpy as np
import requests
import io

# -------- Teşhis --------
st.caption(f"Python: {sys.version}")
st.caption("plotly modülü görüldü mü? → " + ("Evet" if pkgutil.find_loader("plotly") else "Hayır"))
try:
    _names = {d.metadata["Name"].lower() for d in imd.distributions()}
    st.caption("Yüklü plotly sürümü: " + (imd.version("plotly") if "plotly" in _names else "yok"))
except Exception:
    pass

import plotly.graph_objects as go

# ---------- Dropbox Linkleri ----------
DEFAULT_FUTURES_URL = "https://www.dropbox.com/scl/fi/diznny37aq4t88vf62umy/binance_futures_5m-1h-1w-1M_2020-01_2025-08_BTC_ETH.parquet?rlkey=4umoh63qiz3fh0v7xuu86oo5n&st=x0uw3obl&dl=1"
DEFAULT_SPOT_URL    = "https://www.dropbox.com/scl/fi/eavvv8z452i0b6x1c2a5r/binance_spot_5m-1h-1w-1M_2020-01_2025-08_BTC_ETH.parquet?rlkey=swsjkpbp22v4vj68ggzony8yw&st=2sww3kao&dl=1"

# ---------- Yardımcı Fonksiyonlar ----------
def load_parquet_from_dropbox(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))
    except Exception as e:
        st.error(f"Veri alınamadı: {e}")
        return pd.DataFrame()

# EMA hesapları
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

# LRC hesapları
def _lrc_last_point(values: np.ndarray) -> float:
    w = np.asarray(values, dtype=float).ravel()
    n = w.size
    if n < 2 or not np.isfinite(w).all():
        return np.nan
    x = np.arange(n, dtype=float)
    m, b = np.polyfit(x, w, 1)
    return m * (n - 1) + b

def rolling_lrc(series: pd.Series, length: int = 300) -> pd.Series:
    return series.rolling(window=length, min_periods=length).apply(_lrc_last_point, raw=True)

def compute_lrc_bands(df: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['lrc_high'] = rolling_lrc(df['high'], length=length)
    out['lrc_low']  = rolling_lrc(df['low'],  length=length)
    return out

# Backtest
def backtest(df: pd.DataFrame, use_lrc_long: bool, use_lrc_short: bool) -> dict:
    df = df.copy()
    df["ema7"] = ema(df["close"], 7)
    df["ema13"] = ema(df["close"], 13)
    df["ema26"] = ema(df["close"], 26)

    lrc = compute_lrc_bands(df)
    df = pd.concat([df, lrc], axis=1)

    trades = []
    position = None
    entry_price = None
    wins = 0
    losses = 0

    for i in range(len(df)):
        row = df.iloc[i]
        if i < 300:  # warmup
            continue

        # Koşullar: 5m EMA hizalı + 1h EMA hizalı (bu örnekte tek timeframe üzerinden basitçe gösteriyoruz)
        ema_ok = row["ema7"] > row["ema13"] > row["ema26"]

        if position is None:
            # Long
            if ema_ok and use_lrc_long and row["close"] > row["lrc_high"]:
                position = "long"
                entry_price = row["close"]
            # Short
            elif not ema_ok and use_lrc_short and row["close"] < row["lrc_low"]:
                position = "short"
                entry_price = row["close"]

        else:
            # Exit koşulu: ters sinyal
            if position == "long" and row["close"] < row["lrc_low"]:
                if row["close"] > entry_price:
                    wins += 1
                else:
                    losses += 1
                trades.append((position, entry_price, row["close"]))
                position = None

            elif position == "short" and row["close"] > row["lrc_high"]:
                if row["close"] < entry_price:
                    wins += 1
                else:
                    losses += 1
                trades.append((position, entry_price, row["close"]))
                position = None

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) if trades else 0,
    }

# ---------- Streamlit Arayüz ----------
st.title("📊 LRC + EMA Backtest")

data_type = st.radio("Veri seç:", ["Futures", "Spot"])
if data_type == "Futures":
    df = load_parquet_from_dropbox(DEFAULT_FUTURES_URL)
else:
    df = load_parquet_from_dropbox(DEFAULT_SPOT_URL)

if not df.empty:
    st.write("Veri boyutu:", df.shape)

    use_lrc_long = st.checkbox("LRC üstünde long")
    use_lrc_short = st.checkbox("LRC altında short")

    if st.button("Backtest Çalıştır"):
        stats = backtest(df, use_lrc_long, use_lrc_short)
        st.success(f"Toplam İşlem: {stats['trades']} | Win rate: {stats['win_rate']:.2%}")
