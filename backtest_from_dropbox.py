# backtest_from_dropbox.py
# Çalıştır: streamlit run backtest_from_dropbox.py
# Gerekli paketler (requirements.txt):
#   streamlit==1.36.0
#   pandas==2.2.2
#   numpy==1.26.4
#   requests==2.32.3
#   fastparquet

import os
import time
import math
import hashlib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
from typing import Tuple

st.set_page_config(page_title="BTC/ETH Backtest (Dropbox Parquet)", layout="wide")

# ----- Senin verdiğin linkler -----
DEFAULT_FUTURES_URL = (
    "https://www.dropbox.com/scl/fi/diznny37aq4t88vf62umy/"
    "binance_futures_5m-1h-1w-1M_2020-01_2025-08_BTC_ETH.parquet"
    "?rlkey=4umoh63qiz3fh0v7xuu86oo5n&st=5wu5h1x1&dl=0"
)
DEFAULT_SPOT_URL = (
    "https://www.dropbox.com/scl/fi/eavvv8z452i0b6x1c2a5r/"
    "binance_spot_5m-1h-1w-1M_2020-01_2025-08_BTC_ETH.parquet"
    "?rlkey=swsjkpbp22v4vj68ggzony8yw&st=2sww3kao&dl=0"
)

# ---- YAZILABİLİR CACHE DİZİNİ: /tmp ----
DATA_DIR = os.path.join(tempfile.gettempdir(), "dropbox_cache")
os.makedirs(DATA_DIR, exist_ok=True)  # /tmp yazılabilir

# ---------- Yardımcılar ----------
def to_direct_link(url: str) -> str:
    """Dropbox paylaşım linkini doğrudan indirme (dl=1) formatına çevirir."""
    if "dropbox.com" in url and "dl=0" in url:
        url = url.replace("dl=0", "dl=1")
    return url

def stream_download(url: str, dst_path: str, chunk=1024*1024, timeout=120):
    """Dosyayı akış halinde indirir; HTTP hatalarında exception fırlatır."""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0
        t0 = time.time()
        with open(dst_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if not part:
                    continue
                f.write(part)
                downloaded += len(part)
                if total:
                    pct = 100 * downloaded / total
                    st.caption(f"İndiriliyor… {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)")
        st.success(f"İndirme tamamlandı: {downloaded/1e6:.1f} MB, {time.time()-t0:.1f} sn")

def ensure_local_copy(name: str, url: str) -> str:
    """
    Yerelde yoksa indirir, varsa kullanır.
    - HTTP/bağlantı/erişim hatalarında exception fırlatır (UI yakalar).
    """
    local_path = os.path.join(DATA_DIR, name)
    if not os.path.exists(local_path):
        st.info("Yerelde bulunamadı, Dropbox’tan indiriliyor…")
        direct = to_direct_link(url)
        stream_download(direct, local_path)
    else:
        st.caption(f"Yerelde bulundu: {local_path}")
    return local_path

@st.cache_data(show_spinner=False)
def load_parquet(local_path: str) -> pd.DataFrame:
    """
    Parquet'i okur; DatetimeIndex (UTC) kurar; zorunlu kolonları doğrular.
    Beklenen kolonlar: open, high, low, close, volume, symbol, timeframe (+ timestamp veya DatetimeIndex)
    """
    df = pd.read_parquet(local_path)  # engine="auto" -> fastparquet kullanabilir
    # Zaman indeksini hazırla
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            raise ValueError("Parquet’te DatetimeIndex yok ve 'timestamp' kolonu bulunamadı.")
    # UTC garanti
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Zorunlu kolonlar
    needed = {"open","high","low","close","volume","symbol","timeframe"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen kolon(lar) eksik: {missing}")

    # Tip düzeltmeleri
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["timeframe"] = df["timeframe"].astype(str)

    return df.sort_index()

# ---------- İndikatörler ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def is_engulfing(df: pd.DataFrame, bullish=True) -> pd.Series:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    po, pc, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    if bullish:
        cond = (pc < po) & (c > o) & (h > ph) & (l < pl)
    else:
        cond = (pc > po) & (c < o) & (h < ph) & (l > pl)
    return cond.fillna(False)

# ---------- Sinyaller ----------
def make_signals(df: pd.DataFrame,
                 ema_fast=7, ema_mid=13, ema_slow=26,
                 atr_len=14,
                 pullback_atr_thresh=0.5,
                 regime_filter=None) -> pd.DataFrame:
    """
    Basit kural seti:
    - EMA hizası: bull => EMAfast>EMA13>EMA26; bear => ters.
    - Pullback: |close-EMA_mid|/ATR < eşik
    - Engulfing: yönle uyumlu
    - regime_filter: 'long-only', 'short-only' ya da None
    """
    out = df.copy()
    out["ema_f"] = ema(out["close"], ema_fast)
    out["ema_m"] = ema(out["close"], ema_mid)
    out["ema_s"] = ema(out["close"], ema_slow)
    out["ATR"]   = atr(out, atr_len)

    out["bull_align"] = (out["ema_f"] > out["ema_m"]) & (out["ema_m"] > out["ema_s"])
    out["bear_align"] = (out["ema_f"] < out["ema_m"]) & (out["ema_m"] < out["ema_s"])

    dist = (out["close"] - out["ema_m"]).abs() / out["ATR"].replace(0, np.nan)
    out["near_mid"] = dist < pullback_atr_thresh

    out["bull_eng"] = is_engulfing(out, bullish=True)
    out["bear_eng"] = is_engulfing(out, bullish=False)

    long_raw  = out["bull_align"] & out["near_mid"] & out["bull_eng"]
    short_raw = out["bear_align"] & out["near_mid"] & out["bear_eng"]

    if regime_filter == "long-only":
        short_raw = pd.Series(False, index=out.index)
    elif regime_filter == "short-only":
        long_raw = pd.Series(False, index=out.index)

    out["long_signal"]  = long_raw
    out["short_signal"] = short_raw
    return out

# ---------- Backtest ----------
def backtest(df: pd.DataFrame,
             rr=2.0,
             atr_stop_mult=2.0,
             entry_offset_bps=0.0,
             one_trade_at_a_time=True,
             fee_rate=0.0004,      # her bacak (aç/kapa)
             initial_equity=1000.0,
             risk_mode="dynamic",  # "fixed" or "dynamic"
             fixed_amount=100.0,
             risk_pct=0.02,
             leverage=10.0) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Basitleştirilmiş motor:
    - Sinyal mumunun kapanışı sonrası bir sonraki barın açılışında giriş.
    - SL = ATR * atr_stop_mult; TP = rr * SL
    - Aynı anda tek pozisyon.
    - Komisyon: nominal * fee_rate (açılış + kapanış).
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    atrv = df["ATR"].fillna(method="bfill").fillna(method="ffill")
    long_sig  = df["long_signal"].fillna(False)
    short_sig = df["short_signal"].fillna(False)

    trades = []
    in_pos = False
    side = None
    entry_price = sl = tp = np.nan

    equity = initial_equity
    eq_curve = []

    def position_size(price):
        nonlocal equity
        if risk_mode == "fixed":
            nominal = fixed_amount * leverage
        else:
            nominal = (equity * risk_pct) * leverage
        qty = nominal / price if price else 0.0
        return max(qty, 0.0), nominal

    off_mult = 1.0 + (entry_offset_bps / 10000.0)

    for i in range(2, len(df)):
        ts = df.index[i]
        if not in_pos:
            if long_sig.iloc[i-1]:
                ent = o.iloc[i] * off_mult
                sl_dist = atrv.iloc[i] * atr_stop_mult
                entry_price = float(ent); sl = entry_price - sl_dist; tp = entry_price + rr * sl_dist
                side = "long"; in_pos = True
                qty, nominal = position_size(entry_price)
                fee_open = nominal * fee_rate
                trades.append({"time": ts, "side": side, "entry": entry_price, "sl": sl, "tp": tp,
                               "qty": qty, "nominal": nominal, "fee_open": fee_open})
            elif short_sig.iloc[i-1]:
                ent = o.iloc[i] / off_mult
                sl_dist = atrv.iloc[i] * atr_stop_mult
                entry_price = float(ent); sl = entry_price + sl_dist; tp = entry_price - rr * sl_dist
                side = "short"; in_pos = True
                qty, nominal = position_size(entry_price)
                fee_open = nominal * fee_rate
                trades.append({"time": ts, "side": side, "entry": entry_price, "sl": sl, "tp": tp,
                               "qty": qty, "nominal": nominal, "fee_open": fee_open})
        else:
            hi, lo = h.iloc[i], l.iloc[i]
            tp_hit = (hi >= tp) if side == "long" else (lo <= tp)
            sl_hit = (lo <= sl) if side == "long" else (hi >= sl)

            exit_price = None; result = None
            if tp_hit and sl_hit:
                result = "SL_first"; exit_price = sl   # konservatif
            elif tp_hit:
                result = "TP"; exit_price = tp
            elif sl_hit:
                result = "SL"; exit_price = sl

            if exit_price is not None:
                last = trades[-1]; nominal = last["nominal"]; qty = last["qty"]
                fee_close = nominal * fee_rate
                pnl = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
                net = pnl - (last["fee_open"] + fee_close)
                equity += net
                last.update({"exit_time": ts, "exit": exit_price, "result": result,
                             "pnl": pnl, "net": net, "fee_close": fee_close, "equity_after": equity})
                in_pos = False; side = None; entry_price = sl = tp = np.nan

        eq_curve.append((ts, equity))

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq_curve, columns=["time","equity"]).set_index("time")

    if trades_df.empty:
        stats = {"trades": 0, "winrate": 0.0, "net_total": 0.0, "max_dd": 0.0, "final_equity": float(equity)}
    else:
        wins = (trades_df["net"] > 0).sum()
        net_total = trades_df["net"].sum()
        winrate = 100.0 * wins / len(trades_df)
        eq = eq_df["equity"].fillna(method="ffill").fillna(initial_equity)
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min() * 100.0)
        stats = {
            "trades": int(len(trades_df)),
            "winrate": float(winrate),
            "net_total": float(net_total),
            "max_dd": max_dd,
            "final_equity": float(eq.iloc[-1])
        }
    return trades_df, stats, eq_df

# ---------- UI ----------
st.title("BTC/ETH • Spot & Futures Backtest (Dropbox Parquet)")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    dataset_choice = st.radio("Dataset", ["Futures", "Spot"], index=0, horizontal=True)
    fut_url = st.text_input("Futures URL", value=DEFAULT_FUTURES_URL)
    spot_url = st.text_input("Spot URL", value=DEFAULT_SPOT_URL)

    st.subheader("Seçimler")
    wanted_tf = st.selectbox("Timeframe", ["5m","1h","1w","1M"], index=0)
    wanted_symbol = st.selectbox("Sembol", ["BTCUSDT","ETHUSDT"], index=0)
    regime = st.selectbox("Rejim filtresi", ["Hepsi","long-only","short-only"], index=0)

    st.subheader("Sinyal Parametreleri")
    ema_f = st.number_input("EMA Hızlı", min_value=2,  max_value=200, value=7)
    ema_m = st.number_input("EMA Orta",  min_value=2,  max_value=400, value=13)
    ema_s = st.number_input("EMA Yavaş", min_value=3,  max_value=600, value=26)
    atr_len = st.number_input("ATR Periyodu", min_value=2, max_value=200, value=14)
    pb_thr = st.number_input("Pullback Eşiği (|C-EMA13|/ATR)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)

    st.subheader("Risk/Ödül & Pozisyon")
    rr = st.number_input("Risk/Ödül (TP/SL)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    atr_mult = st.number_input("SL = ATR ×", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    entry_off = st.number_input("Giriş Offset (bps)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    st.subheader("Sermaye / İşlem")
    init_eq = st.number_input("Başlangıç Sermaye (USDT)", min_value=100.0, max_value=1_000_000.0, value=1000.0, step=100.0)
    lev = st.number_input("Kaldıraç (x)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    fee = st.number_input("Komisyon (her bacak)", min_value=0.0, max_value=0.005, value=0.0004, step=0.0001, format="%.4f")
    risk_mode = st.selectbox("Risk Modu", ["fixed","dynamic"], index=1)
    fixed_amt = st.number_input("Sabit Nominal (USDT)", min_value=10.0, max_value=100000.0, value=100.0, step=10.0)
    risk_pct = st.number_input("Risk % (dynamic)", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f")

    run_btn = st.button("Yükle & Backtest", type="primary")

# ---------- Ana akış ----------
if run_btn:
    # 1) Linki seç ve indir
    use_url = fut_url if dataset_choice == "Futures" else spot_url
    stub = "futures" if dataset_choice == "Futures" else "spot"
    local_name = f"{stub}_btc_eth.parquet"

    try:
        local_path = ensure_local_copy(local_name, use_url)
    except requests.HTTPError as e:
        st.error(f"HTTP hatası: {e}. Linki ve paylaşım ayarını kontrol et (dl=1).")
        st.stop()
    except requests.RequestException as e:
        st.error(f"Ağ hatası: {e}")
        st.stop()
    except Exception as e:
        st.error(f"İndirme hatası: {e}")
        st.stop()

    # 2) Oku
    try:
        big = load_parquet(local_path)
    except Exception as e:
        st.error(f"Parquet okuma hatası: {e}")
        st.stop()

    st.success(f"Yüklendi: {len(big):,} satır | Semboller: {sorted(big['symbol'].unique())} | TF: {sorted(big['timeframe'].unique())}")

    # 3) Filtrele
    view = big[(big["symbol"] == wanted_symbol) & (big["timeframe"] == wanted_tf)].copy()
    if view.empty:
        st.error("Seçilen sembol/timeframe için veri bulunamadı.")
        st.stop()

    # 4) Sinyaller
    regime_arg = None if regime == "Hepsi" else regime
    sig = make_signals(
        view,
        ema_fast=ema_f, ema_mid=ema_m, ema_slow=ema_s,
        atr_len=atr_len, pullback_atr_thresh=pb_thr,
        regime_filter=regime_arg
    )

    # 5) Backtest
    trades, stats, eq = backtest(
        sig,
        rr=rr, atr_stop_mult=atr_mult, entry_offset_bps=entry_off,
        one_trade_at_a_time=True, fee_rate=fee,
        initial_equity=init_eq, risk_mode=risk_mode, fixed_amount=fixed_amt,
        risk_pct=risk_pct, leverage=lev
    )

    # 6) Sonuçlar
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Equity Eğrisi")
        st.line_chart(eq)
    with c2:
        st.subheader("Özet")
        st.write({
            "Toplam İşlem": stats.get("trades", 0),
            "Winrate %": round(stats.get("winrate", 0.0), 2),
            "Net Toplam (USDT)": round(stats.get("net_total", 0.0), 2),
            "Final Equity": round(stats.get("final_equity", 0.0), 2),
            "Max Drawdown %": round(stats.get("max_dd", 0.0), 2),
        })

    st.subheader("İşlemler (son 200)")
    if len(trades):
        st.dataframe(trades.tail(200), use_container_width=True)
    else:
        st.info("İşlem üretilmedi. Parametreleri değiştirip tekrar dene.")

    with st.expander("Sinyal / İndikatör Görünümü (son 20 bar)"):
        cols = ["ema_f","ema_m","ema_s","ATR","bull_align","bear_align","near_mid","bull_eng","bear_eng","long_signal","short_signal"]
        st.dataframe(sig[cols].tail(20), use_container_width=True)
