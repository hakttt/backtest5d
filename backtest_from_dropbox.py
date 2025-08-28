# backtest_from_dropbox.py
# Çalıştır: streamlit run backtest_from_dropbox.py
# Gerekli paketler (requirements.txt):
#   streamlit==1.36.0
#   pandas==2.2.2
#   numpy==1.26.4
#   requests==2.32.3
#   fastparquet==2024.5.0
#   plotly==5.22.0

import os
import time
import io
import hashlib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
from typing import Tuple
import plotly.graph_objects as go

st.set_page_config(page_title="BTC/ETH Backtest (Dropbox Parquet)", layout="wide")

# ----- Varsayılan linkler -----
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

# ---- Yazılabilir cache dizini (/tmp) ----
DATA_DIR = os.path.join(tempfile.gettempdir(), "dropbox_cache")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Yardımcılar ----------
def to_direct_link(url: str) -> str:
    if "dropbox.com" in url and "dl=0" in url:
        url = url.replace("dl=0", "dl=1")
    return url

def stream_download(url: str, dst_path: str, chunk=1024*1024, timeout=120):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)

def ensure_local_copy(name: str, url: str) -> str:
    local_path = os.path.join(DATA_DIR, name)
    if not os.path.exists(local_path):
        direct = to_direct_link(url)
        stream_download(direct, local_path)
    return local_path

@st.cache_data(show_spinner=False)
def load_parquet(local_path: str) -> pd.DataFrame:
    df = pd.read_parquet(local_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            raise ValueError("DatetimeIndex veya 'timestamp' kolonu yok.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    # Zorunlu kolon kontrolü
    needed = {"open","high","low","close","volume","symbol","timeframe"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Beklenen kolon(lar) eksik: {miss}")
    # Tipler
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["timeframe"] = df["timeframe"].astype(str)
    return df.sort_index()

# ---------- LRC Fonksiyonları ----------
def _lrc_last_point(values: np.ndarray) -> float:
    w = np.asarray(values, dtype=float).ravel()
    n = w.size
    if n < 2 or not np.isfinite(w).all():
        return np.nan
    x = np.arange(n, dtype=float)
    m, b = np.polyfit(x, w, 1)
    return m * (n - 1) + b

def rolling_lrc(series: pd.Series, length: int = 300) -> pd.Series:
    if series is None or len(series) < length:
        return pd.Series(index=getattr(series, "index", None), dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=length, min_periods=length).apply(_lrc_last_point, raw=True)

def compute_lrc_bands(df: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['lrc_high'] = rolling_lrc(df['high'], length=length)
    out['lrc_low']  = rolling_lrc(df['low'],  length=length)
    return out

@st.cache_data(show_spinner=False)
def cached_lrc_bands(df_1d: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    if df_1d is None or df_1d.empty:
        return pd.DataFrame(index=getattr(df_1d, "index", None))
    # Basit içerik hash'i ile cache anahtarı
    key = (str(df_1d.index.min()) + str(df_1d.index.max()) +
           str(float(df_1d["close"].sum())))
    _ = hashlib.md5(key.encode()).hexdigest()
    return compute_lrc_bands(df_1d, length=length)

# ---------- EMA / ATR ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

# ---------- 5m + 1h EMA + Opsiyonel LRC ----------
def make_signals_5m_with_1h_and_lrc(df_5m, df_1h, df_1d,
                                    atr_len=14, regime_filter=None,
                                    use_lrc=False, allow_long=True, allow_short=True):
    """
    Sabit EMA çekirdeği:
      - 5m: EMA(7)>EMA(13)>EMA(26) => long ; EMA(7)<EMA(13)<EMA(26) => short
      - 1h: yön teyidi (EMA7 vs EMA13)
    Opsiyonel LRC filtresi (1D LRC-300):
      - close > lrc_high -> sadece long
      - close < lrc_low  -> sadece short
      - aradaysa -> iki yön de engellenir
    """
    ema_fast, ema_mid, ema_slow = 7, 13, 26

    out = df_5m.copy()
    out["ema_f"] = ema(out["close"], ema_fast)
    out["ema_m"] = ema(out["close"], ema_mid)
    out["ema_s"] = ema(out["close"], ema_slow)

    h = df_1h.copy()
    h["ema_f_h"] = ema(h["close"], ema_fast)
    h["ema_m_h"] = ema(h["close"], ema_mid)
    h["bull_align_h"] = h["ema_f_h"] > h["ema_m_h"]
    h["bear_align_h"] = h["ema_f_h"] < h["ema_m_h"]

    bull_5m = (out["ema_f"] > out["ema_m"]) & (out["ema_m"] > out["ema_s"])
    bear_5m = (out["ema_f"] < out["ema_m"]) & (out["ema_m"] < out["ema_s"])

    key_5m = out.index.floor("H")
    bull_1h_on_5m = h["bull_align_h"].reindex(key_5m).ffill().reindex(out.index)
    bear_1h_on_5m = h["bear_align_h"].reindex(key_5m).ffill().reindex(out.index)

    long_raw  = bull_5m & bull_1h_on_5m
    short_raw = bear_5m & bear_1h_on_5m

    if use_lrc and not df_1d.empty:
        bands = cached_lrc_bands(df_1d, length=300)
        lrc_high = bands['lrc_high'].reindex(out.index, method='ffill')
        lrc_low  = bands['lrc_low'].reindex(out.index, method='ffill')
        above = out["close"] > lrc_high
        below = out["close"] < lrc_low
        long_raw  = long_raw  & above & bool(allow_long)
        short_raw = short_raw & below & bool(allow_short)

    if regime_filter == "long-only":
        short_raw = pd.Series(False, index=out.index)
    elif regime_filter == "short-only":
        long_raw = pd.Series(False, index=out.index)

    out["ATR"] = atr(out, atr_len)
    out["long_signal"]  = long_raw.fillna(False)
    out["short_signal"] = short_raw.fillna(False)
    return out

# ---------- Swing level yardımcıları ----------
def _last_swing_low(high: pd.Series, low: pd.Series, lookback: int = 10) -> float:
    end = len(low) - 1
    start = max(2, end - lookback)
    for i in range(end - 1, start - 1, -1):
        if i-1 >= 0 and i+1 <= end:
            if (low.iloc[i] < low.iloc[i-1]) and (low.iloc[i] < low.iloc[i+1]):
                return float(low.iloc[i])
    return float(low.iloc[start:end].min()) if end > start else float(low.iloc[end])

def _last_swing_high(high: pd.Series, low: pd.Series, lookback: int = 10) -> float:
    end = len(high) - 1
    start = max(2, end - lookback)
    for i in range(end - 1, start - 1, -1):
        if i-1 >= 0 and i+1 <= end:
            if (high.iloc[i] > high.iloc[i-1]) and (high.iloc[i] > high.iloc[i+1]):
                return float(high.iloc[i])
    return float(high.iloc[start:end].max()) if end > start else float(high.iloc[end])

# ---------- Yardımcı metrik: consecutive win/loss ----------
def _consecutive_streaks(win_flags: pd.Series) -> Tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for v in win_flags.astype(bool).tolist():
        if v:
            cur_w += 1; max_w = max(max_w, cur_w)
            cur_l = 0
        else:
            cur_l += 1; max_l = max(max_l, cur_l)
            cur_w = 0
    return max_w, max_l

# ---------- Backtest ----------
def backtest(df: pd.DataFrame,
             stop_type: str = "Swing",     # "Swing" | "ATR"
             tp_percent: float = 10.0,     # Swing modunda kullanılır
             rr: float = 2.0,              # ATR modunda kullanılır
             atr_stop_mult: float = 2.0,   # ATR modunda kullanılır
             entry_offset_bps: float = 0.0,
             fee_rate: float = 0.0004,     # her bacak için
             initial_equity: float = 1000.0,
             risk_mode: str = "dynamic",   # "fixed" or "dynamic"
             fixed_amount: float = 100.0,
             risk_pct: float = 0.02,
             leverage: float = 10.0,
             swing_lookback: int = 10,
             pip_size: float = 0.01) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    - Giriş: sinyal mumunun kapanışı sonrası bir sonraki barın açılışı (offset bps uygulanır).
    - Stop tipi:
        * Swing: SL = son swing low/high ± 1 pip, TP = giriş ± %TP
        * ATR  : SL = ATR × mult, TP = RR × SL
    - Komisyon: (nominal * fee_rate) açılış + kapanış.
    - Pozisyon boyutu: fixed nominal veya equity*risk_pct (her ikisi de leverage ile çarpılır).
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    atrv = df.get("ATR", pd.Series(index=df.index, dtype=float)).fillna(method="ffill")

    long_sig  = df["long_signal"].fillna(False)
    short_sig = df["short_signal"].fillna(False)

    trades = []
    in_pos = False
    side = None
    entry_price = sl = tp = np.nan
    qty = nominal = 0.0

    equity = initial_equity
    eq_curve = []

    def position_size(price):
        nonlocal equity
        nom = fixed_amount * leverage if risk_mode == "fixed" else (equity * risk_pct) * leverage
        q = nom / price if price else 0.0
        return max(q, 0.0), nom

    off_mult = 1.0 + (entry_offset_bps / 10000.0)

    for i in range(2, len(df)):
        ts = df.index[i]

        if not in_pos:
            if long_sig.iloc[i-1] or short_sig.iloc[i-1]:
                ent = o.iloc[i]
                ent = ent * off_mult if long_sig.iloc[i-1] else ent / off_mult

                qty, nominal = position_size(ent)
                fee_open = nominal * fee_rate

                if long_sig.iloc[i-1]:
                    side = "long"
                    if stop_type == "Swing":
                        swing_low = _last_swing_low(h, l, lookback=swing_lookback)
                        sl = max(0.0, swing_low - pip_size)
                        tp = ent * (1.0 + tp_percent / 100.0)
                    else:
                        sl_dist = atrv.iloc[i] * atr_stop_mult
                        sl = ent - sl_dist
                        tp = ent + rr * sl_dist
                else:
                    side = "short"
                    if stop_type == "Swing":
                        swing_high = _last_swing_high(h, l, lookback=swing_lookback)
                        sl = swing_high + pip_size
                        tp = ent * (1.0 - tp_percent / 100.0)
                    else:
                        sl_dist = atrv.iloc[i] * atr_stop_mult
                        sl = ent + sl_dist
                        tp = ent - rr * sl_dist

                entry_price = float(ent)
                in_pos = True
                trades.append({
                    "time": ts, "side": side, "entry": entry_price,
                    "sl": float(sl), "tp": float(tp),
                    "qty": float(qty), "nominal": float(nominal),
                    "fee_open": float(fee_open)
                })

        else:
            hi, lo = h.iloc[i], l.iloc[i]
            tp_hit = (hi >= tp) if side == "long" else (lo <= tp)
            sl_hit = (lo <= sl) if side == "long" else (hi >= sl)

            exit_price = None; result = None
            if tp_hit and sl_hit:
                result = "SL_first"; exit_price = sl
            elif tp_hit:
                result = "TP"; exit_price = tp
            elif sl_hit:
                result = "SL"; exit_price = sl

            if exit_price is not None:
                fee_close = nominal * fee_rate
                pnl = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
                net = pnl - (trades[-1]["fee_open"] + fee_close)
                equity += net
                trades[-1].update({
                    "exit_time": ts, "exit": float(exit_price),
                    "result": result, "pnl": float(pnl), "net": float(net),
                    "fee_close": float(fee_close), "equity_after": float(equity)
                })
                in_pos = False; side = None; entry_price = sl = tp = np.nan

        eq_curve.append((ts, equity))

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq_curve, columns=["time", "equity"]).set_index("time")

    # Özet istatistikler + consecutive seriler
    if trades_df.empty:
        stats = {
            "trades": 0, "wins": 0, "losses": 0,
            "winrate": 0.0, "net_total": 0.0,
            "max_dd": 0.0, "final_equity": float(equity),
            "max_consec_wins": 0, "max_consec_losses": 0
        }
    else:
        wins = (trades_df["net"] > 0).sum()
        losses = (trades_df["net"] <= 0).sum()
        net_total = float(trades_df["net"].sum())
        eq = eq_df["equity"].fillna(method="ffill").fillna(initial_equity)
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min() * 100.0)
        winrate = 100.0 * wins / len(trades_df)
        win_flags = trades_df["net"] > 0
        mcw, mcl = _consecutive_streaks(win_flags)
        stats = {
            "trades": int(len(trades_df)),
            "wins": int(wins),
            "losses": int(losses),
            "winrate": float(winrate),
            "net_total": net_total,
            "max_dd": max_dd,
            "final_equity": float(eq.iloc[-1]),
            "max_consec_wins": int(mcw),
            "max_consec_losses": int(mcl)
        }
    return trades_df, stats, eq_df

# ---------- UI ----------
st.title("BTC/ETH • Spot & Futures Backtest (Dropbox Parquet)")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    dataset_choice = st.radio("Dataset", ["Futures","Spot"], index=0, horizontal=True)
    fut_url = st.text_input("Futures URL", value=DEFAULT_FUTURES_URL)
    spot_url = st.text_input("Spot URL", value=DEFAULT_SPOT_URL)

    st.subheader("Seçimler")
    wanted_symbol = st.selectbox("Sembol", ["BTCUSDT","ETHUSDT"], index=0)
    regime = st.selectbox("Rejim filtresi", ["Hepsi","long-only","short-only"], index=0)

    st.subheader("Tarih Aralığı")
    date_range = st.date_input("İşlem aralığı (UTC)", value=None, help="Başlangıç ve bitiş tarihi seçiniz")

    st.subheader("Opsiyonel LRC Filtre")
    use_lrc = st.checkbox("LRC filtresi (1D LRC-300)", value=False)
    allow_long = st.checkbox("LRC long yalnız", value=True)
    allow_short = st.checkbox("LRC short yalnız", value=True)

    st.subheader("Grafik Overlay")
    show_ema = st.checkbox("Grafikte EMA(7/13/26) çizgilerini göster", value=False)
    show_lrc_overlay = st.checkbox("Grafikte 1D LRC(300) bandını göster", value=False)
    draw_tp_sl = st.checkbox("İşlemlerde TP/SL seviyelerini çiz", value=False)
    plot_n_trades = st.number_input("Son N trade çiz (görsel)", min_value=10, max_value=2000, value=300, step=10)

    st.subheader("Stop / Hedef Ayarları")
    stop_type = st.selectbox("Stop Tipi", ["Swing","ATR"], index=0)
    tp_percent = st.slider("TP % (Swing için)", min_value=1, max_value=50, value=10, step=1)
    rr = st.number_input("Risk/Ödül (TP/SL) [ATR için]", 0.5, 10.0, 2.0, 0.1)
    atr_mult = st.number_input("SL = ATR × [ATR için]", 0.5, 10.0, 2.0, 0.1)
    entry_off = st.number_input("Giriş Offset (bps)", 0.0, 100.0, 0.0, 1.0)

    st.subheader("Sermaye / İşlem")
    init_eq = st.number_input("Başlangıç Sermaye", 100.0, 1_000_000.0, 1000.0, 100.0)
    lev = st.number_input("Kaldıraç (x)", 1.0, 100.0, 10.0, 1.0)
    fee = st.number_input("Komisyon (her bacak)", 0.0, 0.005, 0.0004, 0.0001, format="%.4f")
    risk_mode = st.selectbox("Risk Modu", ["fixed","dynamic"], index=1)
    fixed_amt = st.number_input("Sabit Nominal", 10.0, 100000.0, 100.0, 10.0)
    risk_pct = st.number_input("Risk % (dynamic)", 0.001, 0.2, 0.02, 0.001, format="%.3f")

    run_btn = st.button("Yükle & Backtest", type="primary")

# ---------- Ana akış ----------
if run_btn:
    use_url = fut_url if dataset_choice == "Futures" else spot_url
    stub = "futures" if dataset_choice == "Futures" else "spot"
    try:
        local_path = ensure_local_copy(f"{stub}_btc_eth.parquet", use_url)
        big = load_parquet(local_path)
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        st.stop()

    # --- Sembol/timeframe kırp ---
    df_5m_all = big[(big["symbol"] == wanted_symbol) & (big["timeframe"].str.lower() == "5m")].copy()
    df_1h_all = big[(big["symbol"] == wanted_symbol) & (big["timeframe"].str.lower() == "1h")].copy()
    has_1d = any(tf.lower() == "1d" for tf in big["timeframe"].unique())
    df_1d_all = big[(big["symbol"] == wanted_symbol) & (big["timeframe"].str.lower() == "1d")].copy() if has_1d else pd.DataFrame()

    if df_5m_all.empty or df_1h_all.empty:
        st.error("5m veya 1h veri yok.")
        st.stop()

    # --- Tarih aralığı uygula (varsa) ---
    if date_range:
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date = pd.Timestamp(date_range[0], tz="UTC")
            end_date = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        else:
            start_date = pd.Timestamp(date_range, tz="UTC")
            end_date = start_date + pd.Timedelta(days=1)
        df_5m = df_5m_all.loc[(df_5m_all.index >= start_date) & (df_5m_all.index < end_date)].copy()
        df_1h = df_1h_all.loc[(df_1h_all.index >= start_date) & (df_1h_all.index < end_date)].copy()
        df_1d = df_1d_all.loc[(df_1d_all.index >= start_date) & (df_1d_all.index < end_date)].copy() if has_1d else pd.DataFrame()
    else:
        df_5m, df_1h, df_1d = df_5m_all, df_1h_all, df_1d_all

    # --- Bar yeterliliği kontrolleri ---
    if len(df_5m) < 40:
        st.warning("Seçilen aralıkta 5m bar sayısı yetersiz (>= 40 önerilir).")
    if len(df_1h) < 20:
        st.warning("Seçilen aralıkta 1h bar sayısı yetersiz (>= 20 önerilir).")
    if (use_lrc or show_lrc_overlay):
        if df_1d.empty:
            st.warning("LRC kullanılacak/gösterilecek, ancak 1D veri yok.")
        elif len(df_1d) < 300:
            st.warning("LRC(300) için 1D bar sayısı yetersiz (>= 300 gerekir).")

    # --- Sinyaller ---
    regime_arg = None if regime == "Hepsi" else regime
    sig = make_signals_5m_with_1h_and_lrc(
        df_5m, df_1h, df_1d,
        atr_len=14,
        regime_filter=regime_arg,
        use_lrc=use_lrc,
        allow_long=allow_long,
        allow_short=allow_short
    )

    # --- Backtest ---
    trades, stats, eq = backtest(
        sig,
        stop_type=stop_type,
        tp_percent=float(tp_percent),
        rr=rr,
        atr_stop_mult=atr_mult,
        entry_offset_bps=entry_off,
        fee_rate=fee,
        initial_equity=init_eq,
        risk_mode=risk_mode,
        fixed_amount=fixed_amt,
        risk_pct=risk_pct,
        leverage=lev,
        swing_lookback=10,
        pip_size=0.01
    )

    # ---- Sonuçlar ----
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Equity Eğrisi (Streamlit)")
        st.line_chart(eq)
    with c2:
        st.subheader("Özet")
        st.write({
            "Toplam İşlem": stats.get("trades", 0),
            "Kazanan": stats.get("wins", 0),
            "Kaybeden": stats.get("losses", 0),
            "Winrate %": round(stats.get("winrate", 0.0), 2),
            "Max Consec Wins": stats.get("max_consec_wins", 0),
            "Max Consec Losses": stats.get("max_consec_losses", 0),
            "Net Toplam (USDT)": round(stats.get("net_total", 0.0), 2),
            "Final Equity": round(stats.get("final_equity", 0.0), 2),
            "Max Drawdown %": round(stats.get("max_dd", 0.0), 2),
        })

    # ---- Plotly: Equity (ek) + CSV indir ----
    st.subheader("Equity Eğrisi (Plotly)")
    if not eq.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name="Equity"))
        fig_eq.update_layout(height=350, xaxis_title="Time (UTC)", yaxis_title="Equity (USDT)")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Equity CSV indir
        csv_buf_eq = io.StringIO()
        eq.reset_index().rename(columns={"index":"time"}).to_csv(csv_buf_eq, index=False)
        st.download_button(
            label="Equity CSV indir",
            data=csv_buf_eq.getvalue().encode("utf-8"),
            file_name=f"equity_{wanted_symbol}_{dataset_choice}.csv",
            mime="text/csv"
        )

    # ---- Plotly: fiyat grafiği + giriş/çıkış + opsiyonel EMA/LRC + TP/SL ----
    st.subheader("Fiyat Grafiği (5m) – Giriş/Çıkış Noktaları")
    if not df_5m.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_5m.index, open=df_5m["open"], high=df_5m["high"],
            low=df_5m["low"], close=df_5m["close"], name="Price (5m)"
        )])

        # Çizimde yoğunluğu azalt: yalnızca son N trade
        trades_to_plot = trades.tail(int(plot_n_trades)) if not trades.empty else trades

        # Giriş/çıkış noktaları (yön renkleri)
        if not trades_to_plot.empty:
            # Long ve Short girişleri ayır
            long_entries = trades_to_plot[trades_to_plot["side"]=="long"][["time","entry"]].dropna()
            short_entries = trades_to_plot[trades_to_plot["side"]=="short"][["time","entry"]].dropna()

            if not long_entries.empty:
                fig.add_trace(go.Scatter(
                    x=long_entries["time"], y=long_entries["entry"],
                    mode="markers", name="Long Entry",
                    marker=dict(symbol="triangle-up", size=9, color="green")
                ))
            if not short_entries.empty:
                fig.add_trace(go.Scatter(
                    x=short_entries["time"], y=short_entries["entry"],
                    mode="markers", name="Short Entry",
                    marker=dict(symbol="triangle-down", size=9, color="red")
                ))

            # Çıkışlar (renk: kazanç/zarar)
            if "exit_time" in trades_to_plot.columns:
                exits = trades_to_plot.dropna(subset=["exit_time","exit","net"])
                if not exits.empty:
                    win_exits = exits[exits["net"] > 0]
                    loss_exits = exits[exits["net"] <= 0]
                    if not win_exits.empty:
                        fig.add_trace(go.Scatter(
                            x=win_exits["exit_time"], y=win_exits["exit"],
                            mode="markers", name="Exit (Win)",
                            marker=dict(symbol="x", size=9, color="green")
                        ))
                    if not loss_exits.empty:
                        fig.add_trace(go.Scatter(
                            x=loss_exits["exit_time"], y=loss_exits["exit"],
                            mode="markers", name="Exit (Loss)",
                            marker=dict(symbol="x", size=9, color="red")
                        ))

            # TP/SL seviyeleri (opsiyonel)
            if draw_tp_sl:
                # çok çizgi olmasın diye yine son N trade üzerinden
                for _, tr in trades_to_plot.iterrows():
                    x0 = tr["time"]
                    x1 = tr.get("exit_time", x0)
                    if pd.isna(x1):
                        # kapanmadıysa görsel için az ileri taşıyalım
                        x1 = x0 + pd.Timedelta(minutes=30)
                    # TP
                    if "tp" in tr and not pd.isna(tr["tp"]):
                        fig.add_trace(go.Scatter(
                            x=[x0, x1], y=[tr["tp"], tr["tp"]],
                            mode="lines", name="TP",
                            line=dict(dash="dot", width=1, color="green"),
                            showlegend=False
                        ))
                    # SL
                    if "sl" in tr and not pd.isna(tr["sl"]):
                        fig.add_trace(go.Scatter(
                            x=[x0, x1], y=[tr["sl"], tr["sl"]],
                            mode="lines", name="SL",
                            line=dict(dash="dot", width=1, color="red"),
                            showlegend=False
                        ))

        # EMA overlay (opsiyonel)
        if show_ema and {"ema_f","ema_m","ema_s"}.issubset(sig.columns):
            fig.add_trace(go.Scatter(x=sig.index, y=sig["ema_f"], mode="lines", name="EMA 7"))
            fig.add_trace(go.Scatter(x=sig.index, y=sig["ema_m"], mode="lines", name="EMA 13"))
            fig.add_trace(go.Scatter(x=sig.index, y=sig["ema_s"], mode="lines", name="EMA 26"))

        # LRC overlay (opsiyonel, 1D -> 5m reindex)
        if show_lrc_overlay and not df_1d.empty:
            bands = cached_lrc_bands(df_1d, length=300)
            if not bands.empty:
                lrc_high = bands["lrc_high"].reindex(df_5m.index, method="ffill")
                lrc_low  = bands["lrc_low"].reindex(df_5m.index, method="ffill")
                fig.add_trace(go.Scatter(x=df_5m.index, y=lrc_high, mode="lines", name="LRC High (1D→5m)"))
                fig.add_trace(go.Scatter(x=df_5m.index, y=lrc_low,  mode="lines", name="LRC Low  (1D→5m)"))

        fig.update_layout(height=560, xaxis_title="Time (UTC)", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Grafik için 5m veri yok.")

    # ---- İşlem geçmişi + CSV indir ----
    st.subheader("İşlemler")
    if not trades.empty:
        st.dataframe(trades.tail(200), use_container_width=True)
        csv_buf = io.StringIO()
        trades.to_csv(csv_buf, index=False)
        st.download_button(
            label="İşlemleri CSV indir",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"trades_{wanted_symbol}_{dataset_choice}.csv",
            mime="text/csv"
        )
    else:
        st.info("İşlem bulunamadı. Parametreleri değiştirip tekrar dene.")
