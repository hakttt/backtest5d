# backtest_from_dropbox.py
# Çalıştır: streamlit run backtest_from_dropbox.py
# Gerekli paketler:
#   streamlit==1.36.0
#   pandas==2.2.2
#   numpy==1.26.4
#   requests==2.32.3
#   fastparquet==2024.5.0
#   plotly==5.22.0

import os
import io
import hashlib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
from typing import Tuple
from datetime import date, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="BTC/ETH Backtest (Dropbox Parquet)", layout="wide")

# ----- Varsayılan linkler -----
DEFAULT_FUTURES_URL = (
    "https://www.dropbox.com/scl/fi/58n3fd9syv91z1y2ro1d0/"
    "binance_futures_5m-1h-1w-1M-1d_2019-12_2025-07_BTC_ETH.parquet"
    "?rlkey=fls6fw8ewieqig77ufdhhh1s9&st=xn143hx8&dl=0"
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
    needed = {"open","high","low","close","volume","symbol","timeframe"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Beklenen kolon(lar) eksik: {miss}")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["timeframe"] = df["timeframe"].astype(str)
    return df.sort_index()

# ---------- EMA / ATR ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

# ---------- LRC ----------
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
    out["lrc_high"] = rolling_lrc(df["high"], length=length)
    out["lrc_low"]  = rolling_lrc(df["low"],  length=length)
    return out

@st.cache_data(show_spinner=False)
def cached_lrc_bands(df_1d: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    if df_1d is None or df_1d.empty:
        return pd.DataFrame(index=getattr(df_1d, "index", None))
    df_1d = df_1d.sort_index()
    df_1d = df_1d[~df_1d.index.duplicated(keep="last")]
    key = (str(df_1d.index.min()) + str(df_1d.index.max()) + str(float(df_1d["close"].sum())))
    _ = hashlib.md5(key.encode()).hexdigest()
    return compute_lrc_bands(df_1d, length=length)

# ---------- 5m + 1h EMA + Opsiyonel LRC ----------
def make_signals_5m_with_1h_and_lrc(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d_for_lrc: pd.DataFrame,   # yalnızca dataset'in 1D verisi
    atr_len: int = 14,
    regime_filter: str | None = None,
    use_lrc: bool = False
) -> pd.DataFrame:
    """
    5m: EMA(7)>EMA(13)>EMA(26) => long ; EMA(7)<EMA(13)<EMA(26) => short
    1h: yön teyidi (EMA7 vs EMA13) — 5m zamanına ffill
    LRC (opsiyonel): close>lrc_high -> sadece long; close<lrc_low -> sadece short; aradaysa sinyal yok
    """
    ema_fast, ema_mid, ema_slow = 7, 13, 26

    out = df_5m.copy().sort_index()
    out = out[~out.index.duplicated(keep="last")]

    h = df_1h.copy().sort_index()
    h = h[~h.index.duplicated(keep="last")]

    out["ema_f"] = ema(out["close"], ema_fast)
    out["ema_m"] = ema(out["close"], ema_mid)
    out["ema_s"] = ema(out["close"], ema_slow)

    h["ema_f_h"] = ema(h["close"], ema_fast)
    h["ema_m_h"] = ema(h["close"], ema_mid)
    h["bull_align_h"] = h["ema_f_h"] > h["ema_m_h"]
    h["bear_align_h"] = h["ema_f_h"] < h["ema_m_h"]

    bull_5m = (out["ema_f"] > out["ema_m"]) & (out["ema_m"] > out["ema_s"])
    bear_5m = (out["ema_f"] < out["ema_m"]) & (out["ema_m"] < out["ema_s"])

    bull_1h_on_5m = h["bull_align_h"].reindex(out.index, method="ffill")
    bear_1h_on_5m = h["bear_align_h"].reindex(out.index, method="ffill")

    long_raw  = bull_5m & bull_1h_on_5m
    short_raw = bear_5m & bear_1h_on_5m

    if use_lrc and df_1d_for_lrc is not None and not df_1d_for_lrc.empty:
        bands = cached_lrc_bands(df_1d_for_lrc, length=300)
        if not bands.empty:
            lrc_high = bands["lrc_high"].reindex(out.index, method="ffill")
            lrc_low  = bands["lrc_low"].reindex(out.index, method="ffill")
            above = out["close"] > lrc_high
            below = out["close"] < lrc_low
            long_raw  = long_raw  & above
            short_raw = short_raw & below

    if regime_filter == "long-only":
        short_raw = pd.Series(False, index=out.index)
    elif regime_filter == "short-only":
        long_raw = pd.Series(False, index=out.index)

    out["ATR"] = atr(out, atr_len)
    out["long_signal"]  = long_raw.fillna(False)
    out["short_signal"] = short_raw.fillna(False)
    return out

# ---------- Swing yardımcıları ----------
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

# ---------- Streak yardımcı ----------
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

# ---------- Backtest (TP daima 2×SL) ----------
def backtest(df: pd.DataFrame,
             stop_type: str = "Swing",     # "Swing" | "ATR"
             tp_percent: float = 10.0,     # (KULLANILMIYOR)
             rr: float = 2.0,              # (KULLANILMIYOR, sabit 2)
             atr_stop_mult: float = 2.0,
             entry_offset_bps: float = 0.0,
             fee_rate: float = 0.0004,
             initial_equity: float = 1000.0,
             risk_mode: str = "dynamic",   # "fixed" -> sabit notional
             fixed_amount: float = 100.0,  # (USDT notional)
             risk_pct: float = 0.02,       # dynamic: equity * risk_pct = SL zararı
             leverage: float = 10.0,
             swing_lookback: int = 10,
             pip_size: float = 0.01) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:

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

    off_mult = 1.0 + (entry_offset_bps / 10000.0)

    def cap_by_leverage(q: float, price: float) -> Tuple[float, float]:
        nominal_ = q * price
        max_nominal = max(equity * leverage, 0.0)
        if nominal_ > max_nominal > 0.0:
            scale = max_nominal / nominal_
            q *= scale
            nominal_ = max_nominal
        return q, nominal_

    RR_TARGET = 2.0
    RR_TOL = 1e-6  # float toleransı

    for i in range(2, len(df)):
        ts = df.index[i]

        if not in_pos:
            signal_long = bool(long_sig.iloc[i-1])
            signal_short = bool(short_sig.iloc[i-1])

            if signal_long or signal_short:
                ent = float(o.iloc[i])
                ent = ent * off_mult if signal_long else ent / off_mult

                # SL hesapla ve mesafeyi bul
                if signal_long:
                    if stop_type == "Swing":
                        s_low = _last_swing_low(h, l, lookback=swing_lookback)
                        sl = max(0.0, float(s_low - pip_size))
                    else:
                        sl = float(ent - atrv.iloc[i] * atr_stop_mult)
                    dist = max(ent - sl, 0.0)
                else:
                    if stop_type == "Swing":
                        s_high = _last_swing_high(h, l, lookback=swing_lookback)
                        sl = float(s_high + pip_size)
                    else:
                        sl = float(ent + atrv.iloc[i] * atr_stop_mult)
                    dist = max(sl - ent, 0.0)

                if dist <= 0 or not np.isfinite(dist):
                    eq_curve.append((ts, equity))
                    continue

                # Pozisyon boyutu — risk % SL zararına göre
                if risk_mode == "fixed":
                    q = max(fixed_amount / ent, 0.0)
                else:
                    risk_amount = max(equity * risk_pct, 0.0)
                    q = max(risk_amount / dist, 0.0)

                # Leverage sınırı
                q, nominal = cap_by_leverage(q, ent)
                if q <= 0:
                    eq_curve.append((ts, equity))
                    continue

                # TP = 2×SL ve rr kontrol
                if signal_long:
                    tp = float(ent + RR_TARGET * dist)
                    side = "long"
                    tp_dist = tp - ent
                    sl_dist = ent - sl
                else:
                    tp = float(ent - RR_TARGET * dist)
                    side = "short"
                    tp_dist = ent - tp
                    sl_dist = sl - ent

                rr_eff = (tp_dist / sl_dist) if sl_dist > 0 else np.nan
                if not np.isfinite(rr_eff) or abs(rr_eff - RR_TARGET) > RR_TOL:
                    # Herhangi bir nedenle rr 2 değilse trade’i açma
                    eq_curve.append((ts, equity))
                    continue

                fee_open = q * ent * fee_rate
                entry_price = float(ent)
                in_pos = True

                trades.append({
                    "time": ts, "side": side, "entry": entry_price,
                    "sl": float(sl), "tp": float(tp),
                    "sl_dist": float(sl_dist), "tp_dist": float(tp_dist), "rr": float(rr_eff),
                    "qty": float(q), "nominal": float(nominal),
                    "fee_open": float(fee_open)
                })

        else:
            hi, lo = float(h.iloc[i]), float(l.iloc[i])
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
                fee_close = qty * float(exit_price) * fee_rate
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

    # ---- İstatistikler (TP/SL bazlı kazanım) ----
    if trades_df.empty:
        stats = {
            "trades": 0, "wins": 0, "losses": 0,
            "winrate": 0.0, "net_total": 0.0,
            "max_dd": 0.0, "final_equity": float(equity),
            "max_consec_wins": 0, "max_consec_losses": 0
        }
    else:
        res_str = trades_df["result"].astype(str).str.upper()
        tp_mask = res_str == "TP"
        sl_mask = res_str.isin(["SL", "SL_FIRST"])
        wins = int(tp_mask.sum())
        losses = int(sl_mask.sum())
        closed = wins + losses
        winrate = (100.0 * wins / closed) if closed > 0 else 0.0

        net_total = float(trades_df["net"].sum(skipna=True))
        eq = eq_df["equity"].fillna(method="ffill").fillna(initial_equity)
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min() * 100.0)
        final_eq = float(eq.iloc[-1])

        # Streak'ler de "TP" bazlı
        win_flags = tp_mask.reindex(trades_df.index, fill_value=False)
        max_w = max_l = cur_w = cur_l = 0
        for v in win_flags.astype(bool).tolist():
            if v:
                cur_w += 1; max_w = max(max_w, cur_w); cur_l = 0
            else:
                cur_l += 1; max_l = max(max_l, cur_l); cur_w = 0

        stats = {
            "trades": int(len(trades_df)),
            "wins": wins,
            "losses": losses,
            "winrate": float(winrate),
            "net_total": net_total,
            "max_dd": max_dd,
            "final_equity": final_eq,
            "max_consec_wins": int(max_w),
            "max_consec_losses": int(max_l)
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
    default_range = (date(2020, 1, 1), date.today())
    date_range = st.date_input("İşlem aralığı (UTC)", value=default_range, help="Başlangıç ve bitiş tarihi seçiniz")
    st.caption("Btc/Eth (2019-12.ay/2025 7. ay arası) 5m 1h 1d 1w 1m futures.")

    st.subheader("Opsiyonel LRC Filtre")
    use_lrc = st.checkbox("LRC filtresi (1D LRC-300)", value=False)

    st.subheader("Grafik Ayarı")
    show_graphs = st.checkbox("Grafikleri göster", value=False)
    show_ema = st.checkbox("Grafikte EMA(7/13/26) çizgilerini göster", value=False)
    show_lrc_overlay = st.checkbox("Grafikte 1D LRC(300) bandını göster", value=False)
    draw_tp_sl = st.checkbox("İşlemlerde TP/SL seviyelerini çiz", value=False)
    plot_n_trades = st.number_input("Son N trade çiz (görsel)", min_value=10, max_value=2000, value=300, step=10)

    st.subheader("Stop / Hedef Ayarları")
    stop_type = st.selectbox("Stop Tipi", ["Swing","ATR"], index=0)
    atr_mult = st.number_input("SL = ATR × [ATR için]", 0.5, 10.0, 2.0, 0.1)
    entry_off = st.number_input("Giriş Offset (bps)", 0.0, 100.0, 0.0, 1.0)
    st.caption("Not: TP her zaman SL mesafesinin **2×**'sidir.")

    st.subheader("Sermaye / İşlem")
    init_eq = st.number_input("Başlangıç Sermaye", 100.0, 1_000_000.0, 1000.0, 100.0)
    lev = st.number_input("Kaldıraç (x)", 1.0, 100.0, 10.0, 1.0)
    fee = st.number_input("Komisyon (her bacak)", 0.0, 0.005, 0.0004, 0.0001, format="%.4f")
    risk_mode = st.selectbox("Risk Modu", ["fixed","dynamic"], index=1)
    fixed_amt = st.number_input("Sabit Notional (USDT)", 10.0, 100000.0, 100.0, 10.0)
    risk_pct = st.number_input("Risk % (dynamic, SL zarar hedefi)", 0.001, 0.2, 0.02, 0.001, format="%.3f")

    run_btn = st.button("Yükle & Backtest", type="primary")

# ---------- Önceki (kilitli) sonuçları göster ----------
if "last_stats_text" in st.session_state:
    with st.expander("Son sonuçlar (kopyalanabilir)", expanded=True):
        st.text_area("Özet", st.session_state["last_stats_text"], height=180)
        if "last_trades_csv" in st.session_state:
            st.download_button(
                "İşlemleri CSV indir (Son)",
                st.session_state["last_trades_csv"].encode("utf-8"),
                file_name=st.session_state.get("last_trades_name", "trades_last.csv"),
                mime="text/csv"
            )
        if st.button("Sonuçları temizle"):
            for k in ["last_stats_text", "last_trades_csv", "last_trades_name"]:
                st.session_state.pop(k, None)
            st.rerun()

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

    # --- Tarih aralığını sırala + UTC ---
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        a, b = date_range[0], date_range[1]
        if pd.Timestamp(a) > pd.Timestamp(b):
            a, b = b, a
        start_date = pd.Timestamp(a).tz_localize("UTC")
        end_date = pd.Timestamp(b).tz_localize("UTC") + pd.Timedelta(days=1)
    else:
        start_date = pd.Timestamp(date_range).tz_localize("UTC")
        end_date = start_date + pd.Timedelta(days=1)

    # --- LRC için geçmiş pad'li 1D veriyi hazırla (YALNIZ dataset 1D) ---
    daily_full = df_1d_all.copy()
    if use_lrc and daily_full.empty:
        st.warning("LRC aktif, ancak 1D veri bulunamadı; LRC uygulanmayacak (resample yok).")
    start_ext = (start_date - pd.Timedelta(days=400)).floor("D")
    daily_ext = daily_full.loc[(daily_full.index >= start_ext) & (daily_full.index < end_date)].copy()
    daily_ext = daily_ext.sort_index()
    daily_ext = daily_ext[~daily_ext.index.duplicated(keep="last")]

    # --- 5m/1h: seçilen aralığa kırp ---
    df_5m = df_5m_all.loc[(df_5m_all.index >= start_date) & (df_5m_all.index < end_date)].copy().sort_index()
    df_1h = df_1h_all.loc[(df_1h_all.index >= start_date) & (df_1h_all.index < end_date)].copy().sort_index()
    df_5m = df_5m[~df_5m.index.duplicated(keep="last")]
    df_1h = df_1h[~df_1h.index.duplicated(keep="last")]

    # --- Sinyaller ---
    regime_arg = None if regime == "Hepsi" else regime
    sig = make_signals_5m_with_1h_and_lrc(
        df_5m, df_1h, daily_ext,
        atr_len=14,
        regime_filter=regime_arg,
        use_lrc=use_lrc
    )

    # --- Backtest (TP = 2×SL) ---
    trades, stats, eq = backtest(
        sig,
        stop_type=stop_type,
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
        if show_graphs:
            st.line_chart(eq)
        else:
            st.caption("Grafikler kapalı (sidebar → Grafikleri göster).")
    with c2:
        st.subheader("Özet")
        summary = {
            "Toplam İşlem": stats.get("trades", 0),
            "Kazanan": stats.get("wins", 0),              # TP sayısı
            "Kaybeden": stats.get("losses", 0),           # SL + SL_first sayısı
            "Winrate %": round(stats.get("winrate", 0.0), 2),
            "Max Consec Wins": stats.get("max_consec_wins", 0),
            "Max Consec Losses": stats.get("max_consec_losses", 0),
            "Net Toplam (USDT)": round(stats.get("net_total", 0.0), 2),
            "Final Equity": round(stats.get("final_equity", 0.0), 2),
            "Max Drawdown %": round(stats.get("max_dd", 0.0), 2),
        }
        st.write(summary)

    # ---- Sonuçları kopyalanabilir şekilde kilitle ----
    stats_text = (
        f"Toplam İşlem: {summary['Toplam İşlem']}\n"
        f"Kazanan (TP): {summary['Kazanan']}  |  Kaybeden (SL): {summary['Kaybeden']}\n"
        f"Winrate: {summary['Winrate %']}%\n"
        f"Max Consec Wins: {summary['Max Consec Wins']}  |  Max Consec Losses: {summary['Max Consec Losses']}\n"
        f"Net PnL (USDT): {summary['Net Toplam (USDT)']}\n"
        f"Final Equity: {summary['Final Equity']}\n"
        f"Max DD (%): {summary['Max Drawdown %']}\n"
    )
    st.session_state["last_stats_text"] = stats_text

    csv_buf = io.StringIO()
    trades.to_csv(csv_buf, index=False)
    st.session_state["last_trades_csv"] = csv_buf.getvalue()
    st.session_state["last_trades_name"] = f"trades_{wanted_symbol}_{dataset_choice}.csv"

    # ---- Plotly grafikler (isteğe bağlı) ----
    if show_graphs:
        st.subheader("Equity Eğrisi (Plotly)")
        if not eq.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name="Equity"))
            fig_eq.update_layout(height=350, xaxis_title="Time (UTC)", yaxis_title="Equity (USDT)")
            st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Fiyat Grafiği (5m) – Giriş/Çıkış Noktaları")
        if not df_5m.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df_5m.index, open=df_5m["open"], high=df_5m["high"],
                low=df_5m["low"], close=df_5m["close"], name="Price (5m)"
            )])

            trades_to_plot = trades.tail(int(plot_n_trades)) if not trades.empty else trades
            if not trades_to_plot.empty:
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
                if draw_tp_sl and not trades_to_plot.empty:
                    for _, tr in trades_to_plot.iterrows():
                        x0 = tr["time"]; x1 = tr.get("exit_time", x0 + pd.Timedelta(minutes=30))
                        if "tp" in tr and not pd.isna(tr["tp"]):
                            fig.add_trace(go.Scatter(x=[x0, x1], y=[tr["tp"], tr["tp"]],
                                                     mode="lines", line=dict(dash="dot", width=1, color="green"),
                                                     showlegend=False))
                        if "sl" in tr and not pd.isna(tr["sl"]):
                            fig.add_trace(go.Scatter(x=[x0, x1], y=[tr["sl"], tr["sl"]],
                                                     mode="lines", line=dict(dash="dot", width=1, color="red"),
                                                     showlegend=False))
            if show_ema and {"ema_f","ema_m","ema_s"}.issubset(sig.columns):
                fig.add_trace(go.Scatter(x=sig.index, y=sig["ema_f"], mode="lines", name="EMA 7"))
                fig.add_trace(go.Scatter(x=sig.index, y=sig["ema_m"], mode="lines", name="EMA 13"))
                fig.add_trace(go.Scatter(x=sig.index, y=sig["ema_s"], mode="lines", name="EMA 26"))
            if show_lrc_overlay and not daily_ext.empty:
                bands = cached_lrc_bands(daily_ext, length=300)
                if not bands.empty:
                    lrc_high = bands["lrc_high"].reindex(df_5m.index, method="ffill")
                    lrc_low  = bands["lrc_low"].reindex(df_5m.index, method="ffill")
                    fig.add_trace(go.Scatter(x=df_5m.index, y=lrc_high, mode="lines", name="LRC High (1D→5m)"))
                    fig.add_trace(go.Scatter(x=df_5m.index, y=lrc_low,  mode="lines", name="LRC Low  (1D→5m)"))
            fig.update_layout(height=560, xaxis_title="Time (UTC)", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

    # ---- İşlemler + CSV (her durumda) ----
    st.subheader("İşlemler")
    if not trades.empty:
        st.dataframe(trades.tail(200), use_container_width=True)
        st.download_button(
            "İşlemleri CSV indir",
            st.session_state["last_trades_csv"].encode("utf-8"),
            file_name=st.session_state["last_trades_name"],
            mime="text/csv"
        )
    else:
        st.info("İşlem bulunamadı. Parametreleri değiştirip tekrar dene.")
