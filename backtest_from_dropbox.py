# backtest_from_dropbox.py
import os
import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------
# Streamlit page config (ilk komut)
# -----------------------------------------------------------
st.set_page_config(page_title="EMA + LRC Backtest (5m giriş + 1h onay)", layout="wide")

# -----------------------------------------------------------
# Varsayılan Dropbox linkleri (seninkiler)
# -----------------------------------------------------------
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

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# Yardımcı: güvenli parquet indirme (dl=1 fallback + HTML algı)
# -----------------------------------------------------------
def _force_dl1(url: str) -> str:
    if "dl=" in url:
        return url.replace("dl=0", "dl=1").replace("dl=2", "dl=1")
    sep = "&" if "?" in url else "?"
    return url + f"{sep}dl=1"

@st.cache_data(show_spinner=True)
def load_parquet_from_dropbox(url: str) -> pd.DataFrame:
    """
    - Verilen URL ile dener; HTML/preview gelirse dl=1 fallback yapar.
    - Önce fastparquet ile okur; olmazsa pyarrow denenir (yüklüyse).
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    tried = []
    for step in [url, _force_dl1(url)]:
        try:
            tried.append(step)
            r = requests.get(step, timeout=120, allow_redirects=True, headers=headers)
            r.raise_for_status()
            raw = r.content
            ctype = (r.headers.get("Content-Type") or "").lower()
            size_mb = len(raw) / 1_000_000
            st.caption(f"İndirildi → Content-Type: {ctype} | Boyut: {size_mb:.2f} MB"
                       + (" (dl=1)" if step != url else ""))

            # HTML/preview kontrolü
            if b"<!DOCTYPE html" in raw[:400] or "text/html" in ctype:
                continue  # diğer adım dl=1 ile dene

            # Parquet oku (fastparquet)
            bio = io.BytesIO(raw)
            try:
                df = pd.read_parquet(bio, engine="fastparquet")
                return _standardize_columns(df)
            except Exception:
                bio.seek(0)
                try:
                    # pyarrow varsa çalışır; yoksa bu da hata verir
                    df = pd.read_parquet(bio, engine="pyarrow")
                    return _standardize_columns(df)
                except Exception as e2:
                    raise e2
        except Exception:
            continue

    st.error("Parquet okunamadı. Denenen URL'ler:\n" + "\n".join(tried))
    return pd.DataFrame()

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # timestamp / time
    if "time" in cols:
        tcol = cols["time"]
    elif "timestamp" in cols:
        tcol = cols["timestamp"]
    else:
        raise ValueError("Zaman kolonu bulunamadı (time/timestamp)")
    df = df.rename(columns={tcol: "time"})
    # normalize
    needed = ["open", "high", "low", "close", "volume"]
    for n in needed:
        if n not in df.columns:
            # bazen büyük harf vs olabilir
            maybe = [c for c in df.columns if c.lower() == n]
            if maybe:
                df = df.rename(columns={maybe[0]: n})
            else:
                raise ValueError(f"{n} kolonu yok")
    # interval varsa alt frekansları direkt seçeriz
    if "interval" in [c.lower() for c in df.columns]:
        # Do not rename original; add normalized 'interval_norm'
        iv = None
        for c in df.columns:
            if c.lower() == "interval":
                iv = c
                break
        df["interval_norm"] = df[iv].astype(str).str.lower()
    else:
        df["interval_norm"] = ""
    # symbol normalize
    if "symbol" not in df.columns:
        maybe = [c for c in df.columns if c.lower() == "symbol"]
        if maybe:
            df = df.rename(columns={maybe[0]: "symbol"})
        else:
            # tek sembollü set olabilir
            df["symbol"] = "UNKNOWN"
    # dt
    df["time"] = pd.to_datetime(df["time"], utc=False)
    return df.sort_values("time").reset_index(drop=True)

# -----------------------------------------------------------
# Göstergeler
# -----------------------------------------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # TR = max(H-L, |H-Cprev|, |L-Cprev|)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def tsi(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    m = close.diff()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    abs1 = m.abs().ewm(span=r, adjust=False).mean()
    abs2 = abs1.ewm(span=s, adjust=False).mean()
    return 100 * (ema2 / abs2)

def compute_lrc_bands_daily(df_daily: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    # df_daily: index = day (datetime), cols: open/high/low/close/volume
    def _lrc_last_point(values: np.ndarray) -> float:
        w = np.asarray(values, dtype=float).ravel()
        n = w.size
        if n < 2 or not np.isfinite(w).all():
            return np.nan
        x = np.arange(n, dtype=float)
        m, b = np.polyfit(x, w, 1)
        return m * (n - 1) + b

    def rolling_lrc(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(window=length, min_periods=length).apply(_lrc_last_point, raw=True)

    out = pd.DataFrame(index=df_daily.index)
    out["lrc_high"] = rolling_lrc(df_daily["high"], length)
    out["lrc_low"]  = rolling_lrc(df_daily["low"],  length)
    return out

# -----------------------------------------------------------
# Candlestick yardımcıları
# -----------------------------------------------------------
def is_bull_engulf(prev: pd.Series, cur: pd.Series) -> bool:
    # Bullish engulfing (basit): prev red, cur green, cur body prev body'sini sarar
    if any(pd.isna([prev["open"], prev["close"], cur["open"], cur["close"]])):
        return False
    return (prev["close"] < prev["open"]) and (cur["close"] > cur["open"]) \
        and (cur["close"] >= prev["open"]) and (cur["open"] <= prev["close"])

def is_bear_engulf(prev: pd.Series, cur: pd.Series) -> bool:
    if any(pd.isna([prev["open"], prev["close"], cur["open"], cur["close"]])):
        return False
    return (prev["close"] > prev["open"]) and (cur["close"] < cur["open"]) \
        and (cur["close"] <= prev["open"]) and (cur["open"] >= prev["close"])

def last_swing_low(df5: pd.DataFrame, i: int, lookback: int) -> Optional[float]:
    i0 = max(0, i - lookback)
    window = df5.iloc[i0:i+1]
    return float(window["low"].min()) if len(window) else None

def last_swing_high(df5: pd.DataFrame, i: int, lookback: int) -> Optional[float]:
    i0 = max(0, i - lookback)
    window = df5.iloc[i0:i+1]
    return float(window["high"].max()) if len(window) else None

# -----------------------------------------------------------
# Pozisyon/işlem datası
# -----------------------------------------------------------
@dataclass
class Trade:
    side: str                     # "long" | "short"
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop: float
    tp: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "TP" | "SL" | "RuleExit"
    fee_entry: float = 0.0
    fee_exit: float = 0.0

    @property
    def pnl_usd(self) -> float:
        if self.exit_price is None:
            return 0.0
        direction = 1 if self.side == "long" else -1
        gross = direction * (self.exit_price - self.entry_price) * self.qty
        return gross - self.fee_entry - self.fee_exit

# -----------------------------------------------------------
# Backtest çekirdeği
# -----------------------------------------------------------
def simulate(
    df5: pd.DataFrame,
    df1h: pd.DataFrame,
    dfd: pd.DataFrame,
    *,
    allow_long: bool,
    allow_short: bool,
    use_lrc: bool,
    use_tsi_dw: bool,
    tsi_r: int,
    tsi_s: int,
    vol_filter: bool,
    vol_factor: float,
    need_engulf: bool,
    need_pullback_ema13: bool,
    atr_period: int,
    atr_mult: float,
    swing_lookback: int,
    stop_offset_bps: float,
    rr: float,
    risk_mode: str,            # "dynamic_pct" | "fixed_usd"
    risk_value: float,         # % veya USD
    leverage: float,
    fee_rate: float,           # her bacak %
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    df5: 5m OHLCV (index=Datetime, cols: o,h,l,c,v)
    df1h: 1h OHLCV (index=Datetime)
    dfd: daily OHLCV (index=Date)
    """

    # EMA'lar
    for L in [7, 13, 26]:
        df5[f"ema{L}"] = ema(df5["close"], L)
    df1h["ema7"] = ema(df1h["close"], 7)
    df1h["ema13"] = ema(df1h["close"], 13)

    # ATR (5m)
    df5["atr"] = atr(df5, atr_period)

    # Hacim ort (5m)
    df5["vol_ma20"] = df5["volume"].rolling(20, min_periods=20).mean()

    # TSI (D & W)
    if use_tsi_dw:
        tsi_d = tsi(dfd["close"], tsi_r, tsi_s)
        dfd["tsi_d"] = tsi_d
        # weekly
        dfw = dfd.resample("W").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        dfw["tsi_w"] = tsi(dfw["close"], tsi_r, tsi_s)
    else:
        dfd["tsi_d"] = np.nan
        dfw = dfd.resample("W").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        dfw["tsi_w"] = np.nan

    # LRC (günlük)
    if use_lrc:
        bands = compute_lrc_bands_daily(dfd, 300)
        dfd = dfd.join(bands)

    equity = []
    trades: List[Trade] = []
    cash = 10_000_000.0  # teknik olarak "nakit" değil, burada sadece PnL ekliyoruz
    equity_base = 0.0    # simpliﬁed; gerçek equity için kasa dışarıdan verilecek
    realized_pnl = 0.0

    position: Optional[Trade] = None

    # Hazır referanslar (lookup)
    # 1h ve D eşleşmesi için en yakın geçmiş barı alacağız
    def get_1h_row(ts):
        return df1h.loc[:ts].iloc[-1] if (df1h.index <= ts).any() else None

    def get_d_row(ts):
        day = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)
        # daily index gün saat 00:00; en son günü bul
        return dfd.loc[:day].iloc[-1] if (dfd.index <= day).any() else None

    # İşlem yaratımında risk'e göre adet hesaplama
    def compute_position_qty(side: str, entry: float, stop: float,
                             equity_now: float) -> float:
        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or not np.isfinite(stop_dist):
            return 0.0
        if risk_mode == "dynamic_pct":
            risk_usd = equity_now * (risk_value / 100.0)
        else:
            risk_usd = risk_value
        qty = risk_usd / stop_dist
        # kaldıraç/marjin sınırı: notional/leverage <= equity_now
        notional = qty * entry
        max_notional = equity_now * leverage
        if notional > max_notional:
            qty = max_notional / entry
        return max(qty, 0.0)

    # Basit equity (başlangıç sermayesini UI'dan alacağız; burada sadece PnL toplanacak)
    eq = 0.0

    # Ana döngü
    for i in range(len(df5)):
        ts = df5.index[i]
        if ts < start_dt or ts > end_dt:
            # Warm-up veya aralık dışı
            continue

        row5 = df5.iloc[i]
        # 1h teyit
        row1h = get_1h_row(ts)
        if row1h is None or pd.isna(row1h["ema7"]) or pd.isna(row1h["ema13"]):
            continue

        # EMA şartları
        ema_ok_long  = (row5["ema7"] > row5["ema13"] > row5["ema26"]) and (row1h["ema7"] > row1h["ema13"])
        ema_ok_short = (row5["ema7"] < row5["ema13"] < row5["ema26"]) and (row1h["ema7"] < row1h["ema13"])

        # LRC şartı
        lrc_pass_long = True
        lrc_pass_short = True
        if use_lrc:
            drow = get_d_row(ts)
            if drow is None or (("lrc_high" not in drow) or (pd.isna(drow["lrc_high"]) or pd.isna(drow["lrc_low"]))):
                lrc_pass_long = lrc_pass_short = False
            else:
                lrc_pass_long  = row5["close"] > drow["lrc_high"]
                lrc_pass_short = row5["close"] < drow["lrc_low"]

        # TSI D+W şartı
        tsi_pass_long = True
        tsi_pass_short = True
        if use_tsi_dw:
            drow = get_d_row(ts)
            # weekly karşılığı (en yakın geçmiş Pazar kapanışı)
            wrow = dfw.loc[:ts].iloc[-1] if (dfw.index <= ts).any() else None
            if drow is None or wrow is None:
                tsi_pass_long = tsi_pass_short = False
            else:
                tsi_pass_long  = (drow["tsi_d"] > 0) and (wrow["tsi_w"] > 0)
                tsi_pass_short = (drow["tsi_d"] < 0) and (wrow["tsi_w"] < 0)

        # Hacim filtresi
        vol_pass = True
        if vol_filter:
            if pd.isna(row5["vol_ma20"]):
                vol_pass = False
            else:
                vol_pass = row5["volume"] >= vol_factor * row5["vol_ma20"]

        # Engulf/pullback
        engulf_pass_long = engulf_pass_short = True
        if need_engulf and i >= 1:
            prev = df5.iloc[i-1][["open","high","low","close"]]
            cur  = row5[["open","high","low","close"]]
            engulf_pass_long  = is_bull_engulf(prev, cur)
            engulf_pass_short = is_bear_engulf(prev, cur)

        pullback_pass_long = pullback_pass_short = True
        if need_pullback_ema13:
            pullback_pass_long  = row5["low"]  <= row5["ema13"]
            pullback_pass_short = row5["high"] >= row5["ema13"]

        # Çıkış takip (pozisyon açıksa önce stop/tp kontrolü)
        if position is not None:
            # Bu bar içinde stop veya tp görüldü mü (bar-internal basit kontrol)
            if position.side == "long":
                hit_sl = row5["low"]  <= position.stop
                hit_tp = row5["high"] >= position.tp
                exit_reason = None
                exit_price = None
                if hit_sl and hit_tp:
                    # konservatif: önce SL
                    exit_reason = "SL"
                    exit_price = position.stop
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = position.stop
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = position.tp

                # Kurala göre çıkış (EMA/LRC/TSI bozulduysa) – sadece SL/TP olmadıysa
                if exit_reason is None:
                    rule_ok_long = ema_ok_long and (not use_lrc or lrc_pass_long) and (not use_tsi_dw or tsi_pass_long)
                    if not rule_ok_long:
                        exit_reason = "RuleExit"
                        exit_price = row5["close"]

                if exit_reason:
                    fee_exit = (fee_rate / 100.0) * (exit_price * position.qty)
                    position.exit_time = ts
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    position.fee_exit = fee_exit
                    trades.append(position)
                    eq += position.pnl_usd
                    position = None

            else:  # short
                hit_sl = row5["high"] >= position.stop
                hit_tp = row5["low"]  <= position.tp
                exit_reason = None
                exit_price = None
                if hit_sl and hit_tp:
                    exit_reason = "SL"
                    exit_price = position.stop
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = position.stop
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = position.tp

                if exit_reason is None:
                    rule_ok_short = ema_ok_short and (not use_lrc or lrc_pass_short) and (not use_tsi_dw or tsi_pass_short)
                    if not rule_ok_short:
                        exit_reason = "RuleExit"
                        exit_price = row5["close"]

                if exit_reason:
                    fee_exit = (fee_rate / 100.0) * (exit_price * position.qty)
                    position.exit_time = ts
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason
                    position.fee_exit = fee_exit
                    trades.append(position)
                    eq += position.pnl_usd
                    position = None

        # Giriş – eğer pozisyon yoksa
        if position is None:
            # Long adayı
            if allow_long and ema_ok_long and lrc_pass_long and tsi_pass_long and vol_pass and engulf_pass_long and pullback_pass_long:
                # stop = max(swing_low - offset, entry - ATR*mult)
                entry = row5["close"]
                sl_swing = last_swing_low(df5, i-1, swing_lookback)
                if sl_swing is None or not np.isfinite(sl_swing):
                    pass  # giriş atla
                else:
                    offset = entry * (stop_offset_bps / 10_000.0)
                    sl1 = sl_swing - offset
                    atr_val = row5["atr"]
                    sl2 = entry - atr_mult * atr_val if np.isfinite(atr_val) else entry * 0.99
                    stop = min(sl1, sl2)  # long için daha aşağıdaki (daha geniş) stop
                    # TP
                    sl_dist = entry - stop
                    if sl_dist <= 0:
                        pass
                    else:
                        tp = entry + rr * sl_dist
                        # boyut
                        qty = compute_position_qty("long", entry, stop, equity_now=max(eq, 1.0))
                        if qty > 0:
                            fee_entry = (fee_rate / 100.0) * (entry * qty)
                            position = Trade(side="long", entry_time=ts, entry_price=entry, qty=qty,
                                             stop=stop, tp=tp, fee_entry=fee_entry)

            # Short adayı
            if (position is None) and allow_short and ema_ok_short and lrc_pass_short and tsi_pass_short and vol_pass and engulf_pass_short and pullback_pass_short:
                entry = row5["close"]
                sh_swing = last_swing_high(df5, i-1, swing_lookback)
                if sh_swing is None or not np.isfinite(sh_swing):
                    pass
                else:
                    offset = entry * (stop_offset_bps / 10_000.0)
                    sl1 = sh_swing + offset
                    atr_val = row5["atr"]
                    sl2 = entry + atr_mult * atr_val if np.isfinite(atr_val) else entry * 1.01
                    stop = max(sl1, sl2)  # short için daha yukarıdaki (daha geniş) stop
                    sl_dist = stop - entry
                    if sl_dist <= 0:
                        pass
                    else:
                        tp = entry - rr * sl_dist
                        qty = compute_position_qty("short", entry, stop, equity_now=max(eq, 1.0))
                        if qty > 0:
                            fee_entry = (fee_rate / 100.0) * (entry * qty)
                            position = Trade(side="short", entry_time=ts, entry_price=entry, qty=qty,
                                             stop=stop, tp=tp, fee_entry=fee_entry)

        equity.append({"time": ts, "equity": eq})

    # Kapanışta açık pozisyon varsa kapat
    if position is not None:
        last_close = df5.iloc[-1]["close"]
        fee_exit = (fee_rate / 100.0) * (last_close * position.qty)
        position.exit_time = df5.index[-1]
        position.exit_price = last_close
        position.exit_reason = "LastBarExit"
        position.fee_exit = fee_exit
        trades.append(position)
        eq += position.pnl_usd
        position = None

    equity_df = pd.DataFrame(equity).set_index("time")

    # İstatistikler
    tr_df = pd.DataFrame([{
        "side": t.side,
        "entry_time": t.entry_time,
        "entry": t.entry_price,
        "exit_time": t.exit_time,
        "exit": t.exit_price,
        "exit_reason": t.exit_reason,
        "qty": t.qty,
        "stop": t.stop,
        "tp": t.tp,
        "pnl_usd": t.pnl_usd
    } for t in trades if t.exit_time is not None])

    stats = compute_stats(tr_df)
    return tr_df, equity_df, stats

def compute_stats(trades_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "avg_pnl": 0.0,
            "max_consec_win": 0, "max_consec_loss": 0, "max_dd": 0.0
        }
    wins = (trades_df["pnl_usd"] > 0).astype(int)
    losses = (trades_df["pnl_usd"] < 0).astype(int)
    trades = len(trades_df)
    win_rate = 100.0 * wins.sum() / trades
    net_pnl = trades_df["pnl_usd"].sum()
    avg_pnl = trades_df["pnl_usd"].mean()

    # consecutive streaks
    max_w, max_l = 0, 0
    cur_w, cur_l = 0, 0
    for pnl in trades_df["pnl_usd"]:
        if pnl > 0:
            cur_w += 1; max_w = max(max_w, cur_w)
            cur_l = 0
        elif pnl < 0:
            cur_l += 1; max_l = max(max_l, cur_l)
            cur_w = 0
        else:
            cur_w = cur_l = 0

    # max drawdown (equity akümüle edilerek)
    eq = trades_df["pnl_usd"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()

    return {
        "trades": trades,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "avg_pnl": avg_pnl,
        "max_consec_win": int(max_w),
        "max_consec_loss": int(max_l),
        "max_dd": float(max_dd),
    }

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("📊 EMA + 1H Onay + (opsiyonel) LRC / TSI / Hacim / Engulf Backtest")

with st.expander("🔗 Veri Kaynağı", expanded=True):
    col0, col1 = st.columns(2)
    with col0:
        mode = st.radio("Kaynak", ["Futures", "Spot"], horizontal=True)
        url_input = st.text_input("Dropbox Parquet Linki", DEFAULT_FUTURES_URL if mode=="Futures" else DEFAULT_SPOT_URL)
    with col1:
        if st.button("🔁 Önbelleği temizle"):
            st.cache_data.clear()
            st.success("Önbellek temizlendi.")

df = None
if url_input:
    df = load_parquet_from_dropbox(url_input)

if df is None or df.empty:
    st.stop()

# Sembol listesi
symbols = sorted(list(df["symbol"].dropna().unique()))
if not symbols:
    symbols = ["BTCUSDT", "ETHUSDT"]

colA, colB, colC = st.columns(3)
with colA:
    symbol = st.selectbox("Sembol", symbols, index=(symbols.index("ETHUSDT") if "ETHUSDT" in symbols else 0))
with colB:
    start_date = st.date_input("Başlangıç", pd.Timestamp("2023-01-01"))
with colC:
    end_date = st.date_input("Bitiş", pd.Timestamp("2024-01-01"))

# Warm-up uyarısı
st.caption("⚠️ LRC(300) ve 5m/1h EMA için yeterli warm-up gereklidir. Kod, başlangıçtan "
           "en az 300 günlük (günlük LRC) ve yeterli EMA/ATR geçmişini **otomatik** ekler.")

# Sembolün verisini ayrıştır
df_sym = df[df["symbol"] == symbol].copy()
df_sym["time"] = pd.to_datetime(df_sym["time"])
df_sym = df_sym.sort_values("time")

# 5m ve 1h veri
if (df_sym["interval_norm"] != "").any():
    df5 = df_sym[df_sym["interval_norm"].isin(["5m","5min","5"])]
    df1h = df_sym[df_sym["interval_norm"].isin(["1h","60","60m"])]
    # Güvenli olsun diye yine resample'la normalize et
    df5 = df5.set_index("time").sort_index().resample("5T").last()[["open","high","low","close","volume"]].dropna()
    df1h = df1h.set_index("time").sort_index().resample("1H").last()[["open","high","low","close","volume"]].dropna()
else:
    # Toplu tek frekans ise resample
    base = df_sym.set_index("time").sort_index()[["open","high","low","close","volume"]]
    df5 = base.resample("5T").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    df1h = base.resample("1H").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

# Günlük (1h'den resample – talebin)
dfd_all = df1h.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

# Warm-up: LRC için 300 gün geri
warm_days = 320
start_dt = pd.Timestamp(start_date)
end_dt   = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)

warm_start = start_dt - pd.Timedelta(days=warm_days)
df5_w = df5.loc[warm_start:end_dt].copy()
df1h_w = df1h.loc[warm_start:end_dt].copy()
dfd_w = dfd_all.loc[:end_dt].copy()  # günlüklere warm-start gerekmesin diye üstten alıyoruz

with st.expander("⚙️ Kurallar & Risk Ayarları", expanded=True):
    # Yön ve filtreler
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        allow_long = st.checkbox("Long Aç", value=True)
        allow_short = st.checkbox("Short Aç", value=True)
    with c2:
        use_lrc = st.checkbox("LRC (D, 300) filtresi", value=False)
        use_tsi_dw = st.checkbox("TSI D+W filtresi", value=False)
    with c3:
        vol_filter = st.checkbox("Hacim filtresi (≥ k × MA20)", value=False)
        vol_factor = st.number_input("k (hacim kat sayısı)", value=1.0, step=0.1)
    with c4:
        need_engulf = st.checkbox("Engulf şartı", value=False)
        need_pullback = st.checkbox("EMA13'e çekilme şartı", value=False)

    c5, c6, c7 = st.columns(3)
    with c5:
        tsi_r = st.number_input("TSI r", value=25, step=1, min_value=1)
        tsi_s = st.number_input("TSI s", value=13, step=1, min_value=1)
    with c6:
        atr_period = st.number_input("ATR Periyodu (5m)", value=14, step=1, min_value=1)
        atr_mult = st.number_input("ATR Çarpanı (Stop)", value=2.0, step=0.1, min_value=0.1)
        swing_lookback = st.number_input("Swing lookback (bar)", value=20, step=1, min_value=2)
    with c7:
        stop_offset_bps = st.number_input("Stop offset (bps)", value=5.0, step=1.0, help="1 bps = 0.01%")
        rr = st.number_input("R:R (TP/SL)", value=2.0, step=0.1, min_value=0.1)

with st.expander("💰 Sermaye & Komisyon", expanded=True):
    c8, c9, c10, c11 = st.columns(4)
    with c8:
        risk_mode = st.selectbox("Risk modu", ["dynamic_pct", "fixed_usd"], format_func=lambda x: "Dinamik % (equity)" if x=="dynamic_pct" else "Sabit USD")
    with c9:
        if risk_mode == "dynamic_pct":
            risk_value = st.number_input("Risk %", value=2.0, step=0.1, min_value=0.1)
        else:
            risk_value = st.number_input("Sabit risk (USD)", value=100.0, step=10.0, min_value=10.0)
    with c10:
        leverage = st.number_input("Kaldıraç (x)", value=10.0, step=1.0, min_value=1.0)
    with c11:
        fee_rate = st.number_input("Komisyon (her bacak, %)", value=0.04, step=0.01, min_value=0.0)

run = st.button("▶ Backtest Çalıştır")

if run:
    if df5_w.empty or df1h_w.empty:
        st.error("Veri aralığı boş.")
        st.stop()

    # Simülasyon
    trades_df, equity_df, stats = simulate(
        df5=df5_w, df1h=df1h_w, dfd=dfd_w,
        allow_long=allow_long, allow_short=allow_short,
        use_lrc=use_lrc, use_tsi_dw=use_tsi_dw, tsi_r=tsi_r, tsi_s=tsi_s,
        vol_filter=vol_filter, vol_factor=vol_factor,
        need_engulf=need_engulf, need_pullback_ema13=need_pullback,
        atr_period=atr_period, atr_mult=atr_mult,
        swing_lookback=swing_lookback, stop_offset_bps=stop_offset_bps,
        rr=rr, risk_mode=risk_mode, risk_value=risk_value,
        leverage=leverage, fee_rate=fee_rate,
        start_dt=pd.Timestamp(start_date), end_dt=pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    )

    # Özet
    st.subheader("📈 Sonuç Özeti")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Toplam İşlem", int(stats.get("trades", 0)))
    m2.metric("Win Rate", f"{stats.get('win_rate', 0.0):.2f}%")
    m3.metric("Net PnL (USD)", f"{stats.get('net_pnl', 0.0):.2f}")
    m4.metric("Ort. PnL (USD)", f"{stats.get('avg_pnl', 0.0):.2f}")
    m5.metric("Max Consec Win", int(stats.get("max_consec_win", 0)))
    m6.metric("Max Consec Loss", int(stats.get("max_consec_loss", 0)))

    # Equity grafiği
    st.subheader("Equity Eğrisi")
    if equity_df.empty:
        st.info("Equity verisi boş.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # İşlem günlüğü
    st.subheader("İşlem Günlüğü")
    if trades_df.empty:
        st.info("İşlem bulunmadı.")
    else:
        st.dataframe(trades_df.tail(50), use_container_width=True)

        # CSV kaydet + indir
        out_csv = os.path.join(RESULTS_DIR, f"trades_{symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        trades_df.to_csv(out_csv, index=False)
        st.success(f"CSV kaydedildi: {out_csv}")
        st.download_button("CSV indir", data=trades_df.to_csv(index=False), file_name=os.path.basename(out_csv), mime="text/csv")
