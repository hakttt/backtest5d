# backtest_from_dropbox.py
# Çalıştırma: streamlit run backtest_from_dropbox.py

# --- self-heal: eksik paketleri kur (Cloud bazen requirements'i atlayabiliyor) ---
import sys, subprocess

def ensure(pkg, spec=None):
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec or pkg])
        __import__(pkg)

# Bu ikisi bazen 3.13 ortamında düşebiliyor:
ensure("plotly", "plotly==5.22.0")
ensure("fastparquet", "fastparquet==2024.5.0")

# --- normal importlar ---
import os
import io
import time
import math
import tempfile
from typing import Tuple, Optional

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- SAYFA ----------
st.set_page_config(page_title="LRC + EMA Backtest (5m giriş + 1h onay)", layout="wide")

# ---------- KULLANICI LİNKLERİ ----------
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

# ---------- DİZİNLER (kalıcı değil; reboot'ta silinebilir) ----------
CACHE_DIR = os.path.join(tempfile.gettempdir(), "dropbox_cache")
EXPORT_DIR = os.path.join(tempfile.gettempdir(), "backtest_exports")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------- YARDIMCI FONKSİYONLAR ----------
def to_direct_link(url: str) -> str:
    if "dropbox.com" in url and "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url

def stream_download(url: str, dst_path: str, chunk=1024 * 1024, timeout=180):
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
                    st.caption(f"İndiriliyor… {downloaded/1e6:.1f}/{total/1e6:.1f} MB "
                               f"({downloaded*100/total:.1f}%)")
        st.success(f"İndirme tamam: {downloaded/1e6:.1f} MB, {time.time()-t0:.1f} sn")

def ensure_local_copy(name: str, url: str) -> str:
    local_path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(local_path):
        st.info("Dropbox’tan indiriliyor…")
        direct = to_direct_link(url)
        try:
            stream_download(direct, local_path)
        except requests.HTTPError as e:
            st.error(f"HTTP hatası: {e}")
            raise
        except requests.RequestException as e:
            st.error(f"Ağ hatası: {e}")
            raise
    else:
        st.caption(f"Cache’de bulundu: {local_path}")
    return local_path

@st.cache_data(show_spinner=False)
def load_parquet(local_path: str) -> pd.DataFrame:
    # fastparquet yüklü (self-heal ile)
    df = pd.read_parquet(local_path)
    # Zaman damgası
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            raise ValueError("DatetimeIndex yok ve 'timestamp' kolonu bulunamadı.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    # Kolon kontrolleri
    needed = {"open","high","low","close","volume","symbol","timeframe"}
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Eksik kol(lar): {miss}")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["timeframe"] = df["timeframe"].astype(str)
    return df.sort_index()

# ---------- EMA / ATR / FORMASYON ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def is_engulfing(df: pd.DataFrame, bullish=True) -> pd.Series:
    o,c,h,l = df["open"], df["close"], df["high"], df["low"]
    po, pc, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    if bullish:
        cond = (pc < po) & (c > o) & (h >= ph) & (l <= pl)
    else:
        cond = (pc > po) & (c < o) & (h <= ph) & (l >= pl)
    return cond.fillna(False)

# ---------- SWING & PİP ----------
def estimate_tick(close: pd.Series) -> float:
    diffs = (close - close.shift(1)).dropna().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 0.01
    guess = diffs.quantile(0.1)
    exp = math.floor(math.log10(guess)) if guess > 0 else -2
    base = guess / (10 ** exp)
    for b in [1,2,5,10]:
        if base <= b:
            step = b * (10 ** exp)
            break
    return max(step, 1e-6)

def last_swing_low(high: pd.Series, low: pd.Series, lookback: int = 5) -> pd.Series:
    return low.shift(1).rolling(lookback, min_periods=1).min()

def last_swing_high(high: pd.Series, low: pd.Series, lookback: int = 5) -> pd.Series:
    return high.shift(1).rolling(lookback, min_periods=1).max()

# ---------- LRC (senin verdiğin) ----------
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

def price_position_vs_lrc(close: pd.Series, lrc_high: pd.Series, lrc_low: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    out['dist_to_high'] = close.astype(float) - lrc_high.astype(float)
    out['dist_to_low']  = close.astype(float) - lrc_low.astype(float)
    out['band_mid']  = (lrc_high + lrc_low) / 2.0
    out['band_half'] = (lrc_high - lrc_low) / 2.0
    out['norm_pos'] = (close - out['band_mid']) / out['band_half']
    cond_above = close > lrc_high
    cond_below = close < lrc_low
    out['pos_label'] = np.where(cond_above, 'above_high', np.where(cond_below, 'below_low', 'between'))
    return out

# ---------- VERİ AYIRMA ----------
def split_timeframes(big: pd.DataFrame, symbol: str):
    df5_full = big[(big["symbol"] == symbol) & (big["timeframe"] == "5m")][["open","high","low","close","volume"]].copy()
    df1h_full = big[(big["symbol"] == symbol) & (big["timeframe"] == "1h")][["open","high","low","close","volume"]].copy()
    return df5_full.sort_index(), df1h_full.sort_index()

def compute_indicators_full(df5_full: pd.DataFrame, df1h_full: pd.DataFrame, lrc_length: int):
    # 5m EMA/ATR
    df5_full["ema7"]  = ema(df5_full["close"], 7)
    df5_full["ema13"] = ema(df5_full["close"], 13)
    df5_full["ema26"] = ema(df5_full["close"], 26)
    df5_full["ATR"]   = atr(df5_full, 14)

    # 1h EMA
    df1h_full["ema7"]  = ema(df1h_full["close"], 7)
    df1h_full["ema13"] = ema(df1h_full["close"], 13)

    # 5m LRC bantları
    lrc = compute_lrc_bands(df5_full, length=lrc_length)
    df5_full["lrc_high"] = lrc["lrc_high"]
    df5_full["lrc_low"]  = lrc["lrc_low"]
    pos = price_position_vs_lrc(df5_full["close"], df5_full["lrc_high"], df5_full["lrc_low"])
    df5_full["lrc_mid"]   = pos["band_mid"]
    # Chop ölçüsü: bant genişliği / ATR
    df5_full["lrc_width"] = (df5_full["lrc_high"] - df5_full["lrc_low"]) / df5_full["ATR"].replace(0, np.nan)
    return df5_full, df1h_full

def align_1h_to_5m(df5: pd.DataFrame, df1h_full: pd.DataFrame) -> pd.DataFrame:
    h1 = df1h_full[["ema7","ema13"]].copy()
    h1_aligned = h1.reindex(df5.index, method="ffill")
    df5["h1_ema7"]  = h1_aligned["ema7"]
    df5["h1_ema13"] = h1_aligned["ema13"]
    return df5

# ---------- SİNYALLER ----------
def make_signals(df5: pd.DataFrame,
                 use_lrc_trigger: bool,
                 use_pullback: bool,
                 pullback_thr: float,
                 use_engulf: bool,
                 use_lrc_chop: bool,
                 lrc_chop_min: float) -> pd.DataFrame:
    out = df5.copy()

    # ZORUNLU KOŞULLAR (değiştirilemez):
    # 5m hizası: EMA7 > EMA13 > EMA26 (long) / EMA7 < EMA13 < EMA26 (short)
    out["bull5"]  = (out["ema7"] > out["ema13"]) & (out["ema13"] > out["ema26"])
    out["bear5"]  = (out["ema7"] < out["ema13"]) & (out["ema13"] < out["ema26"])
    # 1h onay: EMA7 > EMA13 (long) / EMA7 < EMA13 (short)
    out["bull1h"] = (out["h1_ema7"] > out["h1_ema13"])
    out["bear1h"] = (out["h1_ema7"] < out["h1_ema13"])

    # Opsiyonel filtreler
    pull_ok = pd.Series(True, index=out.index)
    eng_bull = pd.Series(True, index=out.index)
    eng_bear = pd.Series(True, index=out.index)
    chop_ok = pd.Series(True, index=out.index)

    if use_pullback:
        dist = (out["close"] - out["ema13"]).abs() / out["ATR"].replace(0, np.nan)
        pull_ok = dist <= pullback_thr
    if use_engulf:
        eng_bull = is_engulfing(out, bullish=True)
        eng_bear = is_engulfing(out, bullish=False)
    if use_lrc_chop:
        chop_ok = out["lrc_width"] >= lrc_chop_min

    base_long  = out["bull5"] & out["bull1h"]
    base_short = out["bear5"] & out["bear1h"]

    if use_lrc_trigger:
        long_trg  = out["close"] > out["lrc_high"]
        short_trg = out["close"] < out["lrc_low"]
        base_long  = base_long  & long_trg
        base_short = base_short & short_trg

    out["long_signal"]  = base_long  & pull_ok & chop_ok & eng_bull
    out["short_signal"] = base_short & pull_ok & chop_ok & eng_bear
    return out

# ---------- BACKTEST ----------
def backtest(
    df: pd.DataFrame,
    rr: float = 2.0,
    atr_stop_mult: float = 2.0,
    entry_offset_bps: float = 0.0,
    fee_rate: float = 0.0004,
    initial_equity: float = 1000.0,
    risk_mode: str = "dynamic",   # "dynamic" | "fixed"
    fixed_amount: float = 100.0,
    risk_pct: float = 0.02,
    leverage: float = 10.0,
    swing_override: bool = False,
    swing_lookback: int = 5,
    pip_override: Optional[float] = None,
    cooldown_bars: int = 0
):
    o,h,l,c = df["open"], df["high"], df["low"], df["close"]
    atrv = df["ATR"].fillna(method="bfill").fillna(method="ffill")
    long_sig  = df["long_signal"].fillna(False)
    short_sig = df["short_signal"].fillna(False)

    pip_step = pip_override if (pip_override and pip_override > 0) else estimate_tick(c)
    sw_low  = last_swing_low(df["high"], df["low"], lookback=swing_lookback)
    sw_high = last_swing_high(df["high"], df["low"], lookback=swing_lookback)

    trades = []
    equity = initial_equity
    eq_curve = []
    in_pos = False
    side = None
    entry_price = sl = tp = np.nan
    cooldown_left = 0

    def position_size(price):
        nonlocal equity
        nominal = (equity * risk_pct) * leverage if risk_mode == "dynamic" else (fixed_amount * leverage)
        qty = nominal / price if price else 0.0
        return max(qty, 0.0), nominal

    off_mult_up   = 1.0 + (entry_offset_bps / 10000.0)
    off_mult_down = 1.0 / off_mult_up

    for i in range(2, len(df)):
        ts = df.index[i]
        if cooldown_left > 0:
            cooldown_left -= 1

        if not in_pos and cooldown_left == 0:
            if long_sig.iloc[i-1]:
                ent = o.iloc[i] * off_mult_up
                sl_dist_atr = atrv.iloc[i] * atr_stop_mult
                if swing_override:
                    swing_sl = (sw_low.iloc[i] - pip_step)
                    swing_dist = ent - swing_sl
                    sl_dist = max(sl_dist_atr, swing_dist if np.isfinite(swing_dist) else 0)
                else:
                    sl_dist = sl_dist_atr
                entry_price = float(ent); sl = entry_price - sl_dist; tp = entry_price + rr * sl_dist
                side = "long"; in_pos = True
                qty, nominal = position_size(entry_price)
                fee_open = nominal * fee_rate
                trades.append({"time": ts, "side": side, "entry": entry_price, "sl": sl, "tp": tp,
                               "qty": qty, "nominal": nominal, "fee_open": fee_open})
            elif short_sig.iloc[i-1]:
                ent = o.iloc[i] * off_mult_down
                sl_dist_atr = atrv.iloc[i] * atr_stop_mult
                if swing_override:
                    swing_sl = (sw_high.iloc[i] + pip_step)
                    swing_dist = swing_sl - ent
                    sl_dist = max(sl_dist_atr, swing_dist if np.isfinite(swing_dist) else 0)
                else:
                    sl_dist = sl_dist_atr
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
                result = "SL_first"  # muhafazakar: önce SL'e değdi varsay
                exit_price = sl
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
                if cooldown_bars > 0:
                    cooldown_left = cooldown_bars

        eq_curve.append((ts, equity))

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq_curve, columns=["time","equity"]).set_index("time")

    if trades_df.empty:
        stats = {"trades": 0, "winrate": 0.0, "net_total": 0.0, "max_dd": 0.0,
                 "final_equity": float(equity),
                 "max_win_streak": 0, "max_loss_streak": 0}
    else:
        wins = (trades_df["net"] > 0)
        losses = ~wins

        def max_streak(mask: pd.Series):
            max_len = 0; cur = 0; start = None; best = (0, None, None)
            idx = trades_df["exit_time"].reset_index(drop=True)
            for k, v in enumerate(mask.fillna(False)):
                if v:
                    cur += 1
                    if cur == 1:
                        start = idx[k]
                    if cur > max_len:
                        max_len = cur
                        best = (max_len, start, idx[k])
                else:
                    cur = 0; start = None
            return best

        win_best = max_streak(wins)
        loss_best = max_streak(losses)

        net_total = float(trades_df["net"].sum())
        winrate = 100.0 * wins.sum() / len(trades_df)
        eq = eq_df["equity"].fillna(method="ffill")
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = float(dd.min() * 100.0)
        stats = {
            "trades": int(len(trades_df)),
            "winrate": float(winrate),
            "net_total": net_total,
            "max_dd": max_dd,
            "final_equity": float(eq.iloc[-1]) if len(eq) else float(equity),
            "max_win_streak": int(win_best[0]),
            "max_win_streak_start": win_best[1],
            "max_win_streak_end": win_best[2],
            "max_loss_streak": int(loss_best[0]),
            "max_loss_streak_start": loss_best[1],
            "max_loss_streak_end": loss_best[2],
        }
    return trades_df, stats, eq_df

# ---------- UI ----------
st.title("LRC + EMA Backtest • 5m giriş + 1h onay (BTC/ETH)")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    dataset_choice = st.radio("Dataset", ["Futures", "Spot"], index=0, horizontal=True)
    fut_url = st.text_input("Futures URL", value=DEFAULT_FUTURES_URL)
    spot_url = st.text_input("Spot URL", value=DEFAULT_SPOT_URL)

    st.subheader("Seçimler")
    symbol = st.selectbox("Sembol", ["BTCUSDT","ETHUSDT"], index=1)

    st.caption("Backtest dönemi (UTC)")
    start_date = st.date_input("Başlangıç", value=None)
    end_date   = st.date_input("Bitiş", value=None)

    st.subheader("LRC & Filtreler")
    use_lrc_trg = st.checkbox("LRC tetikleyici (Long: Close > LRC_HIGH, Short: Close < LRC_LOW)", value=True)
    lrc_len = st.number_input("LRC length (5m)", min_value=50, max_value=1000, value=300, step=10)
    use_pullback = st.checkbox("Pullback: |C-EMA13|/ATR ≤ eşik", value=False)
    pull_thr = st.number_input("Pullback eşiği", min_value=0.1, max_value=3.0, value=0.5, step=0.1, disabled=not use_pullback)
    use_engulf = st.checkbox("Engulfing filtresi (5m, yön uyumlu)", value=False)
    use_lrc_chop = st.checkbox("LRC chop filtresi: (LRC_H − LRC_L)/ATR ≥ eşik", value=False)
    lrc_chop_min = st.number_input("Chop min", min_value=0.1, max_value=5.0, value=1.0, step=0.1, disabled=not use_lrc_chop)

    st.subheader("Risk / Pozisyon")
    rr = st.number_input("Risk/Ödül (TP/SL)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    atr_mult = st.number_input("SL = ATR ×", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    entry_off = st.number_input("Giriş offset (bps)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    swing_override = st.checkbox("Swing-stop override (long: swing low − 1 pip, short: swing high + 1 pip)", value=False)
    swing_lb = st.number_input("Swing lookback (bar)", min_value=2, max_value=50, value=5, step=1, disabled=not swing_override)
    pip_manual = st.text_input("Pip adımı (boş=otomatik)", value="")

    st.subheader("Sermaye / İşlem")
    init_eq = st.number_input("Başlangıç Sermaye (USDT)", min_value=100.0, max_value=1_000_000.0, value=1000.0, step=100.0)
    lev = st.number_input("Kaldıraç (x)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    fee = st.number_input("Komisyon (her bacak)", min_value=0.0, max_value=0.005, value=0.0004, step=0.0001, format="%.4f")
    risk_mode = st.selectbox("Risk modu", ["dynamic","fixed"], index=0)
    fixed_amt = st.number_input("Sabit nominal (USDT)", min_value=10.0, max_value=100000.0, value=100.0, step=10.0, disabled=(risk_mode!="fixed"))
    risk_pct = st.number_input("Risk % (dynamic)", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f", disabled=(risk_mode!="dynamic"))
    cooldown = st.number_input("Cooldown (bar)", min_value=0, max_value=100, value=0, step=1)

    run_btn = st.button("Yükle & Backtest", type="primary")

if run_btn:
    # 1) Veri indir/oku
    use_url = fut_url if dataset_choice == "Futures" else spot_url
    stub = "futures" if dataset_choice == "Futures" else "spot"
    local_name = f"{stub}_btc_eth.parquet"
    try:
        local_path = ensure_local_copy(local_name, use_url)
        big = load_parquet(local_path)
    except Exception:
        st.stop()

    st.success(f"Yüklendi: {len(big):,} satır | Semboller: {sorted(big['symbol'].unique())} | TF: {sorted(big['timeframe'].unique())}")

    # 2) 5m/1h ayır (full)
    df5_full, df1h_full = split_timeframes(big, symbol)
    if df5_full.empty or df1h_full.empty:
        st.error("Seçilen sembol için 5m veya 1h veri yok.")
        st.stop()

    # 3) Tüm veri üstünde warm-up + göstergeler
    df5_full, df1h_full = compute_indicators_full(df5_full, df1h_full, lrc_length=int(lrc_len))

    # 4) Tarih filtresi en sonda (warm-up korunsun)
    df5 = df5_full
    if start_date:
        df5 = df5[df5.index >= pd.Timestamp(start_date).tz_localize("UTC")]
    if end_date:
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        df5 = df5[df5.index < end_ts.tz_localize("UTC")]

    # 1h EMA’ları 5m’ye eşle
    df5 = align_1h_to_5m(df5, df1h_full)

    # Kısa aralık uyarısı (sadece çok kısa ise)
    if len(df5) < max(350, int(lrc_len) + 30):
        st.warning("Uyarı: Seçilen aralık kısa olabilir. LRC/EMA warm-up sonrası sinyal az çıkabilir.")

    # 5) Sinyaller
    sig = make_signals(
        df5,
        use_lrc_trigger=use_lrc_trg,
        use_pullback=use_pullback,
        pullback_thr=float(pull_thr),
        use_engulf=use_engulf,
        use_lrc_chop=use_lrc_chop,
        lrc_chop_min=float(lrc_chop_min)
    )

    # 6) Backtest
    try:
        pip_val = float(pip_manual) if pip_manual.strip() else None
    except:
        pip_val = None

    trades, stats, eq = backtest(
        sig,
        rr=rr,
        atr_stop_mult=atr_mult,
        entry_offset_bps=entry_off,
        fee_rate=fee,
        initial_equity=init_eq,
        risk_mode=risk_mode,
        fixed_amount=fixed_amt,
        risk_pct=risk_pct,
        leverage=lev,
        swing_override=swing_override,
        swing_lookback=int(swing_lb),
        pip_override=pip_val,
        cooldown_bars=int(cooldown)
    )

    # 7) Grafik (işlem işaretleri dahil)
    st.subheader("Grafik (5m) — EMA & LRC & İşlemler")
    view = sig.copy()
    fig = go.Figure(data=[go.Candlestick(
        x=view.index, open=view["open"], high=view["high"], low=view["low"], close=view["close"],
        name="5m"
    )])
    # EMA'lar
    for col, nm in [("ema7","EMA7"), ("ema13","EMA13"), ("ema26","EMA26")]:
        if col in view:
            fig.add_trace(go.Scatter(x=view.index, y=view[col], name=nm, mode="lines"))
    # LRC bantları
    if "lrc_high" in view and "lrc_low" in view:
        fig.add_trace(go.Scatter(x=view.index, y=view["lrc_high"], name="LRC High", mode="lines"))
        fig.add_trace(go.Scatter(x=view.index, y=view["lrc_low"],  name="LRC Low",  mode="lines"))
    # Entry/Exit marker'ları
    if len(trades):
        long_ent = trades[trades["side"]=="long"]
        short_ent = trades[trades["side"]=="short"]
        if len(long_ent):
            fig.add_trace(go.Scatter(
                x=long_ent["time"], y=long_ent["entry"], mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=10)))
        if len(short_ent):
            fig.add_trace(go.Scatter(
                x=short_ent["time"], y=short_ent["entry"], mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=10)))
        exited = trades.dropna(subset=["exit_time"])
        if len(exited):
            tp_rows = exited[exited["result"].str.contains("TP", na=False)]
            sl_rows = exited[exited["result"].str.contains("SL", na=False)]
            if len(tp_rows):
                fig.add_trace(go.Scatter(
                    x=tp_rows["exit_time"], y=tp_rows["exit"], mode="markers", name="TP",
                    marker=dict(symbol="circle", size=8)))
            if len(sl_rows):
                fig.add_trace(go.Scatter(
                    x=sl_rows["exit_time"], y=sl_rows["exit"], mode="markers", name="SL",
                    marker=dict(symbol="x", size=9)))
    fig.update_layout(xaxis_rangeslider_visible=False, height=540, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    # 8) Özet + Equity
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Equity Eğrisi")
        st.line_chart(eq)
    with c2:
        st.subheader("Özet")
        pretty = {
            "Toplam İşlem": stats.get("trades", 
