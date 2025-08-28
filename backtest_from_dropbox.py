# backtest_from_dropbox.py
import os, io, math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------
# Streamlit config (İLK komut olmalı)
# -----------------------------------------------------------
st.set_page_config(page_title="EMA+1H Onay + LRC/TSI/Hacim – Backtest", layout="wide")

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

# =========================== Parquet / Veri ============================

def _dropbox_variants(url: str) -> List[str]:
    """Dropbox paylaşımları için işe yarayan birkaç doğrudan indirme varyantı üret."""
    variants = []
    # orijinal
    variants.append(url)
    # dl=1
    if "dl=" in url:
        variants.append(url.replace("dl=0", "dl=1").replace("dl=2", "dl=1"))
    else:
        sep = "&" if "?" in url else "?"
        variants.append(url + f"{sep}dl=1")
    # raw=1
    if "raw=" in url:
        variants.append(url.replace("raw=0", "raw=1"))
    else:
        sep = "&" if "?" in url else "?"
        variants.append(url + f"{sep}raw=1")
    # dl.dropboxusercontent.com
    try:
        from urllib.parse import urlparse, urlunparse, parse_qs
        u = urlparse(url)
        if "dropbox.com" in u.netloc:
            u2 = list(u)
            u2[1] = "dl.dropboxusercontent.com"
            # query'yi sadeleştir (çoğu zaman parametre gerekmez)
            u2[4] = ""
            variants.append(urlunparse(u2))
    except Exception:
        pass
    # Tekrarsız sırala
    seen, clean = set(), []
    for v in variants:
        if v not in seen:
            clean.append(v); seen.add(v)
    return clean

def _is_parquet_bytes(raw: bytes) -> bool:
    """Parquet dosyaları PAR1 ile başlar ve biter."""
    if not raw or len(raw) < 8:
        return False
    return raw[:4] == b"PAR1" and raw[-4:] == b"PAR1"

@st.cache_data(show_spinner=True)
def load_parquet_from_dropbox(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    tried = []
    for cand in _dropbox_variants(url):
        try:
            tried.append(cand)
            r = requests.get(cand, timeout=120, allow_redirects=True, headers=headers)
            r.raise_for_status()
            raw = r.content
            ctype = (r.headers.get("Content-Type") or "").lower()
            size_mb = len(raw)/1_000_000
            st.caption(f"İndirildi: {cand} | Tip: {ctype} | Boyut: {size_mb:.2f} MB")

            # HTML/önizleme ise geç
            if b"<!DOCTYPE html" in raw[:400] or "text/html" in ctype:
                continue

            # Parquet sihir bayt kontrolü
            if not _is_parquet_bytes(raw):
                # Bazı sunucular 'application/octet-stream' veriyor; yine de deneyelim
                bio = io.BytesIO(raw)
                try:
                    df = pd.read_parquet(bio, engine="fastparquet")
                    return _standardize(df)
                except Exception:
                    bio.seek(0)
                    try:
                        df = pd.read_parquet(bio, engine="pyarrow")  # pyarrow yoksa zaten hata
                        return _standardize(df)
                    except Exception:
                        # Gerçekten parquet değil
                        continue
            else:
                bio = io.BytesIO(raw)
                try:
                    df = pd.read_parquet(bio, engine="fastparquet")
                except Exception:
                    bio.seek(0)
                    df = pd.read_parquet(bio, engine="pyarrow")
                return _standardize(df)
        except Exception:
            continue

    st.error("Parquet okunamadı. Denenen URL'ler:\n" + "\n".join(tried))
    return pd.DataFrame()

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    c = {x.lower(): x for x in df.columns}
    # time
    tcol = c.get("time") or c.get("timestamp")
    if not tcol: raise ValueError("time/timestamp kolonu bulunamadı")
    df = df.rename(columns={tcol: "time"})
    # temel kolonlar
    for k in ["open","high","low","close","volume"]:
        if k not in df.columns:
            # büyük/küçük varyasyon
            alts = [col for col in df.columns if col.lower()==k]
            if alts: df = df.rename(columns={alts[0]:k})
            else: raise ValueError(f"{k} kolonu yok")
    # interval
    icol = None
    for col in df.columns:
        if col.lower() == "interval":
            icol = col; break
    if icol is None:
        df["interval_norm"] = ""
    else:
        df["interval_norm"] = df[icol].astype(str).str.lower()

    # symbol
    if "symbol" not in df.columns:
        alts = [col for col in df.columns if col.lower()=="symbol"]
        if alts: df = df.rename(columns={alts[0]:"symbol"})
        else: df["symbol"] = "UNKNOWN"

    df["time"] = pd.to_datetime(df["time"], utc=False)
    return df.sort_values("time").reset_index(drop=True)

# =========================== Göstergeler ============================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def tsi(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    m = close.diff()
    e1 = m.ewm(span=r, adjust=False).mean()
    e2 = e1.ewm(span=s, adjust=False).mean()
    a1 = m.abs().ewm(span=r, adjust=False).mean()
    a2 = a1.ewm(span=s, adjust=False).mean()
    return 100 * (e2 / a2)

# ----- LRC (günlük, 300) -----
def _lrc_last_point(values: np.ndarray) -> float:
    w = np.asarray(values, dtype=float).ravel()
    n = w.size
    if n < 2 or not np.isfinite(w).all(): return np.nan
    x = np.arange(n, dtype=float)
    m, b = np.polyfit(x, w, 1)
    return m*(n-1)+b

def compute_lrc_bands_daily(df_daily: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    def rolling_lrc(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(length, min_periods=length).apply(_lrc_last_point, raw=True)
    out = pd.DataFrame(index=df_daily.index)
    out["lrc_high"] = rolling_lrc(df_daily["high"], length)
    out["lrc_low"]  = rolling_lrc(df_daily["low"],  length)
    return out

# ----- Patternler -----
def is_bull_engulf(prev: pd.Series, cur: pd.Series) -> bool:
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
    return float(df5.iloc[i0:i+1]["low"].min()) if i0 <= i else None

def last_swing_high(df5: pd.DataFrame, i: int, lookback: int) -> Optional[float]:
    i0 = max(0, i - lookback)
    return float(df5.iloc[i0:i+1]["high"].max()) if i0 <= i else None

# =========================== Pozisyon / Simülasyon ============================

@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop: float
    tp: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    fee_entry: float = 0.0
    fee_exit: float = 0.0

    @property
    def pnl_usd(self) -> float:
        if self.exit_price is None: return 0.0
        d = 1 if self.side=="long" else -1
        gross = d * (self.exit_price - self.entry_price) * self.qty
        return gross - self.fee_entry - self.fee_exit

def compute_stats(trades_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df.empty:
        return {"trades":0,"win_rate":0.0,"net_pnl":0.0,"avg_pnl":0.0,"max_consec_win":0,"max_consec_loss":0,"max_dd":0.0}
    wins = (trades_df["pnl_usd"]>0).astype(int)
    trades = len(trades_df)
    win_rate = 100*wins.sum()/trades
    net_pnl = trades_df["pnl_usd"].sum()
    avg_pnl = trades_df["pnl_usd"].mean()

    max_w=max_l=cur_w=cur_l=0
    for pnl in trades_df["pnl_usd"]:
        if pnl>0:
            cur_w+=1; max_w=max(max_w,cur_w); cur_l=0
        elif pnl<0:
            cur_l+=1; max_l=max(max_l,cur_l); cur_w=0
        else:
            cur_w=cur_l=0

    eq = trades_df["pnl_usd"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()

    return {"trades":trades, "win_rate":float(win_rate), "net_pnl":float(net_pnl),
            "avg_pnl":float(avg_pnl), "max_consec_win":int(max_w), "max_consec_loss":int(max_l),
            "max_dd":float(max_dd)}

def simulate(
    df5: pd.DataFrame, df1h: pd.DataFrame, dfd: pd.DataFrame,
    *, allow_long: bool, allow_short: bool,
    use_lrc: bool, use_tsi_dw: bool, tsi_r: int, tsi_s: int,
    vol_filter: bool, vol_factor: float,
    need_engulf: bool, need_pullback_ema13: bool,
    atr_period: int, atr_mult: float, swing_lookback: int, stop_offset_bps: float,
    rr: float, risk_mode: str, risk_value: float, leverage: float, fee_rate: float,
    start_dt: pd.Timestamp, end_dt: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,float]]:

    # EMA'lar
    for L in [7,13,26]:
        df5[f"ema{L}"] = ema(df5["close"], L)
    df1h["ema7"]  = ema(df1h["close"], 7)
    df1h["ema13"] = ema(df1h["close"], 13)

    # ATR/hacim
    df5["atr"] = atr(df5, atr_period)
    df5["vol_ma20"] = df5["volume"].rolling(20, min_periods=20).mean()

    # TSI D+W
    if use_tsi_dw:
        dfd["tsi_d"] = tsi(dfd["close"], tsi_r, tsi_s)
        dfw = dfd.resample("W").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        dfw["tsi_w"] = tsi(dfw["close"], tsi_r, tsi_s)
    else:
        dfd["tsi_d"] = np.nan
        dfw = dfd.resample("W").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        dfw["tsi_w"] = np.nan

    # LRC
    if use_lrc:
        bands = compute_lrc_bands_daily(dfd, 300)
        dfd = dfd.join(bands)

    def last_1h(ts):
        return df1h.loc[:ts].iloc[-1] if (df1h.index<=ts).any() else None
    def last_d(ts):
        day = pd.Timestamp(ts.date())
        return dfd.loc[:day].iloc[-1] if (dfd.index<=day).any() else None

    def pos_qty(entry, stop, equity_now):
        dist = abs(entry-stop)
        if dist<=0 or not np.isfinite(dist): return 0.0
        risk_usd = (equity_now*(risk_value/100.0)) if risk_mode=="dynamic_pct" else risk_value
        qty = risk_usd/dist
        notional = qty*entry
        max_notional = equity_now*leverage
        if notional>max_notional:
            qty = max_notional/entry
        return max(qty,0.0)

    equity_rows, trades = [], []
    eq = 1000.0  # sadece risk yüzdesi için taban; gerçek sermayeyi UI’dan yönetmiyoruz (PnL toplanıyor)
    position: Optional[Trade] = None

    for i in range(len(df5)):
        ts = df5.index[i]
        if ts < start_dt or ts > end_dt:
            continue
        row5 = df5.iloc[i]
        row1h = last_1h(ts)
        if row1h is None or pd.isna(row1h["ema7"]) or pd.isna(row1h["ema13"]):
            continue

        # ZORUNLU EMA koşulları
        ema_ok_long  = (row5["ema7"] > row5["ema13"] > row5["ema26"]) and (row1h["ema7"] > row1h["ema13"])
        ema_ok_short = (row5["ema7"] < row5["ema13"] < row5["ema26"]) and (row1h["ema7"] < row1h["ema13"])

        # LRC
        lrc_long = lrc_short = True
        if use_lrc:
            drow = last_d(ts)
            if drow is None or ("lrc_high" not in drow or pd.isna(drow["lrc_high"]) or pd.isna(drow["lrc_low"])):
                lrc_long = lrc_short = False
            else:
                lrc_long  = row5["close"] > drow["lrc_high"]
                lrc_short = row5["close"] < drow["lrc_low"]

        # TSI
        tsi_long = tsi_short = True
        if use_tsi_dw:
            drow = last_d(ts)
            wrow = dfd.resample("W").last().loc[:ts].iloc[-1] if (dfd.resample("W").last().index<=ts).any() else None
            if drow is None or wrow is None or pd.isna(drow["tsi_d"]) or pd.isna(wrow.get("tsi_w", np.nan)):
                tsi_long = tsi_short = False
            else:
                tsi_long  = (drow["tsi_d"]>0) and (wrow["tsi_w"]>0)
                tsi_short = (drow["tsi_d"]<0) and (wrow["tsi_w"]<0)

        # Hacim
        vol_pass = True
        if vol_filter:
            vol_pass = (not pd.isna(row5["vol_ma20"])) and (row5["volume"] >= vol_factor*row5["vol_ma20"])

        # Engulfing
        eng_long = eng_short = True
        if need_engulf and i>=1:
            prev = df5.iloc[i-1][["open","high","low","close"]]
            cur  = row5[["open","high","low","close"]]
            eng_long  = is_bull_engulf(prev, cur)
            eng_short = is_bear_engulf(prev, cur)

        # Pullback EMA13
        pb_long = pb_short = True
        if need_pullback_ema13:
            pb_long  = row5["low"]  <= row5["ema13"]
            pb_short = row5["high"] >= row5["ema13"]

        # Açık pozisyonu yönet
        if position is not None:
            if position.side=="long":
                hit_sl = row5["low"]  <= position.stop
                hit_tp = row5["high"] >= position.tp
                reason = None; price = None
                if hit_sl and hit_tp: reason, price = "SL", position.stop
                elif hit_sl:          reason, price = "SL", position.stop
                elif hit_tp:          reason, price = "TP", position.tp
                if reason is None:
                    # kural bozuldu mu?
                    if not (ema_ok_long and (not use_lrc or lrc_long) and (not use_tsi_dw or tsi_long)):
                        reason, price = "RuleExit", row5["close"]
                if reason:
                    fee_exit = (fee_rate/100.0) * (price*position.qty)
                    position.exit_time = ts
                    position.exit_price = price
                    position.exit_reason = reason
                    position.fee_exit = fee_exit
                    trades.append(position)
                    eq += position.pnl_usd
                    position = None
            else:
                hit_sl = row5["high"] >= position.stop
                hit_tp = row5["low"]  <= position.tp
                reason = None; price = None
                if hit_sl and hit_tp: reason, price = "SL", position.stop
                elif hit_sl:          reason, price = "SL", position.stop
                elif hit_tp:          reason, price = "TP", position.tp
                if reason is None:
                    if not (ema_ok_short and (not use_lrc or lrc_short) and (not use_tsi_dw or tsi_short)):
                        reason, price = "RuleExit", row5["close"]
                if reason:
                    fee_exit = (fee_rate/100.0) * (price*position.qty)
                    position.exit_time = ts
                    position.exit_price = price
                    position.exit_reason = reason
                    position.fee_exit = fee_exit
                    trades.append(position)
                    eq += position.pnl_usd
                    position = None

        # Giriş (tek pozisyon kuralı)
        if position is None:
            # LONG
            if allow_long and ema_ok_long and lrc_long and tsi_long and vol_pass and eng_long and pb_long:
                entry = row5["close"]
                sl_sw = last_swing_low(df5, i-1, swing_lookback)
                if sl_sw is not None and np.isfinite(sl_sw):
                    off = entry*(stop_offset_bps/10_000.0)
                    sl1 = sl_sw - off
                    atrv = row5["atr"]
                    sl2 = entry - atr_mult*atrv if np.isfinite(atrv) else entry*0.99
                    stop = min(sl1, sl2)  # long: daha aşağı (geniş) stop
                    dist = entry - stop
                    if dist>0:
                        tp = entry + rr*dist
                        qty = pos_qty(entry, stop, equity_now=max(eq,1.0))
                        if qty>0:
                            fee_in = (fee_rate/100.0)*(entry*qty)
                            position = Trade("long", ts, entry, qty, stop, tp, fee_entry=fee_in)
            # SHORT
            if position is None and allow_short and ema_ok_short and lrc_short and tsi_short and vol_pass and eng_short and pb_short:
                entry = row5["close"]
                sh_sw = last_swing_high(df5, i-1, swing_lookback)
                if sh_sw is not None and np.isfinite(sh_sw):
                    off = entry*(stop_offset_bps/10_000.0)
                    sl1 = sh_sw + off
                    atrv = row5["atr"]
                    sl2 = entry + atr_mult*atrv if np.isfinite(atrv) else entry*1.01
                    stop = max(sl1, sl2)  # short: daha yukarı (geniş) stop
                    dist = stop - entry
                    if dist>0:
                        tp = entry - rr*dist
                        qty = pos_qty(entry, stop, equity_now=max(eq,1.0))
                        if qty>0:
                            fee_in = (fee_rate/100.0)*(entry*qty)
                            position = Trade("short", ts, entry, qty, stop, tp, fee_entry=fee_in)

        equity_rows.append({"time": ts, "equity": eq})

    # Son bar kapat
    if position is not None:
        last_close = df5.iloc[-1]["close"]
        fee_exit = (fee_rate/100.0) * (last_close*position.qty)
        position.exit_time = df5.index[-1]
        position.exit_price = last_close
        position.exit_reason = "LastBarExit"
        position.fee_exit = fee_exit
        trades.append(position)
        eq += position.pnl_usd
        position = None

    equity_df = pd.DataFrame(equity_rows).set_index("time")
    trades_df = pd.DataFrame([{
        "side": t.side,
        "entry_time": t.entry_time, "entry": t.entry_price,
        "exit_time": t.exit_time,   "exit": t.exit_price,
        "exit_reason": t.exit_reason,
        "qty": t.qty, "stop": t.stop, "tp": t.tp,
        "pnl_usd": t.pnl_usd
    } for t in trades if t.exit_time is not None])

    stats = compute_stats(trades_df)
    return trades_df, equity_df, stats

# =========================== UI ============================

st.title("📊 EMA (5m+1h Onay) + Opsiyonel LRC/TSI/Hacim/Engulf/Pullback Backtest")

with st.expander("🔗 Veri Kaynağı", expanded=True):
    c0, c1, c2 = st.columns([2,3,2])
    with c0:
        mode = st.radio("Kaynak", ["Futures","Spot"], horizontal=True)
    with c1:
        url_input = st.text_input("Dropbox Parquet Linki",
            DEFAULT_FUTURES_URL if mode=="Futures" else DEFAULT_SPOT_URL)
        st.caption("İpucu: Link herkese açık + **dl=1/raw=1** olmalı. Sorun olursa alttan dosya yükleyebilirsin.")
    with c2:
        uploaded = st.file_uploader("Dosya Yükle (Parquet)", type=["parquet"])
    st.button("🔁 Önbelleği temizle", on_click=st.cache_data.clear)

# Veri oku
df = pd.DataFrame()
if uploaded is not None:
    try:
        df = pd.read_parquet(uploaded, engine="fastparquet")
        df = _standardize(df)
        st.success("Yerel Parquet yüklendi ✅")
    except Exception as e:
        try:
            uploaded.seek(0)
            df = pd.read_parquet(uploaded, engine="pyarrow")
            df = _standardize(df); st.success("Yerel Parquet (pyarrow) yüklendi ✅")
        except Exception as e2:
            st.error(f"Yerel Parquet okunamadı: {e2}")
            st.stop()
elif url_input:
    df = load_parquet_from_dropbox(url_input)
else:
    st.stop()

if df.empty:
    st.stop()

# Semboller
symbols = sorted(list(df["symbol"].dropna().unique()))
if not symbols: symbols = ["BTCUSDT","ETHUSDT"]

cA, cB, cC = st.columns(3)
with cA:
    symbol = st.selectbox("Sembol", symbols, index=(symbols.index("ETHUSDT") if "ETHUSDT" in symbols else 0))
with cB:
    start_date = st.date_input("Başlangıç", pd.Timestamp("2023-01-01"))
with cC:
    end_date = st.date_input("Bitiş", pd.Timestamp("2024-01-01"))

st.caption("⚠️ LRC(300) için ~300 günlük warm-up gerekir. Kod başlangıcı otomatik ~320 gün geriye çeker.")

# Sembole indir
df_sym = df[df["symbol"]==symbol].copy()
df_sym["time"] = pd.to_datetime(df_sym["time"])
df_sym = df_sym.sort_values("time")

# 5m / 1h ayrıştır
if (df_sym["interval_norm"]!="").any():
    df5 = df_sym[df_sym["interval_norm"].isin(["5m","5min","5"])].set_index("time").sort_index()
    df1h= df_sym[df_sym["interval_norm"].isin(["1h","60","60m"])].set_index("time").sort_index()
    df5 = df5.resample("5T").last()[["open","high","low","close","volume"]].dropna()
    df1h= df1h.resample("1H").last()[["open","high","low","close","volume"]].dropna()
else:
    base = df_sym.set_index("time").sort_index()[["open","high","low","close","volume"]]
    df5 = base.resample("5T").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    df1h= base.resample("1H").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

# Günlük (1h'den resample)
dfd_all = df1h.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

# Warm start
warm_days = 320
start_dt = pd.Timestamp(start_date)
end_dt   = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
warm_start = start_dt - pd.Timedelta(days=warm_days)

df5_w  = df5.loc[warm_start:end_dt].copy()
df1h_w = df1h.loc[warm_start:end_dt].copy()
dfd_w  = dfd_all.loc[:end_dt].copy()

with st.expander("⚙️ Kurallar & Filtreler", expanded=True):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        allow_long  = st.checkbox("Long Aç", True)
        allow_short = st.checkbox("Short Aç", True)
    with c2:
        use_lrc     = st.checkbox("LRC (günlük 300)", False)
        use_tsi_dw  = st.checkbox("TSI D+W", False)
    with c3:
        vol_filter  = st.checkbox("Hacim filtresi (≥ k×MA20)", False)
        vol_factor  = st.number_input("k", value=1.0, step=0.1)
        need_engulf = st.checkbox("Engulf şartı", False)
    with c4:
        need_pull   = st.checkbox("EMA13'e çekilme", False)
        atr_period  = st.number_input("ATR(5m) periyodu", value=14, step=1, min_value=1)
        atr_mult    = st.number_input("ATR çarpanı (stop)", value=2.0, step=0.1, min_value=0.1)

    c5,c6,c7 = st.columns(3)
    with c5:
        swing_look = st.number_input("Swing lookback (bar)", value=20, step=1, min_value=2)
        stop_bps   = st.number_input("Stop offset (bps)", value=5.0, step=1.0, help="1 bps = 0.01%")
    with c6:
        rr         = st.number_input("R:R (TP/SL)", value=2.0, step=0.1, min_value=0.1)
        tsi_r      = st.number_input("TSI r", value=25, step=1, min_value=1)
        tsi_s      = st.number_input("TSI s", value=13, step=1, min_value=1)
    with c7:
        risk_mode  = st.selectbox("Risk modu", ["dynamic_pct","fixed_usd"], format_func=lambda x: "Dinamik % (equity)" if x=="dynamic_pct" else "Sabit USD")
        risk_val   = st.number_input("Risk % / Sabit USD", value=2.0 if risk_mode=="dynamic_pct" else 100.0, step=0.1 if risk_mode=="dynamic_pct" else 10.0, min_value=0.1)
        leverage   = st.number_input("Kaldıraç (x)", value=10.0, step=1.0, min_value=1.0)
        fee_rate   = st.number_input("Komisyon (her bacak, %)", value=0.04, step=0.01, min_value=0.0)

run = st.button("▶ Backtest Çalıştır")

if run:
    if df5_w.empty or df1h_w.empty:
        st.error("Veri aralığı boş.")
        st.stop()

    trades_df, equity_df, stats = simulate(
        df5=df5_w, df1h=df1h_w, dfd=dfd_w,
        allow_long=allow_long, allow_short=allow_short,
        use_lrc=use_lrc, use_tsi_dw=use_tsi_dw, tsi_r=tsi_r, tsi_s=tsi_s,
        vol_filter=vol_filter, vol_factor=vol_factor,
        need_engulf=need_engulf, need_pullback_ema13=need_pull,
        atr_period=atr_period, atr_mult=atr_mult,
        swing_lookback=swing_look, stop_offset_bps=stop_bps,
        rr=rr, risk_mode=risk_mode, risk_value=risk_val,
        leverage=leverage, fee_rate=fee_rate,
        start_dt=start_dt, end_dt=end_dt
    )

    st.subheader("📈 Sonuç Özeti")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Toplam İşlem", int(stats.get("trades",0)))
    m2.metric("Win Rate", f"{stats.get('win_rate',0.0):.2f}%")
    m3.metric("Net PnL (USD)", f"{stats.get('net_pnl',0.0):.2f}")
    m4.metric("Ort. PnL (USD)", f"{stats.get('avg_pnl',0.0):.2f}")
    m5.metric("Max Consec Win", int(stats.get("max_consec_win",0)))
    m6.metric("Max Consec Loss", int(stats.get("max_consec_loss",0)))

    st.subheader("Equity")
    if equity_df.empty:
        st.info("Equity verisi boş.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["equity"], mode="lines", name="Equity"))
        fig.update_layout(height=420, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("İşlem Günlüğü")
    if trades_df.empty:
        st.info("İşlem bulunmadı.")
    else:
        st.dataframe(trades_df.tail(100), use_container_width=True)
        out_csv = os.path.join(RESULTS_DIR, f"trades_{symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        trades_df.to_csv(out_csv, index=False)
        st.success(f"CSV kaydedildi: {out_csv}")
        st.download_button("CSV indir", data=trades_df.to_csv(index=False), file_name=os.path.basename(out_csv), mime="text/csv")
