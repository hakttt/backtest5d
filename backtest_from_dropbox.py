# ===============================
# backtest_from_dropbox.py
# ===============================

import streamlit as st
st.set_page_config(
    page_title="LRC(1D,300) + EMA Backtest (5m giriş + 1h onay)",
    layout="wide"
)

import pandas as pd
import numpy as np
import requests
import io
import os
import datetime as dt

# Plotly sessiz import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    st.error("Gerekli paket eksik: plotly. Lütfen requirements.txt içinde `plotly==5.22.0` olduğundan emin olun ve uygulamayı yeniden başlatın.")
    st.stop()


# =========================================================
# ---- Yardımcılar
# =========================================================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

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

def compute_lrc_bands(df_ohlc: pd.DataFrame, length: int = 300) -> pd.DataFrame:
    out = pd.DataFrame(index=df_ohlc.index)
    out["lrc_high"] = rolling_lrc(df_ohlc["high"], length=length)
    out["lrc_low"]  = rolling_lrc(df_ohlc["low"],  length=length)
    return out

@st.cache_data(show_spinner=True)
def load_parquet_from_dropbox(url: str) -> pd.DataFrame:
    """
    Dropbox paylaşımlı linkten tek parquet yükler.
    Tek dosyada birden çok timeframe/symbol olabilir.
    """
    try:
        link = url
        if link.endswith("?dl=0"):
            link = link.replace("?dl=0", "?dl=1")
        elif "dl=" not in link:
            link += "?dl=1"
        r = requests.get(link, timeout=60)
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
        # Zorunlu kolon standartları
        cols = {c.lower(): c for c in df.columns}
        # normalize: timeframe/symbol varsa düzenle
        if "timeframe" in cols:
            tfcol = cols["timeframe"]
            df[tfcol] = df[tfcol].astype(str).str.lower().str.strip()
        if "symbol" in cols:
            sycol = cols["symbol"]
            df[sycol] = df[sycol].astype(str).str.upper().str.strip()
        # index zaman olsun
        if not isinstance(df.index, pd.DatetimeIndex):
            # timestamp veya open_time türü var mı?
            for cand in ["timestamp", "time", "date", "datetime", "open_time"]:
                if cand in df.columns:
                    df[cand] = pd.to_datetime(df[cand], utc=True, errors="coerce")
                    df = df.set_index(cand)
                    break
        # UTC enforce
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        return df.sort_index()
    except Exception as e:
        st.error(f"Veri okunamadı: {e}")
        return pd.DataFrame()


def pick_symbol_timeframes(df: pd.DataFrame):
    # symbol listesi
    if "symbol" in df.columns:
        symbols = sorted(df["symbol"].dropna().unique().tolist())
    else:
        symbols = ["(tek sembol)"]
        df["symbol"] = "(tek sembol)"
    sym = st.selectbox("Sembol", symbols, index=0)

    # timeframe listesi
    if "timeframe" in df.columns:
        tfs = sorted(df["timeframe"].dropna().unique().tolist())
        st.caption(f"Bulunan timeframeler: {tfs}")
        # 5m ve 1h seçimi
        tf_5m = "5m" if "5m" in tfs else st.selectbox("5m için timeframe seçin", tfs, index=0, key="tf5m")
        tf_1h = "1h" if "1h" in tfs else st.selectbox("1h için timeframe seçin", tfs, index=min(1, len(tfs)-1), key="tf1h")
        df5_all  = df[df["timeframe"] == tf_5m]
        df1h_all = df[df["timeframe"] == tf_1h]
    else:
        st.warning("timeframe kolonu bulunamadı: veri tek TF olabilir. 5m varsayımı yapılacak, 1h resample edilecek.")
        df5_all = df.copy()
        # 1h resample
        df1h_all = df5_all.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})

    # sembole göre filtre
    df5  = df5_all[df5_all["symbol"] == sym] if "symbol" in df5_all.columns else df5_all
    df1h = df1h_all[df1h_all["symbol"] == sym] if "symbol" in df1h_all.columns else df1h_all

    # OHLCV zorunlu kolonlar
    needed = ["open","high","low","close","volume"]
    for need in needed:
        if need not in df5.columns:
            st.error(f"5m verisi '{need}' kolonu içermiyor.")
            st.stop()
        if need not in df1h.columns:
            st.error(f"1h verisi '{need}' kolonu içermiyor (veya resample başarısız).")
            st.stop()

    return sym, df5.sort_index(), df1h.sort_index()


def compute_indicators_and_align(df5_full: pd.DataFrame, df1h_full: pd.DataFrame, lrc_len: int = 300):
    """ 5m EMA/ATR, 1h EMA ve 1h->1D→LRC(1D) hesapla; 1D LRC'yi 5m'e ffill ile eşle """
    # 5m EMA'lar
    df5 = df5_full.copy()
    df5["ema7_5m"]  = ema(df5["close"], 7)
    df5["ema13_5m"] = ema(df5["close"], 13)
    df5["ema26_5m"] = ema(df5["close"], 26)

    # 1h EMA'lar
    df1h = df1h_full.copy()
    df1h["ema7_1h"]  = ema(df1h["close"], 7)
    df1h["ema13_1h"] = ema(df1h["close"], 13)

    # 1h -> 1D resample (UTC gün)
    d1 = df1h.copy()
    if d1.index.tz is None:
        d1.index = d1.index.tz_localize("UTC")
    else:
        d1.index = d1.index.tz_convert("UTC")
    d1 = d1.resample("1D", label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","high","low","close"])

    # 1D LRC(300)
    bands_d1 = compute_lrc_bands(d1, length=int(lrc_len))
    d1 = d1.join(bands_d1)

    # 1D LRC'yi 5m index'ine ffill
    df5["lrc_high"] = d1["lrc_high"].reindex(df5.index, method="ffill")
    df5["lrc_low"]  = d1["lrc_low"].reindex(df5.index, method="ffill")

    # 1h EMA'ları 5m'e ffill
    h1 = df1h[["ema7_1h","ema13_1h"]].dropna().sort_index()
    h1_aligned = h1.reindex(df5.index, method="ffill")
    df5 = df5.join(h1_aligned)

    return df5, d1


def make_signals(df5: pd.DataFrame, use_lrc_long: bool, use_lrc_short: bool):
    # EMA zorunlu koşullar
    bull5  = (df5["ema7_5m"] > df5["ema13_5m"]) & (df5["ema13_5m"] > df5["ema26_5m"])
    bear5  = (df5["ema7_5m"] < df5["ema13_5m"]) & (df5["ema13_5m"] < df5["ema26_5m"])
    bull1h = (df5["ema7_1h"] > df5["ema13_1h"]) & df5["ema7_1h"].notna() & df5["ema13_1h"].notna()
    bear1h = (df5["ema7_1h"] < df5["ema13_1h"]) & df5["ema7_1h"].notna() & df5["ema13_1h"].notna()

    ema_ok_long  = bull5 & bull1h
    ema_ok_short = bear5 & bear1h

    # LRC tetikleyici (opsiyonel)
    lrc_ok_long  = (~use_lrc_long)  | (df5["close"] > df5["lrc_high"])
    lrc_ok_short = (~use_lrc_short) | (df5["close"] < df5["lrc_low"])

    long_entry  = ema_ok_long  & lrc_ok_long
    short_entry = ema_ok_short & lrc_ok_short

    diag = {
        "bull5": int(bull5.sum()),
        "bear5": int(bear5.sum()),
        "bull1h": int(bull1h.sum()),
        "bear1h": int(bear1h.sum()),
        "combo_long": int((ema_ok_long).sum()),
        "combo_short": int((ema_ok_short).sum()),
        "lrc_long_true": int(((df5["close"] > df5["lrc_high"]).fillna(False)).sum()),
        "lrc_short_true": int(((df5["close"] < df5["lrc_low"]).fillna(False)).sum()),
        "long_entry": int(long_entry.sum()),
        "short_entry": int(short_entry.sum()),
    }
    out = df5.copy()
    out["long_entry"] = long_entry
    out["short_entry"] = short_entry
    return out, diag


def backtest(df5: pd.DataFrame, stop_on_swing: bool = True, only_one_position: bool = True):
    """
    Basit yürütme:
     - Sinyal bar kapanışında giriş
     - Tek pozisyon kuralı
     - Stop: swing low/high hafif ötesi (tick)
     - TP tanımlı değil (şimdilik)
    """
    trades = []
    position = None
    entry = None
    stop = None

    for ts, row in df5.iterrows():
        # Entry
        if position is None:
            if bool(row.get("long_entry", False)):
                position = "long"
                entry = float(row["close"])
                stop = float(row["low"]) - 0.0001 if stop_on_swing else None
                trades.append({"time": ts, "type": "entry", "side": "long", "price": entry})
            elif bool(row.get("short_entry", False)):
                position = "short"
                entry = float(row["close"])
                stop = float(row["high"]) + 0.0001 if stop_on_swing else None
                trades.append({"time": ts, "type": "entry", "side": "short", "price": entry})
        else:
            # Stop logic
            if position == "long" and stop is not None:
                if float(row["low"]) <= stop:
                    trades.append({"time": ts, "type": "exit", "side": "long", "price": stop})
                    position, entry, stop = None, None, None
            elif position == "short" and stop is not None:
                if float(row["high"]) >= stop:
                    trades.append({"time": ts, "type": "exit", "side": "short", "price": stop})
                    position, entry, stop = None, None, None

    # Kapanmayan son pozisyonu yok sayıyoruz (flat’a kapatma yok)
    # PnL listesi (pip/price farkı)
    results = []
    last_side = None
    last_price = None
    for tr in trades:
        if tr["type"] == "entry":
            last_side = tr["side"]
            last_price = tr["price"]
        elif tr["type"] == "exit" and last_side is not None:
            if last_side == "long":
                results.append(tr["price"] - last_price)
            else:
                results.append(last_price - tr["price"])
            last_side = None
            last_price = None

    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r <= 0)
    total = len(results)

    # ardışık seri
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = cur_loss = 0
    for r in results:
        if r > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    stats = {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "winrate_pct": round((wins / total * 100) if total else 0.0, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }
    return stats, trades


def plot_overview(df5: pd.DataFrame, sym: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(
        x=df5.index,
        open=df5["open"], high=df5["high"], low=df5["low"], close=df5["close"],
        name=f"{sym} 5m"
    ), row=1, col=1)

    # EMA'lar
    fig.add_trace(go.Scatter(x=df5.index, y=df5["ema7_5m"], name="EMA7 (5m)", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df5.index, y=df5["ema13_5m"], name="EMA13 (5m)", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df5.index, y=df5["ema26_5m"], name="EMA26 (5m)", mode="lines"), row=1, col=1)

    # LRC bantları
    if "lrc_high" in df5.columns and df5["lrc_high"].notna().any():
        fig.add_trace(go.Scatter(x=df5.index, y=df5["lrc_high"], name="LRC High (1D,300)", mode="lines"), row=1, col=1)
    if "lrc_low" in df5.columns and df5["lrc_low"].notna().any():
        fig.add_trace(go.Scatter(x=df5.index, y=df5["lrc_low"], name="LRC Low (1D,300)", mode="lines"), row=1, col=1)

    # Hacim
    if "volume" in df5.columns:
        fig.add_trace(go.Bar(x=df5.index, y=df5["volume"], name="Volume"), row=2, col=1)

    fig.update_layout(height=700, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# ---- UI
# =========================================================
st.title("📊 LRC(1D,300) + EMA Backtest")
st.caption("Giriş: 5m (EMA7>EMA13>EMA26) + 1h (EMA7>EMA13). LRC(1D,300) long=Close>LRC_HIGH, short=Close<LRC_LOW (opsiyonel).")

parquet_url = st.text_input("Parquet Dropbox linki (spot veya futures)", value="", help="Tek dosyada 5m ve 1h barlar olmalı (örn: 5m-1h-1w-1M).")
use_lrc_long  = st.checkbox("LRC Long filtresi (Close > LRC_HIGH)", value=False)
use_lrc_short = st.checkbox("LRC Short filtresi (Close < LRC_LOW)", value=False)
stop_on_swing = st.checkbox("Stop: son swing low/high çevresine koy", value=True)
lrc_len = st.number_input("LRC(1D) pencere (gün)", min_value=50, max_value=600, value=300, step=10)

col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input("Başlangıç", value=dt.date(2023, 1, 1))
with col_d2:
    end_date = st.date_input("Bitiş", value=dt.date(2023, 12, 31))

run = st.button("Backtest Çalıştır")

if run:
    df_all = load_parquet_from_dropbox(parquet_url)
    if df_all.empty:
        st.stop()

    sym, df5_full, df1h_full = pick_symbol_timeframes(df_all)

    # İndikatörler önce full veri üstünde (warm-up için), sonra tarih kırp
    df5_full, d1_full = compute_indicators_and_align(df5_full, df1h_full, lrc_len=int(lrc_len))

    # LRC warm-up teşhisi
    try:
        days_have = int(d1_full[["lrc_high","lrc_low"]].dropna().shape[0])
        if days_have < int(lrc_len):
            st.warning(f"Uyarı: LRC(1D) için yeterli gün yok: {days_have}/{int(lrc_len)}. LRC filtreleri açıksa sinyal az olabilir.")
    except Exception:
        pass

    # Tarih filtresi (UTC gün saat 00:00–)
    st.caption(f"Seçilen aralık: {start_date} → {end_date}")
    df5 = df5_full.loc[str(start_date):str(end_date)].copy()

    # Sinyaller
    df5_sig, diag = make_signals(df5, use_lrc_long, use_lrc_short)

    # Teşhis sayaçları
    st.caption(
        f"bull5:{diag['bull5']} | bear5:{diag['bear5']} | "
        f"bull1h:{diag['bull1h']} | bear1h:{diag['bear1h']} | "
        f"combo_long(EMA):{diag['combo_long']} | combo_short(EMA):{diag['combo_short']} | "
        f"LRC_long_true:{diag['lrc_long_true']} | LRC_short_true:{diag['lrc_short_true']} | "
        f"long_entry:{diag['long_entry']} | short_entry:{diag['short_entry']}"
    )

    if diag["long_entry"] + diag["short_entry"] == 0:
        st.warning("Seçilen kurallarla bu aralıkta giriş sinyali üretilmedi. (LRC’yi kapatmayı veya aralığı/genişliği değiştirmeyi deneyin.)")

    # Backtest
    stats, trades = backtest(df5_sig, stop_on_swing=stop_on_swing)

    st.subheader("Özet")
    st.write({
        "Sembol": sym,
        "Toplam İşlem": stats["trades"],
        "Kazanılan": stats["wins"],
        "Kaybedilen": stats["losses"],
        "Winrate (%)": stats["winrate_pct"],
        "Max Win Streak": stats["max_win_streak"],
        "Max Loss Streak": stats["max_loss_streak"],
    })

    # Çizim
    plot_overview(df5_sig, sym)

    # CSV kaydı
    try:
        os.makedirs("artifacts", exist_ok=True)
        out = pd.DataFrame(trades)
        out.to_csv("artifacts/backtest_results.csv", index=False)
        st.success("Sonuçlar kaydedildi: artifacts/backtest_results.csv")
    except Exception as e:
        st.warning(f"CSV yazılamadı: {e}")
