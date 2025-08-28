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
    """Dropbox paylaşımlı linkten parquet yükler."""
    try:
        link = url.strip()
        if link.endswith("?dl=0"):
            link = link.replace("?dl=0", "?dl=1")
        elif "dl=" not in link:
            sep = "&" if "?" in link else "?"
            link = f"{link}{sep}dl=1"
        r = requests.get(link, timeout=60)
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
        # Index zaman olsun
        if not isinstance(df.index, pd.DatetimeIndex):
            # olası kolon adları
            for cand in ["timestamp","time","date","datetime","open_time"]:
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
        # normalize bazı kolonlar
        if "timeframe" in df.columns:
            df["timeframe"] = df["timeframe"].astype(str).str.lower().str.strip()
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        return df.sort_index()
    except Exception as e:
        st.error(f"Veri okunamadı: {e}")
        return pd.DataFrame()

def pick_symbol_timeframes(df: pd.DataFrame):
    # sembol seçimi
    if "symbol" in df.columns:
        symbols = sorted(df["symbol"].dropna().unique().tolist())
    else:
        symbols = ["(tek sembol)"]
        df["symbol"] = "(tek sembol)"
    sym = st.selectbox("Sembol", symbols, index=0)

    # timeframe seçimi
    if "timeframe" in df.columns:
        tfs = sorted(df["timeframe"].dropna().unique().tolist())
        with st.expander("Bulunan timeframeler", expanded=False):
            st.write(tfs)
        tf_5m = "5m" if "5m" in tfs else st.selectbox("5m için TF seçin", tfs, index=0, key="tf5m")
        tf_1h = "1h" if "1h" in tfs else st.selectbox("1h için TF seçin", tfs, index=min(1, len(tfs)-1), key="tf1h")
        df5_all  = df[df["timeframe"] == tf_5m]
        df1h_all = df[df["timeframe"] == tf_1h]
    else:
        st.warning("timeframe kolonu yok. 5m varsayımı yapılıyor; 1h verisi 5m’den resample edilecek.")
        df5_all = df.copy()
        df1h_all = df5_all.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})

    # sembole göre kes
    df5  = df5_all[df5_all["symbol"] == sym] if "symbol" in df5_all.columns else df5_all
    df1h = df1h_all[df1h_all["symbol"] == sym] if "symbol" in df1h_all.columns else df1h_all

    # OHLCV zorunlu kontrol
    for need in ["open","high","low","close","volume"]:
        if need not in df5.columns:
            st.error(f"5m verisi '{need}' kolonu içermiyor.")
            st.stop()
        if need not in df1h.columns:
            st.error(f"1h verisi '{need}' kolonu içermiyor (resample gerekebilir).")
            st.stop()

    return sym, df5.sort_index(), df1h.sort_index()

def compute_indicators_and_align(df5_full: pd.DataFrame, df1h_full: pd.DataFrame, lrc_len: int = 300):
    """ 5m EMA, 1h EMA ve 1D LRC(300) (1h→1D) hesapla; LRC'yi 5m'e ffill ile eşle """
    df5 = df5_full.copy()
    df1h = df1h_full.copy()

    # 5m EMA'lar
    df5["ema7_5m"]  = ema(df5["close"], 7)
    df5["ema13_5m"] = ema(df5["close"], 13)
    df5["ema26_5m"] = ema(df5["close"], 26)

    # 1h EMA'lar
    df1h["ema7_1h"]  = ema(df1h["close"], 7)
    df1h["ema13_1h"] = ema(df1h["close"], 13)

    # 1h -> 1D
    d1 = df1h.copy()
    if d1.index.tz is None:
        d1.index = d1.index.tz_localize("UTC")
    else:
        d1.index = d1.index.tz_convert("UTC")
    d1 = d1.resample("1D", label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","high","low","close"])

    # 1D LRC
    bands_d1 = compute_lrc_bands(d1, length=int(lrc_len))
    d1 = d1.join(bands_d1)

    # 1D LRC'yi 5m'e ffill
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
    out["long_entry_sig"] = long_entry
    out["short_entry_sig"] = short_entry
    return out, diag

def max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0

def profit_factor(pl_list):
    gains = sum(x for x in pl_list if x > 0)
    losses = -sum(x for x in pl_list if x < 0)
    return (gains / losses) if losses > 0 else np.inf if gains > 0 else 0.0


# =========================================================
# ---- Backtest (risk/sermaye/komisyon/kaldıraç dahil)
# =========================================================
def run_backtest_exe(
    df5: pd.DataFrame,
    stop_on_swing: bool,
    use_atr_stop: bool,
    atr_period: int,
    atr_mult: float,
    rr_tp: float,
    starting_equity: float,
    risk_mode: str,      # "fixed" | "percent"
    risk_value: float,   # fixed $ veya yüzde
    leverage: float,
    fee_rate_each_leg: float,  # % -> 0.05 = %0.05
):
    df = df5.copy()
    # ATR (5m) yalnızca gerekirse
    if use_atr_stop:
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(atr_period, min_periods=atr_period).mean()

    equity = starting_equity
    position = None
    entry_price = None
    qty = 0.0
    stop_price = None
    tp_price = None
    trades = []
    equity_curve = []

    price_tick = 0.0001  # sembole göre değişebilir; basit olması için küçük bir adım

    pl_list = []
    max_consec_win = 0
    max_consec_loss = 0
    cur_win = cur_loss = 0

    for ts, row in df.iterrows():
        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])

        # pozisyon yoksa — giriş
        if position is None:
            long_sig = bool(row.get("long_entry_sig", False))
            short_sig = bool(row.get("short_entry_sig", False))

            if long_sig or short_sig:
                side = "long" if long_sig else "short"

                # stop mesafesi
                swing_stop = None
                if stop_on_swing:
                    if side == "long":
                        swing_stop = low - price_tick
                    else:
                        swing_stop = high + price_tick

                atr_stop = None
                if use_atr_stop and not np.isnan(row.get("ATR", np.nan)):
                    dist = float(row["ATR"]) * float(atr_mult)
                    if side == "long":
                        atr_stop = close - dist
                    else:
                        atr_stop = close + dist

                # stop önceliği: swing > atr > yok
                stop_price = swing_stop if swing_stop is not None else atr_stop

                if stop_price is None or (side == "long" and stop_price >= close) or (side == "short" and stop_price <= close):
                    # geçersiz stop (ters yönde), bu sinyali atla
                    continue

                stop_dist = (close - stop_price) if side == "long" else (stop_price - close)
                if stop_dist <= 0:
                    continue

                # risk $ belirle
                if risk_mode == "percent":
                    risk_dollars = equity * (risk_value / 100.0)
                else:
                    risk_dollars = risk_value

                if risk_dollars <= 0:
                    continue

                qty = risk_dollars / stop_dist  # risk tabanlı miktar

                # kaldıraç/margin sınırı
                notional = qty * close
                max_notional = equity * leverage  # kullanılabilir en fazla notional (basit yaklaşım)
                if notional > max_notional:
                    # kıs
                    qty = max_notional / close
                    notional = qty * close
                    # stop_dist aynı kalır; risk fiilen düşer

                # TP
                tp_dist = stop_dist * rr_tp
                tp_price = (close + tp_dist) if side == "long" else (close - tp_dist)

                # giriş komisyonu
                entry_fee = notional * (fee_rate_each_leg / 100.0)

                position = side
                entry_price = close

                trades.append({
                    "time": ts, "type": "entry", "side": side,
                    "price": entry_price, "qty": qty,
                    "stop": stop_price, "tp": tp_price,
                    "fee": entry_fee
                })

                # equity değişmez (fee, PnL bar kapanışında düşebilir) – burada düşürelim:
                equity -= entry_fee
                equity_curve.append({"time": ts, "equity": equity})

        else:
            # pozisyon açık — exit (önce stop, sonra TP)
            if position == "long":
                hit_stop = low <= stop_price if stop_price is not None else False
                hit_tp   = high >= tp_price if tp_price is not None else False
                exit_reason = None
                exit_price = None
                if hit_stop:
                    exit_reason = "SL"
                    exit_price = stop_price
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = tp_price

                if exit_price is not None:
                    notional_exit = qty * exit_price
                    exit_fee = notional_exit * (fee_rate_each_leg / 100.0)
                    pnl = (exit_price - entry_price) * qty  # long

                    # equity güncelle
                    equity += pnl
                    equity -= exit_fee

                    trades.append({
                        "time": ts, "type": "exit", "side": position,
                        "price": exit_price, "qty": qty,
                        "reason": exit_reason, "fee": exit_fee, "pnl": pnl
                    })

                    # win/loss serisi
                    pl_list.append(pnl - 0.0)  # fee zaten düşüldü
                    if pnl > 0:
                        cur_win += 1; cur_loss = 0
                    else:
                        cur_loss += 1; cur_win = 0
                    max_consec_win = max(max_consec_win, cur_win)
                    max_consec_loss = max(max_consec_loss, cur_loss)

                    # reset
                    position = None; entry_price = None; qty = 0.0
                    stop_price = None; tp_price = None

                    equity_curve.append({"time": ts, "equity": equity})

            elif position == "short":
                hit_stop = high >= stop_price if stop_price is not None else False
                hit_tp   = low  <= tp_price if tp_price is not None else False
                exit_reason = None
                exit_price = None
                if hit_stop:
                    exit_reason = "SL"
                    exit_price = stop_price
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = tp_price

                if exit_price is not None:
                    notional_exit = qty * exit_price
                    exit_fee = notional_exit * (fee_rate_each_leg / 100.0)
                    pnl = (entry_price - exit_price) * qty  # short

                    equity += pnl
                    equity -= exit_fee

                    trades.append({
                        "time": ts, "type": "exit", "side": position,
                        "price": exit_price, "qty": qty,
                        "reason": exit_reason, "fee": exit_fee, "pnl": pnl
                    })

                    pl_list.append(pnl - 0.0)
                    if pnl > 0:
                        cur_win += 1; cur_loss = 0
                    else:
                        cur_loss += 1; cur_win = 0
                    max_consec_win = max(max_consec_win, cur_win)
                    max_consec_loss = max(max_consec_loss, cur_loss)

                    position = None; entry_price = None; qty = 0.0
                    stop_price = None; tp_price = None

                    equity_curve.append({"time": ts, "equity": equity})

    # istatistikler
    total_trades = sum(1 for t in trades if t["type"] == "exit")
    wins = sum(1 for t in trades if t["type"] == "exit" and t.get("pnl", 0) > 0)
    losses = total_trades - wins
    winrate = round((wins / total_trades * 100) if total_trades else 0.0, 2)
    pf = profit_factor(pl_list)
    mdd = max_drawdown(pd.Series([e["equity"] for e in equity_curve])) if equity_curve else 0.0
    total_pnl = sum(t.get("pnl", 0) for t in trades if t["type"] == "exit")

    stats = {
        "Toplam İşlem": total_trades,
        "Kazanılan": wins,
        "Kaybedilen": losses,
        "Winrate %": winrate,
        "Profit Factor": round(pf, 2) if np.isfinite(pf) else "∞",
        "Toplam PnL ($)": round(total_pnl, 2),
        "Başlangıç Sermaye": round(starting_equity, 2),
        "Bitiş Sermaye": round(equity, 2),
        "Max Win Streak": max_consec_win,
        "Max Loss Streak": max_consec_loss,
        "Max Drawdown %": round(mdd * 100.0, 2) if mdd else 0.0,
    }

    return stats, trades, pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame()


def plot_overview(df5: pd.DataFrame, sym: str, trades: list):
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

    # İşlem işaretleri
    if trades:
        entries_long_x = [t["time"] for t in trades if t["type"]=="entry" and t["side"]=="long"]
        entries_long_y = [df5.loc[x, "close"] for x in entries_long_x]
        entries_short_x = [t["time"] for t in trades if t["type"]=="entry" and t["side"]=="short"]
        entries_short_y = [df5.loc[x, "close"] for x in entries_short_x]
        exits_x = [t["time"] for t in trades if t["type"]=="exit"]
        exits_y = [df5.loc[x, "close"] if x in df5.index else t["price"] for t in trades if t["type"]=="exit"]

        if entries_long_x:
            fig.add_trace(go.Scatter(x=entries_long_x, y=entries_long_y, mode="markers",
                                     name="Long Entry", marker=dict(symbol="triangle-up", size=9)), row=1, col=1)
        if entries_short_x:
            fig.add_trace(go.Scatter(x=entries_short_x, y=entries_short_y, mode="markers",
                                     name="Short Entry", marker=dict(symbol="triangle-down", size=9)), row=1, col=1)
        if exits_x:
            fig.add_trace(go.Scatter(x=exits_x, y=exits_y, mode="markers",
                                     name="Exit", marker=dict(symbol="x", size=8)), row=1, col=1)

    # Hacim
    if "volume" in df5.columns:
        fig.add_trace(go.Bar(x=df5.index, y=df5["volume"], name="Volume"), row=2, col=1)

    fig.update_layout(height=760, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# ---- UI
# =========================================================
st.title("📊 LRC(1D,300) + EMA Backtest")

# Senin linklerin (varsayılan)
DEFAULT_FUTURES_URL = "https://www.dropbox.com/scl/fi/diznny37aq4t88vf62umy/binance_futures_5m-1h-1w-1M_2020-01_2025-08_BTC_ETH.parquet?rlkey=4umoh63qiz3fh0v7xuu86oo5n&st=5wu5h1x1&dl=0"
DEFAULT_SPOT_URL    = "https://www.dropbox.com/scl/fi/eavvv8z452i0b6x1c2a5r/binance_spot_5m-1h-1w-1M_2020-01_2025-08_BTC_ETH.parquet?rlkey=swsjkpbp22v4vj68ggzony8yw&st=2sww3kao&dl=0"

data_source = st.radio("Veri kaynağı", ["Futures (parquet)", "Spot (parquet)"], index=0, horizontal=True)
url_input = st.text_input("Parquet Dropbox linki", value=DEFAULT_FUTURES_URL if data_source.startswith("Futures") else DEFAULT_SPOT_URL)

col_tf1, col_tf2 = st.columns(2)
with col_tf1:
    lrc_len = st.number_input("LRC(1D) pencere (gün)", min_value=50, max_value=600, value=300, step=10)
with col_tf2:
    rr_tp = st.number_input("Risk/Ödül (TP/SL oranı)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

st.markdown("### Filtreler")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    use_lrc_long  = st.checkbox("LRC Long filtresi (Close > LRC_HIGH)", value=False)
with col_f2:
    use_lrc_short = st.checkbox("LRC Short filtresi (Close < LRC_LOW)", value=False)
with col_f3:
    st.info("EMA koşulları zorunlu: 5m (7>13>26) + 1h (7>13). LRC tikleri opsiyonel tetikleyici.")

st.markdown("### Stop seçenekleri")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    stop_on_swing = st.checkbox("Swing stop (son swing low/high çevresi)", value=True)
with col_s2:
    use_atr_stop = st.checkbox("ATR stop kullan", value=False)
with col_s3:
    atr_period = st.number_input("ATR periyodu (5m)", min_value=5, max_value=200, value=14, step=1)
atr_mult = st.number_input("ATR çarpanı (stop mesafesi için)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

st.markdown("### Sermaye & Risk")
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    starting_equity = st.number_input("Başlangıç Sermaye (USDT)", min_value=10.0, value=1000.0, step=10.0)
with col_r2:
    risk_mode = st.selectbox("Risk modu", ["Sabit $", "Yüzde %"], index=1)
with col_r3:
    risk_value = st.number_input("Risk (Sabit $ veya %)", min_value=0.1, value=2.0, step=0.1)

col_e1, col_e2 = st.columns(2)
with col_e1:
    leverage = st.number_input("Kaldıraç (x)", min_value=1.0, value=10.0, step=1.0)
with col_e2:
    fee_rate_each_leg = st.number_input("Komisyon (her bacak, %)", min_value=0.0, value=0.05, step=0.01, help="%0.05 = 0.05 yazınız")

col_d1, col_d2 = st.columns(2)
with col_d1:
    start_date = st.date_input("Başlangıç", value=dt.date(2023, 1, 1))
with col_d2:
    end_date = st.date_input("Bitiş", value=dt.date(2023, 12, 31))

run = st.button("Backtest Çalıştır")

if run:
    df_all = load_parquet_from_dropbox(url_input)
    if df_all.empty:
        st.stop()

    sym, df5_full, df1h_full = pick_symbol_timeframes(df_all)

    # indikatörleri tüm data üzerinde (warm-up) hesapla
    df5_full, d1_full = compute_indicators_and_align(df5_full, df1h_full, lrc_len=int(lrc_len))

    # LRC warm-up uyarısı
    try:
        days_have = int(d1_full[["lrc_high","lrc_low"]].dropna().shape[0])
        if days_have < int(lrc_len):
            st.warning(f"LRC(1D) için yeterli gün yok: {days_have}/{int(lrc_len)}. LRC filtreleri açıksa sinyal sayısı düşebilir.")
    except Exception:
        pass

    # Tarih filtresi
    st.caption(f"Seçilen aralık: {start_date} → {end_date}")
    df5 = df5_full.loc[str(start_date):str(end_date)].copy()

    # sinyaller
    df5_sig, diag = make_signals(df5, use_lrc_long, use_lrc_short)

    st.caption(
        f"bull5:{diag['bull5']} | bear5:{diag['bear5']} | "
        f"bull1h:{diag['bull1h']} | bear1h:{diag['bear1h']} | "
        f"EMA-long:{diag['combo_long']} | EMA-short:{diag['combo_short']} | "
        f"LRC_long_true:{diag['lrc_long_true']} | LRC_short_true:{diag['lrc_short_true']} | "
        f"long_entry:{diag['long_entry']} | short_entry:{diag['short_entry']}"
    )

    if diag["long_entry"] + diag["short_entry"] == 0:
        st.warning("Bu kurallarla bu aralıkta giriş sinyali üretilmedi. (LRC tiklerini kapatmayı, aralığı veya parametreleri değiştirin.)")

    # yürütme
    stats, trades, eq_curve = run_backtest_exe(
        df5_sig,
        stop_on_swing=stop_on_swing,
        use_atr_stop=use_atr_stop,
        atr_period=int(atr_period),
        atr_mult=float(atr_mult),
        rr_tp=float(rr_tp),
        starting_equity=float(starting_equity),
        risk_mode="percent" if risk_mode.startswith("Yüzde") else "fixed",
        risk_value=float(risk_value),
        leverage=float(leverage),
        fee_rate_each_leg=float(fee_rate_each_leg),
    )

    st.subheader("Özet")
    st.write(stats)

    # grafik
    plot_overview(df5_sig, sym, trades)

    # equity grafiği
    if not eq_curve.empty:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve["equity"], mode="lines", name="Equity"))
        fig_e.update_layout(title="Equity Curve", height=360, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_e, use_container_width=True)

    # CSV çıktıları
    try:
        os.makedirs("artifacts", exist_ok=True)
        pd.DataFrame(trades).to_csv("artifacts/backtest_trades.csv", index=False)
        if not eq_curve.empty:
            eq_curve.to_csv("artifacts/backtest_equity.csv")
        st.success("Kaydedildi: artifacts/backtest_trades.csv ve artifacts/backtest_equity.csv")
    except Exception as e:
        st.warning(f"CSV yazılamadı: {e}")
