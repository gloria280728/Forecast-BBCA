import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import requests

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet


# =========================
# FILE PATHS
# =========================
CKPT_PATH = "tft_model.ckpt"
PARAMS_PATH = "tft_dataset_params.pkl"
FUND_PATH = "bbca_fundamentals_quarterly_2021_2023.csv"

TICKER = "BBCA.JK"


PROTECTED = {
    # time index internals
    "relative_time_idx",
    "time_idx_start",
    "encoder_length",
    "decoder_length",

    # target scaling / centering internals (sering muncul)
    "target_scale",
    "target_center",
    "center",
    "scale",

    # kalau target kamu ret_log, ini yang muncul:
    "ret_log_center",
    "ret_log_scale",

    # kalau targetnya Close, kadang juga ada:
    "Close_center",
    "Close_scale",
}
def drop_protected_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c in PROTECTED]
    return df.drop(columns=cols, errors="ignore")

# =========================
# Yahoo realtime (no yfinance)
# =========================
@st.cache_data(ttl=60 * 15)
def load_bbca_price_from_yahoo(period="5y", interval="1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}"
    params = {"range": period, "interval": interval, "includePrePost": "false"}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()

    result = j.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo chart kosong: {j.get('chart', {}).get('error')}")

    res0 = result[0]
    ts = res0.get("timestamp", [])
    q = res0.get("indicators", {}).get("quote", [{}])[0]

    df = pd.DataFrame({
        "Date": pd.to_datetime(ts, unit="s").tz_localize(None),
        "Open": q.get("open"),
        "High": q.get("high"),
        "Low": q.get("low"),
        "Close": q.get("close"),
        "Volume": q.get("volume"),
    }).dropna()

    df = df.sort_values("Date").groupby("Date", as_index=False).last()
    return df


# =========================
# Feature engineering
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["day_sin"] = np.sin(2*np.pi*out["dayofweek"]/7)
    out["day_cos"] = np.cos(2*np.pi*out["dayofweek"]/7)
    out["mon_sin"] = np.sin(2*np.pi*(out["month"]-1)/12)
    out["mon_cos"] = np.cos(2*np.pi*(out["month"]-1)/12)
    return out

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # out["ret_log"] = np.log(out["Close"]).diff()
    out["ret_log"] = np.log(out["Close"]).diff()

    out["ret_log_cum_5"] = out["ret_log"].rolling(5, min_periods=5).sum()
    out["ret_log_sma_5"] = out["ret_log"].rolling(5, min_periods=5).mean()
    out["ret_log_ema_5"] = out["ret_log"].ewm(span=5, adjust=False).mean()
    out["ret_log_ema_10"] = out["ret_log"].ewm(span=10, adjust=False).mean()

    out["SMA_20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()

    w = 20
    m = out["Close"].rolling(w, min_periods=w).mean()
    s = out["Close"].rolling(w, min_periods=w).std()
    out["Bollinger_upper"] = m + 2*s
    out["Bollinger_lower"] = m - 2*s

    out["BB_percentB"] = (out["Close"] - out["Bollinger_lower"]) / (out["Bollinger_upper"] - out["Bollinger_lower"] + 1e-12)
    out["BB_bandwidth"] = (out["Bollinger_upper"] - out["Bollinger_lower"]) / (m + 1e-12)

    out["RSI_14"] = rsi(out["Close"], 14)
    out["MACD_line"], out["MACD_signal"], out["MACD_hist"] = macd(out["Close"])

    out["roll_std_5"] = out["ret_log"].rolling(5, min_periods=5).std()
    out["roll_std_10"] = out["ret_log"].rolling(10, min_periods=10).std()
    return out

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lag_cols = [
        "Close","Volume","SMA_20","EMA_20","BB_percentB","BB_bandwidth",
        "RSI_14","MACD_line","MACD_signal","MACD_hist",
        "ret_log","roll_std_5","roll_std_10",
        "Bollinger_upper","Bollinger_lower",
    ]
    for c in lag_cols:
        if c in out.columns:
            out[c+"_lag1"] = out[c].shift(1)
            out[c+"_lag5"] = out[c].shift(5)
    return out


# =========================
# Fundamentals: quarterly -> daily + QoQ/YoY
# =========================
def quarter_end_date(q_label: str) -> str:
    q_label = (q_label or "").upper().strip()
    mapping = {"Q1":"03-31","Q2":"06-30","Q3":"09-30","Q4":"12-31"}
    return mapping.get(q_label, "12-31")

def _to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("%", "")
    try:
        return float(s)
    except:
        return np.nan

def load_fundamentals_quarterly(path: str) -> pd.DataFrame:
    f = pd.read_csv(path)
    f.columns = [c.strip() for c in f.columns]
    if "Periode" not in f.columns or "Quartal" not in f.columns:
        raise ValueError("Fund CSV harus punya kolom 'Periode' dan 'Quartal'.")

    years = f["Periode"].astype(str).str.extract(r"(\d{4})")[0]
    qend = f["Quartal"].astype(str).map(quarter_end_date)
    f["Date"] = pd.to_datetime(years + "-" + qend, errors="coerce")

    num_cols = [c for c in f.columns if c not in ["Periode","Quartal","Date"]]
    for c in num_cols:
        f[c] = f[c].map(_to_num)

    f = f.sort_values("Date").reset_index(drop=True)
    f["Periode_QoQ"] = np.arange(len(f), dtype=float)          # 0,1,2,... per quarter
    f["Periode_YoY"] = f["Periode_QoQ"] / 4.0

    for c in num_cols:
        f[c + "_QoQ"] = f[c].pct_change(1)
        f[c + "_YoY"] = f[c].pct_change(4)
    base = ["Date", "Periode_QoQ", "Periode_YoY"]
    return f[base + num_cols + [c+"_QoQ" for c in num_cols] + [c+"_YoY" for c in num_cols]]
    # return f[["Date"] + num_cols + [c+"_QoQ" for c in num_cols] + [c+"_YoY" for c in num_cols]]

def fundamentals_to_daily(fq: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    daily = fq.set_index("Date").sort_index().ffill().bfill()
    daily = daily.reindex(pd.to_datetime(dates)).ffill().bfill()
    daily = daily.reset_index().rename(columns={"index":"Date"})
    return daily


# =========================
# Build df_tft + align to params
# =========================
def build_df_tft(price: pd.DataFrame, fund_daily: pd.DataFrame) -> pd.DataFrame:
    df = price.copy()
    df = add_calendar_feats(df)
    df = add_technicals(df)
    df = add_lags(df)

    # shift fundamentals 7 hari (sesuai training kamu)
    # fund_shift = fund_daily.copy()
    # shift_cols = [c for c in fund_shift.columns if c != "Date"]
    # fund_shift[shift_cols] = fund_shift[shift_cols].shift(7)
    # df = df.merge(fund_shift, on="Date", how="left")
    # fund_shift = fund_daily.copy()

    # # pastikan numeric dan terisi
    # fund_cols = [c for c in fund_shift.columns if c != "Date"]
    # for c in fund_cols:
    #     fund_shift[c] = pd.to_numeric(fund_shift[c], errors="coerce")
    # fund_shift[fund_cols] = fund_shift[fund_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # # ✅ shift 7 HARI (bukan shift 7 baris)
    # fund_shift["Date"] = pd.to_datetime(fund_shift["Date"]) + pd.Timedelta(days=7)

    # df = df.merge(fund_shift, on="Date", how="left")
    # df[fund_cols] = df[fund_cols].ffill().bfill()

    df= df.merge(fund_daily, on="Date", how="left")

    df = df.loc[:, ~df.columns.duplicated()].copy()

    # group_id fixed
    df["group_id"] = "BBCA"
    df["group_id"] = df["group_id"].astype(str)

    # time_idx
    df = df.sort_values("Date").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    return df


def align_df_to_dataset_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Pastikan semua kolom yang dibutuhkan params ada dan tidak NA/inf.
    Kalau ada yang missing, dibuat dan diisi 0 lalu ffill/bfill.
    """
    # kolom-kolom yang biasanya ada di params
    need_cols = set()
    for k in [
        "time_varying_known_reals",
        "time_varying_unknown_reals",
        "static_reals",
        "static_categoricals",
        "time_varying_known_categoricals",
        "time_varying_unknown_categoricals",
    ]:
        v = params.get(k, None)
        if isinstance(v, (list, tuple)):
            need_cols |= set(v)

    # selalu butuh ini
    need_cols |= {"time_idx", "group_id"}

    # buat kolom yang belum ada
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    # casting numeric untuk reals
    reals = set()
    for k in ["time_varying_known_reals", "time_varying_unknown_reals", "static_reals"]:
        v = params.get(k, None)
        if isinstance(v, (list, tuple)):
            reals |= set(v)

    for c in reals:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # handle categoricals
    cats = set()
    for k in ["static_categoricals", "time_varying_known_categoricals", "time_varying_unknown_categoricals"]:
        v = params.get(k, None)
        if isinstance(v, (list, tuple)):
            cats |= set(v)
    for c in cats:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # replace inf -> nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # fill: time-series fill untuk semua columns needed (kecuali Date)
    fill_cols = [c for c in need_cols if c in df.columns and c != "Date"]
    df[fill_cols] = df[fill_cols].ffill().bfill()

    return df


def make_future_rows(df_hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Tambah row future (business days) untuk known features.
    Unknowns akan di-fill oleh aligner (ffill) tapi untuk beberapa (Close/ret_log) akan kita set saat recursive.
    """
    df = df_hist.copy().reset_index(drop=True)
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    last_time = int(df["time_idx"].iloc[-1])
    fut = pd.DataFrame({"Date": pd.to_datetime(future_dates)})
    fut["time_idx"] = np.arange(last_time + 1, last_time + 1 + horizon, dtype=np.int64)
    fut["group_id"] = "BBCA"

    fut = add_calendar_feats(fut)  # known calendar feats

    df_all = pd.concat([df, fut], ignore_index=True, sort=False)
    return df_all


# =========================
# Prediction
# =========================

@st.cache_resource
def load_model_and_params():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"CKPT tidak ketemu: {CKPT_PATH}")
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"Params tidak ketemu: {PARAMS_PATH}")

    # params kamu bener: pickle.dump -> pickle.load
    with open(PARAMS_PATH, "rb") as f:
        params = pickle.load(f)

    # ✅ bypass PyTorch 2.6 weights_only default
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # lightning ckpt biasanya dict dengan keys: "state_dict", "hyper_parameters"
    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt["state_dict"]

    # build model dari hparams yg disimpan di ckpt, lalu load weights
    model = TemporalFusionTransformer(**hparams)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, params


def predict_direct(model, params, df_all: pd.DataFrame) -> np.ndarray:
    """
    Direct predict sesuai max_prediction_length yang ada di ckpt.
    Return array shape (horizon,)
    """
    df_all = drop_protected_cols(df_all)
    ds = TimeSeriesDataSet.from_parameters(params, df_all, predict=True, stop_randomization=True)
    loader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    with torch.no_grad():
        pred = model.predict(loader)  # (N, pred_len)
    arr = pred.detach().cpu().numpy()
    return arr[-1]  # last window


def recursive_forecast(model, params, df_hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Untuk model H=1: predict 1 langkah, update Close dengan exp(ret_pred), ulangi.
    """
    df = df_hist.copy().reset_index(drop=True)

    # pastikan ada kolom inti
    if "Close" not in df.columns or "ret_log" not in df.columns:
        raise ValueError("df_hist wajib punya kolom Close dan ret_log")

    last_close = float(df["Close"].iloc[-1])

    rows = []
    future_dates = pd.bdate_range(pd.to_datetime(df["Date"].iloc[-1]) + pd.Timedelta(days=1), periods=horizon)

    for d in future_dates:
        # tambah 1 row future
        next_time = int(df["time_idx"].iloc[-1]) + 1
        new_row = {"Date": pd.to_datetime(d), "time_idx": next_time, "group_id": "BBCA"}
        new_row = pd.DataFrame([new_row])
        new_row = add_calendar_feats(new_row)

        df = pd.concat([df, new_row], ignore_index=True, sort=False)

        # align columns & fill knowns
        df_aligned = align_df_to_dataset_params(df.copy(), params)

        # predict 1-step
        ret_pred_arr = predict_direct(model, params, df_aligned)
        ret_pred = float(ret_pred_arr[0])  # H=1

        close_pred = last_close * float(np.exp(ret_pred))

        # write back to df for next loop
        df.loc[df.index[-1], "ret_log"] = ret_pred
        df.loc[df.index[-1], "Close"] = close_pred

        rows.append({"Date": pd.to_datetime(d), "ret_pred": ret_pred, "Close_pred": close_pred})
        last_close = close_pred

    return pd.DataFrame(rows)

def add_profit_columns(pred_df: pd.DataFrame, anchor_close: float) -> pd.DataFrame:
    out = pred_df.copy()

    # kalau belum ada kolom numerik % nya, bikin dulu
    if "% Untung/Rugi dari Historical Close_num" not in out.columns:
        out["% Untung/Rugi dari Historical Close_num"] = (out["Close_pred"] / anchor_close - 1.0) * 100.0

    if "% Naik/Turun vs Hari Sebelumnya_num" not in out.columns:
        out["% Naik/Turun vs Hari Sebelumnya_num"] = out["Close_pred"].pct_change() * 100.0
        if len(out) > 0:
            out.loc[out.index[0], "% Naik/Turun vs Hari Sebelumnya_num"] = out["% Untung/Rugi dari Historical Close_num"].iloc[0]

    # kolom display string
    def fmt_pct(x):
        if pd.isna(x):
            return ""
        return f"{x:+.4f}%"

    out["% Untung/Rugi dari Historical Close"] = out["% Untung/Rugi dari Historical Close_num"].apply(fmt_pct)
    out["% Naik/Turun vs Hari Sebelumnya"] = out["% Naik/Turun vs Hari Sebelumnya_num"].apply(fmt_pct)

    return out

def get_feature_importance_table(
    model,
    params,
    df_for_interp: pd.DataFrame,
    top_k: int = 25,
    n_batches: int = 5,
) -> pd.DataFrame:
    """
    Ambil feature importance dari TFT lewat variable selection weights langsung dari forward().
    Lebih kompatibel antar versi pytorch-forecasting dibanding interpret_output.
    """
    df_for_interp = drop_protected_cols(df_for_interp)

    ds = TimeSeriesDataSet.from_parameters(params, df_for_interp, predict=True, stop_randomization=True)
    loader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    # nama feature yang dipakai model (biasanya tersedia)
    enc_names = getattr(model, "encoder_variables", None)
    dec_names = getattr(model, "decoder_variables", None)

    enc_sum = None
    dec_sum = None
    enc_count = 0
    dec_count = 0

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            # dataloader biasanya return (x, y) atau dict
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch

            # pindah ke device
            if isinstance(x, dict):
                x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            else:
                # fallback kalau format aneh
                continue

            out = model(x)

            # beberapa versi pakai key berbeda
            enc_w = out.get("encoder_variable_selection", None) or out.get("encoder_variables", None)
            dec_w = out.get("decoder_variable_selection", None) or out.get("decoder_variables", None)

            # enc_w/dec_w biasanya shape: [B, T, F] atau [B, F]
            if enc_w is not None and torch.is_tensor(enc_w):
                w = enc_w
                while w.dim() > 2:
                    w = w.mean(dim=1)  # rata-rata time
                w = w.mean(dim=0)      # rata-rata batch -> [F]
                enc_sum = w if enc_sum is None else enc_sum + w
                enc_count += 1

            if dec_w is not None and torch.is_tensor(dec_w):
                w = dec_w
                while w.dim() > 2:
                    w = w.mean(dim=1)
                w = w.mean(dim=0)
                dec_sum = w if dec_sum is None else dec_sum + w
                dec_count += 1

    if enc_sum is None and dec_sum is None:
        raise ValueError("Tidak ketemu variable selection weights di output model. Coba cek keys output forward model.")

    rows = []

    if enc_sum is not None and enc_count > 0:
        enc_avg = (enc_sum / enc_count).detach().cpu().numpy()
        if enc_names is None:
            enc_names = [f"enc_{i}" for i in range(len(enc_avg))]
        for n, v in zip(enc_names, enc_avg):
            rows.append((n, float(v), "encoder"))

    if dec_sum is not None and dec_count > 0:
        dec_avg = (dec_sum / dec_count).detach().cpu().numpy()
        if dec_names is None:
            dec_names = [f"dec_{i}" for i in range(len(dec_avg))]
        for n, v in zip(dec_names, dec_avg):
            rows.append((n, float(v), "decoder"))

    imp = pd.DataFrame(rows, columns=["feature", "importance", "part"])

    # gabung encoder+decoder kalau nama sama
    imp = (
        imp.groupby("feature", as_index=False)["importance"]
           .mean()
           .sort_values("importance", ascending=False)
           .head(top_k)
           .reset_index(drop=True)
    )

    # normalize biar gampang dibaca (optional)
    s = imp["importance"].sum()
    if s > 0:
        imp["importance"] = imp["importance"] / s

    return imp

# =========================
# Streamlit UI (REFRESHED UI/UX)
# =========================
st.set_page_config(
    page_title="BBCA TFT Forecast",
    layout="wide",
    page_icon="📈",
)

# --- Optional: Plotly for better interactive charts ---
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ---------- Global Styles ----------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      header { visibility: hidden; height: 0; }
      footer { visibility: hidden; height: 0; }

      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 16px;
        background: rgba(255,255,255,0.03);
      }
      .muted { opacity: 0.75; font-size: 0.95rem; }

      .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.05);
        font-size: 0.85rem;
        margin-right: 6px;
      }

      [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;">
        <div>
          <h2 style="margin:0;">📈 BBCA TFT Forecast Dashboard</h2>
          <div class="muted">Realtime price (Yahoo) + Fundamentals + Technical Indicators → Forecast dengan Temporal Fusion Transformer</div>
          <div style="margin-top:10px;">
            <span class="badge">Realtime Yahoo</span>
            <span class="badge">Fundamental (Quarterly → Daily)</span>
            <span class="badge">TFT Model</span>
          </div>
        </div>
        <div class="muted" style="text-align:right;">
          <div>Target: <b>BBCA.JK</b></div>
          <div>Output: ret_log → Close_pred</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ===== session state init =====
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None
if "anchor_date" not in st.session_state:
    st.session_state.anchor_date = None
if "anchor_close" not in st.session_state:
    st.session_state.anchor_close = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None

# ---------- Load model + params ----------
try:
    model, params = load_model_and_params()
except Exception as e:
    st.error("Gagal load ckpt/params. Pastikan file ada: tft_model.ckpt & tft_dataset_params.pkl")
    st.exception(e)
    st.stop()

# ---------- Sidebar Controls (wrapped in form: better UX, less reruns) ----------
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    with st.form("controls_form", border=False):
        load_period = st.selectbox("Load history (untuk model)", ["2y", "5y", "10y"], index=2)

        today = pd.Timestamp.today().normalize()
        default_view_start = (today - pd.DateOffset(years=1)).date()
        default_view_end = today.date()

        view_range = st.date_input(
            "View range (yang ditampilkan)",
            value=(default_view_start, default_view_end),
        )

        forecast_from_view_end = st.checkbox("Forecast mulai dari end view range", value=True)
        horizon = st.slider("Forecast horizon (trading days)", 1, 20, 20)

        run = st.form_submit_button("🚀 Run Forecast", type="primary")

    st.caption("Tip: pakai view range 1 tahun biar chart enak dibaca. Forecast anchor bisa ikut end view range.")

# ---------- Load price ----------
try:
    price = load_bbca_price_from_yahoo(period=load_period, interval="1d")
except Exception as e:
    st.error("Gagal ambil BBCA.JK dari Yahoo endpoint.")
    st.exception(e)
    st.stop()

# ---------- Load fundamentals ----------
if not os.path.exists(FUND_PATH):
    st.error(f"Fundamentals CSV tidak ketemu: {FUND_PATH}")
    st.stop()

fund_q = load_fundamentals_quarterly(FUND_PATH)

# shift availability 7 hari (Date + 7d)
fund_q = fund_q.sort_values("Date").copy()
fund_q["Date"] = pd.to_datetime(fund_q["Date"]) + pd.Timedelta(days=7)

# align ke trading dates (price dates)
price_dates = pd.to_datetime(price["Date"]).sort_values()
fund_daily = (
    fund_q.set_index("Date")
          .sort_index()
          .reindex(price_dates, method="ffill")
          .bfill()
          .reset_index()
          .rename(columns={"index": "Date"})
)

# ---------- Build dataset ----------
df = build_df_tft(price, fund_daily)
df = align_df_to_dataset_params(df, params)

# ---------- View filter ----------
if isinstance(view_range, tuple) and len(view_range) == 2:
    view_start = pd.to_datetime(view_range[0])
    view_end = pd.to_datetime(view_range[1])
else:
    view_start = pd.to_datetime(default_view_start)
    view_end = pd.to_datetime(default_view_end)

min_d = df["Date"].min()
max_d = df["Date"].max()
vs = max(view_start, min_d)
ve = min(view_end, max_d)

df_view = df[(df["Date"] >= vs) & (df["Date"] <= ve)].copy()

if df_view.empty:
    st.warning("View range kosong (tidak ada data di range itu). Coba geser tanggalnya.")

# ===== pilih data untuk forecast (anchor) =====
if forecast_from_view_end:
    df_model = df[df["Date"] <= ve].copy()
    if df_model.empty:
        df_model = df.copy()
else:
    df_model = df.copy()

anchor_date = pd.to_datetime(df_model["Date"].iloc[-1]).date()
anchor_close = float(df_model["Close"].iloc[-1])

# ---------- Metrics row ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Anchor Date", str(anchor_date))
m2.metric("Anchor Close", f"{anchor_close:,.0f}")
m3.metric("Horizon", f"{horizon} hari")
m4.metric("History Loaded", load_period)

st.write("")

# ---------- Tabs ----------
tab_overview, tab_forecast, tab_importance, tab_data = st.tabs(
    ["📊 Overview", "🔮 Forecast", "🧠 Feature Importance", "🧾 Raw Data"]
)

# ---------- Overview ----------
with tab_overview:
    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.subheader("Historical Price (Close)")
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_view["Date"],
                y=df_view["Close"],
                mode="lines",
                name="Close"
            ))
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Date",
                yaxis_title="Close",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df_view.set_index("Date")["Close"], height=420)

    with c2:
        st.subheader("Snapshot")
        st.dataframe(df_view.tail(12), use_container_width=True, hide_index=True)

    with st.expander("🔧 Debug / Data Quality", expanded=False):
        if "CAR/KPMM (%)" in fund_daily.columns:
            st.write("CAR NA ratio (fund_daily):", float(fund_daily["CAR/KPMM (%)"].isna().mean()))
        else:
            st.info("Kolom 'CAR/KPMM (%)' tidak ditemukan di fund_daily.")

# ---------- Run Forecast action ----------
if run:
    with st.spinner("Running model inference…"):
        pred_len = params.get("max_prediction_length", 1)
        try:
            pred_len = int(pred_len)
        except Exception:
            pred_len = 1

        try:
            if pred_len >= horizon:
                df_all = make_future_rows(df_model, horizon=horizon)
                df_all = align_df_to_dataset_params(df_all, params)

                ret_pred = predict_direct(model, params, df_all)[:horizon]
                cur = anchor_close

                close_path = []
                for r in ret_pred:
                    cur = cur * float(np.exp(float(r)))
                    close_path.append(cur)

                future_dates = pd.bdate_range(
                    pd.to_datetime(df_model["Date"].iloc[-1]) + pd.Timedelta(days=1),
                    periods=horizon
                )
                pred_df = pd.DataFrame({"Date": future_dates, "ret_pred": ret_pred, "Close_pred": close_path})
            else:
                pred_df = recursive_forecast(model, params, df_model, horizon=horizon)

        except Exception as e:
            st.error("Forecast gagal.")
            st.exception(e)
            st.stop()

    # add % profit columns (keep your formatting)
    pred_df["% Untung/Rugi dari Historical Close_num"] = (pred_df["Close_pred"] / anchor_close - 1.0) * 100.0
    pred_df["% Naik/Turun vs Hari Sebelumnya_num"] = pred_df["Close_pred"].pct_change() * 100.0
    pred_df.loc[pred_df.index[0], "% Naik/Turun vs Hari Sebelumnya_num"] = pred_df["% Untung/Rugi dari Historical Close_num"].iloc[0]

    def fmt_pct(x):
        if pd.isna(x):
            return ""
        return f"{x:+.4f}%"

    pred_df["% Untung/Rugi dari Historical Close"] = pred_df["% Untung/Rugi dari Historical Close_num"].apply(fmt_pct)
    pred_df["% Naik/Turun vs Hari Sebelumnya"] = pred_df["% Naik/Turun vs Hari Sebelumnya_num"].apply(fmt_pct)

    # persist
    st.session_state.pred_df = pred_df.copy()
    st.session_state.anchor_date = anchor_date
    st.session_state.anchor_close = anchor_close
    st.session_state.feature_importance = None  # reset biar refresh sesuai data anchor terbaru

    st.success("Forecast selesai! Lihat tab Forecast & Feature Importance.")
    st.rerun()

# ---------- Forecast tab render (persisted) ----------
with tab_forecast:
    if st.session_state.pred_df is None:
        st.markdown(
            """
            <div class="card">
              <h4 style="margin:0;">Belum ada forecast</h4>
              <div class="muted" style="margin-top:6px;">Atur parameter di sidebar lalu klik <b>Run Forecast</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pred_df = st.session_state.pred_df.copy()
        anchor_close = float(st.session_state.anchor_close)

        st.subheader("Forecast Close (future)")
        if _HAS_PLOTLY:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=pred_df["Date"],
                y=pred_df["Close_pred"],
                mode="lines+markers",
                name="Close_pred"
            ))
            fig2.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Date",
                yaxis_title="Close_pred",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(pred_df.set_index("Date")["Close_pred"], height=420)

        st.subheader("Forecast Table")
        show = pred_df[["Date", "Close_pred", "% Untung/Rugi dari Historical Close", "% Naik/Turun vs Hari Sebelumnya"]].copy()
        show = show.reset_index(drop=True)
        show.index = show.index + 1

        def color_pct(val):
            if not isinstance(val, str) or val == "":
                return ""
            if val.startswith("+"):
                return "color: #16a34a; font-weight: 700;"
            if val.startswith("-"):
                return "color: #dc2626; font-weight: 700;"
            return ""

        styled = (
            show.style
            .format({"Close_pred": "{:,.0f}"})
            .applymap(color_pct, subset=["% Untung/Rugi dari Historical Close", "% Naik/Turun vs Hari Sebelumnya"])
        )

        st.dataframe(styled, use_container_width=True)

        st.download_button(
            "⬇️ Download forecast CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name=f"BBCA_forecast_{horizon}d.csv",
            mime="text/csv",
            key="download_forecast_persist",
        )

# ---------- Feature importance ----------
with tab_importance:
    if st.session_state.pred_df is None:
        st.info("Jalankan forecast dulu supaya feature importance relevan dengan window data terbaru.")
    else:
        st.subheader("Feature Importance")
        top_k = st.slider("Top features", 5, 50, 20, key="top_k_slider")

        if st.session_state.feature_importance is None:
            with st.spinner("Menghitung feature importance…"):
                try:
                    df_interp = df_model.tail(400).copy()
                    df_interp = align_df_to_dataset_params(df_interp, params)
                    fi = get_feature_importance_table(model, params, df_interp, top_k=50, n_batches=5)

                    fi["Persentase Perhatian Model (%)"] = (fi["importance"] * 100).round(2)
                    fi = fi.drop(columns=["importance"])
                    st.session_state.feature_importance = fi.copy()
                except Exception as e:
                    st.error("Gagal ambil feature importance.")
                    st.exception(e)

        if st.session_state.feature_importance is not None:
            fi_full = st.session_state.feature_importance.copy()
            fi_show = fi_full.head(top_k).reset_index(drop=True)
            fi_show.index = fi_show.index + 1
            st.dataframe(fi_show, use_container_width=True, hide_index=False)

# ---------- Raw data ----------
with tab_data:
    st.subheader("Latest rows (model dataset)")
    st.dataframe(df.tail(50), use_container_width=True, hide_index=True)
