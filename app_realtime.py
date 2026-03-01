import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NAVIENTA Realtime (Simulated)", layout="wide")
st.title("NAVIENTA — Real-time Risk Dashboard (Simulated)")

CSV_PATH_DEFAULT = "outputs/navienta_demo_with_pred.csv"

# -------------------- Sidebar controls
with st.sidebar:
    st.header("Controls")
    csv_path = st.text_input("CSV path", value=CSV_PATH_DEFAULT)
    refresh_sec = st.slider("Refresh interval (seconds)", 0.1, 2.0, 0.5, 0.1)
    step = st.slider("Rows per tick", 1, 200, 50, 5)
    max_points = st.slider("Max points on chart", 200, 10000, 1500, 100)
    smooth_window = st.slider("Smoothing window (rows)", 1, 200, 10, 1)
    start_btn = st.toggle("Start simulation", value=False)

# -------------------- Helpers
risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
inv_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

def smooth_risk(labels: pd.Series, window: int) -> pd.Series:
    s = labels.map(risk_map).astype(float)
    sm = s.rolling(window, min_periods=1).median().round().astype(int)
    return sm.map(inv_map)

def make_segments(df: pd.DataFrame, pred_col: str, t_col: str, min_d_col: str | None = None, max_close_col: str | None = None) -> pd.DataFrame:
    is_high = (df[pred_col] == "HIGH").astype(int)
    seg_id = (is_high.diff().fillna(0) == 1).cumsum()
    tmp = df.copy()
    tmp["seg_id"] = seg_id.where(is_high == 1, 0)

    d = tmp[tmp["seg_id"] != 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["start_s", "end_s", "duration_s", "min_d_m", "max_close"])

    agg = {
        "start_s": (t_col, "min"),
        "end_s": (t_col, "max"),
        "duration_s": (t_col, lambda s: float(s.max() - s.min())),
    }
    if min_d_col and min_d_col in d.columns:
        agg["min_d_m"] = (min_d_col, "min")
    else:
        agg["min_d_m"] = (t_col, lambda s: np.nan)

    if max_close_col and max_close_col in d.columns:
        agg["max_close"] = (max_close_col, "max")
    else:
        agg["max_close"] = (t_col, lambda s: np.nan)

    seg = d.groupby("seg_id").agg(**agg).reset_index(drop=True)
    return seg.sort_values("duration_s", ascending=False)

# -------------------- Load once (cached)
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

try:
    df_all = load_csv(csv_path)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# Required columns
required = ["timestamp", "d_front_m"]
missing = [c for c in required if c not in df_all.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Available columns:", list(df_all.columns))
    st.stop()

pred_col = "pred_class" if "pred_class" in df_all.columns else ("risk_class" if "risk_class" in df_all.columns else None)
if pred_col is None:
    st.error("CSV must contain `pred_class` or `risk_class`.")
    st.write("Available columns:", list(df_all.columns))
    st.stop()

min_d_col = "min_d_front_1s_clip" if "min_d_front_1s_clip" in df_all.columns else ("min_d_front_1s" if "min_d_front_1s" in df_all.columns else None)
max_close_col = "max_close_1s" if "max_close_1s" in df_all.columns else None

# -------------------- State
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame()

# -------------------- Simulation tick
if start_btn:
    i = st.session_state.idx
    j = min(i + step, len(df_all))
    chunk = df_all.iloc[i:j].copy()
    st.session_state.idx = 0 if j >= len(df_all) else j

    buf = st.session_state.buffer
    buf = pd.concat([buf, chunk], ignore_index=True)

    # keep last max_points
    if len(buf) > max_points:
        buf = buf.iloc[-max_points:].reset_index(drop=True)

    st.session_state.buffer = buf

buf = st.session_state.buffer
if buf.empty:
    st.info("Toggle **Start simulation** on the left to begin streaming.")
    st.stop()

# Relative time for display
buf = buf.sort_values("timestamp").reset_index(drop=True)
buf["t_rel"] = buf["timestamp"] - buf["timestamp"].min()

# Clip distance for sanity (optional, avoids absurd values in demo)
buf["d_front_m_clip"] = buf["d_front_m"].clip(lower=0.5, upper=200)
if min_d_col:
    buf["min_d_front_clip"] = buf[min_d_col].clip(lower=0.5, upper=200)

# Smoothed risk
buf["risk_smooth"] = smooth_risk(buf[pred_col], smooth_window)

# -------------------- KPIs
c1, c2, c3 = st.columns(3)
c1.metric("LOW (%)", f"{(buf['risk_smooth'].eq('LOW').mean()*100):.1f}%")
c2.metric("MEDIUM (%)", f"{(buf['risk_smooth'].eq('MEDIUM').mean()*100):.1f}%")
c3.metric("HIGH (%)", f"{(buf['risk_smooth'].eq('HIGH').mean()*100):.1f}%")

# -------------------- Charts
left, right = st.columns(2)

with left:
    st.subheader("Front distance (m)")
    fig = plt.figure()
    plt.plot(buf["t_rel"], buf["d_front_m_clip"])
    plt.xlabel("Time (s) from start")
    plt.ylabel("Distance (m)")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Risk timeline (smoothed)")
    fig = plt.figure()
    plt.plot(buf["t_rel"], buf["risk_smooth"].map(risk_map))
    plt.yticks([0, 1, 2], ["LOW", "MEDIUM", "HIGH"])
    plt.xlabel("Time (s) from start")
    plt.ylabel("Risk level")
    st.pyplot(fig, clear_figure=True)

# -------------------- High segments table
st.subheader("High-risk segments (from smoothed risk)")
segments = make_segments(
    buf,
    pred_col="risk_smooth",
    t_col="t_rel",
    min_d_col="min_d_front_clip" if ("min_d_front_clip" in buf.columns) else None,
    max_close_col=max_close_col
)
st.write(f"Total HIGH segments: {len(segments)}")
st.dataframe(segments.head(30), use_container_width=True)

with st.expander("Preview latest rows"):
    st.dataframe(buf.tail(50), use_container_width=True)

# Auto-refresh only when running
if start_btn:
    time.sleep(refresh_sec)
    st.rerun()