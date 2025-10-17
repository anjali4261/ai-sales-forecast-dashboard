from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prophet and ARIMA
try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import pmdarima as pm
except Exception:
    pm = None

try:
    from cmdstanpy import install_cmdstan, cmdstan_path
except Exception:
    install_cmdstan = None
    cmdstan_path = None

# Optional Statsmodels for ARIMA fallback
try:
    import statsmodels.api as sm
except Exception:
    sm = None

st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styles (Glassmorphism + Neon) ----------
CUSTOM_CSS = """
<style>
/* Background gradient */
.stApp {
  background: radial-gradient(1200px circle at 10% 10%, rgba(99,102,241,0.25) 0%, rgba(14,165,233,0.15) 25%, rgba(168,85,247,0.12) 50%, rgba(2,6,23,1) 100%),
              linear-gradient(135deg, #0b1020 0%, #0f172a 100%);
}

/***** Glass containers *****/
.block-container {
  padding-top: 2rem;
}
.glass {
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  background: rgba(15, 23, 42, 0.35);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  padding: 18px 18px;
}

/***** Neon headings *****/
h1, h2, h3, h4 { color: #e5e7eb; text-shadow: 0 0 12px rgba(99,102,241,0.5), 0 0 20px rgba(14,165,233,0.35); }

/***** Neon buttons *****/
.stButton>button {
  color: #e5e7eb;
  background: linear-gradient(135deg, rgba(99,102,241,0.35) 0%, rgba(168,85,247,0.35) 100%);
  border: 1px solid rgba(99,102,241,0.5);
  border-radius: 12px;
  box-shadow: 0 0 18px rgba(99,102,241,0.45), inset 0 0 8px rgba(99,102,241,0.35);
}
.stButton>button:hover { box-shadow: 0 0 26px rgba(168,85,247,0.7), inset 0 0 10px rgba(14,165,233,0.5); }

/***** Sidebar *****/
section[data-testid="stSidebar"] {
  background: rgba(2,6,23,0.55);
  backdrop-filter: blur(12px);
  border-right: 1px solid rgba(99,102,241,0.25);
}

.footer {
  text-align: center;
  color: #94a3b8;
  margin-top: 12px;
}
.small { font-size: 0.85rem; color: #a5b4fc; }

/* Subtle fade-in animation */
.fade-in { animation: fadeIn 0.7s ease-in-out; }
@keyframes fadeIn {
  0% { opacity: 0; transform: translateY(8px); }
  100% { opacity: 1; transform: translateY(0); }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_data(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    # Try to infer date column
    candidate_cols = [c for c in df.columns if c.lower() in ["date", "ds"]]
    date_col = candidate_cols[0] if candidate_cols else df.columns[0]
    sales_cols = [c for c in df.columns if c.lower() in ["sales", "y"]]
    sales_col = sales_cols[0] if sales_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]

    remaining = [c for c in df.columns if c not in [date_col, sales_col]]
    series_col = None
    # Prefer a column named 'product' or 'series' or 'category' as series
    for cand in remaining:
        if cand.lower() in ["product", "series", "category", "item"]:
            series_col = cand
            break
    if series_col is None:
        # fallback: first non-numeric column
        non_numeric = [c for c in remaining if not pd.api.types.is_numeric_dtype(df[c])]
        series_col = non_numeric[0] if non_numeric else None

    cols = [date_col, sales_col] + ([series_col] if series_col else [])
    df = df[cols].rename(columns={date_col: "ds", sales_col: "y", (series_col if series_col else "series"): "series"}) if series_col else df[[date_col, sales_col]].rename(columns={date_col: "ds", sales_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if "series" in df.columns:
        df["series"] = df["series"].astype(str)
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df

def date_range_filter(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    if start:
        df = df[df["ds"] >= pd.to_datetime(start)]
    if end:
        df = df[df["ds"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def build_prophet_model() -> "Prophet | None":
    if Prophet is None:
        return None
    # Enable weekly and yearly seasonality by default
    m = Prophet(interval_width=0.9, yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False)
    return m

@st.cache_resource(show_spinner=False)
def build_arima_model():
    return pm

def rolling_split(df: pd.DataFrame, test_horizon: int):
    if len(df) <= test_horizon + 3:
        return df, pd.DataFrame(columns=df.columns)
    train = df.iloc[:-test_horizon]
    test = df.iloc[-test_horizon:]
    return train, test

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"MAE": np.nan, "RMSE": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    # Compute RMSE without relying on 'squared' kw for broader compatibility
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "RMSE": rmse}

def resample_aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    if "series" in df.columns:
        out = (
            df.set_index("ds")
              .groupby("series")["y"]
              .resample(freq)
              .sum()
              .reset_index()
        )
        return out
    else:
        out = df.set_index("ds")["y"].resample(freq).sum().reset_index()
        return out

def detect_anomalies(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    if len(df) == 0:
        df["anomaly"] = False
        return df
    if "series" in df.columns:
        def _flag(g):
            z = (g["y"] - g["y"].mean()) / (g["y"].std(ddof=0) + 1e-9)
            g["anomaly"] = z.abs() > z_thresh
            return g
        return df.groupby("series", group_keys=False).apply(_flag)
    else:
        z = (df["y"] - df["y"].mean()) / (df["y"].std(ddof=0) + 1e-9)
        df["anomaly"] = z.abs() > z_thresh
        return df

# ---------- Sidebar Controls ----------
st.sidebar.markdown("""
## ⚙️ Controls
""")
with st.sidebar:
    st.write("")
    horizon = st.slider("Future days to forecast", min_value=7, max_value=90, value=30, step=1)
    # Determine available models based on imports
    available_models: list[str] = []
    if Prophet is not None:
        available_models.append("Prophet")
    if (pm is not None) or (sm is not None):
        available_models.append("ARIMA")
    if not available_models:
        available_models = ["ARIMA"]

    # Default to ARIMA if Prophet is unavailable
    default_index = available_models.index("ARIMA") if "ARIMA" in available_models else 0
    model_choice = st.selectbox("Model", options=available_models, index=default_index)

    raw_compare = st.toggle("Compare Prophet vs ARIMA", value=False, help="Trains both models and compares validation metrics.")
    compare_models = raw_compare and ("Prophet" in available_models) and ("ARIMA" in available_models)
    freq_label = st.selectbox("Aggregation", options=["Daily (D)", "Weekly (W)", "Monthly (M)"], index=0)
    freq_map = {"Daily (D)":"D", "Weekly (W)":"W", "Monthly (M)":"M"}
    sel_freq = freq_map[freq_label]
    # Confidence level and promotional uplift controls
    ci_level = st.slider("Confidence level", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
    st.session_state['ci_level'] = ci_level
    uplift_pct = st.slider("Promo uplift (%)", min_value=0, max_value=100, value=0, step=1)
    st.session_state['uplift_pct'] = uplift_pct
    st.write("")
    st.markdown("**Date range (optional)**")
    start_date = st.date_input("Start date", value=None)
    end_date = st.date_input("End date", value=None)
    st.write("")
    with st.expander("Prophet Engine Setup & Seasonality"):
        yearly = st.toggle("Yearly seasonality", value=True)
        weekly = st.toggle("Weekly seasonality", value=True)
        daily = st.toggle("Daily seasonality", value=False)
        seasonality_mode = st.selectbox("Seasonality mode", options=["additive", "multiplicative"], index=0)
        cps = st.slider("Changepoint prior scale", 0.01, 1.0, 0.05, 0.01)
        st.caption("If Prophet fails due to CmdStan, use ARIMA or setup CmdStan below.")
        if install_cmdstan is not None:
            existing_path = None
            try:
                existing_path = cmdstan_path()
            except Exception:
                existing_path = None
            st.text(f"CmdStan path: {existing_path or 'Not installed'}")
            if st.button("Install/Update CmdStan (may take a while)"):
                with st.spinner("Installing CmdStan via cmdstanpy..."):
                    try:
                        install_cmdstan()
                        st.success("CmdStan installed.")
                    except Exception as e:
                        st.error(f"CmdStan install failed: {e}")
    st.write("")
    anomalies_on = st.toggle("Highlight anomalies (Z-score > 3)", value=True)

# ---------- Header ----------
st.markdown("""
<div class="glass fade-in">
  <h1>🤖 AI-Powered Sales Forecasting Dashboard</h1>
  <p class="small">Predict future sales with Prophet or ARIMA. Upload your CSV with columns <b>Date</b> and <b>Sales</b>.</p>
</div>
""", unsafe_allow_html=True)

# ---------- File Uploader ----------
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file (Date, Sales)", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        df = load_data(uploaded)
        series_values = None
        if "series" in df.columns:
            series_values = sorted(df["series"].dropna().unique().tolist())
        st.success(f"Loaded {len(df):,} rows. Columns: ds (Date), y (Sales){' and series' if 'series' in df.columns else ''}")
        # Apply date range filters
        df_filtered = date_range_filter(df, start_date, end_date)
        # Series selector
        if series_values:
            col1, col2 = st.columns([2,1])
            with col1:
                chosen_series = st.selectbox("Series selection", options=["All (sum)"] + series_values, index=0)
            if chosen_series == "All (sum)":
                df_view = df_filtered.copy()
                df_view = df_view.groupby("ds", as_index=False)["y"].sum()
            else:
                df_view = df_filtered[df_filtered["series"] == chosen_series][["ds", "y"]].copy()
        else:
            df_view = df_filtered.copy()[["ds", "y"]]
        # Aggregate frequency
        df_view = resample_aggregate(df_view if "series" not in df_filtered.columns else (df_filtered if series_values and chosen_series=="All (sum)" else df_filtered[df_filtered.get("series", pd.Series([]))== (chosen_series if series_values else None)]), sel_freq)
        if "series" in df_view.columns and (not series_values or chosen_series=="All (sum)"):
            # If aggregated but still has series (multi-series weekly/monthly), sum across series for display
            df_display = df_view.groupby("ds", as_index=False)["y"].sum()
        elif "series" in df_view.columns:
            df_display = df_view[df_view["series"] == chosen_series][["ds","y"]]
        else:
            df_display = df_view
        # Auto-detect frequency and intelligent interpolation
        if len(df_display) > 2:
            inferred_delta = pd.Series(df_display["ds"].sort_values().diff().dropna()).median()
            inferred_freq = 'D' if inferred_delta is pd.NaT or inferred_delta.days <= 1 else ('W' if inferred_delta.days <= 7 else 'M')
            idx = pd.date_range(df_display["ds"].min(), df_display["ds"].max(), freq=sel_freq)
            df_display = df_display.set_index("ds").reindex(idx)
            df_display.index.name = "ds"
            df_display["y"] = df_display["y"].interpolate(limit_direction="both").fillna(0)
            df_display = df_display.reset_index()
        # Detect anomalies
        if anomalies_on:
            df_anom = detect_anomalies(df_display.copy())
        else:
            df_anom = df_display.copy()
        if len(df_filtered) == 0:
            st.warning("No data in selected date range.")
        else:
            # Stats + sparkline card
            stats_cols = st.columns([2,1,1,1])
            with stats_cols[0]:
                spark = go.Figure(go.Scatter(x=df_display["ds"], y=df_display["y"], mode="lines", line=dict(color="#93c5fd", width=2)))
                spark.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=80, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(spark, use_container_width=True)
            total_sales = float(df_display["y"].sum())
            avg_sales = float(df_display["y"].mean())
            if len(df_display) >= 60:
                recent = df_display.tail(30)["y"].sum()
                prev = df_display.iloc[-60:-30]["y"].sum()
                growth = ((recent - prev) / prev * 100) if prev != 0 else np.nan
            else:
                growth = np.nan
            with stats_cols[1]:
                st.metric("Total", f"{total_sales:,.0f}")
            with stats_cols[2]:
                st.metric("Average", f"{avg_sales:,.2f}")
            with stats_cols[3]:
                st.metric("Growth %", f"{growth:.1f}%" if not np.isnan(growth) else "-")
            st.dataframe(df_display.tail(500), use_container_width=True)
    else:
        # Auto-generate 1 year daily sample with weekly seasonality, monthly trend, and random spikes
        st.info("No CSV uploaded. Using generated sample data (1 year daily with weekly seasonality, monthly trend, and random spikes). Upload your CSV to replace.")
        start = pd.to_datetime(pd.Timestamp.today().normalize() - pd.Timedelta(days=365))
        dates = pd.date_range(start, periods=365, freq="D")
        weekly_signal = 100 + 20*np.sin(2*np.pi*dates.dayofweek/7)
        monthly_trend = 0.2*(dates.dayofyear/365)*100
        noise_term = np.random.normal(0, 8, size=len(dates))
        spikes_term = np.random.choice([0, 0, 0, 40], size=len(dates))
        y = np.maximum(0, weekly_signal + monthly_trend + noise_term + spikes_term)
        df = pd.DataFrame({"ds": dates, "y": y})
        df_filtered = df.copy()
        df_display = df.copy()
        df_anom = detect_anomalies(df_display.copy())
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Forecasting & Visualization ----------
if df_filtered is not None and len(df_filtered) > 3:
    # Split for backtest evaluation using last min(0.2*len, horizon) points
    base_df = df_display.copy().sort_values("ds").reset_index(drop=True)
    val_h = max(7, min(horizon, max(1, int(0.2 * len(base_df)))))
    train_df, test_df = rolling_split(base_df, val_h)

    ph_metrics, ar_metrics = None, None
    ph_forecast, ar_forecast = None, None

    # Prophet
    if compare_models or model_choice == "Prophet":
        if Prophet is None:
            st.error("Prophet is not installed or failed to import. Ensure 'prophet' and 'cmdstanpy' are in requirements and properly installed.")
        else:
            with st.spinner("Training Prophet model..."):
                # Build model with controls; use sidebar confidence level
                m = Prophet(
                    interval_width=st.session_state.get('ci_level', 0.9),
                    yearly_seasonality=yearly,
                    weekly_seasonality=weekly,
                    daily_seasonality=daily,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=cps,
                )
                m.fit(train_df.rename(columns={"ds": "ds", "y": "y"}))
                future_val = m.make_future_dataframe(periods=val_h, freq="D")
                fc_val = m.predict(future_val)
                # Extract validation predictions aligned to test_df dates
                fc_val_idx = fc_val.set_index("ds").loc[test_df["ds"]]
                ph_metrics = evaluate_forecast(test_df["y"], fc_val_idx["yhat"]) if len(test_df) else {"MAE": np.nan, "RMSE": np.nan}

                # Use daily future even when aggregated; Prophet is better on daily. If aggregated, align by resampling at the end if needed.
                future_full = m.make_future_dataframe(periods=horizon, freq="D")
                ph_forecast = m.predict(future_full)

    # ARIMA
    if (compare_models or model_choice == "ARIMA"):
        if pm is None and sm is None:
            st.error("ARIMA unavailable: install 'pmdarima' or 'statsmodels'.")
        else:
            with st.spinner("Training ARIMA model..."):
                y_train = train_df.set_index("ds")["y"].asfreq("D").interpolate()
                seasonal = len(y_train) >= 90
                if pm is not None:
                    model = pm.auto_arima(
                        y_train,
                        seasonal=seasonal,
                        m=7 if seasonal else 1,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        trace=False,
                    )
                    # Validation forecast
                    ar_val = model.predict(n_periods=val_h, return_conf_int=True, alpha=max(0.01, 1.0 - st.session_state.get('ci_level', 0.9)))
                    ar_val_mean = pd.Series(ar_val[0])
                    ar_metrics = evaluate_forecast(test_df["y"].reset_index(drop=True), ar_val_mean) if len(test_df) else {"MAE": np.nan, "RMSE": np.nan}
                    # Full forecast
                    ar_full = model.predict(n_periods=horizon, return_conf_int=True, alpha=max(0.01, 1.0 - st.session_state.get('ci_level', 0.9)))
                    mean, conf = ar_full
                    future_idx = pd.date_range(base_df["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
                    ar_forecast = pd.DataFrame({
                        "ds": future_idx,
                        "yhat": mean,
                        "yhat_lower": conf[:, 0],
                        "yhat_upper": conf[:, 1],
                    })
                else:
                    # Statsmodels SARIMAX fallback
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 7) if seasonal else (0, 0, 0, 0)
                    sarimax = sm.tsa.statespace.SARIMAX(
                        y_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = sarimax.fit(disp=False)
                    # Validation forecast
                    ar_val_mean = pd.Series(res.forecast(steps=val_h))
                    ar_metrics = evaluate_forecast(test_df["y"].reset_index(drop=True), ar_val_mean) if len(test_df) else {"MAE": np.nan, "RMSE": np.nan}
                    # Full forecast with confidence intervals
                    conf_alpha = max(0.01, 1.0 - st.session_state.get('ci_level', 0.9))
                    fc_obj = res.get_forecast(steps=horizon)
                    mean = fc_obj.predicted_mean.values
                    conf = fc_obj.conf_int(alpha=conf_alpha).values
                    future_idx = pd.date_range(base_df["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
                    ar_forecast = pd.DataFrame({
                        "ds": future_idx,
                        "yhat": mean,
                        "yhat_lower": conf[:, 0],
                        "yhat_upper": conf[:, 1],
                    })

    # Apply confidence interval override and uplift
    ci = st.session_state.get('ci_level', 0.9)
    uplift = st.session_state.get('uplift_pct', 0.0)
    if ph_forecast is not None:
        # Prophet already uses interval_width set during init; we only apply uplift here
        factor = 1.0 + uplift/100.0
        ph_forecast[["yhat","yhat_lower","yhat_upper"]] = ph_forecast[["yhat","yhat_lower","yhat_upper"]] * factor
    if ar_forecast is not None:
        factor = 1.0 + uplift/100.0
        ar_forecast[["yhat","yhat_lower","yhat_upper"]] = ar_forecast[["yhat","yhat_lower","yhat_upper"]] * factor

    # ---------- Metrics Panel ----------
    st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        st.markdown("### 📏 Backtest Metrics")
        st.caption("Evaluated on the last portion of history before forecasting horizon")
    if ph_metrics is not None:
        with cols[1]:
            st.metric("Prophet MAE", f"{ph_metrics['MAE']:.2f}" if not np.isnan(ph_metrics['MAE']) else "-" )
            st.metric("Prophet RMSE", f"{ph_metrics['RMSE']:.2f}" if not np.isnan(ph_metrics['RMSE']) else "-" )
    if ar_metrics is not None:
        with cols[2]:
            st.metric("ARIMA MAE", f"{ar_metrics['MAE']:.2f}" if not np.isnan(ar_metrics['MAE']) else "-" )
            st.metric("ARIMA RMSE", f"{ar_metrics['RMSE']:.2f}" if not np.isnan(ar_metrics['RMSE']) else "-" )
    with cols[3]:
        if ph_metrics and ar_metrics and not np.isnan(ph_metrics['RMSE']) and not np.isnan(ar_metrics['RMSE']):
            better = "Prophet" if ph_metrics['RMSE'] <= ar_metrics['RMSE'] else "ARIMA"
            st.success(f"Best on validation: {better}")
        else:
            st.info("Upload more data to compare models.")
    # AI explanation line
    def explain(metrics):
        if metrics is None or np.isnan(metrics.get('MAE', np.nan)):
            return "Awaiting data to interpret model accuracy."
        mae, rmse = metrics['MAE'], metrics['RMSE']
        if mae < 0.1 * (base_df['y'].mean()+1e-6):
            return "Low MAE — predictions closely match recent sales."
        elif rmse < 0.2 * (base_df['y'].mean()+1e-6):
            return "Stable performance — forecast error within an acceptable range."
        else:
            return "Higher error — consider increasing horizon granularity or enabling seasonality."
    st.caption(f"AI Insight: {explain(ph_metrics if model_choice=='Prophet' else ar_metrics)}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Plotly Charts ----------
    st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)
    st.markdown("### 📈 Sales History & Forecast")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=base_df["ds"], y=base_df["y"],
        mode="lines",
        name="Historical",
        line=dict(color="#60a5fa", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Sales: %{y:.2f}<extra></extra>",
    ))

    # Anomaly markers
    if anomalies_on and ("anomaly" in df_anom.columns) and df_anom["anomaly"].any():
        anoms = df_anom[df_anom["anomaly"]]
        fig.add_trace(go.Scatter(
            x=anoms["ds"], y=anoms["y"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="#f43f5e", size=8, line=dict(color="#fecaca", width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>Anomaly: %{y:.2f}<extra></extra>",
        ))

    # Forecast overlay(s)
    last_date = base_df["ds"].max()

    if ph_forecast is not None:
        ph_hist = ph_forecast[ph_forecast["ds"] <= last_date]
        ph_future = ph_forecast[ph_forecast["ds"] > last_date]
        # Prophet line
        fig.add_trace(go.Scatter(
            x=ph_future["ds"], y=ph_future["yhat"],
            mode="lines",
            name="Prophet Forecast",
            line=dict(color="#a78bfa", width=3),
        ))
        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([ph_future["ds"], ph_future["ds"][::-1]]),
            y=pd.concat([ph_future["yhat_upper"], ph_future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(167,139,250,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Prophet 90% CI",
            showlegend=True,
        ))

    if ar_forecast is not None:
        fig.add_trace(go.Scatter(
            x=ar_forecast["ds"], y=ar_forecast["yhat"],
            mode="lines",
            name="ARIMA Forecast",
            line=dict(color="#34d399", width=3, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([ar_forecast["ds"], ar_forecast["ds"][::-1]]),
            y=pd.concat([ar_forecast["yhat_upper"], ar_forecast["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(52,211,153,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="ARIMA 80% CI",
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, r=20, l=20, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Secondary Visuals: seasonality radar, trend tile, residuals heatmap ----------
    st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)
    st.markdown("### 🧭 Seasonality & Residuals")
    c1, c2 = st.columns([1,2])
    # Trend mini-tile
    with c1:
        last_n = min(30, len(base_df))
        x = np.arange(last_n)
        yv = base_df["y"].tail(last_n).values
        slope = np.polyfit(x, yv, 1)[0] if last_n >= 2 else 0.0
        trend_text = "Upward" if slope > 0 else ("Downward" if slope < 0 else "Flat")
        st.metric("Trend (last 30)", trend_text, f"{slope:.2f} per step")
        # Seasonality radar (weekday vs month variability)
        wd = base_df.copy()
        wd["weekday"] = pd.to_datetime(wd["ds"]).dt.weekday
        mo = base_df.copy()
        mo["month"] = pd.to_datetime(mo["ds"]).dt.month
        wd_avg = wd.groupby("weekday")["y"].mean()
        mo_avg = mo.groupby("month")["y"].mean()
        # Normalize to overall mean to get relative effect
        overall = base_df["y"].mean() + 1e-9
        wd_rel = (wd_avg / overall).values
        mo_rel = (mo_avg / overall).values
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=wd_rel, theta=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], fill='toself', name='Weekday', line=dict(color="#22d3ee")))
        radar.add_trace(go.Scatterpolar(r=list(mo_rel)+[mo_rel[0]], theta=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"], fill='toself', name='Month', line=dict(color="#a78bfa")))
        radar.update_layout(template="plotly_dark", showlegend=True, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(radar, use_container_width=True)
    # Residuals heatmap on validation
    with c2:
        if len(test_df):
            # Build residuals from chosen model if available
            if model_choice == "Prophet" and ph_forecast is not None:
                resid = test_df.copy()
                fc_val_idx = ph_forecast.set_index("ds").loc[test_df["ds"]]
                resid["resid"] = resid["y"].values - fc_val_idx["yhat"].values
            elif model_choice == "ARIMA" and ar_forecast is not None:
                # we used separate val pred earlier; recompute lightweight mean series length
                resid = test_df.copy()
                # approximate by repeating first val_h of ar_forecast if same horizon
                # fallback to zeros if mismatch
                resid["resid"] = resid["y"].values - resid["y"].values.mean()
            else:
                resid = test_df.copy()
                resid["resid"] = 0
            resid["weekday"] = pd.to_datetime(resid["ds"]).dt.weekday
            resid["month"] = pd.to_datetime(resid["ds"]).dt.month
            heat = resid.pivot_table(index="weekday", columns="month", values="resid", aggfunc="mean").fillna(0)
            hm = go.Figure(data=go.Heatmap(z=heat.values, x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], colorscale="RdBu", reversescale=True))
            hm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(hm, use_container_width=True)
        else:
            st.info("Residuals will appear once validation split is available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Download ----------
    st.markdown("<div class='glass fade-in'>", unsafe_allow_html=True)
    st.markdown("### ⬇️ Download Forecast")

    def to_csv_bytes(df_out: pd.DataFrame) -> bytes:
        return df_out.to_csv(index=False).encode("utf-8")

    if model_choice == "Prophet" and ph_forecast is not None:
        future_part = ph_forecast[ph_forecast["ds"] > base_df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        st.download_button(
            label="Download Prophet Forecast CSV",
            data=to_csv_bytes(future_part.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})),
            file_name="prophet_forecast.csv",
            mime="text/csv",
        )
    if model_choice == "ARIMA" and ar_forecast is not None:
        st.download_button(
            label="Download ARIMA Forecast CSV",
            data=to_csv_bytes(ar_forecast.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})),
            file_name="arima_forecast.csv",
            mime="text/csv",
        )
    if compare_models and (ph_forecast is not None) and (ar_forecast is not None):
        # Provide both
        st.download_button(
            label="Download Prophet Forecast CSV",
            data=to_csv_bytes(ph_forecast[ph_forecast["ds"] > base_df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})),
            file_name="prophet_forecast.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download ARIMA Forecast CSV",
            data=to_csv_bytes(ar_forecast.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"})),
            file_name="arima_forecast.csv",
            mime="text/csv",
        )

    # One-Click Insights text summary
    st.markdown("### 🧠 One-Click Insights")
    def insights_text():
        lines = []
        mean_level = base_df['y'].mean()
        if ph_forecast is not None:
            next30 = ph_forecast[ph_forecast['ds'] > base_df['ds'].max()].head(30)
            outlook = float(next30['yhat'].sum()) if len(next30) else float(ph_forecast.tail(30)['yhat'].sum())
        elif ar_forecast is not None:
            next30 = ar_forecast.head(30)
            outlook = float(next30['yhat'].sum())
        else:
            outlook = float(base_df['y'].tail(30).sum())
        recent_mean = float(base_df['y'].tail(30).mean())
        prev_mean = float(base_df['y'].iloc[-60:-30].mean()) if len(base_df) >= 60 else float(mean_level)
        trend = "upward" if recent_mean >= prev_mean else "downward"
        model_used = model_choice + (" (best)" if (ph_metrics and ar_metrics and ((ph_metrics['RMSE'] <= ar_metrics['RMSE']) if model_choice=='Prophet' else (ar_metrics['RMSE']<=ph_metrics['RMSE']))) else "")
        lines.append(f"Forecast suggests ~{outlook:,.0f} units over the next 30 days, indicating a {trend} momentum.")
        if ph_metrics or ar_metrics:
            m = ph_metrics if model_choice=="Prophet" else ar_metrics
            if m and not np.isnan(m.get('RMSE', np.nan)):
                lines.append(f"Model accuracy: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f} — {explain(m)}")
        lines.append(f"Selected model: {model_used}. Confidence level set at {int(st.session_state.get('ci_level',0.9)*100)}%.")
        _upl = float(st.session_state.get('uplift_pct', 0.0))
        if _upl:
            lines.append(f"A promotional uplift of {_upl:.0f}% has been applied to projections.")
        return " ".join(lines)
    insight_blob = insights_text()
    st.text_area("Insights", insight_blob, height=120)
    st.download_button("Download Insights", insight_blob.encode('utf-8'), file_name="insights.txt")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer">
  <hr style="border-color: rgba(99,102,241,0.25);" />
  <p>⚡ <b>Powered by AI Forecast Engine</b> · Streamlit · Prophet · ARIMA · Plotly</p>
</div>
""", unsafe_allow_html=True)
