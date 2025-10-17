# AI Sales Forecasting Dashboard

An interactive Streamlit app for exploring time-series sales data and generating forecasts using Prophet or ARIMA. Includes anomaly highlighting, aggregation (daily/weekly/monthly), model comparison, metrics, Plotly visuals, and downloadable forecasts and insights.

## Features
- Upload CSV with `Date` and `Sales` (optionally a `series`/`product` column)
- Aggregation: Daily, Weekly, Monthly
- Models: Prophet (optional) and ARIMA (pmdarima or statsmodels SARIMAX fallback)
- Backtest metrics: MAE, RMSE
- Anomaly detection (simple Z-score)
- Plotly charts, seasonality radar, residuals heatmap
- Download forecast CSV and insights text

## Tech Stack
- Streamlit, Plotly
- Prophet (optional), cmdstanpy (optional)
- pmdarima or statsmodels (ARIMA)
- pandas, numpy, scikit-learn

## Quickstart

### 1) Create/activate a Python environment (Windows)
```powershell
# Optional but recommended
py -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies
```powershell
py -m pip install -r requirements.txt
```

Notes:
- On Windows with Python 3.13, Prophet wheels may be unavailable. The app will still work using ARIMA via statsmodels. To enable Prophet:
```powershell
py -m pip install prophet cmdstanpy
# Then inside the app sidebar, you can install/update CmdStan if needed
```

### 3) Run the app
```powershell
py -m streamlit run "app.py" --server.port 8501 --server.headless true
```
Open http://localhost:8501 in your browser.

## CSV Format
- Minimum columns: `Date`, `Sales`
- Optional: `series`/`product`/`category` to model multiple series

Example:
```
Date,Sales,product
2024-01-01,120,A
2024-01-02,150,A
2024-01-01,90,B
```

## Development
- Main entry: `app.py`
- Cached helpers: `load_data()`, `build_prophet_model()`, ARIMA section with pmdarima or SARIMAX fallback
- Styling via custom CSS in `app.py`

## Deployment
- Any Streamlit-compatible host (Streamlit Community Cloud, etc.)
- Ensure required packages in `requirements.txt`
- For Prophet, include `prophet` and `cmdstanpy` and ensure the build toolchain is available

## License
MIT
