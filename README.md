# 📈 Time Series Forecasting: AAPL Stock Price

## Project Overview
End-to-end time series forecasting project on 10 years of Apple (AAPL) stock data (2015–2024). The goal was to model historical price behavior, understand its trend and seasonal structure, and compare two different forecasting approaches — ARIMA and Facebook Prophet — using a proper train/test evaluation rather than in-sample fitting alone.

This is **Project 3 of 10** in a 4-month Data Analysis & Machine Learning roadmap.

---

## 🛠️ Tools & Technologies
- **Python** — core analysis language
- **yfinance** — historical stock data source
- **statsmodels** — seasonal decomposition, ADF stationarity testing, ARIMA
- **Prophet** — trend + seasonality forecasting
- **Matplotlib** — visualizations
- **Streamlit** — interactive forecasting dashboard
- **Jupyter Lab** — development environment

---

## 📁 Project Structure
```
stock-forecasting/
├── notebooks/
│   └── 01_stock_forecasting.ipynb   ← full analysis notebook
├── app.py                           ← Streamlit dashboard
└── README.md
```

---

## 📥 Dataset
- **Source:** Yahoo Finance, via the `yfinance` Python library
- **Ticker:** AAPL (Apple Inc.)
- **Period:** January 2015 – December 2024 (10 years)
- **Frequency:** Daily data, resampled to monthly averages for modeling
- **Size:** 2,516 daily trading rows → 120 monthly data points

---

## 🔍 Methodology

### 1. Data Collection & Cleaning
Downloaded 10 years of daily AAPL price data. `yfinance` returns multi-level columns for single-ticker downloads, which were flattened to simple column names (`Close`, `High`, `Low`, `Open`, `Volume`). Verified no missing values and confirmed the date index spanned the expected range (2,516 trading days ≈ 252/year, consistent with market holidays and weekends).

### 2. Resampling to Monthly
Daily price data is noisy; resampling to monthly averages (`resample('M').mean()`) made the underlying trend and seasonal pattern easier to detect, both visually and statistically, without losing the yearly structure we were interested in.

### 3. Seasonal Decomposition
Used `seasonal_decompose()` with a **multiplicative** model (`Observed = Trend × Seasonal × Residual`), rather than additive. This choice was deliberate: AAPL's price grew roughly 10x over the period (~$25 → ~$250), so seasonal swings scale proportionally with price level rather than staying a fixed dollar amount — a multiplicative model handles that correctly, an additive one would not.

**Finding:** A clear, steadily accelerating upward trend, and a strong repeating yearly seasonal pattern peaking in Q4 — consistent with Apple's September iPhone launches and Q4 holiday sales quarter.

### 4. Stationarity Testing (ADF Test)
ARIMA requires stationary input. An Augmented Dickey-Fuller test on the raw monthly series returned a p-value of **0.9975**, confirming strong non-stationarity (consistent with the clear upward trend). First-order differencing (modeling month-over-month *change* instead of raw price level) brought the p-value down to **0.0127** — confirming stationarity was achieved with `d=1`.

### 5. ARIMA Parameter Selection
ACF and PACF plots (24 lags) were used to look for direct/indirect autocorrelation in the differenced series to inform the `p` and `q` parameters. Both plots showed minimal significant autocorrelation beyond lag 0, suggesting weak self-predictive structure in monthly price changes. To make an evidence-based choice, five candidate models — `(0,1,0)`, `(1,1,0)`, `(0,1,1)`, `(1,1,1)`, `(2,1,2)` — were compared by AIC. `(0,1,1)` had the lowest AIC (799.7), though all candidates clustered closely together, reinforcing the limited structure found in the diagnostic plots.

### 6. Model Evaluation (Train/Test Split)
The last 12 months (2024) were held out as a test set, with the model fit only on prior data (2015–2023), to fairly evaluate forecast accuracy against real, unseen values rather than in-sample fit.

### 7. Prophet Modeling
Facebook Prophet was fit on the same training data, using its default automatic trend and yearly-seasonality detection — a different modeling philosophy from ARIMA's manual parameter search. The same train/test split and evaluation metrics were used for a fair, apples-to-apples comparison.

### 8. Interactive Dashboard (Streamlit)
Built a Streamlit app allowing live switching between ARIMA and Prophet forecasts, with an adjustable forecast horizon (3–24 months) and Prophet's confidence intervals visualized as a shaded band.

---

## 📊 Key Findings

| Metric | ARIMA (0,1,1) | Prophet |
|---|---|---|
| MAE | $24.37 | **$15.76** |
| RMSE | $27.57 | **$17.85** |

**Prophet outperformed ARIMA by ~35% on both MAE and RMSE.**

**Why:** ARIMA's forecast came out nearly flat, essentially repeating the last known price forward. This reflects a genuine, well-documented limitation — monthly price *changes* showed minimal autocorrelation (confirmed by both the ACF/PACF plots and the tightly-clustered AIC comparison), consistent with efficient-market behavior. With `p=0`, ARIMA had no mechanism to project a longer-term directional trend, and its one MA correction term faded out almost immediately across a 12-month horizon.

Prophet, by contrast, explicitly models trend and yearly seasonality as separate components — and its automatically-detected yearly pattern closely matched the real Q4 seasonal spike visible in the decomposition step, giving it a structural advantage on data where seasonality is real and consistent, even though short-term price changes alone are hard to predict.

**Honest limitation:** ARIMA's residuals showed significant heteroskedasticity (non-constant variance over time) — a further sign that a simple ARIMA model isn't fully capturing this series' behavior, particularly as volatility increased in later years of the dataset.

---

## ⚠️ Disclaimer

This project is a modeling and forecasting exercise for learning purposes, not financial advice. Historical seasonal patterns are not guarantees of future performance, and this analysis does not account for news, earnings, or macroeconomic events that can dominate short-term price movement.

---

## 🖥️ Streamlit Dashboard
Run locally with:
```bash
streamlit run app.py
```
Features: live ARIMA/Prophet model switching, adjustable forecast horizon, historical price chart, and Prophet confidence intervals.

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/ahmedhussien96/stock-forecasting

# Install dependencies
pip install yfinance pandas numpy matplotlib statsmodels prophet scikit-learn streamlit jupyterlab

# Launch Jupyter
cd stock-forecasting
jupyter lab
```
Then open `notebooks/01_stock_forecasting.ipynb` and run all cells top to bottom.

---

## 👤 Author
Ahmed Hussien — Data Analysis & ML Roadmap, Project 3 of 10
