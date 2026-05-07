import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Stock Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interactive Stock Analysis — Technical Analysis Module")

# --- Sidebar Controls ---
st.sidebar.header("Analysis Controls")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

today = date.today()
start_date = st.sidebar.date_input(
    'Start Date',
    today - timedelta(days=365),
    max_value=today - timedelta(days=1)
)
end_date = st.sidebar.date_input(
    'End Date',
    today,
    min_value=start_date + timedelta(days=1),
    max_value=today
)

# Moving averages (kept from previous)
st.sidebar.subheader("Moving Average Settings")
ma1_window = st.sidebar.number_input("Short-term MA (days)", min_value=5, max_value=100, value=20)
ma2_window = st.sidebar.number_input("Long-term MA (days)", min_value=10, max_value=200, value=50)

# TA toggles
st.sidebar.subheader("Technical Indicators")
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

# Other options
st.sidebar.subheader("Extras")
show_prediction = st.sidebar.checkbox("Show Simple AI Prediction", value=True)
download_data = st.sidebar.checkbox("Enable CSV Download", value=True)


# --- Data Loading ---
@st.cache_data(ttl=60*60)  # cache for 1 hour
def load_data(ticker_symbol, start, end):
    data = yf.download(ticker_symbol, start=start, end=end)
    if not data.empty:
        # flatten MultiIndex if needed and lowercase column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [col.lower() for col in data.columns]
    return data

raw_data = load_data(ticker, start_date, end_date)

if raw_data.empty:
    st.error(f"❌ No data found for ticker '{ticker}' within selected date range.")
    st.stop()

# --- Indicator Computations ---
@st.cache_data
def compute_indicators(df, ma1=20, ma2=50):
    df = df.copy()
    # Moving averages
    df[f'ma{ma1}'] = df['close'].rolling(window=ma1, min_periods=1).mean()
    df[f'ma{ma2}'] = df['close'].rolling(window=ma2, min_periods=1).mean()

    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mult = 2
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_mult * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - bb_mult * df['bb_std']

    # RSI (14)
    rsi_period = 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    # Use Wilder's smoothing (EMA-like)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    # For first values where rolling gives NaN, fallback to EMA-like calculation
    avg_gain = avg_gain.fillna(gain.ewm(alpha=1/rsi_period, adjust=False).mean())
    avg_loss = avg_loss.fillna(loss.ewm(alpha=1/rsi_period, adjust=False).mean())
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # neutral for early rows

    # MACD (12,26,9)
    ema_short = df['close'].ewm(span=12, adjust=False).mean()
    ema_long = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_short - ema_long
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df

data = compute_indicators(raw_data, ma1=ma1_window, ma2=ma2_window)

# --- Plotting: Price + Indicators + Volume ---
st.subheader(f"📈 {ticker} Price Chart with Technical Indicators")

# Build subplots: rows vary depending on what user toggles
# We'll always include the price row. Add RSI and MACD rows if toggled.
rows = 1  # price
if show_volume:
    rows += 1
if show_rsi:
    rows += 1
if show_macd:
    rows += 1

# Create shared x-axis subplots
specs = []
for i in range(rows):
    if i == 0:
        specs.append([{"secondary_y": True}])  # price row will have secondary y for moving averages (not necessary but safe)
    else:
        specs.append([{}])

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.5] + [0.15]*(rows-1),
                    specs=specs)

row_idx = 1

# 1) Price Candlestick
fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=True
    ),
    row=row_idx, col=1
)

# Bollinger Bands (overlay)
if show_bbands:
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], name='BB Upper', line=dict(width=1), opacity=0.6), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_mid'], name='BB Mid', line=dict(width=1), opacity=0.6), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], name='BB Lower', line=dict(width=1), opacity=0.6, fill='tonexty'), row=row_idx, col=1)

# Moving averages overlay
fig.add_trace(go.Scatter(x=data.index, y=data[f'ma{ma1_window}'], name=f'MA {ma1_window}', line=dict(width=1.5), opacity=0.9), row=row_idx, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data[f'ma{ma2_window}'], name=f'MA {ma2_window}', line=dict(width=1.5), opacity=0.9), row=row_idx, col=1)

row_idx += 1

# 2) Volume (if enabled)
if show_volume:
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', showlegend=True), row=row_idx, col=1)
    row_idx += 1

# 3) RSI
if show_rsi:
    fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name='RSI (14)', line=dict(width=1.2)), row=row_idx, col=1)
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", row=row_idx, col=1, annotation_text="Overbought (70)", annotation_position="top left")
    fig.add_hline(y=30, line_dash="dash", row=row_idx, col=1, annotation_text="Oversold (30)", annotation_position="bottom left")
    row_idx += 1

# 4) MACD
if show_macd:
    fig.add_trace(go.Bar(x=data.index, y=data['macd_hist'], name='MACD Hist', showlegend=True), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(width=1.2)), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='Signal', line=dict(width=1.0, dash='dot')), row=row_idx, col=1)
    row_idx += 1

fig.update_layout(
    height=700,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=30, r=30, t=60, b=30)
)

fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends

st.plotly_chart(fig, use_container_width=True)

# --- Recent Data & Prediction (keeps existing RF model) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Recent Market Data (last 10 rows)")
    st.dataframe(data.tail(10))

    if download_data:
        csv = data.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Download Data as CSV", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")

with col2:
    if show_prediction:
        st.subheader("🤖 Simple AI Prediction (RandomForest)")
        df_ml = data.copy().reset_index()
        df_ml['day'] = range(len(df_ml))
        X = df_ml[['day']]
        y = df_ml['close']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        next_day = pd.DataFrame({'day': [len(df_ml)]})
        predicted_price = model.predict(next_day)[0]
        last_close_price = data['close'].iloc[-1]

        st.metric(
            label="Forecasted Next Close Price",
            value=f"${predicted_price:.2f}",
            delta=f"${predicted_price - last_close_price:.2f}"
        )
    else:
        st.info("AI prediction is turned off. Enable 'Show Simple AI Prediction' in the sidebar to view it.")

# --- Small notes / tips ---
st.markdown("""
**TA Notes**
- RSI > 70 typically indicates overbought; < 30 indicates oversold (use with other signals).
- MACD crossing above its signal line is bullish; crossing below is bearish.
- Price above upper Bollinger Band may indicate strength / short-term overextension; below lower band may indicate weakness.
- Indicators should be combined; none are perfect on their own.
""")
