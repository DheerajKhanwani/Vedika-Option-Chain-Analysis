import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go
from dateutil import parser as dateparser

# --------------------------- CONFIG ---------------------------
st.set_page_config(page_title="Option Chain Dashboard", layout="wide")
st.title("\U0001F4CA Option Chain Analysis – NIFTY / BANKNIFTY")

# ------------------------ FUNCTIONS --------------------------
def get_expiries(data):
    return data['records'].get('expiryDates', [])

def filter_weekly_expiries(expiry_list):
    weekly_expiries = []
    for date_str in expiry_list:
        try:
            dt = datetime.strptime(date_str, "%d-%b-%Y")
            if dt.weekday() == 4:  # Friday
                weekly_expiries.append(date_str)
        except:
            continue
    return weekly_expiries

def fetch_option_chain(symbol):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    session = requests.Session()
    session.headers.update(headers)
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

def black_scholes(option_type, S, K, T, r, sigma):
    if sigma == 0 or T == 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_days_to_expiry(expiry_str):
    expiry_date = dateparser.parse(expiry_str).date()
    today = datetime.today().date()
    return max((expiry_date - today).days, 0)

def parse_chain(data, expiry_filter, spot_price, r=0.06):
    T = calculate_days_to_expiry(expiry_filter) / 365
    calls, puts = [], []
    for row in data['records']['data']:
        if row.get('expiryDate') != expiry_filter:
            continue
        strike = row['strikePrice']
        if abs(strike - spot_price) > 500:
            continue
        if 'CE' in row:
            iv = max(row['CE'].get('impliedVolatility', 0), 8) / 100
            ltp = row['CE'].get('lastPrice', 0)
            fair_val = black_scholes('call', spot_price, strike, T, r, iv)
            calls.append({
                "Strike": strike,
                "LTP": ltp,
                "IV%": round(iv * 100, 2),
                "FairValue": round(fair_val, 2),
                "Diff": round(ltp - fair_val, 2),
                "OI": row['CE'].get('openInterest', 0),
                "ChgOI": row['CE'].get('changeinOpenInterest', 0),
                "Volume": row['CE'].get('totalTradedVolume', 0)
            })
        if 'PE' in row:
            iv = max(row['PE'].get('impliedVolatility', 0), 8) / 100
            ltp = row['PE'].get('lastPrice', 0)
            fair_val = black_scholes('put', spot_price, strike, T, r, iv)
            puts.append({
                "Strike": strike,
                "LTP": ltp,
                "IV%": round(iv * 100, 2),
                "FairValue": round(fair_val, 2),
                "Diff": round(ltp - fair_val, 2),
                "OI": row['PE'].get('openInterest', 0),
                "ChgOI": row['PE'].get('changeinOpenInterest', 0),
                "Volume": row['PE'].get('totalTradedVolume', 0)
            })
    return pd.DataFrame(calls), pd.DataFrame(puts)

def plot_iv_skew(df_calls, df_puts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_calls['Strike'], y=df_calls['IV%'], name='Call IV', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_puts['Strike'], y=df_puts['IV%'], name='Put IV', mode='lines+markers'))
    fig.update_layout(title="\U0001F4C9 IV Skew", xaxis_title="Strike Price", yaxis_title="Implied Volatility %")
    fig.update_xaxes(tickformat=".0f")
    return fig

def plot_oi_ladder(df_calls, df_puts):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_calls['Strike'], y=df_calls['OI'], name='Call OI', marker_color='blue'))
    fig.add_trace(go.Bar(x=df_puts['Strike'], y=df_puts['OI'], name='Put OI', marker_color='orange'))
    fig.update_layout(barmode='group', title="\U0001F4CA OI Ladder", xaxis_title="Strike Price", yaxis_title="Open Interest")
    fig.update_xaxes(tickformat=".0f")
    return fig

def compute_max_pain(df_calls, df_puts):
    total_loss = {}
    for strike in df_calls['Strike']:
        call_loss = df_calls.apply(lambda row: max(0, row['Strike'] - strike) * row['OI'], axis=1).sum()
        put_loss = df_puts.apply(lambda row: max(0, strike - row['Strike']) * row['OI'], axis=1).sum()
        total_loss[strike] = call_loss + put_loss
    return min(total_loss, key=total_loss.get)

def suggest_strategy(pcr):
    if pcr > 1:
        return "⬆ High PCR – Bullish outlook. Consider Bull Put Spread, Long Calls."
    elif pcr < 1:
        return "⬇ Low PCR – Bearish outlook. Consider Bear Call Spread, Long Puts."
    else:
        return "⚖ Neutral PCR – Consider Iron Condor, Short Straddle."

def identify_support_resistance(df_calls, df_puts):
    support = df_puts.sort_values("OI", ascending=False).head(3)[["Strike", "OI"]]
    resistance = df_calls.sort_values("OI", ascending=False).head(3)[["Strike", "OI"]]
    return support, resistance

# -------------------------- UI LOGIC --------------------------
symbol = st.selectbox("\U0001F4CC Select Index", ["NIFTY", "BANKNIFTY"])
response = fetch_option_chain(symbol)

if response and 'records' in response:
    expiry_list = get_expiries(response)
    weekly_expiry_list = filter_weekly_expiries(expiry_list)

    if not weekly_expiry_list:
        st.warning(f"⚠️ No Weekly Expiries found for {symbol}")
    else:
        expiry = st.selectbox("\U0001F4C5 Weekly Expiry", weekly_expiry_list)
        spot = response['records'].get('underlyingValue', 0)
        st.markdown(f"### \U0001F50D Selected Expiry: `{expiry}`  Spot: `{spot}`")

        df_calls, df_puts = parse_chain(response, expiry, spot)

        if df_calls.empty or df_puts.empty:
            st.warning(f"⚠️ No Option Chain data for {symbol} at expiry `{expiry}`")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"📞 CALL OPTIONS for {symbol} [{expiry}] (±5 strikes)")
                st.dataframe(df_calls.set_index("Strike"), use_container_width=True)
            with col2:
                st.subheader(f"📱 PUT OPTIONS for {symbol} [{expiry}] (±5 strikes)")
                st.dataframe(df_puts.set_index("Strike"), use_container_width=True)

            st.plotly_chart(plot_iv_skew(df_calls, df_puts), use_container_width=True)
            st.plotly_chart(plot_oi_ladder(df_calls, df_puts), use_container_width=True)

            max_pain = compute_max_pain(df_calls, df_puts)
            pcr = round(df_puts['OI'].sum() / df_calls['OI'].sum(), 2) if df_calls['OI'].sum() else 0

            st.markdown(f"### 🧮 Max Pain: `{max_pain}`  📉 PCR: `{pcr}`")
            st.markdown(f"### 💡 Strategy Suggestion: {suggest_strategy(pcr)}")

            support, resistance = identify_support_resistance(df_calls, df_puts)
            st.markdown("### 🧱 Key OI Zones")
            st.markdown("**Strongest Put (Support) OI Zones:**")
            st.dataframe(support.set_index("Strike"))
            st.markdown("**Strongest Call (Resistance) OI Zones:**")
            st.dataframe(resistance.set_index("Strike"))

            st.markdown("---")
            st.caption("📧 Contact: info@vedikavanijya.com | © 2025 Vedika Stock Broking Pvt. Ltd.")
            st.caption("🌐 Access Live App: [https://vedikaoptionchainanalysis.streamlit.app](https://vedikaoptionchainanalysis.streamlit.app)")
            st.markdown("---")
            st.caption("💻 Developed by **DHEERAJ KHANWANI**")
