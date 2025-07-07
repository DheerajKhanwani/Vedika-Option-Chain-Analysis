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

def get_expiries(data):
    return data['records'].get('expiryDates', [])

def calculate_days_to_expiry(expiry_str):
    expiry_date = dateparser.parse(expiry_str).date()
    today = datetime.today().date()
    return max((expiry_date - today).days, 0)

def black_scholes(option_type, S, K, T, r, sigma):
    if sigma == 0 or T == 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_greeks(option_type, S, K, T, r, sigma):
    if sigma == 0 or T == 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2)) / 100
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

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
            iv = max(row['CE'].get('impliedVolatility', 8), 8) / 100
            ltp = row['CE'].get('lastPrice', 0)
            fair_val = black_scholes('call', spot_price, strike, T, r, iv)
            greeks = option_greeks('call', spot_price, strike, T, r, iv)
            calls.append({"Strike": strike, "LTP": ltp, "IV%": round(iv*100,2), "FairValue": round(fair_val,2), "Diff": round(ltp-fair_val,2),
                          "OI": row['CE'].get('openInterest',0), "ChgOI": row['CE'].get('changeinOpenInterest',0), "Volume": row['CE'].get('totalTradedVolume',0),
                          **greeks})
        if 'PE' in row:
            iv = max(row['PE'].get('impliedVolatility', 8), 8) / 100
            ltp = row['PE'].get('lastPrice', 0)
            fair_val = black_scholes('put', spot_price, strike, T, r, iv)
            greeks = option_greeks('put', spot_price, strike, T, r, iv)
            puts.append({"Strike": strike, "LTP": ltp, "IV%": round(iv*100,2), "FairValue": round(fair_val,2), "Diff": round(ltp-fair_val,2),
                         "OI": row['PE'].get('openInterest',0), "ChgOI": row['PE'].get('changeinOpenInterest',0), "Volume": row['PE'].get('totalTradedVolume',0),
                         **greeks})
    return pd.DataFrame(calls), pd.DataFrame(puts)

def plot_oi_heatmap(df, label):
    fig = go.Figure(data=[
        go.Bar(x=df['Strike'], y=df['ChgOI'], marker=dict(color=df['ChgOI'], colorscale='RdYlGn'), name=label)]
    )
    fig.update_layout(title=f"{label} Change in OI Heatmap", xaxis_title="Strike Price", yaxis_title="Chg in OI", xaxis=dict(tickformat=".0f"))
    return fig

def plot_cumulative_oi(df_calls, df_puts):
    combined = pd.DataFrame()
    combined['Strike'] = df_calls['Strike']
    combined['Total OI'] = df_calls['OI'] + df_puts['OI']
    fig = go.Figure(go.Bar(x=combined['Strike'], y=combined['Total OI'], name='Total OI', marker_color='green'))
    fig.update_layout(title="Cumulative OI Support/Resistance Zones", xaxis_title="Strike Price", yaxis_title="Total OI", xaxis=dict(tickformat=".0f"))
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
        return "\U0001F53C High PCR – Bullish. Consider Bull Put Spread, Long Calls."
    elif pcr < 1:
        return "\U0001F53D Low PCR – Bearish. Consider Bear Call Spread, Long Puts."
    else:
        return "⚖ Neutral PCR – Use Iron Condor, Short Straddle."

# -------------------------- UI LOGIC --------------------------
symbol = st.selectbox("\U0001F4CC Select Index", ["NIFTY", "BANKNIFTY"])
response = fetch_option_chain(symbol)

if response and 'records' in response:
    expiry_list = get_expiries(response)
    expiry = st.selectbox("\U0001F4C5 Select Expiry Date", expiry_list)
    spot = response['records'].get('underlyingValue', 0)
    st.markdown(f"### \U0001F50D Selected Expiry: `{expiry}`  |  Spot: `{spot}`")

    df_calls, df_puts = parse_chain(response, expiry, spot)

    if df_calls.empty or df_puts.empty:
        st.warning(f"⚠ No Option Chain data available for {symbol} [{expiry}]")
    else:
        st.subheader(f"\U0001F4DE CALL OPTIONS for {symbol} [{expiry}]")
        st.dataframe(df_calls.set_index("Strike"))
        st.subheader(f"\U0001F4F1 PUT OPTIONS for {symbol} [{expiry}]")
        st.dataframe(df_puts.set_index("Strike"))

        st.plotly_chart(plot_oi_heatmap(df_calls, "Call"), use_container_width=True)
        st.plotly_chart(plot_oi_heatmap(df_puts, "Put"), use_container_width=True)
        st.plotly_chart(plot_cumulative_oi(df_calls, df_puts), use_container_width=True)

        max_pain = compute_max_pain(df_calls, df_puts)
        pcr = round(df_puts['OI'].sum() / df_calls['OI'].sum(), 2)

        st.markdown(f"### 🧮 Max Pain: `{max_pain}`  📉 PCR: `{pcr}`")
        st.markdown(f"### 💡 Strategy Suggestion: {suggest_strategy(pcr)}")

        if pcr < 0.8 or pcr > 1.2:
            st.warning(f"⚠️ PCR Alert: Unusual value of `{pcr}` detected")

        st.markdown("---")
        st.caption("📧 Contact: info@vedikavanijya.com | © 2025 Vedika Stock Broking Pvt. Ltd.")
        st.caption("🌐 Live App: https://vedikaoptionchainanalysis.streamlit.app")
        st.caption("💻 Developed by DHEERAJ KHANWANI")
