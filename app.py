import streamlit as st
import os
import time
from datetime import datetime
import backend_api as backend 
from signal_generation.signals import get_signal

# --- CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.set_page_config(
    page_title="Sentient | AI Crypto Signals", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

#  THEME MANAGEMENT
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

def toggle_theme():
    if st.session_state.theme == 'Dark':
        st.session_state.theme = 'Light'
    else:
        st.session_state.theme = 'Dark'

#  CSS STYLES
# 1. DARK MODE CSS
dark_mode_css = """
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: white; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px; padding: 30px; 
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 25px;
    }
    .metric-container { background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(255,255,255,0.05); }
    .metric-val { color: #e2e8f0 !important; }
    .metric-lbl { color: #94a3b8 !important; }
    .big-text { color: white !important; }
    .sub-text { color: #94a3b8 !important; }
    h1, h2, h3, p, label, div { color: white; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
</style>
"""

# 2. LIGHT MODE CSS
light_mode_css = """
<style>
    .stApp { background: #f8fafc; color: #0f172a; }
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 24px; padding: 30px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 25px;
    }
    .metric-container { background: #f1f5f9; border: 1px solid #e2e8f0; }
    .metric-val { color: #0f172a !important; }
    .metric-lbl { color: #475569 !important; }
    .big-text { color: #0f172a !important; }
    .sub-text { color: #475569 !important; }
    h1, h2, h3, h4 { color: #0f172a !important; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    section[data-testid="stSidebar"] * { color: #0f172a !important; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #334155 !important; }
    section[data-testid="stSidebar"] small { color: #64748b !important; }
    div[data-testid="stMetricValue"] { color: #0f172a !important; }
</style>
"""

# 3. SHARED STYLES
shared_css = """
<style>
    .metric-container {
        display: flex; flex-direction: column; align-items: center;
        padding: 15px; border-radius: 15px;
    }
    .metric-val { font-size: 24px; font-weight: bold; font-family: 'Courier New', monospace; }
    .metric-lbl { font-size: 11px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .signal-buy { border-left: 5px solid #10b981; }
    .signal-sell { border-left: 5px solid #ef4444; }
    .signal-hold { border-left: 5px solid #9ca3af; }
    .big-text { font-size: 32px; font-weight: 800; letter-spacing: -1px; }
    .sub-text { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white !important; border: none; padding: 12px 24px; border-radius: 12px;
        font-weight: 600; width: 100%; transition: all 0.2s;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }
</style>
"""

if st.session_state.theme == 'Dark':
    st.markdown(dark_mode_css, unsafe_allow_html=True)
else:
    st.markdown(light_mode_css, unsafe_allow_html=True)
st.markdown(shared_css, unsafe_allow_html=True)

# --- CACHING LAYER ---
@st.cache_data(ttl=300, show_spinner=False) 
def cached_market_analysis(ticker):
    df = backend.get_market_data(ticker)
    window_hours = 180 * 24
    latest_prediction = backend.run_ai_training(df, window_hours)
    return df, latest_prediction

#  SIDEBAR
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=60)
    st.title("Crypto Price Prediction")
    st.caption(" Name - D.S.L.Gamage")
    st.caption(" Student ID : 20210143")
    st.caption("IPD Prototype Submission ")
    st.caption(" Supervised By : Dr. Aloka Fernando")
    st.divider()
    
    st.markdown("### ⚙️ Signal Controls")
    
    # --- ASSET SELECTION ---
    ticker = st.selectbox("Asset", ["BTC/USDT", "ADA/USDT"], index=0)
    
    leverage = 10 
    
    st.divider()
    generate_btn = st.button("⚡ Generate AI Signal", type="primary")
    st.markdown("---")
    
    mode_label = "🌙 Dark Mode" if st.session_state.theme == 'Dark' else "☀️ Light Mode"
    st.toggle(mode_label, value=(st.session_state.theme == 'Dark'), on_change=toggle_theme)
    
    st.info("System Status: **Online 🟢**")

#  MAIN DASHBOARD
col1, col2 = st.columns([3, 1])
with col1:
    st.title(f"{ticker.split('/')[0]} Futures Intelligence") # Dynamic Title
    st.markdown("### Real-Time LSTM Predictive Engine")
with col2:
    st.metric(label="SERVER TIME (UTC)", value=datetime.utcnow().strftime("%H:%M:%S"))

if generate_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner(""):
        status_text.text(f"🔗 Connecting to {ticker} API...")
        progress_bar.progress(25)
        
        df, latest = cached_market_analysis(ticker) 
        progress_bar.progress(60)
        
        status_text.text("🧠 Running LSTM Neural Network...")
        signal = get_signal(
            current_price=latest['Actual_Close'], 
            predicted_price=latest['Predicted_Close'], 
            df_history=df, 
            leverage=leverage
        )
        # Update coin name in signal just in case
        signal['coin'] = ticker 
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        st.session_state['latest_signal'] = signal

if 'latest_signal' in st.session_state:
    sig = st.session_state['latest_signal']
    
    if sig['action'] == "Long":
        theme_class, icon, color = "signal-buy", "🟢 BUY", "#10b981"
        main_msg = f"<b>{sig['reason']}</b><br>AI Predicts Uptrend + Technical Confluence."
    elif sig['action'] == "Short":
        theme_class, icon, color = "signal-sell", "🔴 SELL", "#ef4444"
        main_msg = f"<b>{sig['reason']}</b><br>AI Predicts Downtrend + Technical Weakness."
    else:
        theme_class, icon, color = "signal-hold", "⚪ HOLD", "#94a3b8"
        main_msg = f"<b>{sig['reason']}</b><br>Conditions not met. Wait for better setup."

    st.markdown(f"""
    <div class="glass-card {theme_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="sub-text">AI DECISION</div>
                <div class="big-text" style="color:{color}">{icon} {sig['action']}</div>
            </div>
            <div style="text-align:right;">
                <div class="sub-text">CONFIDENCE</div>
                <div class="big-text">{sig['confidence']:.1f}%</div>
            </div>
        </div>
        <hr style="border-color:rgba(128,128,128,0.2); margin: 15px 0;">
        <div style="font-size: 16px;">
            {main_msg} <br>
            <div style="margin-top:10px; font-size:13px; opacity:0.8;">
                RSI: <b>{sig['rsi']:.1f}</b> | Trend: <b>{sig['trend']}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if sig['action'] != "HOLD":
        colA, colB, colC = st.columns(3)
        with colA: st.markdown(f"""<div class="glass-card metric-container"><div class="metric-lbl">📍 ENTRY PRICE</div><div class="metric-val">${sig['entry']:,.4f}</div></div>""", unsafe_allow_html=True)
        with colB: st.markdown(f"""<div class="glass-card metric-container" style="border-bottom: 3px solid #ef4444;"><div class="metric-lbl">🛑 STOP LOSS</div><div class="metric-val" style="color:#fca5a5">${sig['stop_loss']:,.4f}</div></div>""", unsafe_allow_html=True)
        with colC: st.markdown(f"""<div class="glass-card metric-container" style="border-bottom: 3px solid #10b981;"><div class="metric-lbl">🎯 TAKE PROFIT</div><div class="metric-val" style="color:#6ee7b7">${sig['target']:,.4f}</div></div>""", unsafe_allow_html=True)
        
        st.markdown("### ⚖️ Risk Analysis")
        st.write(f"**Estimated Risk (with {sig['leverage']}x Leverage):** {sig['risk_lev']:.2f}%")
        st.progress(min(sig['risk_lev']/20, 1.0)) 
        c1, c2 = st.columns(2)
        c1.info(f"💰 **Reward:** +{sig['reward_lev']:.2f}%")
        c2.warning(f"🛡️ **Risk:** -{sig['risk_lev']:.2f}%")
    else:
        st.image("https://media.tenor.com/GfSX-u7V94wAAAAC/coding-waiting.gif", width=100)
        st.caption(f"AI suggests HOLDING. Reason: {sig.get('reason', 'Wait')}")

else:
    st.info("👈 Click **'Generate AI Signal'** in the sidebar to start.")
    st.markdown("""
    <div style="text-align:center; padding: 50px; opacity: 0.7;">
        <h2>Waiting for Command...</h2>
        <p>System is ready to analyze 180 days of market data.</p>
    </div>
    """, unsafe_allow_html=True)

