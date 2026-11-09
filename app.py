import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import load_dataset, load_models_and_metadata, predict_direction, get_ensemble_prediction
from src.features import compute_all_indicators

# Page Configuration
st.set_page_config(
    page_title="NIFTY 50 Market Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern Bloomberg/FinTech styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    .hero-prediction-bullish {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
    }
    
    .hero-prediction-bearish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.2);
    }
    
    .badge-bullish {
        display: inline-block;
        background-color: #065f46;
        color: #34d399;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    .badge-bearish {
        display: inline-block;
        background-color: #7f1d1d;
        color: #f87171;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    .tech-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        padding: 4px 8px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def get_cached_data():
    return load_dataset()

# Cache model loading
@st.cache_resource
def get_cached_models():
    return load_models_and_metadata()

try:
    market_df = get_cached_data()
    models, scaler, metadata = get_cached_models()
    load_success = True
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    load_success = False

if not load_success:
    st.stop()

# Global Data Variables
latest_row = market_df.iloc[-1]
prev_row = market_df.iloc[-2]
latest_date_str = market_df.index[-1].strftime("%A, %d %B %Y")
latest_nifty_price = latest_row["NIFTY_Close"]
nifty_return_val = latest_row["NIFTY_Return"] * 100
nifty_change_pts = latest_nifty_price - prev_row["NIFTY_Close"]

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("NIFTY 50 AI")
    st.caption("Quantitative Market Direction Intelligence")
    st.markdown("---")
    
    selected_model_name = st.selectbox(
        "🧠 Primary Forecast Model",
        options=["XGBoost (Tuned)", "CatBoost (Tuned)", "LightGBM (Tuned)", "Random Forest", "Logistic Regression"],
        index=0,
        help="Select the machine learning algorithm to generate predictions."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Market Snapshot")
    st.markdown(f"**As of:** `{latest_date_str}`")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric(
            label="NIFTY 50",
            value=f"₹{latest_nifty_price:,.2f}",
            delta=f"{nifty_change_pts:+.2f} ({nifty_return_val:+.2f}%)"
        )
    with col_sb2:
        vix_val = latest_row["VIX_Close"]
        vix_ret = latest_row["VIX_Return"] * 100
        st.metric(
            label="India VIX",
            value=f"{vix_val:.2f}",
            delta=f"{vix_ret:+.2f}%",
            delta_color="inverse"
        )
        
    st.markdown("---")
    st.markdown("### 🌐 Global Influencers")
    sp500_ret = latest_row["SP500_Return"] * 100
    nasdaq_ret = latest_row["NASDAQ_Return"] * 100
    dow_ret = latest_row["DOW_Return"] * 100
    crude_ret = latest_row["CRUDE_Return"] * 100
    usdinr_ret = latest_row["USDINR_Return"] * 100
    
    st.markdown(f"- **S&P 500:** `{'🟢' if sp500_ret >= 0 else '🔴'} {sp500_ret:+.2f}%`")
    st.markdown(f"- **NASDAQ:** `{'🟢' if nasdaq_ret >= 0 else '🔴'} {nasdaq_ret:+.2f}%`")
    st.markdown(f"- **Dow Jones:** `{'🟢' if dow_ret >= 0 else '🔴'} {dow_ret:+.2f}%`")
    st.markdown(f"- **Crude Oil:** `{'🟢' if crude_ret >= 0 else '🔴'} {crude_ret:+.2f}%`")
    st.markdown(f"- **USD / INR:** `{'🟢' if usdinr_ret >= 0 else '🔴'} {usdinr_ret:+.2f}%`")
    
    st.markdown("---")
    st.info("💡 **Hypothesis Validated:** Combining domestic indicators with global macroeconomic signals yields superior directional accuracy (58.47% XGBoost test set).")


# TOP HEADER
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("📈 NIFTY 50 Next-Day Market Direction Predictor")
    st.markdown(
        "An enterprise-grade machine learning application utilizing domestic market dynamics, "
        "global indices (S&P 500, NASDAQ, Dow), currency, commodities, and 25 engineered technical indicators."
    )
with header_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔄 Refresh Application", on_click=st.cache_data.clear)

# MAIN TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 AI Direction Forecast",
    "📊 Interactive Technical Charts",
    "🤖 Model Benchmarks & Metrics",
    "📈 Backtesting & Strategy Sim",
    "📁 Dataset Explorer & Catalog"
])

# ==============================================================================
# TAB 1: AI DIRECTION FORECAST
# ==============================================================================
with tab1:
    st.markdown("### 🎯 Tomorrow's Market Forecast")
    
    # Calculate current prediction with selected model
    features_input = latest_row[metadata["selected_features"]].to_dict()
    pred_res = predict_direction(selected_model_name, features_input, models, scaler, metadata)
    ensemble_res = get_ensemble_prediction(features_input, models, scaler, metadata)
    
    is_bullish = (pred_res["prediction"] == 1)
    card_class = "hero-prediction-bullish" if is_bullish else "hero-prediction-bearish"
    dir_text = "BULLISH (UPWARD MOVE EXPECTED)" if is_bullish else "BEARISH (DOWNWARD MOVE EXPECTED)"
    dir_color = "#10b981" if is_bullish else "#ef4444"
    dir_icon = "🚀" if is_bullish else "🔻"
    
    # Hero Card
    st.markdown(f"""
    <div class="{card_class}">
        <span style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.7);">
            Forecast for Next Trading Session ({selected_model_name})
        </span>
        <h1 style="color: {dir_color}; font-size: 2.8rem; margin: 12px 0 8px 0; font-weight: 800;">
            {dir_icon} {dir_text}
        </h1>
        <p style="font-size: 1.25rem; color: #f1f5f9; margin-bottom: 0;">
            Confidence: <b style="color: {dir_color};">{pred_res['confidence']*100:.1f}%</b> probability of next-day closing higher
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gauge and Probability Metrics
    col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1])
    
    with col_p1:
        st.markdown("#### ⚡ Probability Breakdown")
        prob_up_pct = pred_res["prob_up"] * 100
        prob_down_pct = pred_res["prob_down"] * 100
        
        st.write(f"**Bullish Probability (Up):** `{prob_up_pct:.1f}%`")
        st.progress(pred_res["prob_up"])
        
        st.write(f"**Bearish Probability (Down):** `{prob_down_pct:.1f}%`")
        st.progress(pred_res["prob_down"])
        
    with col_p2:
        st.markdown("#### 🏛️ Multi-Model Consensus")
        st.write(f"**Consensus Direction:** `{ensemble_res['consensus_direction']}`")
        st.write(f"**Average Ensemble Confidence:** `{ensemble_res['avg_prob_up']*100:.1f}% Bullish`")
        
        # Mini table of all models
        consensus_rows = []
        for m_name, m_res in ensemble_res["individual_results"].items():
            consensus_rows.append({
                "Model": m_name,
                "Signal": "🟢 UP" if m_res["prediction"] == 1 else "🔴 DOWN",
                "Up Prob": f"{m_res['prob_up']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(consensus_rows), hide_index=True, use_container_width=True)
        
    with col_p3:
        st.markdown("#### 📐 Current Technical State")
        rsi_val = latest_row["RSI_14"]
        macd_val = latest_row["MACD"]
        bb_pos = latest_row["Price_vs_BB"]
        
        st.markdown(f"- **RSI (14):** `{rsi_val:.1f}` ({'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'})")
        st.markdown(f"- **MACD:** `{macd_val:.2f}` ({'Bullish' if macd_val > 0 else 'Bearish'})")
        st.markdown(f"- **BB Band Position:** `{bb_pos*100:.1f}%`")
        st.markdown(f"- **20D Volatility:** `{latest_row['Rolling_Volatility_20']*100:.2f}%`")

    st.markdown("---")
    
    # WHAT-IF SCENARIO SIMULATOR
    st.markdown("### 🧪 What-If Scenario Simulator")
    st.markdown(
        "Simulate how hypothetical global market shocks, overnight sentiment swings, "
        "or technical indicator shifts would affect the AI model's forecast."
    )
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        sim_sp500 = st.slider("S&P 500 Daily Move (%)", min_value=-5.0, max_value=5.0, value=float(round(sp500_ret, 2)), step=0.1)
        sim_nasdaq = st.slider("NASDAQ Daily Move (%)", min_value=-5.0, max_value=5.0, value=float(round(nasdaq_ret, 2)), step=0.1)
    with sim_col2:
        sim_vix = st.slider("India VIX Change (%)", min_value=-25.0, max_value=25.0, value=float(round(vix_ret, 2)), step=0.5)
        sim_crude = st.slider("Crude Oil Move (%)", min_value=-8.0, max_value=8.0, value=float(round(crude_ret, 2)), step=0.1)
    with sim_col3:
        sim_usdinr = st.slider("USD/INR Move (%)", min_value=-2.0, max_value=2.0, value=float(round(usdinr_ret, 2)), step=0.05)
        sim_rsi = st.slider("NIFTY RSI (14)", min_value=10.0, max_value=90.0, value=float(round(rsi_val, 1)), step=0.5)
        
    # Prepare simulated feature dictionary
    simulated_features = features_input.copy()
    simulated_features["SP500_Return"] = sim_sp500 / 100.0
    simulated_features["NASDAQ_Return"] = sim_nasdaq / 100.0
    simulated_features["VIX_Return"] = sim_vix / 100.0
    simulated_features["CRUDE_Return"] = sim_crude / 100.0
    simulated_features["USDINR_Return"] = sim_usdinr / 100.0
    simulated_features["RSI_14"] = sim_rsi
    
    sim_pred = predict_direction(selected_model_name, simulated_features, models, scaler, metadata)
    
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px; padding: 16px; margin-top: 10px;">
        <span style="font-weight: 600; color: #94a3b8;">Simulated Outcome ({selected_model_name}):</span>
        <h3 style="margin: 6px 0; color: {'#10b981' if sim_pred['prediction']==1 else '#ef4444'};">
            {'🚀 BULLISH' if sim_pred['prediction']==1 else '🔻 BEARISH'} &nbsp;
            (Estimated Up Probability: {sim_pred['prob_up']*100:.1f}%)
        </h3>
        <span style="font-size: 0.9rem; color: #cbd5e1;">
            Under this custom scenario, the model estimates a <b>{sim_pred['confidence']*100:.1f}%</b> probability of the market moving {sim_pred['direction'].split()[0].lower()}.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: INTERACTIVE TECHNICAL CHARTS
# ==============================================================================
with tab2:
    st.markdown("### 📊 NIFTY 50 Technical Analysis & Oscillators")
    
    chart_col1, chart_col2, chart_col3 = st.columns([2, 1, 1])
    with chart_col1:
        time_filter = st.radio(
            "Select Timeframe Range:",
            options=["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "All History"],
            index=3,
            horizontal=True
        )
    with chart_col2:
        chart_type = st.radio("Chart Type:", options=["Candlestick", "Line Chart"], horizontal=True)
    with chart_col3:
        overlays = st.multiselect(
            "Overlay Indicators:",
            options=["SMA 10 & 20", "EMA 10 & 20", "Bollinger Bands"],
            default=["SMA 10 & 20", "Bollinger Bands"]
        )

    # Filter dataframe by timeframe
    max_date = market_df.index[-1]
    if time_filter == "1 Month":
        start_date = max_date - pd.DateOffset(months=1)
    elif time_filter == "3 Months":
        start_date = max_date - pd.DateOffset(months=3)
    elif time_filter == "6 Months":
        start_date = max_date - pd.DateOffset(months=6)
    elif time_filter == "1 Year":
        start_date = max_date - pd.DateOffset(years=1)
    elif time_filter == "3 Years":
        start_date = max_date - pd.DateOffset(years=3)
    else:
        start_date = market_df.index[0]
        
    plot_df = market_df.loc[start_date:max_date]
    
    # Create Subplot Figure: Price/Overlays (row 1), Volume (row 2), RSI (row 3), MACD (row 4)
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=("NIFTY 50 Price & Overlays", "Trading Volume", "RSI (14)", "MACD (12, 26, 9)")
    )
    
    # Row 1: Price
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["NIFTY_Open"],
                high=plot_df["NIFTY_High"],
                low=plot_df["NIFTY_Low"],
                close=plot_df["NIFTY_Close"],
                name="NIFTY 50",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444"
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["NIFTY_Close"],
                mode="lines",
                name="NIFTY 50 Close",
                line=dict(color="#38bdf8", width=2)
            ),
            row=1, col=1
        )
        
    # Overlays
    if "SMA 10 & 20" in overlays:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["NIFTY_SMA_10"], mode="lines", name="SMA 10", line=dict(color="#f59e0b", width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["NIFTY_SMA_20"], mode="lines", name="SMA 20", line=dict(color="#8b5cf6", width=1.5)),
            row=1, col=1
        )
        
    if "EMA 10 & 20" in overlays:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["NIFTY_EMA_10"], mode="lines", name="EMA 10", line=dict(color="#ec4899", width=1.5, dash="dot")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["NIFTY_EMA_20"], mode="lines", name="EMA 20", line=dict(color="#06b6d4", width=1.5, dash="dot")),
            row=1, col=1
        )
        
    if "Bollinger Bands" in overlays:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["BB_High"], mode="lines", name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BB_Low"], mode="lines", name="BB Lower",
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
                fill="tonexty", fillcolor="rgba(148, 163, 184, 0.05)"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["BB_Middle"], mode="lines", name="BB Mid", line=dict(color="rgba(148, 163, 184, 0.6)", width=1, dash="dash")),
            row=1, col=1
        )

    # Row 2: Volume
    colors_vol = ["#10b981" if r >= 0 else "#ef4444" for r in plot_df["NIFTY_Return"]]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["NIFTY_Volume"], name="Volume", marker_color=colors_vol, opacity=0.8),
        row=2, col=1
    )
    
    # Row 3: RSI
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI_14"], mode="lines", name="RSI (14)", line=dict(color="#a855f7", width=1.5)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=3, col=1, annotation_text="Oversold (30)")
    
    # Row 4: MACD
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["MACD"], mode="lines", name="MACD", line=dict(color="#38bdf8", width=1.5)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["MACD_Signal"], mode="lines", name="Signal", line=dict(color="#f97316", width=1.5)),
        row=4, col=1
    )
    hist_colors = ["#10b981" if h >= 0 else "#ef4444" for h in plot_df["MACD_Histogram"]]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["MACD_Histogram"], name="Histogram", marker_color=hist_colors),
        row=4, col=1
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 3: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
with tab3:
    st.markdown("### 🤖 Model Performance & Evaluation Suite")
    st.markdown(
        "Evaluation results on the held-out out-of-sample Test Set (`2025 – 2026`). "
        "Models were chronologically evaluated to prevent temporal lookahead bias."
    )
    
    metrics = metadata["metrics"]
    metrics_df = pd.DataFrame(metrics).T[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    
    # Formatted Benchmark Table
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### 🏆 Test Set Performance Benchmark")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color="#1e3a5f").format("{:.4f}"),
            use_container_width=True
        )
        
        st.markdown("""
        **Key Empirical Takeaways:**
        - **XGBoost (Tuned):** Champion model with the best overall balance across **Accuracy (58.47%)**, **Precision (60.13%)**, and **ROC-AUC (0.5957)**.
        - **CatBoost (Tuned):** Highly competitive boosting alternative (56.83% Accuracy, 0.5928 ROC-AUC).
        - **Random Forest:** Solid ensemble model with highest recall among tree models (65.03%).
        - **Logistic Regression:** Linear baseline that favors high recall (80.87%), demonstrating that tree ensembles capture essential non-linear signals.
        """)
        
    with col_t2:
        st.markdown("#### ⏱️ Chronological Time-Series Split")
        split_data = pd.DataFrame({
            "Dataset Split": ["Training Set", "Validation Set", "Test Set"],
            "Period Range": ["2008 – 2022", "2023 – 2024", "2025 – 2026"],
            "Samples": [3473, 475, 366],
            "Purpose": ["Model Training & Feature Weights", "Hyperparameter Optimization", "Final Out-of-Sample Evaluation"]
        })
        st.dataframe(split_data, hide_index=True, use_container_width=True)
        
    st.markdown("---")
    
    # Confusion Matrix & Feature Importance
    eval_c1, eval_c2 = st.columns(2)
    
    with eval_c1:
        st.markdown("#### 🎯 Confusion Matrix")
        selected_cm_model = st.selectbox("Select Model for Confusion Matrix:", list(metrics.keys()), index=0)
        cm = np.array(metrics[selected_cm_model]["ConfusionMatrix"])
        
        cm_labels = ["Actual DOWN (0)", "Actual UP (1)"]
        pred_labels = ["Pred DOWN (0)", "Pred UP (1)"]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=pred_labels,
            y=cm_labels,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale="Blues",
            showscale=False
        ))
        fig_cm.update_layout(
            template="plotly_dark",
            title=f"Confusion Matrix: {selected_cm_model}",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with eval_c2:
        st.markdown("#### 🌲 Top 15 Informative Features (XGBoost)")
        imp_dict = metadata["feature_importances"]["XGBoost"]
        imp_df = pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True).tail(15)
        
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color="#6366f1"
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            title="Feature Importance Ranking",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
            xaxis_title="Importance Weight"
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# TAB 4: BACKTESTING & STRATEGY SIMULATION
# ==============================================================================
with tab4:
    st.markdown("### 📈 Walk-Forward Backtest & Strategy Simulation")
    st.markdown(
        "Simulating an algorithmic trading strategy on the unseen out-of-sample Test Set (`2025 – 2026`). "
        "Whenever the model predicts **UP (1)**, a long position in NIFTY 50 is held; otherwise, the strategy remains flat in cash."
    )
    
    test_dates = pd.to_datetime(metadata["test_dates"])
    test_actuals = np.array(metadata["test_actuals"])
    
    backtest_model = st.selectbox("Select Model for Backtesting:", list(models.keys()), index=0)
    test_preds = np.array(metadata["test_predictions"][backtest_model])
    
    # Calculate returns
    test_market_df = market_df.loc[test_dates[0]:test_dates[-1]]
    market_returns = test_market_df["NIFTY_Return"].values
    
    # Strategy Return: Long if pred == 1
    # Shift predictions by 1 to represent trading on next day's open/close return
    strat_returns = np.where(test_preds[:-1] == 1, market_returns[1:], 0.0)
    market_eval_returns = market_returns[1:]
    sim_dates = test_dates[1:]
    
    cum_market = np.cumprod(1 + market_eval_returns) - 1
    cum_strat = np.cumprod(1 + strat_returns) - 1
    
    # Metrics
    tot_strat_ret = cum_strat[-1] * 100
    tot_market_ret = cum_market[-1] * 100
    win_trades = np.sum((strat_returns > 0))
    total_trades = np.sum(test_preds[:-1] == 1)
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Max Drawdowns
    strat_equity = np.cumprod(1 + strat_returns)
    strat_peak = np.maximum.accumulate(strat_equity)
    strat_dd = (strat_equity - strat_peak) / strat_peak
    max_dd_strat = np.min(strat_dd) * 100
    
    market_equity = np.cumprod(1 + market_eval_returns)
    market_peak = np.maximum.accumulate(market_equity)
    market_dd = (market_equity - market_peak) / market_peak
    max_dd_market = np.min(market_dd) * 100
    
    # Metric KPI Cards
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.metric(
            label="Strategy Cumulative Return",
            value=f"{tot_strat_ret:+.2f}%",
            delta=f"{tot_strat_ret - tot_market_ret:+.2f}% vs Buy & Hold"
        )
    with col_b2:
        st.metric(
            label="Buy & Hold NIFTY 50",
            value=f"{tot_market_ret:+.2f}%"
        )
    with col_b3:
        st.metric(
            label="Trade Win Rate",
            value=f"{win_rate:.1f}%",
            help="Percentage of long signals that resulted in a positive return."
        )
    with col_b4:
        st.metric(
            label="Strategy Max Drawdown",
            value=f"{max_dd_strat:.2f}%",
            delta=f"{max_dd_strat - max_dd_market:+.2f}% vs Market",
            delta_color="inverse"
        )
        
    # Equity Curve Plot
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=sim_dates,
        y=cum_strat * 100,
        mode="lines",
        name=f"ML Strategy ({backtest_model})",
        line=dict(color="#10b981", width=2.5)
    ))
    fig_equity.add_trace(go.Scatter(
        x=sim_dates,
        y=cum_market * 100,
        mode="lines",
        name="Buy & Hold (NIFTY 50)",
        line=dict(color="#94a3b8", width=1.5, dash="dash")
    ))
    fig_equity.update_layout(
        template="plotly_dark",
        title="Cumulative Return Comparison (2025 - 2026 Out-of-Sample Period)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_equity, use_container_width=True)


# ==============================================================================
# TAB 5: DATASET EXPLORER & FEATURE CATALOG
# ==============================================================================
with tab5:
    st.markdown("### 📁 Dataset Explorer & Feature Catalog")
    st.markdown(
        "Browse the engineered market dataset containing synchronized domestic quotes, "
        "international indices, currency, commodities, and technical indicators."
    )
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.write(f"**Total Records:** `{len(market_df):,}` trading days")
        st.write(f"**Total Features:** `{market_df.shape[1]}` columns")
        st.write(f"**Time Horizon:** `{market_df.index[0].date()}` to `{market_df.index[-1].date()}`")
        
        # Download button
        csv_data = market_df.tail(500).to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Recent 500 Days Data (CSV)",
            data=csv_data,
            file_name="nifty50_recent_data.csv",
            mime="text/csv"
        )
    with col_d2:
        st.markdown("#### 📖 Feature Taxonomy Breakdown")
        tax_col1, tax_col2 = st.columns(2)
        with tax_col1:
            st.markdown("- **Domestic Indices:** NIFTY 50, India VIX")
            st.markdown("- **Global Indices:** S&P 500, NASDAQ, Dow Jones")
            st.markdown("- **Commodities & Forex:** WTI Crude Oil, USD/INR")
        with tax_col2:
            st.markdown("- **Trend:** SMA 10/20, EMA 10/20, Price Ratios")
            st.markdown("- **Momentum:** RSI (14), MACD (12,26,9), ROC (10)")
            st.markdown("- **Volatility:** Bollinger Bands, ATR (14), 20D Volatility")
            
    st.markdown("---")
    st.markdown("#### 🔍 Historical Data Table")
    
    columns_to_show = st.multiselect(
        "Select Columns to Display:",
        options=market_df.columns.tolist(),
        default=["NIFTY_Close", "NIFTY_Return", "Target", "SP500_Return", "VIX_Close", "RSI_14", "MACD"]
    )
    
    st.dataframe(
        market_df[columns_to_show].tail(100).sort_index(ascending=False),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.4); font-size: 0.85rem; padding: 10px;'>"
    "NIFTY 50 Market Direction Machine Learning Pipeline &nbsp;|&nbsp; Built with Streamlit, Scikit-Learn, XGBoost, CatBoost & Plotly"
    "</div>",
    unsafe_allow_html=True
)
