import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ============================================================
# AI FRAUD DETECTION CENTER
# STEP 9
# ============================================================


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #111c35 0%,
                #070b14 40%,
                #05070d 100%
            );

        color: #ffffff;
    }


    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* =========================
       HEADER
       ========================= */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }


    .subtitle {
        color: #8b95a7;
        font-size: 16px;
        margin-bottom: 28px;
    }


    /* =========================
       METRIC CARDS
       ========================= */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                #111827,
                #0a1020
            );

        border: 1px solid #1f2937;

        border-radius: 16px;

        padding: 22px;

        min-height: 125px;

        box-shadow:
            0 10px 30px
            rgba(0, 0, 0, 0.25);
    }


    .metric-title {
        color: #8b95a7;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }


    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
    }


    .metric-danger {
        color: #ff4d5a;
        font-size: 30px;
        font-weight: 800;
    }


    .metric-safe {
        color: #22c55e;
        font-size: 30px;
        font-weight: 800;
    }


    /* =========================
       SECTION
       ========================= */

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* =========================
       RESULT BOXES
       ========================= */

    .fraud-box {

        background:
            linear-gradient(
                135deg,
                #3b0a0a,
                #170707
            );

        border: 1px solid #ef4444;

        border-radius: 18px;

        padding: 30px;

        text-align: center;

        margin-top: 20px;

        box-shadow:
            0 0 30px
            rgba(239, 68, 68, 0.10);
    }


    .safe-box {

        background:
            linear-gradient(
                135deg,
                #062e1a,
                #03150d
            );

        border: 1px solid #22c55e;

        border-radius: 18px;

        padding: 30px;

        text-align: center;

        margin-top: 20px;

        box-shadow:
            0 0 30px
            rgba(34, 197, 94, 0.10);
    }


    .risk-high {
        color: #ef4444;
        font-size: 36px;
        font-weight: 800;
    }


    .risk-medium {
        color: #f59e0b;
        font-size: 36px;
        font-weight: 800;
    }


    .risk-low {
        color: #22c55e;
        font-size: 36px;
        font-weight: 800;
    }


    /* =========================
       BUTTON
       ========================= */

    .stButton > button {

        width: 100%;

        min-height: 48px;

        border-radius: 10px;

        font-weight: 700;

        font-size: 15px;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #171a24,
                #10121a
            );
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            margin-bottom:5px;
        ">
            🛡️ AI Security
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div style="
            color:#8b95a7;
            margin-bottom:25px;
        ">
            Fraud Detection Center
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### System Status")


    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if health_response.status_code == 200:

            st.success("● API ONLINE")

        else:

            st.error("● API ERROR")

    except:

        st.error("● API OFFLINE")


    st.divider()


    st.markdown("### Technology")

    st.write("🐍 Python")

    st.write("🧠 Scikit-learn")

    st.write("🌲 Random Forest")

    st.write("⚡ FastAPI")

    st.write("📊 Streamlit")

    st.write("📈 Plotly")


    st.divider()


    st.markdown("### Session")

    st.write(
        f"Predictions: "
        f"**{len(st.session_state.prediction_history)}**"
    )


    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.rerun()


    st.divider()


    st.caption(
        "AI Fraud Detection Center v2.0"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ AI Fraud Detection Center'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered financial transaction monitoring'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CURRENT SESSION STATISTICS
# ============================================================

history = st.session_state.prediction_history


total_transactions = len(history)

fraud_transactions = sum(
    1
    for item in history
    if item["is_fraud"]
)

safe_transactions = (
    total_transactions -
    fraud_transactions
)


if total_transactions > 0:

    fraud_rate = (
        fraud_transactions /
        total_transactions
    ) * 100

else:

    fraud_rate = 0


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.markdown(
        """
        <div class="metric-card">

        <div class="metric-title">
        AI MODEL
        </div>

        <div class="metric-value">
        Random Forest
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="metric-card">

        <div class="metric-title">
        MODEL ACCURACY
        </div>

        <div class="metric-value">
        100%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        ANALYZED
        </div>

        <div class="metric-value">
        {total_transactions}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        FRAUD DETECTED
        </div>

        <div class="metric-danger">
        {fraud_transactions}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col5:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
        FRAUD RATE
        </div>

        <div class="metric-value">
        {fraud_rate:.1f}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TRANSACTION ANALYZER
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔍 Transaction Analyzer'
    '</div>',
    unsafe_allow_html=True
)


left, right = st.columns(2)


# ============================================================
# LEFT INPUTS
# ============================================================

with left:

    amount = st.number_input(
        "💳 Transaction Amount",
        min_value=1.0,
        max_value=100000.0,
        value=1750.0,
        step=10.0
    )


    transaction_hour = st.slider(
        "🕐 Transaction Hour",
        min_value=0,
        max_value=23,
        value=2
    )


    distance_from_home = st.number_input(
        "📍 Distance From Home (km)",
        min_value=0.0,
        max_value=5000.0,
        value=420.0,
        step=1.0
    )


# ============================================================
# RIGHT INPUTS
# ============================================================

with right:

    merchant_risk = st.slider(
        "🏪 Merchant Risk",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01
    )


    device_trusted = st.selectbox(
        "💻 Device Trusted",
        options=[1, 0],
        format_func=lambda x:
            "Trusted Device"
            if x == 1
            else "Unknown Device"
    )


    previous_transactions = st.number_input(
        "🔄 Previous Transactions",
        min_value=0,
        max_value=1000,
        value=2,
        step=1
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")


analyze = st.button(
    "🚨 ANALYZE TRANSACTION",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    transaction = {

        "amount": amount,

        "transaction_hour":
            transaction_hour,

        "distance_from_home":
            distance_from_home,

        "merchant_risk":
            merchant_risk,

        "device_trusted":
            device_trusted,

        "previous_transactions":
            previous_transactions
    }


    try:

        with st.spinner(
            "🧠 AI model analyzing transaction..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=10
            )


        if response.status_code == 200:

            result = response.json()


            is_fraud = result[
                "is_fraud"
            ]

            probability = result[
                "fraud_probability_percent"
            ]

            risk_level = result[
                "risk_level"
            ]


            # ====================================================
            # SAVE RESULT
            # ====================================================

            history_item = {

                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "amount":
                    amount,

                "transaction_hour":
                    transaction_hour,

                "distance_from_home":
                    distance_from_home,

                "merchant_risk":
                    merchant_risk,

                "device_trusted":
                    device_trusted,

                "previous_transactions":
                    previous_transactions,

                "fraud_probability":
                    probability,

                "risk_level":
                    risk_level,

                "is_fraud":
                    bool(is_fraud)
            }


            st.session_state.prediction_history.append(
                history_item
            )


            # ====================================================
            # RESULT
            # ====================================================

            st.markdown(
                '<div class="section-title">'
                '📊 AI Analysis Result'
                '</div>',
                unsafe_allow_html=True
            )


            if is_fraud:

                st.markdown(
                    f"""
                    <div class="fraud-box">

                    <div style="
                        font-size:55px;
                    ">
                        🚨
                    </div>

                    <div style="
                        font-size:30px;
                        font-weight:800;
                        color:#ef4444;
                    ">
                        FRAUD DETECTED
                    </div>

                    <br>

                    <div style="
                        color:#d1d5db;
                        font-size:16px;
                    ">
                        Fraud Probability
                    </div>

                    <div class="risk-high">
                        {probability:.2f}%
                    </div>

                    <div style="
                        color:#d1d5db;
                        margin-top:10px;
                    ">
                        Risk Level:
                        <b>{risk_level}</b>
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="safe-box">

                    <div style="
                        font-size:55px;
                    ">
                        🟢
                    </div>

                    <div style="
                        font-size:30px;
                        font-weight:800;
                        color:#22c55e;
                    ">
                        TRANSACTION APPEARS SAFE
                    </div>

                    <br>

                    <div style="
                        color:#d1d5db;
                        font-size:16px;
                    ">
                        Fraud Probability
                    </div>

                    <div class="risk-low">
                        {probability:.2f}%
                    </div>

                    <div style="
                        color:#d1d5db;
                        margin-top:10px;
                    ">
                        Risk Level:
                        <b>{risk_level}</b>
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # TRANSACTION SUMMARY
            # ====================================================

            st.markdown(
                '<div class="section-title">'
                '📋 Transaction Summary'
                '</div>',
                unsafe_allow_html=True
            )


            summary_data = {

                "Feature": [

                    "Amount",

                    "Transaction Hour",

                    "Distance From Home",

                    "Merchant Risk",

                    "Device",

                    "Previous Transactions",

                    "Fraud Probability",

                    "Risk Level"
                ],

                "Value": [

                    f"${amount:,.2f}",

                    f"{transaction_hour}:00",

                    f"{distance_from_home:.2f} km",

                    f"{merchant_risk:.2f}",

                    "Trusted"
                    if device_trusted == 1
                    else "Unknown",

                    str(previous_transactions),

                    f"{probability:.2f}%",

                    risk_level
                ]
            }


            summary_df = pd.DataFrame(
                summary_data
            )


            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.error(
                f"API Error: {response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI. "
            "Make sure Uvicorn is running."
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )


# ============================================================
# ANALYTICS SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 AI Monitoring Analytics'
    '</div>',
    unsafe_allow_html=True
)


if len(history) == 0:

    st.info(
        "Analyze some transactions to populate "
        "the monitoring analytics."
    )

else:

    history_df = pd.DataFrame(history)


    chart_col1, chart_col2 = st.columns(2)


    # ========================================================
    # FRAUD PROBABILITY CHART
    # ========================================================

    with chart_col1:

        st.markdown(
            "#### 📈 Fraud Probability"
        )


        chart_df = history_df.copy()

        chart_df["Transaction"] = range(
            1,
            len(chart_df) + 1
        )


        fig = px.line(
            chart_df,
            x="Transaction",
            y="fraud_probability",
            markers=True,
            labels={
                "fraud_probability":
                    "Fraud Probability (%)"
            }
        )


        fig.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="white"
            ),
            yaxis=dict(
                range=[0, 100]
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    with chart_col2:

        st.markdown(
            "#### 🎯 Risk Distribution"
        )


        risk_counts = (
            history_df["risk_level"]
            .value_counts()
            .reset_index()
        )


        risk_counts.columns = [
            "Risk Level",
            "Count"
        ]


        fig2 = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.55
        )


        fig2.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="white"
            )
        )


        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        "#### 🧠 Model Feature Importance"
    )


    feature_names = [

        "Distance From Home",

        "Merchant Risk",

        "Transaction Hour",

        "Amount",

        "Previous Transactions",

        "Device Trusted"
    ]


    feature_values = [

        26.96,

        25.33,

        23.66,

        23.51,

        0.46,

        0.08
    ]


    importance_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            feature_values
    })


    importance_df = importance_df.sort_values(
        "Importance"
    )


    fig3 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance"
    )


    fig3.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig3.update_layout(
        height=380,
        margin=dict(
            l=20,
            r=70,
            t=30,
            b=20
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white"
        )
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # ========================================================
    # TRANSACTION HISTORY
    # ========================================================

    st.markdown(
        "#### 📋 Recent Transaction History"
    )


    display_df = history_df.copy()


    display_df = display_df[
        [
            "timestamp",

            "amount",

            "transaction_hour",

            "merchant_risk",

            "fraud_probability",

            "risk_level",

            "is_fraud"
        ]
    ]


    display_df.columns = [

        "Timestamp",

        "Amount",

        "Hour",

        "Merchant Risk",

        "Fraud Probability",

        "Risk",

        "Fraud"
    ]


    display_df["Fraud"] = display_df[
        "Fraud"
    ].map({

        True: "🚨 YES",

        False: "🟢 NO"
    })


    st.dataframe(
        display_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    csv_data = history_df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="fraud_prediction_history.csv",
        mime="text/csv"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 Model Information'
    '</div>',
    unsafe_allow_html=True
)


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.info(
        """
        **Algorithm**

        Random Forest Classifier
        """
    )


with model_col2:

    st.info(
        """
        **ROC-AUC**

        1.0000
        """
    )


with model_col3:

    st.info(
        """
        **Training Features**

        6 transaction features
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        padding:20px;
    ">

    🛡️ AI Fraud Detection Center

    <br><br>

    Machine Learning • Random Forest • FastAPI • Streamlit • Plotly

    <br>

    <small>
    Educational / demonstration system
    </small>

    </div>
    """,
    unsafe_allow_html=True
)