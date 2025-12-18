import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Parse API URL from command line arguments
# Streamlit command: streamlit run dashboard.py -- --api-url http://localhost:8000
API_BASE_URL = "http://localhost:8000"  # Default value

# Check for --api-url argument in sys.argv
if "--api-url" in sys.argv:
    try:
        api_url_index = sys.argv.index("--api-url")
        if api_url_index + 1 < len(sys.argv):
            API_BASE_URL = sys.argv[api_url_index + 1]
    except (ValueError, IndexError):
        pass

# Also check environment variable as fallback
API_BASE_URL = os.getenv("API_BASE_URL", API_BASE_URL)

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Title
st.title("🚨 Fraud Detection Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Beneficiaries", "Risk Analysis", "Explain Risk"])

# Display API URL in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**API Configuration**")
st.sidebar.info(f"API URL: `{API_BASE_URL}`")

# Helper function to fetch data
@st.cache_data(ttl=60)
def fetch_results():
    """Fetch all results from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/results")
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data["data"])
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot connect to API at `{API_BASE_URL}`. Make sure the API server is running:")
        st.code("uvicorn src.api.main:app --reload", language="bash")
        st.info(f"💡 Tip: You can specify a different API URL using: `streamlit run dashboard.py -- --api-url <URL>`")
        return None

def fetch_explanation(beneficiary_id):
    """Fetch explanation for a beneficiary."""
    try:
        response = requests.get(f"{API_BASE_URL}/explain/{beneficiary_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Overview Page
if page == "Overview":
    st.header("📊 Overview")
    
    df = fetch_results()
    
    if df is not None and not df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Beneficiaries", len(df))
        
        with col2:
            high_risk = (df["risk_level"] == "HIGH").sum()
            st.metric("High Risk", high_risk, delta=f"{high_risk/len(df)*100:.1f}%")
        
        with col3:
            medium_risk = (df["risk_level"] == "MEDIUM").sum()
            st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/len(df)*100:.1f}%")
        
        with col4:
            low_risk = (df["risk_level"] == "LOW").sum()
            st.metric("Low Risk", low_risk, delta=f"{low_risk/len(df)*100:.1f}%")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            risk_counts = df["risk_level"].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={"HIGH": "#FF4444", "MEDIUM": "#FFAA00", "LOW": "#00AA00"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Risk score distribution
            fig_hist = px.histogram(
                df,
                x="final_score",
                nbins=30,
                title="Final Risk Score Distribution",
                labels={"final_score": "Risk Score", "count": "Count"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Risk by state
        st.subheader("Risk by State")
        risk_by_state = df.groupby(["state", "risk_level"]).size().reset_index(name="count")
        fig_bar = px.bar(
            risk_by_state,
            x="state",
            y="count",
            color="risk_level",
            title="Risk Distribution by State",
            color_discrete_map={"HIGH": "#FF4444", "MEDIUM": "#FFAA00", "LOW": "#00AA00"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Beneficiaries Page
elif page == "Beneficiaries":
    st.header("👥 Beneficiaries")
    
    df = fetch_results()
    
    if df is not None and not df.empty:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"]
            )
        
        with col2:
            state_filter = st.multiselect(
                "Filter by State",
                options=sorted(df["state"].unique()),
                default=sorted(df["state"].unique())
            )
        
        # Apply filters
        filtered_df = df[
            (df["risk_level"].isin(risk_filter)) &
            (df["state"].isin(state_filter))
        ]
        
        st.metric("Filtered Results", len(filtered_df))
        
        # Display table
        display_cols = [
            "beneficiary_id", "age", "state", "scheme_name",
            "income", "amount_requested", "risk_level", "final_score"
        ]
        
        st.dataframe(
            filtered_df[display_cols].style.background_gradient(
                subset=["final_score"],
                cmap="RdYlGn_r"
            ),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_beneficiaries.csv",
            mime="text/csv"
        )

# Risk Analysis Page
elif page == "Risk Analysis":
    st.header("🔍 Risk Analysis")
    
    df = fetch_results()
    
    if df is not None and not df.empty:
        # Scatter plot: Income vs Amount Requested
        st.subheader("Income vs Amount Requested")
        fig_scatter = px.scatter(
            df,
            x="income",
            y="amount_requested",
            color="risk_level",
            size="final_score",
            hover_data=["beneficiary_id", "state", "scheme_name"],
            title="Risk Visualization: Income vs Amount Requested",
            color_discrete_map={"HIGH": "#FF4444", "MEDIUM": "#FFAA00", "LOW": "#00AA00"},
            labels={"income": "Income (₹)", "amount_requested": "Amount Requested (₹)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Rule score vs ML anomaly
        st.subheader("Rule Score vs ML Anomaly Detection")
        if "anomaly_score" in df.columns:
            df_plot = df.copy()
            df_plot["is_anomaly"] = df_plot["anomaly_score"] == -1
            fig_box = px.box(
                df_plot,
                x="is_anomaly",
                y="rule_score",
                color="risk_level",
                title="Rule Scores by Anomaly Detection",
                color_discrete_map={"HIGH": "#FF4444", "MEDIUM": "#FFAA00", "LOW": "#00AA00"}
            )
            st.plotly_chart(fig_box, use_container_width=True)

# Explain Risk Page
elif page == "Explain Risk":
    st.header("🧠 Explain Risk")
    
    df = fetch_results()
    
    if df is not None and not df.empty:
        # Select beneficiary
        beneficiary_ids = sorted(df["beneficiary_id"].unique())
        selected_id = st.selectbox("Select Beneficiary ID", beneficiary_ids)
        
        if selected_id:
            # Fetch explanation
            explanation = fetch_explanation(selected_id)
            
            if explanation and "error" not in explanation:
                # Display explanation
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Risk Level", explanation["risk_level"])
                    st.metric("Final Score", explanation["final_score"])
                    st.metric("Rule Score", explanation["rule_score"])
                    st.metric("ML Anomaly", explanation["anomaly_score"])
                
                with col2:
                    st.subheader("Beneficiary Details")
                    details = explanation["details"]
                    st.json(details)
                
                st.markdown("---")
                st.subheader("🚨 Risk Factors")
                
                reasons = explanation["reasons"]
                for i, reason in enumerate(reasons, 1):
                    if "No specific risk factors" in reason:
                        st.success(f"{i}. {reason}")
                    else:
                        st.warning(f"{i}. {reason}")
            else:
                st.error("Could not fetch explanation. Check API connection.")

# Footer
st.markdown("---")
st.markdown("**Fraud Detection System** | Built with Streamlit & FastAPI")


