import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Agent Lightning Dashboard", layout="wide")

st.title("🚀 Agent Lightning Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("API Status", "Healthy", "✅")
    
with col2:
    st.metric("Active Agents", "3", "+1")
    
with col3:
    st.metric("Training Jobs", "2", "0")

st.subheader("System Overview")

# Mock data for demonstration
data = {
    'Service': ['API', 'Database', 'Redis', 'RabbitMQ'],
    'Status': ['Healthy', 'Healthy', 'Healthy', 'Healthy'],
    'Uptime': ['99.9%', '99.8%', '99.9%', '99.7%']
}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

st.subheader("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start Training"):
        st.success("Training job started!")

with col2:
    if st.button("View Logs"):
        st.info("Redirecting to logs...")

with col3:
    if st.button("System Health"):
        st.info("All systems operational")