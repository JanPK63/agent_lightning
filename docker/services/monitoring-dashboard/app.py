import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
st.title("📊 Agent Lightning Monitoring")

# Metrics display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Active Agents", "12", "2")
with col2:
    st.metric("Training Sessions", "5", "1")
with col3:
    st.metric("API Requests/min", "150", "25")
with col4:
    st.metric("System Health", "98%", "1%")

# Charts
st.subheader("Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    # Sample data for demo
    df = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=24, freq='H'),
        'cpu_usage': [20 + i*2 + (i%3)*5 for i in range(24)]
    })
    fig = px.line(df, x='time', y='cpu_usage', title='CPU Usage Over Time')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    df2 = pd.DataFrame({
        'agent': ['Agent A', 'Agent B', 'Agent C', 'Agent D'],
        'performance': [85, 92, 78, 88]
    })
    fig2 = px.bar(df2, x='agent', y='performance', title='Agent Performance')
    st.plotly_chart(fig2, use_container_width=True)