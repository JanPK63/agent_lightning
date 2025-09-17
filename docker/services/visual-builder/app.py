import streamlit as st
import requests
import os

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8001')

st.set_page_config(page_title="Visual Builder", layout="wide")
st.title("🎨 Visual Code Builder")

# Simple visual builder interface
col1, col2 = st.columns(2)

with col1:
    st.header("Components")
    component_type = st.selectbox("Select Component", ["API", "Database", "Service", "Workflow"])
    
    if component_type == "API":
        st.text_input("Endpoint")
        st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
    elif component_type == "Database":
        st.selectbox("Type", ["PostgreSQL", "Redis", "InfluxDB"])
        st.text_input("Connection String")

with col2:
    st.header("Preview")
    st.code("# Generated code will appear here", language="python")

if st.button("Generate Code"):
    st.success("Code generated successfully!")