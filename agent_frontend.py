import streamlit as st
import requests
import json

st.set_page_config(page_title="Agent Lightning Training", layout="wide")

st.title("⚡ Agent Lightning - AI Agent Training Platform")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Agent Training", "Active Agents", "Training History", "System Status"])

if page == "Agent Training":
    st.header("🤖 Create & Train New Agent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Agent Configuration")
        agent_name = st.text_input("Agent Name", "MyAgent")
        agent_type = st.selectbox("Agent Type", ["LangChain", "OpenAI", "AutoGen", "CrewAI", "Custom"])
        framework = st.selectbox("Framework", ["Reinforcement Learning", "Prompt Optimization", "Fine-tuning"])
        
        st.subheader("Training Dataset")
        dataset = st.selectbox("Dataset", ["Calc-X", "Spider SQL", "Custom Upload"])
        if dataset == "Custom Upload":
            uploaded_file = st.file_uploader("Upload training data", type=['json', 'csv'])
    
    with col2:
        st.subheader("Training Parameters")
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])
        max_episodes = st.number_input("Max Episodes", 100, 10000, 1000)
        
        st.subheader("Reward Configuration")
        reward_type = st.selectbox("Reward Type", ["Task Success", "Code Quality", "Custom"])
        
    if st.button("🚀 Start Training", type="primary"):
        training_config = {
            "name": agent_name,
            "type": agent_type,
            "framework": framework,
            "dataset": dataset,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_episodes": max_episodes,
            "reward_type": reward_type
        }
        
        st.success(f"Training started for agent: {agent_name}")
        st.json(training_config)

elif page == "Active Agents":
    st.header("🔄 Active Training Sessions")
    
    # Mock active agents data
    agents = [
        {"name": "SQLAgent", "status": "Training", "progress": 65, "episode": 650, "reward": 0.85},
        {"name": "CodeAgent", "status": "Evaluating", "progress": 100, "episode": 1000, "reward": 0.92},
        {"name": "ChatAgent", "status": "Paused", "progress": 30, "episode": 300, "reward": 0.67}
    ]
    
    for agent in agents:
        with st.expander(f"{agent['name']} - {agent['status']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Progress", f"{agent['progress']}%")
                st.progress(agent['progress'] / 100)
            
            with col2:
                st.metric("Episode", agent['episode'])
                st.metric("Reward", agent['reward'])
            
            with col3:
                if st.button(f"Stop {agent['name']}", key=f"stop_{agent['name']}"):
                    st.warning(f"Stopping {agent['name']}")
                if st.button(f"View Logs {agent['name']}", key=f"logs_{agent['name']}"):
                    st.info(f"Opening logs for {agent['name']}")

elif page == "Training History":
    st.header("📊 Training History & Results")
    
    # Mock training history
    import pandas as pd
    import numpy as np
    
    history_data = {
        'Agent': ['SQLAgent', 'CodeAgent', 'ChatAgent', 'MathAgent'],
        'Training Time': ['2h 15m', '4h 32m', '1h 45m', '3h 20m'],
        'Final Reward': [0.92, 0.88, 0.75, 0.95],
        'Episodes': [1000, 1500, 800, 1200],
        'Status': ['Completed', 'Completed', 'Failed', 'Completed']
    }
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)
    
    # Training metrics chart
    st.subheader("Training Progress")
    chart_data = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['Reward', 'Loss', 'Accuracy']
    )
    st.line_chart(chart_data)

elif page == "System Status":
    st.header("🖥️ System Status & Resources")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Agents", "3", "+1")
    
    with col2:
        st.metric("GPU Usage", "75%", "+5%")
    
    with col3:
        st.metric("Memory Usage", "8.2GB", "+0.5GB")
    
    with col4:
        st.metric("Training Queue", "2", "-1")
    
    # System resources
    st.subheader("Resource Monitoring")
    
    # Mock resource data
    resource_data = pd.DataFrame({
        'Time': pd.date_range('2024-01-01', periods=24, freq='H'),
        'CPU': np.random.randint(20, 80, 24),
        'Memory': np.random.randint(40, 90, 24),
        'GPU': np.random.randint(30, 95, 24)
    })
    
    st.line_chart(resource_data.set_index('Time'))
    
    # Service status
    st.subheader("Service Health")
    services = [
        {"name": "Training Server", "status": "✅ Healthy", "uptime": "99.9%"},
        {"name": "Database", "status": "✅ Healthy", "uptime": "99.8%"},
        {"name": "Redis Cache", "status": "✅ Healthy", "uptime": "99.9%"},
        {"name": "Message Queue", "status": "⚠️ Warning", "uptime": "98.5%"}
    ]
    
    for service in services:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(service["name"])
        with col2:
            st.write(service["status"])
        with col3:
            st.write(service["uptime"])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Agent Lightning v1.0**")
st.sidebar.markdown("🔗 [Documentation](https://github.com/microsoft/agent-lightning)")
st.sidebar.markdown("💬 [Discord](https://discord.gg/RYk7CdvDR7)")