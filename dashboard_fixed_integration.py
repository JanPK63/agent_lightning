#!/usr/bin/env python3
"""
Dashboard Integration with Fixed Agent System
Updates the monitoring dashboard to work with the fixed agents on port 8888
"""

import streamlit as st
import requests
import json
from datetime import datetime

def render_fixed_task_assignment():
    """Render task assignment interface for fixed agent system"""
    st.header("🎯 Task Assignment - Fixed Agent System")
    
    # Connection status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🎯 Connect to Fixed Agent System (Port 8888) - Agents that actually work!")
    with col2:
        if st.button("🔌 Connect"):
            try:
                response = requests.get("http://localhost:8888/health", timeout=5)
                if response.status_code == 200:
                    st.session_state.fixed_connected = True
                    st.success("✅ Connected!")
                else:
                    st.error("❌ Not responding")
            except:
                st.error("❌ Start: python3 fixed_agent_api.py")
    
    if st.session_state.get('fixed_connected', False):
        # Get available agents
        try:
            agents_response = requests.get("http://localhost:8888/agents", timeout=5)
            agents = agents_response.json().get("agents", []) if agents_response.status_code == 200 else []
        except:
            agents = []
        
        # Task form
        st.subheader("📝 Execute Task")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task = st.text_area("Task Description", 
                               placeholder="Create a Python function to sort data...",
                               height=100)
            
            if agents:
                agent_options = ["auto"] + [a["id"] for a in agents]
                agent_labels = ["Auto-select"] + [f"{a['name']}" for a in agents]
                agent = st.selectbox("Agent", agent_options, 
                                   format_func=lambda x: agent_labels[agent_options.index(x)])
            else:
                agent = st.selectbox("Agent", ["auto", "full_stack_developer", "data_scientist"])
        
        with col2:
            st.markdown("**Available Agents:**")
            if agents:
                for a in agents:
                    icon = {"full_stack_developer": "💻", "data_scientist": "📊", 
                           "security_expert": "🔒", "devops_engineer": "🚀"}.get(a["id"], "🤖")
                    st.markdown(f"- {icon} **{a['name']}**")
            
            st.markdown("**✨ What's Different:**")
            st.success("✅ Agents execute tasks")
            st.success("✅ Real implementations")
            st.success("✅ Working solutions")
        
        # Execute button
        if st.button("🚀 Execute Task", type="primary", disabled=not task):
            with st.spinner("Agent executing..."):
                try:
                    response = requests.post("http://localhost:8888/execute",
                                           json={"task": task, "agent_id": agent},
                                           timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store in history
                        if 'task_history' not in st.session_state:
                            st.session_state.task_history = []
                        
                        st.session_state.task_history.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "task": task[:50] + "..." if len(task) > 50 else task,
                            "agent": result.get("agent_name", agent),
                            "result": result.get("result", ""),
                            "status": result.get("status", "completed"),
                            "execution_time": result.get("execution_time_seconds", 0)
                        })
                        
                        # Display result
                        if result.get("status") == "completed":
                            st.success("✅ Task completed!")
                            st.markdown("### 📋 Result:")
                            
                            result_text = result.get("result", "")
                            if len(result_text) > 1000:
                                st.text_area("Result", result_text, height=300)
                            else:
                                st.write(result_text)
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Agent", result.get("agent_name", "Unknown"))
                            with col2:
                                st.metric("Time", f"{result.get('execution_time_seconds', 0):.2f}s")
                            with col3:
                                st.metric("Status", result.get("status", "Unknown"))
                        else:
                            st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ Request failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Quick examples
        st.subheader("💡 Quick Examples")
        examples = [
            ("Create a Python function to calculate fibonacci", "full_stack_developer"),
            ("Build a REST API for user management", "full_stack_developer"),
            ("Analyze sample data and provide insights", "data_scientist"),
            ("Review code for security issues", "security_expert")
        ]
        
        cols = st.columns(2)
        for i, (example_task, example_agent) in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"🚀 {example_task[:30]}...", key=f"ex_{i}"):
                    with st.spinner(f"Running example..."):
                        try:
                            response = requests.post("http://localhost:8888/execute",
                                                   json={"task": example_task, "agent_id": example_agent},
                                                   timeout=60)
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ Completed by {result.get('agent_name', example_agent)}")
                                
                                # Add to history
                                if 'task_history' not in st.session_state:
                                    st.session_state.task_history = []
                                
                                st.session_state.task_history.append({
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "task": example_task,
                                    "agent": example_agent,
                                    "result": result.get("result", ""),
                                    "status": result.get("status", "completed"),
                                    "execution_time": result.get("execution_time_seconds", 0)
                                })
                                
                                # Show preview
                                preview = result.get("result", "")[:200]
                                st.info(f"Preview: {preview}...")
                            else:
                                st.error("Example failed")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Task history
        st.subheader("📜 Recent Tasks")
        if st.session_state.get('task_history', []):
            for i, task in enumerate(reversed(st.session_state.task_history[-5:])):
                status_emoji = {"completed": "✅", "failed": "❌"}.get(task.get("status", "completed"), "✅")
                
                with st.expander(f"{status_emoji} {task['timestamp']} - {task['task']} ({task['agent']})"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Task:** {task['task']}")
                        st.write(f"**Agent:** {task['agent']}")
                    with col2:
                        st.write(f"**Time:** {task['timestamp']}")
                        if 'execution_time' in task:
                            st.write(f"**Duration:** {task['execution_time']:.2f}s")
                    
                    st.write("**Result:**")
                    result_text = task['result']
                    if len(result_text) > 500:
                        st.text_area("Result", result_text, height=200, key=f"hist_{i}")
                    else:
                        st.write(result_text)
        else:
            st.info("No tasks executed yet. Try the examples above!")
    
    else:
        st.warning("⚠️ Not connected. Click 'Connect' and ensure fixed system is running.")
        st.code("""
# Start the fixed system:
python3 fixed_agent_api.py

# Test it works:
python3 test_fixed_agents.py
        """)

if __name__ == "__main__":
    st.set_page_config(page_title="Fixed Agent Dashboard", page_icon="⚡", layout="wide")
    render_fixed_task_assignment()