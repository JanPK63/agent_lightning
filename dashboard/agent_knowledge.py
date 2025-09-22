"""
Agent knowledge management interface for Agent Lightning Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional

from .models import DashboardConfig


class AgentKnowledgeInterface:
    """Handles agent knowledge management and training"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def render_knowledge_management(self):
        """Render agent knowledge management interface"""
        st.header("🧠 Agent Knowledge Management")

        # Import knowledge and agent managers
        try:
            from agent_config import AgentConfigManager
            from knowledge_manager import KnowledgeManager

            config_manager = AgentConfigManager()
            knowledge_manager = KnowledgeManager()
        except ImportError:
            st.error("Knowledge management system not available. Run setup_agents.py first.")
            return

        # Sidebar for agent selection
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Select Agent")

            # Get list of configured agents
            agent_list = config_manager.list_agents()
            if not agent_list:
                st.warning("No agents configured yet")
                if st.button("Setup Default Agents"):
                    import subprocess
                    subprocess.run(["/Users/jankootstra/miniforge3/bin/python", "setup_agents.py"])
                    st.success("Agents configured! Refresh the page.")
            else:
                selected_agent = st.selectbox(
                    "Agent",
                    options=agent_list,
                    help="Select an agent to manage its knowledge"
                )

                # Show agent details
                if selected_agent:
                    agent = config_manager.get_agent(selected_agent)
                    if agent:
                        st.markdown(f"**Role:** {agent.role.value}")
                        st.markdown(f"**Model:** {agent.model}")

                        # Show capabilities
                        capabilities = [k.replace("can_", "").replace("_", " ").title()
                                      for k, v in agent.capabilities.__dict__.items() if v]
                        st.markdown(f"**Capabilities:** {', '.join(capabilities)}")

        with col2:
            if agent_list and selected_agent:
                # Knowledge management tabs
                kb_tab1, kb_tab2, kb_tab3, kb_tab4, kb_tab5 = st.tabs([
                    "📚 View Knowledge",
                    "➕ Add Knowledge",
                    "🎓 Train Agent",
                    "🔍 Search Knowledge",
                    "📊 Statistics"
                ])

                with kb_tab1:
                    self._render_knowledge_view(knowledge_manager, selected_agent)

                with kb_tab2:
                    self._render_add_knowledge(knowledge_manager, selected_agent)

                with kb_tab3:
                    self._render_agent_training(knowledge_manager, selected_agent)

                with kb_tab4:
                    self._render_knowledge_search(knowledge_manager, selected_agent)

                with kb_tab5:
                    self._render_knowledge_statistics(knowledge_manager, selected_agent)

    def _render_knowledge_view(self, knowledge_manager, selected_agent):
        """Render knowledge view tab"""
        st.subheader("📚 Knowledge Base")

        # Get knowledge items
        items = knowledge_manager.knowledge_bases.get(selected_agent, [])

        if items:
            # Group by category
            categories = {}
            for item in items:
                if item.category not in categories:
                    categories[item.category] = []
                categories[item.category].append(item)

            # Display by category
            for category, cat_items in categories.items():
                with st.expander(f"{category} ({len(cat_items)} items)"):
                    for item in cat_items[:5]:  # Show first 5
                        st.markdown(f"**Source:** {item.source}")
                        st.code(item.content[:200] + "..." if len(item.content) > 200 else item.content)
                        st.caption(f"Usage: {item.usage_count} times | Relevance: {item.relevance_score:.2f}")
        else:
            st.info("No knowledge items yet. Add some in the 'Add Knowledge' tab!")

    def _render_add_knowledge(self, knowledge_manager, selected_agent):
        """Render add knowledge tab"""
        st.subheader("➕ Add New Knowledge")

        # Add knowledge form
        category = st.selectbox(
            "Category",
            options=[
                "technical_documentation",
                "code_examples",
                "best_practices",
                "troubleshooting",
                "architecture_patterns",
                "api_references",
                "tutorials",
                "project_specific",
                "domain_knowledge"
            ]
        )

        content = st.text_area(
            "Knowledge Content",
            placeholder="Enter the knowledge item here...",
            height=200
        )

        source = st.text_input(
            "Source",
            placeholder="e.g., official_docs, experience, stackoverflow"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Knowledge", type="primary", disabled=not content):
                item = knowledge_manager.add_knowledge(
                    selected_agent,
                    category,
                    content,
                    source or "manual"
                )
                st.success(f"✅ Added knowledge item: {item.id}")
                st.rerun()

        with col2:
            # Upload file option
            uploaded_file = st.file_uploader(
                "Or upload a file",
                type=["txt", "md", "json", "py", "js", "java", "cpp"]
            )

            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                if st.button("Import from file"):
                    item = knowledge_manager.add_knowledge(
                        selected_agent,
                        "project_specific",
                        content,
                        f"uploaded:{uploaded_file.name}"
                    )
                    st.success(f"✅ Imported knowledge from {uploaded_file.name}")
                    st.rerun()

    def _render_agent_training(self, knowledge_manager, selected_agent):
        """Render agent training tab"""
        st.subheader("🎓 Train Agent with Knowledge")

        # Import knowledge trainer
        try:
            from knowledge_trainer import KnowledgeTrainer
            trainer = KnowledgeTrainer()
        except ImportError:
            st.error("Knowledge trainer not available. Please ensure knowledge_trainer.py is installed.")
            trainer = None

        if trainer:
            # Get consumption stats
            stats = trainer.get_consumption_stats(selected_agent)

            # Show current status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Knowledge", stats["total_knowledge_items"])
            with col2:
                st.metric("New Items", stats["new_knowledge_available"])
            with col3:
                last_consumption = stats.get("last_consumption")
                if last_consumption:
                    from datetime import datetime
                    time_ago = (datetime.now() - last_consumption).total_seconds() / 3600
                    st.metric("Last Trained", f"{time_ago:.1f}h ago")
                else:
                    st.metric("Last Trained", "Never")

            st.markdown("---")

            # Training options
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔄 Consume New Knowledge")
                st.markdown("Make the agent consume and integrate new knowledge items into its configuration.")

                force_all = st.checkbox("Process all knowledge (not just new)", value=False)

                if st.button("🧠 Consume Knowledge", type="primary"):
                    with st.spinner("Processing knowledge..."):
                        result = trainer.consume_knowledge(selected_agent, force_all)

                        if result.knowledge_integrated > 0:
                            st.success(f"✅ Successfully consumed {result.knowledge_consumed} items, integrated {result.knowledge_integrated}")
                            if result.improvements:
                                st.info("Improvements: " + ", ".join(result.improvements))
                        elif result.knowledge_consumed == 0:
                            st.info("No new knowledge to consume")
                        else:
                            st.warning("Knowledge consumed but not integrated")

                        if result.errors:
                            st.error("Errors: " + ", ".join(result.errors))

            with col2:
                st.markdown("### 🎯 Test Training")
                st.markdown("Test the agent with sample queries to verify knowledge integration.")

                test_queries = st.text_area(
                    "Test Queries (one per line)",
                    placeholder="How do I optimize a database query?\nWhat's the best caching strategy?\nHow to handle authentication?",
                    height=100
                )

                if st.button("🧪 Run Training Test"):
                    if test_queries:
                        queries = [q.strip() for q in test_queries.split('\n') if q.strip()]
                        with st.spinner(f"Testing {len(queries)} queries..."):
                            try:
                                results = asyncio.run(trainer.active_training_session(selected_agent, queries))

                                st.success(f"Tested {results['queries_tested']} queries")
                                st.info(f"Knowledge applied in {results['knowledge_applied']} responses")

                                # Show responses
                                for i, response in enumerate(results.get('responses', [])):
                                    with st.expander(f"Query {i+1}: {response.get('query', '')}"):
                                        if 'error' in response:
                                            st.error(response['error'])
                                        else:
                                            st.markdown(f"**Knowledge Used:** {response.get('knowledge_used', 0)} items")
                                            # Show full response or at least 2000 chars
                                            response_text = response.get('response', '')
                                            if len(response_text) > 2000:
                                                st.markdown(response_text[:2000] + "...")
                                                with st.expander("Show full response"):
                                                    st.markdown(response_text)
                                            else:
                                                st.markdown(response_text)
                            except Exception as e:
                                st.error(f"Training test failed: {str(e)}")
                    else:
                        st.warning("Please enter test queries")

            # Knowledge by category chart
            if stats.get("knowledge_by_category"):
                st.markdown("### 📊 Knowledge Distribution")
                df = pd.DataFrame(
                    list(stats["knowledge_by_category"].items()),
                    columns=["Category", "Count"]
                )
                st.bar_chart(df.set_index("Category"))

            # Auto-train all agents button
            st.markdown("---")
            st.markdown("### 🚀 Bulk Operations")
            if st.button("🔄 Auto-Train All Agents"):
                with st.spinner("Training all agents..."):
                    results = trainer.auto_consume_all_agents()

                    success_count = sum(1 for r in results.values() if r.knowledge_integrated > 0)
                    st.success(f"✅ Trained {success_count} agents successfully")

                    for agent_name, result in results.items():
                        if result.knowledge_integrated > 0:
                            st.info(f"{agent_name}: Consumed {result.knowledge_consumed}, integrated {result.knowledge_integrated}")
                        elif result.errors:
                            st.warning(f"{agent_name}: {', '.join(result.errors)}")

    def _render_knowledge_search(self, knowledge_manager, selected_agent):
        """Render knowledge search tab"""
        st.subheader("🔍 Search Knowledge")

        query = st.text_input(
            "Search Query",
            placeholder="Enter keywords to search..."
        )

        search_category = st.selectbox(
            "Filter by Category (optional)",
            options=["All"] + knowledge_manager.categories
        )

        if st.button("Search", disabled=not query):
            results = knowledge_manager.search_knowledge(
                selected_agent,
                query,
                category=search_category if search_category != "All" else None,
                limit=10
            )

            if results:
                st.success(f"Found {len(results)} matching items:")
                for item in results:
                    with st.expander(f"[{item.category}] {item.source}"):
                        st.code(item.content)
                        st.caption(f"Relevance: {item.relevance_score:.2f} | Used: {item.usage_count} times")
            else:
                st.info("No matching knowledge items found")

    def _render_knowledge_statistics(self, knowledge_manager, selected_agent):
        """Render knowledge statistics tab"""
        st.subheader("📊 Knowledge Statistics")

        stats = knowledge_manager.get_statistics(selected_agent)

        if stats.get("total_items", 0) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Items", stats["total_items"])
            with col2:
                st.metric("Total Usage", stats["total_usage"])
            with col3:
                st.metric("Avg Usage", f"{stats['average_usage']:.1f}")

            # Category distribution
            if stats.get("categories"):
                st.subheader("Category Distribution")
                category_df = pd.DataFrame(
                    list(stats["categories"].items()),
                    columns=["Category", "Count"]
                )
                import plotly.express as px
                fig = px.pie(category_df, values="Count", names="Category",
                           title="Knowledge by Category")
                st.plotly_chart(fig, use_container_width=True)

            # Most used items
            if stats.get("most_used"):
                st.subheader("Most Used Knowledge")
                for item in stats["most_used"]:
                    st.markdown(f"- [{item.category}] {item.content[:100]}... (Used: {item.usage_count} times)")

            # Export/Import options
            st.subheader("Data Management")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export Knowledge Base"):
                    export_file = f"{selected_agent}_knowledge.json"
                    knowledge_manager.export_knowledge_base(selected_agent, export_file)
                    st.success(f"✅ Exported to {export_file}")

            with col2:
                st.info("Use the 'Add Knowledge' tab to import files")
        else:
            st.info("No knowledge items yet for this agent")