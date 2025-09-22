"""
Spec-Driven Development interface for Agent Lightning Monitoring Dashboard
"""

import streamlit as st
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

from .models import DashboardConfig


class SpecDrivenDevInterface:
    """Handles spec-driven development interface"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def render_spec_driven_development(self):
        """Render spec-driven development interface"""
        st.header("📋 Spec Driven Development")
        st.markdown("GitHub Spec-Kit methodology for executable specifications")

        # Check if spec service is running
        spec_service_url = "http://localhost:8029"
        service_running = self._check_spec_service_running(spec_service_url)

        if not service_running:
            self._render_service_not_running(spec_service_url)
            return

        st.success("✅ Spec-Driven Service is running")

        # Main workflow tabs
        spec_tab1, spec_tab2, spec_tab3 = st.tabs([
            "📝 Create or Progress Spec",
            "🛠️ Implementation Plan",
            "📚 Browse Existing Specs"
        ])

        with spec_tab1:
            self._render_create_progress_spec(spec_service_url)

        with spec_tab2:
            self._render_implementation_plan(spec_service_url)

        with spec_tab3:
            self._render_browse_specs(spec_service_url)

    def _check_spec_service_running(self, spec_service_url: str) -> bool:
        """Check if spec service is running"""
        try:
            health_response = requests.get(f"{spec_service_url}/health", timeout=2)
            return health_response.status_code == 200
        except:
            return False

    def _render_service_not_running(self, spec_service_url: str):
        """Render service not running message"""
        st.warning("⚠️ Spec-Driven Service not running. Start it with: `python spec_driven_service.py`")
        if st.button("🚀 Start Spec Service"):
            subprocess.Popen(["python", "spec_driven_service.py"], cwd="/Users/jankootstra/agent-lightning-main")
            st.info("Starting service... Refresh in a few seconds.")

    def _render_create_progress_spec(self, spec_service_url: str):
        """Render create or progress spec tab"""
        st.subheader("📝 Create New Spec or Progress on Existing")

        # Workflow selection
        workflow_type = st.radio(
            "What would you like to do?",
            options=["create_new", "progress_existing"],
            format_func=lambda x: {
                "create_new": "🆕 Create a completely new specification",
                "progress_existing": "📈 Progress on an existing specification"
            }[x],
            horizontal=True
        )

        if workflow_type == "create_new":
            self._render_create_new_spec(spec_service_url)
        else:
            self._render_progress_existing_spec(spec_service_url)

    def _render_create_new_spec(self, spec_service_url: str):
        """Render create new spec form"""
        st.markdown("### 🆕 Create New Specification")

        feature_desc = st.text_area(
            "Feature Description",
            placeholder="Describe the new feature you want to build...",
            height=120
        )

        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project Name", placeholder="My Project")
            spec_type = st.selectbox(
                "Specification Type",
                ["Feature Spec", "Comprehensive Spec"],
                help="Feature Spec: Quick feature specification | Comprehensive Spec: Detailed project specification"
            )
        with col2:
            output_dir = st.text_input("Output Directory", value="./specs")
            if spec_type == "Comprehensive Spec":
                project_type = st.selectbox(
                    "Project Type",
                    ["Web Application", "Mobile App", "API Service", "Desktop Application", "Library/Framework"]
                )

        if spec_type == "Comprehensive Spec":
            additional_reqs = st.text_area("Additional Requirements", height=68)

        if st.button("📝 Create New Specification", disabled=not feature_desc, type="primary"):
            with st.spinner("Generating specification..."):
                try:
                    if spec_type == "Feature Spec":
                        response = requests.post(
                            f"{spec_service_url}/new_feature",
                            json={
                                "feature_description": feature_desc,
                                "project_name": project_name,
                                "output_directory": output_dir
                            },
                            timeout=60
                        )
                    else:
                        response = requests.post(
                            f"{spec_service_url}/generate_spec",
                            json={
                                "description": feature_desc,
                                "project_type": project_type,
                                "spec_depth": "Detailed",
                                "additional_requirements": additional_reqs if 'additional_reqs' in locals() else "",
                                "output_directory": output_dir
                            },
                            timeout=120
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        if 'feature_number' in result:
                            st.info(f"Feature Number: {result['feature_number']}")
                            st.info(f"Branch Name: {result['branch_name']}")
                            st.code(f"Spec saved to: {result['spec_file']}")
                        else:
                            st.info(f"Main spec: {result['specification_file']}")
                            if result.get('section_files'):
                                st.info(f"Generated {len(result['section_files'])} section files")
                    else:
                        st.error(f"Failed to create spec: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def _render_progress_existing_spec(self, spec_service_url: str):
        """Render progress existing spec form"""
        st.markdown("### 📈 Progress on Existing Specification")

        # Load existing specs for reference
        existing_specs = self._load_existing_specs(spec_service_url)

        if existing_specs:
            # Reference spec selection
            st.markdown("**📚 Reference Existing Specification:**")
            spec_options = ["None"] + [f"{spec['filename']} ({spec['size']} bytes)" for spec in existing_specs]
            selected_spec_idx = st.selectbox(
                "Select existing spec to reference/elaborate on",
                range(len(spec_options)),
                format_func=lambda x: spec_options[x],
                help="Choose an existing spec to build upon or reference"
            )

            reference_spec = None
            if selected_spec_idx > 0:
                reference_spec = existing_specs[selected_spec_idx - 1]

                # Show preview of selected spec
                with st.expander(f"📖 Preview: {reference_spec['filename']}"):
                    try:
                        with open(reference_spec['path'], 'r') as f:
                            content = f.read()
                        st.code(content[:1000] + "..." if len(content) > 1000 else content, language="markdown")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
        else:
            st.info("No existing specifications found. Create a new one first.")
            reference_spec = None

        # Progress description
        progress_desc = st.text_area(
            "Describe the progress/elaboration you want to make",
            placeholder="I want to elaborate on the authentication system by adding OAuth2 support...\n\nOr: I want to add a new payment processing module to the existing e-commerce spec...",
            height=120
        )

        col1, col2 = st.columns(2)
        with col1:
            progress_type = st.selectbox(
                "Type of Progress",
                ["Elaborate Existing", "Add New Feature", "Refine Details", "Update Requirements"],
                help="Choose how you want to progress on the specification"
            )
        with col2:
            output_name = st.text_input(
                "Output File Name",
                placeholder="enhanced-auth-spec.md",
                help="Name for the new/updated specification file"
            )

        if st.button("🚀 Progress Specification", disabled=not progress_desc, type="primary"):
            with st.spinner("Processing specification progress..."):
                try:
                    # Prepare the request
                    request_data = {
                        "description": progress_desc,
                        "progress_type": progress_type,
                        "output_directory": "./specs",
                        "output_filename": output_name
                    }

                    # Add reference spec if selected
                    if reference_spec:
                        request_data["reference_spec_path"] = reference_spec['path']
                        request_data["reference_spec_name"] = reference_spec['filename']

                    # Use the generate_spec endpoint with progress context
                    response = requests.post(
                        f"{spec_service_url}/generate_spec",
                        json={
                            "description": f"Progress Type: {progress_type}\n\nReference Spec: {reference_spec['filename'] if reference_spec else 'None'}\n\nProgress Description:\n{progress_desc}",
                            "project_type": "Enhancement",
                            "spec_depth": "Detailed",
                            "additional_requirements": f"Build upon existing specification: {reference_spec['path'] if reference_spec else 'N/A'}",
                            "output_directory": "./specs"
                        },
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"Enhanced spec: {result['specification_file']}")
                        if result.get('section_files'):
                            st.info(f"Generated {len(result['section_files'])} section files")

                        # Show what was referenced
                        if reference_spec:
                            st.success(f"📚 Successfully built upon: {reference_spec['filename']}")
                    else:
                        st.error(f"Failed to progress spec: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def _render_implementation_plan(self, spec_service_url: str):
        """Render implementation plan tab"""
        st.subheader("🛠️ Generate Implementation Plan")

        # Check if a spec was selected from the browse tab
        if 'selected_plan_spec' in st.session_state:
            selected_spec = st.session_state.selected_plan_spec
            st.success(f"📚 Using selected spec: {selected_spec['filename']}")
            spec_file_path = st.text_input(
                "Feature Spec File Path",
                value=selected_spec['path'],
                help="Path auto-filled from selected specification"
            )
            if st.button("❌ Clear Selection"):
                del st.session_state.selected_plan_spec
                st.rerun()
        else:
            spec_file_path = st.text_input(
                "Feature Spec File Path",
                placeholder="./specs/001-feature/feature-spec.md",
                help="Enter path manually or select a spec from 'Browse Existing Specs' tab"
            )

        technical_approach = st.text_area(
            "Technical Approach",
            placeholder="Describe your technical approach, frameworks, architecture decisions...",
            height=100
        )

        plan_output_dir = st.text_input("Output Directory", value="./implementation", key="plan_output")

        if st.button("📝 Generate Implementation Plan", disabled=not spec_file_path or not technical_approach):
            with st.spinner("Generating implementation plan..."):
                try:
                    response = requests.post(
                        f"{spec_service_url}/generate_plan",
                        json={
                            "feature_spec_path": spec_file_path,
                            "technical_approach": technical_approach,
                            "output_directory": plan_output_dir
                        },
                        timeout=90
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.info(f"Plan file: {result['plan_file']}")
                        if result.get('supporting_documents'):
                            st.info(f"Generated {len(result['supporting_documents'])} supporting documents")
                    else:
                        st.error(f"Failed to generate plan: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def _render_browse_specs(self, spec_service_url: str):
        """Render browse existing specs tab"""
        st.subheader("📚 Browse Existing Specifications")

        list_dir = st.text_input("Directory to List", value="./specs")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🔄 Refresh List"):
                st.session_state.refresh_specs = True
        with col2:
            view_mode = st.selectbox("View", ["List", "Grid"], key="spec_view_mode")

        # Auto-load specs on first visit or refresh
        if 'refresh_specs' not in st.session_state:
            st.session_state.refresh_specs = True

        if st.session_state.refresh_specs:
            specs = self._load_existing_specs(spec_service_url, list_dir)
            st.session_state.specs_list = specs
            st.session_state.refresh_specs = False

        specs = st.session_state.get('specs_list', [])

        if specs:
            st.success(f"Found {len(specs)} specifications")

            # Search/filter
            search_term = st.text_input("🔍 Search specifications", placeholder="Enter keywords...")
            if search_term:
                specs = [s for s in specs if search_term.lower() in s['filename'].lower()]

            if view_mode == "Grid":
                self._render_specs_grid(specs)
            else:
                self._render_specs_list(specs)

            # Show selected specs for other operations
            if 'selected_progress_spec' in st.session_state:
                st.info(f"📈 Selected for progress: {st.session_state.selected_progress_spec['filename']}")

            if 'selected_plan_spec' in st.session_state:
                st.info(f"🛠️ Selected for planning: {st.session_state.selected_plan_spec['filename']}")

        else:
            st.info("No specifications found in directory")
            st.markdown("**💡 Tip:** Create your first specification in the 'Create or Progress Spec' tab!")

    def _load_existing_specs(self, spec_service_url: str, output_dir: str = "./specs") -> List[Dict]:
        """Load existing specs from service"""
        try:
            response = requests.get(
                f"{spec_service_url}/list_specs",
                params={"output_dir": output_dir},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('specifications', [])
            else:
                return []
        except:
            return []

    def _render_specs_grid(self, specs: List[Dict]):
        """Render specs in grid view"""
        cols = st.columns(2)
        for i, spec in enumerate(specs):
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"**📝 {spec['filename']}**")
                    st.caption(f"{spec['size']} bytes • Modified: {spec['modified'][:10]}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"👁️ View", key=f"view_grid_{i}"):
                            st.session_state[f"viewing_{spec['filename']}"] = True
                    with col_b:
                        if st.button(f"📈 Progress", key=f"progress_grid_{i}"):
                            # Set this spec for progress in tab 1
                            st.session_state.selected_progress_spec = spec
                            st.info(f"Selected {spec['filename']} for progress. Go to 'Create or Progress Spec' tab.")

    def _render_specs_list(self, specs: List[Dict]):
        """Render specs in list view"""
        for spec in specs:
            with st.expander(f"📝 {spec['filename']} ({spec['size']} bytes)"):
                st.text(f"Path: {spec['path']}")
                st.text(f"Modified: {spec['modified']}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"👁️ View Content", key=f"view_{spec['filename']}"):
                        try:
                            with open(spec['path'], 'r') as f:
                                content = f.read()
                            st.code(content, language="markdown")
                        except Exception as e:
                            st.error(f"Error reading file: {e}")

                with col2:
                    if st.button(f"📈 Use for Progress", key=f"progress_{spec['filename']}"):
                        st.session_state.selected_progress_spec = spec
                        st.success(f"Selected {spec['filename']} for progress!")
                        st.info("Go to 'Create or Progress Spec' tab to elaborate on this specification.")

                with col3:
                    if st.button(f"🛠️ Generate Plan", key=f"plan_{spec['filename']}"):
                        st.session_state.selected_plan_spec = spec
                        st.success(f"Selected {spec['filename']} for implementation planning!")
                        st.info("Go to 'Implementation Plan' tab to create a plan for this spec.")