"""
Visual Code Builder interface for Agent Lightning Monitoring Dashboard
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional

from .models import DashboardConfig


class VisualCodeBuilderInterface:
    """Handles visual code builder interface"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def render_visual_code_builder(self):
        """Render Visual Code Builder interface"""
        st.header("🎨 Visual Code Builder")

        if not self._check_visual_code_builder_available():
            st.error("Visual Code Builder components not available. Please ensure all components are installed.")
            return

        # Initialize components in session state
        self._initialize_session_state()

        # Sidebar for component library
        with st.sidebar:
            self._render_component_library()

        # Main interface
        col1, col2 = st.columns([1, 1])

        with col1:
            self._render_visual_program_interface()

        with col2:
            self._render_code_generation_interface()

        # Templates section
        st.divider()
        self._render_code_templates()

        # Visual Debugger Section
        st.divider()
        self._render_visual_debugger()

    def _check_visual_code_builder_available(self) -> bool:
        """Check if visual code builder components are available"""
        try:
            from visual_code_builder import VisualProgram, BlockFactory, BlockType
            from visual_component_library import ComponentLibrary, ComponentCategory
            from visual_to_code_translator import VisualToCodeTranslator, TargetLanguage
            from code_preview_panel import CodePreviewPanel, PreviewSettings, PreviewTheme
            from visual_code_blocks import InteractiveBlock, VisualCanvas, BlockStyle
            from visual_debugger import VisualDebugger, DebugState, BreakpointType
            return True
        except ImportError as e:
            st.error(f"Visual Code Builder import error: {e}")
            return False

    def _initialize_session_state(self):
        """Initialize session state for visual code builder"""
        if 'visual_program' not in st.session_state:
            from visual_code_builder import VisualProgram
            from visual_component_library import ComponentLibrary
            from visual_to_code_translator import VisualToCodeTranslator
            from code_preview_panel import CodePreviewPanel, PreviewSettings, PreviewTheme
            from visual_debugger import VisualDebugger

            st.session_state.visual_program = VisualProgram(name="Agent Task")
            st.session_state.component_library = ComponentLibrary()
            st.session_state.translator = VisualToCodeTranslator()
            st.session_state.preview_panel = CodePreviewPanel(
                PreviewSettings(theme=PreviewTheme.MONOKAI)
            )
            st.session_state.block_factory = BlockFactory()

    def _render_component_library(self):
        """Render component library sidebar"""
        st.subheader("📦 Component Library")

        # Component categories
        category = st.selectbox(
            "Category",
            [cat.value for cat in ComponentCategory],
            key="component_category"
        )

        # Show components in selected category
        selected_category = ComponentCategory(category)
        templates = st.session_state.component_library.get_templates_by_category(selected_category)

        st.write(f"**{len(templates)} components available**")

        # Component search
        search_term = st.text_input("🔍 Search components", key="component_search")
        if search_term:
            templates = st.session_state.component_library.search_templates(search_term)

        # Display component templates
        for template in templates[:10]:  # Limit to 10 for performance
            with st.expander(f"{template.icon} {template.name}"):
                st.write(template.description)
                if st.button(f"Add {template.name}", key=f"add_{template.template_id}"):
                    block = template.create_instance()
                    st.session_state.visual_program.add_block(block)
                    st.success(f"Added {template.name} block")

    def _render_visual_program_interface(self):
        """Render visual program interface"""
        st.subheader("🎯 Visual Program")

        # Program info
        program_name = st.text_input(
            "Program Name",
            value=st.session_state.visual_program.name,
            key="program_name"
        )
        st.session_state.visual_program.name = program_name

        # Quick add common blocks
        st.write("**Quick Add Blocks:**")
        button_col1, button_col2, button_col3, button_col4 = st.columns(4)

        with button_col1:
            if st.button("➕ Function"):
                block = st.session_state.block_factory.create_function_block()
                st.session_state.visual_program.add_block(block)
                st.success("Added Function block")

        with button_col2:
            if st.button("❓ If-Else"):
                block = st.session_state.block_factory.create_if_block()
                st.session_state.visual_program.add_block(block)
                st.success("Added If-Else block")

        with button_col3:
            if st.button("🔁 For Loop"):
                block = st.session_state.block_factory.create_for_loop_block()
                st.session_state.visual_program.add_block(block)
                st.success("Added For Loop block")

        with button_col4:
            if st.button("📦 Variable"):
                block = st.session_state.block_factory.create_variable_block()
                st.session_state.visual_program.add_block(block)
                st.success("Added Variable block")

        # Display current blocks
        st.write(f"**Current Blocks ({len(st.session_state.visual_program.blocks)}):**")

        for i, block in enumerate(st.session_state.visual_program.blocks):
            with st.expander(f"{block.icon} {block.title} ({block.block_type.value})"):
                # Block properties
                for prop_name, prop_value in block.properties.items():
                    new_value = st.text_input(
                        prop_name.replace('_', ' ').title(),
                        value=str(prop_value),
                        key=f"prop_{block.block_id}_{prop_name}"
                    )
                    block.properties[prop_name] = new_value

                # Delete block button
                if st.button(f"🗑️ Delete", key=f"delete_{block.block_id}"):
                    st.session_state.visual_program.blocks.remove(block)
                    st.rerun()

        # Connections (simplified for Streamlit)
        if len(st.session_state.visual_program.blocks) >= 2:
            st.write("**Connect Blocks:**")
            block_names = [f"{b.title} ({i})" for i, b in enumerate(st.session_state.visual_program.blocks)]

            col_from, col_to = st.columns(2)
            with col_from:
                from_block = st.selectbox("From Block", block_names, key="conn_from")
            with col_to:
                to_block = st.selectbox("To Block", block_names, key="conn_to")

            if st.button("🔗 Connect"):
                from_idx = block_names.index(from_block)
                to_idx = block_names.index(to_block)
                from_block_obj = st.session_state.visual_program.blocks[from_idx]
                to_block_obj = st.session_state.visual_program.blocks[to_idx]

                # Simplified connection - connect first output to first input
                if from_block_obj.output_ports and to_block_obj.input_ports:
                    st.session_state.visual_program.connect_blocks(
                        from_block_obj.block_id,
                        from_block_obj.output_ports[0].name,
                        to_block_obj.block_id,
                        to_block_obj.input_ports[0].name
                    )
                    st.success(f"Connected {from_block} to {to_block}")

    def _render_code_generation_interface(self):
        """Render code generation interface"""
        st.subheader("📝 Generated Code")

        # Target language selection
        language = st.selectbox(
            "Target Language",
            ["python", "javascript", "typescript"],
            key="target_language"
        )

        target_lang = TargetLanguage(language)

        # Generate code button
        if st.button("🚀 Generate Code"):
            try:
                # Generate code
                code = st.session_state.translator.translate_program(
                    st.session_state.visual_program,
                    target_lang
                )

                # Validate
                valid, errors = st.session_state.translator.validate_translation(code, target_lang)

                if valid:
                    st.success("✅ Code generated successfully!")
                else:
                    st.warning("⚠️ Generated code has syntax issues:")
                    for error in errors:
                        st.error(error)

                # Display code
                st.code(code, language=language)

                # Download button
                file_extension = "py" if language == "python" else "js" if language == "javascript" else "ts"
                st.download_button(
                    label=f"📥 Download {program_name}.{file_extension}",
                    data=code,
                    file_name=f"{program_name}.{file_extension}",
                    mime="text/plain"
                )

                # Statistics
                lines = code.split('\n')
                st.info(f"📊 Generated {len(lines)} lines of {language} code")

            except Exception as e:
                st.error(f"Error generating code: {str(e)}")

        # Code preview placeholder
        if not st.session_state.visual_program.blocks:
            st.info("Add blocks to generate code")
        else:
            st.write("**Program Structure:**")
            execution_order = st.session_state.visual_program.get_execution_order()
            for i, block in enumerate(execution_order, 1):
                st.write(f"{i}. {block.icon} {block.title}")

    def _render_code_templates(self):
        """Render code templates section"""
        st.subheader("📚 Code Templates")

        template_col1, template_col2, template_col3 = st.columns(3)

        with template_col1:
            if st.button("🔄 Data Processing Pipeline"):
                self._load_data_processing_template()
                st.success("Created Data Processing Pipeline template")
                st.rerun()

        with template_col2:
            if st.button("🌐 API Handler"):
                self._load_api_handler_template()
                st.success("Created API Handler template")
                st.rerun()

        with template_col3:
            if st.button("🤖 Agent Task"):
                self._load_agent_task_template()
                st.success("Created Agent Task template")
                st.rerun()

    def _load_data_processing_template(self):
        """Load data processing pipeline template"""
        from visual_code_builder import VisualProgram
        # Clear current program
        st.session_state.visual_program = VisualProgram(name="Data Pipeline")

        # Add function block
        func = st.session_state.block_factory.create_function_block()
        func.properties["function_name"] = "process_data"
        func.properties["parameters"] = ["data"]
        st.session_state.visual_program.add_block(func)

        # Add for loop
        loop = st.session_state.block_factory.create_for_loop_block()
        loop.properties["variable_name"] = "item"
        st.session_state.visual_program.add_block(loop)

        # Add output
        output = st.session_state.block_factory.create_output_block()
        st.session_state.visual_program.add_block(output)

    def _load_api_handler_template(self):
        """Load API handler template"""
        from visual_code_builder import VisualProgram
        st.session_state.visual_program = VisualProgram(name="API Handler")

        # Add API call block
        api_block = st.session_state.block_factory.create_api_call_block()
        api_block.properties["url"] = "https://api.example.com/data"
        api_block.properties["method"] = "GET"
        st.session_state.visual_program.add_block(api_block)

        # Add if block for error checking
        if_block = st.session_state.block_factory.create_if_block()
        if_block.properties["condition_expression"] = "response.status == 200"
        st.session_state.visual_program.add_block(if_block)

    def _load_agent_task_template(self):
        """Load agent task template"""
        from visual_code_builder import VisualProgram
        st.session_state.visual_program = VisualProgram(name="Agent Task")

        # Add function for agent task
        func = st.session_state.block_factory.create_function_block()
        func.properties["function_name"] = "execute_agent_task"
        func.properties["parameters"] = ["agent_id", "task"]
        st.session_state.visual_program.add_block(func)

        # Add variable for result
        var = st.session_state.block_factory.create_variable_block()
        var.properties["variable_name"] = "result"
        st.session_state.visual_program.add_block(var)

    def _render_visual_debugger(self):
        """Render visual debugger section"""
        st.subheader("🐛 Visual Debugger")

        # Initialize debugger if needed
        if 'visual_debugger' not in st.session_state:
            from visual_debugger import VisualDebugger
            st.session_state.visual_debugger = VisualDebugger(st.session_state.visual_program)

        # Update debugger with current program
        st.session_state.visual_debugger.set_program(st.session_state.visual_program)

        debug_col1, debug_col2, debug_col3 = st.columns([2, 3, 2])

        with debug_col1:
            self._render_debug_controls()

        with debug_col2:
            self._render_debug_information()

        with debug_col3:
            self._render_execution_timeline()

    def _render_debug_controls(self):
        """Render debug controls"""
        st.write("**🎮 Debug Controls**")

        # Debug state display
        state_colors = {
            DebugState.IDLE: "🔵",
            DebugState.RUNNING: "🟢",
            DebugState.PAUSED: "🟡",
            DebugState.ERROR: "🔴",
            DebugState.STEPPING: "🟠",
            DebugState.STOPPED: "⚫"
        }
        current_state = st.session_state.visual_debugger.state
        st.info(f"Status: {state_colors.get(current_state, '⚪')} **{current_state.value.upper()}**")

        # Control buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶️ Start", disabled=current_state != DebugState.IDLE, key="debug_start"):
                try:
                    asyncio.run(st.session_state.visual_debugger.start_debugging())
                except:
                    pass
                st.rerun()

            if st.button("⏸️ Pause", disabled=current_state != DebugState.RUNNING, key="debug_pause"):
                st.session_state.visual_debugger.pause_debugging()
                st.rerun()

        with btn_col2:
            if st.button("⏹️ Stop", disabled=current_state == DebugState.IDLE, key="debug_stop"):
                st.session_state.visual_debugger.stop_debugging()
                st.rerun()

            if st.button("⏭️ Step", disabled=current_state != DebugState.PAUSED, key="debug_step"):
                st.session_state.visual_debugger.step_over()
                st.rerun()

        # Breakpoints section
        st.write("**🔴 Breakpoints**")
        if st.session_state.visual_program.blocks:
            block_names = [f"{b.title}" for b in st.session_state.visual_program.blocks]
            selected_idx = st.selectbox(
                "Add breakpoint to:",
                range(len(block_names)),
                format_func=lambda x: block_names[x],
                key="bp_select"
            )

            if st.button("➕ Add Breakpoint", key="add_bp"):
                from visual_debugger import BreakpointType
                block = st.session_state.visual_program.blocks[selected_idx]
                bp = st.session_state.visual_debugger.add_breakpoint(
                    block_id=block.block_id,
                    breakpoint_type=BreakpointType.LINE
                )
                st.success(f"Added breakpoint")

        # List current breakpoints
        if st.session_state.visual_debugger.breakpoints:
            st.write("Active breakpoints:")
            for i, bp in enumerate(st.session_state.visual_debugger.breakpoints):
                if bp.block_id and bp.enabled:
                    block = next((b for b in st.session_state.visual_program.blocks
                                if b.block_id == bp.block_id), None)
                    if block:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"• {block.title}")
                        with col2:
                            if st.button("❌", key=f"rm_{i}"):
                                st.session_state.visual_debugger.remove_breakpoint(bp.breakpoint_id)
                                st.rerun()

    def _render_debug_information(self):
        """Render debug information"""
        st.write("**📊 Debug Information**")

        # Variables display
        variables = st.session_state.visual_debugger.get_variables()
        if variables:
            with st.expander("🔍 Variables", expanded=True):
                for name, value in list(variables.items())[:10]:
                    st.code(f"{name} = {str(value)[:100]}", language="python")

        # Call stack
        call_stack = st.session_state.visual_debugger.get_call_stack()
        if call_stack:
            with st.expander("📚 Call Stack", expanded=True):
                for frame in call_stack[:5]:
                    st.write(f"→ {frame['function']}")

        # Watch expressions
        with st.expander("👁️ Watch Expressions"):
            expr = st.text_input("Add expression:", key="watch_input")
            if st.button("Add Watch", key="add_watch"):
                st.session_state.visual_debugger.watcher.add_expression(expr)
                st.success(f"Watching: {expr}")

            # Show watched expressions
            if st.session_state.visual_debugger.watcher.watch_expressions:
                st.write("Watched:")
                for expr in st.session_state.visual_debugger.watcher.watch_expressions:
                    result = st.session_state.visual_debugger.evaluate_expression(expr)
                    st.code(f"{expr} = {result}", language="python")

    def _render_execution_timeline(self):
        """Render execution timeline"""
        st.write("**📜 Execution Timeline**")

        timeline = st.session_state.visual_debugger.get_execution_timeline()
        if timeline:
            with st.expander("Recent Execution", expanded=True):
                for trace in timeline[-5:]:
                    if trace.get('block_id'):
                        block = next((b for b in st.session_state.visual_program.blocks
                                    if b.block_id == trace['block_id']), None)
                        if block:
                            st.write(f"• {block.title}")
                            if trace.get('output'):
                                st.code(trace['output'], language="text")
        else:
            st.info("No execution history")

        # Performance metrics
        if current_state != DebugState.IDLE:
            perf = st.session_state.visual_debugger.get_performance_metrics()
            col1, col2 = st.columns(2)
            with col1:
                if perf.get('total_duration'):
                    st.metric("Duration", f"{perf['total_duration']:.2f}s")
            with col2:
                if perf.get('block_statistics'):
                    st.metric("Blocks", len(perf['block_statistics']))

        # Error display
        if st.session_state.visual_debugger.last_error:
            st.error(f"Error: {st.session_state.visual_debugger.last_error.get('message', 'Unknown error')}")