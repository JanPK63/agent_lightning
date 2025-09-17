#!/usr/bin/env python3
"""
Visual Debugger Interface for Agent Lightning
Real-time debugging with visual breakpoints, variable inspection, and execution flow
"""

import os
import sys
import json
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import uuid
from collections import defaultdict, deque
import ast
import inspect
import threading
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    VisualProgram,
    ConnectionType
)
from visual_code_blocks import InteractiveBlock, VisualCanvas
from code_preview_panel import CodePreviewPanel


class DebugState(Enum):
    """States for debugger"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    STOPPED = "stopped"
    ERROR = "error"


class BreakpointType(Enum):
    """Types of breakpoints"""
    LINE = "line"
    CONDITIONAL = "conditional"
    EXCEPTION = "exception"
    FUNCTION_ENTRY = "function_entry"
    FUNCTION_EXIT = "function_exit"
    VARIABLE_CHANGE = "variable_change"


@dataclass
class Breakpoint:
    """Represents a debugging breakpoint"""
    breakpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    block_id: Optional[str] = None
    line_number: Optional[int] = None
    condition: Optional[str] = None
    breakpoint_type: BreakpointType = BreakpointType.LINE
    enabled: bool = True
    hit_count: int = 0
    
    # Visual properties
    position: Optional[Tuple[float, float]] = None
    color: str = "#FF0000"
    
    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate breakpoint condition"""
        if not self.condition:
            return True
        
        try:
            # Safely evaluate condition
            return eval(self.condition, {"__builtins__": {}}, context)
        except:
            return False


@dataclass
class StackFrame:
    """Represents a call stack frame"""
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function_name: str = ""
    file_name: str = ""
    line_number: int = 0
    block_id: Optional[str] = None
    local_variables: Dict[str, Any] = field(default_factory=dict)
    arguments: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionTrace:
    """Trace of code execution"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    block_id: Optional[str] = None
    line_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    variables_snapshot: Dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None
    error: Optional[str] = None


class VariableWatcher:
    """Watches variables for changes"""
    
    def __init__(self):
        self.watched_variables: Dict[str, Any] = {}
        self.watch_expressions: List[str] = []
        self.variable_history: Dict[str, List[Tuple[datetime, Any]]] = defaultdict(list)
    
    def add_watch(self, variable_name: str):
        """Add variable to watch list"""
        if variable_name not in self.watched_variables:
            self.watched_variables[variable_name] = None
    
    def add_expression(self, expression: str):
        """Add expression to watch"""
        if expression not in self.watch_expressions:
            self.watch_expressions.append(expression)
    
    def update_variable(self, name: str, value: Any):
        """Update watched variable value"""
        old_value = self.watched_variables.get(name)
        self.watched_variables[name] = value
        
        # Record history
        self.variable_history[name].append((datetime.now(), value))
        
        # Keep only last 100 values
        if len(self.variable_history[name]) > 100:
            self.variable_history[name] = self.variable_history[name][-100:]
        
        return old_value != value  # Return True if changed
    
    def evaluate_expressions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate watch expressions"""
        results = {}
        for expr in self.watch_expressions:
            try:
                results[expr] = eval(expr, {"__builtins__": {}}, context)
            except Exception as e:
                results[expr] = f"Error: {str(e)}"
        return results
    
    def get_variable_timeline(self, variable_name: str) -> List[Dict[str, Any]]:
        """Get timeline of variable changes"""
        history = self.variable_history.get(variable_name, [])
        return [
            {
                "timestamp": ts.isoformat(),
                "value": str(val)[:100]  # Truncate long values
            }
            for ts, val in history
        ]


class ExecutionProfiler:
    """Profiles code execution"""
    
    def __init__(self):
        self.block_timings: Dict[str, List[float]] = defaultdict(list)
        self.function_calls: Dict[str, int] = defaultdict(int)
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
    
    def start_profiling(self):
        """Start profiling session"""
        self.start_time = datetime.now()
        self.block_timings.clear()
        self.function_calls.clear()
        self.memory_snapshots.clear()
    
    def record_block_execution(self, block_id: str, duration: float):
        """Record block execution time"""
        self.block_timings[block_id].append(duration)
    
    def record_function_call(self, function_name: str):
        """Record function call"""
        self.function_calls[function_name] += 1
    
    def take_memory_snapshot(self):
        """Take memory usage snapshot"""
        import psutil
        process = psutil.Process()
        
        self.memory_snapshots.append({
            "timestamp": datetime.now().isoformat(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent()
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            "total_duration": None,
            "block_statistics": {},
            "function_statistics": dict(self.function_calls),
            "memory_usage": self.memory_snapshots[-10:] if self.memory_snapshots else []
        }
        
        if self.start_time:
            report["total_duration"] = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate block statistics
        for block_id, timings in self.block_timings.items():
            if timings:
                report["block_statistics"][block_id] = {
                    "count": len(timings),
                    "total_time": sum(timings),
                    "average_time": sum(timings) / len(timings),
                    "min_time": min(timings),
                    "max_time": max(timings)
                }
        
        return report


class VisualDebugger:
    """Main visual debugger class"""
    
    def __init__(self, program: Optional[VisualProgram] = None):
        self.program = program
        self.state = DebugState.IDLE
        self.breakpoints: List[Breakpoint] = []
        self.call_stack: List[StackFrame] = []
        self.execution_trace: deque = deque(maxlen=1000)
        self.current_block: Optional[str] = None
        self.current_line: Optional[int] = None
        
        # Components
        self.watcher = VariableWatcher()
        self.profiler = ExecutionProfiler()
        self.canvas = VisualCanvas()
        
        # Callbacks
        self.on_breakpoint_hit: List[Callable] = []
        self.on_variable_change: List[Callable] = []
        self.on_error: List[Callable] = []
        self.on_state_change: List[Callable] = []
        
        # Execution control
        self.step_event = threading.Event()
        self.continue_event = threading.Event()
        self.execution_thread: Optional[threading.Thread] = None
        
        # Error handling
        self.last_error: Optional[Dict[str, Any]] = None
    
    def set_program(self, program: VisualProgram):
        """Set program to debug"""
        self.program = program
        self._update_canvas()
    
    def _update_canvas(self):
        """Update canvas with program blocks"""
        if not self.program:
            return
        
        self.canvas.blocks.clear()
        for block in self.program.blocks:
            # Convert to interactive block
            interactive = InteractiveBlock(
                block_type=block.block_type,
                title=block.title,
                position=block.position
            )
            interactive.block_id = block.block_id
            interactive.properties = block.properties
            self.canvas.add_block(interactive)
    
    def add_breakpoint(
        self,
        block_id: Optional[str] = None,
        line_number: Optional[int] = None,
        condition: Optional[str] = None,
        breakpoint_type: BreakpointType = BreakpointType.LINE
    ) -> Breakpoint:
        """Add a breakpoint"""
        bp = Breakpoint(
            block_id=block_id,
            line_number=line_number,
            condition=condition,
            breakpoint_type=breakpoint_type
        )
        self.breakpoints.append(bp)
        
        # Update visual representation
        if block_id:
            block = self.canvas.get_block(block_id)
            if block:
                block.interaction.error = "🔴"  # Show breakpoint
        
        return bp
    
    def remove_breakpoint(self, breakpoint_id: str):
        """Remove a breakpoint"""
        self.breakpoints = [
            bp for bp in self.breakpoints
            if bp.breakpoint_id != breakpoint_id
        ]
        self._update_breakpoint_visuals()
    
    def _update_breakpoint_visuals(self):
        """Update visual breakpoint indicators"""
        # Clear all breakpoint indicators
        for block in self.canvas.blocks:
            block.interaction.error = None
        
        # Set breakpoint indicators
        for bp in self.breakpoints:
            if bp.enabled and bp.block_id:
                block = self.canvas.get_block(bp.block_id)
                if block:
                    block.interaction.error = "🔴"
    
    async def start_debugging(self):
        """Start debugging session"""
        if self.state != DebugState.IDLE:
            return
        
        self.state = DebugState.RUNNING
        self.profiler.start_profiling()
        self.call_stack.clear()
        self.execution_trace.clear()
        
        await self._trigger_state_change()
        
        # Start execution in separate thread
        self.execution_thread = threading.Thread(
            target=self._run_program,
            daemon=True
        )
        self.execution_thread.start()
    
    def _run_program(self):
        """Run the program with debugging"""
        try:
            if not self.program:
                return
            
            # Get execution order
            blocks = self.program.get_execution_order()
            
            for block in blocks:
                if self.state == DebugState.STOPPED:
                    break
                
                # Check breakpoints
                if self._check_breakpoint(block.block_id):
                    self._handle_breakpoint_hit(block.block_id)
                
                # Execute block
                self._execute_block(block)
                
                # Update trace
                self._add_trace(block.block_id)
                
                # Handle stepping
                if self.state == DebugState.STEPPING:
                    self.state = DebugState.PAUSED
                    self.step_event.clear()
                    self.step_event.wait()
            
            self.state = DebugState.IDLE
            
        except Exception as e:
            self._handle_error(e)
    
    def _check_breakpoint(self, block_id: str) -> bool:
        """Check if breakpoint should trigger"""
        for bp in self.breakpoints:
            if bp.enabled and bp.block_id == block_id:
                if bp.evaluate_condition(self._get_current_context()):
                    bp.hit_count += 1
                    return True
        return False
    
    def _handle_breakpoint_hit(self, block_id: str):
        """Handle breakpoint hit"""
        self.state = DebugState.PAUSED
        self.current_block = block_id
        
        # Highlight block
        block = self.canvas.get_block(block_id)
        if block:
            block.interaction.highlighted = True
        
        # Trigger callbacks
        asyncio.create_task(self._trigger_breakpoint())
        
        # Wait for continue or step
        self.continue_event.clear()
        self.continue_event.wait()
    
    def _execute_block(self, block: VisualBlock):
        """Execute a block (simulated)"""
        start_time = time.time()
        
        # Create stack frame
        frame = StackFrame(
            function_name=block.title,
            block_id=block.block_id,
            local_variables=dict(block.properties)
        )
        self.call_stack.append(frame)
        
        # Simulate execution based on block type
        if block.block_type == BlockType.VARIABLE:
            var_name = block.properties.get("variable_name", "var")
            var_value = block.properties.get("initial_value", None)
            self.watcher.update_variable(var_name, var_value)
        
        elif block.block_type == BlockType.OUTPUT:
            output = block.properties.get("value", "")
            self._add_output(output)
        
        # Record timing
        duration = time.time() - start_time
        self.profiler.record_block_execution(block.block_id, duration)
        
        # Pop stack frame
        if self.call_stack:
            self.call_stack.pop()
    
    def _add_trace(self, block_id: str):
        """Add to execution trace"""
        trace = ExecutionTrace(
            block_id=block_id,
            variables_snapshot=dict(self.watcher.watched_variables)
        )
        self.execution_trace.append(trace)
    
    def _add_output(self, output: str):
        """Add output to trace"""
        if self.execution_trace:
            self.execution_trace[-1].output = output
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current execution context"""
        context = dict(self.watcher.watched_variables)
        if self.call_stack:
            context.update(self.call_stack[-1].local_variables)
        return context
    
    def _handle_error(self, error: Exception):
        """Handle execution error"""
        self.state = DebugState.ERROR
        self.last_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "block_id": self.current_block,
            "line_number": self.current_line
        }
        
        # Highlight error block
        if self.current_block:
            block = self.canvas.get_block(self.current_block)
            if block:
                block.interaction.error = str(error)
        
        asyncio.create_task(self._trigger_error())
    
    def step_over(self):
        """Step to next block"""
        if self.state == DebugState.PAUSED:
            self.state = DebugState.STEPPING
            self.step_event.set()
            self.continue_event.set()
    
    def step_into(self):
        """Step into function (if applicable)"""
        # For visual blocks, similar to step_over
        self.step_over()
    
    def step_out(self):
        """Step out of current function"""
        # Continue until current stack frame is popped
        if self.state == DebugState.PAUSED and self.call_stack:
            target_depth = len(self.call_stack) - 1
            # Continue execution until stack depth decreases
            self.continue_debugging()
    
    def continue_debugging(self):
        """Continue execution"""
        if self.state == DebugState.PAUSED:
            self.state = DebugState.RUNNING
            self.continue_event.set()
    
    def pause_debugging(self):
        """Pause execution"""
        if self.state == DebugState.RUNNING:
            self.state = DebugState.PAUSED
    
    def stop_debugging(self):
        """Stop debugging session"""
        self.state = DebugState.STOPPED
        self.continue_event.set()
        self.step_event.set()
        
        # Clear highlights
        for block in self.canvas.blocks:
            block.interaction.highlighted = False
            block.interaction.error = None
    
    def get_call_stack(self) -> List[Dict[str, Any]]:
        """Get formatted call stack"""
        return [
            {
                "frame_id": frame.frame_id,
                "function": frame.function_name,
                "block_id": frame.block_id,
                "line": frame.line_number,
                "variables": frame.local_variables
            }
            for frame in self.call_stack
        ]
    
    def get_variables(self) -> Dict[str, Any]:
        """Get all variables in current scope"""
        variables = dict(self.watcher.watched_variables)
        if self.call_stack:
            variables.update(self.call_stack[-1].local_variables)
        return variables
    
    def evaluate_expression(self, expression: str) -> Any:
        """Evaluate expression in current context"""
        try:
            context = self._get_current_context()
            return eval(expression, {"__builtins__": {}}, context)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        """Get execution timeline"""
        timeline = []
        for trace in self.execution_trace:
            timeline.append({
                "trace_id": trace.trace_id,
                "block_id": trace.block_id,
                "timestamp": trace.timestamp.isoformat(),
                "variables": trace.variables_snapshot,
                "output": trace.output,
                "error": trace.error
            })
        return timeline
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.profiler.get_performance_report()
    
    def export_debug_session(self) -> Dict[str, Any]:
        """Export complete debug session data"""
        return {
            "program": self.program.to_dict() if self.program else None,
            "state": self.state.value,
            "breakpoints": [asdict(bp) for bp in self.breakpoints],
            "call_stack": self.get_call_stack(),
            "variables": self.get_variables(),
            "execution_timeline": self.get_execution_timeline(),
            "performance_metrics": self.get_performance_metrics(),
            "last_error": self.last_error
        }
    
    async def _trigger_breakpoint(self):
        """Trigger breakpoint callbacks"""
        for callback in self.on_breakpoint_hit:
            if asyncio.iscoroutinefunction(callback):
                await callback(self.current_block)
            else:
                callback(self.current_block)
    
    async def _trigger_error(self):
        """Trigger error callbacks"""
        for callback in self.on_error:
            if asyncio.iscoroutinefunction(callback):
                await callback(self.last_error)
            else:
                callback(self.last_error)
    
    async def _trigger_state_change(self):
        """Trigger state change callbacks"""
        for callback in self.on_state_change:
            if asyncio.iscoroutinefunction(callback):
                await callback(self.state)
            else:
                callback(self.state)


class DebuggerUI:
    """UI wrapper for visual debugger"""
    
    def __init__(self, debugger: VisualDebugger):
        self.debugger = debugger
    
    def generate_html(self) -> str:
        """Generate HTML for debugger interface"""
        
        # Get debugger state
        state = self.debugger.export_debug_session()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Visual Debugger</title>
    <style>
        body {{
            font-family: 'Monaco', 'Courier New', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            margin: 0;
            padding: 20px;
        }}
        
        .debugger-container {{
            display: grid;
            grid-template-columns: 1fr 300px;
            grid-template-rows: 50px 1fr 200px;
            gap: 10px;
            height: calc(100vh - 40px);
        }}
        
        .toolbar {{
            grid-column: 1 / -1;
            background: #2d2d30;
            border-radius: 5px;
            padding: 10px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .canvas-area {{
            background: #1e1e1e;
            border: 1px solid #3e3e42;
            border-radius: 5px;
            overflow: auto;
            position: relative;
        }}
        
        .sidebar {{
            background: #252526;
            border-radius: 5px;
            padding: 10px;
            overflow-y: auto;
        }}
        
        .console {{
            grid-column: 1 / -1;
            background: #1e1e1e;
            border: 1px solid #3e3e42;
            border-radius: 5px;
            padding: 10px;
            overflow-y: auto;
            font-size: 12px;
        }}
        
        .debug-button {{
            background: #0e639c;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 3px;
            cursor: pointer;
        }}
        
        .debug-button:hover {{
            background: #1177bb;
        }}
        
        .debug-button:disabled {{
            background: #3e3e42;
            cursor: not-allowed;
        }}
        
        .state-indicator {{
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        .state-idle {{ background: #3e3e42; }}
        .state-running {{ background: #16825d; }}
        .state-paused {{ background: #cc6633; }}
        .state-error {{ background: #f14c4c; }}
        
        .variable-list {{
            font-size: 12px;
        }}
        
        .variable-item {{
            padding: 3px;
            margin: 2px 0;
            background: #2d2d30;
            border-radius: 3px;
        }}
        
        .breakpoint {{
            color: #f14c4c;
            font-weight: bold;
        }}
        
        .stack-frame {{
            padding: 5px;
            margin: 2px 0;
            background: #2d2d30;
            border-left: 3px solid #0e639c;
        }}
    </style>
</head>
<body>
    <div class="debugger-container">
        <!-- Toolbar -->
        <div class="toolbar">
            <button class="debug-button" onclick="startDebug()">▶️ Start</button>
            <button class="debug-button" onclick="pauseDebug()">⏸️ Pause</button>
            <button class="debug-button" onclick="stepOver()">⏭️ Step Over</button>
            <button class="debug-button" onclick="stepInto()">📥 Step Into</button>
            <button class="debug-button" onclick="stepOut()">📤 Step Out</button>
            <button class="debug-button" onclick="stopDebug()">⏹️ Stop</button>
            <div class="state-indicator state-{state['state']}">{state['state'].upper()}</div>
        </div>
        
        <!-- Canvas Area -->
        <div class="canvas-area" id="canvas">
            <!-- Visual blocks rendered here -->
        </div>
        
        <!-- Sidebar -->
        <div class="sidebar">
            <h3>🔍 Variables</h3>
            <div class="variable-list">
        """
        
        # Add variables
        for name, value in state['variables'].items():
            html += f"""
                <div class="variable-item">
                    <strong>{name}:</strong> {str(value)[:50]}
                </div>
            """
        
        html += """
            </div>
            
            <h3>📚 Call Stack</h3>
            <div class="stack-list">
        """
        
        # Add call stack
        for frame in state['call_stack']:
            html += f"""
                <div class="stack-frame">
                    {frame['function']}
                </div>
            """
        
        html += """
            </div>
            
            <h3>🔴 Breakpoints</h3>
            <div class="breakpoint-list">
        """
        
        # Add breakpoints
        for bp in state['breakpoints']:
            if bp['enabled']:
                html += f"""
                    <div class="breakpoint">
                        Block: {bp['block_id'] or 'N/A'}
                    </div>
                """
        
        html += """
            </div>
        </div>
        
        <!-- Console -->
        <div class="console">
            <h3>📝 Output Console</h3>
        """
        
        # Add execution timeline
        for trace in state['execution_timeline'][-10:]:
            if trace['output']:
                html += f"<div>► {trace['output']}</div>"
            if trace['error']:
                html += f"<div style='color: #f14c4c'>✖ {trace['error']}</div>"
        
        html += """
        </div>
    </div>
    
    <script>
        function startDebug() {
            fetch('/debug/start', {method: 'POST'});
        }
        
        function pauseDebug() {
            fetch('/debug/pause', {method: 'POST'});
        }
        
        function stepOver() {
            fetch('/debug/step-over', {method: 'POST'});
        }
        
        function stepInto() {
            fetch('/debug/step-into', {method: 'POST'});
        }
        
        function stepOut() {
            fetch('/debug/step-out', {method: 'POST'});
        }
        
        function stopDebug() {
            fetch('/debug/stop', {method: 'POST'});
        }
        
        // Auto-refresh
        setInterval(() => {
            location.reload();
        }, 2000);
    </script>
</body>
</html>
        """
        
        return html


# Test the visual debugger
def test_visual_debugger():
    """Test the visual debugger"""
    print("\n" + "="*60)
    print("Visual Debugger Test")
    print("="*60)
    
    from visual_code_builder import VisualProgram, BlockFactory
    
    # Create a test program
    program = VisualProgram(name="Debug Test Program")
    factory = BlockFactory()
    
    # Add blocks
    var_block = factory.create_variable_block()
    var_block.properties["variable_name"] = "counter"
    var_block.properties["initial_value"] = 0
    program.add_block(var_block)
    
    loop_block = factory.create_for_loop_block()
    loop_block.properties["variable_name"] = "i"
    loop_block.properties["items_expression"] = "range(5)"
    program.add_block(loop_block)
    
    output_block = factory.create_output_block()
    output_block.properties["value"] = "counter"
    program.add_block(output_block)
    
    print(f"\n📦 Created test program with {len(program.blocks)} blocks")
    
    # Create debugger
    debugger = VisualDebugger(program)
    
    # Add breakpoints
    bp1 = debugger.add_breakpoint(
        block_id=loop_block.block_id,
        breakpoint_type=BreakpointType.LINE
    )
    print(f"\n🔴 Added breakpoint at loop block")
    
    # Add variable watches
    debugger.watcher.add_watch("counter")
    debugger.watcher.add_watch("i")
    print(f"👁️ Watching variables: counter, i")
    
    # Start debugging
    print(f"\n▶️ Starting debug session...")
    asyncio.run(debugger.start_debugging())
    
    # Wait a bit for execution
    import time
    time.sleep(0.5)
    
    # Get debug info
    print(f"\n📊 Debug State: {debugger.state.value}")
    print(f"📚 Call Stack Depth: {len(debugger.call_stack)}")
    print(f"🔍 Variables: {debugger.get_variables()}")
    
    # Generate performance report
    perf = debugger.get_performance_metrics()
    print(f"\n⚡ Performance Metrics:")
    print(f"   Total Duration: {perf.get('total_duration', 0):.3f}s")
    print(f"   Blocks Executed: {len(perf.get('block_statistics', {}))}")
    
    # Export session
    session_data = debugger.export_debug_session()
    
    with open("debug_session.json", "w") as f:
        json.dump(session_data, f, indent=2, default=str)
    print(f"\n💾 Exported debug session to debug_session.json")
    
    # Generate UI
    ui = DebuggerUI(debugger)
    html = ui.generate_html()
    
    with open("visual_debugger.html", "w") as f:
        f.write(html)
    print(f"📄 Generated debugger UI at visual_debugger.html")
    
    return debugger


if __name__ == "__main__":
    print("Visual Debugger Interface for Agent Lightning")
    print("="*60)
    
    debugger = test_visual_debugger()
    
    print("\n✅ Visual Debugger ready!")
    print("\nFeatures:")
    print("  • Visual breakpoints with conditions")
    print("  • Step-by-step execution")
    print("  • Variable watching and history")
    print("  • Call stack inspection")
    print("  • Performance profiling")
    print("  • Execution timeline")
    print("  • Error handling and visualization")