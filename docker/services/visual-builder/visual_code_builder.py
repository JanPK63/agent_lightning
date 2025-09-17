#!/usr/bin/env python3
"""
Visual Code Builder for Agent Lightning
Drag-and-drop interface for visual programming and code generation
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import uuid
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BlockType(Enum):
    """Types of visual code blocks"""
    # Control Flow
    IF_CONDITION = "if_condition"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    FUNCTION_DEF = "function_def"
    TRY_CATCH = "try_catch"
    
    # Data Operations
    VARIABLE = "variable"
    ASSIGNMENT = "assignment"
    EXPRESSION = "expression"
    RETURN = "return"
    
    # I/O Operations
    INPUT = "input"
    OUTPUT = "output"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    
    # API Operations
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    HTTP_REQUEST = "http_request"
    
    # Special Blocks
    COMMENT = "comment"
    IMPORT = "import"
    CLASS_DEF = "class_def"
    DECORATOR = "decorator"


class ConnectionType(Enum):
    """Types of connections between blocks"""
    CONTROL_FLOW = "control_flow"  # Execution order
    DATA_FLOW = "data_flow"         # Data passing
    CONDITION = "condition"         # Conditional branching
    LOOP_BODY = "loop_body"         # Loop content
    FUNCTION_BODY = "function_body" # Function content
    ERROR_HANDLER = "error_handler" # Exception handling


@dataclass
class BlockPort:
    """Input/Output port for a block"""
    port_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    port_type: str = "any"  # any, string, number, boolean, object, array
    direction: str = "input"  # input or output
    required: bool = False
    multiple: bool = False  # Can accept multiple connections
    value: Any = None
    connected_to: List[str] = field(default_factory=list)


@dataclass
class VisualBlock:
    """A visual programming block"""
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    block_type: BlockType = BlockType.EXPRESSION
    title: str = ""
    description: str = ""
    position: Tuple[float, float] = (0, 0)
    size: Tuple[float, float] = (200, 100)
    
    # Ports
    input_ports: List[BlockPort] = field(default_factory=list)
    output_ports: List[BlockPort] = field(default_factory=list)
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Visual properties
    color: str = "#4A90E2"
    icon: str = "📦"
    collapsed: bool = False
    
    # Generated code
    generated_code: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_input_port(self, name: str, port_type: str = "any", required: bool = False) -> BlockPort:
        """Add an input port to the block"""
        port = BlockPort(
            name=name,
            port_type=port_type,
            direction="input",
            required=required
        )
        self.input_ports.append(port)
        return port
    
    def add_output_port(self, name: str, port_type: str = "any") -> BlockPort:
        """Add an output port to the block"""
        port = BlockPort(
            name=name,
            port_type=port_type,
            direction="output"
        )
        self.output_ports.append(port)
        return port
    
    def connect_to(self, other_block: 'VisualBlock', from_port: str, to_port: str):
        """Connect this block to another block"""
        # Find ports
        output_port = next((p for p in self.output_ports if p.name == from_port), None)
        input_port = next((p for p in other_block.input_ports if p.name == to_port), None)
        
        if output_port and input_port:
            # Check type compatibility
            if output_port.port_type == input_port.port_type or input_port.port_type == "any":
                output_port.connected_to.append(input_port.port_id)
                input_port.connected_to.append(output_port.port_id)
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            "block_id": self.block_id,
            "block_type": self.block_type.value,
            "title": self.title,
            "position": self.position,
            "properties": self.properties,
            "input_ports": [{"name": p.name, "type": p.port_type, "value": p.value} 
                           for p in self.input_ports],
            "output_ports": [{"name": p.name, "type": p.port_type} 
                            for p in self.output_ports]
        }


class BlockFactory:
    """Factory for creating different types of blocks"""
    
    @staticmethod
    def create_if_block() -> VisualBlock:
        """Create an IF condition block"""
        block = VisualBlock(
            block_type=BlockType.IF_CONDITION,
            title="If Condition",
            description="Conditional branching",
            color="#FF6B6B",
            icon="❓"
        )
        block.add_input_port("condition", "boolean", required=True)
        block.add_output_port("true_branch", "control_flow")
        block.add_output_port("false_branch", "control_flow")
        block.properties["condition_expression"] = ""
        return block
    
    @staticmethod
    def create_for_loop_block() -> VisualBlock:
        """Create a FOR loop block"""
        block = VisualBlock(
            block_type=BlockType.FOR_LOOP,
            title="For Loop",
            description="Iterate over items",
            color="#51CF66",
            icon="🔁"
        )
        block.add_input_port("items", "array", required=True)
        block.add_output_port("item", "any")
        block.add_output_port("index", "number")
        block.add_output_port("loop_body", "control_flow")
        block.properties["variable_name"] = "item"
        return block
    
    @staticmethod
    def create_function_block() -> VisualBlock:
        """Create a function definition block"""
        block = VisualBlock(
            block_type=BlockType.FUNCTION_DEF,
            title="Function",
            description="Define a function",
            color="#845EC2",
            icon="📋"
        )
        block.add_output_port("body", "control_flow")
        block.add_output_port("return", "any")
        block.properties["function_name"] = "my_function"
        block.properties["parameters"] = []
        block.properties["return_type"] = "any"
        return block
    
    @staticmethod
    def create_variable_block() -> VisualBlock:
        """Create a variable block"""
        block = VisualBlock(
            block_type=BlockType.VARIABLE,
            title="Variable",
            description="Store a value",
            color="#4A90E2",
            icon="📦"
        )
        block.add_input_port("value", "any")
        block.add_output_port("value", "any")
        block.properties["variable_name"] = "var"
        block.properties["variable_type"] = "any"
        block.properties["initial_value"] = None
        return block
    
    @staticmethod
    def create_api_call_block() -> VisualBlock:
        """Create an API call block"""
        block = VisualBlock(
            block_type=BlockType.API_CALL,
            title="API Call",
            description="Make an API request",
            color="#FFB84D",
            icon="🌐"
        )
        block.add_input_port("url", "string", required=True)
        block.add_input_port("method", "string")
        block.add_input_port("headers", "object")
        block.add_input_port("body", "any")
        block.add_output_port("response", "object")
        block.add_output_port("status", "number")
        block.properties["method"] = "GET"
        block.properties["timeout"] = 30
        return block
    
    @staticmethod
    def create_output_block() -> VisualBlock:
        """Create an output/print block"""
        block = VisualBlock(
            block_type=BlockType.OUTPUT,
            title="Output",
            description="Print to console",
            color="#00C9A7",
            icon="📤"
        )
        block.add_input_port("value", "any", required=True)
        block.properties["format"] = "text"
        return block
    
    @staticmethod
    def create_return_block() -> VisualBlock:
        """Create a return statement block"""
        block = VisualBlock(
            block_type=BlockType.RETURN,
            title="Return",
            description="Return value from function",
            color="#845EC2",
            icon="↩️"
        )
        block.add_input_port("value", "any", required=False)
        block.properties["value"] = None
        return block
    
    @staticmethod
    def create_try_catch_block() -> VisualBlock:
        """Create a try-catch error handling block"""
        block = VisualBlock(
            block_type=BlockType.TRY_CATCH,
            title="Try-Catch",
            description="Error handling",
            color="#FF4757",
            icon="🛡️"
        )
        block.add_output_port("try_body", "control_flow")
        block.add_output_port("catch_body", "control_flow")
        block.properties["exception_type"] = "Exception"
        block.properties["exception_var"] = "e"
        return block
    
    @staticmethod
    def create_database_query_block() -> VisualBlock:
        """Create a database query block"""
        block = VisualBlock(
            block_type=BlockType.DATABASE_QUERY,
            title="Database Query",
            description="Execute database query",
            color="#10B981",
            icon="🗄️"
        )
        block.add_input_port("query", "string", required=True)
        block.add_input_port("params", "object", required=False)
        block.add_output_port("result", "any")
        block.properties["query_type"] = "SELECT"
        block.properties["connection"] = ""
        return block
    
    @staticmethod
    def create_file_read_block() -> VisualBlock:
        """Create a file read block"""
        block = VisualBlock(
            block_type=BlockType.FILE_READ,
            title="Read File",
            description="Read file contents",
            color="#00C9A7",
            icon="📂"
        )
        block.add_input_port("path", "string", required=True)
        block.add_output_port("content", "string")
        block.properties["encoding"] = "utf-8"
        return block
    
    @staticmethod
    def create_expression_block() -> VisualBlock:
        """Create an expression evaluation block"""
        block = VisualBlock(
            block_type=BlockType.EXPRESSION,
            title="Expression",
            description="Evaluate expression",
            color="#4A90E2",
            icon="🔢"
        )
        block.add_output_port("result", "any")
        block.properties["expression"] = ""
        return block
    
    @staticmethod
    def create_while_loop_block() -> VisualBlock:
        """Create a WHILE loop block"""
        block = VisualBlock(
            block_type=BlockType.WHILE_LOOP,
            title="While Loop",
            description="Loop while condition is true",
            color="#51CF66",
            icon="⭕"
        )
        block.add_input_port("condition", "boolean", required=True)
        block.add_output_port("loop_body", "control_flow")
        block.properties["condition_expression"] = "True"
        return block


@dataclass
class VisualProgram:
    """A complete visual program"""
    program_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Program"
    description: str = ""
    blocks: List[VisualBlock] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Canvas properties
    canvas_position: Tuple[float, float] = (0, 0)
    canvas_zoom: float = 1.0
    
    # Generated code
    target_language: str = "python"
    generated_code: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    def add_block(self, block: VisualBlock) -> None:
        """Add a block to the program"""
        self.blocks.append(block)
        self.updated_at = datetime.now()
    
    def connect_blocks(
        self, 
        from_block_id: str, 
        from_port: str,
        to_block_id: str,
        to_port: str,
        connection_type: ConnectionType = ConnectionType.DATA_FLOW
    ) -> bool:
        """Connect two blocks"""
        from_block = next((b for b in self.blocks if b.block_id == from_block_id), None)
        to_block = next((b for b in self.blocks if b.block_id == to_block_id), None)
        
        if from_block and to_block:
            if from_block.connect_to(to_block, from_port, to_port):
                self.connections.append({
                    "from_block": from_block_id,
                    "from_port": from_port,
                    "to_block": to_block_id,
                    "to_port": to_port,
                    "type": connection_type.value
                })
                self.updated_at = datetime.now()
                return True
        return False
    
    def get_execution_order(self) -> List[VisualBlock]:
        """Get blocks in execution order using topological sort"""
        # Build adjacency list
        graph = {block.block_id: [] for block in self.blocks}
        in_degree = {block.block_id: 0 for block in self.blocks}
        
        for conn in self.connections:
            if conn["type"] == ConnectionType.CONTROL_FLOW.value:
                graph[conn["from_block"]].append(conn["to_block"])
                in_degree[conn["to_block"]] += 1
        
        # Topological sort
        queue = [block_id for block_id, degree in in_degree.items() if degree == 0]
        ordered = []
        
        while queue:
            block_id = queue.pop(0)
            ordered.append(block_id)
            
            for neighbor in graph[block_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Return blocks in order
        return [next(b for b in self.blocks if b.block_id == bid) for bid in ordered]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the visual program"""
        errors = []
        
        # Check for required connections
        for block in self.blocks:
            for port in block.input_ports:
                if port.required and not port.connected_to:
                    errors.append(f"Block '{block.title}' missing required input '{port.name}'")
        
        # Check for cycles in control flow
        if self._has_cycle():
            errors.append("Program contains a cycle in control flow")
        
        # Check for unconnected blocks
        connected_blocks = set()
        for conn in self.connections:
            connected_blocks.add(conn["from_block"])
            connected_blocks.add(conn["to_block"])
        
        for block in self.blocks:
            if block.block_id not in connected_blocks and len(self.blocks) > 1:
                errors.append(f"Block '{block.title}' is not connected")
        
        return len(errors) == 0, errors
    
    def _has_cycle(self) -> bool:
        """Check if the program has a cycle"""
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(block_id):
            visited.add(block_id)
            rec_stack.add(block_id)
            
            for conn in self.connections:
                if conn["from_block"] == block_id and conn["type"] == ConnectionType.CONTROL_FLOW.value:
                    neighbor = conn["to_block"]
                    if neighbor not in visited:
                        if has_cycle_util(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack.remove(block_id)
            return False
        
        for block in self.blocks:
            if block.block_id not in visited:
                if has_cycle_util(block.block_id):
                    return True
        return False
    
    def to_json(self) -> str:
        """Convert program to JSON"""
        return json.dumps({
            "program_id": self.program_id,
            "name": self.name,
            "description": self.description,
            "blocks": [block.to_dict() for block in self.blocks],
            "connections": self.connections,
            "target_language": self.target_language,
            "version": self.version
        }, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VisualProgram':
        """Create program from JSON"""
        data = json.loads(json_str)
        program = cls(
            program_id=data.get("program_id"),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            target_language=data.get("target_language", "python"),
            version=data.get("version", "1.0.0")
        )
        
        # Recreate blocks
        for block_data in data.get("blocks", []):
            block = VisualBlock(
                block_id=block_data["block_id"],
                block_type=BlockType(block_data["block_type"]),
                title=block_data["title"],
                position=tuple(block_data["position"]),
                properties=block_data["properties"]
            )
            
            # Recreate ports
            for port_data in block_data.get("input_ports", []):
                block.add_input_port(port_data["name"], port_data["type"])
            for port_data in block_data.get("output_ports", []):
                block.add_output_port(port_data["name"], port_data["type"])
            
            program.blocks.append(block)
        
        # Recreate connections
        program.connections = data.get("connections", [])
        
        return program


class VisualCodeBuilder:
    """Main visual code builder interface"""
    
    def __init__(self):
        self.programs: Dict[str, VisualProgram] = {}
        self.block_factory = BlockFactory()
        self.current_program: Optional[VisualProgram] = None
    
    def create_new_program(self, name: str = "New Program") -> VisualProgram:
        """Create a new visual program"""
        program = VisualProgram(name=name)
        self.programs[program.program_id] = program
        self.current_program = program
        return program
    
    def load_program(self, program_id: str) -> Optional[VisualProgram]:
        """Load an existing program"""
        if program_id in self.programs:
            self.current_program = self.programs[program_id]
            return self.current_program
        return None
    
    def add_block_to_current(
        self, 
        block_type: BlockType,
        position: Tuple[float, float] = (100, 100)
    ) -> Optional[VisualBlock]:
        """Add a block to the current program"""
        if not self.current_program:
            return None
        
        # Create block based on type
        if block_type == BlockType.IF_CONDITION:
            block = self.block_factory.create_if_block()
        elif block_type == BlockType.FOR_LOOP:
            block = self.block_factory.create_for_loop_block()
        elif block_type == BlockType.FUNCTION_DEF:
            block = self.block_factory.create_function_block()
        elif block_type == BlockType.VARIABLE:
            block = self.block_factory.create_variable_block()
        elif block_type == BlockType.API_CALL:
            block = self.block_factory.create_api_call_block()
        elif block_type == BlockType.OUTPUT:
            block = self.block_factory.create_output_block()
        else:
            block = VisualBlock(block_type=block_type)
        
        block.position = position
        self.current_program.add_block(block)
        return block
    
    def connect_blocks_in_current(
        self,
        from_block_id: str,
        from_port: str,
        to_block_id: str,
        to_port: str
    ) -> bool:
        """Connect two blocks in the current program"""
        if not self.current_program:
            return False
        
        return self.current_program.connect_blocks(
            from_block_id, from_port,
            to_block_id, to_port
        )
    
    def validate_current_program(self) -> Tuple[bool, List[str]]:
        """Validate the current program"""
        if not self.current_program:
            return False, ["No program loaded"]
        
        return self.current_program.validate()
    
    def get_program_preview(self) -> Dict[str, Any]:
        """Get a preview of the current program"""
        if not self.current_program:
            return {"error": "No program loaded"}
        
        valid, errors = self.current_program.validate()
        
        return {
            "program_id": self.current_program.program_id,
            "name": self.current_program.name,
            "blocks": len(self.current_program.blocks),
            "connections": len(self.current_program.connections),
            "valid": valid,
            "errors": errors,
            "execution_order": [b.title for b in self.current_program.get_execution_order()]
        }
    
    def save_program(self, filepath: str) -> bool:
        """Save the current program to a file"""
        if not self.current_program:
            return False
        
        try:
            with open(filepath, 'w') as f:
                f.write(self.current_program.to_json())
            return True
        except Exception as e:
            print(f"Error saving program: {e}")
            return False
    
    def load_program_from_file(self, filepath: str) -> Optional[VisualProgram]:
        """Load a program from a file"""
        try:
            with open(filepath, 'r') as f:
                json_str = f.read()
            
            program = VisualProgram.from_json(json_str)
            self.programs[program.program_id] = program
            self.current_program = program
            return program
        except Exception as e:
            print(f"Error loading program: {e}")
            return None


# Example usage and testing
def test_visual_code_builder():
    """Test the visual code builder"""
    print("\n" + "="*60)
    print("Visual Code Builder Test")
    print("="*60)
    
    # Create builder
    builder = VisualCodeBuilder()
    
    # Create a new program
    program = builder.create_new_program("Calculate Fibonacci")
    print(f"\n✅ Created program: {program.name}")
    
    # Add blocks
    func_block = builder.add_block_to_current(BlockType.FUNCTION_DEF, (100, 100))
    func_block.properties["function_name"] = "fibonacci"
    func_block.properties["parameters"] = ["n"]
    
    if_block = builder.add_block_to_current(BlockType.IF_CONDITION, (100, 200))
    if_block.properties["condition_expression"] = "n <= 1"
    
    return_block1 = builder.add_block_to_current(BlockType.RETURN, (50, 300))
    return_block1.properties["value"] = "n"
    
    var_block = builder.add_block_to_current(BlockType.VARIABLE, (200, 300))
    var_block.properties["variable_name"] = "result"
    
    output_block = builder.add_block_to_current(BlockType.OUTPUT, (200, 400))
    
    print(f"\n📦 Added {len(program.blocks)} blocks")
    for block in program.blocks:
        print(f"   - {block.title} ({block.block_type.value})")
    
    # Connect blocks
    builder.connect_blocks_in_current(
        func_block.block_id, "body",
        if_block.block_id, "condition"
    )
    
    builder.connect_blocks_in_current(
        if_block.block_id, "true_branch",
        return_block1.block_id, "value"
    )
    
    print(f"\n🔗 Created {len(program.connections)} connections")
    
    # Validate program
    valid, errors = builder.validate_current_program()
    print(f"\n✅ Program valid: {valid}")
    if errors:
        print("❌ Errors:")
        for error in errors:
            print(f"   - {error}")
    
    # Get preview
    preview = builder.get_program_preview()
    print(f"\n📊 Program Preview:")
    print(f"   Name: {preview['name']}")
    print(f"   Blocks: {preview['blocks']}")
    print(f"   Connections: {preview['connections']}")
    print(f"   Valid: {preview['valid']}")
    
    # Save program
    filepath = "visual_program_example.json"
    if builder.save_program(filepath):
        print(f"\n💾 Program saved to {filepath}")
    
    return builder


if __name__ == "__main__":
    print("Visual Code Builder for Agent Lightning")
    print("="*60)
    
    builder = test_visual_code_builder()
    
    print("\n✅ Visual Code Builder initialized successfully!")