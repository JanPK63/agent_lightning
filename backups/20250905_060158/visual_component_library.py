#!/usr/bin/env python3
"""
Visual Component Library for Agent Lightning
Drag-and-drop components with templates and presets
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    BlockPort,
    VisualProgram,
    ConnectionType
)


class ComponentCategory(Enum):
    """Categories for organizing components"""
    BASIC = "basic"
    CONTROL_FLOW = "control_flow"
    DATA_STRUCTURES = "data_structures"
    STRING_OPERATIONS = "string_operations"
    MATH_OPERATIONS = "math_operations"
    FILE_OPERATIONS = "file_operations"
    NETWORK = "network"
    DATABASE = "database"
    AI_ML = "ai_ml"
    TESTING = "testing"
    UTILITIES = "utilities"
    CUSTOM = "custom"


@dataclass
class ComponentTemplate:
    """Template for a reusable component"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: ComponentCategory = ComponentCategory.BASIC
    description: str = ""
    icon: str = "📦"
    tags: List[str] = field(default_factory=list)
    
    # Block configuration
    block_type: BlockType = BlockType.EXPRESSION
    default_properties: Dict[str, Any] = field(default_factory=dict)
    input_ports: List[Dict[str, Any]] = field(default_factory=list)
    output_ports: List[Dict[str, Any]] = field(default_factory=list)
    
    # Code generation template
    code_template: str = ""
    language: str = "python"
    
    # Visual properties
    default_color: str = "#4A90E2"
    default_size: tuple = (200, 100)
    expandable: bool = False
    
    # Usage statistics
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def create_instance(self) -> VisualBlock:
        """Create an instance of this template"""
        block = VisualBlock(
            block_type=self.block_type,
            title=self.name,
            description=self.description,
            color=self.default_color,
            icon=self.icon,
            size=self.default_size
        )
        
        # Add ports
        for port_config in self.input_ports:
            block.add_input_port(
                name=port_config["name"],
                port_type=port_config.get("type", "any"),
                required=port_config.get("required", False)
            )
        
        for port_config in self.output_ports:
            block.add_output_port(
                name=port_config["name"],
                port_type=port_config.get("type", "any")
            )
        
        # Set properties
        block.properties = self.default_properties.copy()
        
        # Update usage
        self.usage_count += 1
        self.last_used = datetime.now()
        
        return block


class ComponentLibrary:
    """Library of reusable visual components"""
    
    def __init__(self):
        self.templates: Dict[str, ComponentTemplate] = {}
        self.categories: Dict[ComponentCategory, List[str]] = {
            category: [] for category in ComponentCategory
        }
        self._initialize_standard_components()
    
    def _initialize_standard_components(self):
        """Initialize the standard component library"""
        
        # Basic Components
        self.add_template(ComponentTemplate(
            name="Print",
            category=ComponentCategory.BASIC,
            description="Print value to console",
            icon="🖨️",
            tags=["output", "console", "debug"],
            block_type=BlockType.OUTPUT,
            input_ports=[{"name": "value", "type": "any", "required": True}],
            code_template="print({value})",
            default_color="#00C9A7"
        ))
        
        self.add_template(ComponentTemplate(
            name="Input",
            category=ComponentCategory.BASIC,
            description="Get user input",
            icon="⌨️",
            tags=["input", "user", "console"],
            block_type=BlockType.INPUT,
            input_ports=[{"name": "prompt", "type": "string"}],
            output_ports=[{"name": "value", "type": "string"}],
            code_template="{value} = input({prompt})",
            default_color="#667EEA"
        ))
        
        self.add_template(ComponentTemplate(
            name="Variable",
            category=ComponentCategory.BASIC,
            description="Store a value",
            icon="📦",
            tags=["variable", "storage", "data"],
            block_type=BlockType.VARIABLE,
            input_ports=[{"name": "value", "type": "any"}],
            output_ports=[{"name": "value", "type": "any"}],
            default_properties={
                "variable_name": "var",
                "variable_type": "any"
            },
            code_template="{variable_name} = {value}",
            default_color="#4A90E2"
        ))
        
        # Control Flow Components
        self.add_template(ComponentTemplate(
            name="If-Else",
            category=ComponentCategory.CONTROL_FLOW,
            description="Conditional branching",
            icon="❓",
            tags=["if", "condition", "branch"],
            block_type=BlockType.IF_CONDITION,
            input_ports=[{"name": "condition", "type": "boolean", "required": True}],
            output_ports=[
                {"name": "then", "type": "control_flow"},
                {"name": "else", "type": "control_flow"}
            ],
            code_template="if {condition}:\n    {then}\nelse:\n    {else}",
            default_color="#FF6B6B",
            expandable=True
        ))
        
        self.add_template(ComponentTemplate(
            name="For Loop",
            category=ComponentCategory.CONTROL_FLOW,
            description="Iterate over items",
            icon="🔁",
            tags=["loop", "iteration", "for"],
            block_type=BlockType.FOR_LOOP,
            input_ports=[{"name": "items", "type": "array", "required": True}],
            output_ports=[
                {"name": "item", "type": "any"},
                {"name": "body", "type": "control_flow"}
            ],
            default_properties={
                "variable_name": "item"
            },
            code_template="for {variable_name} in {items}:\n    {body}",
            default_color="#51CF66",
            expandable=True
        ))
        
        self.add_template(ComponentTemplate(
            name="While Loop",
            category=ComponentCategory.CONTROL_FLOW,
            description="Loop while condition is true",
            icon="⭕",
            tags=["loop", "while", "condition"],
            block_type=BlockType.WHILE_LOOP,
            input_ports=[{"name": "condition", "type": "boolean", "required": True}],
            output_ports=[{"name": "body", "type": "control_flow"}],
            code_template="while {condition}:\n    {body}",
            default_color="#339AF0",
            expandable=True
        ))
        
        self.add_template(ComponentTemplate(
            name="Try-Catch",
            category=ComponentCategory.CONTROL_FLOW,
            description="Error handling",
            icon="🛡️",
            tags=["error", "exception", "try", "catch"],
            block_type=BlockType.TRY_CATCH,
            output_ports=[
                {"name": "try", "type": "control_flow"},
                {"name": "catch", "type": "control_flow"},
                {"name": "error", "type": "object"}
            ],
            code_template="try:\n    {try}\nexcept Exception as e:\n    {catch}",
            default_color="#FF6B9D",
            expandable=True
        ))
        
        # Data Structure Components
        self.add_template(ComponentTemplate(
            name="List",
            category=ComponentCategory.DATA_STRUCTURES,
            description="Create a list",
            icon="📝",
            tags=["list", "array", "collection"],
            block_type=BlockType.EXPRESSION,
            output_ports=[{"name": "list", "type": "array"}],
            default_properties={
                "initial_values": []
            },
            code_template="[{initial_values}]",
            default_color="#7950F2"
        ))
        
        self.add_template(ComponentTemplate(
            name="Dictionary",
            category=ComponentCategory.DATA_STRUCTURES,
            description="Create a dictionary",
            icon="📖",
            tags=["dict", "map", "object"],
            block_type=BlockType.EXPRESSION,
            output_ports=[{"name": "dict", "type": "object"}],
            default_properties={
                "key_value_pairs": {}
            },
            code_template="{key_value_pairs}",
            default_color="#9C36B5"
        ))
        
        self.add_template(ComponentTemplate(
            name="List Append",
            category=ComponentCategory.DATA_STRUCTURES,
            description="Add item to list",
            icon="➕",
            tags=["append", "add", "list"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "list", "type": "array", "required": True},
                {"name": "item", "type": "any", "required": True}
            ],
            output_ports=[{"name": "list", "type": "array"}],
            code_template="{list}.append({item})",
            default_color="#7950F2"
        ))
        
        # String Operations
        self.add_template(ComponentTemplate(
            name="String Concat",
            category=ComponentCategory.STRING_OPERATIONS,
            description="Concatenate strings",
            icon="🔗",
            tags=["string", "concat", "join"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "str1", "type": "string", "required": True},
                {"name": "str2", "type": "string", "required": True}
            ],
            output_ports=[{"name": "result", "type": "string"}],
            code_template="{str1} + {str2}",
            default_color="#20C997"
        ))
        
        self.add_template(ComponentTemplate(
            name="String Format",
            category=ComponentCategory.STRING_OPERATIONS,
            description="Format string with variables",
            icon="📐",
            tags=["format", "string", "template"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "template", "type": "string", "required": True},
                {"name": "values", "type": "object"}
            ],
            output_ports=[{"name": "result", "type": "string"}],
            code_template="{template}.format(**{values})",
            default_color="#20C997"
        ))
        
        # Math Operations
        self.add_template(ComponentTemplate(
            name="Addition",
            category=ComponentCategory.MATH_OPERATIONS,
            description="Add two numbers",
            icon="➕",
            tags=["math", "add", "plus"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "a", "type": "number", "required": True},
                {"name": "b", "type": "number", "required": True}
            ],
            output_ports=[{"name": "result", "type": "number"}],
            code_template="{a} + {b}",
            default_color="#FFB84D"
        ))
        
        self.add_template(ComponentTemplate(
            name="Comparison",
            category=ComponentCategory.MATH_OPERATIONS,
            description="Compare two values",
            icon="⚖️",
            tags=["compare", "equals", "greater", "less"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "a", "type": "any", "required": True},
                {"name": "b", "type": "any", "required": True}
            ],
            output_ports=[{"name": "result", "type": "boolean"}],
            default_properties={
                "operator": "=="
            },
            code_template="{a} {operator} {b}",
            default_color="#FFB84D"
        ))
        
        # File Operations
        self.add_template(ComponentTemplate(
            name="Read File",
            category=ComponentCategory.FILE_OPERATIONS,
            description="Read file contents",
            icon="📂",
            tags=["file", "read", "load"],
            block_type=BlockType.FILE_READ,
            input_ports=[{"name": "path", "type": "string", "required": True}],
            output_ports=[{"name": "content", "type": "string"}],
            code_template="with open({path}, 'r') as f:\n    {content} = f.read()",
            default_color="#868E96"
        ))
        
        self.add_template(ComponentTemplate(
            name="Write File",
            category=ComponentCategory.FILE_OPERATIONS,
            description="Write to file",
            icon="💾",
            tags=["file", "write", "save"],
            block_type=BlockType.FILE_WRITE,
            input_ports=[
                {"name": "path", "type": "string", "required": True},
                {"name": "content", "type": "string", "required": True}
            ],
            code_template="with open({path}, 'w') as f:\n    f.write({content})",
            default_color="#868E96"
        ))
        
        # Network Components
        self.add_template(ComponentTemplate(
            name="HTTP GET",
            category=ComponentCategory.NETWORK,
            description="Make HTTP GET request",
            icon="🌐",
            tags=["http", "api", "request", "get"],
            block_type=BlockType.HTTP_REQUEST,
            input_ports=[
                {"name": "url", "type": "string", "required": True},
                {"name": "headers", "type": "object"}
            ],
            output_ports=[
                {"name": "response", "type": "object"},
                {"name": "status", "type": "number"}
            ],
            default_properties={
                "method": "GET",
                "timeout": 30
            },
            code_template="response = requests.get({url}, headers={headers})",
            default_color="#FF8787"
        ))
        
        self.add_template(ComponentTemplate(
            name="HTTP POST",
            category=ComponentCategory.NETWORK,
            description="Make HTTP POST request",
            icon="📮",
            tags=["http", "api", "request", "post"],
            block_type=BlockType.HTTP_REQUEST,
            input_ports=[
                {"name": "url", "type": "string", "required": True},
                {"name": "body", "type": "any", "required": True},
                {"name": "headers", "type": "object"}
            ],
            output_ports=[
                {"name": "response", "type": "object"},
                {"name": "status", "type": "number"}
            ],
            default_properties={
                "method": "POST",
                "timeout": 30
            },
            code_template="response = requests.post({url}, json={body}, headers={headers})",
            default_color="#FF8787"
        ))
        
        # Database Components
        self.add_template(ComponentTemplate(
            name="SQL Query",
            category=ComponentCategory.DATABASE,
            description="Execute SQL query",
            icon="🗄️",
            tags=["sql", "database", "query"],
            block_type=BlockType.DATABASE_QUERY,
            input_ports=[
                {"name": "connection", "type": "object", "required": True},
                {"name": "query", "type": "string", "required": True},
                {"name": "params", "type": "array"}
            ],
            output_ports=[{"name": "results", "type": "array"}],
            code_template="cursor = {connection}.cursor()\ncursor.execute({query}, {params})\n{results} = cursor.fetchall()",
            default_color="#00B4D8"
        ))
        
        # AI/ML Components
        self.add_template(ComponentTemplate(
            name="AI Prompt",
            category=ComponentCategory.AI_ML,
            description="Send prompt to AI model",
            icon="🤖",
            tags=["ai", "prompt", "llm", "gpt"],
            block_type=BlockType.API_CALL,
            input_ports=[
                {"name": "prompt", "type": "string", "required": True},
                {"name": "model", "type": "string"},
                {"name": "temperature", "type": "number"}
            ],
            output_ports=[{"name": "response", "type": "string"}],
            default_properties={
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 150
            },
            code_template="response = openai.chat.completions.create(model={model}, messages=[{'role': 'user', 'content': {prompt}}])",
            default_color="#10B981"
        ))
        
        self.add_template(ComponentTemplate(
            name="Embedding",
            category=ComponentCategory.AI_ML,
            description="Generate text embeddings",
            icon="🔢",
            tags=["embedding", "vector", "ai"],
            block_type=BlockType.API_CALL,
            input_ports=[
                {"name": "text", "type": "string", "required": True},
                {"name": "model", "type": "string"}
            ],
            output_ports=[{"name": "embedding", "type": "array"}],
            default_properties={
                "model": "text-embedding-ada-002"
            },
            code_template="response = openai.embeddings.create(input={text}, model={model})\n{embedding} = response.data[0].embedding",
            default_color="#10B981"
        ))
        
        # Testing Components
        self.add_template(ComponentTemplate(
            name="Assert",
            category=ComponentCategory.TESTING,
            description="Assert condition is true",
            icon="✅",
            tags=["test", "assert", "check"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "condition", "type": "boolean", "required": True},
                {"name": "message", "type": "string"}
            ],
            code_template="assert {condition}, {message}",
            default_color="#F59E0B"
        ))
        
        self.add_template(ComponentTemplate(
            name="Test Case",
            category=ComponentCategory.TESTING,
            description="Define a test case",
            icon="🧪",
            tags=["test", "unittest", "case"],
            block_type=BlockType.FUNCTION_DEF,
            output_ports=[{"name": "body", "type": "control_flow"}],
            default_properties={
                "test_name": "test_example"
            },
            code_template="def {test_name}():\n    {body}",
            default_color="#F59E0B",
            expandable=True
        ))
        
        # Utility Components
        self.add_template(ComponentTemplate(
            name="Comment",
            category=ComponentCategory.UTILITIES,
            description="Add a comment",
            icon="💬",
            tags=["comment", "note", "documentation"],
            block_type=BlockType.COMMENT,
            default_properties={
                "comment_text": "Add your comment here"
            },
            code_template="# {comment_text}",
            default_color="#94A3B8"
        ))
        
        self.add_template(ComponentTemplate(
            name="Import",
            category=ComponentCategory.UTILITIES,
            description="Import module",
            icon="📦",
            tags=["import", "module", "library"],
            block_type=BlockType.IMPORT,
            default_properties={
                "module_name": "module",
                "import_as": ""
            },
            code_template="import {module_name}" + (" as {import_as}" if "{import_as}" else ""),
            default_color="#94A3B8"
        ))
        
        self.add_template(ComponentTemplate(
            name="Function",
            category=ComponentCategory.UTILITIES,
            description="Define a function",
            icon="📋",
            tags=["function", "def", "method"],
            block_type=BlockType.FUNCTION_DEF,
            output_ports=[
                {"name": "body", "type": "control_flow"},
                {"name": "return", "type": "any"}
            ],
            default_properties={
                "function_name": "my_function",
                "parameters": [],
                "return_type": "any"
            },
            code_template="def {function_name}({parameters}):\n    {body}",
            default_color="#845EC2",
            expandable=True
        ))
    
    def add_template(self, template: ComponentTemplate) -> None:
        """Add a template to the library"""
        self.templates[template.template_id] = template
        self.categories[template.category].append(template.template_id)
    
    def get_template(self, template_id: str) -> Optional[ComponentTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: ComponentCategory) -> List[ComponentTemplate]:
        """Get all templates in a category"""
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def search_templates(self, query: str) -> List[ComponentTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def get_popular_templates(self, limit: int = 10) -> List[ComponentTemplate]:
        """Get most used templates"""
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )
        return sorted_templates[:limit]
    
    def get_recent_templates(self, limit: int = 10) -> List[ComponentTemplate]:
        """Get recently used templates"""
        used_templates = [t for t in self.templates.values() if t.last_used]
        sorted_templates = sorted(
            used_templates,
            key=lambda t: t.last_used,
            reverse=True
        )
        return sorted_templates[:limit]
    
    def create_custom_template(
        self,
        name: str,
        description: str,
        block_config: Dict[str, Any]
    ) -> ComponentTemplate:
        """Create a custom template"""
        template = ComponentTemplate(
            name=name,
            category=ComponentCategory.CUSTOM,
            description=description,
            **block_config
        )
        
        self.add_template(template)
        return template
    
    def export_template(self, template_id: str) -> Dict[str, Any]:
        """Export a template as JSON"""
        template = self.get_template(template_id)
        if template:
            return asdict(template)
        return {}
    
    def import_template(self, template_data: Dict[str, Any]) -> Optional[ComponentTemplate]:
        """Import a template from JSON"""
        try:
            template = ComponentTemplate(**template_data)
            self.add_template(template)
            return template
        except Exception as e:
            print(f"Error importing template: {e}")
            return None
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics"""
        total_usage = sum(t.usage_count for t in self.templates.values())
        
        return {
            "total_templates": len(self.templates),
            "categories": {
                cat.value: len(ids) 
                for cat, ids in self.categories.items()
            },
            "total_usage": total_usage,
            "most_used": self.get_popular_templates(1)[0].name if self.templates else None
        }


# Specialized component collections
class DataProcessingComponents:
    """Components for data processing pipelines"""
    
    @staticmethod
    def create_data_pipeline_template() -> ComponentTemplate:
        """Create a data processing pipeline template"""
        return ComponentTemplate(
            name="Data Pipeline",
            category=ComponentCategory.DATA_STRUCTURES,
            description="Complete data processing pipeline",
            icon="🔄",
            tags=["pipeline", "etl", "data"],
            block_type=BlockType.FUNCTION_DEF,
            input_ports=[
                {"name": "input_data", "type": "array", "required": True}
            ],
            output_ports=[
                {"name": "processed_data", "type": "array"},
                {"name": "metrics", "type": "object"}
            ],
            expandable=True
        )
    
    @staticmethod
    def create_data_filter_template() -> ComponentTemplate:
        """Create a data filter template"""
        return ComponentTemplate(
            name="Filter Data",
            category=ComponentCategory.DATA_STRUCTURES,
            description="Filter data based on condition",
            icon="🔍",
            tags=["filter", "select", "where"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "data", "type": "array", "required": True},
                {"name": "condition", "type": "string", "required": True}
            ],
            output_ports=[{"name": "filtered", "type": "array"}],
            code_template="[item for item in {data} if {condition}]"
        )
    
    @staticmethod
    def create_data_transform_template() -> ComponentTemplate:
        """Create a data transformation template"""
        return ComponentTemplate(
            name="Transform Data",
            category=ComponentCategory.DATA_STRUCTURES,
            description="Transform data items",
            icon="🔄",
            tags=["map", "transform", "convert"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "data", "type": "array", "required": True},
                {"name": "transform", "type": "string", "required": True}
            ],
            output_ports=[{"name": "transformed", "type": "array"}],
            code_template="[{transform} for item in {data}]"
        )


class APIComponents:
    """Components for API development"""
    
    @staticmethod
    def create_api_endpoint_template() -> ComponentTemplate:
        """Create an API endpoint template"""
        return ComponentTemplate(
            name="API Endpoint",
            category=ComponentCategory.NETWORK,
            description="Define API endpoint",
            icon="🔌",
            tags=["api", "endpoint", "route"],
            block_type=BlockType.FUNCTION_DEF,
            default_properties={
                "method": "GET",
                "path": "/api/resource",
                "auth_required": True
            },
            output_ports=[{"name": "handler", "type": "control_flow"}],
            code_template="@app.route('{path}', methods=['{method}'])\ndef handler():\n    {handler}",
            expandable=True
        )
    
    @staticmethod
    def create_api_validation_template() -> ComponentTemplate:
        """Create API validation template"""
        return ComponentTemplate(
            name="Validate Request",
            category=ComponentCategory.NETWORK,
            description="Validate API request",
            icon="✔️",
            tags=["validate", "check", "request"],
            block_type=BlockType.EXPRESSION,
            input_ports=[
                {"name": "request", "type": "object", "required": True},
                {"name": "schema", "type": "object", "required": True}
            ],
            output_ports=[
                {"name": "valid", "type": "boolean"},
                {"name": "errors", "type": "array"}
            ],
            code_template="validate_request({request}, {schema})"
        )


# Test the component library
def test_component_library():
    """Test the visual component library"""
    print("\n" + "="*60)
    print("Visual Component Library Test")
    print("="*60)
    
    # Create library
    library = ComponentLibrary()
    
    # Get library stats
    stats = library.get_library_stats()
    print(f"\n📚 Library Statistics:")
    print(f"   Total templates: {stats['total_templates']}")
    print(f"   Categories: {len(stats['categories'])}")
    
    # Show categories
    print(f"\n📁 Categories:")
    for category, count in stats['categories'].items():
        if count > 0:
            print(f"   - {category}: {count} components")
    
    # Search for components
    search_results = library.search_templates("loop")
    print(f"\n🔍 Search results for 'loop': {len(search_results)} found")
    for template in search_results:
        print(f"   - {template.name}: {template.description}")
    
    # Get templates by category
    control_flow = library.get_templates_by_category(ComponentCategory.CONTROL_FLOW)
    print(f"\n🎮 Control Flow Components: {len(control_flow)}")
    for template in control_flow:
        print(f"   - {template.name} ({template.icon}): {template.description}")
    
    # Create instance from template
    if_template = library.search_templates("If-Else")[0]
    if_block = if_template.create_instance()
    print(f"\n✅ Created block from template: {if_block.title}")
    print(f"   Type: {if_block.block_type.value}")
    print(f"   Input ports: {[p.name for p in if_block.input_ports]}")
    print(f"   Output ports: {[p.name for p in if_block.output_ports]}")
    
    # Create custom template
    custom = library.create_custom_template(
        name="My Custom Block",
        description="A custom component",
        block_config={
            "block_type": BlockType.EXPRESSION,
            "input_ports": [{"name": "input1", "type": "string"}],
            "output_ports": [{"name": "output1", "type": "string"}],
            "code_template": "custom_function({input1})"
        }
    )
    print(f"\n🛠️ Created custom template: {custom.name}")
    
    return library


if __name__ == "__main__":
    print("Visual Component Library for Agent Lightning")
    print("="*60)
    
    library = test_component_library()
    
    print("\n✅ Visual Component Library ready with {} templates!".format(
        len(library.templates)
    ))