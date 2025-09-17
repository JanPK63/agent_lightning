#!/usr/bin/env python3
"""
Visual Code Blocks Implementation
Interactive visual components for code generation
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import uuid
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    BlockPort,
    VisualProgram,
    ConnectionType
)
from visual_component_library import ComponentTemplate, ComponentCategory


class BlockStyle:
    """Visual styling for blocks"""
    
    # Color schemes
    COLORS = {
        "control_flow": "#FF6B6B",
        "data": "#4A90E2",
        "function": "#845EC2",
        "io": "#00C9A7",
        "network": "#FFB84D",
        "ai": "#10B981",
        "error": "#FF4757",
        "comment": "#94A3B8"
    }
    
    # Icon mappings
    ICONS = {
        BlockType.IF_CONDITION: "❓",
        BlockType.FOR_LOOP: "🔁",
        BlockType.WHILE_LOOP: "⭕",
        BlockType.FUNCTION_DEF: "📋",
        BlockType.TRY_CATCH: "🛡️",
        BlockType.VARIABLE: "📦",
        BlockType.ASSIGNMENT: "📝",
        BlockType.EXPRESSION: "🔢",
        BlockType.RETURN: "↩️",
        BlockType.INPUT: "⌨️",
        BlockType.OUTPUT: "🖨️",
        BlockType.FILE_READ: "📂",
        BlockType.FILE_WRITE: "💾",
        BlockType.API_CALL: "🌐",
        BlockType.DATABASE_QUERY: "🗄️",
        BlockType.HTTP_REQUEST: "📮",
        BlockType.COMMENT: "💬",
        BlockType.IMPORT: "📦",
        BlockType.CLASS_DEF: "🏗️",
        BlockType.DECORATOR: "🎯"
    }
    
    @staticmethod
    def get_color(block_type: BlockType) -> str:
        """Get color for a block type"""
        if block_type in [BlockType.IF_CONDITION, BlockType.FOR_LOOP, BlockType.WHILE_LOOP, BlockType.TRY_CATCH]:
            return BlockStyle.COLORS["control_flow"]
        elif block_type in [BlockType.VARIABLE, BlockType.ASSIGNMENT, BlockType.EXPRESSION]:
            return BlockStyle.COLORS["data"]
        elif block_type in [BlockType.FUNCTION_DEF, BlockType.CLASS_DEF, BlockType.DECORATOR]:
            return BlockStyle.COLORS["function"]
        elif block_type in [BlockType.INPUT, BlockType.OUTPUT, BlockType.FILE_READ, BlockType.FILE_WRITE]:
            return BlockStyle.COLORS["io"]
        elif block_type in [BlockType.API_CALL, BlockType.HTTP_REQUEST]:
            return BlockStyle.COLORS["network"]
        elif block_type == BlockType.DATABASE_QUERY:
            return BlockStyle.COLORS["ai"]
        elif block_type == BlockType.COMMENT:
            return BlockStyle.COLORS["comment"]
        else:
            return BlockStyle.COLORS["data"]
    
    @staticmethod
    def get_icon(block_type: BlockType) -> str:
        """Get icon for a block type"""
        return BlockStyle.ICONS.get(block_type, "📦")


@dataclass
class BlockInteraction:
    """Interaction state for a visual block"""
    block_id: str
    selected: bool = False
    dragging: bool = False
    highlighted: bool = False
    error: Optional[str] = None
    expanded: bool = True
    
    # Interaction data
    drag_offset: Tuple[float, float] = (0, 0)
    connection_preview: Optional[str] = None
    
    # Edit state
    editing_property: Optional[str] = None
    property_value: Any = None


class InteractiveBlock(VisualBlock):
    """Enhanced visual block with interactivity"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction = BlockInteraction(block_id=self.block_id)
        self.render_cache: Optional[Dict[str, Any]] = None
        self.animation_state: Dict[str, Any] = {}
    
    def handle_click(self, position: Tuple[float, float]) -> bool:
        """Handle click event"""
        if self._is_within_bounds(position):
            self.interaction.selected = not self.interaction.selected
            return True
        return False
    
    def handle_drag_start(self, position: Tuple[float, float]):
        """Handle drag start"""
        if self._is_within_bounds(position):
            self.interaction.dragging = True
            self.interaction.drag_offset = (
                position[0] - self.position[0],
                position[1] - self.position[1]
            )
    
    def handle_drag(self, position: Tuple[float, float]):
        """Handle drag movement"""
        if self.interaction.dragging:
            self.position = (
                position[0] - self.interaction.drag_offset[0],
                position[1] - self.interaction.drag_offset[1]
            )
            self.updated_at = datetime.now()
    
    def handle_drag_end(self):
        """Handle drag end"""
        self.interaction.dragging = False
        self.interaction.drag_offset = (0, 0)
    
    def handle_double_click(self):
        """Handle double click to expand/collapse"""
        if self.collapsed:
            self.expand()
        else:
            self.collapse()
    
    def collapse(self):
        """Collapse the block"""
        self.collapsed = True
        self.size = (self.size[0], 50)  # Minimize height
    
    def expand(self):
        """Expand the block"""
        self.collapsed = False
        self.size = (self.size[0], 100 + len(self.input_ports) * 25)
    
    def _is_within_bounds(self, position: Tuple[float, float]) -> bool:
        """Check if position is within block bounds"""
        x, y = position
        bx, by = self.position
        bw, bh = self.size
        return bx <= x <= bx + bw and by <= y <= by + bh
    
    def get_port_position(self, port: BlockPort) -> Tuple[float, float]:
        """Get visual position of a port"""
        x, y = self.position
        w, h = self.size
        
        if port.direction == "input":
            # Input ports on the left
            port_index = self.input_ports.index(port)
            port_y = y + 30 + port_index * 25
            return (x, port_y)
        else:
            # Output ports on the right
            port_index = self.output_ports.index(port)
            port_y = y + 30 + port_index * 25
            return (x + w, port_y)
    
    def render_data(self) -> Dict[str, Any]:
        """Get rendering data for the block"""
        return {
            "id": self.block_id,
            "type": self.block_type.value,
            "title": self.title,
            "position": self.position,
            "size": self.size,
            "color": self.color,
            "icon": self.icon,
            "collapsed": self.collapsed,
            "selected": self.interaction.selected,
            "highlighted": self.interaction.highlighted,
            "error": self.interaction.error,
            "ports": {
                "input": [
                    {
                        "id": p.port_id,
                        "name": p.name,
                        "type": p.port_type,
                        "required": p.required,
                        "connected": len(p.connected_to) > 0,
                        "position": self.get_port_position(p)
                    }
                    for p in self.input_ports
                ],
                "output": [
                    {
                        "id": p.port_id,
                        "name": p.name,
                        "type": p.port_type,
                        "connected": len(p.connected_to) > 0,
                        "position": self.get_port_position(p)
                    }
                    for p in self.output_ports
                ]
            },
            "properties": self.properties
        }


class CodeBlockRenderer:
    """Renders visual blocks to different formats"""
    
    @staticmethod
    def render_to_svg(block: InteractiveBlock) -> str:
        """Render block as SVG"""
        data = block.render_data()
        x, y = data["position"]
        w, h = data["size"]
        
        svg_parts = [
            f'<g transform="translate({x},{y})" class="visual-block" id="{data["id"]}">'
        ]
        
        # Main rectangle
        stroke_color = "#FF0000" if data["error"] else ("#000" if data["selected"] else "#CCC")
        svg_parts.append(
            f'<rect x="0" y="0" width="{w}" height="{h}" '
            f'fill="{data["color"]}" stroke="{stroke_color}" stroke-width="2" rx="5"/>'
        )
        
        # Title bar
        svg_parts.append(
            f'<rect x="0" y="0" width="{w}" height="30" '
            f'fill="{data["color"]}" opacity="0.8" rx="5"/>'
        )
        
        # Icon and title
        svg_parts.append(
            f'<text x="10" y="20" font-size="16">{data["icon"]}</text>'
        )
        svg_parts.append(
            f'<text x="35" y="20" font-size="14" font-weight="bold">{data["title"]}</text>'
        )
        
        if not data["collapsed"]:
            # Input ports
            for port in data["ports"]["input"]:
                px, py = port["position"]
                rel_x, rel_y = px - x, py - y
                fill_color = "#4CAF50" if port["connected"] else "#FFF"
                svg_parts.append(
                    f'<circle cx="{rel_x}" cy="{rel_y}" r="6" '
                    f'fill="{fill_color}" stroke="#333" stroke-width="1"/>'
                )
                svg_parts.append(
                    f'<text x="{rel_x + 10}" y="{rel_y + 4}" font-size="10">{port["name"]}</text>'
                )
            
            # Output ports
            for port in data["ports"]["output"]:
                px, py = port["position"]
                rel_x, rel_y = px - x, py - y
                fill_color = "#4CAF50" if port["connected"] else "#FFF"
                svg_parts.append(
                    f'<circle cx="{rel_x}" cy="{rel_y}" r="6" '
                    f'fill="{fill_color}" stroke="#333" stroke-width="1"/>'
                )
                svg_parts.append(
                    f'<text x="{rel_x - 40}" y="{rel_y + 4}" font-size="10" text-anchor="end">{port["name"]}</text>'
                )
        
        svg_parts.append('</g>')
        
        return '\n'.join(svg_parts)
    
    @staticmethod
    def render_to_html(block: InteractiveBlock) -> str:
        """Render block as HTML"""
        data = block.render_data()
        
        style = f"""
            position: absolute;
            left: {data['position'][0]}px;
            top: {data['position'][1]}px;
            width: {data['size'][0]}px;
            height: {data['size'][1]}px;
            background-color: {data['color']};
            border: 2px solid {'red' if data['error'] else 'black' if data['selected'] else '#ccc'};
            border-radius: 5px;
            padding: 5px;
            cursor: move;
        """
        
        html = f"""
        <div class="visual-block" id="{data['id']}" style="{style}">
            <div class="block-header">
                <span class="block-icon">{data['icon']}</span>
                <span class="block-title">{data['title']}</span>
            </div>
        """
        
        if not data['collapsed']:
            html += '<div class="block-ports">'
            
            # Input ports
            html += '<div class="input-ports">'
            for port in data['ports']['input']:
                connected_class = "connected" if port['connected'] else ""
                required_class = "required" if port['required'] else ""
                html += f"""
                <div class="port input-port {connected_class} {required_class}" data-port-id="{port['id']}">
                    <span class="port-connector">●</span>
                    <span class="port-name">{port['name']}</span>
                </div>
                """
            html += '</div>'
            
            # Output ports
            html += '<div class="output-ports">'
            for port in data['ports']['output']:
                connected_class = "connected" if port['connected'] else ""
                html += f"""
                <div class="port output-port {connected_class}" data-port-id="{port['id']}">
                    <span class="port-name">{port['name']}</span>
                    <span class="port-connector">●</span>
                </div>
                """
            html += '</div>'
            
            html += '</div>'
        
        html += '</div>'
        
        return html


class BlockConnectionRenderer:
    """Renders connections between blocks"""
    
    @staticmethod
    def render_bezier_path(
        start: Tuple[float, float],
        end: Tuple[float, float],
        connection_type: ConnectionType
    ) -> str:
        """Render a bezier curve connection"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate control points for bezier curve
        dx = abs(x2 - x1)
        control_offset = min(dx * 0.5, 100)
        
        cx1 = x1 + control_offset
        cy1 = y1
        cx2 = x2 - control_offset
        cy2 = y2
        
        # Color based on connection type
        colors = {
            ConnectionType.CONTROL_FLOW: "#FF6B6B",
            ConnectionType.DATA_FLOW: "#4A90E2",
            ConnectionType.CONDITION: "#FFB84D",
            ConnectionType.LOOP_BODY: "#51CF66",
            ConnectionType.FUNCTION_BODY: "#845EC2",
            ConnectionType.ERROR_HANDLER: "#FF4757"
        }
        color = colors.get(connection_type, "#666")
        
        return f"""
        <path d="M {x1} {y1} C {cx1} {cy1}, {cx2} {cy2}, {x2} {y2}"
              stroke="{color}" stroke-width="2" fill="none"
              class="connection {connection_type.value}"/>
        """
    
    @staticmethod
    def render_straight_line(
        start: Tuple[float, float],
        end: Tuple[float, float],
        connection_type: ConnectionType
    ) -> str:
        """Render a straight line connection"""
        x1, y1 = start
        x2, y2 = end
        
        colors = {
            ConnectionType.CONTROL_FLOW: "#FF6B6B",
            ConnectionType.DATA_FLOW: "#4A90E2"
        }
        color = colors.get(connection_type, "#666")
        
        return f"""
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
              stroke="{color}" stroke-width="2"
              class="connection {connection_type.value}"/>
        """


class VisualCanvas:
    """Canvas for visual programming"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.blocks: List[InteractiveBlock] = []
        self.connections: List[Dict[str, Any]] = []
        self.grid_size = 20
        self.zoom = 1.0
        self.pan_offset = (0, 0)
        
        # Selection state
        self.selected_blocks: List[str] = []
        self.connection_start: Optional[Tuple[str, str]] = None  # (block_id, port_id)
    
    def add_block(self, block: InteractiveBlock):
        """Add a block to the canvas"""
        # Snap to grid
        block.position = self._snap_to_grid(block.position)
        self.blocks.append(block)
    
    def remove_block(self, block_id: str):
        """Remove a block from the canvas"""
        self.blocks = [b for b in self.blocks if b.block_id != block_id]
        self.connections = [c for c in self.connections 
                          if c["from_block"] != block_id and c["to_block"] != block_id]
    
    def connect_blocks(
        self,
        from_block_id: str,
        from_port: str,
        to_block_id: str,
        to_port: str,
        connection_type: ConnectionType = ConnectionType.DATA_FLOW
    ):
        """Create a connection between blocks"""
        from_block = self.get_block(from_block_id)
        to_block = self.get_block(to_block_id)
        
        if from_block and to_block:
            # Check if connection is valid
            if from_block.connect_to(to_block, from_port, to_port):
                self.connections.append({
                    "from_block": from_block_id,
                    "from_port": from_port,
                    "to_block": to_block_id,
                    "to_port": to_port,
                    "type": connection_type
                })
                return True
        return False
    
    def get_block(self, block_id: str) -> Optional[InteractiveBlock]:
        """Get a block by ID"""
        return next((b for b in self.blocks if b.block_id == block_id), None)
    
    def _snap_to_grid(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Snap position to grid"""
        x, y = position
        grid = self.grid_size
        return (
            round(x / grid) * grid,
            round(y / grid) * grid
        )
    
    def handle_mouse_down(self, position: Tuple[float, float]):
        """Handle mouse down event"""
        # Check if clicking on a block
        for block in reversed(self.blocks):  # Check top blocks first
            if block.handle_click(position):
                if block.interaction.selected:
                    self.selected_blocks.append(block.block_id)
                else:
                    self.selected_blocks.remove(block.block_id)
                return
        
        # Clear selection if clicking on empty space
        self.selected_blocks = []
        for block in self.blocks:
            block.interaction.selected = False
    
    def handle_mouse_move(self, position: Tuple[float, float]):
        """Handle mouse move event"""
        # Handle dragging
        for block in self.blocks:
            if block.interaction.dragging:
                block.handle_drag(position)
    
    def handle_mouse_up(self):
        """Handle mouse up event"""
        for block in self.blocks:
            block.handle_drag_end()
    
    def render_to_svg(self) -> str:
        """Render canvas as SVG"""
        svg_parts = [
            f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">',
            f'<g transform="translate({self.pan_offset[0]},{self.pan_offset[1]}) scale({self.zoom})">'
        ]
        
        # Render grid
        svg_parts.append(self._render_grid_svg())
        
        # Render connections
        for conn in self.connections:
            from_block = self.get_block(conn["from_block"])
            to_block = self.get_block(conn["to_block"])
            
            if from_block and to_block:
                from_port = next((p for p in from_block.output_ports if p.name == conn["from_port"]), None)
                to_port = next((p for p in to_block.input_ports if p.name == conn["to_port"]), None)
                
                if from_port and to_port:
                    start = from_block.get_port_position(from_port)
                    end = to_block.get_port_position(to_port)
                    
                    svg_parts.append(
                        BlockConnectionRenderer.render_bezier_path(
                            start, end, conn["type"]
                        )
                    )
        
        # Render blocks
        for block in self.blocks:
            svg_parts.append(CodeBlockRenderer.render_to_svg(block))
        
        svg_parts.append('</g>')
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _render_grid_svg(self) -> str:
        """Render grid pattern"""
        grid = self.grid_size
        lines = []
        
        # Vertical lines
        for x in range(0, self.width, grid):
            lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{self.height}" stroke="#E0E0E0" stroke-width="0.5"/>')
        
        # Horizontal lines
        for y in range(0, self.height, grid):
            lines.append(f'<line x1="0" y1="{y}" x2="{self.width}" y2="{y}" stroke="#E0E0E0" stroke-width="0.5"/>')
        
        return '<g class="grid">\n' + '\n'.join(lines) + '\n</g>'


# Test the visual code blocks
def test_visual_code_blocks():
    """Test visual code blocks implementation"""
    print("\n" + "="*60)
    print("Visual Code Blocks Test")
    print("="*60)
    
    # Create canvas
    canvas = VisualCanvas(1200, 800)
    
    # Create interactive blocks
    if_block = InteractiveBlock(
        block_type=BlockType.IF_CONDITION,
        title="Check Value",
        position=(100, 100)
    )
    if_block.color = BlockStyle.get_color(BlockType.IF_CONDITION)
    if_block.icon = BlockStyle.get_icon(BlockType.IF_CONDITION)
    if_block.add_input_port("condition", "boolean", required=True)
    if_block.add_output_port("true", "control_flow")
    if_block.add_output_port("false", "control_flow")
    
    print_block = InteractiveBlock(
        block_type=BlockType.OUTPUT,
        title="Print Result",
        position=(400, 150)
    )
    print_block.color = BlockStyle.get_color(BlockType.OUTPUT)
    print_block.icon = BlockStyle.get_icon(BlockType.OUTPUT)
    print_block.add_input_port("value", "any", required=True)
    
    # Add blocks to canvas
    canvas.add_block(if_block)
    canvas.add_block(print_block)
    
    print(f"\n📦 Added {len(canvas.blocks)} interactive blocks")
    
    # Connect blocks
    success = canvas.connect_blocks(
        if_block.block_id, "true",
        print_block.block_id, "value",
        ConnectionType.CONTROL_FLOW
    )
    print(f"\n🔗 Connection created: {success}")
    
    # Test interaction
    print(f"\n🖱️ Testing interactions:")
    
    # Click on block
    canvas.handle_mouse_down((110, 110))
    print(f"   Block selected: {if_block.interaction.selected}")
    
    # Drag block
    if_block.handle_drag_start((110, 110))
    if_block.handle_drag((150, 150))
    print(f"   Block moved to: {if_block.position}")
    if_block.handle_drag_end()
    
    # Collapse/expand
    if_block.handle_double_click()
    print(f"   Block collapsed: {if_block.collapsed}")
    if_block.handle_double_click()
    print(f"   Block expanded: {not if_block.collapsed}")
    
    # Render to SVG
    svg = canvas.render_to_svg()
    print(f"\n🎨 Generated SVG ({len(svg)} chars)")
    
    # Save SVG
    with open("visual_code_blocks.svg", "w") as f:
        f.write(svg)
    print(f"   Saved to visual_code_blocks.svg")
    
    return canvas


if __name__ == "__main__":
    print("Visual Code Blocks Implementation")
    print("="*60)
    
    canvas = test_visual_code_blocks()
    
    print("\n✅ Visual Code Blocks ready!")