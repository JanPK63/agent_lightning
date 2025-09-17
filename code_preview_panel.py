#!/usr/bin/env python3
"""
Code Preview Panel for Agent Lightning
Real-time code preview with syntax highlighting and live updates
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
import hashlib
import difflib
from pygments import highlight
from pygments.lexers import (
    PythonLexer,
    JavascriptLexer,
    TypeScriptLexer,
    JavaLexer,
    CppLexer,
    GoLexer,
    RustLexer,
    get_lexer_by_name
)
from pygments.formatters import HtmlFormatter, Terminal256Formatter
from pygments.styles import get_style_by_name

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import VisualProgram
from visual_to_code_translator import (
    VisualToCodeTranslator,
    TargetLanguage
)


class PreviewTheme(Enum):
    """Available themes for code preview"""
    MONOKAI = "monokai"
    GITHUB = "github"
    DRACULA = "dracula"
    SOLARIZED_DARK = "solarized-dark"
    SOLARIZED_LIGHT = "solarized-light"
    MATERIAL = "material"
    NORD = "nord"
    ONE_DARK = "one-dark"


@dataclass
class PreviewSettings:
    """Settings for code preview panel"""
    theme: PreviewTheme = PreviewTheme.MONOKAI
    font_size: int = 14
    font_family: str = "Monaco, 'Courier New', monospace"
    line_numbers: bool = True
    word_wrap: bool = False
    tab_size: int = 4
    auto_refresh: bool = True
    refresh_delay: float = 0.5  # seconds
    show_minimap: bool = True
    show_breadcrumbs: bool = True
    highlight_current_line: bool = True


@dataclass
class CodeSection:
    """A section of generated code"""
    start_line: int
    end_line: int
    block_id: str
    block_type: str
    code: str
    indentation: int
    is_error: bool = False
    error_message: Optional[str] = None


@dataclass
class PreviewState:
    """State of the code preview"""
    current_code: str = ""
    previous_code: str = ""
    language: TargetLanguage = TargetLanguage.PYTHON
    sections: List[CodeSection] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    cursor_position: Tuple[int, int] = (1, 0)  # line, column
    selection: Optional[Tuple[int, int, int, int]] = None  # start_line, start_col, end_line, end_col
    folded_regions: List[Tuple[int, int]] = field(default_factory=list)
    bookmarks: List[int] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class SyntaxHighlighter:
    """Syntax highlighter for different languages"""
    
    def __init__(self, theme: PreviewTheme = PreviewTheme.MONOKAI):
        self.theme = theme
        self.lexers = {
            TargetLanguage.PYTHON: PythonLexer(),
            TargetLanguage.JAVASCRIPT: JavascriptLexer(),
            TargetLanguage.TYPESCRIPT: TypeScriptLexer(),
        }
        
        # Map theme to Pygments style
        self.style_map = {
            PreviewTheme.MONOKAI: "monokai",
            PreviewTheme.GITHUB: "github-dark",
            PreviewTheme.DRACULA: "dracula",
            PreviewTheme.SOLARIZED_DARK: "solarized-dark",
            PreviewTheme.SOLARIZED_LIGHT: "solarized-light",
            PreviewTheme.MATERIAL: "material",
            PreviewTheme.NORD: "nord",
            PreviewTheme.ONE_DARK: "one-dark-pro"
        }
    
    def highlight_code(
        self,
        code: str,
        language: TargetLanguage,
        format_type: str = "html"
    ) -> str:
        """Highlight code with syntax colors"""
        lexer = self.lexers.get(language, PythonLexer())
        
        if format_type == "html":
            formatter = HtmlFormatter(
                style=self.style_map.get(self.theme, "monokai"),
                linenos='table',
                cssclass="highlight",
                lineanchors="line",
                anchorlinenos=True
            )
        else:
            formatter = Terminal256Formatter(
                style=self.style_map.get(self.theme, "monokai")
            )
        
        return highlight(code, lexer, formatter)
    
    def get_css(self) -> str:
        """Get CSS for syntax highlighting"""
        formatter = HtmlFormatter(
            style=self.style_map.get(self.theme, "monokai")
        )
        return formatter.get_style_defs('.highlight')


class CodeDiffer:
    """Calculates differences between code versions"""
    
    @staticmethod
    def get_diff(old_code: str, new_code: str) -> List[Dict[str, Any]]:
        """Get line-by-line diff between old and new code"""
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        
        differ = difflib.unified_diff(
            old_lines, new_lines,
            lineterm='',
            n=0
        )
        
        changes = []
        for line in differ:
            if line.startswith('+'):
                changes.append({'type': 'add', 'line': line[1:]})
            elif line.startswith('-'):
                changes.append({'type': 'remove', 'line': line[1:]})
            elif line.startswith('@@'):
                # Parse line numbers
                parts = line.split(' ')
                if len(parts) >= 3:
                    changes.append({'type': 'context', 'line': line})
        
        return changes
    
    @staticmethod
    def get_inline_diff(old_code: str, new_code: str) -> str:
        """Get inline diff with highlighting"""
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result.extend(old_lines[i1:i2])
            elif tag == 'delete':
                result.extend([f"- {line}" for line in old_lines[i1:i2]])
            elif tag == 'insert':
                result.extend([f"+ {line}" for line in new_lines[j1:j2]])
            elif tag == 'replace':
                result.extend([f"- {line}" for line in old_lines[i1:i2]])
                result.extend([f"+ {line}" for line in new_lines[j1:j2]])
        
        return '\n'.join(result)


class CodePreviewPanel:
    """Main code preview panel"""
    
    def __init__(self, settings: Optional[PreviewSettings] = None):
        self.settings = settings or PreviewSettings()
        self.state = PreviewState()
        self.highlighter = SyntaxHighlighter(self.settings.theme)
        self.translator = VisualToCodeTranslator()
        self.differ = CodeDiffer()
        
        # Callbacks
        self.on_code_change_callbacks = []
        self.on_error_callbacks = []
        self.on_selection_callbacks = []
        
        # Auto-refresh task
        self.refresh_task: Optional[asyncio.Task] = None
        self.current_program: Optional[VisualProgram] = None
    
    async def update_program(self, program: VisualProgram):
        """Update the preview with a new program"""
        self.current_program = program
        
        if self.settings.auto_refresh:
            await self.refresh_preview()
    
    async def refresh_preview(self):
        """Refresh the code preview"""
        if not self.current_program:
            return
        
        try:
            # Store previous code
            self.state.previous_code = self.state.current_code
            
            # Generate new code
            new_code = self.translator.translate_program(
                self.current_program,
                self.state.language
            )
            
            # Validate code
            valid, errors = self.translator.validate_translation(
                new_code,
                self.state.language
            )
            
            # Update state
            self.state.current_code = new_code
            self.state.errors = [{'message': err, 'line': 0} for err in errors]
            self.state.last_updated = datetime.now()
            
            # Parse sections
            self._parse_code_sections()
            
            # Trigger callbacks
            await self._trigger_code_change()
            
            if errors:
                await self._trigger_errors()
        
        except Exception as e:
            self.state.errors.append({
                'message': str(e),
                'line': 0,
                'type': 'generation_error'
            })
            await self._trigger_errors()
    
    def _parse_code_sections(self):
        """Parse code into sections based on blocks"""
        self.state.sections.clear()
        
        lines = self.state.current_code.splitlines()
        current_section = None
        
        for i, line in enumerate(lines, 1):
            # Simple heuristic - detect function/class definitions
            if line.strip().startswith(('def ', 'class ', 'function ', 'if ', 'for ', 'while ')):
                if current_section:
                    current_section.end_line = i - 1
                    self.state.sections.append(current_section)
                
                current_section = CodeSection(
                    start_line=i,
                    end_line=i,
                    block_id=f"section_{i}",
                    block_type=self._detect_block_type(line),
                    code=line,
                    indentation=len(line) - len(line.lstrip())
                )
        
        if current_section:
            current_section.end_line = len(lines)
            self.state.sections.append(current_section)
    
    def _detect_block_type(self, line: str) -> str:
        """Detect block type from code line"""
        line_stripped = line.strip()
        
        if line_stripped.startswith('def '):
            return 'function'
        elif line_stripped.startswith('class '):
            return 'class'
        elif line_stripped.startswith('if '):
            return 'condition'
        elif line_stripped.startswith('for '):
            return 'for_loop'
        elif line_stripped.startswith('while '):
            return 'while_loop'
        elif line_stripped.startswith('try:'):
            return 'try_catch'
        else:
            return 'statement'
    
    def get_highlighted_html(self) -> str:
        """Get HTML with syntax highlighting"""
        if not self.state.current_code:
            return "<pre>No code generated yet</pre>"
        
        highlighted = self.highlighter.highlight_code(
            self.state.current_code,
            self.state.language,
            format_type="html"
        )
        
        # Add error markers
        if self.state.errors:
            highlighted = self._add_error_markers(highlighted)
        
        # Add selection highlighting
        if self.state.selection:
            highlighted = self._add_selection(highlighted)
        
        return highlighted
    
    def _add_error_markers(self, html: str) -> str:
        """Add error markers to HTML"""
        lines = html.split('\n')
        
        for error in self.state.errors:
            line_num = error.get('line', 0)
            if 0 < line_num <= len(lines):
                lines[line_num - 1] = (
                    f'<span class="error-line" title="{error["message"]}">'
                    f'{lines[line_num - 1]}'
                    f'</span>'
                )
        
        return '\n'.join(lines)
    
    def _add_selection(self, html: str) -> str:
        """Add selection highlighting to HTML"""
        if not self.state.selection:
            return html
        
        start_line, start_col, end_line, end_col = self.state.selection
        lines = html.split('\n')
        
        for i in range(start_line - 1, min(end_line, len(lines))):
            lines[i] = f'<span class="selected-line">{lines[i]}</span>'
        
        return '\n'.join(lines)
    
    def get_terminal_output(self) -> str:
        """Get terminal-formatted output with colors"""
        if not self.state.current_code:
            return "No code generated yet"
        
        return self.highlighter.highlight_code(
            self.state.current_code,
            self.state.language,
            format_type="terminal"
        )
    
    def get_diff_view(self) -> str:
        """Get diff view between previous and current code"""
        if not self.state.previous_code:
            return self.state.current_code
        
        return self.differ.get_inline_diff(
            self.state.previous_code,
            self.state.current_code
        )
    
    def fold_region(self, start_line: int, end_line: int):
        """Fold a region of code"""
        self.state.folded_regions.append((start_line, end_line))
    
    def unfold_region(self, start_line: int):
        """Unfold a region starting at the given line"""
        self.state.folded_regions = [
            (s, e) for s, e in self.state.folded_regions
            if s != start_line
        ]
    
    def toggle_bookmark(self, line: int):
        """Toggle bookmark on a line"""
        if line in self.state.bookmarks:
            self.state.bookmarks.remove(line)
        else:
            self.state.bookmarks.append(line)
            self.state.bookmarks.sort()
    
    def go_to_line(self, line: int):
        """Move cursor to specific line"""
        max_line = len(self.state.current_code.splitlines())
        line = max(1, min(line, max_line))
        self.state.cursor_position = (line, 0)
    
    def find_in_code(self, search_term: str, case_sensitive: bool = False) -> List[Tuple[int, int]]:
        """Find occurrences of search term in code"""
        matches = []
        lines = self.state.current_code.splitlines()
        
        for i, line in enumerate(lines, 1):
            search_line = line if case_sensitive else line.lower()
            search_str = search_term if case_sensitive else search_term.lower()
            
            col = 0
            while True:
                pos = search_line.find(search_str, col)
                if pos == -1:
                    break
                matches.append((i, pos))
                col = pos + 1
        
        return matches
    
    def export_code(self, filepath: str):
        """Export generated code to file"""
        with open(filepath, 'w') as f:
            f.write(self.state.current_code)
    
    def copy_to_clipboard(self) -> str:
        """Get code for clipboard copy"""
        return self.state.current_code
    
    def get_minimap_data(self) -> Dict[str, Any]:
        """Get data for minimap visualization"""
        lines = self.state.current_code.splitlines()
        
        minimap_data = {
            'total_lines': len(lines),
            'sections': [
                {
                    'start': s.start_line,
                    'end': s.end_line,
                    'type': s.block_type
                }
                for s in self.state.sections
            ],
            'errors': [e['line'] for e in self.state.errors],
            'bookmarks': self.state.bookmarks,
            'folded': self.state.folded_regions,
            'cursor': self.state.cursor_position[0]
        }
        
        return minimap_data
    
    def get_breadcrumbs(self) -> List[str]:
        """Get breadcrumb navigation for current position"""
        line, _ = self.state.cursor_position
        breadcrumbs = []
        
        for section in self.state.sections:
            if section.start_line <= line <= section.end_line:
                breadcrumbs.append(f"{section.block_type}:{section.start_line}")
        
        return breadcrumbs
    
    async def _trigger_code_change(self):
        """Trigger code change callbacks"""
        for callback in self.on_code_change_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(self.state.current_code)
            else:
                callback(self.state.current_code)
    
    async def _trigger_errors(self):
        """Trigger error callbacks"""
        for callback in self.on_error_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(self.state.errors)
            else:
                callback(self.state.errors)
    
    async def start_auto_refresh(self):
        """Start auto-refresh task"""
        if self.refresh_task:
            self.refresh_task.cancel()
        
        async def refresh_loop():
            while self.settings.auto_refresh:
                await asyncio.sleep(self.settings.refresh_delay)
                if self.current_program:
                    await self.refresh_preview()
        
        self.refresh_task = asyncio.create_task(refresh_loop())
    
    def stop_auto_refresh(self):
        """Stop auto-refresh task"""
        if self.refresh_task:
            self.refresh_task.cancel()
            self.refresh_task = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get code statistics"""
        lines = self.state.current_code.splitlines()
        
        return {
            'total_lines': len(lines),
            'blank_lines': sum(1 for l in lines if not l.strip()),
            'comment_lines': sum(1 for l in lines if l.strip().startswith('#')),
            'code_lines': sum(1 for l in lines if l.strip() and not l.strip().startswith('#')),
            'sections': len(self.state.sections),
            'errors': len(self.state.errors),
            'warnings': len(self.state.warnings),
            'language': self.state.language.value,
            'last_updated': self.state.last_updated.isoformat()
        }


class CodePreviewUI:
    """UI wrapper for code preview panel"""
    
    def __init__(self, panel: CodePreviewPanel):
        self.panel = panel
    
    def generate_html(self) -> str:
        """Generate complete HTML for preview panel"""
        css = self.panel.highlighter.get_css()
        highlighted_code = self.panel.get_highlighted_html()
        stats = self.panel.get_stats()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        {css}
        .preview-container {{
            font-family: {self.panel.settings.font_family};
            font-size: {self.panel.settings.font_size}px;
            background: #272822;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 5px;
        }}
        .preview-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #444;
        }}
        .stats {{
            color: #75715e;
            font-size: 12px;
        }}
        .error-line {{
            background-color: rgba(255, 0, 0, 0.2);
            border-left: 3px solid red;
        }}
        .selected-line {{
            background-color: rgba(255, 255, 0, 0.1);
        }}
    </style>
</head>
<body>
    <div class="preview-container">
        <div class="preview-header">
            <span>Code Preview - {stats['language']}</span>
            <span class="stats">
                {stats['total_lines']} lines | 
                {stats['code_lines']} code | 
                {stats['errors']} errors
            </span>
        </div>
        <div class="code-content">
            {highlighted_code}
        </div>
    </div>
</body>
</html>
        """
        
        return html


# Test the code preview panel
def test_code_preview_panel():
    """Test the code preview panel"""
    print("\n" + "="*60)
    print("Code Preview Panel Test")
    print("="*60)
    
    from visual_code_builder import VisualProgram, BlockFactory
    
    # Create preview panel
    settings = PreviewSettings(
        theme=PreviewTheme.MONOKAI,
        line_numbers=True,
        auto_refresh=False
    )
    panel = CodePreviewPanel(settings)
    
    # Create a sample program
    program = VisualProgram(name="Preview Test")
    factory = BlockFactory()
    
    # Add some blocks
    func = factory.create_function_block()
    func.properties["function_name"] = "fibonacci"
    func.properties["parameters"] = ["n"]
    program.add_block(func)
    
    if_block = factory.create_if_block()
    if_block.properties["condition_expression"] = "n <= 1"
    program.add_block(if_block)
    
    output = factory.create_output_block()
    program.add_block(output)
    
    print(f"\n📦 Created program with {len(program.blocks)} blocks")
    
    # Update preview
    asyncio.run(panel.update_program(program))
    
    # Get terminal output
    print("\n🖥️ Terminal Preview:")
    print(panel.get_terminal_output())
    
    # Get stats
    stats = panel.get_stats()
    print(f"\n📊 Code Statistics:")
    print(f"   Total lines: {stats['total_lines']}")
    print(f"   Code lines: {stats['code_lines']}")
    print(f"   Sections: {stats['sections']}")
    
    # Search in code
    matches = panel.find_in_code("def")
    print(f"\n🔍 Found 'def' at {len(matches)} locations")
    
    # Export code
    panel.export_code("preview_output.py")
    print(f"\n💾 Exported code to preview_output.py")
    
    # Generate HTML
    ui = CodePreviewUI(panel)
    html = ui.generate_html()
    
    with open("code_preview.html", "w") as f:
        f.write(html)
    print(f"📄 Generated HTML preview saved to code_preview.html")
    
    return panel


if __name__ == "__main__":
    print("Code Preview Panel for Agent Lightning")
    print("="*60)
    
    panel = test_code_preview_panel()
    
    print("\n✅ Code Preview Panel ready!")