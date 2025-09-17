#!/usr/bin/env python3
"""
Visual Git Diff Visualization for Agent Lightning
Visualize code changes and git diffs in the visual code builder
"""

import os
import sys
import json
import difflib
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import uuid
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    VisualProgram,
    BlockFactory
)
from visual_code_blocks import InteractiveBlock, VisualCanvas


class DiffType(Enum):
    """Types of diff operations"""
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    RENAMED = "renamed"
    UNCHANGED = "unchanged"


@dataclass
class DiffLine:
    """Represents a single line in a diff"""
    line_number_old: Optional[int] = None
    line_number_new: Optional[int] = None
    content: str = ""
    diff_type: DiffType = DiffType.UNCHANGED
    context: bool = False  # Is this a context line?


@dataclass
class FileDiff:
    """Represents diff for a single file"""
    file_path: str
    old_path: Optional[str] = None  # For renames
    diff_type: DiffType = DiffType.MODIFIED
    additions: int = 0
    deletions: int = 0
    lines: List[DiffLine] = field(default_factory=list)
    hunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VisualDiffBlock:
    """Visual representation of a diff block"""
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_diff: Optional[FileDiff] = None
    position: Tuple[float, float] = (0, 0)
    size: Tuple[float, float] = (200, 150)
    expanded: bool = False
    selected_lines: List[int] = field(default_factory=list)
    
    def get_color(self) -> str:
        """Get color based on diff type"""
        if not self.file_diff:
            return "#808080"
        
        colors = {
            DiffType.ADDED: "#28a745",      # Green
            DiffType.DELETED: "#dc3545",    # Red
            DiffType.MODIFIED: "#ffc107",   # Yellow
            DiffType.RENAMED: "#17a2b8",    # Cyan
            DiffType.UNCHANGED: "#6c757d"   # Gray
        }
        return colors.get(self.file_diff.diff_type, "#808080")


class GitDiffVisualizer:
    """Main git diff visualizer class"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.canvas = VisualCanvas()
        self.diff_blocks: List[VisualDiffBlock] = []
        self.current_commit = "HEAD"
        self.compare_commit = "HEAD~1"
        self.file_diffs: List[FileDiff] = []
        
    def get_git_diff(self, commit1: str = "HEAD~1", commit2: str = "HEAD") -> List[FileDiff]:
        """Get git diff between two commits"""
        try:
            # Get list of changed files
            cmd = ["git", "diff", "--name-status", commit1, commit2]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                print(f"Error getting git diff: {result.stderr}")
                return []
            
            file_diffs = []
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('\t')
                status = parts[0]
                
                if status == 'A':  # Added
                    file_diff = FileDiff(
                        file_path=parts[1],
                        diff_type=DiffType.ADDED
                    )
                elif status == 'D':  # Deleted
                    file_diff = FileDiff(
                        file_path=parts[1],
                        diff_type=DiffType.DELETED
                    )
                elif status == 'M':  # Modified
                    file_diff = FileDiff(
                        file_path=parts[1],
                        diff_type=DiffType.MODIFIED
                    )
                elif status.startswith('R'):  # Renamed
                    file_diff = FileDiff(
                        file_path=parts[2],
                        old_path=parts[1],
                        diff_type=DiffType.RENAMED
                    )
                else:
                    continue
                
                # Get detailed diff for the file
                self._get_file_diff_details(file_diff, commit1, commit2)
                file_diffs.append(file_diff)
            
            return file_diffs
            
        except Exception as e:
            print(f"Error getting git diff: {e}")
            return []
    
    def _get_file_diff_details(self, file_diff: FileDiff, commit1: str, commit2: str):
        """Get detailed diff for a specific file"""
        try:
            # Get unified diff
            cmd = ["git", "diff", commit1, commit2, "--", file_diff.file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0:
                return
            
            lines = result.stdout.split('\n')
            current_line_old = 0
            current_line_new = 0
            
            for line in lines:
                if line.startswith('@@'):
                    # Parse hunk header
                    parts = line.split('@@')
                    if len(parts) >= 2:
                        hunk_info = parts[1].strip()
                        # Parse line numbers from hunk header like "-10,7 +10,8"
                        try:
                            old_info, new_info = hunk_info.split(' ')
                            old_start = int(old_info.split(',')[0][1:])
                            new_start = int(new_info.split(',')[0][1:])
                            current_line_old = old_start
                            current_line_new = new_start
                        except:
                            pass
                        
                        file_diff.hunks.append({
                            'header': line,
                            'old_start': current_line_old,
                            'new_start': current_line_new
                        })
                
                elif line.startswith('+'):
                    # Added line
                    if not line.startswith('+++'):
                        diff_line = DiffLine(
                            line_number_new=current_line_new,
                            content=line[1:],
                            diff_type=DiffType.ADDED
                        )
                        file_diff.lines.append(diff_line)
                        file_diff.additions += 1
                        current_line_new += 1
                
                elif line.startswith('-'):
                    # Deleted line
                    if not line.startswith('---'):
                        diff_line = DiffLine(
                            line_number_old=current_line_old,
                            content=line[1:],
                            diff_type=DiffType.DELETED
                        )
                        file_diff.lines.append(diff_line)
                        file_diff.deletions += 1
                        current_line_old += 1
                
                elif line.startswith(' '):
                    # Context line
                    diff_line = DiffLine(
                        line_number_old=current_line_old,
                        line_number_new=current_line_new,
                        content=line[1:],
                        diff_type=DiffType.UNCHANGED,
                        context=True
                    )
                    file_diff.lines.append(diff_line)
                    current_line_old += 1
                    current_line_new += 1
                    
        except Exception as e:
            print(f"Error getting file diff details: {e}")
    
    def create_visual_diff(self, commit1: str = "HEAD~1", commit2: str = "HEAD"):
        """Create visual representation of git diff"""
        self.current_commit = commit2
        self.compare_commit = commit1
        self.file_diffs = self.get_git_diff(commit1, commit2)
        
        # Clear existing blocks
        self.diff_blocks.clear()
        
        # Create visual blocks for each file diff
        x_offset = 50
        y_offset = 50
        max_per_row = 4
        
        for i, file_diff in enumerate(self.file_diffs):
            row = i // max_per_row
            col = i % max_per_row
            
            x = x_offset + col * 250
            y = y_offset + row * 200
            
            visual_block = VisualDiffBlock(
                file_diff=file_diff,
                position=(x, y)
            )
            
            self.diff_blocks.append(visual_block)
    
    def generate_visual_program(self) -> VisualProgram:
        """Generate a visual program from the diff"""
        program = VisualProgram(name=f"Git Diff: {self.compare_commit}..{self.current_commit}")
        factory = BlockFactory()
        
        for diff_block in self.diff_blocks:
            if not diff_block.file_diff:
                continue
            
            # Create a visual block for each file change
            if diff_block.file_diff.diff_type == DiffType.ADDED:
                block = factory.create_file_write_block()
                block.properties["file_path"] = diff_block.file_diff.file_path
                block.properties["operation"] = "create"
            
            elif diff_block.file_diff.diff_type == DiffType.DELETED:
                block = factory.create_expression_block()
                block.properties["expression"] = f"delete_file('{diff_block.file_diff.file_path}')"
            
            elif diff_block.file_diff.diff_type == DiffType.MODIFIED:
                block = factory.create_file_write_block()
                block.properties["file_path"] = diff_block.file_diff.file_path
                block.properties["operation"] = "modify"
                block.properties["changes"] = {
                    "additions": diff_block.file_diff.additions,
                    "deletions": diff_block.file_diff.deletions
                }
            
            elif diff_block.file_diff.diff_type == DiffType.RENAMED:
                block = factory.create_expression_block()
                old_path = diff_block.file_diff.old_path or "unknown"
                new_path = diff_block.file_diff.file_path
                block.properties["expression"] = f"rename_file('{old_path}', '{new_path}')"
            
            else:
                continue
            
            # Set visual properties
            block.position = diff_block.position
            block.metadata["diff_type"] = diff_block.file_diff.diff_type.value
            block.metadata["additions"] = diff_block.file_diff.additions
            block.metadata["deletions"] = diff_block.file_diff.deletions
            
            program.add_block(block)
        
        return program
    
    def generate_html(self) -> str:
        """Generate HTML visualization of git diff"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Git Diff Visualization</title>
    <style>
        body {{
            font-family: 'Monaco', 'Courier New', monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            margin: 0;
            padding: 20px;
        }}
        
        .diff-header {{
            background: #2d2d30;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        
        .diff-header h1 {{
            margin: 0 0 10px 0;
            color: #4fc3f7;
        }}
        
        .diff-stats {{
            display: flex;
            gap: 20px;
            font-size: 14px;
        }}
        
        .stat {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .additions {{ color: #28a745; }}
        .deletions {{ color: #dc3545; }}
        .modifications {{ color: #ffc107; }}
        
        .diff-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .file-diff {{
            background: #2d2d30;
            border-radius: 5px;
            padding: 15px;
            border-left: 4px solid #4fc3f7;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .file-diff:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        
        .file-diff.added {{ border-left-color: #28a745; }}
        .file-diff.deleted {{ border-left-color: #dc3545; }}
        .file-diff.modified {{ border-left-color: #ffc107; }}
        .file-diff.renamed {{ border-left-color: #17a2b8; }}
        
        .file-name {{
            font-weight: bold;
            margin-bottom: 10px;
            word-break: break-all;
        }}
        
        .file-stats {{
            display: flex;
            gap: 15px;
            font-size: 12px;
            margin-top: 10px;
        }}
        
        .diff-lines {{
            max-height: 200px;
            overflow-y: auto;
            margin-top: 10px;
            padding: 10px;
            background: #1e1e1e;
            border-radius: 3px;
            font-size: 12px;
            display: none;
        }}
        
        .diff-lines.expanded {{
            display: block;
        }}
        
        .diff-line {{
            margin: 2px 0;
            padding: 2px 5px;
            white-space: pre;
            overflow-x: auto;
        }}
        
        .diff-line.added {{
            background: rgba(40, 167, 69, 0.2);
            color: #28a745;
        }}
        
        .diff-line.deleted {{
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }}
        
        .diff-line.context {{
            color: #6c757d;
        }}
        
        .line-number {{
            display: inline-block;
            width: 40px;
            text-align: right;
            margin-right: 10px;
            color: #6c757d;
        }}
        
        .expand-button {{
            background: #0e639c;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            margin-top: 10px;
        }}
        
        .expand-button:hover {{
            background: #1177bb;
        }}
    </style>
</head>
<body>
    <div class="diff-header">
        <h1>🔄 Git Diff Visualization</h1>
        <div class="diff-stats">
            <div class="stat">
                <span>📍 Comparing:</span>
                <code>{self.compare_commit} → {self.current_commit}</code>
            </div>
            <div class="stat additions">
                <span>+</span>
                <span>{sum(f.additions for f in self.file_diffs)} additions</span>
            </div>
            <div class="stat deletions">
                <span>-</span>
                <span>{sum(f.deletions for f in self.file_diffs)} deletions</span>
            </div>
            <div class="stat modifications">
                <span>📝</span>
                <span>{len(self.file_diffs)} files changed</span>
            </div>
        </div>
    </div>
    
    <div class="diff-container">
"""
        
        for diff_block in self.diff_blocks:
            if not diff_block.file_diff:
                continue
            
            file_diff = diff_block.file_diff
            diff_type_class = file_diff.diff_type.value
            
            html += f"""
        <div class="file-diff {diff_type_class}" onclick="toggleDiff('{diff_block.block_id}')">
            <div class="file-name">
                {self._get_file_icon(file_diff.file_path)} {file_diff.file_path}
            </div>
            <div class="file-stats">
                <span class="additions">+{file_diff.additions}</span>
                <span class="deletions">-{file_diff.deletions}</span>
                <span>{file_diff.diff_type.value.upper()}</span>
            </div>
            <button class="expand-button" onclick="event.stopPropagation(); toggleDiff('{diff_block.block_id}')">
                View Changes
            </button>
            <div class="diff-lines" id="diff-{diff_block.block_id}">
"""
            
            # Add diff lines
            for line in file_diff.lines[:50]:  # Limit to first 50 lines for performance
                line_class = "context" if line.context else line.diff_type.value
                line_num = ""
                
                if line.line_number_old:
                    line_num = f"{line.line_number_old}"
                elif line.line_number_new:
                    line_num = f"{line.line_number_new}"
                
                # Escape HTML
                content = line.content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                html += f"""
                <div class="diff-line {line_class}">
                    <span class="line-number">{line_num}</span>{content}
                </div>
"""
            
            if len(file_diff.lines) > 50:
                html += f"""
                <div class="diff-line context">
                    ... {len(file_diff.lines) - 50} more lines ...
                </div>
"""
            
            html += """
            </div>
        </div>
"""
        
        html += """
    </div>
    
    <script>
        function toggleDiff(blockId) {
            const diffElement = document.getElementById('diff-' + blockId);
            if (diffElement) {
                diffElement.classList.toggle('expanded');
            }
        }
    </script>
</body>
</html>
"""
        
        return html
    
    def _get_file_icon(self, file_path: str) -> str:
        """Get icon for file type"""
        ext = Path(file_path).suffix.lower()
        
        icons = {
            '.py': '🐍',
            '.js': '📜',
            '.ts': '📘',
            '.html': '🌐',
            '.css': '🎨',
            '.json': '📋',
            '.md': '📝',
            '.yaml': '⚙️',
            '.yml': '⚙️',
            '.sh': '🔧',
            '.sql': '🗄️',
            '.go': '🐹',
            '.java': '☕',
            '.cpp': '⚡',
            '.c': '⚡',
            '.rs': '🦀',
            '.rb': '💎',
            '.php': '🐘'
        }
        
        return icons.get(ext, '📄')
    
    def export_diff_data(self, filepath: str):
        """Export diff data to JSON"""
        data = {
            'compare_from': self.compare_commit,
            'compare_to': self.current_commit,
            'timestamp': datetime.now().isoformat(),
            'files': []
        }
        
        for file_diff in self.file_diffs:
            file_data = {
                'path': file_diff.file_path,
                'old_path': file_diff.old_path,
                'type': file_diff.diff_type.value,
                'additions': file_diff.additions,
                'deletions': file_diff.deletions,
                'hunks': file_diff.hunks,
                'lines': [
                    {
                        'old_line': line.line_number_old,
                        'new_line': line.line_number_new,
                        'content': line.content,
                        'type': line.diff_type.value,
                        'context': line.context
                    }
                    for line in file_diff.lines
                ]
            }
            data['files'].append(file_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def test_git_diff_visualizer():
    """Test the git diff visualizer"""
    print("\n" + "="*60)
    print("Git Diff Visualizer Test")
    print("="*60)
    
    visualizer = GitDiffVisualizer()
    
    # Create visual diff
    print("\n📊 Creating visual diff for HEAD~1..HEAD")
    visualizer.create_visual_diff()
    
    if visualizer.file_diffs:
        print(f"\n📁 Found {len(visualizer.file_diffs)} file changes:")
        for file_diff in visualizer.file_diffs[:5]:  # Show first 5
            icon = "➕" if file_diff.diff_type == DiffType.ADDED else \
                   "➖" if file_diff.diff_type == DiffType.DELETED else \
                   "📝" if file_diff.diff_type == DiffType.MODIFIED else "↔️"
            print(f"   {icon} {file_diff.file_path}")
            print(f"      +{file_diff.additions} -{file_diff.deletions}")
        
        # Generate visual program
        program = visualizer.generate_visual_program()
        print(f"\n🎨 Generated visual program with {len(program.blocks)} blocks")
        
        # Generate HTML
        html = visualizer.generate_html()
        with open("git_diff_visualization.html", "w") as f:
            f.write(html)
        print(f"📄 Generated HTML visualization: git_diff_visualization.html")
        
        # Export data
        visualizer.export_diff_data("git_diff_data.json")
        print(f"💾 Exported diff data: git_diff_data.json")
    else:
        print("\n❌ No changes found between commits")
    
    return visualizer


if __name__ == "__main__":
    print("Visual Git Diff Visualization for Agent Lightning")
    print("="*60)
    
    visualizer = test_git_diff_visualizer()
    
    print("\n✅ Git Diff Visualizer ready!")
    print("\nFeatures:")
    print("  • Visual representation of git diffs")
    print("  • Color-coded file changes")
    print("  • Interactive diff exploration")
    print("  • Integration with visual code builder")
    print("  • Export diff data for analysis")