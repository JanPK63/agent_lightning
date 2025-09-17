#!/usr/bin/env python3
"""
Train agents on your custom codebase and documentation
Automatically extract knowledge from your project files
"""

import os
import json
import ast
from pathlib import Path
from typing import List, Dict, Optional
from agent_config import AgentConfigManager
from knowledge_manager import KnowledgeManager
import re


class CodebaseTrainer:
    """Train agents on custom codebases and documentation"""
    
    def __init__(self):
        self.config_manager = AgentConfigManager()
        self.knowledge_manager = KnowledgeManager()
        self.file_patterns = {
            # Code files
            ".py": self._extract_python_knowledge,
            ".js": self._extract_javascript_knowledge,
            ".jsx": self._extract_javascript_knowledge,
            ".ts": self._extract_typescript_knowledge,
            ".tsx": self._extract_typescript_knowledge,
            ".java": self._extract_java_knowledge,
            ".cpp": self._extract_cpp_knowledge,
            ".c": self._extract_c_knowledge,
            ".go": self._extract_go_knowledge,
            ".rs": self._extract_rust_knowledge,
            ".swift": self._extract_swift_knowledge,
            ".kt": self._extract_kotlin_knowledge,
            
            # Documentation
            ".md": self._extract_markdown_knowledge,
            ".rst": self._extract_rst_knowledge,
            ".txt": self._extract_text_knowledge,
            
            # Configuration
            ".json": self._extract_json_knowledge,
            ".yaml": self._extract_yaml_knowledge,
            ".yml": self._extract_yaml_knowledge,
            ".toml": self._extract_toml_knowledge,
            
            # Web
            ".html": self._extract_html_knowledge,
            ".css": self._extract_css_knowledge,
            ".scss": self._extract_scss_knowledge,
        }
    
    def train_on_directory(self, agent_name: str, directory_path: str, 
                          recursive: bool = True, file_extensions: Optional[List[str]] = None):
        """Train an agent on all files in a directory"""
        
        if agent_name not in self.config_manager.list_agents():
            print(f"❌ Agent '{agent_name}' not found")
            return
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"❌ Directory '{directory_path}' not found")
            return
        
        print(f"🚀 Training {agent_name} on {directory_path}")
        
        # Collect files
        files_to_process = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                # Skip hidden files and common ignore patterns
                if any(part.startswith('.') for part in file_path.parts[len(directory.parts):]):
                    continue
                if any(ignore in str(file_path) for ignore in ['node_modules', '__pycache__', 'venv', '.git']):
                    continue
                
                # Check file extension
                if file_extensions:
                    if file_path.suffix in file_extensions:
                        files_to_process.append(file_path)
                else:
                    if file_path.suffix in self.file_patterns:
                        files_to_process.append(file_path)
        
        print(f"📁 Found {len(files_to_process)} files to process")
        
        # Process each file
        knowledge_items_added = 0
        for file_path in files_to_process:
            try:
                knowledge_items = self._extract_knowledge_from_file(file_path)
                
                for item in knowledge_items:
                    self.knowledge_manager.add_knowledge(
                        agent_name=agent_name,
                        category=item["category"],
                        content=item["content"],
                        source=f"codebase:{file_path.name}",
                        metadata=item.get("metadata", {})
                    )
                    knowledge_items_added += 1
                
                print(f"  ✅ Processed {file_path.name} ({len(knowledge_items)} items)")
                
            except Exception as e:
                print(f"  ⚠️ Error processing {file_path.name}: {e}")
        
        print(f"\n✅ Training complete! Added {knowledge_items_added} knowledge items to {agent_name}")
        
        # Save knowledge base
        self.knowledge_manager.save_knowledge_base(agent_name)
        
        return knowledge_items_added
    
    def _extract_knowledge_from_file(self, file_path: Path) -> List[Dict]:
        """Extract knowledge from a single file"""
        
        if file_path.suffix in self.file_patterns:
            extractor = self.file_patterns[file_path.suffix]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return extractor(content, file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return []
    
    def _extract_python_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Python files"""
        knowledge_items = []
        
        try:
            tree = ast.parse(content)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        knowledge_items.append({
                            "category": "code_examples",
                            "content": f"Class {node.name}:\n{class_doc}\n\nMethods: {[m.name for m in node.body if isinstance(m, ast.FunctionDef)]}",
                            "metadata": {"type": "class", "file": str(file_path)}
                        })
                
                # Extract functions with docstrings
                elif isinstance(node, ast.FunctionDef):
                    func_doc = ast.get_docstring(node)
                    if func_doc:
                        # Get function signature
                        args = [arg.arg for arg in node.args.args]
                        signature = f"def {node.name}({', '.join(args)})"
                        
                        knowledge_items.append({
                            "category": "api_references",
                            "content": f"{signature}\n{func_doc}",
                            "metadata": {"type": "function", "file": str(file_path)}
                        })
        
        except SyntaxError:
            # If parsing fails, extract patterns
            patterns = self._extract_code_patterns(content)
            for pattern in patterns:
                knowledge_items.append({
                    "category": "code_examples",
                    "content": pattern,
                    "metadata": {"file": str(file_path)}
                })
        
        return knowledge_items
    
    def _extract_javascript_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from JavaScript/JSX files"""
        knowledge_items = []
        
        # Extract React components
        component_pattern = r'(?:function|const|class)\s+(\w+).*?(?:\{[\s\S]*?\n\}|\=\>[\s\S]*?\n\})'
        components = re.findall(component_pattern, content)
        
        for component in components[:5]:  # Limit to first 5
            knowledge_items.append({
                "category": "code_examples",
                "content": f"React component: {component}",
                "metadata": {"type": "component", "file": str(file_path)}
            })
        
        # Extract exports
        export_pattern = r'export\s+(?:default\s+)?(?:function|const|class)\s+(\w+)'
        exports = re.findall(export_pattern, content)
        
        if exports:
            knowledge_items.append({
                "category": "api_references",
                "content": f"Exports from {file_path.name}: {', '.join(exports)}",
                "metadata": {"type": "exports", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_typescript_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from TypeScript files"""
        knowledge_items = []
        
        # Extract interfaces
        interface_pattern = r'interface\s+(\w+)\s*\{([^}]*)\}'
        interfaces = re.findall(interface_pattern, content)
        
        for name, body in interfaces[:5]:
            knowledge_items.append({
                "category": "code_examples",
                "content": f"TypeScript interface {name}:\n{body}",
                "metadata": {"type": "interface", "file": str(file_path)}
            })
        
        # Extract types
        type_pattern = r'type\s+(\w+)\s*=\s*([^;]+);'
        types = re.findall(type_pattern, content)
        
        for name, definition in types[:5]:
            knowledge_items.append({
                "category": "code_examples",
                "content": f"Type {name} = {definition}",
                "metadata": {"type": "type_definition", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_java_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Java files"""
        knowledge_items = []
        
        # Extract class definitions
        class_pattern = r'(?:public\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?'
        classes = re.findall(class_pattern, content)
        
        for class_info in classes[:3]:
            class_name = class_info[0]
            extends = class_info[1] if class_info[1] else "None"
            implements = class_info[2] if class_info[2] else "None"
            
            knowledge_items.append({
                "category": "code_examples",
                "content": f"Java class {class_name}, extends: {extends}, implements: {implements}",
                "metadata": {"type": "class", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_cpp_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from C++ files"""
        return self._extract_c_knowledge(content, file_path)  # Similar patterns
    
    def _extract_c_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from C files"""
        knowledge_items = []
        
        # Extract function signatures
        func_pattern = r'(?:[\w\s\*]+)\s+(\w+)\s*\([^)]*\)\s*\{'
        functions = re.findall(func_pattern, content)
        
        if functions:
            knowledge_items.append({
                "category": "api_references",
                "content": f"Functions in {file_path.name}: {', '.join(functions[:10])}",
                "metadata": {"type": "functions", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_go_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Go files"""
        knowledge_items = []
        
        # Extract function definitions
        func_pattern = r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)'
        functions = re.findall(func_pattern, content)
        
        if functions:
            knowledge_items.append({
                "category": "api_references",
                "content": f"Go functions: {', '.join(functions[:10])}",
                "metadata": {"type": "functions", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_rust_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Rust files"""
        knowledge_items = []
        
        # Extract function and impl blocks
        impl_pattern = r'impl(?:<[^>]+>)?\s+(?:for\s+)?(\w+)'
        impls = re.findall(impl_pattern, content)
        
        for impl_name in impls[:3]:
            knowledge_items.append({
                "category": "code_examples",
                "content": f"Rust implementation for {impl_name}",
                "metadata": {"type": "impl", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_swift_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Swift files"""
        knowledge_items = []
        
        # Extract classes and structs
        pattern = r'(?:class|struct|protocol)\s+(\w+)'
        types = re.findall(pattern, content)
        
        if types:
            knowledge_items.append({
                "category": "code_examples",
                "content": f"Swift types in {file_path.name}: {', '.join(types[:10])}",
                "metadata": {"type": "types", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_kotlin_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Kotlin files"""
        knowledge_items = []
        
        # Extract classes and data classes
        class_pattern = r'(?:data\s+)?class\s+(\w+)'
        classes = re.findall(class_pattern, content)
        
        for class_name in classes[:5]:
            knowledge_items.append({
                "category": "code_examples",
                "content": f"Kotlin class: {class_name}",
                "metadata": {"type": "class", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_markdown_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from Markdown files"""
        knowledge_items = []
        
        # Extract headers and their content
        sections = re.split(r'\n(?=#)', content)
        
        for section in sections[:10]:  # Limit sections
            lines = section.strip().split('\n')
            if lines:
                header = lines[0].strip('#').strip()
                content_preview = '\n'.join(lines[1:4])  # First 3 lines
                
                if header and content_preview:
                    knowledge_items.append({
                        "category": "technical_documentation",
                        "content": f"{header}:\n{content_preview}",
                        "metadata": {"type": "documentation", "file": str(file_path)}
                    })
        
        return knowledge_items
    
    def _extract_rst_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from reStructuredText files"""
        return self._extract_text_knowledge(content, file_path)
    
    def _extract_text_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from plain text files"""
        knowledge_items = []
        
        # Take first 500 characters as a sample
        if len(content) > 100:
            knowledge_items.append({
                "category": "domain_knowledge",
                "content": content[:500],
                "metadata": {"type": "text", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_json_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from JSON files"""
        knowledge_items = []
        
        try:
            data = json.loads(content)
            
            # For package.json
            if file_path.name == "package.json":
                if "dependencies" in data:
                    deps = list(data["dependencies"].keys())[:20]
                    knowledge_items.append({
                        "category": "project_specific",
                        "content": f"Project dependencies: {', '.join(deps)}",
                        "metadata": {"type": "dependencies", "file": str(file_path)}
                    })
            
            # For other JSON configs
            elif len(str(data)) < 500:
                knowledge_items.append({
                    "category": "project_specific",
                    "content": f"Configuration in {file_path.name}: {json.dumps(data, indent=2)[:500]}",
                    "metadata": {"type": "config", "file": str(file_path)}
                })
        
        except json.JSONDecodeError:
            pass
        
        return knowledge_items
    
    def _extract_yaml_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from YAML files"""
        knowledge_items = []
        
        # For Docker Compose, CI/CD configs, etc.
        if any(keyword in file_path.name.lower() for keyword in ['docker', 'compose', 'ci', 'github', 'gitlab']):
            knowledge_items.append({
                "category": "project_specific",
                "content": f"Configuration file {file_path.name}:\n{content[:500]}",
                "metadata": {"type": "config", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_toml_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from TOML files"""
        return self._extract_yaml_knowledge(content, file_path)
    
    def _extract_html_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from HTML files"""
        knowledge_items = []
        
        # Extract meta tags
        meta_pattern = r'<meta\s+(?:name|property)="([^"]+)"\s+content="([^"]+)"'
        metas = re.findall(meta_pattern, content)
        
        if metas:
            meta_info = [f"{name}: {content}" for name, content in metas[:5]]
            knowledge_items.append({
                "category": "project_specific",
                "content": f"HTML metadata: {'; '.join(meta_info)}",
                "metadata": {"type": "html", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_css_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from CSS files"""
        knowledge_items = []
        
        # Extract class names
        class_pattern = r'\.([a-zA-Z][\w-]*)\s*\{'
        classes = re.findall(class_pattern, content)
        
        if classes:
            unique_classes = list(set(classes))[:20]
            knowledge_items.append({
                "category": "project_specific",
                "content": f"CSS classes in {file_path.name}: {', '.join(unique_classes)}",
                "metadata": {"type": "styles", "file": str(file_path)}
            })
        
        return knowledge_items
    
    def _extract_scss_knowledge(self, content: str, file_path: Path) -> List[Dict]:
        """Extract knowledge from SCSS files"""
        return self._extract_css_knowledge(content, file_path)
    
    def _extract_code_patterns(self, content: str) -> List[str]:
        """Extract common code patterns from any file"""
        patterns = []
        
        # Extract imports
        import_pattern = r'(?:import|from|require|use|include)\s+[^\n]+'
        imports = re.findall(import_pattern, content)
        if imports:
            patterns.append(f"Imports: {'; '.join(imports[:5])}")
        
        return patterns


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python train_agent_on_codebase.py <agent_name> <directory_path> [file_extensions]")
        print("\nExample:")
        print("  python train_agent_on_codebase.py full_stack_developer ./my-project")
        print("  python train_agent_on_codebase.py full_stack_developer ./my-project .py,.js,.md")
        return
    
    agent_name = sys.argv[1]
    directory_path = sys.argv[2]
    
    file_extensions = None
    if len(sys.argv) > 3:
        file_extensions = sys.argv[3].split(',')
    
    trainer = CodebaseTrainer()
    
    # First ensure the agent exists
    if agent_name not in trainer.config_manager.list_agents():
        print(f"Agent '{agent_name}' not found. Available agents:")
        for agent in trainer.config_manager.list_agents():
            print(f"  - {agent}")
        return
    
    # Train the agent
    items_added = trainer.train_on_directory(
        agent_name=agent_name,
        directory_path=directory_path,
        recursive=True,
        file_extensions=file_extensions
    )
    
    # Show statistics
    stats = trainer.knowledge_manager.get_statistics(agent_name)
    print(f"\n📊 Knowledge Base Statistics for {agent_name}:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Categories: {stats['categories']}")


if __name__ == "__main__":
    main()