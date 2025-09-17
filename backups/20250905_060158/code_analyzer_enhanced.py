#!/usr/bin/env python3
"""
Enhanced Code Analyzer for Test Generation
Deep code analysis with dependency tracking, flow analysis, and pattern detection
"""

import os
import sys
import ast
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import networkx as nx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CodeComplexity(Enum):
    """Code complexity levels"""
    TRIVIAL = "trivial"      # < 5 cyclomatic complexity
    SIMPLE = "simple"        # 5-10 
    MODERATE = "moderate"    # 10-20
    COMPLEX = "complex"      # 20-50
    VERY_COMPLEX = "very_complex"  # > 50


class DependencyType(Enum):
    """Types of code dependencies"""
    IMPORT = "import"
    FUNCTION_CALL = "function_call"
    CLASS_INHERITANCE = "class_inheritance"
    VARIABLE_REFERENCE = "variable_reference"
    ATTRIBUTE_ACCESS = "attribute_access"
    METHOD_CALL = "method_call"


@dataclass
class CodeMetrics:
    """Metrics for code analysis"""
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    dependencies_count: int = 0
    test_coverage_estimate: float = 0.0


@dataclass
class DataFlow:
    """Data flow information"""
    variable: str
    definition_line: int
    usage_lines: List[int] = field(default_factory=list)
    modifications: List[int] = field(default_factory=list)
    scope: str = "global"
    type_hints: Optional[str] = None
    possible_values: List[Any] = field(default_factory=list)


@dataclass
class ControlFlow:
    """Control flow information"""
    entry_points: List[int] = field(default_factory=list)
    exit_points: List[int] = field(default_factory=list)
    branches: List[Tuple[int, int, str]] = field(default_factory=list)  # (from_line, to_line, condition)
    loops: List[Tuple[int, int, str]] = field(default_factory=list)  # (start_line, end_line, type)
    exception_handlers: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Dependency:
    """Code dependency information"""
    source: str
    target: str
    dep_type: DependencyType
    line_number: int
    context: str = ""


@dataclass
class Pattern:
    """Code pattern detection"""
    pattern_type: str
    location: int
    description: str
    severity: str = "info"  # info, warning, error
    suggestion: str = ""


class EnhancedCodeAnalyzer:
    """Enhanced code analyzer with deep analysis capabilities"""
    
    def __init__(self):
        self.ast_tree = None
        self.source_code = ""
        self.metrics = CodeMetrics()
        self.data_flows = {}
        self.control_flow = ControlFlow()
        self.dependencies = []
        self.patterns = []
        self.call_graph = nx.DiGraph()
        self.symbol_table = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive file analysis"""
        try:
            with open(file_path, 'r') as f:
                self.source_code = f.read()
            
            self.ast_tree = ast.parse(self.source_code, filename=file_path)
            
            # Run all analysis phases
            analysis = {
                "file_path": file_path,
                "metrics": self._calculate_metrics(),
                "structure": self._analyze_structure(),
                "data_flow": self._analyze_data_flow(),
                "control_flow": self._analyze_control_flow(),
                "dependencies": self._analyze_dependencies(),
                "patterns": self._detect_patterns(),
                "call_graph": self._build_call_graph(),
                "test_suggestions": self._generate_test_suggestions(),
                "complexity_hotspots": self._find_complexity_hotspots(),
                "refactoring_opportunities": self._find_refactoring_opportunities()
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_metrics(self) -> CodeMetrics:
        """Calculate various code metrics"""
        self.metrics.lines_of_code = len(self.source_code.split('\n'))
        
        # Cyclomatic complexity
        self.metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(self.ast_tree)
        
        # Cognitive complexity
        self.metrics.cognitive_complexity = self._calculate_cognitive_complexity(self.ast_tree)
        
        # Halstead metrics
        operators, operands = self._extract_halstead_elements(self.ast_tree)
        self.metrics.halstead_volume = self._calculate_halstead_volume(operators, operands)
        self.metrics.halstead_difficulty = self._calculate_halstead_difficulty(operators, operands)
        
        # Maintainability index
        self.metrics.maintainability_index = self._calculate_maintainability_index()
        
        return self.metrics
    
    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze code structure"""
        structure = {
            "classes": [],
            "functions": [],
            "async_functions": [],
            "decorators": [],
            "imports": [],
            "global_variables": [],
            "constants": [],
            "type_annotations": []
        }
        
        class StructureVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.current_class = None
                self.current_function = None
                self.scope_stack = []
                
            def visit_ClassDef(self, node):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [],
                    "attributes": [],
                    "bases": [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in node.bases],
                    "decorators": [ast.unparse(dec) if hasattr(ast, 'unparse') else str(dec) for dec in node.decorator_list],
                    "docstring": ast.get_docstring(node),
                    "complexity": self.analyzer._calculate_cyclomatic_complexity(node),
                    "is_dataclass": any('dataclass' in str(dec) for dec in node.decorator_list),
                    "is_abstract": any(isinstance(base, ast.Name) and base.id == 'ABC' for base in node.bases)
                }
                
                old_class = self.current_class
                self.current_class = class_info
                self.scope_stack.append(node.name)
                
                # Visit methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = self.analyzer._extract_function_details(item)
                        method_info["is_method"] = True
                        method_info["class_name"] = node.name
                        
                        # Identify special methods
                        if item.name.startswith('__') and item.name.endswith('__'):
                            method_info["is_special"] = True
                        if item.name == '__init__':
                            method_info["is_constructor"] = True
                        if 'staticmethod' in str(item.decorator_list):
                            method_info["is_static"] = True
                        if 'classmethod' in str(item.decorator_list):
                            method_info["is_class_method"] = True
                        if 'property' in str(item.decorator_list):
                            method_info["is_property"] = True
                            
                        class_info["methods"].append(method_info)
                
                structure["classes"].append(class_info)
                self.generic_visit(node)
                self.scope_stack.pop()
                self.current_class = old_class
                
            def visit_FunctionDef(self, node):
                if self.current_class is None:  # Only top-level functions
                    func_info = self.analyzer._extract_function_details(node)
                    structure["functions"].append(func_info)
                self.generic_visit(node)
                
            def visit_AsyncFunctionDef(self, node):
                if self.current_class is None:  # Only top-level async functions
                    func_info = self.analyzer._extract_function_details(node)
                    func_info["is_async"] = True
                    structure["async_functions"].append(func_info)
                self.generic_visit(node)
                
            def visit_Import(self, node):
                for alias in node.names:
                    structure["imports"].append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
                    
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    structure["imports"].append({
                        "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                        "from": node.module,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
                    
            def visit_Assign(self, node):
                # Detect global variables and constants
                if len(self.scope_stack) == 0:  # Global scope
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_info = {
                                "name": target.id,
                                "line": node.lineno,
                                "value": ast.unparse(node.value) if hasattr(ast, 'unparse') else None
                            }
                            
                            # Check if it's a constant (UPPER_CASE naming)
                            if target.id.isupper():
                                structure["constants"].append(var_info)
                            else:
                                structure["global_variables"].append(var_info)
                self.generic_visit(node)
                
            def visit_AnnAssign(self, node):
                # Type annotated assignments
                if isinstance(node.target, ast.Name):
                    structure["type_annotations"].append({
                        "name": node.target.id,
                        "type": ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation),
                        "line": node.lineno
                    })
                self.generic_visit(node)
        
        visitor = StructureVisitor(self)
        visitor.visit(self.ast_tree)
        
        return structure
    
    def _extract_function_details(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract detailed function information"""
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "end_line": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            "parameters": [],
            "return_type": None,
            "docstring": ast.get_docstring(node),
            "decorators": [ast.unparse(dec) if hasattr(ast, 'unparse') else str(dec) for dec in node.decorator_list],
            "complexity": self._calculate_cyclomatic_complexity(node),
            "calls": [],
            "is_generator": self._is_generator(node),
            "is_recursive": self._is_recursive(node),
            "has_side_effects": self._has_side_effects(node),
            "pure_function": self._is_pure_function(node)
        }
        
        # Extract parameters with defaults and type hints
        defaults = node.args.defaults
        num_defaults = len(defaults)
        num_args = len(node.args.args)
        
        for i, arg in enumerate(node.args.args):
            param = {
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation and hasattr(ast, 'unparse') else None,
                "has_default": i >= (num_args - num_defaults),
                "default": None
            }
            
            if param["has_default"]:
                default_index = i - (num_args - num_defaults)
                param["default"] = ast.unparse(defaults[default_index]) if hasattr(ast, 'unparse') else str(defaults[default_index])
                
            func_info["parameters"].append(param)
        
        # Extract return type
        if node.returns:
            func_info["return_type"] = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Extract function calls
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    func_info["calls"].append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    func_info["calls"].append(f"{ast.unparse(child.func.value) if hasattr(ast, 'unparse') else '?'}.{child.func.attr}")
        
        return func_info
    
    def _analyze_data_flow(self) -> Dict[str, DataFlow]:
        """Analyze data flow through the code"""
        data_flows = {}
        
        class DataFlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_scope = "global"
                self.scope_stack = []
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in data_flows:
                            data_flows[target.id] = DataFlow(
                                variable=target.id,
                                definition_line=node.lineno,
                                scope=self.current_scope
                            )
                        else:
                            data_flows[target.id].modifications.append(node.lineno)
                            
                        # Try to infer possible values
                        if isinstance(node.value, ast.Constant):
                            data_flows[target.id].possible_values.append(node.value.value)
                            
                self.generic_visit(node)
                
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id in data_flows:
                    data_flows[node.id].usage_lines.append(node.lineno)
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                old_scope = self.current_scope
                self.current_scope = node.name
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
                self.current_scope = old_scope
                
        visitor = DataFlowVisitor()
        visitor.visit(self.ast_tree)
        
        self.data_flows = data_flows
        return data_flows
    
    def _analyze_control_flow(self) -> ControlFlow:
        """Analyze control flow"""
        control_flow = ControlFlow()
        
        class ControlFlowVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                control_flow.entry_points.append(node.lineno)
                
                # Find return statements
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        control_flow.exit_points.append(child.lineno)
                        
                self.generic_visit(node)
                
            def visit_If(self, node):
                # Record branch
                control_flow.branches.append((
                    node.lineno,
                    node.body[0].lineno if node.body else node.lineno + 1,
                    ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
                ))
                
                if node.orelse:
                    control_flow.branches.append((
                        node.lineno,
                        node.orelse[0].lineno,
                        f"not ({ast.unparse(node.test) if hasattr(ast, 'unparse') else 'condition'})"
                    ))
                    
                self.generic_visit(node)
                
            def visit_For(self, node):
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 1
                control_flow.loops.append((node.lineno, end_line, "for"))
                self.generic_visit(node)
                
            def visit_While(self, node):
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 1
                control_flow.loops.append((node.lineno, end_line, "while"))
                self.generic_visit(node)
                
            def visit_Try(self, node):
                for handler in node.handlers:
                    control_flow.exception_handlers.append((node.lineno, handler.lineno))
                self.generic_visit(node)
                
        visitor = ControlFlowVisitor()
        visitor.visit(self.ast_tree)
        
        self.control_flow = control_flow
        return control_flow
    
    def _analyze_dependencies(self) -> List[Dependency]:
        """Analyze code dependencies"""
        dependencies = []
        
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_context = "global"
                
            def visit_Import(self, node):
                for alias in node.names:
                    dependencies.append(Dependency(
                        source=self.current_context,
                        target=alias.name,
                        dep_type=DependencyType.IMPORT,
                        line_number=node.lineno
                    ))
                    
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    dependencies.append(Dependency(
                        source=self.current_context,
                        target=node.func.id,
                        dep_type=DependencyType.FUNCTION_CALL,
                        line_number=node.lineno
                    ))
                elif isinstance(node.func, ast.Attribute):
                    dependencies.append(Dependency(
                        source=self.current_context,
                        target=node.func.attr,
                        dep_type=DependencyType.METHOD_CALL,
                        line_number=node.lineno
                    ))
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        dependencies.append(Dependency(
                            source=node.name,
                            target=base.id,
                            dep_type=DependencyType.CLASS_INHERITANCE,
                            line_number=node.lineno
                        ))
                        
                old_context = self.current_context
                self.current_context = node.name
                self.generic_visit(node)
                self.current_context = old_context
                
            def visit_FunctionDef(self, node):
                old_context = self.current_context
                self.current_context = f"{self.current_context}.{node.name}"
                self.generic_visit(node)
                self.current_context = old_context
                
        visitor = DependencyVisitor()
        visitor.visit(self.ast_tree)
        
        self.dependencies = dependencies
        return dependencies
    
    def _detect_patterns(self) -> List[Pattern]:
        """Detect code patterns and anti-patterns"""
        patterns = []
        
        # Long method detection
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno'):
                    method_length = node.end_lineno - node.lineno
                    if method_length > 50:
                        patterns.append(Pattern(
                            pattern_type="long_method",
                            location=node.lineno,
                            description=f"Method '{node.name}' is {method_length} lines long",
                            severity="warning",
                            suggestion="Consider breaking this method into smaller functions"
                        ))
        
        # Nested loops detection
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        patterns.append(Pattern(
                            pattern_type="nested_loops",
                            location=node.lineno,
                            description="Nested loops detected",
                            severity="info",
                            suggestion="Consider extracting inner loop to a separate function"
                        ))
                        break
        
        # Magic numbers detection
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in [0, 1, -1, 2, 10, 100, 1000]:  # Common acceptable values
                    patterns.append(Pattern(
                        pattern_type="magic_number",
                        location=node.lineno,
                        description=f"Magic number {node.value} detected",
                        severity="info",
                        suggestion="Consider extracting to a named constant"
                    ))
        
        # Exception swallowing
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == 'Exception'):
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        patterns.append(Pattern(
                            pattern_type="exception_swallowing",
                            location=node.lineno,
                            description="Exception being swallowed without handling",
                            severity="warning",
                            suggestion="Log or handle the exception appropriately"
                        ))
        
        self.patterns = patterns
        return patterns
    
    def _build_call_graph(self) -> Dict[str, List[str]]:
        """Build function call graph"""
        call_graph = defaultdict(list)
        
        class CallGraphVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                
                # Find all function calls within this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            call_graph[self.current_function].append(child.func.id)
                            
                self.current_function = old_function
                
        visitor = CallGraphVisitor()
        visitor.visit(self.ast_tree)
        
        # Convert to networkx graph for analysis
        for caller, callees in call_graph.items():
            for callee in callees:
                self.call_graph.add_edge(caller, callee)
        
        return dict(call_graph)
    
    def _generate_test_suggestions(self) -> List[Dict[str, Any]]:
        """Generate intelligent test suggestions based on analysis"""
        suggestions = []
        
        # Suggest tests for complex functions
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                
                if complexity > 10:
                    suggestions.append({
                        "type": "high_complexity_test",
                        "function": node.name,
                        "reason": f"High complexity ({complexity})",
                        "suggested_tests": [
                            "Test all branch conditions",
                            "Test boundary values",
                            "Test error paths",
                            "Consider property-based testing"
                        ]
                    })
                    
                # Check for recursive functions
                if self._is_recursive(node):
                    suggestions.append({
                        "type": "recursive_test",
                        "function": node.name,
                        "reason": "Recursive function detected",
                        "suggested_tests": [
                            "Test base case",
                            "Test recursive case",
                            "Test stack overflow protection",
                            "Test with large inputs"
                        ]
                    })
                    
                # Check for functions with side effects
                if self._has_side_effects(node):
                    suggestions.append({
                        "type": "side_effect_test",
                        "function": node.name,
                        "reason": "Function has side effects",
                        "suggested_tests": [
                            "Mock external dependencies",
                            "Test state changes",
                            "Verify side effects occur",
                            "Test rollback scenarios"
                        ]
                    })
        
        # Suggest integration tests for highly connected functions
        if self.call_graph.nodes():
            centrality = nx.degree_centrality(self.call_graph) if len(self.call_graph) > 0 else {}
            for func, score in centrality.items():
                if score > 0.5:
                    suggestions.append({
                        "type": "integration_test",
                        "function": func,
                        "reason": f"Highly connected function (centrality: {score:.2f})",
                        "suggested_tests": [
                            "Integration test with dependencies",
                            "Test interaction patterns",
                            "Mock external calls"
                        ]
                    })
        
        return suggestions
    
    def _find_complexity_hotspots(self) -> List[Dict[str, Any]]:
        """Find areas of high complexity that need attention"""
        hotspots = []
        
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 15:
                    hotspots.append({
                        "type": "function",
                        "name": node.name,
                        "line": node.lineno,
                        "complexity": complexity,
                        "risk": "high" if complexity > 30 else "medium",
                        "recommendation": "Consider refactoring into smaller functions"
                    })
                    
            elif isinstance(node, ast.ClassDef):
                class_complexity = sum(
                    self._calculate_cyclomatic_complexity(item)
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                )
                if class_complexity > 50:
                    hotspots.append({
                        "type": "class",
                        "name": node.name,
                        "line": node.lineno,
                        "complexity": class_complexity,
                        "risk": "high" if class_complexity > 100 else "medium",
                        "recommendation": "Consider splitting into multiple classes"
                    })
        
        return sorted(hotspots, key=lambda x: x["complexity"], reverse=True)
    
    def _find_refactoring_opportunities(self) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities"""
        opportunities = []
        
        # Find duplicate code patterns (simplified)
        function_bodies = {}
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                body_str = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                if body_str in function_bodies:
                    opportunities.append({
                        "type": "duplicate_code",
                        "functions": [function_bodies[body_str], node.name],
                        "suggestion": "Extract common code to shared function"
                    })
                else:
                    function_bodies[body_str] = node.name
        
        # Find long parameter lists
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    opportunities.append({
                        "type": "long_parameter_list",
                        "function": node.name,
                        "parameter_count": len(node.args.args),
                        "suggestion": "Consider using a configuration object or builder pattern"
                    })
        
        # Find classes with too many methods
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                if method_count > 20:
                    opportunities.append({
                        "type": "large_class",
                        "class": node.name,
                        "method_count": method_count,
                        "suggestion": "Consider splitting responsibilities using composition"
                    })
        
        return opportunities
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)"""
        complexity = 0
        nesting_level = 0
        
        class CognitiveComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0
                
            def visit_If(self, node):
                self.complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_While(self, node):
                self.complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_For(self, node):
                self.complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_ExceptHandler(self, node):
                self.complexity += (1 + self.nesting_level)
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1
                
        visitor = CognitiveComplexityVisitor()
        visitor.visit(node)
        return visitor.complexity
    
    def _extract_halstead_elements(self, node: ast.AST) -> Tuple[List[str], List[str]]:
        """Extract operators and operands for Halstead metrics"""
        operators = []
        operands = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp):
                operators.append(type(child.op).__name__)
            elif isinstance(child, ast.UnaryOp):
                operators.append(type(child.op).__name__)
            elif isinstance(child, ast.BoolOp):
                operators.append(type(child.op).__name__)
            elif isinstance(child, ast.Compare):
                for op in child.ops:
                    operators.append(type(op).__name__)
            elif isinstance(child, ast.Name):
                operands.append(child.id)
            elif isinstance(child, ast.Constant):
                operands.append(str(child.value))
                
        return operators, operands
    
    def _calculate_halstead_volume(self, operators: List[str], operands: List[str]) -> float:
        """Calculate Halstead volume"""
        n1 = len(set(operators))  # Unique operators
        n2 = len(set(operands))    # Unique operands
        N1 = len(operators)        # Total operators
        N2 = len(operands)         # Total operands
        
        if n1 + n2 == 0:
            return 0
            
        n = n1 + n2  # Program vocabulary
        N = N1 + N2  # Program length
        
        import math
        return N * math.log2(n) if n > 0 else 0
    
    def _calculate_halstead_difficulty(self, operators: List[str], operands: List[str]) -> float:
        """Calculate Halstead difficulty"""
        n1 = len(set(operators))  # Unique operators
        n2 = len(set(operands))    # Unique operands
        N2 = len(operands)         # Total operands
        
        if n2 == 0:
            return 0
            
        return (n1 / 2) * (N2 / n2)
    
    def _calculate_maintainability_index(self) -> float:
        """Calculate maintainability index"""
        # Simplified version of the maintainability index
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        import math
        
        HV = self.metrics.halstead_volume
        CC = self.metrics.cyclomatic_complexity
        LOC = self.metrics.lines_of_code
        
        if HV <= 0 or LOC <= 0:
            return 100  # Maximum maintainability for trivial code
            
        MI = 171 - 5.2 * math.log(HV) - 0.23 * CC - 16.2 * math.log(LOC)
        
        # Normalize to 0-100 scale
        return max(0, min(100, MI))
    
    def _is_generator(self, node: ast.FunctionDef) -> bool:
        """Check if function is a generator"""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _is_recursive(self, node: ast.FunctionDef) -> bool:
        """Check if function is recursive"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == node.name:
                    return True
        return False
    
    def _has_side_effects(self, node: ast.FunctionDef) -> bool:
        """Check if function has side effects"""
        for child in ast.walk(node):
            # Check for print statements
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ['print', 'open', 'write']:
                    return True
            # Check for global variable modifications
            elif isinstance(child, ast.Global):
                return True
            # Check for attribute assignments (modifying object state)
            elif isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store):
                return True
        return False
    
    def _is_pure_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is pure (no side effects, deterministic)"""
        # A pure function should:
        # 1. Not have side effects
        # 2. Not use global variables
        # 3. Not use random or time-based functions
        # 4. Only depend on its parameters
        
        if self._has_side_effects(node):
            return False
            
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in ['random', 'time', 'datetime', 'uuid']:
                        return False
                        
        return True


# Example usage
def test_enhanced_analyzer():
    """Test the enhanced code analyzer"""
    print("\n" + "="*60)
    print("Testing Enhanced Code Analyzer")
    print("="*60)
    
    analyzer = EnhancedCodeAnalyzer()
    
    # Test with test_generator.py
    analysis = analyzer.analyze_file("test_generator.py")
    
    if "error" not in analysis:
        print(f"\n📊 Analysis Results:")
        print(f"\nMetrics:")
        metrics = analysis["metrics"]
        print(f"  Lines of code: {metrics.lines_of_code}")
        print(f"  Cyclomatic complexity: {metrics.cyclomatic_complexity}")
        print(f"  Cognitive complexity: {metrics.cognitive_complexity}")
        print(f"  Maintainability index: {metrics.maintainability_index:.2f}")
        
        print(f"\nStructure:")
        structure = analysis["structure"]
        print(f"  Classes: {len(structure['classes'])}")
        print(f"  Functions: {len(structure['functions'])}")
        print(f"  Async functions: {len(structure['async_functions'])}")
        
        print(f"\nComplexity Hotspots:")
        for hotspot in analysis["complexity_hotspots"][:3]:
            print(f"  {hotspot['name']}: {hotspot['complexity']} ({hotspot['risk']} risk)")
        
        print(f"\nTest Suggestions:")
        for suggestion in analysis["test_suggestions"][:3]:
            print(f"  {suggestion['function']}: {suggestion['reason']}")
            
        print(f"\nRefactoring Opportunities:")
        for opp in analysis["refactoring_opportunities"][:3]:
            print(f"  {opp['type']}: {opp.get('suggestion', 'N/A')}")
    else:
        print(f"❌ Error: {analysis['error']}")
    
    return analyzer


if __name__ == "__main__":
    print("Enhanced Code Analyzer for Test Generation")
    print("="*60)
    
    analyzer = test_enhanced_analyzer()
    
    print("\n✅ Enhanced Code Analyzer ready!")