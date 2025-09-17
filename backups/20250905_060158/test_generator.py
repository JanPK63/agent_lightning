#!/usr/bin/env python3
"""
Automated Test Generator for Agent Lightning
Generates comprehensive test suites for code automatically
"""

import os
import sys
import ast
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import inspect
import importlib.util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestFramework(Enum):
    """Supported test frameworks"""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JASMINE = "jasmine"


class TestType(Enum):
    """Types of tests to generate"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"


class AssertionType(Enum):
    """Types of assertions"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    IS_NONE = "is_none"
    IS_NOT_NONE = "is_not_none"
    RAISES = "raises"
    NOT_RAISES = "not_raises"
    TYPE_CHECK = "type_check"
    LENGTH = "length"
    REGEX_MATCH = "regex_match"


@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    description: str
    test_type: TestType
    function_under_test: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    assertions: List[Tuple[AssertionType, Any]] = field(default_factory=list)
    setup: Optional[str] = None
    teardown: Optional[str] = None
    mocks: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1-5, 1 being highest
    timeout: Optional[int] = None
    skip_condition: Optional[str] = None


@dataclass
class TestSuite:
    """Collection of test cases for a module"""
    module_name: str
    module_path: str
    framework: TestFramework
    test_cases: List[TestCase] = field(default_factory=list)
    fixtures: Dict[str, str] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    setup_module: Optional[str] = None
    teardown_module: Optional[str] = None
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    coverage_target: float = 80.0


class CodeAnalyzer:
    """Analyzes code to understand structure for test generation"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.methods = []
        self.imports = []
        self.global_vars = []
        self.dependencies = []
    
    def analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file to extract testable components"""
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        try:
            tree = ast.parse(source_code)
            return self._analyze_ast(tree, file_path)
        except SyntaxError as e:
            return {"error": f"Syntax error in file: {e}"}
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze AST to extract code structure"""
        analysis = {
            "file_path": file_path,
            "functions": [],
            "classes": [],
            "imports": [],
            "global_variables": [],
            "decorators": [],
            "async_functions": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._extract_function_info(node)
                analysis["functions"].append(func_info)
                
            elif isinstance(node, ast.AsyncFunctionDef):
                func_info = self._extract_function_info(node)
                func_info["is_async"] = True
                analysis["async_functions"].append(func_info)
                
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node)
                analysis["classes"].append(class_info)
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_info = self._extract_import_info(node)
                analysis["imports"].append(import_info)
                
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if self._is_global_scope(node):
                    var_info = {
                        "name": node.targets[0].id,
                        "type": type(node.value).__name__
                    }
                    analysis["global_variables"].append(var_info)
        
        return analysis
    
    def _extract_function_info(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict[str, Any]:
        """Extract information about a function"""
        func_info = {
            "name": node.name,
            "parameters": [],
            "return_type": None,
            "docstring": ast.get_docstring(node),
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "line_number": node.lineno,
            "complexity": self._calculate_complexity(node)
        }
        
        # Extract parameters
        for arg in node.args.args:
            param = {
                "name": arg.arg,
                "type": None,
                "default": None
            }
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            func_info["parameters"].append(param)
        
        # Extract return type
        if node.returns:
            func_info["return_type"] = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        return func_info
    
    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract information about a class"""
        class_info = {
            "name": node.name,
            "methods": [],
            "attributes": [],
            "base_classes": [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in node.bases],
            "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
            "docstring": ast.get_docstring(node),
            "line_number": node.lineno
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_function_info(item)
                method_info["is_method"] = True
                class_info["methods"].append(method_info)
        
        return class_info
    
    def _extract_import_info(self, node: Union[ast.Import, ast.ImportFrom]) -> Dict[str, Any]:
        """Extract import information"""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "names": [alias.name for alias in node.names],
                "aliases": {alias.name: alias.asname for alias in node.names if alias.asname}
            }
        else:  # ImportFrom
            return {
                "type": "from_import",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "aliases": {alias.name: alias.asname for alias in node.names if alias.asname}
            }
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name as string"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        else:
            return ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator)
    
    def _is_global_scope(self, node: ast.AST) -> bool:
        """Check if node is in global scope"""
        # Simplified check - would need more sophisticated analysis for accuracy
        return True  # Placeholder
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def analyze_javascript_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a JavaScript file (basic regex-based analysis)"""
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        analysis = {
            "file_path": file_path,
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": []
        }
        
        # Find functions
        func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(func_pattern, source_code):
            analysis["functions"].append({
                "name": match.group(1),
                "is_async": "async" in match.group(0)
            })
        
        # Find arrow functions
        arrow_pattern = r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'
        for match in re.finditer(arrow_pattern, source_code):
            analysis["functions"].append({
                "name": match.group(1),
                "is_arrow": True,
                "is_async": "async" in match.group(0)
            })
        
        # Find classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, source_code):
            analysis["classes"].append({"name": match.group(1)})
        
        # Find imports
        import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, source_code):
            analysis["imports"].append({"module": match.group(1)})
        
        # Find exports
        export_pattern = r'export\s+(?:default\s+)?(\w+)'
        for match in re.finditer(export_pattern, source_code):
            analysis["exports"].append({"name": match.group(1)})
        
        return analysis


class TestCaseGenerator:
    """Generates test cases based on code analysis"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.test_patterns = self._load_test_patterns()
    
    def _load_test_patterns(self) -> Dict[str, Any]:
        """Load common test patterns"""
        return {
            "boundary_values": ["min", "max", "zero", "negative", "overflow"],
            "edge_cases": ["empty", "null", "undefined", "single_element"],
            "error_cases": ["invalid_input", "type_error", "timeout", "permission"],
            "happy_path": ["normal_input", "expected_use"],
            "stress_test": ["large_input", "concurrent", "performance"]
        }
    
    def generate_test_cases(
        self,
        code_analysis: Dict[str, Any],
        test_type: TestType = TestType.UNIT
    ) -> List[TestCase]:
        """Generate test cases from code analysis"""
        test_cases = []
        
        # Generate tests for functions
        for func in code_analysis.get("functions", []):
            test_cases.extend(self._generate_function_tests(func, test_type))
        
        # Generate tests for async functions
        for func in code_analysis.get("async_functions", []):
            test_cases.extend(self._generate_async_function_tests(func, test_type))
        
        # Generate tests for classes
        for cls in code_analysis.get("classes", []):
            test_cases.extend(self._generate_class_tests(cls, test_type))
        
        return test_cases
    
    def _generate_function_tests(self, func_info: Dict[str, Any], test_type: TestType) -> List[TestCase]:
        """Generate test cases for a function"""
        test_cases = []
        func_name = func_info["name"]
        
        # Happy path test
        test_cases.append(TestCase(
            name=f"test_{func_name}_happy_path",
            description=f"Test {func_name} with valid inputs",
            test_type=test_type,
            function_under_test=func_name,
            inputs=self._generate_valid_inputs(func_info),
            assertions=[(AssertionType.IS_NOT_NONE, None)],
            tags=["happy_path"],
            priority=1
        ))
        
        # Edge cases
        if func_info["parameters"]:
            test_cases.append(TestCase(
                name=f"test_{func_name}_edge_cases",
                description=f"Test {func_name} with edge case inputs",
                test_type=test_type,
                function_under_test=func_name,
                inputs=self._generate_edge_case_inputs(func_info),
                assertions=[(AssertionType.NOT_RAISES, None)],
                tags=["edge_case"],
                priority=2
            ))
        
        # Error cases
        test_cases.append(TestCase(
            name=f"test_{func_name}_invalid_input",
            description=f"Test {func_name} with invalid inputs",
            test_type=test_type,
            function_under_test=func_name,
            inputs=self._generate_invalid_inputs(func_info),
            assertions=[(AssertionType.RAISES, "Exception")],
            tags=["error_case"],
            priority=3
        ))
        
        # Performance test if complexity is high
        if func_info.get("complexity", 0) > 5:
            test_cases.append(TestCase(
                name=f"test_{func_name}_performance",
                description=f"Test {func_name} performance",
                test_type=TestType.PERFORMANCE,
                function_under_test=func_name,
                inputs=self._generate_large_inputs(func_info),
                assertions=[(AssertionType.LESS_THAN, 1000)],  # ms
                tags=["performance"],
                priority=4,
                timeout=5000
            ))
        
        return test_cases
    
    def _generate_async_function_tests(self, func_info: Dict[str, Any], test_type: TestType) -> List[TestCase]:
        """Generate test cases for async functions"""
        test_cases = self._generate_function_tests(func_info, test_type)
        
        # Mark all as async tests
        for test_case in test_cases:
            test_case.tags.append("async")
            test_case.name = test_case.name.replace("test_", "test_async_")
        
        return test_cases
    
    def _generate_class_tests(self, class_info: Dict[str, Any], test_type: TestType) -> List[TestCase]:
        """Generate test cases for a class"""
        test_cases = []
        class_name = class_info["name"]
        
        # Test instantiation
        test_cases.append(TestCase(
            name=f"test_{class_name}_instantiation",
            description=f"Test {class_name} can be instantiated",
            test_type=test_type,
            function_under_test=f"{class_name}.__init__",
            assertions=[(AssertionType.IS_NOT_NONE, None), (AssertionType.TYPE_CHECK, class_name)],
            tags=["instantiation"],
            priority=1
        ))
        
        # Test methods
        for method in class_info.get("methods", []):
            if method["name"] != "__init__":
                method_test_cases = self._generate_function_tests(method, test_type)
                for test_case in method_test_cases:
                    test_case.function_under_test = f"{class_name}.{method['name']}"
                    test_case.setup = f"self.instance = {class_name}()"
                test_cases.extend(method_test_cases)
        
        return test_cases
    
    def _generate_valid_inputs(self, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate valid inputs for a function"""
        inputs = {}
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type")
            
            if param_type:
                if "str" in str(param_type):
                    inputs[param_name] = "test_string"
                elif "int" in str(param_type):
                    inputs[param_name] = 42
                elif "float" in str(param_type):
                    inputs[param_name] = 3.14
                elif "bool" in str(param_type):
                    inputs[param_name] = True
                elif "list" in str(param_type).lower():
                    inputs[param_name] = [1, 2, 3]
                elif "dict" in str(param_type).lower():
                    inputs[param_name] = {"key": "value"}
                else:
                    inputs[param_name] = None
            else:
                # Default values based on parameter name
                if "name" in param_name or "string" in param_name:
                    inputs[param_name] = "test_value"
                elif "count" in param_name or "number" in param_name:
                    inputs[param_name] = 10
                elif "flag" in param_name or "is_" in param_name:
                    inputs[param_name] = True
                else:
                    inputs[param_name] = "default_value"
        
        return inputs
    
    def _generate_edge_case_inputs(self, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate edge case inputs"""
        inputs = {}
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type")
            
            if param_type:
                if "str" in str(param_type):
                    inputs[param_name] = ""
                elif "int" in str(param_type):
                    inputs[param_name] = 0
                elif "list" in str(param_type).lower():
                    inputs[param_name] = []
                elif "dict" in str(param_type).lower():
                    inputs[param_name] = {}
                else:
                    inputs[param_name] = None
            else:
                inputs[param_name] = None
        
        return inputs
    
    def _generate_invalid_inputs(self, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate invalid inputs"""
        inputs = {}
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type")
            
            if param_type:
                # Intentionally wrong types
                if "str" in str(param_type):
                    inputs[param_name] = 123  # Number instead of string
                elif "int" in str(param_type):
                    inputs[param_name] = "not_a_number"
                elif "list" in str(param_type).lower():
                    inputs[param_name] = "not_a_list"
                else:
                    inputs[param_name] = object()  # Random object
            else:
                inputs[param_name] = None
        
        return inputs
    
    def _generate_large_inputs(self, func_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate large inputs for performance testing"""
        inputs = {}
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type")
            
            if param_type:
                if "str" in str(param_type):
                    inputs[param_name] = "x" * 10000
                elif "int" in str(param_type):
                    inputs[param_name] = 999999
                elif "list" in str(param_type).lower():
                    inputs[param_name] = list(range(10000))
                elif "dict" in str(param_type).lower():
                    inputs[param_name] = {str(i): i for i in range(1000)}
                else:
                    inputs[param_name] = None
            else:
                inputs[param_name] = "large_value" * 1000
        
        return inputs


class TestGenerator:
    """Main test generator class"""
    
    def __init__(self, framework: TestFramework = TestFramework.PYTEST):
        self.framework = framework
        self.analyzer = CodeAnalyzer()
        self.case_generator = TestCaseGenerator()
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load test file templates for different frameworks"""
        templates = {}
        
        # Pytest template
        templates[TestFramework.PYTEST] = """\"\"\"
Auto-generated test file for {module_name}
Generated on: {timestamp}
\"\"\"

import pytest
import sys
import os
{imports}

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from {module_name} import *

{fixtures}

{test_cases}
"""
        
        # Jest template
        templates[TestFramework.JEST] = """/**
 * Auto-generated test file for {module_name}
 * Generated on: {timestamp}
 */

const {{ {imports} }} = require('../{module_name}');

{setup}

describe('{module_name}', () => {{
{test_cases}
}});

{teardown}
"""
        
        return templates
    
    def generate_tests_for_file(
        self,
        file_path: str,
        output_dir: str = "tests",
        test_types: List[TestType] = None
    ) -> Tuple[bool, str, List[str]]:
        """Generate tests for a specific file"""
        if test_types is None:
            test_types = [TestType.UNIT]
        
        try:
            # Determine file type
            if file_path.endswith('.py'):
                analysis = self.analyzer.analyze_python_file(file_path)
            elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                analysis = self.analyzer.analyze_javascript_file(file_path)
            else:
                return False, "", [f"Unsupported file type: {file_path}"]
            
            if "error" in analysis:
                return False, "", [analysis["error"]]
            
            # Generate test cases
            all_test_cases = []
            for test_type in test_types:
                test_cases = self.case_generator.generate_test_cases(analysis, test_type)
                all_test_cases.extend(test_cases)
            
            # Create test suite
            module_name = os.path.basename(file_path).rsplit('.', 1)[0]
            test_suite = TestSuite(
                module_name=module_name,
                module_path=file_path,
                framework=self.framework,
                test_cases=all_test_cases
            )
            
            # Generate test code
            test_code = self._generate_test_code(test_suite)
            
            # Save test file
            os.makedirs(output_dir, exist_ok=True)
            test_file_name = f"test_{module_name}.{'py' if self.framework in [TestFramework.PYTEST, TestFramework.UNITTEST] else 'js'}"
            test_file_path = os.path.join(output_dir, test_file_name)
            
            with open(test_file_path, 'w') as f:
                f.write(test_code)
            
            return True, test_file_path, []
            
        except Exception as e:
            return False, "", [str(e)]
    
    def _generate_test_code(self, test_suite: TestSuite) -> str:
        """Generate actual test code from test suite"""
        if test_suite.framework == TestFramework.PYTEST:
            return self._generate_pytest_code(test_suite)
        elif test_suite.framework == TestFramework.JEST:
            return self._generate_jest_code(test_suite)
        else:
            raise NotImplementedError(f"Framework {test_suite.framework} not yet implemented")
    
    def _generate_pytest_code(self, test_suite: TestSuite) -> str:
        """Generate pytest code"""
        test_functions = []
        
        for test_case in test_suite.test_cases:
            test_code = self._generate_pytest_test_function(test_case)
            test_functions.append(test_code)
        
        # Format template
        test_code = self.templates[TestFramework.PYTEST].format(
            module_name=test_suite.module_name,
            timestamp=datetime.now().isoformat(),
            imports="\n".join(test_suite.imports),
            fixtures=self._generate_pytest_fixtures(test_suite.fixtures),
            test_cases="\n\n".join(test_functions)
        )
        
        return test_code
    
    def _generate_pytest_test_function(self, test_case: TestCase) -> str:
        """Generate a single pytest test function"""
        # Build test function
        lines = []
        
        # Add decorators
        if test_case.skip_condition:
            lines.append(f"@pytest.mark.skipif({test_case.skip_condition}, reason='Conditional skip')")
        
        if "async" in test_case.tags:
            lines.append("@pytest.mark.asyncio")
            lines.append(f"async def {test_case.name}():")
        else:
            lines.append(f"def {test_case.name}():")
        
        lines.append(f'    """')
        lines.append(f'    {test_case.description}')
        lines.append(f'    """')
        
        # Setup
        if test_case.setup:
            lines.append(f"    # Setup")
            lines.append(f"    {test_case.setup}")
        
        # Prepare inputs
        if test_case.inputs:
            lines.append(f"    # Arrange")
            for key, value in test_case.inputs.items():
                if isinstance(value, str):
                    lines.append(f"    {key} = '{value}'")
                else:
                    lines.append(f"    {key} = {value}")
        
        # Call function
        lines.append(f"    # Act")
        if test_case.inputs:
            params = ", ".join(f"{k}={k}" for k in test_case.inputs.keys())
            if "async" in test_case.tags:
                lines.append(f"    result = await {test_case.function_under_test}({params})")
            else:
                lines.append(f"    result = {test_case.function_under_test}({params})")
        else:
            if "async" in test_case.tags:
                lines.append(f"    result = await {test_case.function_under_test}()")
            else:
                lines.append(f"    result = {test_case.function_under_test}()")
        
        # Assertions
        lines.append(f"    # Assert")
        for assertion_type, expected in test_case.assertions:
            lines.append(self._generate_pytest_assertion(assertion_type, expected))
        
        # Teardown
        if test_case.teardown:
            lines.append(f"    # Teardown")
            lines.append(f"    {test_case.teardown}")
        
        return "\n".join(lines)
    
    def _generate_pytest_assertion(self, assertion_type: AssertionType, expected: Any) -> str:
        """Generate pytest assertion"""
        if assertion_type == AssertionType.EQUALS:
            return f"    assert result == {expected}"
        elif assertion_type == AssertionType.NOT_EQUALS:
            return f"    assert result != {expected}"
        elif assertion_type == AssertionType.IS_NONE:
            return f"    assert result is None"
        elif assertion_type == AssertionType.IS_NOT_NONE:
            return f"    assert result is not None"
        elif assertion_type == AssertionType.IS_TRUE:
            return f"    assert result is True"
        elif assertion_type == AssertionType.IS_FALSE:
            return f"    assert result is False"
        elif assertion_type == AssertionType.RAISES:
            return f"    # Should raise {expected}"
        elif assertion_type == AssertionType.TYPE_CHECK:
            return f"    assert isinstance(result, {expected})"
        else:
            return f"    assert result  # {assertion_type.value}"
    
    def _generate_pytest_fixtures(self, fixtures: Dict[str, str]) -> str:
        """Generate pytest fixtures"""
        fixture_code = []
        for name, code in fixtures.items():
            fixture_code.append(f"@pytest.fixture")
            fixture_code.append(f"def {name}():")
            fixture_code.append(f"    {code}")
            fixture_code.append("")
        return "\n".join(fixture_code)
    
    def _generate_jest_code(self, test_suite: TestSuite) -> str:
        """Generate Jest code"""
        # TODO: Implement Jest code generation
        return "// Jest tests to be implemented"


# Example usage
def test_test_generator():
    """Test the test generator"""
    print("\n" + "="*60)
    print("Testing Automated Test Generator")
    print("="*60)
    
    # Create test generator
    generator = TestGenerator(TestFramework.PYTEST)
    
    # Test with visual_code_builder.py
    success, test_file, errors = generator.generate_tests_for_file(
        "visual_code_builder.py",
        output_dir="generated_tests"
    )
    
    if success:
        print(f"\n✅ Generated test file: {test_file}")
        
        # Show preview
        with open(test_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"\n📝 Preview (first 50 lines):")
            print("-" * 40)
            for line in lines[:50]:
                print(line)
    else:
        print(f"❌ Failed to generate tests: {errors}")
    
    return generator


if __name__ == "__main__":
    print("Automated Test Generator for Agent Lightning")
    print("="*60)
    
    generator = test_test_generator()
    
    print("\n✅ Test Generator initialized successfully!")