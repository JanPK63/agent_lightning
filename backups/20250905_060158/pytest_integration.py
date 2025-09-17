#!/usr/bin/env python3
"""
PyTest Integration for Python Testing
Automated pytest test generation and execution for Python projects
"""

import os
import sys
import json
import subprocess
import ast
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import shutil
import tempfile
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_generator import TestCase, TestType, AssertionType, TestFramework
from test_case_generator_advanced import TestDataGenerator
from assertion_builder import AssertionBuilder
from mock_data_generator import MockDataGenerator, MockDataType, MockConfig
from code_analyzer_enhanced import EnhancedCodeAnalyzer


class PytestMarker(Enum):
    """Pytest markers"""
    SKIP = "skip"
    SKIPIF = "skipif"
    XFAIL = "xfail"
    PARAMETRIZE = "parametrize"
    TIMEOUT = "timeout"
    SLOW = "slow"
    FAST = "fast"
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    SMOKE = "smoke"
    REGRESSION = "regression"
    ASYNCIO = "asyncio"
    DJANGO_DB = "django_db"
    BENCHMARK = "benchmark"


class PytestFixtureScope(Enum):
    """Pytest fixture scopes"""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PACKAGE = "package"
    SESSION = "session"


@dataclass
class PytestConfig:
    """Pytest configuration"""
    min_version: str = "7.0"
    test_paths: List[str] = field(default_factory=lambda: ["tests"])
    python_files: List[str] = field(default_factory=lambda: ["test_*.py", "*_test.py"])
    python_classes: List[str] = field(default_factory=lambda: ["Test*"])
    python_functions: List[str] = field(default_factory=lambda: ["test_*"])
    markers: Dict[str, str] = field(default_factory=lambda: {
        "slow": "marks tests as slow (deselect with '-m \"not slow\"')",
        "unit": "marks tests as unit tests",
        "integration": "marks tests as integration tests",
        "e2e": "marks tests as end-to-end tests"
    })
    addopts: List[str] = field(default_factory=lambda: [
        "-ra",
        "--strict-markers",
        "--strict-config",
        "--cov",
        "--cov-branch",
        "--cov-report=term-missing:skip-covered",
        "--cov-report=html",
        "--cov-report=xml"
    ])
    testpaths: List[str] = field(default_factory=lambda: ["tests"])
    norecursedirs: List[str] = field(default_factory=lambda: [
        "*.egg", ".eggs", "dist", "build", "docs", ".tox", ".git", "__pycache__"
    ])
    cache_dir: str = ".pytest_cache"
    console_output_style: str = "progress"
    log_cli: bool = True
    log_cli_level: str = "INFO"


@dataclass
class PytestFixture:
    """Represents a pytest fixture"""
    name: str
    scope: PytestFixtureScope = PytestFixtureScope.FUNCTION
    params: Optional[List[Any]] = None
    autouse: bool = False
    ids: Optional[List[str]] = None
    code: str = ""


@dataclass
class PytestTestFile:
    """Represents a pytest test file"""
    file_path: str
    source_file: str
    test_class: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    fixtures: List[PytestFixture] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""


class PytestTestGenerator:
    """Generates pytest test files"""
    
    def __init__(self):
        self.analyzer = EnhancedCodeAnalyzer()
        self.data_generator = TestDataGenerator()
        self.mock_generator = MockDataGenerator(MockConfig(realistic=True))
        self.assertion_builder = AssertionBuilder(TestFramework.PYTEST)
    
    def generate_test_file(
        self,
        source_file: str,
        output_dir: str = "tests",
        analyze_code: bool = True
    ) -> Tuple[bool, str, List[str]]:
        """Generate a pytest test file for a source file"""
        try:
            errors = []
            
            # Analyze source code if requested
            test_cases = []
            if analyze_code and os.path.exists(source_file):
                analysis = self.analyzer.analyze_file(source_file)
                if "error" not in analysis:
                    test_cases = self._generate_test_cases_from_analysis(analysis)
                else:
                    errors.append(f"Code analysis error: {analysis['error']}")
            
            # Prepare test file
            test_file = self._prepare_test_file(source_file, test_cases)
            
            # Generate test code
            test_code = self._generate_pytest_code(test_file)
            
            # Write test file
            output_path = self._write_test_file(test_file, test_code, output_dir)
            
            return True, output_path, errors
            
        except Exception as e:
            return False, "", [str(e)]
    
    def _generate_test_cases_from_analysis(self, analysis: Dict) -> List[TestCase]:
        """Generate test cases from code analysis"""
        test_cases = []
        
        # Generate tests for functions
        for func in analysis.get("structure", {}).get("functions", []):
            test_cases.extend(self._generate_function_test_cases(func))
        
        # Generate tests for async functions
        for func in analysis.get("structure", {}).get("async_functions", []):
            test_cases.extend(self._generate_async_function_test_cases(func))
        
        # Generate tests for classes
        for cls in analysis.get("structure", {}).get("classes", []):
            test_cases.extend(self._generate_class_test_cases(cls))
        
        return test_cases
    
    def _generate_function_test_cases(self, func_info: Dict) -> List[TestCase]:
        """Generate test cases for a function"""
        test_cases = []
        func_name = func_info["name"]
        
        # Happy path test
        test_cases.append(TestCase(
            name=f"test_{func_name}_happy_path",
            description=f"Test {func_name} with valid inputs",
            test_type=TestType.UNIT,
            function_under_test=func_name,
            inputs=self._generate_valid_inputs(func_info),
            assertions=[
                (AssertionType.IS_NOT_NONE, None),
                (AssertionType.NOT_RAISES, None)
            ],
            tags=["unit", "happy_path"]
        ))
        
        # Edge cases
        if func_info.get("parameters"):
            test_cases.append(TestCase(
                name=f"test_{func_name}_edge_cases",
                description=f"Test {func_name} with edge case inputs",
                test_type=TestType.UNIT,
                function_under_test=func_name,
                inputs=self._generate_edge_inputs(func_info),
                assertions=[(AssertionType.NOT_RAISES, None)],
                tags=["unit", "edge_case"]
            ))
        
        # Error cases
        if func_info.get("complexity", 0) > 5:
            test_cases.append(TestCase(
                name=f"test_{func_name}_error_handling",
                description=f"Test {func_name} error handling",
                test_type=TestType.UNIT,
                function_under_test=func_name,
                inputs=self._generate_invalid_inputs(func_info),
                assertions=[(AssertionType.RAISES, "Exception")],
                tags=["unit", "error"]
            ))
        
        return test_cases
    
    def _generate_async_function_test_cases(self, func_info: Dict) -> List[TestCase]:
        """Generate test cases for async functions"""
        test_cases = self._generate_function_test_cases(func_info)
        
        # Mark all as async
        for test_case in test_cases:
            test_case.tags.append("asyncio")
            test_case.name = test_case.name.replace("test_", "test_async_")
        
        return test_cases
    
    def _generate_class_test_cases(self, class_info: Dict) -> List[TestCase]:
        """Generate test cases for a class"""
        test_cases = []
        class_name = class_info["name"]
        
        # Test instantiation
        test_cases.append(TestCase(
            name=f"test_{class_name}_instantiation",
            description=f"Test {class_name} instantiation",
            test_type=TestType.UNIT,
            function_under_test=class_name,
            assertions=[
                (AssertionType.IS_NOT_NONE, None),
                (AssertionType.TYPE_CHECK, class_name)
            ],
            tags=["unit", "instantiation"]
        ))
        
        # Test methods
        for method in class_info.get("methods", []):
            if not method["name"].startswith("_"):  # Skip private methods
                method_test_cases = self._generate_function_test_cases(method)
                for tc in method_test_cases:
                    tc.function_under_test = f"{class_name}.{method['name']}"
                    tc.setup = f"instance = {class_name}()"
                test_cases.extend(method_test_cases)
        
        return test_cases
    
    def _generate_valid_inputs(self, func_info: Dict) -> Dict[str, Any]:
        """Generate valid inputs for a function"""
        inputs = {}
        
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "Any")
            
            # Generate appropriate data based on type
            test_data = self.data_generator.generate_test_data(
                param_name, param_type
            )
            
            if test_data.valid_values:
                inputs[param_name] = test_data.valid_values[0]
            else:
                # Use mock data generator
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
        
        return inputs
    
    def _generate_edge_inputs(self, func_info: Dict) -> Dict[str, Any]:
        """Generate edge case inputs"""
        inputs = {}
        
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "Any")
            
            test_data = self.data_generator.generate_test_data(
                param_name, param_type
            )
            
            if test_data.edge_cases:
                inputs[param_name] = test_data.edge_cases[0]
            elif test_data.boundary_values:
                inputs[param_name] = test_data.boundary_values[0]
            else:
                inputs[param_name] = None
        
        return inputs
    
    def _generate_invalid_inputs(self, func_info: Dict) -> Dict[str, Any]:
        """Generate invalid inputs"""
        inputs = {}
        
        for param in func_info.get("parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "Any")
            
            test_data = self.data_generator.generate_test_data(
                param_name, param_type
            )
            
            if test_data.invalid_values:
                inputs[param_name] = test_data.invalid_values[0]
            else:
                # Wrong type
                if "str" in str(param_type):
                    inputs[param_name] = 123
                elif "int" in str(param_type):
                    inputs[param_name] = "not_a_number"
                else:
                    inputs[param_name] = object()
        
        return inputs
    
    def _prepare_test_file(
        self,
        source_file: str,
        test_cases: List[TestCase]
    ) -> PytestTestFile:
        """Prepare test file structure"""
        test_file = PytestTestFile(
            file_path="",
            source_file=source_file,
            test_cases=test_cases
        )
        
        # Generate imports
        test_file.imports = self._generate_imports(source_file, test_cases)
        
        # Generate fixtures
        test_file.fixtures = self._generate_fixtures(test_cases)
        
        # Generate setup/teardown
        test_file.setup_code = self._generate_setup()
        test_file.teardown_code = self._generate_teardown()
        
        return test_file
    
    def _generate_imports(
        self,
        source_file: str,
        test_cases: List[TestCase]
    ) -> List[str]:
        """Generate import statements"""
        imports = [
            "import pytest",
            "import sys",
            "import os",
            "from unittest.mock import Mock, MagicMock, patch",
            "from datetime import datetime, timedelta"
        ]
        
        # Add path to source
        imports.append("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
        
        # Import from source file
        module_name = Path(source_file).stem
        imports.append(f"from {module_name} import *")
        
        # Check for async tests
        if any("asyncio" in tc.tags for tc in test_cases):
            imports.append("import asyncio")
            imports.append("import pytest_asyncio")
        
        # Check for parametrized tests
        if any("parametrize" in str(tc) for tc in test_cases):
            imports.append("from itertools import product")
        
        return imports
    
    def _generate_fixtures(self, test_cases: List[TestCase]) -> List[PytestFixture]:
        """Generate pytest fixtures"""
        fixtures = []
        
        # Common fixtures
        fixtures.append(PytestFixture(
            name="setup_data",
            scope=PytestFixtureScope.FUNCTION,
            code="""@pytest.fixture
def setup_data():
    \"\"\"Provide test data\"\"\"
    return {
        "test_string": "test_value",
        "test_int": 42,
        "test_list": [1, 2, 3],
        "test_dict": {"key": "value"}
    }"""
        ))
        
        # Mock fixtures
        fixtures.append(PytestFixture(
            name="mock_api",
            scope=PytestFixtureScope.FUNCTION,
            code="""@pytest.fixture
def mock_api():
    \"\"\"Mock API client\"\"\"
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {"status": "success"}
        mock_get.return_value.status_code = 200
        yield mock_get"""
        ))
        
        # Database fixture (if needed)
        if any("database" in str(tc.function_under_test).lower() for tc in test_cases):
            fixtures.append(PytestFixture(
                name="test_db",
                scope=PytestFixtureScope.SESSION,
                code="""@pytest.fixture(scope="session")
def test_db():
    \"\"\"Test database connection\"\"\"
    # Setup test database
    db = create_test_database()
    yield db
    # Teardown
    db.close()
    cleanup_test_database()"""
            ))
        
        # Async fixture (if needed)
        if any("asyncio" in tc.tags for tc in test_cases):
            fixtures.append(PytestFixture(
                name="event_loop",
                scope=PytestFixtureScope.SESSION,
                code="""@pytest.fixture(scope="session")
def event_loop():
    \"\"\"Create event loop for async tests\"\"\"
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()"""
            ))
        
        return fixtures
    
    def _generate_setup(self) -> str:
        """Generate setup code"""
        return """
def setup_module(module):
    \"\"\"Setup for the entire module\"\"\"
    print("\\nSetting up module...")

def setup_function(function):
    \"\"\"Setup for each function\"\"\"
    pass"""
    
    def _generate_teardown(self) -> str:
        """Generate teardown code"""
        return """
def teardown_function(function):
    \"\"\"Teardown for each function\"\"\"
    pass

def teardown_module(module):
    \"\"\"Teardown for the entire module\"\"\"
    print("\\nTearing down module...")"""
    
    def _generate_pytest_code(self, test_file: PytestTestFile) -> str:
        """Generate complete pytest code"""
        code_parts = []
        
        # Header
        code_parts.append('"""')
        code_parts.append(f"Test file for {test_file.source_file}")
        code_parts.append(f"Generated on: {datetime.now().isoformat()}")
        code_parts.append('"""')
        code_parts.append("")
        
        # Imports
        code_parts.extend(test_file.imports)
        code_parts.append("")
        
        # Fixtures
        for fixture in test_file.fixtures:
            code_parts.append(fixture.code)
            code_parts.append("")
        
        # Setup/Teardown
        if test_file.setup_code:
            code_parts.append(test_file.setup_code)
            code_parts.append("")
        if test_file.teardown_code:
            code_parts.append(test_file.teardown_code)
            code_parts.append("")
        
        # Test class or functions
        if test_file.test_class:
            code_parts.append(f"class Test{test_file.test_class}:")
            code_parts.append("")
            
            # Class-level fixtures
            code_parts.append("    @classmethod")
            code_parts.append("    def setup_class(cls):")
            code_parts.append('        """Setup for the test class"""')
            code_parts.append("        pass")
            code_parts.append("")
            
            # Generate test methods
            for test_case in test_file.test_cases:
                test_code = self._generate_test_method(test_case, indent=1)
                code_parts.append(test_code)
        else:
            # Generate test functions
            for test_case in test_file.test_cases:
                test_code = self._generate_test_function(test_case)
                code_parts.append(test_code)
        
        # Parametrized tests
        code_parts.append(self._generate_parametrized_tests())
        
        # Performance tests
        code_parts.append(self._generate_performance_tests())
        
        return "\n".join(code_parts)
    
    def _generate_test_function(self, test_case: TestCase) -> str:
        """Generate a test function"""
        lines = []
        
        # Decorators
        if test_case.skip_condition:
            lines.append(f"@pytest.mark.skipif({test_case.skip_condition}, reason='Conditional skip')")
        
        if "slow" in test_case.tags:
            lines.append("@pytest.mark.slow")
        
        if "asyncio" in test_case.tags:
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
        
        # Arrange
        if test_case.inputs:
            lines.append(f"    # Arrange")
            for key, value in test_case.inputs.items():
                value_repr = repr(value)
                lines.append(f"    {key} = {value_repr}")
        
        # Act
        lines.append(f"    # Act")
        if test_case.inputs:
            params = ", ".join(f"{k}={k}" for k in test_case.inputs.keys())
            if "asyncio" in test_case.tags:
                lines.append(f"    result = await {test_case.function_under_test}({params})")
            else:
                lines.append(f"    result = {test_case.function_under_test}({params})")
        else:
            if "asyncio" in test_case.tags:
                lines.append(f"    result = await {test_case.function_under_test}()")
            else:
                lines.append(f"    result = {test_case.function_under_test}()")
        
        # Assert
        lines.append(f"    # Assert")
        for assertion_type, expected in test_case.assertions:
            assertion_code = self.assertion_builder.build_assertion(
                assertion_type, expected
            )
            lines.append(f"    {self.assertion_builder.to_code(assertion_code)}")
        
        # Teardown
        if test_case.teardown:
            lines.append(f"    # Teardown")
            lines.append(f"    {test_case.teardown}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _generate_test_method(self, test_case: TestCase, indent: int = 1) -> str:
        """Generate a test method for a test class"""
        lines = self._generate_test_function(test_case).split("\n")
        indent_str = "    " * indent
        return "\n".join(indent_str + line if line else line for line in lines)
    
    def _generate_parametrized_tests(self) -> str:
        """Generate parametrized test examples"""
        return """
# Parametrized test example
@pytest.mark.parametrize("input_value,expected", [
    (0, 0),
    (1, 1),
    (-1, 1),
    (10, 100),
    (-10, 100),
])
def test_square_parametrized(input_value, expected):
    \"\"\"Test square function with multiple inputs\"\"\"
    result = input_value ** 2
    assert result == expected"""
    
    def _generate_performance_tests(self) -> str:
        """Generate performance test examples"""
        return """
# Performance test example
@pytest.mark.timeout(5)
@pytest.mark.benchmark
def test_performance():
    \"\"\"Test function performance\"\"\"
    import time
    start = time.time()
    # Run function
    result = expensive_operation()
    duration = time.time() - start
    assert duration < 1.0, f"Operation took {duration:.2f}s, expected < 1s\""""
    
    def _write_test_file(
        self,
        test_file: PytestTestFile,
        test_code: str,
        output_dir: str
    ) -> str:
        """Write test file to disk"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        source_name = Path(test_file.source_file).stem
        test_filename = f"test_{source_name}.py"
        output_path = os.path.join(output_dir, test_filename)
        
        # Write file
        with open(output_path, 'w') as f:
            f.write(test_code)
        
        return output_path


class PytestRunner:
    """Runs pytest tests and collects results"""
    
    def __init__(self, config: PytestConfig = None):
        self.config = config or PytestConfig()
    
    def run_tests(
        self,
        test_path: str = None,
        markers: str = None,
        coverage: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run pytest tests"""
        cmd = ["python", "-m", "pytest"]
        
        if test_path:
            cmd.append(test_path)
        
        if markers:
            cmd.extend(["-m", markers])
        
        if coverage:
            cmd.extend(["--cov", "--cov-report=json"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend(["--json-report", "--json-report-file=pytest-report.json"])
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse results
            results = self._parse_results()
            
            return results
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_results(self) -> Dict[str, Any]:
        """Parse pytest results"""
        results = {
            "success": False,
            "summary": {},
            "tests": [],
            "coverage": {}
        }
        
        # Parse JSON report if available
        if os.path.exists("pytest-report.json"):
            with open("pytest-report.json", 'r') as f:
                report = json.load(f)
            
            results["success"] = report.get("exitcode") == 0
            results["summary"] = report.get("summary", {})
            results["tests"] = report.get("tests", [])
            
            # Clean up
            os.remove("pytest-report.json")
        
        # Parse coverage if available
        if os.path.exists("coverage.json"):
            with open("coverage.json", 'r') as f:
                coverage = json.load(f)
            
            results["coverage"] = self._parse_coverage(coverage)
            
            # Clean up
            os.remove("coverage.json")
        
        return results
    
    def _parse_coverage(self, coverage_data: Dict) -> Dict[str, Any]:
        """Parse coverage data"""
        files = coverage_data.get("files", {})
        
        total_lines = 0
        covered_lines = 0
        
        for file_data in files.values():
            executed = set(file_data.get("executed_lines", []))
            missing = set(file_data.get("missing_lines", []))
            
            file_total = len(executed) + len(missing)
            file_covered = len(executed)
            
            total_lines += file_total
            covered_lines += file_covered
        
        coverage_percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            "percent": round(coverage_percent, 2),
            "lines_covered": covered_lines,
            "lines_total": total_lines
        }


class PytestConfigGenerator:
    """Generates pytest configuration files"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.config = PytestConfig()
    
    def generate_pytest_ini(self) -> bool:
        """Generate pytest.ini file"""
        try:
            ini_content = [
                "[pytest]",
                f"minversion = {self.config.min_version}",
                f"testpaths = {' '.join(self.config.testpaths)}",
                f"python_files = {' '.join(self.config.python_files)}",
                f"python_classes = {' '.join(self.config.python_classes)}",
                f"python_functions = {' '.join(self.config.python_functions)}",
                f"addopts = {' '.join(self.config.addopts)}",
                f"norecursedirs = {' '.join(self.config.norecursedirs)}",
                f"cache_dir = {self.config.cache_dir}",
                f"console_output_style = {self.config.console_output_style}",
                f"log_cli = {self.config.log_cli}",
                f"log_cli_level = {self.config.log_cli_level}",
                "",
                "markers ="
            ]
            
            for marker, description in self.config.markers.items():
                ini_content.append(f"    {marker}: {description}")
            
            # Write file
            ini_path = self.project_root / "pytest.ini"
            ini_path.write_text("\n".join(ini_content))
            
            return True
            
        except Exception as e:
            print(f"Error generating pytest.ini: {e}")
            return False
    
    def generate_conftest(self) -> bool:
        """Generate conftest.py with shared fixtures"""
        try:
            conftest_content = '''"""
Shared pytest fixtures and configuration
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return test data directory"""
    return project_root / "tests" / "data"


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for tests"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test"""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# Custom markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
'''
            
            # Write file
            conftest_path = self.project_root / "tests" / "conftest.py"
            conftest_path.parent.mkdir(parents=True, exist_ok=True)
            conftest_path.write_text(conftest_content)
            
            return True
            
        except Exception as e:
            print(f"Error generating conftest.py: {e}")
            return False
    
    def generate_tox_ini(self) -> bool:
        """Generate tox.ini for testing multiple Python versions"""
        try:
            tox_content = """[tox]
envlist = py38, py39, py310, py311, lint, coverage

[testenv]
deps =
    pytest
    pytest-cov
    pytest-asyncio
    pytest-timeout
    pytest-mock
commands =
    pytest {posargs}

[testenv:lint]
deps =
    flake8
    black
    mypy
commands =
    flake8 .
    black --check .
    mypy .

[testenv:coverage]
deps =
    pytest
    pytest-cov
commands =
    pytest --cov --cov-report=html --cov-report=term
"""
            
            # Write file
            tox_path = self.project_root / "tox.ini"
            tox_path.write_text(tox_content)
            
            return True
            
        except Exception as e:
            print(f"Error generating tox.ini: {e}")
            return False


# Example usage
def test_pytest_integration():
    """Test pytest integration"""
    print("\n" + "="*60)
    print("Testing PyTest Integration")
    print("="*60)
    
    # Create test generator
    generator = PytestTestGenerator()
    
    # Generate test file
    success, output_path, errors = generator.generate_test_file(
        "visual_code_builder.py",
        output_dir="tests"
    )
    
    if success:
        print(f"\n✅ Generated test file: {output_path}")
        
        # Show preview
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"\n📝 Preview (first 50 lines):")
                print("-" * 40)
                for line in lines[:50]:
                    print(line)
    else:
        print(f"❌ Failed to generate tests: {errors}")
    
    # Test configuration generation
    print("\n" + "-"*40)
    print("PyTest Configuration:")
    
    config_gen = PytestConfigGenerator(".")
    
    if config_gen.generate_pytest_ini():
        print("✅ Generated pytest.ini")
    
    if config_gen.generate_conftest():
        print("✅ Generated conftest.py")
    
    if config_gen.generate_tox_ini():
        print("✅ Generated tox.ini")
    
    return generator


if __name__ == "__main__":
    print("PyTest Integration for Python Testing")
    print("="*60)
    
    generator = test_pytest_integration()
    
    print("\n✅ PyTest Integration ready!")