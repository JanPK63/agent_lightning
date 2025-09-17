#!/usr/bin/env python3
"""
Jest Integration for JavaScript Testing
Automated Jest test generation and execution for JavaScript/TypeScript projects
"""

import os
import sys
import json
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_generator import TestCase, TestType, AssertionType
from test_case_generator_advanced import TestDataGenerator
from assertion_builder import AssertionBuilder, AssertionCategory
from mock_data_generator import MockDataGenerator, MockDataType


class JestTestType(Enum):
    """Jest-specific test types"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SNAPSHOT = "snapshot"
    COMPONENT = "component"
    E2E = "e2e"
    PERFORMANCE = "performance"


class JestMatcher(Enum):
    """Jest assertion matchers"""
    TO_BE = "toBe"
    TO_EQUAL = "toEqual"
    TO_BE_NULL = "toBeNull"
    TO_BE_UNDEFINED = "toBeUndefined"
    TO_BE_DEFINED = "toBeDefined"
    TO_BE_TRUTHY = "toBeTruthy"
    TO_BE_FALSY = "toBeFalsy"
    TO_BE_GREATER_THAN = "toBeGreaterThan"
    TO_BE_LESS_THAN = "toBeLessThan"
    TO_BE_GREATER_THAN_OR_EQUAL = "toBeGreaterThanOrEqual"
    TO_BE_LESS_THAN_OR_EQUAL = "toBeLessThanOrEqual"
    TO_BE_CLOSE_TO = "toBeCloseTo"
    TO_MATCH = "toMatch"
    TO_CONTAIN = "toContain"
    TO_THROW = "toThrow"
    TO_HAVE_LENGTH = "toHaveLength"
    TO_HAVE_PROPERTY = "toHaveProperty"
    TO_BE_INSTANCE_OF = "toBeInstanceOf"
    TO_MATCH_SNAPSHOT = "toMatchSnapshot"
    TO_MATCH_INLINE_SNAPSHOT = "toMatchInlineSnapshot"


@dataclass
class JestConfig:
    """Jest configuration"""
    test_environment: str = "node"  # node, jsdom, jest-environment-jsdom-fifteen
    coverage: bool = True
    coverage_threshold: Dict[str, int] = field(default_factory=lambda: {
        "branches": 80,
        "functions": 80,
        "lines": 80,
        "statements": 80
    })
    test_match: List[str] = field(default_factory=lambda: [
        "**/__tests__/**/*.[jt]s?(x)",
        "**/?(*.)+(spec|test).[jt]s?(x)"
    ])
    transform: Dict[str, str] = field(default_factory=lambda: {
        "^.+\\.tsx?$": "ts-jest",
        "^.+\\.jsx?$": "babel-jest"
    })
    module_name_mapper: Dict[str, str] = field(default_factory=dict)
    setup_files: List[str] = field(default_factory=list)
    test_timeout: int = 5000
    verbose: bool = True
    collect_coverage_from: List[str] = field(default_factory=lambda: [
        "src/**/*.{js,jsx,ts,tsx}",
        "!src/**/*.d.ts",
        "!src/index.js",
        "!src/serviceWorker.js"
    ])


@dataclass
class JestTestFile:
    """Represents a Jest test file"""
    file_path: str
    source_file: str
    test_cases: List[TestCase] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    mocks: Dict[str, str] = field(default_factory=dict)


class JestTestGenerator:
    """Generates Jest test files"""
    
    def __init__(self):
        self.data_generator = TestDataGenerator()
        self.mock_generator = MockDataGenerator()
        self.assertion_builder = AssertionBuilder()
    
    def generate_test_file(
        self,
        source_file: str,
        test_cases: List[TestCase],
        output_dir: str = "__tests__"
    ) -> Tuple[bool, str, List[str]]:
        """Generate a Jest test file for given test cases"""
        try:
            # Prepare test file
            test_file = self._prepare_test_file(source_file, test_cases)
            
            # Generate test code
            test_code = self._generate_jest_code(test_file)
            
            # Write test file
            output_path = self._write_test_file(test_file, test_code, output_dir)
            
            return True, output_path, []
            
        except Exception as e:
            return False, "", [str(e)]
    
    def _prepare_test_file(
        self,
        source_file: str,
        test_cases: List[TestCase]
    ) -> JestTestFile:
        """Prepare test file structure"""
        test_file = JestTestFile(
            file_path="",
            source_file=source_file,
            test_cases=test_cases
        )
        
        # Determine imports
        test_file.imports = self._generate_imports(source_file, test_cases)
        
        # Generate mocks if needed
        test_file.mocks = self._generate_mocks(test_cases)
        
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
        imports = []
        
        # Import from source file
        module_name = Path(source_file).stem
        relative_path = self._get_relative_import_path(source_file)
        imports.append(f"import {{ {module_name} }} from '{relative_path}';")
        
        # Check if we need additional imports
        needs_react = any("component" in str(tc.test_type).lower() for tc in test_cases)
        needs_enzyme = any("shallow" in str(tc).lower() or "mount" in str(tc).lower() 
                           for tc in test_cases if hasattr(tc, 'tags'))
        needs_axios = any("api" in str(tc.function_under_test).lower() for tc in test_cases)
        
        if needs_react:
            imports.append("import React from 'react';")
            imports.append("import { render, screen, fireEvent } from '@testing-library/react';")
        
        if needs_enzyme:
            imports.append("import { shallow, mount } from 'enzyme';")
        
        if needs_axios:
            imports.append("import axios from 'axios';")
            imports.append("jest.mock('axios');")
        
        return imports
    
    def _generate_mocks(self, test_cases: List[TestCase]) -> Dict[str, str]:
        """Generate mock implementations"""
        mocks = {}
        
        for test_case in test_cases:
            for mock_name, mock_value in test_case.mocks.items():
                if mock_name not in mocks:
                    mocks[mock_name] = self._generate_mock_implementation(
                        mock_name,
                        mock_value
                    )
        
        return mocks
    
    def _generate_mock_implementation(
        self,
        mock_name: str,
        mock_value: Any
    ) -> str:
        """Generate mock implementation code"""
        if "api" in mock_name.lower() or "fetch" in mock_name.lower():
            return f"""
const mock{mock_name} = jest.fn().mockResolvedValue({{
  data: {json.dumps(self.mock_generator.generate(MockDataType.API_RESPONSE))},
  status: 200
}});"""
        elif "database" in mock_name.lower() or "db" in mock_name.lower():
            return f"""
const mock{mock_name} = {{
  query: jest.fn().mockResolvedValue([]),
  insert: jest.fn().mockResolvedValue({{ id: 1 }}),
  update: jest.fn().mockResolvedValue({{ affected: 1 }}),
  delete: jest.fn().mockResolvedValue({{ affected: 1 }})
}};"""
        else:
            return f"const mock{mock_name} = jest.fn();"
    
    def _generate_setup(self) -> str:
        """Generate setup code"""
        return """
beforeEach(() => {
  jest.clearAllMocks();
});"""
    
    def _generate_teardown(self) -> str:
        """Generate teardown code"""
        return """
afterEach(() => {
  jest.restoreAllMocks();
});"""
    
    def _generate_jest_code(self, test_file: JestTestFile) -> str:
        """Generate complete Jest test code"""
        code_parts = []
        
        # Add imports
        code_parts.append("\n".join(test_file.imports))
        code_parts.append("")
        
        # Add mocks
        if test_file.mocks:
            code_parts.append("// Mocks")
            for mock_code in test_file.mocks.values():
                code_parts.append(mock_code)
            code_parts.append("")
        
        # Add describe block
        module_name = Path(test_file.source_file).stem
        code_parts.append(f"describe('{module_name}', () => {{")
        
        # Add setup/teardown
        if test_file.setup_code:
            code_parts.append(test_file.setup_code)
        if test_file.teardown_code:
            code_parts.append(test_file.teardown_code)
        code_parts.append("")
        
        # Group test cases by function
        tests_by_function = {}
        for test_case in test_file.test_cases:
            func_name = test_case.function_under_test
            if func_name not in tests_by_function:
                tests_by_function[func_name] = []
            tests_by_function[func_name].append(test_case)
        
        # Generate test suites
        for func_name, test_cases in tests_by_function.items():
            code_parts.append(f"  describe('{func_name}', () => {{")
            
            for test_case in test_cases:
                test_code = self._generate_test_case_code(test_case)
                code_parts.append(test_code)
            
            code_parts.append("  });")
            code_parts.append("")
        
        code_parts.append("});")
        
        return "\n".join(code_parts)
    
    def _generate_test_case_code(self, test_case: TestCase) -> str:
        """Generate code for a single test case"""
        lines = []
        
        # Determine if async
        is_async = "async" in test_case.tags or test_case.test_type == TestType.E2E
        
        # Test declaration
        test_it = "it" if not test_case.skip_condition else "it.skip"
        async_prefix = "async " if is_async else ""
        
        lines.append(f"    {test_it}('{test_case.description}', {async_prefix}() => {{")
        
        # Setup
        if test_case.setup:
            lines.append(f"      // Setup")
            lines.append(f"      {test_case.setup}")
        
        # Arrange
        if test_case.inputs:
            lines.append(f"      // Arrange")
            for key, value in test_case.inputs.items():
                value_str = self._format_value(value)
                lines.append(f"      const {key} = {value_str};")
        
        # Act
        lines.append(f"      // Act")
        func_call = self._generate_function_call(test_case)
        if is_async:
            lines.append(f"      const result = await {func_call};")
        else:
            lines.append(f"      const result = {func_call};")
        
        # Assert
        lines.append(f"      // Assert")
        for assertion_type, expected in test_case.assertions:
            assertion_code = self._generate_jest_assertion(assertion_type, expected)
            lines.append(f"      {assertion_code}")
        
        # Teardown
        if test_case.teardown:
            lines.append(f"      // Teardown")
            lines.append(f"      {test_case.teardown}")
        
        lines.append("    });")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_function_call(self, test_case: TestCase) -> str:
        """Generate function call code"""
        func_name = test_case.function_under_test
        
        # Handle class methods
        if "." in func_name:
            parts = func_name.split(".")
            if len(parts) == 2:
                class_name, method_name = parts
                instance_name = class_name[0].lower() + class_name[1:]
                
                # Check if we need to instantiate
                if method_name == "__init__" or method_name == "constructor":
                    params = self._format_parameters(test_case.inputs)
                    return f"new {class_name}({params})"
                else:
                    params = self._format_parameters(test_case.inputs)
                    return f"{instance_name}.{method_name}({params})"
        
        # Regular function
        params = self._format_parameters(test_case.inputs)
        return f"{func_name}({params})"
    
    def _format_parameters(self, inputs: Dict[str, Any]) -> str:
        """Format function parameters"""
        if not inputs:
            return ""
        
        params = []
        for key, value in inputs.items():
            params.append(key)
        
        return ", ".join(params)
    
    def _format_value(self, value: Any) -> str:
        """Format a value for JavaScript"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            # Escape quotes and special characters
            escaped = value.replace("'", "\\'").replace('"', '\\"')
            return f"'{escaped}'"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            items = [self._format_value(item) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            pairs = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return f"{{{', '.join(pairs)}}}"
        else:
            return json.dumps(value)
    
    def _generate_jest_assertion(
        self,
        assertion_type: AssertionType,
        expected: Any
    ) -> str:
        """Generate Jest assertion code"""
        matcher_map = {
            AssertionType.EQUALS: JestMatcher.TO_BE,
            AssertionType.NOT_EQUALS: JestMatcher.TO_BE,
            AssertionType.GREATER_THAN: JestMatcher.TO_BE_GREATER_THAN,
            AssertionType.LESS_THAN: JestMatcher.TO_BE_LESS_THAN,
            AssertionType.CONTAINS: JestMatcher.TO_CONTAIN,
            AssertionType.NOT_CONTAINS: JestMatcher.TO_CONTAIN,
            AssertionType.IS_TRUE: JestMatcher.TO_BE_TRUTHY,
            AssertionType.IS_FALSE: JestMatcher.TO_BE_FALSY,
            AssertionType.IS_NONE: JestMatcher.TO_BE_NULL,
            AssertionType.IS_NOT_NONE: JestMatcher.TO_BE_DEFINED,
            AssertionType.RAISES: JestMatcher.TO_THROW,
            AssertionType.TYPE_CHECK: JestMatcher.TO_BE_INSTANCE_OF,
            AssertionType.LENGTH: JestMatcher.TO_HAVE_LENGTH,
            AssertionType.REGEX_MATCH: JestMatcher.TO_MATCH
        }
        
        matcher = matcher_map.get(assertion_type, JestMatcher.TO_BE)
        
        # Handle special cases
        if assertion_type == AssertionType.NOT_EQUALS:
            return f"expect(result).not.{matcher.value}({self._format_value(expected)});"
        elif assertion_type == AssertionType.NOT_CONTAINS:
            return f"expect(result).not.{matcher.value}({self._format_value(expected)});"
        elif assertion_type == AssertionType.RAISES:
            return f"expect(() => result).{matcher.value}();"
        elif assertion_type == AssertionType.TYPE_CHECK:
            return f"expect(result).{matcher.value}({expected});"
        elif assertion_type in [AssertionType.IS_TRUE, AssertionType.IS_FALSE,
                                AssertionType.IS_NONE, AssertionType.IS_NOT_NONE]:
            return f"expect(result).{matcher.value}();"
        else:
            return f"expect(result).{matcher.value}({self._format_value(expected)});"
    
    def _get_relative_import_path(self, source_file: str) -> str:
        """Get relative import path for source file"""
        # Remove extension
        path = Path(source_file).with_suffix("")
        
        # Convert to relative import
        if path.parts[0] == "src":
            return f"../{'/'.join(path.parts)}"
        else:
            return f"./{'/'.join(path.parts)}"
    
    def _write_test_file(
        self,
        test_file: JestTestFile,
        test_code: str,
        output_dir: str
    ) -> str:
        """Write test file to disk"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        source_name = Path(test_file.source_file).stem
        test_filename = f"{source_name}.test.js"
        output_path = os.path.join(output_dir, test_filename)
        
        # Write file
        with open(output_path, 'w') as f:
            f.write(test_code)
        
        return output_path


class JestRunner:
    """Runs Jest tests and collects results"""
    
    def __init__(self, config: JestConfig = None):
        self.config = config or JestConfig()
    
    def run_tests(
        self,
        test_path: str = None,
        coverage: bool = True,
        watch: bool = False
    ) -> Dict[str, Any]:
        """Run Jest tests"""
        # Build Jest command
        cmd = ["npx", "jest"]
        
        if test_path:
            cmd.append(test_path)
        
        if coverage:
            cmd.append("--coverage")
        
        if watch:
            cmd.append("--watch")
        
        if self.config.verbose:
            cmd.append("--verbose")
        
        cmd.extend(["--json", "--outputFile=jest-results.json"])
        
        try:
            # Run Jest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse results
            if os.path.exists("jest-results.json"):
                with open("jest-results.json", 'r') as f:
                    results = json.load(f)
                os.remove("jest-results.json")
                return self._parse_jest_results(results)
            else:
                return {
                    "success": False,
                    "error": result.stderr or "Jest execution failed"
                }
                
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
    
    def _parse_jest_results(self, results: Dict) -> Dict[str, Any]:
        """Parse Jest test results"""
        return {
            "success": results.get("success", False),
            "numTotalTests": results.get("numTotalTests", 0),
            "numPassedTests": results.get("numPassedTests", 0),
            "numFailedTests": results.get("numFailedTests", 0),
            "numPendingTests": results.get("numPendingTests", 0),
            "testResults": self._summarize_test_results(
                results.get("testResults", [])
            ),
            "coverage": self._parse_coverage(results.get("coverageMap", {}))
        }
    
    def _summarize_test_results(self, test_results: List) -> List[Dict]:
        """Summarize test results"""
        summary = []
        
        for result in test_results:
            summary.append({
                "testFilePath": result.get("testFilePath", ""),
                "status": result.get("status", ""),
                "numPassingTests": result.get("numPassingTests", 0),
                "numFailingTests": result.get("numFailingTests", 0),
                "numPendingTests": result.get("numPendingTests", 0),
                "perfStats": {
                    "runtime": result.get("perfStats", {}).get("runtime", 0)
                }
            })
        
        return summary
    
    def _parse_coverage(self, coverage_map: Dict) -> Dict[str, Any]:
        """Parse coverage information"""
        if not coverage_map:
            return {}
        
        total_coverage = {
            "lines": {"percent": 0, "covered": 0, "total": 0},
            "statements": {"percent": 0, "covered": 0, "total": 0},
            "functions": {"percent": 0, "covered": 0, "total": 0},
            "branches": {"percent": 0, "covered": 0, "total": 0}
        }
        
        # Aggregate coverage from all files
        for file_coverage in coverage_map.values():
            for metric in ["lines", "statements", "functions", "branches"]:
                if metric in file_coverage:
                    data = file_coverage[metric]
                    total_coverage[metric]["covered"] += data.get("covered", 0)
                    total_coverage[metric]["total"] += data.get("total", 0)
        
        # Calculate percentages
        for metric in total_coverage:
            total = total_coverage[metric]["total"]
            if total > 0:
                covered = total_coverage[metric]["covered"]
                total_coverage[metric]["percent"] = round((covered / total) * 100, 2)
        
        return total_coverage


class JestProjectManager:
    """Manages Jest configuration for projects"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.config = JestConfig()
    
    def setup_jest(self, typescript: bool = False) -> Tuple[bool, List[str]]:
        """Set up Jest in a project"""
        messages = []
        
        try:
            # Check if package.json exists
            package_json_path = self.project_root / "package.json"
            if not package_json_path.exists():
                messages.append("No package.json found. Initializing npm project...")
                self._init_npm_project()
            
            # Install Jest dependencies
            messages.append("Installing Jest dependencies...")
            if not self._install_jest_dependencies(typescript):
                return False, ["Failed to install Jest dependencies"]
            
            # Create Jest configuration
            messages.append("Creating Jest configuration...")
            self._create_jest_config(typescript)
            
            # Update package.json scripts
            messages.append("Updating package.json scripts...")
            self._update_package_scripts()
            
            # Create test directories
            messages.append("Creating test directories...")
            self._create_test_directories()
            
            messages.append("Jest setup complete!")
            return True, messages
            
        except Exception as e:
            return False, [str(e)]
    
    def _init_npm_project(self):
        """Initialize npm project"""
        subprocess.run(
            ["npm", "init", "-y"],
            cwd=self.project_root,
            check=True
        )
    
    def _install_jest_dependencies(self, typescript: bool) -> bool:
        """Install Jest and related dependencies"""
        dependencies = [
            "jest",
            "@types/jest",
            "babel-jest",
            "@babel/core",
            "@babel/preset-env"
        ]
        
        if typescript:
            dependencies.extend([
                "ts-jest",
                "typescript",
                "@babel/preset-typescript"
            ])
        
        try:
            subprocess.run(
                ["npm", "install", "--save-dev"] + dependencies,
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _create_jest_config(self, typescript: bool):
        """Create jest.config.js file"""
        config = {
            "testEnvironment": self.config.test_environment,
            "collectCoverage": self.config.coverage,
            "coverageDirectory": "coverage",
            "coverageThreshold": {
                "global": self.config.coverage_threshold
            },
            "testMatch": self.config.test_match,
            "collectCoverageFrom": self.config.collect_coverage_from,
            "testTimeout": self.config.test_timeout,
            "verbose": self.config.verbose
        }
        
        if typescript:
            config["preset"] = "ts-jest"
            config["transform"] = self.config.transform
        
        # Write config file
        config_content = f"module.exports = {json.dumps(config, indent=2)};"
        config_path = self.project_root / "jest.config.js"
        config_path.write_text(config_content)
    
    def _update_package_scripts(self):
        """Update package.json with test scripts"""
        package_json_path = self.project_root / "package.json"
        
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        if "scripts" not in package_data:
            package_data["scripts"] = {}
        
        package_data["scripts"].update({
            "test": "jest",
            "test:watch": "jest --watch",
            "test:coverage": "jest --coverage",
            "test:debug": "node --inspect-brk node_modules/.bin/jest --runInBand"
        })
        
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)
    
    def _create_test_directories(self):
        """Create test directory structure"""
        test_dirs = [
            "__tests__",
            "__tests__/unit",
            "__tests__/integration",
            "__tests__/e2e",
            "__mocks__"
        ]
        
        for dir_name in test_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep to preserve empty directories
            gitkeep = dir_path / ".gitkeep"
            gitkeep.touch()


# Example usage
def test_jest_integration():
    """Test Jest integration"""
    print("\n" + "="*60)
    print("Testing Jest Integration")
    print("="*60)
    
    # Create test generator
    generator = JestTestGenerator()
    
    # Create sample test cases
    test_cases = [
        TestCase(
            name="test_add_function",
            description="should add two numbers correctly",
            test_type=TestType.UNIT,
            function_under_test="add",
            inputs={"a": 2, "b": 3},
            expected_output=5,
            assertions=[(AssertionType.EQUALS, 5)]
        ),
        TestCase(
            name="test_multiply_function",
            description="should multiply two numbers",
            test_type=TestType.UNIT,
            function_under_test="multiply",
            inputs={"x": 4, "y": 5},
            expected_output=20,
            assertions=[(AssertionType.EQUALS, 20)]
        ),
        TestCase(
            name="test_divide_by_zero",
            description="should throw error when dividing by zero",
            test_type=TestType.UNIT,
            function_under_test="divide",
            inputs={"a": 10, "b": 0},
            assertions=[(AssertionType.RAISES, "Error")]
        )
    ]
    
    # Generate test file
    success, output_path, errors = generator.generate_test_file(
        "src/math.js",
        test_cases,
        "__tests__"
    )
    
    if success:
        print(f"\n✅ Generated test file: {output_path}")
        
        # Show preview
        with open(output_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"\n📝 Preview (first 40 lines):")
            print("-" * 40)
            for line in lines[:40]:
                print(line)
    else:
        print(f"❌ Failed to generate tests: {errors}")
    
    # Test Jest configuration
    print("\n" + "-"*40)
    print("Jest Configuration:")
    
    config = JestConfig()
    print(f"  Test Environment: {config.test_environment}")
    print(f"  Coverage Enabled: {config.coverage}")
    print(f"  Coverage Thresholds:")
    for metric, threshold in config.coverage_threshold.items():
        print(f"    - {metric}: {threshold}%")
    
    return generator


if __name__ == "__main__":
    print("Jest Integration for JavaScript Testing")
    print("="*60)
    
    generator = test_jest_integration()
    
    print("\n✅ Jest Integration ready!")