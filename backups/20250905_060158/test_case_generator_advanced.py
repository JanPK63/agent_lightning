#!/usr/bin/env python3
"""
Advanced Test Case Generator Algorithm
Intelligent test case generation using code analysis and ML patterns
"""

import os
import sys
import ast
import json
import random
import string
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import re
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code_analyzer_enhanced import EnhancedCodeAnalyzer, CodeComplexity
from test_generator import TestCase, TestType, AssertionType, TestFramework


class TestStrategy(Enum):
    """Test generation strategies"""
    BOUNDARY_VALUE = "boundary_value"
    EQUIVALENCE_PARTITION = "equivalence_partition"
    DECISION_TABLE = "decision_table"
    STATE_TRANSITION = "state_transition"
    PAIRWISE = "pairwise"
    RANDOM = "random"
    MUTATION = "mutation"
    PROPERTY_BASED = "property_based"
    METAMORPHIC = "metamorphic"
    FUZZING = "fuzzing"


class DataCategory(Enum):
    """Categories of test data"""
    VALID = "valid"
    INVALID = "invalid"
    BOUNDARY = "boundary"
    EDGE_CASE = "edge_case"
    NULL_EMPTY = "null_empty"
    EXTREME = "extreme"
    MALICIOUS = "malicious"


@dataclass
class TestScenario:
    """Comprehensive test scenario"""
    name: str
    description: str
    strategy: TestStrategy
    preconditions: List[str] = field(default_factory=list)
    test_data: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    postconditions: List[str] = field(default_factory=list)
    edge_cases: List[Dict[str, Any]] = field(default_factory=list)
    negative_cases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TestDataSet:
    """Collection of test data for a parameter"""
    parameter_name: str
    parameter_type: str
    valid_values: List[Any] = field(default_factory=list)
    invalid_values: List[Any] = field(default_factory=list)
    boundary_values: List[Any] = field(default_factory=list)
    edge_cases: List[Any] = field(default_factory=list)
    null_values: List[Any] = field(default_factory=list)


class TestDataGenerator:
    """Generates test data based on type and constraints"""
    
    def __init__(self):
        self.type_generators = {
            "str": self._generate_string_data,
            "int": self._generate_integer_data,
            "float": self._generate_float_data,
            "bool": self._generate_boolean_data,
            "list": self._generate_list_data,
            "dict": self._generate_dict_data,
            "tuple": self._generate_tuple_data,
            "set": self._generate_set_data,
            "datetime": self._generate_datetime_data,
            "None": self._generate_none_data,
            "Any": self._generate_any_data
        }
    
    def generate_test_data(
        self,
        param_name: str,
        param_type: str,
        constraints: Dict[str, Any] = None
    ) -> TestDataSet:
        """Generate comprehensive test data for a parameter"""
        dataset = TestDataSet(
            parameter_name=param_name,
            parameter_type=param_type
        )
        
        # Clean up type string
        clean_type = self._clean_type_string(param_type)
        
        # Get appropriate generator
        generator = self.type_generators.get(
            clean_type,
            self._generate_generic_data
        )
        
        # Generate data
        generated = generator(param_name, constraints or {})
        
        # Populate dataset
        dataset.valid_values = generated.get("valid", [])
        dataset.invalid_values = generated.get("invalid", [])
        dataset.boundary_values = generated.get("boundary", [])
        dataset.edge_cases = generated.get("edge", [])
        dataset.null_values = generated.get("null", [None])
        
        return dataset
    
    def _clean_type_string(self, type_str: str) -> str:
        """Clean type string to basic type"""
        if not type_str:
            return "Any"
        
        # Remove Optional, Union, etc.
        type_str = type_str.replace("Optional[", "").replace("]", "")
        type_str = type_str.replace("Union[", "").replace("]", "")
        type_str = type_str.replace("List[", "list").replace("]", "")
        type_str = type_str.replace("Dict[", "dict").replace("]", "")
        type_str = type_str.replace("Tuple[", "tuple").replace("]", "")
        type_str = type_str.replace("Set[", "set").replace("]", "")
        
        # Map common types
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict"
        }
        
        for old, new in type_mapping.items():
            if old in type_str.lower():
                return new
        
        # Extract first type if multiple
        if "," in type_str:
            type_str = type_str.split(",")[0].strip()
        
        return type_str
    
    def _generate_string_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate string test data"""
        data = {
            "valid": [],
            "invalid": [],
            "boundary": [],
            "edge": [],
            "null": [None, ""]
        }
        
        # Valid strings based on parameter name
        if "email" in param_name.lower():
            data["valid"] = [
                "user@example.com",
                "test.user@domain.co.uk",
                "valid+tag@email.org"
            ]
            data["invalid"] = [
                "invalid.email",
                "@nodomain.com",
                "user@",
                "user @email.com"
            ]
        elif "url" in param_name.lower():
            data["valid"] = [
                "https://www.example.com",
                "http://localhost:8080",
                "ftp://files.server.org"
            ]
            data["invalid"] = [
                "not a url",
                "htp://malformed.com",
                "//no-protocol.com"
            ]
        elif "phone" in param_name.lower():
            data["valid"] = [
                "+1-234-567-8900",
                "555-1234",
                "(555) 123-4567"
            ]
            data["invalid"] = [
                "123",
                "phone number",
                "555-CALL"
            ]
        elif "password" in param_name.lower():
            data["valid"] = [
                "SecureP@ss123",
                "ValidPassword!",
                "Str0ng&P@ssw0rd"
            ]
            data["invalid"] = [
                "weak",
                "12345",
                "password"
            ]
        elif "name" in param_name.lower():
            data["valid"] = [
                "John Doe",
                "Alice",
                "Bob Smith-Jones"
            ]
            data["invalid"] = [
                "123",
                "",
                "@#$%"
            ]
        else:
            # Generic strings
            data["valid"] = [
                "normal string",
                "Test Value 123",
                "Multiple\nLines\nHere"
            ]
            data["invalid"] = [
                123,  # Wrong type
                [],   # Wrong type
                {}    # Wrong type
            ]
        
        # Boundary values
        max_length = constraints.get("max_length", 255)
        min_length = constraints.get("min_length", 0)
        
        data["boundary"] = [
            "x" * min_length,  # Minimum length
            "x" * max_length,  # Maximum length
            "x" * (max_length - 1),  # Just under max
            "x" * (min_length + 1),  # Just over min
        ]
        
        # Edge cases
        data["edge"] = [
            "",  # Empty string
            " ",  # Single space
            "   ",  # Only spaces
            "\n\t\r",  # Only whitespace
            "' OR '1'='1",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../../etc/passwd",  # Path traversal
            "A" * 10000,  # Very long string
            "emoji: 😀🎉🚀",  # Emoji characters
            "unicode: Ñoño José",  # Unicode characters
            "\x00\x01\x02",  # Control characters
        ]
        
        return data
    
    def _generate_integer_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate integer test data"""
        data = {
            "valid": [],
            "invalid": [],
            "boundary": [],
            "edge": [],
            "null": [None]
        }
        
        # Get constraints
        min_val = constraints.get("min", -sys.maxsize)
        max_val = constraints.get("max", sys.maxsize)
        
        # Valid values based on parameter name
        if "age" in param_name.lower():
            data["valid"] = [25, 30, 50, 65]
            data["boundary"] = [0, 1, 120, 150]
            data["invalid"] = [-1, -10, 200, 1000]
        elif "count" in param_name.lower() or "size" in param_name.lower():
            data["valid"] = [0, 1, 10, 100]
            data["boundary"] = [0, 1, sys.maxsize]
            data["invalid"] = [-1, -100]
        elif "id" in param_name.lower():
            data["valid"] = [1, 42, 9999, 123456]
            data["boundary"] = [0, 1, sys.maxsize]
            data["invalid"] = [-1, -999]
        elif "port" in param_name.lower():
            data["valid"] = [80, 443, 8080, 3000]
            data["boundary"] = [0, 1, 65535]
            data["invalid"] = [-1, 65536, 99999]
        else:
            # Generic integers
            data["valid"] = [0, 1, -1, 42, 100, -100]
            data["boundary"] = [
                min_val,
                max_val,
                min_val + 1,
                max_val - 1
            ]
        
        # Invalid values (wrong type)
        data["invalid"].extend([
            "not a number",
            3.14,
            [],
            {},
            True
        ])
        
        # Edge cases
        data["edge"] = [
            0,
            -0,  # Negative zero
            sys.maxsize,
            -sys.maxsize,
            2**31 - 1,  # Max 32-bit signed
            -2**31,     # Min 32-bit signed
            2**63 - 1,  # Max 64-bit signed
            -2**63,     # Min 64-bit signed
        ]
        
        return data
    
    def _generate_float_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate float test data"""
        data = {
            "valid": [],
            "invalid": [],
            "boundary": [],
            "edge": [],
            "null": [None]
        }
        
        # Valid values
        data["valid"] = [
            0.0,
            1.0,
            -1.0,
            3.14159,
            0.0001,
            -0.0001,
            1234.5678
        ]
        
        # Invalid values (wrong type)
        data["invalid"] = [
            "not a number",
            [],
            {},
            True
        ]
        
        # Boundary values
        min_val = constraints.get("min", -float('inf'))
        max_val = constraints.get("max", float('inf'))
        
        data["boundary"] = [
            min_val,
            max_val,
            0.0
        ]
        
        # Edge cases
        data["edge"] = [
            float('inf'),
            float('-inf'),
            float('nan'),
            sys.float_info.max,
            sys.float_info.min,
            sys.float_info.epsilon,
            1e308,  # Near overflow
            1e-308  # Near underflow
        ]
        
        return data
    
    def _generate_boolean_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate boolean test data"""
        return {
            "valid": [True, False],
            "invalid": [1, 0, "true", "false", "yes", "no", [], {}],
            "boundary": [True, False],
            "edge": [],
            "null": [None]
        }
    
    def _generate_list_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate list test data"""
        data = {
            "valid": [],
            "invalid": [],
            "boundary": [],
            "edge": [],
            "null": [None, []]
        }
        
        # Valid lists
        data["valid"] = [
            [],
            [1],
            [1, 2, 3],
            ["a", "b", "c"],
            [1, "mixed", 3.14, True],
            list(range(10))
        ]
        
        # Invalid values (wrong type)
        data["invalid"] = [
            "not a list",
            123,
            {},
            True
        ]
        
        # Boundary values
        max_length = constraints.get("max_length", 1000)
        data["boundary"] = [
            [],  # Empty
            [1],  # Single element
            list(range(max_length)),  # Maximum size
        ]
        
        # Edge cases
        data["edge"] = [
            [],  # Empty list
            [None],  # List with None
            [None, None, None],  # Multiple None
            [[]], # Nested empty list
            [[], [], []],  # Multiple nested empty
            [[1, 2], [3, 4]],  # Nested lists
            list(range(10000)),  # Large list
            [1] * 1000,  # Repeated elements
            [float('inf'), float('-inf'), float('nan')],  # Special floats
        ]
        
        return data
    
    def _generate_dict_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate dictionary test data"""
        data = {
            "valid": [],
            "invalid": [],
            "boundary": [],
            "edge": [],
            "null": [None, {}]
        }
        
        # Valid dictionaries
        data["valid"] = [
            {},
            {"key": "value"},
            {"a": 1, "b": 2, "c": 3},
            {"nested": {"inner": "value"}},
            {"mixed": [1, 2, 3], "types": True, "here": 3.14}
        ]
        
        # Invalid values (wrong type)
        data["invalid"] = [
            "not a dict",
            123,
            [],
            True
        ]
        
        # Edge cases
        data["edge"] = [
            {},  # Empty dict
            {"": "empty key"},
            {"None": None},
            {"nested": {"deep": {"deeper": {"deepest": "value"}}}},
            {str(i): i for i in range(1000)},  # Large dict
            {"key with spaces": "value"},
            {"special!@#$%": "characters"},
        ]
        
        return data
    
    def _generate_tuple_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate tuple test data"""
        return {
            "valid": [
                (),
                (1,),
                (1, 2, 3),
                ("a", "b", "c"),
                (1, "mixed", 3.14)
            ],
            "invalid": [
                "not a tuple",
                123,
                [],
                {}
            ],
            "boundary": [
                (),
                (1,),
                tuple(range(100))
            ],
            "edge": [
                (),
                (None,),
                ((), ())
            ],
            "null": [None, ()]
        }
    
    def _generate_set_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate set test data"""
        return {
            "valid": [
                set(),
                {1},
                {1, 2, 3},
                {"a", "b", "c"}
            ],
            "invalid": [
                "not a set",
                123,
                [],
                {}
            ],
            "boundary": [
                set(),
                {1},
                set(range(100))
            ],
            "edge": [
                set(),
                {None},
                {1, 1, 1}  # Duplicates removed
            ],
            "null": [None, set()]
        }
    
    def _generate_datetime_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate datetime test data"""
        now = datetime.now()
        return {
            "valid": [
                now,
                datetime(2024, 1, 1),
                datetime(2000, 1, 1, 0, 0, 0),
                now + timedelta(days=30),
                now - timedelta(days=30)
            ],
            "invalid": [
                "2024-01-01",
                "not a date",
                123,
                []
            ],
            "boundary": [
                datetime.min,
                datetime.max,
                datetime(1970, 1, 1)  # Unix epoch
            ],
            "edge": [
                datetime(9999, 12, 31, 23, 59, 59),
                datetime(1, 1, 1, 0, 0, 0),
                datetime(2000, 2, 29)  # Leap year
            ],
            "null": [None]
        }
    
    def _generate_none_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate None/null test data"""
        return {
            "valid": [None],
            "invalid": ["None", 0, False, ""],
            "boundary": [None],
            "edge": [],
            "null": [None]
        }
    
    def _generate_any_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate data for Any type"""
        return {
            "valid": [
                "string",
                123,
                3.14,
                True,
                [],
                {},
                None
            ],
            "invalid": [],  # Any type accepts everything
            "boundary": [],
            "edge": [
                object(),
                lambda x: x,
                type
            ],
            "null": [None]
        }
    
    def _generate_generic_data(self, param_name: str, constraints: Dict) -> Dict[str, List]:
        """Generate generic test data for unknown types"""
        return {
            "valid": [
                "generic_value",
                123,
                {"generic": "data"}
            ],
            "invalid": [None],
            "boundary": [],
            "edge": [],
            "null": [None]
        }


class AdvancedTestCaseGenerator:
    """Advanced test case generator with multiple strategies"""
    
    def __init__(self):
        self.analyzer = EnhancedCodeAnalyzer()
        self.data_generator = TestDataGenerator()
        self.strategies = {
            TestStrategy.BOUNDARY_VALUE: self._generate_boundary_value_tests,
            TestStrategy.EQUIVALENCE_PARTITION: self._generate_equivalence_tests,
            TestStrategy.DECISION_TABLE: self._generate_decision_table_tests,
            TestStrategy.STATE_TRANSITION: self._generate_state_tests,
            TestStrategy.PAIRWISE: self._generate_pairwise_tests,
            TestStrategy.RANDOM: self._generate_random_tests,
            TestStrategy.MUTATION: self._generate_mutation_tests,
            TestStrategy.PROPERTY_BASED: self._generate_property_tests,
            TestStrategy.FUZZING: self._generate_fuzz_tests
        }
    
    def generate_comprehensive_tests(
        self,
        code_analysis: Dict[str, Any],
        strategies: List[TestStrategy] = None,
        coverage_target: float = 80.0
    ) -> List[TestCase]:
        """Generate comprehensive test cases using multiple strategies"""
        
        if strategies is None:
            # Default strategies based on complexity
            strategies = self._select_strategies(code_analysis)
        
        all_test_cases = []
        
        # Generate tests for each function/method
        for func in code_analysis.get("structure", {}).get("functions", []):
            func_tests = self._generate_function_tests(func, strategies)
            all_test_cases.extend(func_tests)
        
        # Generate tests for classes
        for cls in code_analysis.get("structure", {}).get("classes", []):
            class_tests = self._generate_class_tests(cls, strategies)
            all_test_cases.extend(class_tests)
        
        # Generate integration tests based on call graph
        if code_analysis.get("call_graph"):
            integration_tests = self._generate_integration_tests(
                code_analysis["call_graph"]
            )
            all_test_cases.extend(integration_tests)
        
        # Generate tests based on detected patterns
        if code_analysis.get("patterns"):
            pattern_tests = self._generate_pattern_based_tests(
                code_analysis["patterns"]
            )
            all_test_cases.extend(pattern_tests)
        
        # Prioritize tests
        prioritized = self._prioritize_test_cases(all_test_cases, coverage_target)
        
        return prioritized
    
    def _select_strategies(self, code_analysis: Dict) -> List[TestStrategy]:
        """Select appropriate strategies based on code characteristics"""
        strategies = [TestStrategy.BOUNDARY_VALUE]  # Always include
        
        # Add strategies based on complexity
        complexity = code_analysis.get("metrics", {}).get("cyclomatic_complexity", 0)
        
        if complexity > 10:
            strategies.append(TestStrategy.EQUIVALENCE_PARTITION)
            strategies.append(TestStrategy.DECISION_TABLE)
        
        if complexity > 20:
            strategies.append(TestStrategy.STATE_TRANSITION)
            strategies.append(TestStrategy.PAIRWISE)
        
        if complexity > 30:
            strategies.append(TestStrategy.PROPERTY_BASED)
            strategies.append(TestStrategy.MUTATION)
        
        # Add fuzzing for security-sensitive functions
        if self._has_security_concerns(code_analysis):
            strategies.append(TestStrategy.FUZZING)
        
        return strategies
    
    def _has_security_concerns(self, code_analysis: Dict) -> bool:
        """Check if code has security concerns"""
        security_keywords = [
            "password", "auth", "token", "secret", "key",
            "encrypt", "decrypt", "hash", "sql", "query",
            "file", "upload", "download", "exec", "eval"
        ]
        
        # Check function names
        for func in code_analysis.get("structure", {}).get("functions", []):
            func_name = func.get("name", "").lower()
            if any(keyword in func_name for keyword in security_keywords):
                return True
        
        return False
    
    def _generate_function_tests(
        self,
        func_info: Dict,
        strategies: List[TestStrategy]
    ) -> List[TestCase]:
        """Generate tests for a function using multiple strategies"""
        test_cases = []
        func_name = func_info["name"]
        
        for strategy in strategies:
            strategy_func = self.strategies.get(strategy)
            if strategy_func:
                tests = strategy_func(func_info)
                test_cases.extend(tests)
        
        return test_cases
    
    def _generate_class_tests(
        self,
        class_info: Dict,
        strategies: List[TestStrategy]
    ) -> List[TestCase]:
        """Generate tests for a class"""
        test_cases = []
        class_name = class_info["name"]
        
        # Test class instantiation
        test_cases.append(TestCase(
            name=f"test_{class_name}_instantiation",
            description=f"Test {class_name} instantiation",
            test_type=TestType.UNIT,
            function_under_test=f"{class_name}.__init__",
            assertions=[
                (AssertionType.IS_NOT_NONE, None),
                (AssertionType.TYPE_CHECK, class_name)
            ],
            priority=1
        ))
        
        # Test each method
        for method in class_info.get("methods", []):
            method_tests = self._generate_function_tests(method, strategies)
            # Adjust for class context
            for test in method_tests:
                test.function_under_test = f"{class_name}.{method['name']}"
                test.setup = f"self.obj = {class_name}()"
            test_cases.extend(method_tests)
        
        # Test inheritance if applicable
        if class_info.get("base_classes"):
            test_cases.append(TestCase(
                name=f"test_{class_name}_inheritance",
                description=f"Test {class_name} inheritance",
                test_type=TestType.UNIT,
                function_under_test=class_name,
                assertions=[
                    (AssertionType.TYPE_CHECK, base)
                    for base in class_info["base_classes"]
                ],
                priority=2
            ))
        
        return test_cases
    
    def _generate_boundary_value_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate boundary value tests"""
        test_cases = []
        func_name = func_info["name"]
        
        for param in func_info.get("parameters", []):
            # Generate test data
            test_data = self.data_generator.generate_test_data(
                param["name"],
                param.get("type", "Any")
            )
            
            # Create test for each boundary value
            for i, boundary_value in enumerate(test_data.boundary_values):
                test_cases.append(TestCase(
                    name=f"test_{func_name}_boundary_{param['name']}_{i}",
                    description=f"Boundary test for {param['name']}",
                    test_type=TestType.UNIT,
                    function_under_test=func_name,
                    inputs={param["name"]: boundary_value},
                    assertions=[(AssertionType.NOT_RAISES, None)],
                    tags=["boundary", param["name"]],
                    priority=2
                ))
        
        return test_cases
    
    def _generate_equivalence_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate equivalence partition tests"""
        test_cases = []
        func_name = func_info["name"]
        
        # Create partitions for each parameter
        for param in func_info.get("parameters", []):
            test_data = self.data_generator.generate_test_data(
                param["name"],
                param.get("type", "Any")
            )
            
            # Test representative from each partition
            partitions = {
                "valid": test_data.valid_values[:1],
                "invalid": test_data.invalid_values[:1],
                "edge": test_data.edge_cases[:1]
            }
            
            for partition_name, values in partitions.items():
                for value in values:
                    test_cases.append(TestCase(
                        name=f"test_{func_name}_equiv_{param['name']}_{partition_name}",
                        description=f"Equivalence partition test: {partition_name}",
                        test_type=TestType.UNIT,
                        function_under_test=func_name,
                        inputs={param["name"]: value},
                        assertions=[
                            (AssertionType.RAISES if partition_name == "invalid" 
                             else AssertionType.NOT_RAISES, None)
                        ],
                        tags=["equivalence", partition_name],
                        priority=3
                    ))
        
        return test_cases
    
    def _generate_decision_table_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate decision table tests for complex conditions"""
        test_cases = []
        func_name = func_info["name"]
        
        # Simple decision table for boolean parameters
        bool_params = [
            p for p in func_info.get("parameters", [])
            if "bool" in str(p.get("type", "")).lower()
        ]
        
        if bool_params:
            # Generate all combinations
            num_combinations = 2 ** len(bool_params)
            for i in range(num_combinations):
                inputs = {}
                for j, param in enumerate(bool_params):
                    inputs[param["name"]] = bool(i & (1 << j))
                
                test_cases.append(TestCase(
                    name=f"test_{func_name}_decision_{i}",
                    description=f"Decision table test case {i}",
                    test_type=TestType.UNIT,
                    function_under_test=func_name,
                    inputs=inputs,
                    assertions=[(AssertionType.NOT_RAISES, None)],
                    tags=["decision_table"],
                    priority=3
                ))
        
        return test_cases
    
    def _generate_state_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate state transition tests"""
        test_cases = []
        func_name = func_info["name"]
        
        # Check if function modifies state
        if func_info.get("has_side_effects"):
            test_cases.append(TestCase(
                name=f"test_{func_name}_state_transition",
                description="Test state transitions",
                test_type=TestType.INTEGRATION,
                function_under_test=func_name,
                setup="initial_state = self.get_state()",
                assertions=[
                    (AssertionType.NOT_EQUALS, "initial_state")
                ],
                teardown="self.reset_state()",
                tags=["state", "side_effects"],
                priority=2
            ))
        
        return test_cases
    
    def _generate_pairwise_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate pairwise combination tests"""
        test_cases = []
        func_name = func_info["name"]
        
        if len(func_info.get("parameters", [])) >= 2:
            # Simplified pairwise - test first two parameters
            params = func_info["parameters"][:2]
            
            for i in range(3):  # 3 combinations
                inputs = {}
                for param in params:
                    test_data = self.data_generator.generate_test_data(
                        param["name"],
                        param.get("type", "Any")
                    )
                    if test_data.valid_values:
                        inputs[param["name"]] = test_data.valid_values[
                            i % len(test_data.valid_values)
                        ]
                
                test_cases.append(TestCase(
                    name=f"test_{func_name}_pairwise_{i}",
                    description=f"Pairwise combination test {i}",
                    test_type=TestType.UNIT,
                    function_under_test=func_name,
                    inputs=inputs,
                    assertions=[(AssertionType.NOT_RAISES, None)],
                    tags=["pairwise"],
                    priority=4
                ))
        
        return test_cases
    
    def _generate_random_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate random test cases"""
        test_cases = []
        func_name = func_info["name"]
        
        for i in range(5):  # Generate 5 random tests
            inputs = {}
            for param in func_info.get("parameters", []):
                test_data = self.data_generator.generate_test_data(
                    param["name"],
                    param.get("type", "Any")
                )
                all_values = (
                    test_data.valid_values +
                    test_data.boundary_values +
                    test_data.edge_cases
                )
                if all_values:
                    inputs[param["name"]] = random.choice(all_values)
            
            test_cases.append(TestCase(
                name=f"test_{func_name}_random_{i}",
                description=f"Random test case {i}",
                test_type=TestType.UNIT,
                function_under_test=func_name,
                inputs=inputs,
                assertions=[(AssertionType.NOT_RAISES, None)],
                tags=["random"],
                priority=5
            ))
        
        return test_cases
    
    def _generate_mutation_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate mutation tests"""
        test_cases = []
        func_name = func_info["name"]
        
        # Create mutations of valid inputs
        for param in func_info.get("parameters", []):
            test_data = self.data_generator.generate_test_data(
                param["name"],
                param.get("type", "Any")
            )
            
            if test_data.valid_values:
                base_value = test_data.valid_values[0]
                
                # Create mutations
                mutations = self._mutate_value(base_value)
                
                for i, mutated in enumerate(mutations[:3]):
                    test_cases.append(TestCase(
                        name=f"test_{func_name}_mutation_{param['name']}_{i}",
                        description=f"Mutation test for {param['name']}",
                        test_type=TestType.UNIT,
                        function_under_test=func_name,
                        inputs={param["name"]: mutated},
                        assertions=[(AssertionType.NOT_RAISES, None)],
                        tags=["mutation"],
                        priority=4
                    ))
        
        return test_cases
    
    def _mutate_value(self, value: Any) -> List[Any]:
        """Create mutations of a value"""
        mutations = []
        
        if isinstance(value, str):
            mutations = [
                value.upper(),
                value.lower(),
                value[::-1],  # Reversed
                value + value,  # Doubled
                value[::2],  # Every other char
            ]
        elif isinstance(value, (int, float)):
            mutations = [
                value * -1,  # Negated
                value * 2,   # Doubled
                value // 2,  # Halved
                value + 1,   # Incremented
                value - 1,   # Decremented
            ]
        elif isinstance(value, list):
            mutations = [
                value[::-1],  # Reversed
                value + value,  # Doubled
                value[::2],  # Every other element
                [],  # Empty
                value[:1],  # First element only
            ]
        elif isinstance(value, dict):
            mutations = [
                {k: v for k, v in list(value.items())[::-1]},  # Reversed
                {},  # Empty
                {k: None for k in value},  # Null values
            ]
        else:
            mutations = [value]  # No mutation
        
        return mutations
    
    def _generate_property_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate property-based tests"""
        test_cases = []
        func_name = func_info["name"]
        
        # Define properties to test
        properties = []
        
        # Idempotence: f(f(x)) = f(x)
        if func_info.get("pure_function"):
            properties.append({
                "name": "idempotence",
                "description": "Function should be idempotent",
                "assertion": AssertionType.EQUALS
            })
        
        # Commutativity: f(a, b) = f(b, a)
        if len(func_info.get("parameters", [])) >= 2:
            param_types = [p.get("type", "") for p in func_info["parameters"][:2]]
            if param_types[0] == param_types[1]:
                properties.append({
                    "name": "commutativity",
                    "description": "Function should be commutative",
                    "assertion": AssertionType.EQUALS
                })
        
        # Generate tests for each property
        for prop in properties:
            test_cases.append(TestCase(
                name=f"test_{func_name}_property_{prop['name']}",
                description=prop["description"],
                test_type=TestType.UNIT,
                function_under_test=func_name,
                assertions=[(prop["assertion"], None)],
                tags=["property", prop["name"]],
                priority=3
            ))
        
        return test_cases
    
    def _generate_fuzz_tests(self, func_info: Dict) -> List[TestCase]:
        """Generate fuzz tests for security testing"""
        test_cases = []
        func_name = func_info["name"]
        
        # Fuzz payloads
        fuzz_payloads = [
            "A" * 10000,  # Buffer overflow attempt
            "\x00\x01\x02\x03",  # Binary data
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('XSS')</script>",  # XSS
            "../../../etc/passwd",  # Path traversal
            "\\x41\\x41\\x41\\x41",  # Hex encoding
            "%00",  # Null byte
            "${7*7}",  # Template injection
            "{{7*7}}",  # Template injection
            "__import__('os').system('ls')",  # Python code injection
        ]
        
        for i, payload in enumerate(fuzz_payloads):
            inputs = {}
            for param in func_info.get("parameters", []):
                if "str" in str(param.get("type", "")).lower():
                    inputs[param["name"]] = payload
                    break
            
            if inputs:
                test_cases.append(TestCase(
                    name=f"test_{func_name}_fuzz_{i}",
                    description=f"Fuzz test with payload {i}",
                    test_type=TestType.SECURITY,
                    function_under_test=func_name,
                    inputs=inputs,
                    assertions=[(AssertionType.NOT_RAISES, "SecurityException")],
                    tags=["fuzz", "security"],
                    priority=2
                ))
        
        return test_cases
    
    def _generate_integration_tests(self, call_graph: Dict) -> List[TestCase]:
        """Generate integration tests based on call graph"""
        test_cases = []
        
        # Find functions that call multiple other functions
        for caller, callees in call_graph.items():
            if len(callees) >= 2:
                test_cases.append(TestCase(
                    name=f"test_integration_{caller}",
                    description=f"Integration test for {caller}",
                    test_type=TestType.INTEGRATION,
                    function_under_test=caller,
                    mocks={callee: f"Mock({callee})" for callee in callees},
                    assertions=[(AssertionType.NOT_RAISES, None)],
                    tags=["integration"],
                    priority=2
                ))
        
        return test_cases
    
    def _generate_pattern_based_tests(self, patterns: List) -> List[TestCase]:
        """Generate tests based on detected patterns"""
        test_cases = []
        
        for pattern in patterns:
            # Check if pattern is a dict or object
            if hasattr(pattern, 'pattern_type'):
                pattern_type = pattern.pattern_type
                location = pattern.location
            else:
                pattern_type = pattern.get("pattern_type")
                location = pattern.get("location")
            
            if pattern_type == "long_method":
                # Test timeout for long methods
                test_cases.append(TestCase(
                    name=f"test_timeout_{location}",
                    description="Test method timeout",
                    test_type=TestType.PERFORMANCE,
                    function_under_test="method_at_line_" + str(location),
                    assertions=[(AssertionType.LESS_THAN, 5000)],  # 5 seconds
                    timeout=5000,
                    tags=["performance", "timeout"],
                    priority=3
                ))
            elif pattern_type == "nested_loops":
                # Test performance with large inputs
                test_cases.append(TestCase(
                    name=f"test_nested_loop_perf_{location}",
                    description="Test nested loop performance",
                    test_type=TestType.PERFORMANCE,
                    function_under_test="method_at_line_" + str(location),
                    inputs={"data": list(range(1000))},
                    assertions=[(AssertionType.LESS_THAN, 1000)],  # 1 second
                    tags=["performance", "nested_loops"],
                    priority=3
                ))
        
        return test_cases
    
    def _prioritize_test_cases(
        self,
        test_cases: List[TestCase],
        coverage_target: float
    ) -> List[TestCase]:
        """Prioritize test cases based on importance"""
        # Sort by priority (1 is highest)
        prioritized = sorted(test_cases, key=lambda x: x.priority)
        
        # Calculate estimated coverage
        estimated_coverage = min(len(prioritized) * 2, 100)  # Rough estimate
        
        # If we exceed coverage target, we can be selective
        if estimated_coverage > coverage_target:
            # Keep high priority tests
            essential = [tc for tc in prioritized if tc.priority <= 2]
            optional = [tc for tc in prioritized if tc.priority > 2]
            
            # Add optional tests until we reach target
            tests_needed = int((coverage_target / 100) * len(prioritized))
            return essential + optional[:tests_needed - len(essential)]
        
        return prioritized


# Example usage
def test_advanced_generator():
    """Test the advanced test case generator"""
    print("\n" + "="*60)
    print("Testing Advanced Test Case Generator")
    print("="*60)
    
    # Analyze code first
    analyzer = EnhancedCodeAnalyzer()
    analysis = analyzer.analyze_file("visual_code_builder.py")
    
    if "error" not in analysis:
        # Generate comprehensive tests
        generator = AdvancedTestCaseGenerator()
        test_cases = generator.generate_comprehensive_tests(
            analysis,
            strategies=[
                TestStrategy.BOUNDARY_VALUE,
                TestStrategy.EQUIVALENCE_PARTITION,
                TestStrategy.FUZZING
            ]
        )
        
        print(f"\n✅ Generated {len(test_cases)} test cases")
        
        # Group by type
        by_type = defaultdict(list)
        for tc in test_cases:
            by_type[tc.test_type].append(tc)
        
        print("\nTest Cases by Type:")
        for test_type, cases in by_type.items():
            print(f"  {test_type.value}: {len(cases)} cases")
        
        # Show sample test cases
        print("\nSample Test Cases:")
        for tc in test_cases[:5]:
            print(f"  - {tc.name}: {tc.description}")
            print(f"    Type: {tc.test_type.value}, Priority: {tc.priority}")
            print(f"    Tags: {', '.join(tc.tags)}")
        
        # Test data generation
        print("\n" + "-"*40)
        print("Testing Data Generation:")
        data_gen = TestDataGenerator()
        
        test_params = [
            ("email", "str"),
            ("age", "int"),
            ("is_active", "bool"),
            ("items", "List[str]"),
            ("config", "Dict[str, Any]")
        ]
        
        for param_name, param_type in test_params:
            dataset = data_gen.generate_test_data(param_name, param_type)
            print(f"\n{param_name} ({param_type}):")
            print(f"  Valid: {dataset.valid_values[:3]}")
            print(f"  Invalid: {dataset.invalid_values[:3]}")
            print(f"  Boundary: {dataset.boundary_values[:3]}")
            
    else:
        print(f"❌ Error: {analysis['error']}")
    
    return generator


if __name__ == "__main__":
    print("Advanced Test Case Generator Algorithm")
    print("="*60)
    
    generator = test_advanced_generator()
    
    print("\n✅ Advanced Test Case Generator ready!")