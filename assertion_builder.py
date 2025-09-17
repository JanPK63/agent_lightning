#!/usr/bin/env python3
"""
Assertion Builder for Test Generation
Intelligent assertion generation based on function behavior and return types
"""

import os
import sys
import ast
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import inspect

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_generator import AssertionType, TestFramework


class AssertionCategory(Enum):
    """Categories of assertions"""
    EQUALITY = "equality"
    COMPARISON = "comparison"
    CONTAINMENT = "containment"
    TYPE_CHECKING = "type_checking"
    EXCEPTION = "exception"
    STATE = "state"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CUSTOM = "custom"


class AssertionComplexity(Enum):
    """Complexity levels of assertions"""
    SIMPLE = "simple"          # Single condition
    COMPOUND = "compound"       # Multiple conditions
    NESTED = "nested"          # Nested assertions
    CHAINED = "chained"        # Chained assertions
    CONDITIONAL = "conditional" # Conditional assertions


@dataclass
class Assertion:
    """Represents a test assertion"""
    assertion_type: AssertionType
    category: AssertionCategory
    expected_value: Any = None
    actual_expression: str = "result"
    operator: str = "=="
    message: str = ""
    complexity: AssertionComplexity = AssertionComplexity.SIMPLE
    framework: TestFramework = TestFramework.PYTEST
    is_negative: bool = False
    tolerance: Optional[float] = None  # For float comparisons
    timeout: Optional[int] = None      # For performance assertions
    custom_matcher: Optional[str] = None  # Custom matcher function


@dataclass
class AssertionChain:
    """Chain of related assertions"""
    assertions: List[Assertion] = field(default_factory=list)
    logic_operator: str = "AND"  # AND or OR
    description: str = ""
    
    def add(self, assertion: Assertion) -> 'AssertionChain':
        """Add an assertion to the chain"""
        self.assertions.append(assertion)
        return self
    
    def to_code(self, framework: TestFramework) -> str:
        """Generate code for assertion chain"""
        if framework == TestFramework.PYTEST:
            if self.logic_operator == "AND":
                return " and ".join(a.to_code() for a in self.assertions)
            else:
                return " or ".join(a.to_code() for a in self.assertions)
        return ""


class AssertionBuilder:
    """Builds assertions based on function analysis"""
    
    def __init__(self, framework: TestFramework = TestFramework.PYTEST):
        self.framework = framework
        self.assertion_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load assertion templates for different frameworks"""
        templates = {}
        
        # Pytest templates
        templates["pytest"] = {
            AssertionType.EQUALS: "assert {actual} == {expected}",
            AssertionType.NOT_EQUALS: "assert {actual} != {expected}",
            AssertionType.GREATER_THAN: "assert {actual} > {expected}",
            AssertionType.LESS_THAN: "assert {actual} < {expected}",
            AssertionType.CONTAINS: "assert {expected} in {actual}",
            AssertionType.NOT_CONTAINS: "assert {expected} not in {actual}",
            AssertionType.IS_TRUE: "assert {actual} is True",
            AssertionType.IS_FALSE: "assert {actual} is False",
            AssertionType.IS_NONE: "assert {actual} is None",
            AssertionType.IS_NOT_NONE: "assert {actual} is not None",
            AssertionType.RAISES: "with pytest.raises({expected}): {actual}",
            AssertionType.NOT_RAISES: "# Should not raise {expected}",
            AssertionType.TYPE_CHECK: "assert isinstance({actual}, {expected})",
            AssertionType.LENGTH: "assert len({actual}) == {expected}",
            AssertionType.REGEX_MATCH: "assert re.match(r'{expected}', {actual})"
        }
        
        # Jest templates
        templates["jest"] = {
            AssertionType.EQUALS: "expect({actual}).toBe({expected})",
            AssertionType.NOT_EQUALS: "expect({actual}).not.toBe({expected})",
            AssertionType.GREATER_THAN: "expect({actual}).toBeGreaterThan({expected})",
            AssertionType.LESS_THAN: "expect({actual}).toBeLessThan({expected})",
            AssertionType.CONTAINS: "expect({actual}).toContain({expected})",
            AssertionType.NOT_CONTAINS: "expect({actual}).not.toContain({expected})",
            AssertionType.IS_TRUE: "expect({actual}).toBeTruthy()",
            AssertionType.IS_FALSE: "expect({actual}).toBeFalsy()",
            AssertionType.IS_NONE: "expect({actual}).toBeNull()",
            AssertionType.IS_NOT_NONE: "expect({actual}).not.toBeNull()",
            AssertionType.RAISES: "expect(() => {actual}).toThrow({expected})",
            AssertionType.TYPE_CHECK: "expect(typeof {actual}).toBe('{expected}')",
            AssertionType.LENGTH: "expect({actual}.length).toBe({expected})",
            AssertionType.REGEX_MATCH: "expect({actual}).toMatch(/{expected}/)"
        }
        
        # Unittest templates
        templates["unittest"] = {
            AssertionType.EQUALS: "self.assertEqual({actual}, {expected})",
            AssertionType.NOT_EQUALS: "self.assertNotEqual({actual}, {expected})",
            AssertionType.GREATER_THAN: "self.assertGreater({actual}, {expected})",
            AssertionType.LESS_THAN: "self.assertLess({actual}, {expected})",
            AssertionType.CONTAINS: "self.assertIn({expected}, {actual})",
            AssertionType.NOT_CONTAINS: "self.assertNotIn({expected}, {actual})",
            AssertionType.IS_TRUE: "self.assertTrue({actual})",
            AssertionType.IS_FALSE: "self.assertFalse({actual})",
            AssertionType.IS_NONE: "self.assertIsNone({actual})",
            AssertionType.IS_NOT_NONE: "self.assertIsNotNone({actual})",
            AssertionType.RAISES: "with self.assertRaises({expected}): {actual}",
            AssertionType.TYPE_CHECK: "self.assertIsInstance({actual}, {expected})",
            AssertionType.LENGTH: "self.assertEqual(len({actual}), {expected})",
            AssertionType.REGEX_MATCH: "self.assertRegex({actual}, r'{expected}')"
        }
        
        return templates
    
    def build_assertion(
        self,
        assertion_type: AssertionType,
        expected: Any = None,
        actual: str = "result",
        message: str = ""
    ) -> Assertion:
        """Build a single assertion"""
        
        # Determine category
        category = self._determine_category(assertion_type)
        
        # Create assertion
        assertion = Assertion(
            assertion_type=assertion_type,
            category=category,
            expected_value=expected,
            actual_expression=actual,
            message=message,
            framework=self.framework
        )
        
        return assertion
    
    def build_assertions_for_function(
        self,
        func_info: Dict[str, Any],
        test_input: Dict[str, Any],
        expected_output: Any = None
    ) -> List[Assertion]:
        """Build assertions based on function analysis"""
        assertions = []
        
        # Determine return type
        return_type = func_info.get("return_type", "Any")
        
        # Basic return value assertion
        if expected_output is not None:
            assertions.append(self.build_assertion(
                AssertionType.EQUALS,
                expected=expected_output,
                actual="result"
            ))
        else:
            # Build assertions based on return type
            type_assertions = self._build_type_based_assertions(return_type)
            assertions.extend(type_assertions)
        
        # Add performance assertions for complex functions
        if func_info.get("complexity", 0) > 10:
            assertions.append(self.build_performance_assertion(
                max_time_ms=1000,
                function_name=func_info["name"]
            ))
        
        # Add exception assertions for functions with error handling
        if self._has_error_handling(func_info):
            assertions.extend(self._build_exception_assertions(func_info))
        
        # Add state assertions for functions with side effects
        if func_info.get("has_side_effects"):
            assertions.extend(self._build_state_assertions(func_info))
        
        # Add security assertions for sensitive functions
        if self._is_security_sensitive(func_info):
            assertions.extend(self._build_security_assertions(func_info))
        
        return assertions
    
    def _determine_category(self, assertion_type: AssertionType) -> AssertionCategory:
        """Determine assertion category from type"""
        category_map = {
            AssertionType.EQUALS: AssertionCategory.EQUALITY,
            AssertionType.NOT_EQUALS: AssertionCategory.EQUALITY,
            AssertionType.GREATER_THAN: AssertionCategory.COMPARISON,
            AssertionType.LESS_THAN: AssertionCategory.COMPARISON,
            AssertionType.CONTAINS: AssertionCategory.CONTAINMENT,
            AssertionType.NOT_CONTAINS: AssertionCategory.CONTAINMENT,
            AssertionType.IS_TRUE: AssertionCategory.STATE,
            AssertionType.IS_FALSE: AssertionCategory.STATE,
            AssertionType.IS_NONE: AssertionCategory.STATE,
            AssertionType.IS_NOT_NONE: AssertionCategory.STATE,
            AssertionType.RAISES: AssertionCategory.EXCEPTION,
            AssertionType.NOT_RAISES: AssertionCategory.EXCEPTION,
            AssertionType.TYPE_CHECK: AssertionCategory.TYPE_CHECKING,
            AssertionType.LENGTH: AssertionCategory.COMPARISON,
            AssertionType.REGEX_MATCH: AssertionCategory.CONTAINMENT
        }
        return category_map.get(assertion_type, AssertionCategory.CUSTOM)
    
    def _build_type_based_assertions(self, return_type: str) -> List[Assertion]:
        """Build assertions based on return type"""
        assertions = []
        
        if "str" in return_type.lower():
            assertions.extend([
                self.build_assertion(AssertionType.TYPE_CHECK, expected="str"),
                self.build_assertion(AssertionType.IS_NOT_NONE)
            ])
        elif "int" in return_type.lower():
            assertions.extend([
                self.build_assertion(AssertionType.TYPE_CHECK, expected="int"),
                self.build_assertion(AssertionType.IS_NOT_NONE)
            ])
        elif "float" in return_type.lower():
            assertions.extend([
                self.build_assertion(AssertionType.TYPE_CHECK, expected="float"),
                self.build_assertion(AssertionType.IS_NOT_NONE)
            ])
        elif "bool" in return_type.lower():
            assertions.append(
                self.build_assertion(AssertionType.TYPE_CHECK, expected="bool")
            )
        elif "list" in return_type.lower():
            assertions.extend([
                self.build_assertion(AssertionType.TYPE_CHECK, expected="list"),
                self.build_assertion(AssertionType.IS_NOT_NONE)
            ])
        elif "dict" in return_type.lower():
            assertions.extend([
                self.build_assertion(AssertionType.TYPE_CHECK, expected="dict"),
                self.build_assertion(AssertionType.IS_NOT_NONE)
            ])
        elif "none" in return_type.lower():
            assertions.append(
                self.build_assertion(AssertionType.IS_NONE)
            )
        else:
            # Generic non-None assertion
            assertions.append(
                self.build_assertion(AssertionType.IS_NOT_NONE)
            )
        
        return assertions
    
    def _has_error_handling(self, func_info: Dict) -> bool:
        """Check if function has error handling"""
        # Simple heuristic: check for try/except in function
        return "error" in str(func_info).lower() or "exception" in str(func_info).lower()
    
    def _is_security_sensitive(self, func_info: Dict) -> bool:
        """Check if function is security sensitive"""
        sensitive_keywords = [
            "password", "auth", "token", "secret", "key",
            "encrypt", "decrypt", "hash", "validate", "sanitize"
        ]
        func_name = func_info.get("name", "").lower()
        return any(keyword in func_name for keyword in sensitive_keywords)
    
    def _build_exception_assertions(self, func_info: Dict) -> List[Assertion]:
        """Build exception-related assertions"""
        assertions = []
        
        # Test that valid inputs don't raise
        assertions.append(
            self.build_assertion(AssertionType.NOT_RAISES, expected="Exception")
        )
        
        # Test that invalid inputs do raise
        assertions.append(Assertion(
            assertion_type=AssertionType.RAISES,
            category=AssertionCategory.EXCEPTION,
            expected_value="ValueError",
            actual_expression="func_with_invalid_input()",
            message="Should raise ValueError for invalid input",
            framework=self.framework
        ))
        
        return assertions
    
    def _build_state_assertions(self, func_info: Dict) -> List[Assertion]:
        """Build state-related assertions"""
        assertions = []
        
        # Check state before and after
        assertions.append(Assertion(
            assertion_type=AssertionType.NOT_EQUALS,
            category=AssertionCategory.STATE,
            expected_value="initial_state",
            actual_expression="final_state",
            message="State should change after function call",
            framework=self.framework,
            complexity=AssertionComplexity.COMPOUND
        ))
        
        return assertions
    
    def _build_security_assertions(self, func_info: Dict) -> List[Assertion]:
        """Build security-related assertions"""
        assertions = []
        
        # Check for SQL injection prevention
        if "sql" in func_info.get("name", "").lower():
            assertions.append(Assertion(
                assertion_type=AssertionType.NOT_CONTAINS,
                category=AssertionCategory.SECURITY,
                expected_value="DROP TABLE",
                actual_expression="result",
                message="Should prevent SQL injection",
                framework=self.framework
            ))
        
        # Check for XSS prevention
        if "html" in func_info.get("name", "").lower() or "render" in func_info.get("name", "").lower():
            assertions.append(Assertion(
                assertion_type=AssertionType.NOT_CONTAINS,
                category=AssertionCategory.SECURITY,
                expected_value="<script>",
                actual_expression="result",
                message="Should prevent XSS attacks",
                framework=self.framework
            ))
        
        # Check for path traversal prevention
        if "file" in func_info.get("name", "").lower() or "path" in func_info.get("name", "").lower():
            assertions.append(Assertion(
                assertion_type=AssertionType.NOT_CONTAINS,
                category=AssertionCategory.SECURITY,
                expected_value="../",
                actual_expression="result",
                message="Should prevent path traversal",
                framework=self.framework
            ))
        
        return assertions
    
    def build_performance_assertion(
        self,
        max_time_ms: int,
        function_name: str
    ) -> Assertion:
        """Build performance assertion"""
        return Assertion(
            assertion_type=AssertionType.LESS_THAN,
            category=AssertionCategory.PERFORMANCE,
            expected_value=max_time_ms,
            actual_expression="execution_time_ms",
            message=f"{function_name} should complete within {max_time_ms}ms",
            framework=self.framework,
            timeout=max_time_ms
        )
    
    def build_compound_assertion(
        self,
        assertions: List[Assertion],
        logic: str = "AND"
    ) -> AssertionChain:
        """Build compound assertion from multiple assertions"""
        chain = AssertionChain(logic_operator=logic)
        for assertion in assertions:
            chain.add(assertion)
        return chain
    
    def build_nested_assertion(
        self,
        outer_assertion: Assertion,
        inner_assertions: List[Assertion]
    ) -> Assertion:
        """Build nested assertion"""
        outer_assertion.complexity = AssertionComplexity.NESTED
        # Store inner assertions as custom matcher
        outer_assertion.custom_matcher = self._generate_nested_code(
            outer_assertion,
            inner_assertions
        )
        return outer_assertion
    
    def _generate_nested_code(
        self,
        outer: Assertion,
        inner: List[Assertion]
    ) -> str:
        """Generate code for nested assertions"""
        if self.framework == TestFramework.PYTEST:
            inner_code = " and ".join(self.to_code(a) for a in inner)
            return f"if {self.to_code(outer)}: assert {inner_code}"
        return ""
    
    def to_code(self, assertion: Assertion) -> str:
        """Convert assertion to code"""
        framework_name = self.framework.value
        templates = self.assertion_templates.get(framework_name, {})
        template = templates.get(assertion.assertion_type, "")
        
        if not template:
            return f"# Unsupported assertion type: {assertion.assertion_type}"
        
        # Handle custom matcher
        if assertion.custom_matcher:
            return assertion.custom_matcher
        
        # Format the template
        code = template.format(
            actual=assertion.actual_expression,
            expected=self._format_expected_value(assertion.expected_value)
        )
        
        # Add message if present
        if assertion.message and self.framework == TestFramework.PYTEST:
            code += f", '{assertion.message}'"
        
        return code
    
    def _format_expected_value(self, value: Any) -> str:
        """Format expected value for code generation"""
        if value is None:
            return "None"
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return str(value)
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)


class SmartAssertionGenerator:
    """Generates intelligent assertions based on code behavior"""
    
    def __init__(self):
        self.builder = AssertionBuilder()
        self.patterns = self._load_assertion_patterns()
    
    def _load_assertion_patterns(self) -> Dict[str, List[AssertionType]]:
        """Load common assertion patterns"""
        return {
            "getter": [AssertionType.IS_NOT_NONE, AssertionType.TYPE_CHECK],
            "setter": [AssertionType.NOT_RAISES, AssertionType.EQUALS],
            "validator": [AssertionType.IS_TRUE, AssertionType.IS_FALSE, AssertionType.RAISES],
            "calculator": [AssertionType.TYPE_CHECK, AssertionType.GREATER_THAN],
            "transformer": [AssertionType.NOT_EQUALS, AssertionType.TYPE_CHECK],
            "filter": [AssertionType.LENGTH, AssertionType.CONTAINS],
            "searcher": [AssertionType.CONTAINS, AssertionType.IS_NOT_NONE],
            "parser": [AssertionType.TYPE_CHECK, AssertionType.NOT_RAISES],
            "builder": [AssertionType.IS_NOT_NONE, AssertionType.TYPE_CHECK],
            "factory": [AssertionType.TYPE_CHECK, AssertionType.IS_NOT_NONE]
        }
    
    def generate_assertions(
        self,
        func_name: str,
        func_info: Dict[str, Any],
        test_scenario: str = "happy_path"
    ) -> List[Assertion]:
        """Generate assertions based on function name and scenario"""
        assertions = []
        
        # Detect function pattern
        pattern = self._detect_pattern(func_name)
        
        # Get base assertions for pattern
        if pattern in self.patterns:
            base_types = self.patterns[pattern]
            for assertion_type in base_types:
                assertions.append(self.builder.build_assertion(assertion_type))
        
        # Add scenario-specific assertions
        scenario_assertions = self._generate_scenario_assertions(
            test_scenario,
            func_info
        )
        assertions.extend(scenario_assertions)
        
        # Add invariant assertions
        invariants = self._generate_invariant_assertions(func_info)
        assertions.extend(invariants)
        
        return assertions
    
    def _detect_pattern(self, func_name: str) -> str:
        """Detect function pattern from name"""
        func_lower = func_name.lower()
        
        if func_lower.startswith("get_") or func_lower.startswith("fetch_"):
            return "getter"
        elif func_lower.startswith("set_") or func_lower.startswith("update_"):
            return "setter"
        elif "validate" in func_lower or "check" in func_lower:
            return "validator"
        elif "calculate" in func_lower or "compute" in func_lower:
            return "calculator"
        elif "transform" in func_lower or "convert" in func_lower:
            return "transformer"
        elif "filter" in func_lower:
            return "filter"
        elif "search" in func_lower or "find" in func_lower:
            return "searcher"
        elif "parse" in func_lower:
            return "parser"
        elif "build" in func_lower:
            return "builder"
        elif "create" in func_lower or "make" in func_lower:
            return "factory"
        
        return "generic"
    
    def _generate_scenario_assertions(
        self,
        scenario: str,
        func_info: Dict
    ) -> List[Assertion]:
        """Generate assertions for specific test scenarios"""
        assertions = []
        
        if scenario == "happy_path":
            assertions.extend([
                self.builder.build_assertion(AssertionType.NOT_RAISES),
                self.builder.build_assertion(AssertionType.IS_NOT_NONE)
            ])
        elif scenario == "edge_case":
            assertions.extend([
                self.builder.build_assertion(AssertionType.NOT_RAISES),
                self.builder.build_assertion(
                    AssertionType.TYPE_CHECK,
                    expected=func_info.get("return_type", "Any")
                )
            ])
        elif scenario == "error_case":
            assertions.append(
                self.builder.build_assertion(
                    AssertionType.RAISES,
                    expected="Exception"
                )
            )
        elif scenario == "boundary":
            assertions.extend([
                self.builder.build_assertion(AssertionType.NOT_RAISES),
                self.builder.build_assertion(AssertionType.IS_NOT_NONE),
                self.builder.build_assertion(
                    AssertionType.GREATER_THAN,
                    expected=0,
                    actual="len(result)" if "list" in str(func_info.get("return_type", "")) else "result"
                )
            ])
        
        return assertions
    
    def _generate_invariant_assertions(self, func_info: Dict) -> List[Assertion]:
        """Generate invariant assertions that should always hold"""
        invariants = []
        
        # Type invariants
        if func_info.get("return_type"):
            return_type = func_info["return_type"]
            if "Optional" not in return_type and "None" not in return_type:
                invariants.append(
                    self.builder.build_assertion(AssertionType.IS_NOT_NONE)
                )
        
        # Pure function invariants
        if func_info.get("pure_function"):
            # Idempotence: f(f(x)) == f(x)
            invariants.append(Assertion(
                assertion_type=AssertionType.EQUALS,
                category=AssertionCategory.STATE,
                expected_value="first_call_result",
                actual_expression="second_call_result",
                message="Pure function should be idempotent",
                framework=self.builder.framework
            ))
        
        # Collection invariants
        if "list" in str(func_info.get("return_type", "")).lower():
            invariants.append(Assertion(
                assertion_type=AssertionType.TYPE_CHECK,
                category=AssertionCategory.TYPE_CHECKING,
                expected_value="list",
                actual_expression="result",
                framework=self.builder.framework
            ))
        
        return invariants


class PropertyBasedAssertionGenerator:
    """Generates property-based assertions"""
    
    def __init__(self):
        self.builder = AssertionBuilder()
        self.properties = self._define_properties()
    
    def _define_properties(self) -> Dict[str, Callable]:
        """Define common properties to test"""
        return {
            "idempotence": self._check_idempotence,
            "commutativity": self._check_commutativity,
            "associativity": self._check_associativity,
            "distributivity": self._check_distributivity,
            "identity": self._check_identity,
            "inverse": self._check_inverse,
            "monotonicity": self._check_monotonicity,
            "symmetry": self._check_symmetry
        }
    
    def generate_property_assertions(
        self,
        func_info: Dict,
        property_names: List[str] = None
    ) -> List[Assertion]:
        """Generate property-based assertions"""
        assertions = []
        
        if property_names is None:
            # Auto-detect applicable properties
            property_names = self._detect_applicable_properties(func_info)
        
        for prop_name in property_names:
            if prop_name in self.properties:
                prop_assertions = self.properties[prop_name](func_info)
                assertions.extend(prop_assertions)
        
        return assertions
    
    def _detect_applicable_properties(self, func_info: Dict) -> List[str]:
        """Detect which properties apply to function"""
        applicable = []
        
        # Check parameter count
        param_count = len(func_info.get("parameters", []))
        
        if func_info.get("pure_function"):
            applicable.append("idempotence")
        
        if param_count >= 2:
            # Check if parameters have same type
            params = func_info.get("parameters", [])
            if len(params) >= 2:
                if params[0].get("type") == params[1].get("type"):
                    applicable.append("commutativity")
                    applicable.append("symmetry")
        
        if param_count >= 3:
            applicable.append("associativity")
        
        # Check for mathematical operations
        func_name = func_info.get("name", "").lower()
        if any(op in func_name for op in ["add", "multiply", "concat"]):
            applicable.extend(["identity", "associativity"])
        
        if "sort" in func_name or "order" in func_name:
            applicable.append("monotonicity")
        
        return applicable
    
    def _check_idempotence(self, func_info: Dict) -> List[Assertion]:
        """Generate idempotence assertion: f(f(x)) = f(x)"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="func(result)",
            actual_expression="func(func(result))",
            message="Function should be idempotent",
            framework=self.builder.framework,
            complexity=AssertionComplexity.COMPOUND
        )]
    
    def _check_commutativity(self, func_info: Dict) -> List[Assertion]:
        """Generate commutativity assertion: f(a,b) = f(b,a)"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="func(a, b)",
            actual_expression="func(b, a)",
            message="Function should be commutative",
            framework=self.builder.framework,
            complexity=AssertionComplexity.COMPOUND
        )]
    
    def _check_associativity(self, func_info: Dict) -> List[Assertion]:
        """Generate associativity assertion: f(f(a,b),c) = f(a,f(b,c))"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="func(func(a, b), c)",
            actual_expression="func(a, func(b, c))",
            message="Function should be associative",
            framework=self.builder.framework,
            complexity=AssertionComplexity.NESTED
        )]
    
    def _check_distributivity(self, func_info: Dict) -> List[Assertion]:
        """Generate distributivity assertion"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="func1(a, func2(b, c))",
            actual_expression="func2(func1(a, b), func1(a, c))",
            message="Function should be distributive",
            framework=self.builder.framework,
            complexity=AssertionComplexity.NESTED
        )]
    
    def _check_identity(self, func_info: Dict) -> List[Assertion]:
        """Generate identity assertion: f(x, identity) = x"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="x",
            actual_expression="func(x, identity_element)",
            message="Function should have identity element",
            framework=self.builder.framework
        )]
    
    def _check_inverse(self, func_info: Dict) -> List[Assertion]:
        """Generate inverse assertion: f(f_inv(x)) = x"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="x",
            actual_expression="func(func_inverse(x))",
            message="Function should have inverse",
            framework=self.builder.framework,
            complexity=AssertionComplexity.COMPOUND
        )]
    
    def _check_monotonicity(self, func_info: Dict) -> List[Assertion]:
        """Generate monotonicity assertion: if a <= b then f(a) <= f(b)"""
        return [Assertion(
            assertion_type=AssertionType.LESS_THAN,
            category=AssertionCategory.CUSTOM,
            expected_value="func(b)",
            actual_expression="func(a)",
            message="Function should be monotonic",
            framework=self.builder.framework,
            complexity=AssertionComplexity.CONDITIONAL
        )]
    
    def _check_symmetry(self, func_info: Dict) -> List[Assertion]:
        """Generate symmetry assertion: f(a,b) = f(b,a)"""
        return [Assertion(
            assertion_type=AssertionType.EQUALS,
            category=AssertionCategory.CUSTOM,
            expected_value="func(a, b)",
            actual_expression="func(b, a)",
            message="Function should be symmetric",
            framework=self.builder.framework
        )]


# Example usage
def test_assertion_builder():
    """Test the assertion builder"""
    print("\n" + "="*60)
    print("Testing Assertion Builder")
    print("="*60)
    
    # Test basic assertion building
    builder = AssertionBuilder(TestFramework.PYTEST)
    
    # Build various assertions
    assertions = [
        builder.build_assertion(AssertionType.EQUALS, expected=42),
        builder.build_assertion(AssertionType.IS_NOT_NONE),
        builder.build_assertion(AssertionType.GREATER_THAN, expected=0),
        builder.build_assertion(AssertionType.CONTAINS, expected="test", actual="result_string"),
        builder.build_assertion(AssertionType.RAISES, expected="ValueError"),
        builder.build_assertion(AssertionType.TYPE_CHECK, expected="str")
    ]
    
    print("\nGenerated Assertions (Pytest):")
    for assertion in assertions:
        code = builder.to_code(assertion)
        print(f"  {assertion.assertion_type.value}: {code}")
    
    # Test with Jest framework
    builder_jest = AssertionBuilder(TestFramework.JEST)
    
    print("\nGenerated Assertions (Jest):")
    for assertion in assertions[:3]:
        assertion.framework = TestFramework.JEST
        code = builder_jest.to_code(assertion)
        print(f"  {assertion.assertion_type.value}: {code}")
    
    # Test smart assertion generation
    smart_gen = SmartAssertionGenerator()
    
    test_functions = [
        ("get_user_name", {"return_type": "str"}),
        ("validate_email", {"return_type": "bool"}),
        ("calculate_total", {"return_type": "float"}),
        ("parse_json", {"return_type": "dict"}),
        ("filter_items", {"return_type": "List[str]"})
    ]
    
    print("\n" + "-"*40)
    print("Smart Assertion Generation:")
    
    for func_name, func_info in test_functions:
        assertions = smart_gen.generate_assertions(func_name, func_info)
        print(f"\n{func_name}:")
        for assertion in assertions[:3]:
            print(f"  - {assertion.assertion_type.value}")
    
    # Test property-based assertions
    prop_gen = PropertyBasedAssertionGenerator()
    
    print("\n" + "-"*40)
    print("Property-Based Assertions:")
    
    func_info = {
        "name": "add_numbers",
        "parameters": [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ],
        "return_type": "int",
        "pure_function": True
    }
    
    prop_assertions = prop_gen.generate_property_assertions(func_info)
    print(f"\nFunction: {func_info['name']}")
    print("Properties to test:")
    for assertion in prop_assertions:
        print(f"  - {assertion.message}")
    
    # Test assertion chaining
    chain = builder.build_compound_assertion([
        builder.build_assertion(AssertionType.IS_NOT_NONE),
        builder.build_assertion(AssertionType.TYPE_CHECK, expected="str"),
        builder.build_assertion(AssertionType.LENGTH, expected=10, actual="result")
    ])
    
    print("\n" + "-"*40)
    print("Chained Assertions:")
    print(f"  Chain has {len(chain.assertions)} assertions")
    print(f"  Logic: {chain.logic_operator}")
    
    return builder


if __name__ == "__main__":
    print("Assertion Builder for Test Generation")
    print("="*60)
    
    builder = test_assertion_builder()
    
    print("\n✅ Assertion Builder initialized successfully!")