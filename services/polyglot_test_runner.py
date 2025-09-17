#!/usr/bin/env python3
"""
Polyglot Test Runner Service - Execute tests across multiple programming languages
Supports Python, Node.js, Go, Java, and more
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
from enum import Enum
import re
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class TestFramework(str, Enum):
    """Test frameworks"""
    # Python
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE = "nose"
    # JavaScript/TypeScript
    JEST = "jest"
    MOCHA = "mocha"
    JASMINE = "jasmine"
    VITEST = "vitest"
    # Go
    GO_TEST = "go_test"
    TESTIFY = "testify"
    # Java
    JUNIT = "junit"
    TESTNG = "testng"
    # Ruby
    RSPEC = "rspec"
    MINITEST = "minitest"
    # C#
    NUNIT = "nunit"
    XUNIT = "xunit"
    # Rust
    CARGO_TEST = "cargo_test"
    # PHP
    PHPUNIT = "phpunit"


class TestRequest(BaseModel):
    """Test execution request"""
    project_path: str = Field(description="Path to project directory")
    language: Optional[Language] = Field(default=None, description="Programming language")
    framework: Optional[TestFramework] = Field(default=None, description="Test framework")
    test_pattern: Optional[str] = Field(default=None, description="Test file pattern")
    coverage: bool = Field(default=True, description="Generate coverage report")
    timeout: int = Field(default=300, description="Timeout in seconds")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")


class TestResult(BaseModel):
    """Test execution result"""
    language: Language
    framework: TestFramework
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage: Optional[Dict[str, Any]] = None
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    output: str
    exit_code: int
    timestamp: datetime


class PolyglotTestRunner:
    """Polyglot Test Runner Service"""
    
    def __init__(self):
        self.app = FastAPI(title="Polyglot Test Runner", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("test_runner")
        self.cache = get_cache()
        
        # Test runners mapping
        self.runners = {
            Language.PYTHON: self._run_python_tests,
            Language.JAVASCRIPT: self._run_javascript_tests,
            Language.TYPESCRIPT: self._run_typescript_tests,
            Language.GO: self._run_go_tests,
            Language.JAVA: self._run_java_tests,
            Language.RUST: self._run_rust_tests,
            Language.RUBY: self._run_ruby_tests,
            Language.CSHARP: self._run_csharp_tests,
            Language.PHP: self._run_php_tests,
            Language.SWIFT: self._run_swift_tests,
            Language.KOTLIN: self._run_kotlin_tests
        }
        
        logger.info("✅ Polyglot Test Runner initialized")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _detect_language(self, project_path: str) -> Optional[Language]:
        """Detect project language based on files"""
        path = Path(project_path)
        
        # Check for language-specific files
        if (path / "package.json").exists():
            return Language.JAVASCRIPT
        elif (path / "tsconfig.json").exists():
            return Language.TYPESCRIPT
        elif (path / "requirements.txt").exists() or (path / "setup.py").exists() or (path / "pyproject.toml").exists():
            return Language.PYTHON
        elif (path / "go.mod").exists():
            return Language.GO
        elif (path / "pom.xml").exists() or (path / "build.gradle").exists():
            return Language.JAVA
        elif (path / "Cargo.toml").exists():
            return Language.RUST
        elif (path / "Gemfile").exists():
            return Language.RUBY
        elif (path / "composer.json").exists():
            return Language.PHP
        elif (path / "Package.swift").exists():
            return Language.SWIFT
        elif any(f.endswith(".csproj") for f in path.glob("*.csproj")):
            return Language.CSHARP
        elif (path / "build.gradle.kts").exists():
            return Language.KOTLIN
        
        return None
    
    def _detect_framework(self, project_path: str, language: Language) -> Optional[TestFramework]:
        """Detect test framework based on project files"""
        path = Path(project_path)
        
        if language == Language.PYTHON:
            if (path / "pytest.ini").exists() or (path / "setup.cfg").exists():
                return TestFramework.PYTEST
            elif any(path.glob("**/test_*.py")) or any(path.glob("**/*_test.py")):
                return TestFramework.PYTEST
            return TestFramework.UNITTEST
            
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            package_json_path = path / "package.json"
            if package_json_path.exists():
                with open(package_json_path) as f:
                    package = json.load(f)
                    deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
                    if "jest" in deps:
                        return TestFramework.JEST
                    elif "mocha" in deps:
                        return TestFramework.MOCHA
                    elif "vitest" in deps:
                        return TestFramework.VITEST
                    elif "jasmine" in deps:
                        return TestFramework.JASMINE
            return TestFramework.JEST
            
        elif language == Language.GO:
            return TestFramework.GO_TEST
            
        elif language == Language.JAVA:
            if (path / "pom.xml").exists():
                return TestFramework.JUNIT
            return TestFramework.JUNIT
            
        elif language == Language.RUST:
            return TestFramework.CARGO_TEST
            
        elif language == Language.RUBY:
            if (path / ".rspec").exists() or (path / "spec").exists():
                return TestFramework.RSPEC
            return TestFramework.MINITEST
            
        elif language == Language.CSHARP:
            return TestFramework.XUNIT
            
        elif language == Language.PHP:
            return TestFramework.PHPUNIT
            
        return None
    
    async def _run_python_tests(self, project_path: str, framework: TestFramework, 
                                request: TestRequest) -> TestResult:
        """Run Python tests"""
        start_time = datetime.now()
        
        # Prepare command based on framework
        if framework == TestFramework.PYTEST:
            cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
            if request.coverage:
                cmd.extend(["--cov=.", "--cov-report=json", "--cov-report=term"])
            if request.test_pattern:
                cmd.append(request.test_pattern)
        elif framework == TestFramework.UNITTEST:
            cmd = ["python", "-m", "unittest", "discover", "-v"]
            if request.test_pattern:
                cmd.extend(["-p", request.test_pattern])
        else:  # nose
            cmd = ["python", "-m", "nose", "-v"]
            if request.coverage:
                cmd.extend(["--with-coverage", "--cover-package=."])
        
        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            # Parse output
            output = result.stdout + result.stderr
            test_result = self._parse_python_output(output, framework)
            
            # Get coverage if available
            coverage = None
            if request.coverage and framework == TestFramework.PYTEST:
                coverage_file = Path(project_path) / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage = json.load(f)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.PYTHON,
                framework=framework,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                coverage=coverage,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_javascript_tests(self, project_path: str, framework: TestFramework,
                                   request: TestRequest) -> TestResult:
        """Run JavaScript/TypeScript tests"""
        start_time = datetime.now()
        
        # Check if npm/yarn is available
        package_manager = "npm"
        if (Path(project_path) / "yarn.lock").exists():
            package_manager = "yarn"
        
        # Prepare command based on framework
        if framework == TestFramework.JEST:
            cmd = [package_manager, "test", "--", "--json", "--outputFile=test-results.json"]
            if request.coverage:
                cmd.append("--coverage")
        elif framework == TestFramework.MOCHA:
            cmd = [package_manager, "test"]
            if request.coverage:
                cmd = ["nyc", package_manager, "test"]
        elif framework == TestFramework.VITEST:
            cmd = [package_manager, "test", "--", "--reporter=json", "--outputFile=test-results.json"]
        else:
            cmd = [package_manager, "test"]
        
        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            
            # Parse results
            test_result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
            
            if framework == TestFramework.JEST:
                results_file = Path(project_path) / "test-results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        jest_results = json.load(f)
                        test_result = self._parse_jest_results(jest_results)
            else:
                test_result = self._parse_javascript_output(output, framework)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.JAVASCRIPT,
                framework=framework,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_typescript_tests(self, project_path: str, framework: TestFramework,
                                   request: TestRequest) -> TestResult:
        """Run TypeScript tests (similar to JavaScript)"""
        return await self._run_javascript_tests(project_path, framework, request)
    
    async def _run_go_tests(self, project_path: str, framework: TestFramework,
                           request: TestRequest) -> TestResult:
        """Run Go tests"""
        start_time = datetime.now()
        
        cmd = ["go", "test", "-v", "-json", "./..."]
        if request.coverage:
            cmd.extend(["-cover", "-coverprofile=coverage.out"])
        if request.test_pattern:
            cmd.extend(["-run", request.test_pattern])
        
        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            test_result = self._parse_go_output(output)
            
            # Get coverage if available
            coverage = None
            if request.coverage:
                cov_result = subprocess.run(
                    ["go", "tool", "cover", "-func=coverage.out"],
                    cwd=project_path,
                    capture_output=True,
                    text=True
                )
                if cov_result.returncode == 0:
                    coverage = {"report": cov_result.stdout}
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.GO,
                framework=TestFramework.GO_TEST,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                coverage=coverage,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_java_tests(self, project_path: str, framework: TestFramework,
                             request: TestRequest) -> TestResult:
        """Run Java tests"""
        start_time = datetime.now()
        
        # Detect build tool
        build_tool = "maven"
        if (Path(project_path) / "build.gradle").exists():
            build_tool = "gradle"
        
        if build_tool == "maven":
            cmd = ["mvn", "test"]
            if request.coverage:
                cmd.append("-Djacoco.skip=false")
        else:
            cmd = ["gradle", "test"]
            if request.coverage:
                cmd.append("jacocoTestReport")
        
        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            test_result = self._parse_java_output(output, build_tool)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.JAVA,
                framework=TestFramework.JUNIT,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_rust_tests(self, project_path: str, framework: TestFramework,
                             request: TestRequest) -> TestResult:
        """Run Rust tests"""
        start_time = datetime.now()
        
        cmd = ["cargo", "test", "--", "--nocapture"]
        if request.test_pattern:
            cmd.append(request.test_pattern)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            test_result = self._parse_rust_output(output)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.RUST,
                framework=TestFramework.CARGO_TEST,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_ruby_tests(self, project_path: str, framework: TestFramework,
                             request: TestRequest) -> TestResult:
        """Run Ruby tests"""
        start_time = datetime.now()
        
        if framework == TestFramework.RSPEC:
            cmd = ["bundle", "exec", "rspec", "--format", "json", "--out", "rspec-results.json"]
        else:
            cmd = ["ruby", "-Itest", "test/test_*.rb"]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            
            test_result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
            if framework == TestFramework.RSPEC:
                results_file = Path(project_path) / "rspec-results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        rspec_results = json.load(f)
                        test_result = self._parse_rspec_results(rspec_results)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.RUBY,
                framework=framework,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_csharp_tests(self, project_path: str, framework: TestFramework,
                                request: TestRequest) -> TestResult:
        """Run C# tests"""
        start_time = datetime.now()
        
        cmd = ["dotnet", "test", "--logger", "json"]
        if request.coverage:
            cmd.extend(["--collect", "Code Coverage"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            test_result = self._parse_dotnet_output(output)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.CSHARP,
                framework=framework,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_php_tests(self, project_path: str, framework: TestFramework,
                            request: TestRequest) -> TestResult:
        """Run PHP tests"""
        start_time = datetime.now()
        
        cmd = ["vendor/bin/phpunit", "--testdox"]
        if request.coverage:
            cmd.extend(["--coverage-text", "--coverage-html=coverage"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            test_result = self._parse_phpunit_output(output)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.PHP,
                framework=TestFramework.PHPUNIT,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_swift_tests(self, project_path: str, framework: TestFramework,
                               request: TestRequest) -> TestResult:
        """Run Swift tests"""
        start_time = datetime.now()
        
        cmd = ["swift", "test"]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                env={**os.environ, **request.env_vars}
            )
            
            output = result.stdout + result.stderr
            test_result = self._parse_swift_output(output)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                language=Language.SWIFT,
                framework=TestFramework.XUNIT,
                total_tests=test_result["total"],
                passed=test_result["passed"],
                failed=test_result["failed"],
                skipped=test_result["skipped"],
                duration=duration,
                failures=test_result["failures"],
                output=output,
                exit_code=result.returncode,
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Tests timed out after {request.timeout} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_kotlin_tests(self, project_path: str, framework: TestFramework,
                                request: TestRequest) -> TestResult:
        """Run Kotlin tests"""
        # Similar to Java tests
        return await self._run_java_tests(project_path, framework, request)
    
    # Output parsing methods
    def _parse_python_output(self, output: str, framework: TestFramework) -> Dict:
        """Parse Python test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        if framework == TestFramework.PYTEST:
            # Parse pytest output
            match = re.search(r'(\d+) passed', output)
            if match:
                result["passed"] = int(match.group(1))
            match = re.search(r'(\d+) failed', output)
            if match:
                result["failed"] = int(match.group(1))
            match = re.search(r'(\d+) skipped', output)
            if match:
                result["skipped"] = int(match.group(1))
            
            result["total"] = result["passed"] + result["failed"] + result["skipped"]
            
            # Extract failures
            failure_pattern = r'FAILED (.+?) - (.+)'
            for match in re.finditer(failure_pattern, output):
                result["failures"].append({
                    "test": match.group(1),
                    "error": match.group(2)
                })
        
        return result
    
    def _parse_javascript_output(self, output: str, framework: TestFramework) -> Dict:
        """Parse JavaScript test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        # Simple regex parsing for common patterns
        if "passing" in output:
            match = re.search(r'(\d+) passing', output)
            if match:
                result["passed"] = int(match.group(1))
        if "failing" in output:
            match = re.search(r'(\d+) failing', output)
            if match:
                result["failed"] = int(match.group(1))
        if "pending" in output:
            match = re.search(r'(\d+) pending', output)
            if match:
                result["skipped"] = int(match.group(1))
        
        result["total"] = result["passed"] + result["failed"] + result["skipped"]
        return result
    
    def _parse_jest_results(self, jest_results: Dict) -> Dict:
        """Parse Jest JSON results"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        if "numTotalTests" in jest_results:
            result["total"] = jest_results["numTotalTests"]
            result["passed"] = jest_results.get("numPassedTests", 0)
            result["failed"] = jest_results.get("numFailedTests", 0)
            result["skipped"] = jest_results.get("numPendingTests", 0)
        
        return result
    
    def _parse_go_output(self, output: str) -> Dict:
        """Parse Go test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        # Parse Go test JSON output
        for line in output.split('\n'):
            if line.strip():
                try:
                    event = json.loads(line)
                    if event.get("Action") == "pass":
                        result["passed"] += 1
                    elif event.get("Action") == "fail":
                        result["failed"] += 1
                        result["failures"].append({
                            "test": event.get("Test", ""),
                            "package": event.get("Package", "")
                        })
                    elif event.get("Action") == "skip":
                        result["skipped"] += 1
                except json.JSONDecodeError:
                    continue
        
        result["total"] = result["passed"] + result["failed"] + result["skipped"]
        return result
    
    def _parse_java_output(self, output: str, build_tool: str) -> Dict:
        """Parse Java test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        if build_tool == "maven":
            # Parse Maven Surefire output
            match = re.search(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)', output)
            if match:
                result["total"] = int(match.group(1))
                result["failed"] = int(match.group(2)) + int(match.group(3))
                result["skipped"] = int(match.group(4))
                result["passed"] = result["total"] - result["failed"] - result["skipped"]
        else:
            # Parse Gradle output
            match = re.search(r'(\d+) tests completed, (\d+) failed', output)
            if match:
                result["total"] = int(match.group(1))
                result["failed"] = int(match.group(2))
                result["passed"] = result["total"] - result["failed"]
        
        return result
    
    def _parse_rust_output(self, output: str) -> Dict:
        """Parse Rust test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        match = re.search(r'test result: .*? (\d+) passed; (\d+) failed', output)
        if match:
            result["passed"] = int(match.group(1))
            result["failed"] = int(match.group(2))
        
        # Look for ignored tests
        match = re.search(r'(\d+) ignored', output)
        if match:
            result["skipped"] = int(match.group(1))
        
        result["total"] = result["passed"] + result["failed"] + result["skipped"]
        return result
    
    def _parse_rspec_results(self, rspec_results: Dict) -> Dict:
        """Parse RSpec JSON results"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        summary = rspec_results.get("summary", {})
        result["total"] = summary.get("example_count", 0)
        result["failed"] = summary.get("failure_count", 0)
        result["skipped"] = summary.get("pending_count", 0)
        result["passed"] = result["total"] - result["failed"] - result["skipped"]
        
        return result
    
    def _parse_dotnet_output(self, output: str) -> Dict:
        """Parse .NET test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        # Parse dotnet test output
        match = re.search(r'Total tests: (\d+)', output)
        if match:
            result["total"] = int(match.group(1))
        match = re.search(r'Passed: (\d+)', output)
        if match:
            result["passed"] = int(match.group(1))
        match = re.search(r'Failed: (\d+)', output)
        if match:
            result["failed"] = int(match.group(1))
        match = re.search(r'Skipped: (\d+)', output)
        if match:
            result["skipped"] = int(match.group(1))
        
        return result
    
    def _parse_phpunit_output(self, output: str) -> Dict:
        """Parse PHPUnit output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        # Parse PHPUnit output
        match = re.search(r'OK \((\d+) tests', output)
        if match:
            result["total"] = int(match.group(1))
            result["passed"] = result["total"]
        else:
            match = re.search(r'Tests: (\d+), Assertions: \d+, Failures: (\d+)', output)
            if match:
                result["total"] = int(match.group(1))
                result["failed"] = int(match.group(2))
                result["passed"] = result["total"] - result["failed"]
        
        return result
    
    def _parse_swift_output(self, output: str) -> Dict:
        """Parse Swift test output"""
        result = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "failures": []}
        
        # Parse Swift test output
        match = re.search(r'Executed (\d+) tests', output)
        if match:
            result["total"] = int(match.group(1))
        
        if "Test Suite 'All tests' passed" in output:
            result["passed"] = result["total"]
        else:
            # Count failures
            failures = re.findall(r'error:', output)
            result["failed"] = len(failures)
            result["passed"] = result["total"] - result["failed"]
        
        return result
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "polyglot_test_runner",
                "status": "healthy",
                "supported_languages": list(Language),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/run")
        async def run_tests(request: TestRequest):
            """Run tests for a project"""
            project_path = request.project_path
            
            # Validate project path
            if not os.path.exists(project_path):
                raise HTTPException(status_code=404, detail=f"Project path not found: {project_path}")
            
            # Detect language if not specified
            language = request.language
            if not language:
                language = self._detect_language(project_path)
                if not language:
                    raise HTTPException(status_code=400, detail="Could not detect project language")
            
            # Detect framework if not specified
            framework = request.framework
            if not framework:
                framework = self._detect_framework(project_path, language)
                if not framework:
                    raise HTTPException(status_code=400, detail="Could not detect test framework")
            
            # Get test runner
            runner = self.runners.get(language)
            if not runner:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
            
            # Run tests
            try:
                result = await runner(project_path, framework, request)
                
                # Cache result
                cache_key = f"test_result:{project_path}:{language}:{framework}"
                self.cache.set(cache_key, result.dict(), ttl=3600)
                
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/languages")
        async def list_languages():
            """List supported languages"""
            return {
                "languages": [
                    {
                        "id": lang.value,
                        "name": lang.value.title(),
                        "frameworks": [f.value for f in TestFramework if lang.value in f.value]
                    }
                    for lang in Language
                ]
            }
        
        @self.app.get("/detect")
        async def detect_project(project_path: str):
            """Detect project language and framework"""
            if not os.path.exists(project_path):
                raise HTTPException(status_code=404, detail=f"Project path not found: {project_path}")
            
            language = self._detect_language(project_path)
            framework = None
            if language:
                framework = self._detect_framework(project_path, language)
            
            return {
                "project_path": project_path,
                "language": language.value if language else None,
                "framework": framework.value if framework else None
            }
        
        @self.app.get("/history/{project_path:path}")
        async def get_test_history(project_path: str):
            """Get test history for a project"""
            # Get all cached results for this project
            pattern = f"test_result:{project_path}:*"
            keys = self.cache.redis_client.keys(pattern)
            
            history = []
            for key in keys:
                result = self.cache.get(key.decode())
                if result:
                    history.append(result)
            
            # Sort by timestamp
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return {
                "project_path": project_path,
                "history": history[:10]  # Last 10 results
            }
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Polyglot Test Runner starting up...")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Polyglot Test Runner shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = PolyglotTestRunner()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("TEST_RUNNER_PORT", 8018))
    logger.info(f"Starting Polyglot Test Runner on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()