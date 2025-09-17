#!/usr/bin/env python3
"""
Test Coverage Analyzer
Analyzes test coverage and generates detailed reports with actionable insights
"""

import os
import sys
import json
import ast
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CoverageLevel(Enum):
    """Coverage level categories"""
    EXCELLENT = "excellent"    # >= 90%
    GOOD = "good"              # >= 75%
    MODERATE = "moderate"      # >= 60%
    LOW = "low"                # >= 40%
    CRITICAL = "critical"      # < 40%


class CoverageType(Enum):
    """Types of coverage metrics"""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    STATEMENT = "statement"
    CONDITION = "condition"
    PATH = "path"


@dataclass
class FileCoverage:
    """Coverage data for a single file"""
    file_path: str
    total_lines: int = 0
    covered_lines: int = 0
    missed_lines: List[int] = field(default_factory=list)
    total_branches: int = 0
    covered_branches: int = 0
    missed_branches: List[Tuple[int, str]] = field(default_factory=list)
    total_functions: int = 0
    covered_functions: int = 0
    missed_functions: List[str] = field(default_factory=list)
    line_rate: float = 0.0
    branch_rate: float = 0.0
    function_rate: float = 0.0
    complexity: int = 0
    
    def calculate_rates(self):
        """Calculate coverage rates"""
        self.line_rate = (self.covered_lines / self.total_lines * 100) if self.total_lines > 0 else 0
        self.branch_rate = (self.covered_branches / self.total_branches * 100) if self.total_branches > 0 else 0
        self.function_rate = (self.covered_functions / self.total_functions * 100) if self.total_functions > 0 else 0
    
    def get_coverage_level(self) -> CoverageLevel:
        """Get coverage level based on line rate"""
        if self.line_rate >= 90:
            return CoverageLevel.EXCELLENT
        elif self.line_rate >= 75:
            return CoverageLevel.GOOD
        elif self.line_rate >= 60:
            return CoverageLevel.MODERATE
        elif self.line_rate >= 40:
            return CoverageLevel.LOW
        else:
            return CoverageLevel.CRITICAL


@dataclass
class PackageCoverage:
    """Coverage data for a package"""
    package_name: str
    files: Dict[str, FileCoverage] = field(default_factory=dict)
    total_lines: int = 0
    covered_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    total_functions: int = 0
    covered_functions: int = 0
    
    def calculate_totals(self):
        """Calculate total coverage metrics"""
        self.total_lines = sum(f.total_lines for f in self.files.values())
        self.covered_lines = sum(f.covered_lines for f in self.files.values())
        self.total_branches = sum(f.total_branches for f in self.files.values())
        self.covered_branches = sum(f.covered_branches for f in self.files.values())
        self.total_functions = sum(f.total_functions for f in self.files.values())
        self.covered_functions = sum(f.covered_functions for f in self.files.values())
    
    def get_line_rate(self) -> float:
        """Get package line coverage rate"""
        return (self.covered_lines / self.total_lines * 100) if self.total_lines > 0 else 0
    
    def get_branch_rate(self) -> float:
        """Get package branch coverage rate"""
        return (self.covered_branches / self.total_branches * 100) if self.total_branches > 0 else 0


@dataclass
class CoverageReport:
    """Complete coverage report"""
    timestamp: datetime
    packages: Dict[str, PackageCoverage] = field(default_factory=dict)
    total_lines: int = 0
    covered_lines: int = 0
    missed_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    missed_branches: int = 0
    total_functions: int = 0
    covered_functions: int = 0
    missed_functions: int = 0
    line_rate: float = 0.0
    branch_rate: float = 0.0
    function_rate: float = 0.0
    coverage_level: CoverageLevel = CoverageLevel.CRITICAL
    
    def calculate_totals(self):
        """Calculate total coverage metrics"""
        for package in self.packages.values():
            package.calculate_totals()
        
        self.total_lines = sum(p.total_lines for p in self.packages.values())
        self.covered_lines = sum(p.covered_lines for p in self.packages.values())
        self.missed_lines = self.total_lines - self.covered_lines
        
        self.total_branches = sum(p.total_branches for p in self.packages.values())
        self.covered_branches = sum(p.covered_branches for p in self.packages.values())
        self.missed_branches = self.total_branches - self.covered_branches
        
        self.total_functions = sum(p.total_functions for p in self.packages.values())
        self.covered_functions = sum(p.covered_functions for p in self.packages.values())
        self.missed_functions = self.total_functions - self.covered_functions
        
        self.line_rate = (self.covered_lines / self.total_lines * 100) if self.total_lines > 0 else 0
        self.branch_rate = (self.covered_branches / self.total_branches * 100) if self.total_branches > 0 else 0
        self.function_rate = (self.covered_functions / self.total_functions * 100) if self.total_functions > 0 else 0
        
        # Determine coverage level
        if self.line_rate >= 90:
            self.coverage_level = CoverageLevel.EXCELLENT
        elif self.line_rate >= 75:
            self.coverage_level = CoverageLevel.GOOD
        elif self.line_rate >= 60:
            self.coverage_level = CoverageLevel.MODERATE
        elif self.line_rate >= 40:
            self.coverage_level = CoverageLevel.LOW
        else:
            self.coverage_level = CoverageLevel.CRITICAL


class CoverageAnalyzer:
    """Main coverage analyzer"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.coverage_data = None
        self.report = None
    
    def run_coverage(
        self,
        test_command: str = None,
        source_dirs: List[str] = None,
        omit_patterns: List[str] = None
    ) -> Tuple[bool, CoverageReport, List[str]]:
        """Run coverage analysis"""
        errors = []
        
        try:
            # Determine test command
            if not test_command:
                test_command = self._detect_test_command()
            
            # Run coverage
            coverage_file = self._run_coverage_command(test_command, source_dirs, omit_patterns)
            
            if not coverage_file:
                errors.append("Failed to generate coverage data")
                return False, None, errors
            
            # Parse coverage data
            self.report = self._parse_coverage_data(coverage_file)
            
            if not self.report:
                errors.append("Failed to parse coverage data")
                return False, None, errors
            
            # Calculate totals
            self.report.calculate_totals()
            
            return True, self.report, errors
            
        except Exception as e:
            errors.append(str(e))
            return False, None, errors
    
    def _detect_test_command(self) -> str:
        """Detect appropriate test command"""
        if (self.project_root / "package.json").exists():
            # JavaScript/TypeScript project
            return "npm test -- --coverage"
        elif (self.project_root / "setup.py").exists() or (self.project_root / "requirements.txt").exists():
            # Python project
            return "pytest --cov --cov-report=xml"
        else:
            # Default to pytest
            return "pytest --cov --cov-report=xml"
    
    def _run_coverage_command(
        self,
        test_command: str,
        source_dirs: List[str] = None,
        omit_patterns: List[str] = None
    ) -> Optional[str]:
        """Run coverage command and return coverage file path"""
        try:
            # Prepare command
            cmd = test_command
            
            # Add source directories if specified
            if source_dirs and "pytest" in cmd:
                src_args = " ".join(f"--cov={d}" for d in source_dirs)
                cmd = cmd.replace("--cov", src_args)
            
            # Run command
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minutes timeout
            )
            
            # Check for coverage files
            coverage_files = [
                "coverage.xml",
                "coverage.json",
                ".coverage",
                "coverage/lcov.info"
            ]
            
            for file in coverage_files:
                file_path = self.project_root / file
                if file_path.exists():
                    return str(file_path)
            
            return None
            
        except subprocess.TimeoutExpired:
            print("Coverage command timed out")
            return None
        except Exception as e:
            print(f"Error running coverage: {e}")
            return None
    
    def _parse_coverage_data(self, coverage_file: str) -> Optional[CoverageReport]:
        """Parse coverage data from file"""
        file_path = Path(coverage_file)
        
        if not file_path.exists():
            return None
        
        # Determine format
        if file_path.suffix == ".xml":
            return self._parse_xml_coverage(file_path)
        elif file_path.suffix == ".json":
            return self._parse_json_coverage(file_path)
        elif file_path.name == ".coverage":
            return self._parse_sqlite_coverage(file_path)
        elif "lcov" in file_path.name:
            return self._parse_lcov_coverage(file_path)
        else:
            return None
    
    def _parse_xml_coverage(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse XML coverage (Cobertura format)"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            report = CoverageReport(timestamp=datetime.now())
            
            # Parse packages
            for package_elem in root.findall(".//package"):
                package_name = package_elem.get("name", "default")
                package = PackageCoverage(package_name=package_name)
                
                # Parse classes/files
                for class_elem in package_elem.findall(".//class"):
                    filename = class_elem.get("filename", "")
                    
                    file_cov = FileCoverage(file_path=filename)
                    
                    # Parse lines
                    lines = class_elem.findall(".//line")
                    covered_lines = set()
                    all_lines = set()
                    
                    for line in lines:
                        line_num = int(line.get("number", 0))
                        hits = int(line.get("hits", 0))
                        all_lines.add(line_num)
                        if hits > 0:
                            covered_lines.add(line_num)
                    
                    file_cov.total_lines = len(all_lines)
                    file_cov.covered_lines = len(covered_lines)
                    file_cov.missed_lines = sorted(all_lines - covered_lines)
                    
                    # Parse methods
                    methods = class_elem.findall(".//method")
                    file_cov.total_functions = len(methods)
                    file_cov.covered_functions = sum(
                        1 for m in methods 
                        if int(m.get("hits", 0)) > 0
                    )
                    file_cov.missed_functions = [
                        m.get("name", "") for m in methods 
                        if int(m.get("hits", 0)) == 0
                    ]
                    
                    file_cov.calculate_rates()
                    package.files[filename] = file_cov
                
                report.packages[package_name] = package
            
            return report
            
        except Exception as e:
            print(f"Error parsing XML coverage: {e}")
            return None
    
    def _parse_json_coverage(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse JSON coverage"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            report = CoverageReport(timestamp=datetime.now())
            
            # Parse files
            files = data.get("files", {})
            
            for file_path, file_data in files.items():
                # Determine package
                parts = Path(file_path).parts
                package_name = parts[0] if len(parts) > 1 else "default"
                
                if package_name not in report.packages:
                    report.packages[package_name] = PackageCoverage(package_name=package_name)
                
                file_cov = FileCoverage(file_path=file_path)
                
                # Parse line coverage
                executed = set(file_data.get("executed_lines", []))
                missing = set(file_data.get("missing_lines", []))
                all_lines = executed | missing
                
                file_cov.total_lines = len(all_lines)
                file_cov.covered_lines = len(executed)
                file_cov.missed_lines = sorted(missing)
                
                # Parse branch coverage if available
                if "branches" in file_data:
                    branches = file_data["branches"]
                    file_cov.total_branches = branches.get("total", 0)
                    file_cov.covered_branches = branches.get("covered", 0)
                
                file_cov.calculate_rates()
                report.packages[package_name].files[file_path] = file_cov
            
            return report
            
        except Exception as e:
            print(f"Error parsing JSON coverage: {e}")
            return None
    
    def _parse_sqlite_coverage(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse SQLite coverage (.coverage file)"""
        try:
            # Use coverage.py to export to JSON
            temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            temp_file.close()
            
            subprocess.run(
                f"coverage json -o {temp_file.name}",
                shell=True,
                cwd=file_path.parent,
                capture_output=True
            )
            
            # Parse the JSON
            report = self._parse_json_coverage(Path(temp_file.name))
            
            # Clean up
            os.unlink(temp_file.name)
            
            return report
            
        except Exception as e:
            print(f"Error parsing SQLite coverage: {e}")
            return None
    
    def _parse_lcov_coverage(self, file_path: Path) -> Optional[CoverageReport]:
        """Parse LCOV coverage"""
        try:
            report = CoverageReport(timestamp=datetime.now())
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse LCOV format
            current_file = None
            current_package = "default"
            
            for line in content.split('\n'):
                if line.startswith("SF:"):
                    # Source file
                    file_name = line[3:]
                    current_file = FileCoverage(file_path=file_name)
                    
                    # Determine package
                    parts = Path(file_name).parts
                    current_package = parts[0] if len(parts) > 1 else "default"
                    
                    if current_package not in report.packages:
                        report.packages[current_package] = PackageCoverage(package_name=current_package)
                    
                elif line.startswith("DA:") and current_file:
                    # Line data
                    parts = line[3:].split(',')
                    if len(parts) == 2:
                        line_num = int(parts[0])
                        hits = int(parts[1])
                        
                        current_file.total_lines += 1
                        if hits > 0:
                            current_file.covered_lines += 1
                        else:
                            current_file.missed_lines.append(line_num)
                
                elif line.startswith("FN:") and current_file:
                    # Function data
                    current_file.total_functions += 1
                
                elif line.startswith("FNDA:") and current_file:
                    # Function hit data
                    parts = line[5:].split(',')
                    if len(parts) >= 2:
                        hits = int(parts[0])
                        func_name = ','.join(parts[1:])
                        
                        if hits > 0:
                            current_file.covered_functions += 1
                        else:
                            current_file.missed_functions.append(func_name)
                
                elif line == "end_of_record" and current_file:
                    # End of file record
                    current_file.calculate_rates()
                    report.packages[current_package].files[current_file.file_path] = current_file
                    current_file = None
            
            return report
            
        except Exception as e:
            print(f"Error parsing LCOV coverage: {e}")
            return None
    
    def analyze_uncovered_code(self, report: CoverageReport) -> Dict[str, List[Dict]]:
        """Analyze uncovered code and provide insights"""
        insights = defaultdict(list)
        
        for package in report.packages.values():
            for file_path, file_cov in package.files.items():
                if file_cov.missed_lines:
                    # Analyze missed lines
                    analysis = self._analyze_missed_lines(file_path, file_cov.missed_lines)
                    if analysis:
                        insights[file_path].extend(analysis)
                
                if file_cov.missed_functions:
                    # Analyze missed functions
                    for func in file_cov.missed_functions:
                        insights[file_path].append({
                            "type": "uncovered_function",
                            "function": func,
                            "priority": "high",
                            "suggestion": f"Add tests for function '{func}'"
                        })
        
        return dict(insights)
    
    def _analyze_missed_lines(self, file_path: str, missed_lines: List[int]) -> List[Dict]:
        """Analyze missed lines in a file"""
        insights = []
        
        if not os.path.exists(file_path):
            return insights
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Group consecutive missed lines
            line_groups = []
            current_group = []
            
            for line_num in sorted(missed_lines):
                if not current_group or line_num == current_group[-1] + 1:
                    current_group.append(line_num)
                else:
                    line_groups.append(current_group)
                    current_group = [line_num]
            
            if current_group:
                line_groups.append(current_group)
            
            # Analyze each group
            for group in line_groups:
                start_line = group[0]
                end_line = group[-1]
                
                # Get code context
                context_start = max(0, start_line - 2)
                context_end = min(len(lines), end_line + 1)
                code_context = ''.join(lines[context_start:context_end])
                
                # Determine type of uncovered code
                if self._is_error_handling(code_context):
                    insights.append({
                        "type": "uncovered_error_handling",
                        "lines": f"{start_line}-{end_line}",
                        "priority": "high",
                        "suggestion": "Add tests for error handling code"
                    })
                elif self._is_edge_case(code_context):
                    insights.append({
                        "type": "uncovered_edge_case",
                        "lines": f"{start_line}-{end_line}",
                        "priority": "medium",
                        "suggestion": "Add tests for edge cases"
                    })
                elif len(group) > 10:
                    insights.append({
                        "type": "large_uncovered_block",
                        "lines": f"{start_line}-{end_line}",
                        "priority": "high",
                        "suggestion": f"Large block of {len(group)} uncovered lines - consider refactoring or adding comprehensive tests"
                    })
            
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
        
        return insights
    
    def _is_error_handling(self, code: str) -> bool:
        """Check if code is error handling"""
        error_patterns = [
            r'\btry\b',
            r'\bcatch\b',
            r'\bexcept\b',
            r'\bfinally\b',
            r'\braise\b',
            r'\bthrow\b',
            r'\.catch\(',
            r'\.error\(',
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False
    
    def _is_edge_case(self, code: str) -> bool:
        """Check if code handles edge cases"""
        edge_patterns = [
            r'if.*[<>]=?\s*0',         # Boundary checks
            r'if.*is\s+None',           # None checks
            r'if\s+not\s+',             # Negation checks
            r'len\([^)]+\)\s*==\s*0',   # Empty checks
            r'\.length\s*===?\s*0',     # JS empty checks
        ]
        
        for pattern in edge_patterns:
            if re.search(pattern, code):
                return True
        return False
    
    def generate_report(
        self,
        report: CoverageReport,
        format: str = "text",
        output_file: str = None
    ) -> str:
        """Generate coverage report in specified format"""
        if format == "text":
            content = self._generate_text_report(report)
        elif format == "html":
            content = self._generate_html_report(report)
        elif format == "json":
            content = self._generate_json_report(report)
        elif format == "markdown":
            content = self._generate_markdown_report(report)
        else:
            content = self._generate_text_report(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(content)
        
        return content
    
    def _generate_text_report(self, report: CoverageReport) -> str:
        """Generate text coverage report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST COVERAGE REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Overall summary
        lines.append("OVERALL COVERAGE")
        lines.append("-" * 40)
        lines.append(f"Line Coverage:     {report.line_rate:.1f}% ({report.covered_lines}/{report.total_lines})")
        lines.append(f"Branch Coverage:   {report.branch_rate:.1f}% ({report.covered_branches}/{report.total_branches})")
        lines.append(f"Function Coverage: {report.function_rate:.1f}% ({report.covered_functions}/{report.total_functions})")
        lines.append(f"Coverage Level:    {report.coverage_level.value.upper()}")
        lines.append("")
        
        # Package breakdown
        lines.append("PACKAGE BREAKDOWN")
        lines.append("-" * 40)
        
        for package_name, package in sorted(report.packages.items()):
            package.calculate_totals()
            lines.append(f"\n📦 {package_name}")
            lines.append(f"   Line Coverage: {package.get_line_rate():.1f}%")
            lines.append(f"   Files: {len(package.files)}")
            
            # File details
            for file_path, file_cov in sorted(package.files.items()):
                level_icon = self._get_level_icon(file_cov.get_coverage_level())
                lines.append(f"   {level_icon} {Path(file_path).name}: {file_cov.line_rate:.1f}%")
                
                if file_cov.missed_lines and len(file_cov.missed_lines) <= 10:
                    lines.append(f"      Missing lines: {', '.join(map(str, file_cov.missed_lines))}")
                elif file_cov.missed_lines:
                    lines.append(f"      Missing: {len(file_cov.missed_lines)} lines")
        
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if report.coverage_level == CoverageLevel.CRITICAL:
            lines.append("⚠️  Critical: Coverage is below 40%. Immediate attention required!")
            lines.append("   - Focus on testing core functionality first")
            lines.append("   - Add unit tests for all public methods")
        elif report.coverage_level == CoverageLevel.LOW:
            lines.append("⚠️  Low coverage: Consider adding more tests")
            lines.append("   - Target untested functions and methods")
            lines.append("   - Add edge case and error handling tests")
        elif report.coverage_level == CoverageLevel.MODERATE:
            lines.append("📊 Moderate coverage: Room for improvement")
            lines.append("   - Aim for at least 75% coverage")
            lines.append("   - Focus on critical paths")
        elif report.coverage_level == CoverageLevel.GOOD:
            lines.append("✅ Good coverage: Keep it up!")
            lines.append("   - Consider reaching for 90% coverage")
        else:
            lines.append("🌟 Excellent coverage!")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report: CoverageReport) -> str:
        """Generate HTML coverage report"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .excellent {{ color: #22c55e; }}
        .good {{ color: #84cc16; }}
        .moderate {{ color: #eab308; }}
        .low {{ color: #f97316; }}
        .critical {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
        .progress {{ background: #e0e0e0; border-radius: 5px; overflow: hidden; }}
        .progress-bar {{ height: 20px; background: #4CAF50; text-align: center; color: white; }}
    </style>
</head>
<body>
    <h1>Test Coverage Report</h1>
    <div class="summary">
        <h2>Overall Coverage: <span class="{report.coverage_level.value}">{report.line_rate:.1f}%</span></h2>
        <p>Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <ul>
            <li>Line Coverage: {report.line_rate:.1f}% ({report.covered_lines}/{report.total_lines})</li>
            <li>Branch Coverage: {report.branch_rate:.1f}% ({report.covered_branches}/{report.total_branches})</li>
            <li>Function Coverage: {report.function_rate:.1f}% ({report.covered_functions}/{report.total_functions})</li>
        </ul>
    </div>
    
    <h2>File Coverage</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Line Coverage</th>
            <th>Branch Coverage</th>
            <th>Missing Lines</th>
        </tr>
"""
        
        for package in report.packages.values():
            for file_path, file_cov in package.files.items():
                level_class = file_cov.get_coverage_level().value
                html += f"""
        <tr>
            <td>{file_path}</td>
            <td>
                <div class="progress">
                    <div class="progress-bar" style="width: {file_cov.line_rate}%">
                        {file_cov.line_rate:.1f}%
                    </div>
                </div>
            </td>
            <td>{file_cov.branch_rate:.1f}%</td>
            <td>{len(file_cov.missed_lines)}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>"""
        
        return html
    
    def _generate_json_report(self, report: CoverageReport) -> str:
        """Generate JSON coverage report"""
        data = {
            "timestamp": report.timestamp.isoformat(),
            "summary": {
                "line_rate": report.line_rate,
                "branch_rate": report.branch_rate,
                "function_rate": report.function_rate,
                "total_lines": report.total_lines,
                "covered_lines": report.covered_lines,
                "missed_lines": report.missed_lines,
                "coverage_level": report.coverage_level.value
            },
            "packages": {}
        }
        
        for package_name, package in report.packages.items():
            package_data = {
                "line_rate": package.get_line_rate(),
                "files": {}
            }
            
            for file_path, file_cov in package.files.items():
                package_data["files"][file_path] = {
                    "line_rate": file_cov.line_rate,
                    "branch_rate": file_cov.branch_rate,
                    "function_rate": file_cov.function_rate,
                    "missed_lines": file_cov.missed_lines,
                    "missed_functions": file_cov.missed_functions
                }
            
            data["packages"][package_name] = package_data
        
        return json.dumps(data, indent=2)
    
    def _generate_markdown_report(self, report: CoverageReport) -> str:
        """Generate Markdown coverage report"""
        lines = []
        lines.append("# Test Coverage Report")
        lines.append("")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Overall Coverage:** {report.line_rate:.1f}% ({report.coverage_level.value.upper()})")
        lines.append("")
        lines.append("| Metric | Coverage | Details |")
        lines.append("|--------|----------|---------|")
        lines.append(f"| Lines | {report.line_rate:.1f}% | {report.covered_lines}/{report.total_lines} |")
        lines.append(f"| Branches | {report.branch_rate:.1f}% | {report.covered_branches}/{report.total_branches} |")
        lines.append(f"| Functions | {report.function_rate:.1f}% | {report.covered_functions}/{report.total_functions} |")
        lines.append("")
        
        # Files
        lines.append("## File Coverage")
        lines.append("")
        lines.append("| File | Line Coverage | Status |")
        lines.append("|------|---------------|--------|")
        
        for package in report.packages.values():
            for file_path, file_cov in package.files.items():
                status = self._get_level_icon(file_cov.get_coverage_level())
                lines.append(f"| {file_path} | {file_cov.line_rate:.1f}% | {status} |")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _get_level_icon(self, level: CoverageLevel) -> str:
        """Get icon for coverage level"""
        icons = {
            CoverageLevel.EXCELLENT: "🌟",
            CoverageLevel.GOOD: "✅",
            CoverageLevel.MODERATE: "📊",
            CoverageLevel.LOW: "⚠️",
            CoverageLevel.CRITICAL: "❌"
        }
        return icons.get(level, "❔")


# Example usage
def test_coverage_analyzer():
    """Test the coverage analyzer"""
    print("\n" + "="*60)
    print("Testing Coverage Analyzer")
    print("="*60)
    
    analyzer = CoverageAnalyzer()
    
    # Run coverage analysis
    print("\n🔍 Running coverage analysis...")
    success, report, errors = analyzer.run_coverage(
        test_command="pytest --cov=. --cov-report=xml tests/",
        source_dirs=["."],
        omit_patterns=["tests/*", "setup.py"]
    )
    
    if success and report:
        # Generate text report
        text_report = analyzer.generate_report(report, format="text")
        print(text_report)
        
        # Analyze uncovered code
        insights = analyzer.analyze_uncovered_code(report)
        
        if insights:
            print("\n📊 Uncovered Code Insights:")
            print("-" * 40)
            
            for file_path, file_insights in insights.items():
                print(f"\n📄 {file_path}:")
                for insight in file_insights:
                    print(f"   {insight['type']}: {insight['suggestion']}")
                    if 'lines' in insight:
                        print(f"      Lines: {insight['lines']}")
        
        # Save reports in different formats
        analyzer.generate_report(report, format="html", output_file="coverage.html")
        analyzer.generate_report(report, format="json", output_file="coverage.json")
        analyzer.generate_report(report, format="markdown", output_file="coverage.md")
        
        print("\n✅ Coverage reports generated:")
        print("   - coverage.html")
        print("   - coverage.json")
        print("   - coverage.md")
        
    else:
        print(f"❌ Coverage analysis failed: {errors}")
    
    return analyzer


if __name__ == "__main__":
    print("Test Coverage Analyzer")
    print("="*60)
    
    analyzer = test_coverage_analyzer()
    
    print("\n✅ Coverage Analyzer ready!")