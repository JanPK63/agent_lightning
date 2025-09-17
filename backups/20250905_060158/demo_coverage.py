#!/usr/bin/env python3
"""
Demo of the coverage analyzer with synthetic data
"""

import json
import os
from datetime import datetime
from coverage_analyzer import (
    CoverageAnalyzer, CoverageReport, PackageCoverage, 
    FileCoverage, CoverageLevel
)


def create_demo_coverage_report():
    """Create a demo coverage report with synthetic data"""
    
    # Create report
    report = CoverageReport(timestamp=datetime.now())
    
    # Package 1: Agent System (Good coverage)
    agent_package = PackageCoverage(package_name="agent_system")
    
    # File 1: agent_collaboration.py
    file1 = FileCoverage(
        file_path="agent_collaboration.py",
        total_lines=450,
        covered_lines=405,
        missed_lines=[45, 67, 89, 123, 156, 178, 234, 267, 289, 301, 
                      312, 334, 356, 378, 390, 401, 412, 423, 434, 445],
        total_branches=80,
        covered_branches=72,
        total_functions=25,
        covered_functions=23,
        missed_functions=["handle_edge_case", "cleanup_resources"]
    )
    file1.calculate_rates()
    agent_package.files["agent_collaboration.py"] = file1
    
    # File 2: orchestrator.py
    file2 = FileCoverage(
        file_path="orchestrator.py",
        total_lines=320,
        covered_lines=288,
        missed_lines=list(range(250, 282)),  # 32 missed lines
        total_branches=60,
        covered_branches=54,
        total_functions=18,
        covered_functions=17,
        missed_functions=["emergency_shutdown"]
    )
    file2.calculate_rates()
    agent_package.files["orchestrator.py"] = file2
    
    report.packages["agent_system"] = agent_package
    
    # Package 2: Visual Builder (Moderate coverage)
    visual_package = PackageCoverage(package_name="visual_builder")
    
    # File 3: visual_code_builder.py
    file3 = FileCoverage(
        file_path="visual_code_builder.py",
        total_lines=680,
        covered_lines=408,  # 60% coverage
        missed_lines=list(range(400, 672)),  # Large uncovered block
        total_branches=120,
        covered_branches=72,
        total_functions=35,
        covered_functions=21,
        missed_functions=[
            "handle_complex_flow", "optimize_layout", "validate_connections",
            "export_to_format", "import_from_file", "merge_diagrams",
            "diff_diagrams", "analyze_complexity", "suggest_improvements",
            "auto_layout", "detect_patterns", "refactor_blocks",
            "generate_documentation", "create_test_cases"
        ]
    )
    file3.calculate_rates()
    visual_package.files["visual_code_builder.py"] = file3
    
    report.packages["visual_builder"] = visual_package
    
    # Package 3: Testing Tools (Excellent coverage)
    testing_package = PackageCoverage(package_name="testing")
    
    # File 4: test_generator.py
    file4 = FileCoverage(
        file_path="test_generator.py",
        total_lines=280,
        covered_lines=266,  # 95% coverage
        missed_lines=[45, 67, 89, 101, 123, 145, 167, 189, 201, 223, 245, 256, 267, 278],
        total_branches=45,
        covered_branches=43,
        total_functions=15,
        covered_functions=15
    )
    file4.calculate_rates()
    testing_package.files["test_generator.py"] = file4
    
    # File 5: assertion_builder.py
    file5 = FileCoverage(
        file_path="assertion_builder.py",
        total_lines=180,
        covered_lines=171,  # 95% coverage
        missed_lines=[34, 56, 78, 90, 112, 134, 156, 167, 178],
        total_branches=30,
        covered_branches=29,
        total_functions=12,
        covered_functions=12
    )
    file5.calculate_rates()
    testing_package.files["assertion_builder.py"] = file5
    
    report.packages["testing"] = testing_package
    
    # Package 4: Utilities (Low coverage - needs attention)
    utils_package = PackageCoverage(package_name="utils")
    
    # File 6: helper_functions.py
    file6 = FileCoverage(
        file_path="helper_functions.py",
        total_lines=150,
        covered_lines=45,  # 30% coverage - critical!
        missed_lines=list(range(46, 151)),  # Most of the file uncovered
        total_branches=25,
        covered_branches=5,
        total_functions=10,
        covered_functions=3,
        missed_functions=[
            "parse_config", "validate_input", "sanitize_output",
            "format_response", "cache_result", "log_error", "retry_operation"
        ]
    )
    file6.calculate_rates()
    utils_package.files["helper_functions.py"] = file6
    
    report.packages["utils"] = utils_package
    
    return report


def main():
    """Demo the coverage analyzer"""
    print("="*80)
    print("COVERAGE ANALYZER DEMONSTRATION")
    print("="*80)
    
    # Create demo report
    print("\n📊 Creating demo coverage report...")
    report = create_demo_coverage_report()
    
    # Calculate totals
    report.calculate_totals()
    
    # Create analyzer
    analyzer = CoverageAnalyzer()
    
    # Generate different report formats
    print("\n📝 Generating reports...")
    
    # Text report (display on console)
    text_report = analyzer.generate_report(report, format="text")
    print("\n" + text_report)
    
    # HTML report
    html_report = analyzer.generate_report(report, format="html", output_file="demo_coverage.html")
    print("\n✅ Generated HTML report: demo_coverage.html")
    
    # JSON report
    json_report = analyzer.generate_report(report, format="json", output_file="demo_coverage.json")
    print("✅ Generated JSON report: demo_coverage.json")
    
    # Markdown report
    md_report = analyzer.generate_report(report, format="markdown", output_file="demo_coverage.md")
    print("✅ Generated Markdown report: demo_coverage.md")
    
    # Analyze uncovered code
    print("\n🔍 Analyzing uncovered code...")
    insights = analyzer.analyze_uncovered_code(report)
    
    if insights:
        print("\n📊 UNCOVERED CODE INSIGHTS")
        print("-" * 40)
        
        for file_path, file_insights in insights.items():
            print(f"\n📄 {file_path}:")
            for insight in file_insights[:3]:  # Show first 3 insights per file
                priority_icon = "🔴" if insight['priority'] == 'high' else "🟡"
                print(f"   {priority_icon} {insight['type']}")
                print(f"      {insight['suggestion']}")
                if 'lines' in insight:
                    print(f"      Lines: {insight['lines']}")
    
    # Coverage trends (simulated)
    print("\n📈 COVERAGE TRENDS")
    print("-" * 40)
    print("   Last Week:  72.3% ↗️")
    print("   Yesterday:  74.8% ↗️")
    print(f"   Today:      {report.line_rate:.1f}% ", end="")
    if report.line_rate > 74.8:
        print("↗️ Improving!")
    else:
        print("↘️ Needs attention")
    
    # Priority recommendations
    print("\n⚡ PRIORITY RECOMMENDATIONS")
    print("-" * 40)
    
    critical_files = []
    for package in report.packages.values():
        for file_path, file_cov in package.files.items():
            if file_cov.get_coverage_level() == CoverageLevel.CRITICAL:
                critical_files.append((file_path, file_cov.line_rate))
    
    if critical_files:
        print("🔴 Critical files needing immediate attention:")
        for file_path, coverage in sorted(critical_files, key=lambda x: x[1]):
            print(f"   - {file_path}: {coverage:.1f}% coverage")
    
    low_coverage_funcs = []
    for package in report.packages.values():
        for file_path, file_cov in package.files.items():
            if file_cov.missed_functions:
                for func in file_cov.missed_functions[:2]:  # Top 2 per file
                    low_coverage_funcs.append((file_path, func))
    
    if low_coverage_funcs:
        print("\n🟡 Key untested functions:")
        for file_path, func in low_coverage_funcs[:5]:  # Show top 5
            print(f"   - {func} in {file_path}")
    
    print("\n" + "="*80)
    print("✅ Coverage analysis demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()