#!/usr/bin/env python3
"""
Security Gates Service
Provides SAST (Static Application Security Testing) and dependency scanning
for the Reviewer Agent to ensure code security
"""

import os
import sys
import json
import asyncio
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import tempfile
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityScanRequest(BaseModel):
    """Security scan request"""
    project_path: str = Field(description="Project path to scan")
    scan_type: str = Field(default="full", description="Scan type: full, sast, dependencies, secrets")
    language: Optional[str] = Field(default=None, description="Programming language")
    severity_threshold: str = Field(default="medium", description="Minimum severity: low, medium, high, critical")
    ignore_patterns: Optional[List[str]] = Field(default=None, description="Patterns to ignore")


class DependencyScanRequest(BaseModel):
    """Dependency vulnerability scan request"""
    project_path: str = Field(description="Project path")
    package_file: Optional[str] = Field(default=None, description="Package file (package.json, requirements.txt, etc)")
    check_licenses: bool = Field(default=True, description="Check license compliance")


class CodeScanRequest(BaseModel):
    """Static code analysis request"""
    file_path: str = Field(description="File or directory to scan")
    language: str = Field(description="Programming language")
    rules: Optional[List[str]] = Field(default=None, description="Specific rules to check")


class SecurityIssue(BaseModel):
    """Security issue found during scanning"""
    type: str = Field(description="Issue type: vulnerability, secret, code_smell, etc")
    severity: str = Field(description="Severity: low, medium, high, critical")
    file: Optional[str] = Field(default=None, description="File path")
    line: Optional[int] = Field(default=None, description="Line number")
    column: Optional[int] = Field(default=None, description="Column number")
    rule: str = Field(description="Security rule violated")
    message: str = Field(description="Issue description")
    recommendation: Optional[str] = Field(default=None, description="Fix recommendation")
    cwe: Optional[str] = Field(default=None, description="CWE ID")
    owasp: Optional[str] = Field(default=None, description="OWASP category")


class SecurityGatesService:
    """Security Gates Service for SAST and dependency scanning"""
    
    def __init__(self):
        self.app = FastAPI(title="Security Gates Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("security_gates")
        self.cache = get_cache()
        
        # Security rules database
        self.security_rules = self._load_security_rules()
        
        # Vulnerability patterns
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        
        # License compatibility matrix
        self.license_matrix = self._load_license_matrix()
        
        logger.info("✅ Security Gates Service initialized")
        
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
    
    def _load_security_rules(self) -> Dict[str, List[Dict]]:
        """Load security scanning rules"""
        return {
            "python": [
                {"id": "PY001", "pattern": r"eval\s*\(", "severity": "critical", "message": "Use of eval() is dangerous", "cwe": "CWE-95"},
                {"id": "PY002", "pattern": r"exec\s*\(", "severity": "critical", "message": "Use of exec() is dangerous", "cwe": "CWE-95"},
                {"id": "PY003", "pattern": r"pickle\.loads?\s*\(", "severity": "high", "message": "Pickle deserialization vulnerability", "cwe": "CWE-502"},
                {"id": "PY004", "pattern": r"os\.system\s*\(", "severity": "high", "message": "Command injection risk", "cwe": "CWE-78"},
                {"id": "PY005", "pattern": r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", "severity": "high", "message": "Shell injection risk", "cwe": "CWE-78"},
                {"id": "PY006", "pattern": r"sqlite3\.connect\s*\([^)]*check_same_thread\s*=\s*False", "severity": "medium", "message": "SQLite thread safety issue"},
                {"id": "PY007", "pattern": r"hashlib\.(md5|sha1)\s*\(", "severity": "medium", "message": "Weak cryptographic hash", "cwe": "CWE-327"},
                {"id": "PY008", "pattern": r"random\.\w+\s*\(", "severity": "low", "message": "Use secrets module for cryptographic randomness", "cwe": "CWE-330"},
            ],
            "javascript": [
                {"id": "JS001", "pattern": r"eval\s*\(", "severity": "critical", "message": "Use of eval() is dangerous", "cwe": "CWE-95"},
                {"id": "JS002", "pattern": r"innerHTML\s*=", "severity": "high", "message": "Potential XSS vulnerability", "cwe": "CWE-79"},
                {"id": "JS003", "pattern": r"document\.write\s*\(", "severity": "high", "message": "Potential XSS vulnerability", "cwe": "CWE-79"},
                {"id": "JS004", "pattern": r"new\s+Function\s*\(", "severity": "high", "message": "Dynamic code execution", "cwe": "CWE-95"},
                {"id": "JS005", "pattern": r"child_process\.\w+\s*\(", "severity": "high", "message": "Command execution risk", "cwe": "CWE-78"},
                {"id": "JS006", "pattern": r"fs\.\w+Sync\s*\(", "severity": "low", "message": "Synchronous file operation"},
                {"id": "JS007", "pattern": r"require\s*\([^)]*\+[^)]*\)", "severity": "medium", "message": "Dynamic require vulnerability"},
                {"id": "JS008", "pattern": r"crypto\.createHash\s*\(['\"]md5['\"]\)", "severity": "medium", "message": "Weak hash algorithm", "cwe": "CWE-327"},
            ],
            "go": [
                {"id": "GO001", "pattern": r"fmt\.Sprintf.*%s.*sql", "severity": "critical", "message": "SQL injection risk", "cwe": "CWE-89"},
                {"id": "GO002", "pattern": r"exec\.Command\s*\(", "severity": "high", "message": "Command injection risk", "cwe": "CWE-78"},
                {"id": "GO003", "pattern": r"md5\.New\s*\(", "severity": "medium", "message": "Weak hash algorithm", "cwe": "CWE-327"},
                {"id": "GO004", "pattern": r"math/rand", "severity": "low", "message": "Use crypto/rand for security", "cwe": "CWE-330"},
                {"id": "GO005", "pattern": r"tls\.Config.*InsecureSkipVerify.*true", "severity": "high", "message": "TLS verification disabled", "cwe": "CWE-295"},
            ],
            "java": [
                {"id": "JA001", "pattern": r"Runtime\.getRuntime\(\)\.exec", "severity": "critical", "message": "Command injection risk", "cwe": "CWE-78"},
                {"id": "JA002", "pattern": r"MessageDigest\.getInstance\s*\(['\"]MD5['\"]\)", "severity": "medium", "message": "Weak hash algorithm", "cwe": "CWE-327"},
                {"id": "JA003", "pattern": r"new\s+Random\s*\(", "severity": "low", "message": "Use SecureRandom for security", "cwe": "CWE-330"},
                {"id": "JA004", "pattern": r"Statement.*execute.*\+", "severity": "critical", "message": "SQL injection risk", "cwe": "CWE-89"},
            ]
        }
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[Dict]]:
        """Load patterns for detecting vulnerabilities"""
        return {
            "secrets": [
                {"pattern": r"(?i)(api[_\s\-]?key|apikey)\s*[:=]\s*['\"][^'\"]{20,}['\"]", "type": "api_key"},
                {"pattern": r"(?i)(secret|password|passwd|pwd)\s*[:=]\s*['\"][^'\"]+['\"]", "type": "password"},
                {"pattern": r"(?i)aws[_\s]?access[_\s]?key[_\s]?id\s*[:=]\s*['\"][A-Z0-9]{20}['\"]", "type": "aws_key"},
                {"pattern": r"(?i)aws[_\s]?secret[_\s]?access[_\s]?key\s*[:=]\s*['\"][A-Za-z0-9/+=]{40}['\"]", "type": "aws_secret"},
                {"pattern": r"github[_\s]token\s*[:=]\s*['\"]ghp_[a-zA-Z0-9]{36}['\"]", "type": "github_token"},
                {"pattern": r"(?i)private[_\s]?key\s*[:=]\s*['\"]-----BEGIN", "type": "private_key"},
                {"pattern": r"mongodb(\+srv)?://[^:]+:[^@]+@", "type": "database_url"},
                {"pattern": r"postgres(ql)?://[^:]+:[^@]+@", "type": "database_url"},
            ],
            "injections": [
                {"pattern": r"f['\"].*{.*}.*['\"].*sql", "type": "sql_injection", "severity": "critical"},
                {"pattern": r"os\.path\.join\([^,)]*request\.", "type": "path_traversal", "severity": "high"},
                {"pattern": r"open\([^)]*request\.", "type": "file_disclosure", "severity": "high"},
            ]
        }
    
    def _load_license_matrix(self) -> Dict[str, Dict]:
        """Load license compatibility matrix"""
        return {
            "MIT": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "type": "permissive"},
            "Apache-2.0": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "type": "permissive"},
            "BSD": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "type": "permissive"},
            "GPL-3.0": {"compatible": ["GPL-3.0", "AGPL-3.0"], "type": "copyleft"},
            "LGPL-3.0": {"compatible": ["LGPL-3.0", "GPL-3.0", "MIT", "BSD"], "type": "weak-copyleft"},
            "AGPL-3.0": {"compatible": ["AGPL-3.0", "GPL-3.0"], "type": "strong-copyleft"},
            "ISC": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "type": "permissive"},
        }
    
    async def scan_code(self, file_path: str, language: str, 
                       rules: Optional[List[str]] = None) -> List[SecurityIssue]:
        """Perform static code analysis"""
        issues = []
        
        try:
            # Get language-specific rules
            lang_rules = self.security_rules.get(language.lower(), [])
            
            if rules:
                # Filter to specific rules if provided
                lang_rules = [r for r in lang_rules if r["id"] in rules]
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check each rule
            for rule in lang_rules:
                pattern = re.compile(rule["pattern"])
                
                for line_num, line in enumerate(lines, 1):
                    matches = pattern.finditer(line)
                    for match in matches:
                        issue = SecurityIssue(
                            type="vulnerability",
                            severity=rule["severity"],
                            file=file_path,
                            line=line_num,
                            column=match.start() + 1,
                            rule=rule["id"],
                            message=rule["message"],
                            cwe=rule.get("cwe"),
                            owasp=self._get_owasp_category(rule.get("cwe"))
                        )
                        issues.append(issue)
            
            # Check for secrets
            for secret_pattern in self.vulnerability_patterns["secrets"]:
                pattern = re.compile(secret_pattern["pattern"])
                for line_num, line in enumerate(lines, 1):
                    if pattern.search(line):
                        issue = SecurityIssue(
                            type="secret",
                            severity="critical",
                            file=file_path,
                            line=line_num,
                            column=1,
                            rule="SECRET001",
                            message=f"Potential {secret_pattern['type']} exposed",
                            recommendation="Remove secret and rotate credentials",
                            cwe="CWE-798"
                        )
                        issues.append(issue)
            
            logger.info(f"Found {len(issues)} security issues in {file_path}")
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
        
        return issues
    
    async def scan_dependencies(self, project_path: str, 
                               package_file: Optional[str] = None,
                               check_licenses: bool = True) -> Dict[str, Any]:
        """Scan project dependencies for vulnerabilities"""
        vulnerabilities = []
        license_issues = []
        
        try:
            # Auto-detect package file if not specified
            if not package_file:
                package_files = self._detect_package_files(project_path)
            else:
                package_files = [os.path.join(project_path, package_file)]
            
            for pkg_file in package_files:
                if not os.path.exists(pkg_file):
                    continue
                
                file_name = os.path.basename(pkg_file)
                
                # Python dependencies
                if file_name == "requirements.txt":
                    vulns = await self._scan_python_deps(pkg_file)
                    vulnerabilities.extend(vulns)
                
                # Node.js dependencies
                elif file_name == "package.json":
                    vulns = await self._scan_node_deps(pkg_file)
                    vulnerabilities.extend(vulns)
                
                # Go dependencies
                elif file_name == "go.mod":
                    vulns = await self._scan_go_deps(pkg_file)
                    vulnerabilities.extend(vulns)
                
                # Java dependencies
                elif file_name == "pom.xml":
                    vulns = await self._scan_java_deps(pkg_file)
                    vulnerabilities.extend(vulns)
                
                # Check licenses if requested
                if check_licenses:
                    license_issues.extend(
                        await self._check_license_compliance(pkg_file)
                    )
            
            return {
                "vulnerabilities": vulnerabilities,
                "license_issues": license_issues,
                "total_issues": len(vulnerabilities) + len(license_issues),
                "scanned_files": package_files
            }
            
        except Exception as e:
            logger.error(f"Error scanning dependencies: {e}")
            raise
    
    def _detect_package_files(self, project_path: str) -> List[str]:
        """Detect package files in project"""
        package_files = []
        patterns = [
            "requirements.txt", "requirements*.txt", "Pipfile", "pyproject.toml",
            "package.json", "yarn.lock", "package-lock.json",
            "go.mod", "go.sum",
            "pom.xml", "build.gradle", "build.gradle.kts",
            "Gemfile", "Gemfile.lock",
            "Cargo.toml", "Cargo.lock"
        ]
        
        for pattern in patterns:
            for file_path in Path(project_path).rglob(pattern):
                package_files.append(str(file_path))
        
        return package_files
    
    async def _scan_python_deps(self, requirements_file: str) -> List[Dict]:
        """Scan Python dependencies"""
        vulnerabilities = []
        
        try:
            # Use safety or pip-audit for vulnerability scanning
            result = subprocess.run(
                ["pip-audit", "--requirement", requirements_file, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                audit_results = json.loads(result.stdout)
                for vuln in audit_results:
                    vulnerabilities.append({
                        "package": vuln.get("name"),
                        "version": vuln.get("version"),
                        "vulnerability": vuln.get("vulnerability_id"),
                        "severity": self._normalize_severity(vuln.get("severity", "unknown")),
                        "description": vuln.get("description"),
                        "fixed_version": vuln.get("fixed_version")
                    })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to basic checks
            logger.warning("pip-audit not available, using basic checks")
            
            # Read requirements and check against known vulnerable versions
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Check against known vulnerabilities (simplified)
                        if 'django==2.2.0' in line.lower():
                            vulnerabilities.append({
                                "package": "django",
                                "version": "2.2.0",
                                "vulnerability": "CVE-2019-14232",
                                "severity": "high",
                                "description": "Django SQL injection vulnerability",
                                "fixed_version": ">=2.2.4"
                            })
        
        return vulnerabilities
    
    async def _scan_node_deps(self, package_json: str) -> List[Dict]:
        """Scan Node.js dependencies"""
        vulnerabilities = []
        
        try:
            # Use npm audit or yarn audit
            project_dir = os.path.dirname(package_json)
            
            # Check if using npm or yarn
            if os.path.exists(os.path.join(project_dir, "package-lock.json")):
                result = subprocess.run(
                    ["npm", "audit", "--json"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            elif os.path.exists(os.path.join(project_dir, "yarn.lock")):
                result = subprocess.run(
                    ["yarn", "audit", "--json"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                return vulnerabilities
            
            if result.stdout:
                # Parse audit results
                for line in result.stdout.split('\n'):
                    if line.strip():
                        try:
                            audit_data = json.loads(line)
                            if audit_data.get("type") == "auditAdvisory":
                                advisory = audit_data.get("data", {}).get("advisory", {})
                                vulnerabilities.append({
                                    "package": advisory.get("module_name"),
                                    "severity": advisory.get("severity"),
                                    "vulnerability": advisory.get("cves", [""])[0],
                                    "description": advisory.get("title"),
                                    "recommendation": advisory.get("recommendation")
                                })
                        except json.JSONDecodeError:
                            continue
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("npm/yarn audit not available")
        
        return vulnerabilities
    
    async def _scan_go_deps(self, go_mod: str) -> List[Dict]:
        """Scan Go dependencies"""
        vulnerabilities = []
        
        try:
            project_dir = os.path.dirname(go_mod)
            
            # Use govulncheck
            result = subprocess.run(
                ["govulncheck", "./..."],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                # Parse govulncheck output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Vulnerability' in line:
                        # Parse vulnerability info
                        vulnerabilities.append({
                            "package": "go-dependency",
                            "vulnerability": line,
                            "severity": "medium"
                        })
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("govulncheck not available")
        
        return vulnerabilities
    
    async def _scan_java_deps(self, pom_xml: str) -> List[Dict]:
        """Scan Java dependencies"""
        vulnerabilities = []
        
        try:
            project_dir = os.path.dirname(pom_xml)
            
            # Use OWASP dependency check
            result = subprocess.run(
                ["mvn", "org.owasp:dependency-check-maven:check"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse results from generated report
            report_path = os.path.join(project_dir, "target/dependency-check-report.json")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    for dep in report.get("dependencies", []):
                        for vuln in dep.get("vulnerabilities", []):
                            vulnerabilities.append({
                                "package": dep.get("fileName"),
                                "vulnerability": vuln.get("name"),
                                "severity": vuln.get("severity"),
                                "description": vuln.get("description")
                            })
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Maven dependency check not available")
        
        return vulnerabilities
    
    async def _check_license_compliance(self, package_file: str) -> List[Dict]:
        """Check license compliance"""
        issues = []
        
        # This is a simplified implementation
        # In production, use tools like license-checker, pip-licenses, etc.
        
        return issues
    
    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity levels"""
        severity_map = {
            "critical": "critical",
            "high": "high",
            "moderate": "medium",
            "medium": "medium",
            "low": "low",
            "info": "low"
        }
        return severity_map.get(severity.lower(), "medium")
    
    def _get_owasp_category(self, cwe: Optional[str]) -> Optional[str]:
        """Map CWE to OWASP Top 10 category"""
        if not cwe:
            return None
        
        owasp_map = {
            "CWE-89": "A03:2021 - Injection",
            "CWE-78": "A03:2021 - Injection",
            "CWE-79": "A03:2021 - Injection",
            "CWE-95": "A03:2021 - Injection",
            "CWE-502": "A08:2021 - Software and Data Integrity Failures",
            "CWE-327": "A02:2021 - Cryptographic Failures",
            "CWE-330": "A02:2021 - Cryptographic Failures",
            "CWE-798": "A07:2021 - Identification and Authentication Failures",
            "CWE-295": "A07:2021 - Identification and Authentication Failures"
        }
        
        return owasp_map.get(cwe)
    
    async def full_security_scan(self, project_path: str, 
                                 severity_threshold: str = "medium") -> Dict[str, Any]:
        """Perform full security scan of project"""
        all_issues = []
        scan_results = {
            "project_path": project_path,
            "scan_time": datetime.utcnow().isoformat(),
            "severity_threshold": severity_threshold
        }
        
        # SAST - Static Application Security Testing
        sast_issues = []
        for root, dirs, files in os.walk(project_path):
            # Skip common directories to ignore
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv', '.venv']]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Determine language
                language = self._detect_language(file_path)
                if language:
                    issues = await self.scan_code(file_path, language)
                    sast_issues.extend(issues)
        
        # Dependency scanning
        dep_scan = await self.scan_dependencies(project_path)
        
        # Secret scanning
        secret_issues = await self._scan_for_secrets(project_path)
        
        # Compile results
        scan_results["sast"] = {
            "issues": [issue.dict() for issue in sast_issues],
            "count": len(sast_issues)
        }
        
        scan_results["dependencies"] = dep_scan
        
        scan_results["secrets"] = {
            "issues": secret_issues,
            "count": len(secret_issues)
        }
        
        # Calculate overall risk score
        scan_results["risk_score"] = self._calculate_risk_score(
            sast_issues, dep_scan["vulnerabilities"], secret_issues
        )
        
        # Filter by severity threshold
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        threshold_level = severity_levels.get(severity_threshold, 2)
        
        filtered_issues = [
            issue for issue in sast_issues
            if severity_levels.get(issue.severity, 1) >= threshold_level
        ]
        
        scan_results["filtered_issues"] = {
            "count": len(filtered_issues),
            "issues": [issue.dict() for issue in filtered_issues]
        }
        
        # Cache results
        cache_key = f"security_scan:{hashlib.md5(project_path.encode()).hexdigest()}"
        self.cache.set(cache_key, scan_results, ttl=3600)
        
        logger.info(f"Security scan complete: {len(filtered_issues)} issues above threshold")
        
        return scan_results
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.go': 'go',
            '.java': 'java',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext)
    
    async def _scan_for_secrets(self, project_path: str) -> List[Dict]:
        """Scan for exposed secrets"""
        secrets = []
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv']]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip binary files
                if self._is_binary_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                    
                    for pattern_info in self.vulnerability_patterns["secrets"]:
                        pattern = re.compile(pattern_info["pattern"])
                        
                        for line_num, line in enumerate(lines, 1):
                            if pattern.search(line):
                                secrets.append({
                                    "type": pattern_info["type"],
                                    "file": file_path,
                                    "line": line_num,
                                    "severity": "critical",
                                    "message": f"Potential {pattern_info['type']} exposed"
                                })
                
                except Exception as e:
                    logger.debug(f"Could not scan {file_path}: {e}")
        
        return secrets
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    return True
                return False
        except:
            return True
    
    def _calculate_risk_score(self, sast_issues: List, dep_vulns: List, secrets: List) -> int:
        """Calculate overall security risk score (0-100)"""
        score = 0
        
        # Weight different issue types
        severity_weights = {
            "critical": 25,
            "high": 15,
            "medium": 5,
            "low": 1
        }
        
        # SAST issues
        for issue in sast_issues:
            score += severity_weights.get(issue.severity, 1)
        
        # Dependency vulnerabilities
        for vuln in dep_vulns:
            score += severity_weights.get(vuln.get("severity", "medium"), 5)
        
        # Secrets (always critical)
        score += len(secrets) * 25
        
        # Cap at 100
        return min(score, 100)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "security_gates",
                "status": "healthy",
                "rules_loaded": sum(len(rules) for rules in self.security_rules.values()),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/scan")
        async def security_scan(request: SecurityScanRequest):
            """Perform security scan"""
            if request.scan_type == "full":
                return await self.full_security_scan(
                    request.project_path,
                    request.severity_threshold
                )
            elif request.scan_type == "sast":
                # SAST only
                issues = []
                for root, dirs, files in os.walk(request.project_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        language = self._detect_language(file_path)
                        if language:
                            file_issues = await self.scan_code(file_path, language)
                            issues.extend(file_issues)
                return {"issues": [i.dict() for i in issues], "count": len(issues)}
            elif request.scan_type == "dependencies":
                return await self.scan_dependencies(request.project_path)
            elif request.scan_type == "secrets":
                secrets = await self._scan_for_secrets(request.project_path)
                return {"secrets": secrets, "count": len(secrets)}
            else:
                raise HTTPException(status_code=400, detail="Invalid scan type")
        
        @self.app.post("/scan/code")
        async def scan_code_endpoint(request: CodeScanRequest):
            """Scan specific code file"""
            issues = await self.scan_code(
                request.file_path,
                request.language,
                request.rules
            )
            return {
                "file": request.file_path,
                "language": request.language,
                "issues": [issue.dict() for issue in issues],
                "count": len(issues)
            }
        
        @self.app.post("/scan/dependencies")
        async def scan_dependencies_endpoint(request: DependencyScanRequest):
            """Scan dependencies"""
            return await self.scan_dependencies(
                request.project_path,
                request.package_file,
                request.check_licenses
            )
        
        @self.app.get("/rules/{language}")
        async def get_rules(language: str):
            """Get security rules for language"""
            rules = self.security_rules.get(language.lower(), [])
            return {
                "language": language,
                "rules": rules,
                "count": len(rules)
            }
        
        @self.app.get("/report/{scan_id}")
        async def get_report(scan_id: str):
            """Get cached scan report"""
            cache_key = f"security_scan:{scan_id}"
            report = self.cache.get(cache_key)
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return report
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Security Gates Service starting up...")
        logger.info(f"Loaded {sum(len(r) for r in self.security_rules.values())} security rules")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Security Gates Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = SecurityGatesService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("SECURITY_GATES_PORT", 8020))
    logger.info(f"Starting Security Gates Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()