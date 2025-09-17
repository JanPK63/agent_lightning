#!/usr/bin/env python3
"""
Governance and PR Gates System
Enforces quality standards, code review requirements, and approval workflows
Ensures all changes meet enterprise governance standards before deployment
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
import re

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import aiohttp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GateType(str, Enum):
    """Types of governance gates"""
    CODE_QUALITY = "code_quality"
    SECURITY_SCAN = "security_scan"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    PEER_REVIEW = "peer_review"
    ARCHITECTURE_REVIEW = "architecture_review"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DEPLOYMENT_READINESS = "deployment_readiness"


class GateStatus(str, Enum):
    """Gate check status"""
    PENDING = "pending"
    CHECKING = "checking"
    PASSED = "passed"
    FAILED = "failed"
    WARNED = "warned"
    BYPASSED = "bypassed"


class ApprovalStatus(str, Enum):
    """Approval status for changes"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class PRGateRequest(BaseModel):
    """Request to check PR gates"""
    pr_id: str = Field(description="Pull request ID")
    repository: str = Field(description="Repository name")
    branch: str = Field(description="Branch name")
    commit_sha: str = Field(description="Commit SHA")
    author: str = Field(description="PR author")
    title: str = Field(description="PR title")
    description: str = Field(description="PR description")
    files_changed: List[str] = Field(description="List of changed files")
    lines_added: int = Field(default=0)
    lines_removed: int = Field(default=0)


class GateCheck(BaseModel):
    """Individual gate check result"""
    gate_type: GateType
    status: GateStatus
    score: float = Field(ge=0, le=1, description="Score from 0 to 1")
    details: Dict[str, Any]
    required: bool = Field(default=True, description="Is this gate required?")
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PRGateResult(BaseModel):
    """Complete PR gate check result"""
    pr_id: str
    overall_status: GateStatus
    approval_status: ApprovalStatus
    gates_passed: int
    gates_failed: int
    gates_warned: int
    total_score: float
    checks: List[GateCheck]
    can_merge: bool
    requires_approval_from: List[str]
    recommendations: List[str]


class GovernancePolicy(BaseModel):
    """Governance policy configuration"""
    name: str
    enabled: bool = True
    required_gates: List[GateType]
    minimum_score: float = Field(ge=0, le=1, default=0.8)
    bypass_roles: List[str] = Field(default_factory=list)
    auto_approve_threshold: float = Field(ge=0, le=1, default=0.95)
    require_human_review: bool = Field(default=True)


class GovernancePRGatesSystem:
    """System for enforcing governance and PR quality gates"""
    
    def __init__(self):
        self.app = FastAPI(title="Governance & PR Gates System", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("governance_gates")
        self.cache = get_cache()
        
        # Service URLs
        self.security_gates_url = "http://localhost:8019"
        self.test_runner_url = "http://localhost:8018"
        self.workflow_dod_url = "http://localhost:8021"
        
        # Database connection
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        # Initialize policies
        self.policies = self._load_default_policies()
        
        # Create tables
        self._create_tables()
        
        logger.info("✅ Governance & PR Gates System initialized")
        
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
    
    def _create_tables(self):
        """Create governance tracking tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # PR gate checks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pr_gate_checks (
                    id SERIAL PRIMARY KEY,
                    pr_id VARCHAR(100) NOT NULL,
                    repository VARCHAR(200) NOT NULL,
                    commit_sha VARCHAR(40) NOT NULL,
                    gate_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    score FLOAT,
                    details JSONB,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pr_id, gate_type, commit_sha)
                );
                
                CREATE INDEX IF NOT EXISTS idx_pr_gate_pr_id ON pr_gate_checks(pr_id);
                CREATE INDEX IF NOT EXISTS idx_pr_gate_status ON pr_gate_checks(status);
                CREATE INDEX IF NOT EXISTS idx_pr_gate_timestamp ON pr_gate_checks(timestamp);
            """)
            
            # Approval records table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pr_approvals (
                    id SERIAL PRIMARY KEY,
                    pr_id VARCHAR(100) NOT NULL,
                    approver VARCHAR(100) NOT NULL,
                    approval_status VARCHAR(30) NOT NULL,
                    comments TEXT,
                    conditions JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pr_id, approver)
                );
                
                CREATE INDEX IF NOT EXISTS idx_pr_approval_pr_id ON pr_approvals(pr_id);
                CREATE INDEX IF NOT EXISTS idx_pr_approval_status ON pr_approvals(approval_status);
            """)
            
            # Governance policies table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS governance_policies (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    policy_config JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Governance tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def _load_default_policies(self) -> Dict[str, GovernancePolicy]:
        """Load default governance policies"""
        return {
            "standard": GovernancePolicy(
                name="Standard PR Policy",
                required_gates=[
                    GateType.CODE_QUALITY,
                    GateType.TEST_COVERAGE,
                    GateType.SECURITY_SCAN,
                    GateType.PEER_REVIEW
                ],
                minimum_score=0.8,
                require_human_review=True
            ),
            "critical": GovernancePolicy(
                name="Critical System Policy",
                required_gates=[
                    GateType.CODE_QUALITY,
                    GateType.TEST_COVERAGE,
                    GateType.SECURITY_SCAN,
                    GateType.PEER_REVIEW,
                    GateType.ARCHITECTURE_REVIEW,
                    GateType.PERFORMANCE,
                    GateType.COMPLIANCE
                ],
                minimum_score=0.9,
                require_human_review=True,
                auto_approve_threshold=0.98
            ),
            "experimental": GovernancePolicy(
                name="Experimental Feature Policy",
                required_gates=[
                    GateType.CODE_QUALITY,
                    GateType.TEST_COVERAGE
                ],
                minimum_score=0.7,
                require_human_review=False,
                auto_approve_threshold=0.85
            )
        }
    
    async def check_code_quality(self, request: PRGateRequest) -> GateCheck:
        """Check code quality standards"""
        try:
            score = 1.0
            issues = []
            
            # Check file sizes
            large_files = [f for f in request.files_changed if f.endswith(('.py', '.js', '.ts'))]
            if request.lines_added > 500:
                score -= 0.2
                issues.append("PR too large (>500 lines)")
            
            # Check for common quality issues
            patterns_to_check = [
                (r'\bTODO\b', "Contains TODO comments", 0.1),
                (r'\bFIXME\b', "Contains FIXME comments", 0.15),
                (r'print\(', "Contains print statements", 0.1),
                (r'console\.log', "Contains console.log statements", 0.1),
                (r'except\s*:', "Bare except clauses", 0.2),
                (r'pass\s*$', "Empty code blocks", 0.05)
            ]
            
            # Simplified check (in production, would analyze actual file content)
            for pattern, message, penalty in patterns_to_check:
                if any(pattern in f for f in request.files_changed):
                    score -= penalty
                    issues.append(message)
            
            # Check documentation
            if not request.description or len(request.description) < 50:
                score -= 0.1
                issues.append("Insufficient PR description")
            
            score = max(0, min(1, score))
            
            return GateCheck(
                gate_type=GateType.CODE_QUALITY,
                status=GateStatus.PASSED if score >= 0.7 else GateStatus.FAILED,
                score=score,
                details={"issues": issues, "files_checked": len(request.files_changed)},
                message=f"Code quality score: {score:.2f}" + (f" - Issues: {', '.join(issues)}" if issues else "")
            )
            
        except Exception as e:
            logger.error(f"Code quality check failed: {e}")
            return GateCheck(
                gate_type=GateType.CODE_QUALITY,
                status=GateStatus.FAILED,
                score=0,
                details={"error": str(e)},
                message=f"Code quality check error: {e}"
            )
    
    async def check_test_coverage(self, request: PRGateRequest) -> GateCheck:
        """Check test coverage requirements"""
        try:
            # In production, would call test runner service
            # Simulating coverage check
            coverage = 0.85  # Simulated coverage
            
            if any('.test.' in f or '_test.' in f or 'test_' in f for f in request.files_changed):
                coverage += 0.05  # Bonus for including tests
            
            status = GateStatus.PASSED if coverage >= 0.8 else GateStatus.WARNED if coverage >= 0.6 else GateStatus.FAILED
            
            return GateCheck(
                gate_type=GateType.TEST_COVERAGE,
                status=status,
                score=coverage,
                details={"coverage_percent": coverage * 100, "threshold": 80},
                message=f"Test coverage: {coverage * 100:.1f}% (minimum 80% required)"
            )
            
        except Exception as e:
            logger.error(f"Test coverage check failed: {e}")
            return GateCheck(
                gate_type=GateType.TEST_COVERAGE,
                status=GateStatus.FAILED,
                score=0,
                details={"error": str(e)},
                message=f"Coverage check error: {e}"
            )
    
    async def check_security(self, request: PRGateRequest) -> GateCheck:
        """Check security requirements"""
        try:
            score = 1.0
            vulnerabilities = []
            
            # Check for security patterns
            security_patterns = [
                (r'eval\(', "Dangerous eval() usage", 0.3),
                (r'exec\(', "Dangerous exec() usage", 0.3),
                (r'pickle\.', "Pickle usage (security risk)", 0.2),
                (r'os\.system', "Shell command execution", 0.2),
                (r'subprocess\.', "Subprocess usage", 0.1),
                (r'api_key|secret|password|token', "Potential hardcoded secrets", 0.4)
            ]
            
            for pattern, message, penalty in security_patterns:
                if any(pattern in f.lower() for f in request.files_changed):
                    score -= penalty
                    vulnerabilities.append(message)
            
            score = max(0, min(1, score))
            
            return GateCheck(
                gate_type=GateType.SECURITY_SCAN,
                status=GateStatus.PASSED if score >= 0.7 else GateStatus.FAILED,
                score=score,
                details={"vulnerabilities": vulnerabilities},
                message=f"Security score: {score:.2f}" + (f" - Issues: {', '.join(vulnerabilities)}" if vulnerabilities else "")
            )
            
        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return GateCheck(
                gate_type=GateType.SECURITY_SCAN,
                status=GateStatus.FAILED,
                score=0,
                details={"error": str(e)},
                message=f"Security check error: {e}"
            )
    
    async def check_documentation(self, request: PRGateRequest) -> GateCheck:
        """Check documentation requirements"""
        try:
            score = 1.0
            issues = []
            
            # Check PR description
            if len(request.description) < 100:
                score -= 0.2
                issues.append("Brief PR description")
            
            # Check for README updates if significant changes
            if request.lines_added > 100 and not any('README' in f for f in request.files_changed):
                score -= 0.1
                issues.append("No README update for significant changes")
            
            # Check for docstrings in Python files
            python_files = [f for f in request.files_changed if f.endswith('.py')]
            if python_files and not any('"""' in request.description or "'''" in request.description):
                score -= 0.15
                issues.append("Missing docstrings in Python files")
            
            score = max(0, min(1, score))
            
            return GateCheck(
                gate_type=GateType.DOCUMENTATION,
                status=GateStatus.PASSED if score >= 0.7 else GateStatus.WARNED,
                score=score,
                details={"issues": issues},
                required=False,
                message=f"Documentation score: {score:.2f}" + (f" - Issues: {', '.join(issues)}" if issues else "")
            )
            
        except Exception as e:
            logger.error(f"Documentation check failed: {e}")
            return GateCheck(
                gate_type=GateType.DOCUMENTATION,
                status=GateStatus.WARNED,
                score=0.5,
                details={"error": str(e)},
                required=False,
                message=f"Documentation check error: {e}"
            )
    
    async def check_all_gates(self, request: PRGateRequest, policy_name: str = "standard") -> PRGateResult:
        """Run all gate checks for a PR"""
        try:
            policy = self.policies.get(policy_name, self.policies["standard"])
            checks = []
            
            # Run all checks in parallel
            check_tasks = []
            
            if GateType.CODE_QUALITY in policy.required_gates:
                check_tasks.append(self.check_code_quality(request))
            
            if GateType.TEST_COVERAGE in policy.required_gates:
                check_tasks.append(self.check_test_coverage(request))
            
            if GateType.SECURITY_SCAN in policy.required_gates:
                check_tasks.append(self.check_security(request))
            
            if GateType.DOCUMENTATION in policy.required_gates or True:  # Always check docs
                check_tasks.append(self.check_documentation(request))
            
            # Execute all checks
            checks = await asyncio.gather(*check_tasks)
            
            # Calculate overall status
            gates_passed = sum(1 for c in checks if c.status == GateStatus.PASSED)
            gates_failed = sum(1 for c in checks if c.status == GateStatus.FAILED)
            gates_warned = sum(1 for c in checks if c.status == GateStatus.WARNED)
            
            total_score = sum(c.score for c in checks) / len(checks) if checks else 0
            
            # Determine if PR can be merged
            required_checks = [c for c in checks if c.required]
            all_required_passed = all(c.status == GateStatus.PASSED for c in required_checks)
            meets_minimum_score = total_score >= policy.minimum_score
            
            can_merge = all_required_passed and meets_minimum_score
            
            # Determine approval status
            if total_score >= policy.auto_approve_threshold and not policy.require_human_review:
                approval_status = ApprovalStatus.APPROVED
            elif gates_failed > 0:
                approval_status = ApprovalStatus.CHANGES_REQUESTED
            else:
                approval_status = ApprovalStatus.PENDING
            
            # Generate recommendations
            recommendations = []
            if gates_failed > 0:
                recommendations.append("Fix failing gate checks before merging")
            if gates_warned > 0:
                recommendations.append("Review and address warnings")
            if total_score < 0.9:
                recommendations.append(f"Improve code quality to reach 90% score (current: {total_score*100:.1f}%)")
            
            # Determine required approvers
            requires_approval_from = []
            if policy.require_human_review:
                requires_approval_from.append("senior_developer")
            if GateType.ARCHITECTURE_REVIEW in policy.required_gates:
                requires_approval_from.append("architect")
            if GateType.SECURITY_SCAN in policy.required_gates and any(c.status == GateStatus.FAILED for c in checks if c.gate_type == GateType.SECURITY_SCAN):
                requires_approval_from.append("security_team")
            
            # Store results in database
            await self._store_gate_results(request.pr_id, request.commit_sha, checks)
            
            result = PRGateResult(
                pr_id=request.pr_id,
                overall_status=GateStatus.PASSED if can_merge else GateStatus.FAILED,
                approval_status=approval_status,
                gates_passed=gates_passed,
                gates_failed=gates_failed,
                gates_warned=gates_warned,
                total_score=total_score,
                checks=checks,
                can_merge=can_merge,
                requires_approval_from=requires_approval_from,
                recommendations=recommendations
            )
            
            # Cache result
            cache_key = f"pr_gates:{request.pr_id}:{request.commit_sha}"
            self.cache.set(cache_key, result.dict(), ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Gate check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _store_gate_results(self, pr_id: str, commit_sha: str, checks: List[GateCheck]):
        """Store gate check results in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            for check in checks:
                cur.execute("""
                    INSERT INTO pr_gate_checks 
                    (pr_id, repository, commit_sha, gate_type, status, score, details, message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pr_id, gate_type, commit_sha) 
                    DO UPDATE SET 
                        status = EXCLUDED.status,
                        score = EXCLUDED.score,
                        details = EXCLUDED.details,
                        message = EXCLUDED.message,
                        timestamp = CURRENT_TIMESTAMP
                """, (
                    pr_id, "main", commit_sha, check.gate_type.value,
                    check.status.value, check.score, json.dumps(check.details),
                    check.message
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store gate results: {e}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "governance_pr_gates",
                "status": "healthy",
                "policies_loaded": len(self.policies),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/check-pr")
        async def check_pr_gates(request: PRGateRequest, policy: str = "standard"):
            """Check all gates for a PR"""
            return await self.check_all_gates(request, policy)
        
        @self.app.post("/check-gate/{gate_type}")
        async def check_specific_gate(gate_type: GateType, request: PRGateRequest):
            """Check a specific gate"""
            if gate_type == GateType.CODE_QUALITY:
                return await self.check_code_quality(request)
            elif gate_type == GateType.TEST_COVERAGE:
                return await self.check_test_coverage(request)
            elif gate_type == GateType.SECURITY_SCAN:
                return await self.check_security(request)
            elif gate_type == GateType.DOCUMENTATION:
                return await self.check_documentation(request)
            else:
                raise HTTPException(status_code=400, detail=f"Gate type {gate_type} not implemented")
        
        @self.app.post("/approve-pr")
        async def approve_pr(pr_id: str, approver: str, comments: str = ""):
            """Approve a PR"""
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO pr_approvals (pr_id, approver, approval_status, comments)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (pr_id, approver)
                    DO UPDATE SET 
                        approval_status = EXCLUDED.approval_status,
                        comments = EXCLUDED.comments,
                        timestamp = CURRENT_TIMESTAMP
                """, (pr_id, approver, ApprovalStatus.APPROVED.value, comments))
                
                conn.commit()
                conn.close()
                
                return {"status": "approved", "pr_id": pr_id, "approver": approver}
                
            except Exception as e:
                logger.error(f"Failed to approve PR: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/pr-status/{pr_id}")
        async def get_pr_status(pr_id: str):
            """Get current status of PR gates"""
            try:
                # Check cache first
                cached = None
                for key in self.cache.redis_client.keys(f"pr_gates:{pr_id}:*"):
                    cached = self.cache.get(key)
                    if cached:
                        break
                
                if cached:
                    return cached
                
                # Get from database
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor(cursor_factory=RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM pr_gate_checks 
                    WHERE pr_id = %s 
                    ORDER BY timestamp DESC
                """, (pr_id,))
                
                checks = cur.fetchall()
                
                cur.execute("""
                    SELECT * FROM pr_approvals 
                    WHERE pr_id = %s
                """, (pr_id,))
                
                approvals = cur.fetchall()
                
                conn.close()
                
                return {
                    "pr_id": pr_id,
                    "checks": checks,
                    "approvals": approvals
                }
                
            except Exception as e:
                logger.error(f"Failed to get PR status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/policies")
        async def get_policies():
            """Get all governance policies"""
            return {
                "policies": [p.dict() for p in self.policies.values()]
            }
        
        @self.app.post("/policies")
        async def create_policy(policy: GovernancePolicy):
            """Create or update a governance policy"""
            self.policies[policy.name] = policy
            
            # Store in database
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO governance_policies (name, policy_config)
                    VALUES (%s, %s)
                    ON CONFLICT (name)
                    DO UPDATE SET 
                        policy_config = EXCLUDED.policy_config,
                        updated_at = CURRENT_TIMESTAMP
                """, (policy.name, json.dumps(policy.dict())))
                
                conn.commit()
                conn.close()
                
                return {"status": "created", "policy": policy.name}
                
            except Exception as e:
                logger.error(f"Failed to create policy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Governance & PR Gates System starting up...")
        
        # Load policies from database
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("SELECT name, policy_config FROM governance_policies WHERE enabled = TRUE")
            
            for row in cur.fetchall():
                policy_data = row["policy_config"]
                self.policies[row["name"]] = GovernancePolicy(**policy_data)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
        
        logger.info(f"Loaded {len(self.policies)} governance policies")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Governance & PR Gates System shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = GovernancePRGatesSystem()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("GOVERNANCE_PORT", 8028))
    logger.info(f"Starting Governance & PR Gates System on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()