#!/usr/bin/env python3
"""
Workflow Definition of Done (DoD) Service
Defines and validates completion criteria for each workflow step
Ensures quality gates are met before progression
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStep(str, Enum):
    """Workflow step identifiers"""
    ROUTING = "routing"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    CODING = "coding"
    TESTING = "testing"
    REVIEWING = "reviewing"
    INTEGRATION = "integration"


class CriteriaType(str, Enum):
    """Types of completion criteria"""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class CriteriaStatus(str, Enum):
    """Status of criteria validation"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DoDCriteria:
    """Definition of Done criteria"""
    id: str
    name: str
    description: str
    type: CriteriaType
    validator: str  # Name of validation function
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class StepDoD:
    """DoD for a workflow step"""
    step: WorkflowStep
    criteria: List[DoDCriteria]
    min_required_pass_rate: float = 1.0  # 100% of required criteria must pass
    min_recommended_pass_rate: float = 0.8  # 80% of recommended should pass


class DoDValidationRequest(BaseModel):
    """Request to validate DoD for a step"""
    workflow_id: str = Field(description="Workflow instance ID")
    step: WorkflowStep = Field(description="Step to validate")
    context: Dict[str, Any] = Field(description="Context for validation")
    skip_optional: bool = Field(default=True, description="Skip optional criteria")


class DoDValidationResult(BaseModel):
    """Result of DoD validation"""
    step: WorkflowStep
    passed: bool
    required_passed: int
    required_total: int
    recommended_passed: int
    recommended_total: int
    optional_passed: int
    optional_total: int
    criteria_results: List[Dict[str, Any]]
    timestamp: str
    duration_ms: int


class WorkflowDoDService:
    """Service for managing workflow Definition of Done"""
    
    def __init__(self):
        self.app = FastAPI(title="Workflow DoD Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("workflow_dod")
        self.cache = get_cache()
        
        # Define DoD for each workflow step
        self.step_dods = self._initialize_dods()
        
        # Validation functions registry
        self.validators = self._initialize_validators()
        
        logger.info("✅ Workflow DoD Service initialized")
        
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
    
    def _initialize_dods(self) -> Dict[WorkflowStep, StepDoD]:
        """Initialize Definition of Done for each workflow step"""
        return {
            WorkflowStep.ROUTING: StepDoD(
                step=WorkflowStep.ROUTING,
                criteria=[
                    DoDCriteria(
                        id="route_001",
                        name="Request Classified",
                        description="Request type has been identified",
                        type=CriteriaType.REQUIRED,
                        validator="validate_request_classified"
                    ),
                    DoDCriteria(
                        id="route_002",
                        name="Workflow Selected",
                        description="Appropriate workflow has been selected",
                        type=CriteriaType.REQUIRED,
                        validator="validate_workflow_selected"
                    ),
                    DoDCriteria(
                        id="route_003",
                        name="Context Initialized",
                        description="Workflow context has been initialized",
                        type=CriteriaType.REQUIRED,
                        validator="validate_context_initialized"
                    ),
                    DoDCriteria(
                        id="route_004",
                        name="Priority Set",
                        description="Task priority has been determined",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_priority_set"
                    )
                ]
            ),
            
            WorkflowStep.PLANNING: StepDoD(
                step=WorkflowStep.PLANNING,
                criteria=[
                    DoDCriteria(
                        id="plan_001",
                        name="Task Breakdown",
                        description="Task broken down into subtasks",
                        type=CriteriaType.REQUIRED,
                        validator="validate_task_breakdown"
                    ),
                    DoDCriteria(
                        id="plan_002",
                        name="Dependencies Identified",
                        description="Task dependencies have been identified",
                        type=CriteriaType.REQUIRED,
                        validator="validate_dependencies_identified"
                    ),
                    DoDCriteria(
                        id="plan_003",
                        name="Resource Requirements",
                        description="Required resources have been identified",
                        type=CriteriaType.REQUIRED,
                        validator="validate_resources_identified"
                    ),
                    DoDCriteria(
                        id="plan_004",
                        name="Time Estimation",
                        description="Time estimates provided for tasks",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_time_estimates"
                    ),
                    DoDCriteria(
                        id="plan_005",
                        name="Risk Assessment",
                        description="Potential risks have been identified",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_risk_assessment"
                    )
                ]
            ),
            
            WorkflowStep.RETRIEVAL: StepDoD(
                step=WorkflowStep.RETRIEVAL,
                criteria=[
                    DoDCriteria(
                        id="retrieve_001",
                        name="Context Retrieved",
                        description="Relevant context has been retrieved",
                        type=CriteriaType.REQUIRED,
                        validator="validate_context_retrieved"
                    ),
                    DoDCriteria(
                        id="retrieve_002",
                        name="Code Examples Found",
                        description="Relevant code examples have been found",
                        type=CriteriaType.REQUIRED,
                        validator="validate_code_examples"
                    ),
                    DoDCriteria(
                        id="retrieve_003",
                        name="Documentation Retrieved",
                        description="Relevant documentation has been retrieved",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_documentation_retrieved"
                    ),
                    DoDCriteria(
                        id="retrieve_004",
                        name="Similarity Threshold",
                        description="Retrieved content meets similarity threshold",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_similarity_threshold",
                        metadata={"threshold": 0.7}
                    )
                ]
            ),
            
            WorkflowStep.CODING: StepDoD(
                step=WorkflowStep.CODING,
                criteria=[
                    DoDCriteria(
                        id="code_001",
                        name="Code Generated",
                        description="Code has been generated/modified",
                        type=CriteriaType.REQUIRED,
                        validator="validate_code_generated"
                    ),
                    DoDCriteria(
                        id="code_002",
                        name="Syntax Valid",
                        description="Code has valid syntax",
                        type=CriteriaType.REQUIRED,
                        validator="validate_syntax"
                    ),
                    DoDCriteria(
                        id="code_003",
                        name="Imports Resolved",
                        description="All imports/dependencies are resolved",
                        type=CriteriaType.REQUIRED,
                        validator="validate_imports"
                    ),
                    DoDCriteria(
                        id="code_004",
                        name="Type Checking",
                        description="Code passes type checking",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_type_checking"
                    ),
                    DoDCriteria(
                        id="code_005",
                        name="Comments Added",
                        description="Code includes appropriate comments",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_comments"
                    ),
                    DoDCriteria(
                        id="code_006",
                        name="Naming Conventions",
                        description="Code follows naming conventions",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_naming_conventions"
                    )
                ]
            ),
            
            WorkflowStep.TESTING: StepDoD(
                step=WorkflowStep.TESTING,
                criteria=[
                    DoDCriteria(
                        id="test_001",
                        name="Tests Written",
                        description="Tests have been written",
                        type=CriteriaType.REQUIRED,
                        validator="validate_tests_written"
                    ),
                    DoDCriteria(
                        id="test_002",
                        name="Tests Pass",
                        description="All tests pass successfully",
                        type=CriteriaType.REQUIRED,
                        validator="validate_tests_pass"
                    ),
                    DoDCriteria(
                        id="test_003",
                        name="Code Coverage",
                        description="Code coverage meets threshold",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_code_coverage",
                        metadata={"threshold": 80}
                    ),
                    DoDCriteria(
                        id="test_004",
                        name="Edge Cases",
                        description="Edge cases are tested",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_edge_cases"
                    ),
                    DoDCriteria(
                        id="test_005",
                        name="Performance Tests",
                        description="Performance tests pass",
                        type=CriteriaType.OPTIONAL,
                        validator="validate_performance_tests"
                    )
                ]
            ),
            
            WorkflowStep.REVIEWING: StepDoD(
                step=WorkflowStep.REVIEWING,
                criteria=[
                    DoDCriteria(
                        id="review_001",
                        name="Security Scan",
                        description="Security scan completed with no critical issues",
                        type=CriteriaType.REQUIRED,
                        validator="validate_security_scan"
                    ),
                    DoDCriteria(
                        id="review_002",
                        name="Linting Passed",
                        description="Code passes linting rules",
                        type=CriteriaType.REQUIRED,
                        validator="validate_linting"
                    ),
                    DoDCriteria(
                        id="review_003",
                        name="No Hardcoded Secrets",
                        description="No hardcoded secrets detected",
                        type=CriteriaType.REQUIRED,
                        validator="validate_no_secrets"
                    ),
                    DoDCriteria(
                        id="review_004",
                        name="Dependency Check",
                        description="Dependencies are secure and up-to-date",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_dependencies"
                    ),
                    DoDCriteria(
                        id="review_005",
                        name="Code Complexity",
                        description="Code complexity is within acceptable limits",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_complexity",
                        metadata={"max_complexity": 10}
                    ),
                    DoDCriteria(
                        id="review_006",
                        name="Documentation Complete",
                        description="Documentation is complete and accurate",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_documentation"
                    )
                ]
            ),
            
            WorkflowStep.INTEGRATION: StepDoD(
                step=WorkflowStep.INTEGRATION,
                criteria=[
                    DoDCriteria(
                        id="integrate_001",
                        name="Branch Created",
                        description="Feature branch has been created",
                        type=CriteriaType.REQUIRED,
                        validator="validate_branch_created"
                    ),
                    DoDCriteria(
                        id="integrate_002",
                        name="Changes Committed",
                        description="Changes have been committed",
                        type=CriteriaType.REQUIRED,
                        validator="validate_changes_committed"
                    ),
                    DoDCriteria(
                        id="integrate_003",
                        name="PR Created",
                        description="Pull request has been created",
                        type=CriteriaType.REQUIRED,
                        validator="validate_pr_created"
                    ),
                    DoDCriteria(
                        id="integrate_004",
                        name="CI/CD Passing",
                        description="CI/CD pipeline is passing",
                        type=CriteriaType.REQUIRED,
                        validator="validate_cicd_passing"
                    ),
                    DoDCriteria(
                        id="integrate_005",
                        name="No Merge Conflicts",
                        description="No merge conflicts exist",
                        type=CriteriaType.REQUIRED,
                        validator="validate_no_conflicts"
                    ),
                    DoDCriteria(
                        id="integrate_006",
                        name="Review Approved",
                        description="PR has been approved by reviewers",
                        type=CriteriaType.RECOMMENDED,
                        validator="validate_review_approved"
                    )
                ]
            )
        }
    
    def _initialize_validators(self) -> Dict[str, callable]:
        """Initialize validation functions"""
        return {
            # Routing validators
            "validate_request_classified": self._validate_request_classified,
            "validate_workflow_selected": self._validate_workflow_selected,
            "validate_context_initialized": self._validate_context_initialized,
            "validate_priority_set": self._validate_priority_set,
            
            # Planning validators
            "validate_task_breakdown": self._validate_task_breakdown,
            "validate_dependencies_identified": self._validate_dependencies_identified,
            "validate_resources_identified": self._validate_resources_identified,
            "validate_time_estimates": self._validate_time_estimates,
            "validate_risk_assessment": self._validate_risk_assessment,
            
            # Retrieval validators
            "validate_context_retrieved": self._validate_context_retrieved,
            "validate_code_examples": self._validate_code_examples,
            "validate_documentation_retrieved": self._validate_documentation_retrieved,
            "validate_similarity_threshold": self._validate_similarity_threshold,
            
            # Coding validators
            "validate_code_generated": self._validate_code_generated,
            "validate_syntax": self._validate_syntax,
            "validate_imports": self._validate_imports,
            "validate_type_checking": self._validate_type_checking,
            "validate_comments": self._validate_comments,
            "validate_naming_conventions": self._validate_naming_conventions,
            
            # Testing validators
            "validate_tests_written": self._validate_tests_written,
            "validate_tests_pass": self._validate_tests_pass,
            "validate_code_coverage": self._validate_code_coverage,
            "validate_edge_cases": self._validate_edge_cases,
            "validate_performance_tests": self._validate_performance_tests,
            
            # Review validators
            "validate_security_scan": self._validate_security_scan,
            "validate_linting": self._validate_linting,
            "validate_no_secrets": self._validate_no_secrets,
            "validate_dependencies": self._validate_dependencies,
            "validate_complexity": self._validate_complexity,
            "validate_documentation": self._validate_documentation,
            
            # Integration validators
            "validate_branch_created": self._validate_branch_created,
            "validate_changes_committed": self._validate_changes_committed,
            "validate_pr_created": self._validate_pr_created,
            "validate_cicd_passing": self._validate_cicd_passing,
            "validate_no_conflicts": self._validate_no_conflicts,
            "validate_review_approved": self._validate_review_approved
        }
    
    async def validate_step_dod(self, request: DoDValidationRequest) -> DoDValidationResult:
        """Validate DoD for a workflow step"""
        start_time = datetime.utcnow()
        
        # Get DoD for the step
        step_dod = self.step_dods.get(request.step)
        if not step_dod:
            raise HTTPException(status_code=404, detail=f"DoD not defined for step: {request.step}")
        
        # Validate each criteria
        criteria_results = []
        required_passed = 0
        required_total = 0
        recommended_passed = 0
        recommended_total = 0
        optional_passed = 0
        optional_total = 0
        
        for criteria in step_dod.criteria:
            # Skip optional if requested
            if request.skip_optional and criteria.type == CriteriaType.OPTIONAL:
                continue
            
            # Get validator function
            validator = self.validators.get(criteria.validator)
            if not validator:
                logger.warning(f"Validator not found: {criteria.validator}")
                result = {
                    "criteria_id": criteria.id,
                    "name": criteria.name,
                    "type": criteria.type.value,
                    "status": CriteriaStatus.FAILED.value,
                    "message": "Validator not implemented"
                }
            else:
                # Run validation
                try:
                    validation_result = await validator(request.context, criteria.metadata)
                    result = {
                        "criteria_id": criteria.id,
                        "name": criteria.name,
                        "type": criteria.type.value,
                        "status": CriteriaStatus.PASSED.value if validation_result else CriteriaStatus.FAILED.value,
                        "message": f"Validation {'passed' if validation_result else 'failed'}"
                    }
                    
                    # Update counters
                    if criteria.type == CriteriaType.REQUIRED:
                        required_total += 1
                        if validation_result:
                            required_passed += 1
                    elif criteria.type == CriteriaType.RECOMMENDED:
                        recommended_total += 1
                        if validation_result:
                            recommended_passed += 1
                    elif criteria.type == CriteriaType.OPTIONAL:
                        optional_total += 1
                        if validation_result:
                            optional_passed += 1
                            
                except Exception as e:
                    logger.error(f"Validation error for {criteria.id}: {e}")
                    result = {
                        "criteria_id": criteria.id,
                        "name": criteria.name,
                        "type": criteria.type.value,
                        "status": CriteriaStatus.FAILED.value,
                        "message": str(e)
                    }
                    
                    if criteria.type == CriteriaType.REQUIRED:
                        required_total += 1
            
            criteria_results.append(result)
        
        # Determine if step passes DoD
        required_pass_rate = required_passed / required_total if required_total > 0 else 0
        recommended_pass_rate = recommended_passed / recommended_total if recommended_total > 0 else 0
        
        passed = (
            required_pass_rate >= step_dod.min_required_pass_rate and
            recommended_pass_rate >= step_dod.min_recommended_pass_rate
        )
        
        # Calculate duration
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Store result in cache
        cache_key = f"dod_result:{request.workflow_id}:{request.step.value}"
        result = DoDValidationResult(
            step=request.step,
            passed=passed,
            required_passed=required_passed,
            required_total=required_total,
            recommended_passed=recommended_passed,
            recommended_total=recommended_total,
            optional_passed=optional_passed,
            optional_total=optional_total,
            criteria_results=criteria_results,
            timestamp=datetime.utcnow().isoformat(),
            duration_ms=duration_ms
        )
        
        self.cache.set(cache_key, result.dict(), ttl=3600)
        
        logger.info(f"DoD validation for {request.step}: {'PASSED' if passed else 'FAILED'}")
        
        return result
    
    # Validation functions (simplified implementations)
    
    async def _validate_request_classified(self, context: Dict, metadata: Dict) -> bool:
        """Validate that request has been classified"""
        return "request_type" in context and context["request_type"] is not None
    
    async def _validate_workflow_selected(self, context: Dict, metadata: Dict) -> bool:
        """Validate that workflow has been selected"""
        return "workflow_id" in context and context["workflow_id"] is not None
    
    async def _validate_context_initialized(self, context: Dict, metadata: Dict) -> bool:
        """Validate that context has been initialized"""
        return "initialized" in context and context["initialized"] is True
    
    async def _validate_priority_set(self, context: Dict, metadata: Dict) -> bool:
        """Validate that priority has been set"""
        return "priority" in context
    
    async def _validate_task_breakdown(self, context: Dict, metadata: Dict) -> bool:
        """Validate task breakdown"""
        return "subtasks" in context and len(context.get("subtasks", [])) > 0
    
    async def _validate_dependencies_identified(self, context: Dict, metadata: Dict) -> bool:
        """Validate dependencies identified"""
        return "dependencies" in context
    
    async def _validate_resources_identified(self, context: Dict, metadata: Dict) -> bool:
        """Validate resources identified"""
        return "resources" in context
    
    async def _validate_time_estimates(self, context: Dict, metadata: Dict) -> bool:
        """Validate time estimates"""
        return "time_estimate" in context
    
    async def _validate_risk_assessment(self, context: Dict, metadata: Dict) -> bool:
        """Validate risk assessment"""
        return "risks" in context
    
    async def _validate_context_retrieved(self, context: Dict, metadata: Dict) -> bool:
        """Validate context retrieved"""
        return "retrieved_context" in context and len(context.get("retrieved_context", [])) > 0
    
    async def _validate_code_examples(self, context: Dict, metadata: Dict) -> bool:
        """Validate code examples found"""
        return "code_examples" in context and len(context.get("code_examples", [])) > 0
    
    async def _validate_documentation_retrieved(self, context: Dict, metadata: Dict) -> bool:
        """Validate documentation retrieved"""
        return "documentation" in context
    
    async def _validate_similarity_threshold(self, context: Dict, metadata: Dict) -> bool:
        """Validate similarity threshold"""
        threshold = metadata.get("threshold", 0.7)
        return context.get("max_similarity", 0) >= threshold
    
    async def _validate_code_generated(self, context: Dict, metadata: Dict) -> bool:
        """Validate code generated"""
        return "generated_code" in context and context["generated_code"] is not None
    
    async def _validate_syntax(self, context: Dict, metadata: Dict) -> bool:
        """Validate syntax"""
        return context.get("syntax_valid", False)
    
    async def _validate_imports(self, context: Dict, metadata: Dict) -> bool:
        """Validate imports resolved"""
        return context.get("imports_resolved", False)
    
    async def _validate_type_checking(self, context: Dict, metadata: Dict) -> bool:
        """Validate type checking"""
        return context.get("type_check_passed", False)
    
    async def _validate_comments(self, context: Dict, metadata: Dict) -> bool:
        """Validate comments added"""
        return context.get("has_comments", False)
    
    async def _validate_naming_conventions(self, context: Dict, metadata: Dict) -> bool:
        """Validate naming conventions"""
        return context.get("naming_convention_valid", False)
    
    async def _validate_tests_written(self, context: Dict, metadata: Dict) -> bool:
        """Validate tests written"""
        return context.get("tests_written", False)
    
    async def _validate_tests_pass(self, context: Dict, metadata: Dict) -> bool:
        """Validate tests pass"""
        return context.get("tests_passed", False)
    
    async def _validate_code_coverage(self, context: Dict, metadata: Dict) -> bool:
        """Validate code coverage"""
        threshold = metadata.get("threshold", 80)
        return context.get("code_coverage", 0) >= threshold
    
    async def _validate_edge_cases(self, context: Dict, metadata: Dict) -> bool:
        """Validate edge cases tested"""
        return context.get("edge_cases_tested", False)
    
    async def _validate_performance_tests(self, context: Dict, metadata: Dict) -> bool:
        """Validate performance tests"""
        return context.get("performance_tests_passed", False)
    
    async def _validate_security_scan(self, context: Dict, metadata: Dict) -> bool:
        """Validate security scan"""
        return context.get("security_issues_critical", 0) == 0
    
    async def _validate_linting(self, context: Dict, metadata: Dict) -> bool:
        """Validate linting passed"""
        return context.get("linting_passed", False)
    
    async def _validate_no_secrets(self, context: Dict, metadata: Dict) -> bool:
        """Validate no secrets"""
        return context.get("secrets_found", 0) == 0
    
    async def _validate_dependencies(self, context: Dict, metadata: Dict) -> bool:
        """Validate dependencies"""
        return context.get("vulnerable_dependencies", 0) == 0
    
    async def _validate_complexity(self, context: Dict, metadata: Dict) -> bool:
        """Validate code complexity"""
        max_complexity = metadata.get("max_complexity", 10)
        return context.get("max_complexity", 0) <= max_complexity
    
    async def _validate_documentation(self, context: Dict, metadata: Dict) -> bool:
        """Validate documentation"""
        return context.get("documentation_complete", False)
    
    async def _validate_branch_created(self, context: Dict, metadata: Dict) -> bool:
        """Validate branch created"""
        return context.get("branch_created", False)
    
    async def _validate_changes_committed(self, context: Dict, metadata: Dict) -> bool:
        """Validate changes committed"""
        return context.get("changes_committed", False)
    
    async def _validate_pr_created(self, context: Dict, metadata: Dict) -> bool:
        """Validate PR created"""
        return context.get("pr_created", False)
    
    async def _validate_cicd_passing(self, context: Dict, metadata: Dict) -> bool:
        """Validate CI/CD passing"""
        return context.get("cicd_status", "") == "passing"
    
    async def _validate_no_conflicts(self, context: Dict, metadata: Dict) -> bool:
        """Validate no merge conflicts"""
        return context.get("merge_conflicts", 0) == 0
    
    async def _validate_review_approved(self, context: Dict, metadata: Dict) -> bool:
        """Validate review approved"""
        return context.get("review_approved", False)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "workflow_dod",
                "status": "healthy",
                "steps_defined": len(self.step_dods),
                "validators_registered": len(self.validators),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/validate")
        async def validate_dod(request: DoDValidationRequest):
            """Validate DoD for a workflow step"""
            return await self.validate_step_dod(request)
        
        @self.app.get("/dod/{step}")
        async def get_step_dod(step: WorkflowStep):
            """Get DoD criteria for a step"""
            step_dod = self.step_dods.get(step)
            if not step_dod:
                raise HTTPException(status_code=404, detail=f"DoD not defined for step: {step}")
            
            return {
                "step": step.value,
                "criteria": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": c.description,
                        "type": c.type.value,
                        "dependencies": c.dependencies
                    }
                    for c in step_dod.criteria
                ],
                "min_required_pass_rate": step_dod.min_required_pass_rate,
                "min_recommended_pass_rate": step_dod.min_recommended_pass_rate
            }
        
        @self.app.get("/dod")
        async def get_all_dods():
            """Get all DoD definitions"""
            return {
                step.value: {
                    "criteria_count": len(dod.criteria),
                    "required_count": sum(1 for c in dod.criteria if c.type == CriteriaType.REQUIRED),
                    "recommended_count": sum(1 for c in dod.criteria if c.type == CriteriaType.RECOMMENDED),
                    "optional_count": sum(1 for c in dod.criteria if c.type == CriteriaType.OPTIONAL)
                }
                for step, dod in self.step_dods.items()
            }
        
        @self.app.get("/validation/{workflow_id}/{step}")
        async def get_validation_result(workflow_id: str, step: WorkflowStep):
            """Get cached validation result"""
            cache_key = f"dod_result:{workflow_id}:{step.value}"
            result = self.cache.get(cache_key)
            
            if not result:
                raise HTTPException(status_code=404, detail="Validation result not found")
            
            return result
        
        @self.app.post("/batch-validate")
        async def batch_validate(workflow_id: str, context: Dict[str, Any]):
            """Validate all steps for a workflow"""
            results = {}
            
            for step in WorkflowStep:
                request = DoDValidationRequest(
                    workflow_id=workflow_id,
                    step=step,
                    context=context
                )
                
                try:
                    result = await self.validate_step_dod(request)
                    results[step.value] = result.dict()
                except Exception as e:
                    logger.error(f"Failed to validate {step}: {e}")
                    results[step.value] = {"error": str(e)}
            
            return results
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Workflow DoD Service starting up...")
        logger.info(f"Loaded DoD for {len(self.step_dods)} workflow steps")
        logger.info(f"Registered {len(self.validators)} validators")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Workflow DoD Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = WorkflowDoDService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("WORKFLOW_DOD_PORT", 8023))
    logger.info(f"Starting Workflow DoD Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()