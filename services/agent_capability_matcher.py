#!/usr/bin/env python3
"""
Agent Capability Matcher Service
Ensures proper agent selection based on task requirements and agent capabilities
Prevents mismatched assignments like security_expert getting web development tasks
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Task categories"""
    WEB_DEVELOPMENT = "web_development"
    SECURITY = "security"
    DATA_ANALYSIS = "data_analysis"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    DATABASE = "database"
    API_DEVELOPMENT = "api_development"
    INFRASTRUCTURE = "infrastructure"
    MACHINE_LEARNING = "machine_learning"
    GENERAL = "general"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_id: str
    agent_name: str
    categories: List[TaskCategory]
    keywords: List[str]
    confidence_score: float = 1.0


class AgentCapabilityMatcher:
    """Matches tasks to appropriate agents based on capabilities"""
    
    def __init__(self):
        self.agent_capabilities = self._initialize_capabilities()
        self.task_patterns = self._initialize_task_patterns()
        
    def _initialize_capabilities(self) -> Dict[str, AgentCapability]:
        """Initialize agent capabilities"""
        return {
            "security_expert": AgentCapability(
                agent_id="security_expert",
                agent_name="Security Expert",
                categories=[TaskCategory.SECURITY, TaskCategory.CODE_REVIEW],
                keywords=["security", "vulnerability", "audit", "encryption", "authentication", 
                         "authorization", "penetration", "exploit", "firewall", "ssl", "tls",
                         "owasp", "cve", "patch", "compliance", "gdpr", "pci"],
                confidence_score=0.95
            ),
            "web_developer": AgentCapability(
                agent_id="web_developer",
                agent_name="Web Developer",
                categories=[TaskCategory.WEB_DEVELOPMENT, TaskCategory.API_DEVELOPMENT],
                keywords=["website", "web", "html", "css", "javascript", "frontend", "backend",
                         "react", "vue", "angular", "node", "express", "django", "flask",
                         "api", "rest", "graphql", "ui", "ux", "responsive", "browser"],
                confidence_score=0.95
            ),
            "coder-agent": AgentCapability(
                agent_id="coder-agent",
                agent_name="Coder Agent",
                categories=[TaskCategory.WEB_DEVELOPMENT, TaskCategory.API_DEVELOPMENT, 
                           TaskCategory.GENERAL],
                keywords=["code", "program", "function", "class", "method", "algorithm",
                         "implementation", "develop", "create", "build", "write", "hello world"],
                confidence_score=0.90
            ),
            "data_analyst": AgentCapability(
                agent_id="data_analyst",
                agent_name="Data Analyst",
                categories=[TaskCategory.DATA_ANALYSIS, TaskCategory.MACHINE_LEARNING],
                keywords=["data", "analysis", "statistics", "ml", "machine learning", "ai",
                         "visualization", "report", "metrics", "kpi", "dashboard", "etl",
                         "pandas", "numpy", "scikit", "tensorflow", "pytorch"],
                confidence_score=0.92
            ),
            "tester-agent": AgentCapability(
                agent_id="tester-agent",
                agent_name="Tester Agent",
                categories=[TaskCategory.TESTING],
                keywords=["test", "qa", "quality", "bug", "debug", "unit test", "integration",
                         "e2e", "selenium", "pytest", "jest", "coverage", "regression"],
                confidence_score=0.93
            ),
            "devops_engineer": AgentCapability(
                agent_id="devops_engineer",
                agent_name="DevOps Engineer",
                categories=[TaskCategory.DEPLOYMENT, TaskCategory.INFRASTRUCTURE],
                keywords=["deploy", "kubernetes", "docker", "ci/cd", "jenkins", "aws", "azure",
                         "gcp", "terraform", "ansible", "monitoring", "logging", "scaling"],
                confidence_score=0.94
            ),
            "database_admin": AgentCapability(
                agent_id="database_admin",
                agent_name="Database Administrator",
                categories=[TaskCategory.DATABASE],
                keywords=["database", "sql", "postgres", "mysql", "mongodb", "redis", "query",
                         "index", "migration", "backup", "restore", "optimization", "schema"],
                confidence_score=0.91
            ),
            "reviewer-agent": AgentCapability(
                agent_id="reviewer-agent",
                agent_name="Reviewer Agent",
                categories=[TaskCategory.CODE_REVIEW, TaskCategory.DOCUMENTATION],
                keywords=["review", "audit", "check", "validate", "approve", "pr", "pull request",
                         "code quality", "standards", "best practices", "refactor"],
                confidence_score=0.88
            )
        }
    
    def _initialize_task_patterns(self) -> Dict[TaskCategory, List[re.Pattern]]:
        """Initialize task categorization patterns"""
        return {
            TaskCategory.WEB_DEVELOPMENT: [
                re.compile(r'\b(create|build|develop)\s+.*\b(website|web\s*app|web\s*page|html|frontend)\b', re.I),
                re.compile(r'\bhello\s*world\s*(website|page|site)\b', re.I),
                re.compile(r'\b(react|vue|angular|javascript|css|html)\b', re.I)
            ],
            TaskCategory.SECURITY: [
                re.compile(r'\b(security|vulnerability|penetration|exploit|audit)\b', re.I),
                re.compile(r'\b(encrypt|decrypt|authentication|authorization)\b', re.I),
                re.compile(r'\b(owasp|cve|patch|compliance)\b', re.I)
            ],
            TaskCategory.DATA_ANALYSIS: [
                re.compile(r'\b(analyze|analysis|data|statistics|report|metrics)\b', re.I),
                re.compile(r'\b(machine\s*learning|ml|ai|neural|model)\b', re.I)
            ],
            TaskCategory.TESTING: [
                re.compile(r'\b(test|testing|qa|quality|bug|debug)\b', re.I),
                re.compile(r'\b(unit\s*test|integration\s*test|e2e)\b', re.I)
            ],
            TaskCategory.DATABASE: [
                re.compile(r'\b(database|sql|query|table|schema|migration)\b', re.I),
                re.compile(r'\b(postgres|mysql|mongodb|redis)\b', re.I)
            ],
            TaskCategory.DEPLOYMENT: [
                re.compile(r'\b(deploy|deployment|release|rollout)\b', re.I),
                re.compile(r'\b(kubernetes|k8s|docker|container)\b', re.I)
            ]
        }
    
    def categorize_task(self, task_description: str) -> Tuple[TaskCategory, float]:
        """
        Categorize a task based on its description
        Returns: (category, confidence_score)
        """
        task_lower = task_description.lower()
        category_scores = {}
        
        # Check patterns
        for category, patterns in self.task_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern.search(task_description):
                    score += 1.0
            if score > 0:
                category_scores[category] = score / len(patterns)
        
        # Check keywords
        for agent_id, capability in self.agent_capabilities.items():
            for keyword in capability.keywords:
                if keyword.lower() in task_lower:
                    for category in capability.categories:
                        if category not in category_scores:
                            category_scores[category] = 0
                        category_scores[category] += 0.1
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return best_category, min(category_scores[best_category], 1.0)
        
        return TaskCategory.GENERAL, 0.5
    
    def find_best_agent(self, task_description: str) -> Tuple[str, float, str]:
        """
        Find the best agent for a task
        Returns: (agent_id, confidence_score, reason)
        """
        category, task_confidence = self.categorize_task(task_description)
        
        logger.info(f"Task categorized as {category.value} with confidence {task_confidence:.2f}")
        
        # Find agents with matching categories
        matching_agents = []
        for agent_id, capability in self.agent_capabilities.items():
            if category in capability.categories:
                # Calculate match score based on keywords
                keyword_matches = sum(
                    1 for keyword in capability.keywords 
                    if keyword.lower() in task_description.lower()
                )
                match_score = (
                    task_confidence * 0.4 + 
                    capability.confidence_score * 0.3 + 
                    min(keyword_matches / 5, 1.0) * 0.3
                )
                matching_agents.append((agent_id, match_score, capability))
        
        if not matching_agents:
            # Fallback to general coder agent
            if "coder-agent" in self.agent_capabilities:
                return ("coder-agent", 0.5, 
                       f"No specialist found for {category.value}, using general coder")
            return ("router-agent", 0.3, 
                   f"No suitable agent found for {category.value}")
        
        # Sort by match score
        matching_agents.sort(key=lambda x: x[1], reverse=True)
        best_agent = matching_agents[0]
        
        reason = f"Best match for {category.value} task with {best_agent[1]:.2f} confidence"
        
        # Log warning if low confidence
        if best_agent[1] < 0.6:
            logger.warning(f"Low confidence match: {best_agent[0]} for task: {task_description[:100]}")
        
        return best_agent[0], best_agent[1], reason
    
    def validate_assignment(self, agent_id: str, task_description: str) -> Tuple[bool, str]:
        """
        Validate if an agent assignment is appropriate
        Returns: (is_valid, reason)
        """
        best_agent, confidence, _ = self.find_best_agent(task_description)
        
        if agent_id not in self.agent_capabilities:
            return False, f"Unknown agent: {agent_id}"
        
        if best_agent == agent_id:
            return True, "Correct agent assigned"
        
        # Check if agent has any relevant capabilities
        task_category, _ = self.categorize_task(task_description)
        agent_capability = self.agent_capabilities[agent_id]
        
        if task_category in agent_capability.categories:
            return True, f"Agent has capability for {task_category.value}"
        
        # Check for severe mismatch
        if agent_id == "security_expert" and task_category == TaskCategory.WEB_DEVELOPMENT:
            return False, "SEVERE MISMATCH: Security expert assigned to web development task"
        
        if confidence > 0.7:
            return False, f"Wrong agent: {agent_id} assigned but {best_agent} is better suited"
        
        return True, "Assignment acceptable with low confidence"


# Test the matcher
if __name__ == "__main__":
    matcher = AgentCapabilityMatcher()
    
    # Test cases
    test_tasks = [
        "create a hello world website in this directory /Users/jankootstra/Identity_blockchain/agent-backbone-architecture/docs",
        "perform security audit on the authentication system",
        "analyze user engagement data for the last quarter",
        "deploy the application to kubernetes cluster",
        "write unit tests for the payment module"
    ]
    
    print("Testing Agent Capability Matcher\n" + "="*50)
    
    for task in test_tasks:
        print(f"\nTask: {task[:80]}...")
        
        # Find best agent
        agent_id, confidence, reason = matcher.find_best_agent(task)
        print(f"  Best Agent: {agent_id}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Reason: {reason}")
        
        # Test validation with wrong agent
        is_valid, validation_reason = matcher.validate_assignment("security_expert", task)
        print(f"  Security Expert Valid? {is_valid} - {validation_reason}")