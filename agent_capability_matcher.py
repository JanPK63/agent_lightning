#!/usr/bin/env python3
"""
Agent Capability Matcher Module

This module provides capability-based agent selection to prevent wrong agent assignments.
It matches tasks to agents based on their capabilities, expertise, and task requirements.

Key Features:
- Keyword-based capability matching
- Confidence scoring for agent-task compatibility
- Validation of agent assignments
- Extensible agent capability definitions

Author: Agent Lightning Team
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Represents an agent's capabilities and expertise areas"""
    agent_id: str
    name: str
    capabilities: List[str]
    expertise_keywords: List[str]
    weak_keywords: List[str]  # Keywords this agent should avoid
    base_confidence: float = 0.5
    # Optional health/probe URL for runtime availability checks (e.g., http://host:port/health)
    health_url: Optional[str] = None


class AgentCapabilityMatcher:
    """
    Matches tasks to agents based on their capabilities and expertise.

    This class prevents wrong agent assignments by validating task-agent compatibility
    before assignment occurs.
    """

    def __init__(self):
        """Initialize the capability matcher with predefined agents"""
        self.agents = self._initialize_agents()
        logger.info(f"Initialized capability matcher with {len(self.agents)} agents")

    def _initialize_agents(self) -> Dict[str, AgentCapability]:
        """Initialize the predefined set of agents and their capabilities"""
        # health_url values are defaults pointing to localhost mock agent endpoints.
        # If you run real agent services elsewhere, update these URLs accordingly.
        return {
            "web_developer": AgentCapability(
                agent_id="web_developer",
                name="Web Developer",
                capabilities=["web development", "frontend", "backend", "html", "css", "javascript"],
                expertise_keywords=["website", "web", "html", "css", "javascript", "frontend", "backend", "api"],
                weak_keywords=["security", "audit", "penetration", "encryption"],
                base_confidence=0.8,
                health_url="http://localhost:9001/health"
            ),
            "security_expert": AgentCapability(
                agent_id="security_expert",
                name="Security Expert",
                capabilities=["security", "penetration testing", "vulnerability assessment", "encryption"],
                expertise_keywords=["security", "audit", "penetration", "vulnerability", "encryption", "authentication"],
                weak_keywords=["website", "web", "html", "css", "frontend"],
                base_confidence=0.9,
                health_url="http://localhost:9003/health"
            ),
            "data_analyst": AgentCapability(
                agent_id="data_analyst",
                name="Data Analyst",
                capabilities=["data analysis", "statistics", "visualization", "machine learning"],
                expertise_keywords=["data", "analysis", "statistics", "visualization", "ml", "machine learning", "dataset"],
                weak_keywords=["security", "web", "deployment"],
                base_confidence=0.7,
                health_url="http://localhost:9002/health"
            ),
            "devops_engineer": AgentCapability(
                agent_id="devops_engineer",
                name="DevOps Engineer",
                capabilities=["deployment", "infrastructure", "ci/cd", "monitoring"],
                expertise_keywords=["deploy", "infrastructure", "ci", "cd", "monitoring", "docker", "kubernetes"],
                weak_keywords=["frontend", "ui", "design"],
                base_confidence=0.75,
                health_url="http://localhost:9004/health"
            ),
            "qa_tester": AgentCapability(
                agent_id="qa_tester",
                name="QA Tester",
                capabilities=["testing", "quality assurance", "automation"],
                expertise_keywords=["test", "testing", "qa", "quality", "automation", "bug", "regression"],
                weak_keywords=["production", "deployment", "security"],
                base_confidence=0.6,
                health_url="http://localhost:9005/health"
            ),
            "general_assistant": AgentCapability(
                agent_id="general_assistant",
                name="General Assistant",
                capabilities=["general tasks", "coordination", "documentation"],
                expertise_keywords=["general", "help", "assist", "coordinate", "document"],
                weak_keywords=[],  # Can handle most things
                base_confidence=0.4,
                health_url="http://localhost:9006/health"
            )
        }

    def find_best_agent(self, task_description: str) -> Tuple[str, float, str]:
        """
        Find the best agent for a given task based on capability matching.

        Args:
            task_description: Description of the task to be performed

        Returns:
            Tuple of (agent_id, confidence_score, reason)
        """
        task_lower = task_description.lower()
        best_agent = "general_assistant"
        best_confidence = 0.0
        best_reason = "Default fallback agent"

        for agent_id, agent in self.agents.items():
            confidence, reason = self._calculate_match_confidence(task_lower, agent)

            if confidence > best_confidence:
                best_confidence = confidence
                best_agent = agent_id
                best_reason = reason

        # Ensure minimum confidence
        if best_confidence < 0.3:
            best_agent = "general_assistant"
            best_confidence = 0.3
            best_reason = "Task complexity too low for specialized agents"

        logger.info(f"Best agent for '{task_description[:50]}...': {best_agent} (confidence: {best_confidence:.2f})")

        return best_agent, best_confidence, best_reason

    def validate_assignment(self, agent_id: str, task_description: str) -> Tuple[bool, str]:
        """
        Validate if an agent assignment is appropriate for the task.

        Args:
            agent_id: The agent being assigned
            task_description: Description of the task

        Returns:
            Tuple of (is_valid, validation_reason)
        """
        if agent_id not in self.agents:
            return False, f"Unknown agent: {agent_id}"

        agent = self.agents[agent_id]
        task_lower = task_description.lower()

        # Check for weak keywords (things this agent should avoid)
        for weak_keyword in agent.weak_keywords:
            if weak_keyword in task_lower:
                return False, f"Agent {agent.name} should avoid tasks containing '{weak_keyword}'"

        # Check for expertise keywords
        expertise_matches = sum(1 for keyword in agent.expertise_keywords if keyword in task_lower)
        weak_matches = sum(1 for keyword in agent.weak_keywords if keyword in task_lower)

        if expertise_matches == 0 and agent_id != "general_assistant":
            return False, f"Task doesn't match {agent.name}'s expertise areas"

        if weak_matches > 0:
            return False, f"Task contains keywords that {agent.name} should avoid"

        # Calculate confidence for validation
        confidence, _ = self._calculate_match_confidence(task_lower, agent)

        if confidence < 0.2:
            return False, f"Confidence too low ({confidence:.2f}) for {agent.name}"

        return True, f"Valid assignment for {agent.name} (confidence: {confidence:.2f})"

    def _calculate_match_confidence(self, task_lower: str, agent: AgentCapability) -> Tuple[float, str]:
        """
        Calculate how well a task matches an agent's capabilities.

        Returns:
            Tuple of (confidence_score, reason)
        """
        confidence = agent.base_confidence
        reasons = []

        # Check expertise keywords
        expertise_matches = 0
        for keyword in agent.expertise_keywords:
            if keyword in task_lower:
                expertise_matches += 1
                confidence += 0.1

        # Check weak keywords (penalize)
        weak_matches = 0
        for keyword in agent.weak_keywords:
            if keyword in task_lower:
                weak_matches += 1
                confidence -= 0.2

        # Domain-specific markers for additional signal
        web_markers = ["web", "website", "frontend", "html", "css", "javascript", "react", "nextjs", "vite"]
        devops_markers = ["deploy", "deployment", "kubernetes", "k8s", "helm", "docker", "terraform", "ci/cd", "ci", "cd"]
        data_markers = ["data", "dataset", "analysis", "analytics", "pandas", "sql", "snowflake"]
        security_markers = ["security", "audit", "vulnerability", "vuln", "pentest", "owasp", "encryption", "authentication"]
        test_markers = ["test", "unit test", "pytest", "coverage"]
        docs_context = ["docs/", "/docs", "readme", "documentation"]

        # Apply small boosts based on markers aligned to each agent
        if agent.agent_id == "web_developer":
            if any(m in task_lower for m in web_markers):
                confidence += 0.10
                reasons.append("web-related markers")
            if any(m in task_lower for m in docs_context) and ("website" in task_lower or "web" in task_lower):
                confidence += 0.05
                reasons.append("docs context")
        elif agent.agent_id == "devops_engineer":
            if any(m in task_lower for m in devops_markers):
                confidence += 0.10
                reasons.append("devops markers")
        elif agent.agent_id == "data_analyst":
            if any(m in task_lower for m in data_markers):
                confidence += 0.10
                reasons.append("data markers")
        elif agent.agent_id == "security_expert":
            if any(m in task_lower for m in security_markers):
                confidence += 0.10
                reasons.append("security markers")
        elif agent.agent_id == "qa_tester":
            if any(m in task_lower for m in test_markers):
                confidence += 0.05
                reasons.append("testing markers")

        # Domain-specific markers for additional signal
        web_markers = ["web", "website", "frontend", "html", "css", "javascript", "react", "nextjs", "vite"]
        devops_markers = ["deploy", "deployment", "kubernetes", "k8s", "helm", "docker", "terraform", "ci/cd", "ci", "cd"]
        data_markers = ["data", "dataset", "analysis", "analytics", "pandas", "sql", "snowflake"]
        security_markers = ["security", "audit", "vulnerability", "vuln", "pentest", "owasp", "encryption", "authentication"]
        test_markers = ["test", "unit test", "pytest", "coverage"]
        docs_context = ["docs/", "/docs", "readme", "documentation"]

        # Apply small boosts based on markers aligned to each agent
        if agent.agent_id == "web_developer":
            if any(m in task_lower for m in web_markers):
                confidence += 0.10
                reasons.append("web-related markers")
            if any(m in task_lower for m in docs_context) and ("website" in task_lower or "web" in task_lower):
                confidence += 0.05
                reasons.append("docs context")
        elif agent.agent_id == "devops_engineer":
            if any(m in task_lower for m in devops_markers):
                confidence += 0.10
                reasons.append("devops markers")
        elif agent.agent_id == "data_analyst":
            if any(m in task_lower for m in data_markers):
                confidence += 0.10
                reasons.append("data markers")
        elif agent.agent_id == "security_expert":
            if any(m in task_lower for m in security_markers):
                confidence += 0.10
                reasons.append("security markers")
        elif agent.agent_id == "qa_tester":
            if any(m in task_lower for m in test_markers):
                confidence += 0.05
                reasons.append("testing markers")

        # Bonus for multiple expertise matches
        if expertise_matches > 1:
            confidence += 0.1 * (expertise_matches - 1)

        # Penalty for weak keyword matches
        if weak_matches > 0:
            confidence -= 0.1 * weak_matches

        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))

        # Build reason string
        if expertise_matches > 0:
            reasons.append(f"Matched {expertise_matches} expertise keywords")
        if weak_matches > 0:
            reasons.append(f"Contains {weak_matches} weak keywords")

        reason = f"{agent.name}: " + ", ".join(reasons) if reasons else f"{agent.name}: Base capability match"

        return confidence, reason

    def get_available_agents(self) -> List[str]:
        """Get list of all available agent IDs"""
        return list(self.agents.keys())

    def get_agent_info(self, agent_id: str) -> Optional[AgentCapability]:
        """Get detailed information about a specific agent"""
        return self.agents.get(agent_id)

    def add_agent(self, agent: AgentCapability):
        """Add a new agent to the matcher (for extensibility)"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added new agent: {agent.name} ({agent.agent_id})")

    def remove_agent(self, agent_id: str):
        """Remove an agent from the matcher"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent: {agent_id}")


# Global instance for easy access
_capability_matcher = None

def get_capability_matcher() -> AgentCapabilityMatcher:
    """Get the global capability matcher instance"""
    global _capability_matcher
    if _capability_matcher is None:
        _capability_matcher = AgentCapabilityMatcher()
    return _capability_matcher


if __name__ == "__main__":
    # Test the capability matcher
    matcher = AgentCapabilityMatcher()

    test_tasks = [
        "Create a hello world website",
        "Perform security audit on the system",
        "Analyze sales data and create visualizations",
        "Deploy application to production",
        "Write automated tests for the API",
        "Document the project architecture"
    ]

    print("Testing Agent Capability Matcher:")
    print("=" * 50)

    for task in test_tasks:
        agent, confidence, reason = matcher.find_best_agent(task)
        is_valid, validation_reason = matcher.validate_assignment(agent, task)

        print(f"\nTask: {task}")
        print(f"Assigned Agent: {agent}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reason: {reason}")
        print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'} - {validation_reason}")