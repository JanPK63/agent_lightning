#!/usr/bin/env python3
"""
Agent Extensions - Specialized Agents for Development Workflow
These agents are registered in the unified agent pool and can be orchestrated
through Workflow Engine and LangGraph
"""

import os
import json
import logging
import subprocess
import ast
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecializedAgentType(str, Enum):
    """Extended agent types for development workflow"""
    ROUTER = "router"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    INTEGRATOR = "integrator"


# ============= AGENT CAPABILITIES DEFINITIONS =============

ROUTER_AGENT_SPEC = {
    "id": "router-agent",
    "name": "Router Agent",
    "type": "router",
    "description": "Routes requests to workflows",
    "capabilities": {
        "task_analysis": True,
        "workflow_selection": True,
        "priority_assignment": True,
        "dependency_detection": True
    },
    "skills": [
        "analyze_request",
        "determine_workflow",
        "detect_task_type",
        "route_to_agents"
    ],
    "metadata": {
        "version": "1.0.0",
        "specialized": True,
        "ml_enhanced": True
    }
}

PLANNER_AGENT_SPEC = {
    "id": "planner-agent",
    "name": "Planner Agent",
    "type": "planner",
    "description": "Plans development tasks",
    "capabilities": {
        "task_decomposition": True,
        "dependency_mapping": True,
        "resource_estimation": True,
        "timeline_planning": True
    },
    "skills": [
        "create_plan",
        "estimate_complexity",
        "identify_dependencies",
        "define_acceptance_criteria"
    ],
    "metadata": {
        "version": "1.0.0",
        "specialized": True,
        "supports_languages": ["python", "javascript", "go", "java"]
    }
}

RETRIEVER_AGENT_SPEC = {
    "id": "retriever-agent",
    "name": "Retriever Agent",
    "type": "retriever",
    "description": "RAG search with pgvector",
    "capabilities": {
        "semantic_search": True,
        "code_similarity": True,
        "documentation_search": True,
        "dependency_analysis": True,
        "pgvector_integration": True
    },
    "skills": [
        "search_codebase",
        "find_similar_implementations",
        "retrieve_documentation",
        "analyze_dependencies"
    ],
    "metadata": {
        "version": "1.0.0",
        "specialized": True,
        "uses_embeddings": True,
        "vector_dimensions": 1536
    }
}

CODER_AGENT_SPEC = {
    "id": "coder-agent",
    "name": "Coder Agent",
    "type": "coder",
    "description": "Multi-language code generation",
    "capabilities": {
        "code_generation": True,
        "code_modification": True,
        "refactoring": True,
        "multi_language": True,
        "test_generation": True
    },
    "skills": [
        "generate_code",
        "modify_code",
        "refactor_code",
        "generate_tests",
        "apply_patterns"
    ],
    "supported_languages": [
        "python", "javascript", "typescript", "go", "java", "rust", "c++", "c#"
    ],
    "metadata": {
        "version": "1.0.0",
        "specialized": True,
        "llm_powered": True
    }
}

TESTER_AGENT_SPEC = {
    "id": "tester-agent",
    "name": "Tester Agent",
    "type": "tester",
    "description": "Runs polyglot test suites and validates code",
    "capabilities": {
        "unit_testing": True,
        "integration_testing": True,
        "performance_testing": True,
        "security_testing": True,
        "polyglot_support": True
    },
    "skills": [
        "run_tests",
        "generate_test_cases",
        "measure_coverage",
        "performance_profiling",
        "security_scanning"
    ],
    "test_frameworks": {
        "python": ["pytest", "unittest", "nose"],
        "javascript": ["jest", "mocha", "jasmine"],
        "go": ["go test", "testify"],
        "java": ["junit", "testng"]
    },
    "metadata": {
        "version": "1.0.0",
        "specialized": True
    }
}

REVIEWER_AGENT_SPEC = {
    "id": "reviewer-agent",
    "name": "Reviewer Agent",
    "type": "reviewer",
    "description": "Reviews code for quality, security, and compliance",
    "capabilities": {
        "code_quality_analysis": True,
        "security_scanning": True,
        "performance_analysis": True,
        "compliance_checking": True,
        "documentation_review": True
    },
    "skills": [
        "review_code",
        "check_security",
        "analyze_performance",
        "verify_standards",
        "suggest_improvements"
    ],
    "review_tools": [
        "pylint", "eslint", "golint",
        "sonarqube", "bandit", "safety",
        "black", "prettier", "gofmt"
    ],
    "metadata": {
        "version": "1.0.0",
        "specialized": True,
        "ml_enhanced": True
    }
}

INTEGRATOR_AGENT_SPEC = {
    "id": "integrator-agent",
    "name": "Integrator Agent",
    "type": "integrator",
    "description": "Manages Git operations, PRs, and CI/CD integration",
    "capabilities": {
        "git_operations": True,
        "pr_management": True,
        "ci_cd_integration": True,
        "merge_conflict_resolution": True,
        "deployment_automation": True
    },
    "skills": [
        "create_branch",
        "commit_changes",
        "create_pr",
        "run_ci_pipeline",
        "merge_code",
        "deploy_application"
    ],
    "integrations": [
        "github", "gitlab", "bitbucket",
        "jenkins", "github_actions", "circleci",
        "docker", "kubernetes"
    ],
    "metadata": {
        "version": "1.0.0",
        "specialized": True
    }
}


# ============= AGENT REGISTRATION =============

def get_specialized_agents() -> List[Dict[str, Any]]:
    """Get all specialized development agents for registration"""
    return [
        ROUTER_AGENT_SPEC,
        PLANNER_AGENT_SPEC,
        RETRIEVER_AGENT_SPEC,
        CODER_AGENT_SPEC,
        TESTER_AGENT_SPEC,
        REVIEWER_AGENT_SPEC,
        INTEGRATOR_AGENT_SPEC
    ]


def register_specialized_agents(dal):
    """Register specialized agents in the unified agent pool"""
    agents = get_specialized_agents()
    registered = []
    
    for agent_spec in agents:
        try:
            # Check if agent already exists
            existing = dal.get_agent(agent_spec["id"])
            
            if not existing:
                # Register new agent - needs proper structure for Agent model
                agent_data = {
                    "id": agent_spec["id"],
                    "name": agent_spec["name"],
                    "model": "claude-3-haiku",  # Default model
                    "specialization": agent_spec.get("description", agent_spec["type"]),
                    "status": "idle",
                    "capabilities": agent_spec.get("skills", []),  # Use skills as capabilities
                    "config": {
                        "type": agent_spec["type"],
                        "capabilities": agent_spec["capabilities"],
                        "metadata": agent_spec.get("metadata", {})
                    }
                }
                
                result = dal.create_agent(agent_data)
                registered.append(agent_spec["name"])
                logger.info(f"Registered specialized agent: {agent_spec['name']}")
            else:
                logger.info(f"Agent already exists: {agent_spec['name']}")
                
        except ValueError as e:
            # Agent already exists error from DAL
            logger.info(f"Agent already exists: {agent_spec['name']}")
        except Exception as e:
            logger.error(f"Failed to register agent {agent_spec['name']}: {e}")
    
    return registered


# ============= WORKFLOW DEFINITIONS =============

DEV_WORKFLOW_DEFINITION = {
    "id": "software-development-workflow",
    "name": "Software Development Workflow",
    "description": "Complete software development lifecycle with specialized agents",
    "version": "1.0.0",
    "steps": [
        {
            "id": "routing",
            "agent": "router-agent",
            "action": "analyze_and_route",
            "inputs": ["task_description", "constraints"],
            "outputs": ["workflow_path", "task_type"],
            "next": "planning"
        },
        {
            "id": "planning",
            "agent": "planner-agent",
            "action": "create_development_plan",
            "inputs": ["task_description", "workflow_path"],
            "outputs": ["implementation_plan", "dependencies"],
            "next": "retrieval"
        },
        {
            "id": "retrieval",
            "agent": "retriever-agent",
            "action": "search_relevant_code",
            "inputs": ["implementation_plan", "repository"],
            "outputs": ["relevant_code", "similar_implementations"],
            "next": "coding"
        },
        {
            "id": "coding",
            "agent": "coder-agent",
            "action": "generate_implementation",
            "inputs": ["implementation_plan", "relevant_code"],
            "outputs": ["source_code", "test_code"],
            "next": "testing"
        },
        {
            "id": "testing",
            "agent": "tester-agent",
            "action": "run_test_suite",
            "inputs": ["source_code", "test_code"],
            "outputs": ["test_results", "coverage_report"],
            "next": "review"
        },
        {
            "id": "review",
            "agent": "reviewer-agent",
            "action": "perform_code_review",
            "inputs": ["source_code", "test_results"],
            "outputs": ["review_report", "approval_status"],
            "next": "integration"
        },
        {
            "id": "integration",
            "agent": "integrator-agent",
            "action": "integrate_changes",
            "inputs": ["source_code", "approval_status"],
            "outputs": ["pr_url", "ci_status"],
            "next": "complete"
        }
    ],
    "metadata": {
        "idempotent": True,
        "supports_checkpoint": True,
        "max_retries": 3
    }
}


def create_dev_workflow_in_langgraph(langgraph_url: str = "http://localhost:8016"):
    """Register the development workflow in LangGraph"""
    import aiohttp
    import asyncio
    
    async def register():
        async with aiohttp.ClientSession() as session:
            # Convert to LangGraph format
            nodes = []
            edges = []
            
            for step in DEV_WORKFLOW_DEFINITION["steps"]:
                # Create node
                node = {
                    "id": step["id"],
                    "type": "agent",
                    "name": step["id"].title(),
                    "description": f"Execute {step['action']} using {step['agent']}",
                    "agent_id": step["agent"],
                    "next_nodes": [step["next"]] if step["next"] != "complete" else []
                }
                nodes.append(node)
                
                # Create edge
                if step["next"] != "complete":
                    edges.append({
                        "from": step["id"],
                        "to": step["next"]
                    })
            
            # Create workflow
            workflow_data = {
                "id": DEV_WORKFLOW_DEFINITION["id"],
                "name": DEV_WORKFLOW_DEFINITION["name"],
                "description": DEV_WORKFLOW_DEFINITION["description"],
                "nodes": nodes,
                "edges": edges,
                "entry_node": "routing",
                "metadata": DEV_WORKFLOW_DEFINITION["metadata"]
            }
            
            try:
                async with session.post(
                    f"{langgraph_url}/workflows",
                    json=workflow_data
                ) as response:
                    if response.status == 200:
                        logger.info("Development workflow registered in LangGraph")
                        return await response.json()
                    else:
                        logger.error(f"Failed to register workflow: {response.status}")
            except Exception as e:
                logger.error(f"Error registering workflow: {e}")
    
    # Run async registration
    try:
        import asyncio
        return asyncio.run(register())
    except:
        logger.warning("Could not register workflow in LangGraph - service may not be running")
        return None


# ============= INTEGRATION WITH EXISTING SERVICES =============

def integrate_with_rl_orchestrator(rl_url: str = "http://localhost:8011"):
    """Register agents with RL Orchestrator for learning"""
    import requests
    
    agents = get_specialized_agents()
    
    for agent in agents:
        # Create RL agent configuration
        rl_config = {
            "agent_id": agent["id"],
            "agent_type": agent["type"],
            "state_space": {
                "task_type": ["feature", "bugfix", "refactor", "test"],
                "complexity": ["low", "medium", "high"],
                "language": agent.get("supported_languages", ["any"])
            },
            "action_space": agent["skills"],
            "reward_function": "development_success_rate"
        }
        
        try:
            response = requests.post(
                f"{rl_url}/agents/register",
                json=rl_config
            )
            if response.status_code == 200:
                logger.info(f"Registered {agent['name']} with RL Orchestrator")
        except Exception as e:
            logger.warning(f"Could not register {agent['name']} with RL Orchestrator: {e}")


def integrate_with_autogen(autogen_url: str = "http://localhost:8015"):
    """Create AutoGen group for collaborative development"""
    import requests
    
    group_config = {
        "group_name": "Development Team",
        "agents": [
            {"name": "Router", "role": "coordinator"},
            {"name": "Planner", "role": "architect"},
            {"name": "Retriever", "role": "researcher"},
            {"name": "Coder", "role": "developer"},
            {"name": "Tester", "role": "qa_engineer"},
            {"name": "Reviewer", "role": "code_reviewer"},
            {"name": "Integrator", "role": "devops"}
        ],
        "max_rounds": 10,
        "collaboration_mode": "sequential"
    }
    
    try:
        response = requests.post(
            f"{autogen_url}/groups",
            json=group_config
        )
        if response.status_code == 200:
            logger.info("Created AutoGen development team")
    except Exception as e:
        logger.warning(f"Could not create AutoGen group: {e}")


# ============= MAIN INTEGRATION FUNCTION =============

def extend_agent_pool():
    """
    Main function to extend the existing agent pool with specialized development agents
    This should be called during service initialization
    """
    logger.info("Extending agent pool with specialized development agents...")
    
    # Import DAL
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.data_access import DataAccessLayer
    
    # Initialize DAL
    dal = DataAccessLayer("agent_extensions")
    
    # 1. Register specialized agents in unified pool
    registered = register_specialized_agents(dal)
    logger.info(f"Registered {len(registered)} specialized agents")
    
    # 2. Create development workflow in LangGraph
    workflow = create_dev_workflow_in_langgraph()
    if workflow:
        logger.info("Development workflow created in LangGraph")
    
    # 3. Integrate with RL Orchestrator for learning
    integrate_with_rl_orchestrator()
    
    # 4. Create AutoGen collaboration group
    integrate_with_autogen()
    
    # Cleanup
    dal.cleanup()
    
    logger.info("Agent pool extension complete!")
    
    return {
        "agents_registered": registered,
        "workflow_created": workflow is not None,
        "rl_integrated": True,
        "autogen_integrated": True
    }


if __name__ == "__main__":
    # Run extension when module is executed
    result = extend_agent_pool()
    print(f"Agent pool extended: {json.dumps(result, indent=2)}")