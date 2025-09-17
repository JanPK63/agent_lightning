"""
Agent Configuration System for Agent Lightning
Define specialized agents with specific knowledge domains and capabilities
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class AgentRole(Enum):
    """Predefined agent roles"""
    FULL_STACK_DEVELOPER = "full_stack_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    DATA_SCIENTIST = "data_scientist"
    DEVOPS_ENGINEER = "devops_engineer"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    ARCHITECT = "system_architect"
    ANALYST = "information_analyst"
    CUSTOM = "custom"


@dataclass
class KnowledgeBase:
    """Knowledge configuration for an agent"""
    domains: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    context_files: List[str] = field(default_factory=list)  # Files to load as context
    documentation_urls: List[str] = field(default_factory=list)
    custom_instructions: str = ""


@dataclass
class AgentCapabilities:
    """Define what an agent can do"""
    can_write_code: bool = False
    can_debug: bool = False
    can_review_code: bool = False
    can_optimize: bool = False
    can_test: bool = False
    can_deploy: bool = False
    can_design_architecture: bool = False
    can_write_documentation: bool = False
    can_analyze_data: bool = False
    can_generate_reports: bool = False


@dataclass
class AgentConfig:
    """Complete configuration for an agent"""
    name: str
    role: AgentRole
    description: str
    model: str = "gpt-4o"  # Default model
    temperature: float = 0.7
    max_tokens: int = 4000
    knowledge_base: KnowledgeBase = field(default_factory=KnowledgeBase)
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    system_prompt: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)  # Available tools/functions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "role": self.role.value,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "knowledge_base": {
                "domains": self.knowledge_base.domains,
                "technologies": self.knowledge_base.technologies,
                "frameworks": self.knowledge_base.frameworks,
                "best_practices": self.knowledge_base.best_practices,
                "context_files": self.knowledge_base.context_files,
                "documentation_urls": self.knowledge_base.documentation_urls,
                "custom_instructions": self.knowledge_base.custom_instructions
            },
            "capabilities": {
                "can_write_code": self.capabilities.can_write_code,
                "can_debug": self.capabilities.can_debug,
                "can_review_code": self.capabilities.can_review_code,
                "can_optimize": self.capabilities.can_optimize,
                "can_test": self.capabilities.can_test,
                "can_deploy": self.capabilities.can_deploy,
                "can_design_architecture": self.capabilities.can_design_architecture,
                "can_write_documentation": self.capabilities.can_write_documentation,
                "can_analyze_data": self.capabilities.can_analyze_data,
                "can_generate_reports": self.capabilities.can_generate_reports
            },
            "system_prompt": self.system_prompt,
            "examples": self.examples,
            "tools": self.tools
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentConfig":
        """Create from dictionary"""
        knowledge_base = KnowledgeBase(**data.get("knowledge_base", {}))
        capabilities = AgentCapabilities(**data.get("capabilities", {}))
        
        return cls(
            name=data["name"],
            role=AgentRole(data["role"]),
            description=data["description"],
            model=data.get("model", "gpt-4o"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4000),
            knowledge_base=knowledge_base,
            capabilities=capabilities,
            system_prompt=data.get("system_prompt", ""),
            examples=data.get("examples", []),
            tools=data.get("tools", [])
        )


class AgentConfigManager:
    """Manage agent configurations"""
    
    def __init__(self, config_dir: str = ".agent-configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.agents: Dict[str, AgentConfig] = {}
        self.load_configs()
    
    def load_configs(self):
        """Load all agent configurations from disk"""
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    agent = AgentConfig.from_dict(data)
                    self.agents[agent.name] = agent
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
    
    def save_agent(self, agent: AgentConfig):
        """Save an agent configuration"""
        config_file = self.config_dir / f"{agent.name}.json"
        with open(config_file, 'w') as f:
            json.dump(agent.to_dict(), f, indent=2)
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get an agent configuration by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all available agents"""
        return list(self.agents.keys())
    
    def delete_agent(self, name: str):
        """Delete an agent configuration"""
        if name in self.agents:
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
            del self.agents[name]
    
    def create_full_stack_developer(self) -> AgentConfig:
        """Create a pre-configured full-stack developer agent"""
        knowledge_base = KnowledgeBase(
            domains=[
                "Web Development",
                "Software Architecture",
                "Database Design",
                "API Development",
                "Cloud Computing",
                "DevOps"
            ],
            technologies=[
                "Python", "JavaScript", "TypeScript", "SQL", "NoSQL",
                "HTML5", "CSS3", "Node.js", "Docker", "Kubernetes",
                "Git", "CI/CD", "REST", "GraphQL", "WebSockets"
            ],
            frameworks=[
                "React", "Vue.js", "Angular", "Next.js",
                "Django", "FastAPI", "Flask", "Express.js",
                "Spring Boot", "Ruby on Rails",
                "PostgreSQL", "MongoDB", "Redis",
                "AWS", "Azure", "Google Cloud"
            ],
            best_practices=[
                "Clean Code principles",
                "SOLID principles",
                "Design Patterns",
                "Test-Driven Development (TDD)",
                "Agile methodologies",
                "Security best practices",
                "Performance optimization",
                "Responsive design",
                "Accessibility standards",
                "Code review practices"
            ],
            custom_instructions="""You are an expert full-stack developer with deep knowledge of both frontend and backend technologies. 
            You write clean, maintainable, and efficient code. You follow best practices and design patterns.
            You can architect complete applications from scratch and debug complex issues.
            You provide detailed explanations and consider security, performance, and scalability in your solutions."""
        )
        
        capabilities = AgentCapabilities(
            can_write_code=True,
            can_debug=True,
            can_review_code=True,
            can_optimize=True,
            can_test=True,
            can_deploy=True,
            can_design_architecture=True,
            can_write_documentation=True,
            can_analyze_data=False,
            can_generate_reports=True
        )
        
        system_prompt = """You are a Senior Full-Stack Developer with 10+ years of experience.

Your expertise includes:
- Frontend: React, Vue, Angular, HTML5, CSS3, JavaScript/TypeScript
- Backend: Python (Django, FastAPI), Node.js, Java (Spring Boot)
- Databases: PostgreSQL, MongoDB, Redis, MySQL
- Cloud: AWS, Azure, GCP, Docker, Kubernetes
- DevOps: CI/CD, GitHub Actions, Jenkins, Terraform
- Architecture: Microservices, REST APIs, GraphQL, WebSockets

When solving problems:
1. First understand the requirements completely
2. Consider multiple approaches and trade-offs
3. Write clean, maintainable code with proper error handling
4. Include tests when appropriate
5. Consider security, performance, and scalability
6. Provide clear documentation and comments
7. Follow industry best practices and design patterns

You can handle any full-stack development task from database design to frontend UI implementation."""
        
        examples = [
            {
                "input": "Create a user authentication system",
                "output": "I'll create a secure authentication system with JWT tokens, password hashing, and role-based access control..."
            },
            {
                "input": "Optimize database query performance",
                "output": "Let me analyze the query execution plan and suggest optimizations including indexes, query restructuring..."
            }
        ]
        
        tools = [
            "code_generation",
            "code_review",
            "debugging",
            "testing",
            "documentation",
            "architecture_design",
            "performance_analysis"
        ]
        
        return AgentConfig(
            name="full_stack_developer",
            role=AgentRole.FULL_STACK_DEVELOPER,
            description="Expert full-stack developer for complete web application development",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4000,
            knowledge_base=knowledge_base,
            capabilities=capabilities,
            system_prompt=system_prompt,
            examples=examples,
            tools=tools
        )
    
    def create_data_scientist(self) -> AgentConfig:
        """Create a pre-configured data scientist agent"""
        knowledge_base = KnowledgeBase(
            domains=[
                "Machine Learning",
                "Deep Learning",
                "Statistics",
                "Data Analysis",
                "Data Visualization",
                "Big Data"
            ],
            technologies=[
                "Python", "R", "SQL", "Spark", "Hadoop",
                "TensorFlow", "PyTorch", "Scikit-learn",
                "Pandas", "NumPy", "Matplotlib", "Seaborn",
                "Jupyter", "Apache Airflow"
            ],
            frameworks=[
                "XGBoost", "LightGBM", "CatBoost",
                "Keras", "FastAI", "Transformers",
                "MLflow", "Kubeflow", "Ray"
            ],
            best_practices=[
                "Feature engineering",
                "Model evaluation metrics",
                "Cross-validation",
                "Hyperparameter tuning",
                "A/B testing",
                "Statistical significance",
                "Data preprocessing",
                "Model interpretability"
            ],
            custom_instructions="You are an expert data scientist who can analyze complex datasets, build ML models, and provide actionable insights."
        )
        
        capabilities = AgentCapabilities(
            can_write_code=True,
            can_debug=True,
            can_review_code=True,
            can_optimize=True,
            can_test=True,
            can_analyze_data=True,
            can_generate_reports=True
        )
        
        return AgentConfig(
            name="data_scientist",
            role=AgentRole.DATA_SCIENTIST,
            description="Expert in data analysis, machine learning, and statistical modeling",
            model="gpt-4o",
            temperature=0.6,
            max_tokens=4000,
            knowledge_base=knowledge_base,
            capabilities=capabilities
        )


# Pre-configured agent templates
AGENT_TEMPLATES = {
    "full_stack_developer": {
        "creator": lambda: AgentConfigManager().create_full_stack_developer(),
        "description": "Full-stack web developer with frontend and backend expertise"
    },
    "data_scientist": {
        "creator": lambda: AgentConfigManager().create_data_scientist(),
        "description": "Data analysis and machine learning expert"
    },
    "frontend_developer": {
        "description": "Frontend specialist focusing on UI/UX and modern frameworks"
    },
    "backend_developer": {
        "description": "Backend specialist for APIs, databases, and server architecture"
    },
    "devops_engineer": {
        "description": "Infrastructure, deployment, and automation expert"
    }
}

# Note: This file is deprecated. All agent management has been migrated to the Agent Designer Service (microservice).
# The Test Engineer agent has been created directly in the Agent Designer Service on port 8002.
# All agents are now managed through the unified microservice architecture.

if __name__ == "__main__":
    # Example: Create and save a full-stack developer agent
    manager = AgentConfigManager()
    
    # Create full-stack developer
    full_stack = manager.create_full_stack_developer()
    manager.save_agent(full_stack)
    
    print(f"Created agent: {full_stack.name}")
    print(f"Role: {full_stack.role.value}")
    print(f"Technologies: {', '.join(full_stack.knowledge_base.technologies[:5])}...")
    print(f"Capabilities: {sum(1 for k, v in full_stack.capabilities.__dict__.items() if v)} enabled")
    
    # List all agents
    print(f"\nAvailable agents: {manager.list_agents()}")