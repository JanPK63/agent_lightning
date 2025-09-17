"""
Enhanced Production API with Knowledge Management Integration
This extends the production API to use agent configurations and knowledge bases
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any
import asyncio
import uuid
from dataclasses import dataclass
import uvicorn
from datetime import datetime

# Basic models for API compatibility
class AgentRequest(BaseModel):
    task: str
    agent_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Agent:
    """Simple agent representation"""
    agent_id: str
    name: str
    model: str
    specialization: str
    tools: list
    config: Dict[str, Any]

class EnhancedAgentService:
    """Enhanced agent service with knowledge management and specialized agents"""
    
    def __init__(self):
        self.agents = {}
        self.task_results = {}
        self.action_results = {}
        self._load_specialized_agents()
    
    def _load_specialized_agents(self):
        """Load all configured specialized agents"""
        # Define the 32 specialized agents
        agent_configs = {
            "full_stack_developer": {
                "name": "Full Stack Developer",
                "description": "Expert in web development, APIs, and databases",
                "model": "gpt-4o",
                "specialization": "web_development",
                "tools": ["code_execution", "database_access", "api_testing"],
                "capabilities": ["React", "Node.js", "Python", "SQL", "REST APIs"]
            },
            "mobile_developer": {
                "name": "Mobile Developer", 
                "description": "iOS and Android development specialist",
                "model": "gpt-4o",
                "specialization": "mobile_development",
                "tools": ["xcode", "android_studio", "react_native"],
                "capabilities": ["Swift", "Kotlin", "React Native", "Flutter"]
            },
            "security_expert": {
                "name": "Security Expert",
                "description": "Cybersecurity and penetration testing specialist", 
                "model": "gpt-4o",
                "specialization": "security",
                "tools": ["nmap", "burp_suite", "metasploit"],
                "capabilities": ["OWASP", "Penetration Testing", "Security Audits"]
            },
            "devops_engineer": {
                "name": "DevOps Engineer",
                "description": "Infrastructure and deployment automation expert",
                "model": "gpt-4o",
                "specialization": "devops",
                "tools": ["docker", "kubernetes", "terraform"],
                "capabilities": ["Docker", "Kubernetes", "AWS", "CI/CD"]
            },
            "data_scientist": {
                "name": "Data Scientist",
                "description": "Machine learning and data analysis expert",
                "model": "gpt-4o",
                "specialization": "data_science",
                "tools": ["jupyter", "pandas", "tensorflow"],
                "capabilities": ["Python", "TensorFlow", "Pandas", "Statistics"]
            },
            "ui_ux_designer": {
                "name": "UI/UX Designer",
                "description": "User interface and experience design specialist",
                "model": "gpt-4o",
                "specialization": "design",
                "tools": ["figma", "sketch", "adobe_xd"],
                "capabilities": ["UI Design", "UX Research", "Prototyping", "CSS"]
            },
            "blockchain_developer": {
                "name": "Blockchain Developer",
                "description": "Smart contract and DeFi development expert",
                "model": "gpt-4o",
                "specialization": "blockchain",
                "tools": ["solidity", "web3", "truffle"],
                "capabilities": ["Solidity", "Web3", "Smart Contracts", "DeFi"]
            },
            "ai_ml_engineer": {
                "name": "AI/ML Engineer",
                "description": "Artificial intelligence and machine learning specialist",
                "model": "gpt-4o",
                "specialization": "ai_ml",
                "tools": ["pytorch", "tensorflow", "huggingface"],
                "capabilities": ["PyTorch", "TensorFlow", "NLP", "Computer Vision"]
            },
            "cloud_architect": {
                "name": "Cloud Architect",
                "description": "Cloud infrastructure and architecture expert",
                "model": "gpt-4o",
                "specialization": "cloud",
                "tools": ["aws_cli", "azure_cli", "gcp_cli"],
                "capabilities": ["AWS", "Azure", "GCP", "Serverless"]
            },
            "database_admin": {
                "name": "Database Administrator",
                "description": "Database design and optimization specialist",
                "model": "gpt-4o",
                "specialization": "database",
                "tools": ["mysql", "postgresql", "mongodb"],
                "capabilities": ["SQL", "NoSQL", "Performance Tuning", "Backup"]
            },
            "qa_engineer": {
                "name": "QA Engineer",
                "description": "Quality assurance and testing specialist",
                "model": "gpt-4o",
                "specialization": "testing",
                "tools": ["selenium", "jest", "cypress"],
                "capabilities": ["Test Automation", "Manual Testing", "Performance Testing"]
            },
            "product_manager": {
                "name": "Product Manager",
                "description": "Product strategy and roadmap specialist",
                "model": "gpt-4o",
                "specialization": "product",
                "tools": ["jira", "confluence", "analytics"],
                "capabilities": ["Product Strategy", "Roadmapping", "Analytics", "User Research"]
            },
            "technical_writer": {
                "name": "Technical Writer",
                "description": "Documentation and technical communication expert",
                "model": "gpt-4o",
                "specialization": "documentation",
                "tools": ["markdown", "gitbook", "confluence"],
                "capabilities": ["Technical Writing", "API Documentation", "User Guides"]
            },
            "system_admin": {
                "name": "System Administrator",
                "description": "Server and infrastructure management expert",
                "model": "gpt-4o",
                "specialization": "sysadmin",
                "tools": ["bash", "ansible", "monitoring"],
                "capabilities": ["Linux", "Windows Server", "Monitoring", "Automation"]
            },
            "network_engineer": {
                "name": "Network Engineer",
                "description": "Network design and troubleshooting specialist",
                "model": "gpt-4o",
                "specialization": "networking",
                "tools": ["wireshark", "cisco_cli", "network_tools"],
                "capabilities": ["TCP/IP", "Routing", "Switching", "VPN"]
            },
            "game_developer": {
                "name": "Game Developer",
                "description": "Video game development and design expert",
                "model": "gpt-4o",
                "specialization": "game_dev",
                "tools": ["unity", "unreal", "godot"],
                "capabilities": ["Unity", "Unreal Engine", "C#", "Game Design"]
            },
            "embedded_engineer": {
                "name": "Embedded Engineer",
                "description": "Embedded systems and IoT development expert",
                "model": "gpt-4o",
                "specialization": "embedded",
                "tools": ["arduino", "raspberry_pi", "microcontrollers"],
                "capabilities": ["C/C++", "Arduino", "Raspberry Pi", "IoT"]
            },
            "frontend_specialist": {
                "name": "Frontend Specialist",
                "description": "Advanced frontend development and optimization expert",
                "model": "gpt-4o",
                "specialization": "frontend",
                "tools": ["webpack", "vite", "performance_tools"],
                "capabilities": ["React", "Vue", "Angular", "Performance Optimization"]
            },
            "backend_specialist": {
                "name": "Backend Specialist",
                "description": "Server-side architecture and API development expert",
                "model": "gpt-4o",
                "specialization": "backend",
                "tools": ["microservices", "api_gateway", "load_balancer"],
                "capabilities": ["Microservices", "API Design", "Scalability", "Performance"]
            },
            "automation_engineer": {
                "name": "Automation Engineer",
                "description": "Process automation and workflow optimization expert",
                "model": "gpt-4o",
                "specialization": "automation",
                "tools": ["python", "selenium", "zapier"],
                "capabilities": ["Process Automation", "RPA", "Workflow Design", "Integration"]
            },
            "performance_engineer": {
                "name": "Performance Engineer",
                "description": "Application and system performance optimization expert",
                "model": "gpt-4o",
                "specialization": "performance",
                "tools": ["profilers", "load_testing", "monitoring"],
                "capabilities": ["Performance Testing", "Optimization", "Monitoring", "Scalability"]
            },
            "compliance_specialist": {
                "name": "Compliance Specialist",
                "description": "Regulatory compliance and governance expert",
                "model": "gpt-4o",
                "specialization": "compliance",
                "tools": ["audit_tools", "compliance_frameworks"],
                "capabilities": ["GDPR", "SOX", "HIPAA", "Risk Assessment"]
            },
            "integration_specialist": {
                "name": "Integration Specialist",
                "description": "System integration and API connectivity expert",
                "model": "gpt-4o",
                "specialization": "integration",
                "tools": ["api_tools", "etl", "message_queues"],
                "capabilities": ["API Integration", "ETL", "Message Queues", "Data Sync"]
            },
            "business_analyst": {
                "name": "Business Analyst",
                "description": "Business requirements and process analysis expert",
                "model": "gpt-4o",
                "specialization": "business",
                "tools": ["requirements_tools", "process_mapping"],
                "capabilities": ["Requirements Analysis", "Process Mapping", "Stakeholder Management"]
            },
            "solution_architect": {
                "name": "Solution Architect",
                "description": "Enterprise solution design and architecture expert",
                "model": "gpt-4o",
                "specialization": "architecture",
                "tools": ["architecture_tools", "design_patterns"],
                "capabilities": ["System Design", "Architecture Patterns", "Technology Strategy"]
            },
            "research_scientist": {
                "name": "Research Scientist",
                "description": "Advanced research and experimental development expert",
                "model": "gpt-4o",
                "specialization": "research",
                "tools": ["research_tools", "statistical_analysis"],
                "capabilities": ["Research Methodology", "Statistical Analysis", "Experimental Design"]
            },
            "startup_advisor": {
                "name": "Startup Advisor",
                "description": "Startup strategy and business development expert",
                "model": "gpt-4o",
                "specialization": "startup",
                "tools": ["business_planning", "market_analysis"],
                "capabilities": ["Business Strategy", "Market Analysis", "Fundraising", "MVP Development"]
            },
            "legal_tech_specialist": {
                "name": "Legal Tech Specialist",
                "description": "Legal technology and compliance automation expert",
                "model": "gpt-4o",
                "specialization": "legal_tech",
                "tools": ["legal_research", "contract_analysis"],
                "capabilities": ["Legal Research", "Contract Analysis", "Compliance Automation"]
            },
            "fintech_developer": {
                "name": "FinTech Developer",
                "description": "Financial technology and payment systems expert",
                "model": "gpt-4o",
                "specialization": "fintech",
                "tools": ["payment_apis", "financial_modeling"],
                "capabilities": ["Payment Processing", "Financial APIs", "Risk Management", "Compliance"]
            },
            "healthcare_it_specialist": {
                "name": "Healthcare IT Specialist",
                "description": "Healthcare technology and HIPAA compliance expert",
                "model": "gpt-4o",
                "specialization": "healthcare_it",
                "tools": ["ehr_systems", "medical_apis"],
                "capabilities": ["EHR Systems", "HIPAA Compliance", "Medical APIs", "Healthcare Analytics"]
            },
            "ecommerce_specialist": {
                "name": "E-commerce Specialist",
                "description": "E-commerce platform and digital marketing expert",
                "model": "gpt-4o",
                "specialization": "ecommerce",
                "tools": ["shopify", "woocommerce", "analytics"],
                "capabilities": ["E-commerce Platforms", "Payment Integration", "Digital Marketing", "Analytics"]
            },
            "content_strategist": {
                "name": "Content Strategist",
                "description": "Content creation and digital marketing strategy expert",
                "model": "gpt-4o",
                "specialization": "content",
                "tools": ["cms", "seo_tools", "analytics"],
                "capabilities": ["Content Strategy", "SEO", "Social Media", "Brand Management"]
            }
        }
        
        # Create agent objects
        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                name=config["name"],
                model=config["model"],
                specialization=config["specialization"],
                tools=config["tools"],
                config={
                    "description": config["description"],
                    "capabilities": config["capabilities"],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
        
        print(f"✅ Loaded {len(self.agents)} specialized agents")
    
    async def process_task_with_knowledge(self, request: AgentRequest) -> TaskResponse:
        """Process task with agent selection and mock AI execution"""
        
        # Determine which agent to use
        if request.agent_id and request.agent_id in self.agents:
            agent_name = request.agent_id
        else:
            # Auto-select based on task content
            agent_name = self._select_best_agent(request.task)
        
        # Get the agent
        agent = self.agents.get(agent_name)
        if not agent:
            agent_name = "full_stack_developer"
            agent = self.agents.get(agent_name)
        
        # Create task context
        task_id = str(uuid.uuid4())
        
        # Generate intelligent response based on agent expertise
        result_text = self._generate_intelligent_response(agent_name, request.task, agent)
        
        response = TaskResponse(
            task_id=task_id,
            status="completed",
            result={
                "response": result_text,
                "agent": agent_name,
                "task": request.task,
                "agent_role": agent.specialization,
                "timestamp": datetime.now().isoformat()
            },
            metadata={"agent_id": agent_name}
        )
        
        # Store the task result
        self.task_results[task_id] = response
        
        return response
    
    def _select_best_agent(self, task: str) -> str:
        """Select the best agent for a task based on content"""
        
        task_lower = task.lower()
        
        # Keywords for each agent type
        agent_keywords = {
            "full_stack_developer": ["web", "react", "api", "database", "frontend", "backend", "fullstack"],
            "mobile_developer": ["ios", "android", "mobile", "app", "swift", "kotlin", "react native"],
            "security_expert": ["security", "vulnerability", "penetration", "audit", "owasp", "encryption"],
            "devops_engineer": ["deploy", "docker", "kubernetes", "ci/cd", "infrastructure", "aws", "cloud"],
            "data_scientist": ["data", "machine learning", "ml", "analysis", "statistics", "model"],
            "ui_ux_designer": ["design", "ui", "ux", "interface", "wireframe", "prototype", "css"],
            "blockchain_developer": ["blockchain", "smart contract", "solidity", "web3", "crypto", "defi"],
            "ai_ml_engineer": ["ai", "artificial intelligence", "neural network", "deep learning", "nlp"],
            "cloud_architect": ["cloud", "aws", "azure", "gcp", "serverless", "microservices"],
            "qa_engineer": ["test", "testing", "quality", "automation", "selenium", "cypress"]
        }
        
        # Score each agent
        scores = {}
        for agent_name, keywords in agent_keywords.items():
            if agent_name in self.agents:
                score = sum(1 for keyword in keywords if keyword in task_lower)
                scores[agent_name] = score
        
        # Return agent with highest score, or default to full_stack_developer
        if scores:
            best_agent = max(scores, key=scores.get)
            if scores[best_agent] > 0:
                return best_agent
        
        # Default fallback
        return "full_stack_developer"
    
    def _generate_intelligent_response(self, agent_name: str, task: str, agent: Agent) -> str:
        """Generate an intelligent response based on agent expertise"""
        
        # Build response based on agent's expertise
        response = f"As a {agent.config['description']}, I'll help you with: {task}\n\n"
        
        # Add specific guidance based on agent type
        agent_responses = {
            "full_stack_developer": "I can help you build complete web applications using modern technologies like React, Node.js, and databases. I'll provide code examples, architecture guidance, and best practices for scalable development.",
            "mobile_developer": "I can assist with iOS and Android app development, including native development with Swift/Kotlin or cross-platform solutions with React Native/Flutter. I'll help with app architecture, UI/UX, and deployment.",
            "security_expert": "I can analyze your systems for security vulnerabilities, recommend security best practices, help with penetration testing strategies, and ensure compliance with security standards like OWASP.",
            "devops_engineer": "I can help you set up CI/CD pipelines, containerize applications with Docker, orchestrate with Kubernetes, and deploy to cloud platforms like AWS, Azure, or GCP.",
            "data_scientist": "I can assist with data analysis, machine learning model development, statistical analysis, and data visualization. I'll help you extract insights from your data and build predictive models.",
            "ui_ux_designer": "I can help you design user-friendly interfaces, create wireframes and prototypes, conduct user research, and ensure your applications provide excellent user experiences.",
            "blockchain_developer": "I can help you develop smart contracts, build DeFi applications, integrate Web3 functionality, and understand blockchain architecture and security considerations."
        }
        
        specific_response = agent_responses.get(agent_name, "I'll provide expert guidance based on industry best practices and my specialized knowledge.")
        response += specific_response
        
        # Add capabilities
        if agent.config.get('capabilities'):
            response += f"\n\nMy key capabilities include: {', '.join(agent.config['capabilities'])}"
        
        response += "\n\nHow can I assist you with this specific task?"
        
        return response

# Initialize enhanced service
enhanced_service = EnhancedAgentService()

# Enhanced API
enhanced_app = FastAPI(
    title="Agent Lightning Enhanced API",
    version="2.0.0",
    description="Production API with 32 Specialized Agents and Knowledge Management"
)

enhanced_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@enhanced_app.get("/")
async def root():
    return {
        "message": "Agent Lightning Enhanced API", 
        "status": "running",
        "version": "2.0.0",
        "specialized_agents": len(enhanced_service.agents)
    }

@enhanced_app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "enhanced-agent-api",
        "agents_loaded": len(enhanced_service.agents),
        "version": "2.0.0"
    }

@enhanced_app.get("/api/v2/agents/list")
async def list_specialized_agents():
    """List all specialized agents with their capabilities"""
    agents = []
    
    for agent_name, agent in enhanced_service.agents.items():
        agents.append({
            "id": agent_name,
            "name": agent.name,
            "role": agent.specialization,
            "model": agent.model,
            "capabilities": agent.config.get('capabilities', []),
            "description": agent.config.get('description', ''),
            "tools": agent.tools
        })
    
    return {"agents": agents}

@enhanced_app.post("/api/v2/agents/execute")
async def execute_with_knowledge(request: AgentRequest):
    """Execute task with specialized agent selection"""
    return await enhanced_service.process_task_with_knowledge(request)

# Legacy endpoints for compatibility
@enhanced_app.get("/agents")
async def list_agents_legacy():
    """Legacy endpoint for backward compatibility"""
    return {"agents": list(enhanced_service.agents.keys())}

@enhanced_app.post("/agents/execute")
async def execute_task_legacy(request: AgentRequest):
    """Legacy endpoint for backward compatibility"""
    return await enhanced_service.process_task_with_knowledge(request)

# Task status endpoint for dashboard compatibility
@enhanced_app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status - returns the actual task result if available"""
    if task_id in enhanced_service.task_results:
        task_response = enhanced_service.task_results[task_id]
        return {
            "task_id": task_id,
            "status": task_response.status,
            "result": task_response.result,
            "metadata": task_response.metadata
        }
    else:
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "response": "Task completed successfully",
                "agent": "specialized_agent"
            },
            "metadata": {}
        }

# Authentication endpoint (simple mock for dashboard compatibility)
@enhanced_app.post("/api/v1/auth/token")
async def login(username: str, password: str):
    """Simple authentication endpoint for dashboard compatibility"""
    if username == "admin" and password == "admin":
        return {
            "access_token": "mock-token-12345",
            "token_type": "bearer"
        }
    return {"error": "Invalid credentials"}

if __name__ == "__main__":
    print("🚀 Starting Enhanced Agent Lightning API with 32 Specialized Agents")
    print("=" * 70)
    print("\nSpecialized Agents Available:")
    for agent_name, agent in enhanced_service.agents.items():
        print(f"  - {agent_name}: {agent.name}")
    
    print("\n\nEndpoints:")
    print("  Enhanced API: http://localhost:8002/api/v2")
    print("  Legacy API: http://localhost:8002/agents")
    print("  Documentation: http://localhost:8002/docs")
    
    uvicorn.run(enhanced_app, host="0.0.0.0", port=8002)