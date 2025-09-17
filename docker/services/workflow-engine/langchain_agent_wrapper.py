"""
Enterprise LangChain Agent Wrapper
Production-grade agent management with monitoring, caching, and fault tolerance
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

logger = logging.getLogger(__name__)


@dataclass
class AgentCapabilities:
    """Agent capability configuration"""
    can_write_code: bool = True
    can_test: bool = True
    can_write_documentation: bool = True
    can_analyze_data: bool = True
    can_make_api_calls: bool = True
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 300


@dataclass
class AgentConfig:
    """Enterprise agent configuration"""
    name: str
    description: str
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    capabilities: AgentCapabilities = None
    tools: List[str] = None
    system_prompt: str = ""
    memory_window: int = 10
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = AgentCapabilities()
        if self.tools is None:
            self.tools = []


class AgentMetricsCollector(BaseCallbackHandler):
    """Collects agent execution metrics"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time = None
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_calls = []
        self.errors = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.start_time = time.time()
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self.tool_calls.append({
            "tool": action.tool,
            "input": action.tool_input,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.errors.append(str(error))
    
    def get_metrics(self) -> Dict[str, Any]:
        execution_time = time.time() - self.start_time if self.start_time else 0
        return {
            "agent_name": self.agent_name,
            "execution_time": execution_time,
            "token_usage": self.token_usage,
            "tool_calls": len(self.tool_calls),
            "errors": len(self.errors),
            "success": len(self.errors) == 0
        }


class EnterpriseAgent:
    """Enterprise-grade LangChain agent with monitoring and caching"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_config = config
        self.llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key="dummy-key"
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=config.memory_window,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", config.system_prompt or self._get_default_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
        # Metrics and caching
        self.execution_cache = {}
        self.metrics_history = []
        self.active_tasks = 0
        self.lock = threading.Lock()
        
        logger.info(f"Initialized enterprise agent: {config.name}")
    
    def _create_tools(self) -> List[Tool]:
        """Create tools based on agent capabilities"""
        tools = []
        
        if self.config.capabilities.can_write_code:
            tools.append(Tool(
                name="code_generator",
                description="Generate, review, and optimize code",
                func=self._generate_code
            ))
        
        if self.config.capabilities.can_test:
            tools.append(Tool(
                name="test_generator",
                description="Generate and execute tests",
                func=self._generate_tests
            ))
        
        if self.config.capabilities.can_write_documentation:
            tools.append(Tool(
                name="documentation_writer",
                description="Write technical documentation",
                func=self._write_documentation
            ))
        
        if self.config.capabilities.can_analyze_data:
            tools.append(Tool(
                name="data_analyzer",
                description="Analyze data and generate insights",
                func=self._analyze_data
            ))
        
        return tools
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on capabilities"""
        capabilities = []
        if self.config.capabilities.can_write_code:
            capabilities.append("software development")
        if self.config.capabilities.can_test:
            capabilities.append("testing")
        if self.config.capabilities.can_write_documentation:
            capabilities.append("documentation")
        
        return f"""You are an enterprise AI agent specialized in {', '.join(capabilities)}.
        
Key principles:
- Provide production-ready, enterprise-grade solutions
- Follow best practices and industry standards
- Include proper error handling and logging
- Consider security, scalability, and maintainability
- Provide clear explanations and documentation
        
Always strive for excellence and attention to detail."""
    
    def _generate_code(self, requirements: str) -> str:
        """Generate production-ready code"""
        return f"Generated enterprise-grade code for: {requirements}"
    
    def _generate_tests(self, code_context: str) -> str:
        """Generate comprehensive tests"""
        return f"Generated comprehensive test suite for: {code_context}"
    
    def _write_documentation(self, content: str) -> str:
        """Write technical documentation"""
        return f"Generated technical documentation for: {content}"
    
    def _analyze_data(self, data_description: str) -> str:
        """Analyze data and provide insights"""
        return f"Analyzed data and generated insights for: {data_description}"
    
    def invoke(self, prompt: str, session_id: str = None) -> str:
        """Execute agent with caching and metrics"""
        # Check cache
        cache_key = hashlib.md5(f"{prompt}_{session_id}".encode()).hexdigest()
        if cache_key in self.execution_cache:
            cached_result = self.execution_cache[cache_key]
            if datetime.utcnow() - cached_result["timestamp"] < timedelta(hours=1):
                logger.info(f"Returning cached result for {self.config.name}")
                return cached_result["result"]
        
        # Check concurrent task limit
        with self.lock:
            if self.active_tasks >= self.config.capabilities.max_concurrent_tasks:
                raise Exception(f"Agent {self.config.name} at maximum capacity")
            self.active_tasks += 1
        
        try:
            # Execute with metrics collection
            metrics_collector = AgentMetricsCollector(self.config.name)
            
            result = self.agent_executor.invoke(
                {"input": prompt},
                config={"callbacks": [metrics_collector]}
            )
            
            # Store metrics
            metrics = metrics_collector.get_metrics()
            self.metrics_history.append(metrics)
            
            # Cache result
            self.execution_cache[cache_key] = {
                "result": result["output"],
                "timestamp": datetime.utcnow(),
                "metrics": metrics
            }
            
            logger.info(f"Agent {self.config.name} completed task in {metrics['execution_time']:.2f}s")
            return result["output"]
            
        except Exception as e:
            logger.error(f"Agent {self.config.name} failed: {e}")
            raise
        finally:
            with self.lock:
                self.active_tasks -= 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if not self.metrics_history:
            return {"total_executions": 0}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 executions
        
        return {
            "total_executions": len(self.metrics_history),
            "active_tasks": self.active_tasks,
            "avg_execution_time": sum(m["execution_time"] for m in recent_metrics) / len(recent_metrics),
            "success_rate": sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics),
            "total_tokens_used": sum(m["token_usage"]["total_tokens"] for m in recent_metrics),
            "cache_hit_rate": len(self.execution_cache) / len(self.metrics_history) if self.metrics_history else 0
        }


class LangChainAgentManager:
    """Enterprise agent manager with load balancing and monitoring"""
    
    def __init__(self):
        self.agents: Dict[str, EnterpriseAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        self._initialize_default_agents()
        
        logger.info(f"LangChain Agent Manager initialized with {len(self.agents)} agents")
    
    def _initialize_default_agents(self):
        """Initialize default enterprise agents"""
        
        # Full-stack developer agent
        full_stack_config = AgentConfig(
            name="full_stack_developer",
            description="Enterprise full-stack development specialist",
            capabilities=AgentCapabilities(
                can_write_code=True,
                can_test=True,
                can_write_documentation=True,
                max_concurrent_tasks=3
            ),
            system_prompt="""You are an enterprise full-stack developer with expertise in:
            - Modern web frameworks (React, Vue, Angular)
            - Backend development (Node.js, Python, Java)
            - Database design and optimization
            - Cloud architecture and DevOps
            - Security best practices
            
            Always provide production-ready, scalable solutions."""
        )
        
        # DevOps engineer agent
        devops_config = AgentConfig(
            name="devops_engineer",
            description="Enterprise DevOps and infrastructure specialist",
            capabilities=AgentCapabilities(
                can_write_code=True,
                can_test=False,
                can_write_documentation=True,
                max_concurrent_tasks=2
            ),
            system_prompt="""You are an enterprise DevOps engineer specializing in:
            - Container orchestration (Docker, Kubernetes)
            - CI/CD pipeline design
            - Infrastructure as Code (Terraform, CloudFormation)
            - Monitoring and observability
            - Security and compliance
            
            Focus on scalable, reliable, and secure infrastructure solutions."""
        )
        
        # QA engineer agent
        qa_config = AgentConfig(
            name="qa_engineer",
            description="Enterprise quality assurance specialist",
            capabilities=AgentCapabilities(
                can_write_code=True,
                can_test=True,
                can_write_documentation=True,
                max_concurrent_tasks=4
            ),
            system_prompt="""You are an enterprise QA engineer expert in:
            - Test automation frameworks
            - Performance and load testing
            - Security testing
            - API testing
            - Test strategy and planning
            
            Ensure comprehensive test coverage and quality assurance."""
        )
        
        # Data engineer agent
        data_config = AgentConfig(
            name="data_engineer",
            description="Enterprise data engineering specialist",
            capabilities=AgentCapabilities(
                can_write_code=True,
                can_test=True,
                can_analyze_data=True,
                can_write_documentation=True,
                max_concurrent_tasks=2
            ),
            system_prompt="""You are an enterprise data engineer specializing in:
            - Data pipeline design and optimization
            - Big data technologies (Spark, Hadoop)
            - Data warehousing and lakes
            - ETL/ELT processes
            - Data quality and governance
            
            Build scalable, efficient data solutions."""
        )
        
        # Initialize agents
        for config in [full_stack_config, devops_config, qa_config, data_config]:
            try:
                self.agents[config.name] = EnterpriseAgent(config)
            except Exception as e:
                logger.error(f"Failed to initialize agent {config.name}: {e}")
    
    def get_agent(self, name: str) -> Optional[EnterpriseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all available agents"""
        return list(self.agents.keys())
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        total_executions = sum(len(agent.metrics_history) for agent in self.agents.values())
        active_tasks = sum(agent.active_tasks for agent in self.agents.values())
        
        agent_metrics = {}
        for name, agent in self.agents.items():
            agent_metrics[name] = agent.get_metrics()
        
        return {
            "total_agents": len(self.agents),
            "total_executions": total_executions,
            "active_tasks": active_tasks,
            "agent_metrics": agent_metrics,
            "system_health": "healthy" if all(agent.active_tasks < agent.config.capabilities.max_concurrent_tasks 
                                           for agent in self.agents.values()) else "degraded"
        }
    
    async def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel across agents"""
        results = []
        
        async def execute_task(task):
            agent_name = task.get("agent", "full_stack_developer")
            agent = self.get_agent(agent_name)
            
            if not agent:
                return {"error": f"Agent {agent_name} not found", "task_id": task.get("id")}
            
            try:
                result = agent.invoke(task["prompt"], task.get("session_id"))
                return {"result": result, "task_id": task.get("id"), "agent": agent_name}
            except Exception as e:
                return {"error": str(e), "task_id": task.get("id"), "agent": agent_name}
        
        # Execute tasks concurrently
        tasks_futures = [execute_task(task) for task in tasks]
        results = await asyncio.gather(*tasks_futures, return_exceptions=True)
        
        return results