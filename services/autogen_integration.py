#!/usr/bin/env python3
"""
AutoGen Integration Service - Enhanced multi-agent collaboration
Integrates Microsoft AutoGen for sophisticated agent conversations and group problem-solving
"""

import os
import sys
import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# AutoGen imports - using new import structure
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import GroupChat, GroupChatManager
except ImportError:
    # Fallback to older import structure
    try:
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    except ImportError:
        # Create mock classes for testing without AutoGen
        class AssistantAgent:
            def __init__(self, **kwargs):
                self.name = kwargs.get('name', 'assistant')
                self.system_message = kwargs.get('system_message', '')
                self.chat_messages = {}
            def initiate_chat(self, recipient, **kwargs):
                pass
            def last_message(self):
                return {"content": "Mock response"}
        
        class UserProxyAgent(AssistantAgent):
            pass
        
        class GroupChat:
            def __init__(self, **kwargs):
                self.agents = kwargs.get('agents', [])
                self.messages = []
        
        class GroupChatManager:
            def __init__(self, **kwargs):
                pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent roles in AutoGen"""
    ASSISTANT = "assistant"
    USER_PROXY = "user_proxy"
    CRITIC = "critic"
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"


class ConversationType(str, Enum):
    """Types of agent conversations"""
    TWO_AGENT = "two_agent"
    GROUP_CHAT = "group_chat"
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"


@dataclass
class AutoGenAgentConfig:
    """Configuration for AutoGen agent"""
    name: str
    role: AgentRole
    system_message: str
    llm_config: Dict[str, Any]
    human_input_mode: str = "NEVER"  # NEVER, TERMINATE, or ALWAYS
    max_consecutive_auto_reply: int = 10
    code_execution_config: Optional[Dict] = None
    function_map: Optional[Dict[str, Callable]] = None
    is_termination_msg: Optional[Callable] = None


@dataclass
class ConversationConfig:
    """Configuration for agent conversation"""
    conversation_type: ConversationType
    agents: List[AutoGenAgentConfig]
    max_round: int = 20
    admin_name: Optional[str] = None
    speaker_selection_method: str = "auto"  # auto, manual, random, round_robin
    allow_repeat_speaker: bool = True
    send_introductions: bool = False


class AutoGenAgent:
    """Wrapper for AutoGen agents with our framework integration"""
    
    def __init__(self, config: AutoGenAgentConfig, dal: DataAccessLayer):
        self.config = config
        self.dal = dal
        self.agent = None
        self.conversation_history = []
        self._create_agent()
    
    def _create_agent(self):
        """Create the AutoGen agent based on configuration"""
        if self.config.role == AgentRole.USER_PROXY:
            self.agent = UserProxyAgent(
                name=self.config.name,
                system_message=self.config.system_message,
                human_input_mode=self.config.human_input_mode,
                max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
                code_execution_config=self.config.code_execution_config or False,
                llm_config=self.config.llm_config,
                function_map=self.config.function_map,
                is_termination_msg=self.config.is_termination_msg
            )
        else:
            self.agent = AssistantAgent(
                name=self.config.name,
                system_message=self.config.system_message,
                llm_config=self.config.llm_config,
                max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
                is_termination_msg=self.config.is_termination_msg
            )
    
    def update_system_message(self, message: str):
        """Update agent's system message"""
        self.config.system_message = message
        if hasattr(self.agent, 'update_system_message'):
            self.agent.update_system_message(message)
    
    def add_function(self, name: str, func: Callable):
        """Add a function to the agent's function map"""
        if self.config.function_map is None:
            self.config.function_map = {}
        self.config.function_map[name] = func
        if hasattr(self.agent, 'register_function'):
            self.agent.register_function({name: func})


class ConversationOrchestrator:
    """Orchestrates multi-agent conversations"""
    
    def __init__(self, dal: DataAccessLayer):
        self.dal = dal
        self.active_conversations = {}
        self.agent_registry = {}
        
    def create_agent(self, config: AutoGenAgentConfig) -> AutoGenAgent:
        """Create and register an agent"""
        agent = AutoGenAgent(config, self.dal)
        self.agent_registry[config.name] = agent
        
        # Store in database
        self.dal.create_agent({
            "id": config.name,
            "name": config.name,
            "model": config.llm_config.get("model", "gpt-4"),
            "specialization": config.role.value,
            "config": {
                "system_message": config.system_message,
                "llm_config": config.llm_config,
                "human_input_mode": config.human_input_mode
            }
        })
        
        return agent
    
    async def start_conversation(
        self, 
        config: ConversationConfig,
        initial_message: str
    ) -> str:
        """Start a multi-agent conversation"""
        conversation_id = str(uuid.uuid4())
        
        # Create agents
        agents = []
        for agent_config in config.agents:
            if agent_config.name in self.agent_registry:
                agent = self.agent_registry[agent_config.name]
            else:
                agent = self.create_agent(agent_config)
            agents.append(agent)
        
        # Start conversation based on type
        if config.conversation_type == ConversationType.TWO_AGENT:
            result = await self._two_agent_chat(agents[0], agents[1], initial_message)
        elif config.conversation_type == ConversationType.GROUP_CHAT:
            result = await self._group_chat(agents, config, initial_message)
        elif config.conversation_type == ConversationType.SEQUENTIAL:
            result = await self._sequential_chat(agents, initial_message)
        elif config.conversation_type == ConversationType.HIERARCHICAL:
            result = await self._hierarchical_chat(agents, config, initial_message)
        else:
            result = await self._collaborative_chat(agents, initial_message)
        
        # Store conversation
        self.active_conversations[conversation_id] = {
            "id": conversation_id,
            "config": config,
            "agents": [a.config.name for a in agents],
            "started_at": datetime.now(),
            "result": result
        }
        
        return conversation_id
    
    async def _two_agent_chat(
        self, 
        agent1: AutoGenAgent, 
        agent2: AutoGenAgent,
        message: str
    ) -> Dict[str, Any]:
        """Handle two-agent conversation"""
        try:
            # Initiate chat
            agent1.agent.initiate_chat(
                agent2.agent,
                message=message
            )
            
            # Get conversation history
            history = agent1.agent.chat_messages[agent2.agent]
            
            return {
                "type": "two_agent",
                "agents": [agent1.config.name, agent2.config.name],
                "history": history,
                "summary": self._summarize_conversation(history)
            }
            
        except Exception as e:
            logger.error(f"Two-agent chat failed: {e}")
            return {"error": str(e)}
    
    async def _group_chat(
        self,
        agents: List[AutoGenAgent],
        config: ConversationConfig,
        message: str
    ) -> Dict[str, Any]:
        """Handle group chat with multiple agents"""
        try:
            # Create group chat
            groupchat = GroupChat(
                agents=[a.agent for a in agents],
                messages=[],
                max_round=config.max_round,
                speaker_selection_method=config.speaker_selection_method,
                allow_repeat_speaker=config.allow_repeat_speaker,
                send_introductions=config.send_introductions
            )
            
            # Create manager
            manager = GroupChatManager(
                groupchat=groupchat,
                llm_config=agents[0].config.llm_config,
                is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0
            )
            
            # Start conversation
            agents[0].agent.initiate_chat(
                manager,
                message=message
            )
            
            return {
                "type": "group_chat",
                "agents": [a.config.name for a in agents],
                "history": groupchat.messages,
                "rounds": len(groupchat.messages),
                "summary": self._summarize_conversation(groupchat.messages)
            }
            
        except Exception as e:
            logger.error(f"Group chat failed: {e}")
            return {"error": str(e)}
    
    async def _sequential_chat(
        self,
        agents: List[AutoGenAgent],
        message: str
    ) -> Dict[str, Any]:
        """Handle sequential conversation through agents"""
        history = []
        current_message = message
        
        try:
            for i, agent in enumerate(agents[:-1]):
                next_agent = agents[i + 1]
                
                # Pass message to next agent
                agent.agent.initiate_chat(
                    next_agent.agent,
                    message=current_message,
                    clear_history=False
                )
                
                # Get response
                agent_history = agent.agent.chat_messages[next_agent.agent]
                history.extend(agent_history)
                
                # Update message for next iteration
                if agent_history:
                    current_message = agent_history[-1].get("content", "")
            
            return {
                "type": "sequential",
                "agents": [a.config.name for a in agents],
                "history": history,
                "final_response": current_message,
                "summary": self._summarize_conversation(history)
            }
            
        except Exception as e:
            logger.error(f"Sequential chat failed: {e}")
            return {"error": str(e)}
    
    async def _hierarchical_chat(
        self,
        agents: List[AutoGenAgent],
        config: ConversationConfig,
        message: str
    ) -> Dict[str, Any]:
        """Handle hierarchical conversation with supervisor"""
        try:
            # First agent is supervisor
            supervisor = agents[0]
            workers = agents[1:]
            
            results = []
            
            # Supervisor delegates to workers
            for worker in workers:
                supervisor.agent.initiate_chat(
                    worker.agent,
                    message=f"Please handle this task: {message}"
                )
                
                history = supervisor.agent.chat_messages[worker.agent]
                results.append({
                    "worker": worker.config.name,
                    "result": history[-1].get("content", "") if history else ""
                })
            
            # Supervisor synthesizes results
            synthesis_prompt = f"Based on these worker results, provide a final answer:\n"
            for r in results:
                synthesis_prompt += f"{r['worker']}: {r['result']}\n"
            
            # Use supervisor to synthesize
            final_response = synthesis_prompt  # In practice, would call LLM
            
            return {
                "type": "hierarchical",
                "supervisor": supervisor.config.name,
                "workers": [w.config.name for w in workers],
                "worker_results": results,
                "final_response": final_response
            }
            
        except Exception as e:
            logger.error(f"Hierarchical chat failed: {e}")
            return {"error": str(e)}
    
    async def _collaborative_chat(
        self,
        agents: List[AutoGenAgent],
        message: str
    ) -> Dict[str, Any]:
        """Handle collaborative problem-solving"""
        try:
            # Create specialized agents for collaboration
            planner = next((a for a in agents if a.config.role == AgentRole.PLANNER), agents[0])
            executor = next((a for a in agents if a.config.role == AgentRole.EXECUTOR), agents[1] if len(agents) > 1 else agents[0])
            reviewer = next((a for a in agents if a.config.role == AgentRole.REVIEWER), agents[2] if len(agents) > 2 else agents[0])
            
            # Planning phase
            planner.agent.initiate_chat(
                executor.agent,
                message=f"Let's plan how to solve: {message}"
            )
            plan = planner.agent.last_message()["content"]
            
            # Execution phase
            executor.agent.initiate_chat(
                reviewer.agent,
                message=f"Execute this plan: {plan}"
            )
            execution = executor.agent.last_message()["content"]
            
            # Review phase
            review_result = f"Reviewing execution: {execution}"  # Simplified
            
            return {
                "type": "collaborative",
                "agents": [a.config.name for a in agents],
                "plan": plan,
                "execution": execution,
                "review": review_result
            }
            
        except Exception as e:
            logger.error(f"Collaborative chat failed: {e}")
            return {"error": str(e)}
    
    def _summarize_conversation(self, messages: List[Dict]) -> str:
        """Summarize a conversation"""
        if not messages:
            return "No conversation"
        
        # Simple summary - in practice, use LLM
        num_messages = len(messages)
        participants = set(m.get("name", "Unknown") for m in messages)
        
        return f"Conversation with {num_messages} messages between {', '.join(participants)}"
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation details"""
        return self.active_conversations.get(conversation_id)


class AutoGenIntegrationService:
    """FastAPI service for AutoGen integration"""
    
    def __init__(self):
        self.app = FastAPI(title="AutoGen Integration Service", version="1.0.0")
        
        # Initialize Data Access Layer
        self.dal = DataAccessLayer("autogen_integration")
        
        # Cache
        self.cache = get_cache()
        
        # Conversation orchestrator
        self.orchestrator = ConversationOrchestrator(self.dal)
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Default LLM config
        self.default_llm_config = {
            "model": os.getenv("AUTOGEN_MODEL", "gpt-4"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        logger.info("✅ AutoGen Integration Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
        self._create_default_agents()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _create_default_agents(self):
        """Create default AutoGen agents"""
        # Create a planner agent
        planner_config = AutoGenAgentConfig(
            name="planner",
            role=AgentRole.PLANNER,
            system_message="You are a strategic planner. Break down complex problems into actionable steps.",
            llm_config=self.default_llm_config
        )
        self.orchestrator.create_agent(planner_config)
        
        # Create an executor agent
        executor_config = AutoGenAgentConfig(
            name="executor",
            role=AgentRole.EXECUTOR,
            system_message="You are an executor. Implement the planned steps and produce concrete results.",
            llm_config=self.default_llm_config,
            code_execution_config={"use_docker": False}
        )
        self.orchestrator.create_agent(executor_config)
        
        # Create a critic agent
        critic_config = AutoGenAgentConfig(
            name="critic",
            role=AgentRole.CRITIC,
            system_message="You are a critic. Review work and provide constructive feedback for improvement.",
            llm_config=self.default_llm_config
        )
        self.orchestrator.create_agent(critic_config)
        
        logger.info("Created default AutoGen agents: planner, executor, critic")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "autogen_integration",
                "status": "healthy",
                "agents": list(self.orchestrator.agent_registry.keys()),
                "active_conversations": len(self.orchestrator.active_conversations)
            }
        
        @self.app.post("/agents/create")
        async def create_agent(
            name: str,
            role: AgentRole,
            system_message: str,
            model: str = "gpt-4"
        ):
            """Create a new AutoGen agent"""
            try:
                config = AutoGenAgentConfig(
                    name=name,
                    role=role,
                    system_message=system_message,
                    llm_config={
                        **self.default_llm_config,
                        "model": model
                    }
                )
                
                agent = self.orchestrator.create_agent(config)
                
                return {
                    "agent_name": name,
                    "role": role.value,
                    "status": "created"
                }
                
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/agents")
        async def list_agents():
            """List all registered agents"""
            agents = []
            for name, agent in self.orchestrator.agent_registry.items():
                agents.append({
                    "name": name,
                    "role": agent.config.role.value,
                    "system_message": agent.config.system_message[:100] + "..."
                })
            return {"agents": agents}
        
        @self.app.post("/conversations/start")
        async def start_conversation(
            conversation_type: ConversationType,
            agent_names: List[str],
            initial_message: str,
            max_round: int = 20
        ):
            """Start a multi-agent conversation"""
            try:
                # Get agent configs
                agent_configs = []
                for name in agent_names:
                    if name not in self.orchestrator.agent_registry:
                        raise ValueError(f"Agent {name} not found")
                    agent_configs.append(self.orchestrator.agent_registry[name].config)
                
                # Create conversation config
                conv_config = ConversationConfig(
                    conversation_type=conversation_type,
                    agents=agent_configs,
                    max_round=max_round
                )
                
                # Start conversation
                conversation_id = await self.orchestrator.start_conversation(
                    conv_config,
                    initial_message
                )
                
                return {
                    "conversation_id": conversation_id,
                    "status": "started",
                    "type": conversation_type.value,
                    "agents": agent_names
                }
                
            except Exception as e:
                logger.error(f"Failed to start conversation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/conversations/{conversation_id}")
        async def get_conversation(conversation_id: str):
            """Get conversation details"""
            conversation = self.orchestrator.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            return conversation
        
        @self.app.post("/conversations/group")
        async def start_group_chat(
            agent_names: List[str],
            topic: str,
            max_round: int = 20,
            selection_method: str = "auto"
        ):
            """Start a group chat with multiple agents"""
            try:
                return await start_conversation(
                    conversation_type=ConversationType.GROUP_CHAT,
                    agent_names=agent_names,
                    initial_message=f"Let's discuss: {topic}",
                    max_round=max_round
                )
                
            except Exception as e:
                logger.error(f"Failed to start group chat: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/conversations/solve")
        async def collaborative_solve(
            problem: str,
            agent_names: Optional[List[str]] = None
        ):
            """Collaboratively solve a problem"""
            try:
                # Use default agents if not specified
                if not agent_names:
                    agent_names = ["planner", "executor", "critic"]
                
                return await start_conversation(
                    conversation_type=ConversationType.COLLABORATIVE,
                    agent_names=agent_names,
                    initial_message=problem,
                    max_round=30
                )
                
            except Exception as e:
                logger.error(f"Failed to solve problem: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time conversation updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    async def broadcast_update(self, update: Dict):
        """Broadcast update to all WebSocket connections"""
        for connection in self.websocket_connections:
            try:
                await connection.send_json(update)
            except:
                self.websocket_connections.remove(connection)
    
    async def startup(self):
        """Startup tasks"""
        logger.info("AutoGen Integration Service starting up...")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("AutoGen Integration Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = AutoGenIntegrationService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("AUTOGEN_SERVICE_PORT", 8015))
    logger.info(f"Starting AutoGen Integration Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()