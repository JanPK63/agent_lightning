"""
LangChain Agent Wrapper for Charter Compliance
Integrates existing Agent Lightning agents with LangChain framework
"""

from typing import Dict, Any, Optional, List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_openai_functions_agent, AgentExecutor
import json
import os
import time

from agent_config import AgentConfig
from memory_manager import MemoryManager
from shared_memory_system import SharedMemorySystem
from langchain_tools import get_agent_tools

# Import Prometheus metrics
from monitoring.metrics import get_metrics_collector


class LangChainAgentWrapper:
    """
    Wraps existing Agent Lightning agents with LangChain framework
    Provides charter-compliant prompt templates and memory management
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.memory_manager = MemoryManager()
        self.shared_memory = SharedMemorySystem()

        # Initialize Prometheus metrics collector
        self.metrics_collector = get_metrics_collector("langchain_agent")

        # Charter-compliant system prompt
        self.charter_prompt = """
Jij bent een AI-agent die software genereert. Je doel is om altijd
betrouwbaar, transparant en iteratief te werken. Je breekt complexe
taken op in de kleinst mogelijke subtaken, bouwt actief kennis en
geheugen op, en levert altijd eerlijke en kwaliteitsvolle output.

Werkprincipes:
1. Taakreductie: Breek opdrachten op in de kleinst mogelijke subtaken
2. Eerlijkheid & Transparantie: Wees eerlijk, vermeld aannames, rapporteer fouten
3. Kennisopbouw: Documenteer keuzes, hergebruik patronen, leer van fouten
4. Geheugenopbouw: Houd keuzes, conventies en afhankelijkheden persistent bij
5. Codekwaliteit: Leesbare, modulaire code, documentatie, tests, coding standards
6. Iteratief werken: Lever MVP, verbeter in kleine iteraties, test & documenteer
7. Samenwerking: Stel vragen, geef alternatieven, vat samen

Jouw specialisatie: {agent_role}
{agent_expertise}
        """

        # Get tools for this agent
        capabilities_dict = {
            "can_write_code": agent_config.capabilities.can_write_code,
            "can_write_documentation": agent_config.capabilities.can_write_documentation,
            "can_debug": agent_config.capabilities.can_debug,
            "can_analyze_data": agent_config.capabilities.can_analyze_data
        }
        self.tools = get_agent_tools(capabilities_dict)

        # Create LangChain prompt template for agent with tools
        if self.tools:
            # Agent with tools needs agent_scratchpad
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.charter_prompt + "\n\nJe hebt toegang tot de volgende tools: {tools}\nGebruik tools wanneer nodig om taken uit te voeren."),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
        else:
            # Simple agent without tools
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.charter_prompt),
                ("human", "{input}"),
                ("assistant", "Ik ga deze taak aanpakken volgens de charter principes. Laat me beginnen met het opdelen in subtaken:")
            ])

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=agent_config.model,
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens
        )

        # Create agent with tools
        if self.tools:
            self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt_template)
            self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        else:
            # Fallback to simple chain if no tools
            self.chain = self.prompt_template | self.llm
            self.agent_executor = None

        # Memory setup
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Session store for RunnableWithMessageHistory
        self.session_store = {}

        # Create conversational chain with memory
        if self.agent_executor:
            # Use agent executor with tools
            self.conversational_chain = RunnableWithMessageHistory(
                self.agent_executor,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
        else:
            # Fallback to simple chain
            self.conversational_chain = RunnableWithMessageHistory(
                self.chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for session"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def _format_agent_expertise(self) -> str:
        """Format agent expertise from knowledge base"""
        kb = self.agent_config.knowledge_base
        expertise = f"""
Expertise gebieden: {', '.join(kb.domains)}
Technologieën: {', '.join(kb.technologies[:10])}  # Limit for prompt size
Frameworks: {', '.join(kb.frameworks[:5])}
        """
        return expertise.strip()

    def invoke(self, input_text: str, session_id: str = "default") -> str:
        """
        Invoke agent with LangChain integration
        """
        try:
            self.metrics_collector.increment_request("invoke", "POST", "200")

            # Format prompt with agent-specific information
            formatted_input = {
                "input": input_text,
                "agent_role": self.agent_config.description,
                "agent_expertise": self._format_agent_expertise(),
                "tools": ", ".join([tool.name for tool in self.tools]) if self.tools else "geen tools beschikbaar"
            }

            # Get response from conversational chain
            response = self.conversational_chain.invoke(
                formatted_input,
                config={"configurable": {"session_id": session_id}}
            )

            # Handle different response types
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict):
                response_text = response.get('output', str(response))
            else:
                response_text = str(response)

            # Store in memory systems
            self._store_interaction(input_text, response_text, session_id)

            return response_text

        except Exception as e:
            error_msg = f"Error in LangChain agent wrapper: {str(e)}"
            print(error_msg)
            self.metrics_collector.increment_error("invoke", type(e).__name__)
            return error_msg

    def _store_interaction(self, input_text: str, response: str, session_id: str):
        """Store interaction in memory systems"""
        try:
            # Store in agent memory as episodic memory
            self.memory_manager.store_episodic(
                content={
                    "agent_id": self.agent_config.name,
                    "input": input_text,
                    "response": response,
                    "session_id": session_id,
                    "task_type": "conversation",
                    "timestamp": time.time()
                },
                importance=0.6
            )

            # Store in shared memory
            self.shared_memory.add_conversation_turn(
                agent_id=self.agent_config.name,
                user_message=input_text,
                agent_response=response
            )

        except Exception as e:
            print(f"Warning: Could not store interaction in memory: {e}")

    def get_memory_context(self, session_id: str = "default") -> str:
        """Get relevant memory context for current session"""
        try:
            self.metrics_collector.increment_request("get_memory_context", "GET", "200")

            # Get conversation history
            history = self._get_session_history(session_id)

            # Get relevant memories from memory manager
            query = {
                "task": "conversation",
                "agent_id": self.agent_config.name,
                "session_id": session_id
            }
            memories = self.memory_manager.retrieve_relevant(query, k=5)

            context = "Relevante geheugencontext:\n"

            # Add recent conversation
            if history.messages:
                context += "Recente conversatie:\n"
                for msg in history.messages[-4:]:  # Last 2 exchanges
                    role = "Gebruiker" if msg.type == "human" else "Agent"
                    context += f"{role}: {msg.content[:100]}...\n"

            # Add relevant memories
            if memories:
                context += "\nRelevante eerdere ervaringen:\n"
                for memory in memories[:3]:
                    content = memory.content
                    summary = content.get('input', 'Geen samenvatting')[:50]
                    context += f"- {summary}...\n"

            return context

        except Exception as e:
            print(f"Warning: Could not retrieve memory context: {e}")
            self.metrics_collector.increment_error("get_memory_context", type(e).__name__)
            return ""

    def clear_session(self, session_id: str = "default"):
        """Clear session history"""
        try:
            self.metrics_collector.increment_request("clear_session", "DELETE", "200")
            if session_id in self.session_store:
                del self.session_store[session_id]
        except Exception as e:
            self.metrics_collector.increment_error("clear_session", type(e).__name__)


class LangChainAgentManager:
    """
    Manages multiple LangChain-wrapped agents
    """

    def __init__(self):
        # Initialize Prometheus metrics collector
        self.metrics_collector = get_metrics_collector("langchain_manager")
        self.agents: Dict[str, LangChainAgentWrapper] = {}
        self.load_agents()

    def load_agents(self):
        """Load all existing agents from database and wrap them with LangChain"""
        from shared.database import db_manager
        from shared.models import Agent
        from agent_config import AgentConfig, AgentRole, KnowledgeBase, AgentCapabilities

        try:
            with db_manager.get_db() as db:
                db_agents = db.query(Agent).all()

                for db_agent in db_agents:
                    try:
                        # Convert database agent to AgentConfig
                        agent_config = AgentConfig(
                            name=db_agent.name.lower().replace(' ', '_'),
                            role=AgentRole.CUSTOM,
                            description=db_agent.specialization or f"{db_agent.name} specialist",
                            model=db_agent.model or "gpt-4o",
                            temperature=0.7,
                            max_tokens=4000,
                            knowledge_base=KnowledgeBase(
                                domains=[db_agent.name],
                                custom_instructions=db_agent.specialization or ""
                            ),
                            capabilities=AgentCapabilities(
                                can_write_code=True,
                                can_debug=True,
                                can_review_code=True,
                                can_optimize=True,
                                can_test=True,
                                can_write_documentation=True
                            ),
                            system_prompt=f"You are a {db_agent.name} with expertise in your domain."
                        )

                        self.agents[agent_config.name] = LangChainAgentWrapper(agent_config)
                        print(f"✅ Loaded LangChain wrapper for {agent_config.name}")

                    except Exception as e:
                        print(f"❌ Failed to load {db_agent.name}: {e}")

        except Exception as e:
            print(f"❌ Failed to load agents from database: {e}")
            # Fallback to file-based agents
            self._load_file_based_agents()

    def _load_file_based_agents(self):
        """Fallback: Load file-based agents"""
        from agent_config import AgentConfigManager

        config_manager = AgentConfigManager()
        agent_configs = config_manager.list_agents()

        for agent_name in agent_configs:
            try:
                agent_config = config_manager.get_agent(agent_name)
                if agent_config:
                    self.agents[agent_name] = LangChainAgentWrapper(agent_config)
                    print(f"✅ Loaded LangChain wrapper for {agent_name} (file-based)")
            except Exception as e:
                print(f"❌ Failed to load {agent_name}: {e}")

    def get_agent(self, agent_name: str) -> Optional[LangChainAgentWrapper]:
        """Get wrapped agent by name"""
        try:
            self.metrics_collector.increment_request("get_agent", "GET", "200")
            return self.agents.get(agent_name)
        except Exception as e:
            self.metrics_collector.increment_error("get_agent", type(e).__name__)
            return None

    def list_agents(self) -> List[str]:
        """List all available wrapped agents"""
        try:
            self.metrics_collector.increment_request("list_agents", "GET", "200")
            return list(self.agents.keys())
        except Exception as e:
            self.metrics_collector.increment_error("list_agents", type(e).__name__)
            return []

    def invoke_agent(self, agent_name: str, input_text: str, session_id: str = "default") -> str:
        """Invoke specific agent with input"""
        try:
            self.metrics_collector.increment_request("invoke_agent", "POST", "200")

            agent = self.get_agent(agent_name)
            if not agent:
                return f"Agent '{agent_name}' not found. Available agents: {', '.join(self.list_agents())}"

            return agent.invoke(input_text, session_id)

        except Exception as e:
            error_msg = f"Error invoking agent {agent_name}: {str(e)}"
            print(error_msg)
            self.metrics_collector.increment_error("invoke_agent", type(e).__name__)
            return error_msg


# Example usage and testing
def test_langchain_integration():
    """Test LangChain integration with existing agents"""

    print("🧪 Testing LangChain Agent Wrapper...")

    # Initialize manager
    manager = LangChainAgentManager()

    if not manager.agents:
        print("❌ No agents loaded")
        return

    # Test with first available agent
    agent_name = list(manager.agents.keys())[0]
    print(f"Testing with agent: {agent_name}")

    # Test basic invocation
    test_input = "Schrijf een Python-functie die Fibonacci berekent en voeg unittests toe"
    response = manager.invoke_agent(agent_name, test_input, "test_session")

    print(f"Input: {test_input}")
    print(f"Response: {response[:200]}...")

    # Test memory retrieval
    agent = manager.get_agent(agent_name)
    if agent:
        memory_context = agent.get_memory_context("test_session")
        print(f"Memory context: {memory_context[:200]}...")

    print("✅ LangChain integration test completed")


if __name__ == "__main__":
    test_langchain_integration()