"""
Enhanced Production API with Knowledge Management Integration
This extends the production API to use agent configurations and knowledge bases
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_api import *
from agent_config import AgentConfigManager, AgentConfig
from knowledge_manager import KnowledgeManager
from agent_visual_integration import (
    AgentVisualIntegration, 
    TaskComplexityAnalyzer,
    EnhancedAgentRequest,
    enhance_agent_with_visual
)
from typing import Dict, Optional, Any
import asyncio
import uuid
from fastapi import FastAPI
from dataclasses import dataclass


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
    """Enhanced agent service with knowledge management, shared memory, and visual planning"""
    
    def __init__(self):
        self.agents = {}
        self.config_manager = AgentConfigManager()
        self.knowledge_manager = KnowledgeManager()
        self.visual_integration = AgentVisualIntegration()  # Add visual planning
        self.task_analyzer = TaskComplexityAnalyzer()  # Add task analyzer
        self.task_results = {}  # Store task results for retrieval
        self.action_results = {}  # Store action execution results
        
        # Initialize shared memory system
        try:
            from shared_memory_system import SharedMemorySystem
            self.memory_system = SharedMemorySystem()
        except ImportError:
            print("Warning: Shared memory system not available")
            self.memory_system = None
        
        self._load_specialized_agents()
    
    def _load_specialized_agents(self):
        """Load all configured specialized agents"""
        # First setup the specialized agents if not already done
        try:
            from specialized_agents import setup_all_specialized_agents
            if not self.config_manager.list_agents():
                print("Setting up specialized agents...")
                setup_all_specialized_agents()
        except Exception as e:
            print(f"Error setting up specialized agents: {e}")
        
        # Load all configured agents
        for agent_name in self.config_manager.list_agents():
            agent_config = self.config_manager.get_agent(agent_name)
            if agent_config:
                # Create an enhanced agent with knowledge
                self.agents[agent_name] = self._create_enhanced_agent(agent_config)
                print(f"✅ Loaded specialized agent: {agent_name}")
    
    def _create_enhanced_agent(self, config: AgentConfig) -> Agent:
        """Create an agent with knowledge integration"""
        
        # Get agent's knowledge base
        knowledge_items = self.knowledge_manager.knowledge_bases.get(config.name, [])
        
        # Build enhanced system prompt with knowledge
        enhanced_prompt = config.system_prompt + "\n\n"
        
        if knowledge_items:
            enhanced_prompt += "You have access to the following knowledge:\n\n"
            
            # Group knowledge by category
            categories = {}
            for item in knowledge_items[:20]:  # Limit to top 20 items
                if item.category not in categories:
                    categories[item.category] = []
                categories[item.category].append(item)
            
            for category, items in categories.items():
                enhanced_prompt += f"\n{category.upper()}:\n"
                for item in items[:3]:  # Top 3 per category
                    enhanced_prompt += f"- {item.content[:200]}...\n"
        
        # Create agent with enhanced configuration
        agent = Agent(
            agent_id=config.name,
            name=config.description,
            model=config.model,
            specialization=config.role.value,
            tools=config.tools,
            config={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "system_prompt": enhanced_prompt,
                "capabilities": config.capabilities.__dict__
            }
        )
        
        return agent
    
    async def process_task_with_knowledge(self, request: AgentRequest, execute_code: bool = False) -> TaskResponse:
        """Process a task using agent's knowledge base with actual AI execution"""
        import openai
        import os
        
        # Determine which agent to use
        if request.agent_id and request.agent_id in self.agents:
            agent_name = request.agent_id
        else:
            # Auto-select based on task content
            agent_name = self._select_best_agent(request.task)
        
        # Get the agent configuration
        agent = self.agents.get(agent_name)
        if not agent:
            # Fallback to a default agent if not found
            agent_name = list(self.agents.keys())[0] if self.agents else "full_stack_developer"
            agent = self.agents.get(agent_name)
        
        # Create context with relevant knowledge
        context = self.knowledge_manager.create_context(
            task_id=str(uuid.uuid4()),
            agent_id=agent_name,
            initial_query=request.task
        )
        
        # Check if deployment is requested
        deployment_config = request.context.get("deployment") if hasattr(request, 'context') and request.context else None
        
        # Get relevant knowledge for the task
        relevant_knowledge = self.knowledge_manager.search_knowledge(
            agent_name=agent_name,
            query=request.task,
            limit=5
        )
        
        # Build the enhanced prompt with knowledge
        knowledge_context = ""
        if relevant_knowledge:
            knowledge_context = "\n\nRelevant Knowledge:\n"
            for item in relevant_knowledge[:3]:
                knowledge_context += f"- {item.category}: {item.content[:200]}...\n"
        
        # Add shared memory context
        memory_context = ""
        if self.memory_system:
            try:
                agent_memory_context = self.memory_system.get_agent_context(agent_name)
                if agent_memory_context:
                    memory_context = f"\n\n## Shared Memory Context:\n{agent_memory_context}"
            except Exception as e:
                print(f"Error getting memory context: {e}")
        
        # Build the full prompt with deployment context
        system_prompt = agent.config.get("system_prompt", "") if agent else ""
        
        # Add deployment context if deployment is configured
        deployment_context = ""
        if deployment_config:
            if deployment_config.get("type") == "local":
                deployment_context = f"\n\nYou have the ability to deploy code locally to: {deployment_config.get('path', '/tmp/agent_workspace')}"
            elif deployment_config.get("type") == "ubuntu_server":
                deployment_context = f"\n\nYou have the ability to deploy code to Ubuntu server at: {deployment_config.get('server_ip')}"
            elif deployment_config.get("type") == "aws_ec2":
                deployment_context = f"\n\nYou have the ability to deploy code to AWS EC2 instance in region: {deployment_config.get('region', 'us-east-1')}"
        
        enhanced_prompt = f"{system_prompt}\n{memory_context}{knowledge_context}{deployment_context}\n\nTask: {request.task}"
        
        try:
            # Check if OpenAI API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                # Use actual OpenAI API
                client = openai.OpenAI(api_key=api_key)
                
                # Prepare system message with deployment capabilities
                full_system_prompt = system_prompt
                if deployment_config:
                    full_system_prompt += "\n\nIMPORTANT: You have code deployment capabilities. When asked to implement or deploy code, acknowledge that you will generate and deploy the code to the specified location."
                
                completion = client.chat.completions.create(
                    model=agent.model if agent else "gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": f"{knowledge_context}{deployment_context}\n\nTask: {request.task}"}
                    ],
                    temperature=agent.config.get("temperature", 0.7) if agent else 0.7,
                    max_tokens=agent.config.get("max_tokens", 1000) if agent else 1000
                )
                
                result_text = completion.choices[0].message.content
            else:
                # Fallback to intelligent response without API
                result_text = self._generate_intelligent_response(agent_name, request.task, relevant_knowledge)
            
            # NEW: Use action-based system to ACTUALLY EXECUTE tasks
            action_result = None
            if deployment_config:
                try:
                    # Import action system
                    from agent_actions import AgentActionExecutor, ActionRequest, ActionClassifier, ActionType
                    
                    # Classify the task into a concrete action
                    action_type = ActionClassifier.classify_request(request.task)
                    print(f"🎯 Task classified as action: {action_type.value}")
                    
                    # Create action executor
                    action_executor = AgentActionExecutor()
                    
                    # Determine target path based on task content
                    target_path = deployment_config.get('working_directory', '/home/ubuntu')
                    if 'fabric-api-gateway-modular' in request.task.lower():
                        target_path = '/home/ubuntu/fabric-api-gateway-modular'
                    
                    # Prepare deployment config for action
                    action_deployment_config = {
                        'type': deployment_config.get('type'),
                        'server_ip': deployment_config.get('server_ip'),
                        'username': deployment_config.get('username', 'ubuntu'),
                        'key_path': deployment_config.get('key_path'),
                        'working_directory': deployment_config.get('working_directory', '/home/ubuntu'),
                        'path': deployment_config.get('path')  # for local deployments
                    }
                    
                    # Handle key path properly
                    if action_deployment_config.get('key_path'):
                        key_path = action_deployment_config['key_path']
                        # Remove quotes if present
                        key_path = key_path.strip('"').strip("'")
                        # Expand user directory
                        key_path = os.path.expanduser(key_path)
                        # Make absolute if needed
                        if not os.path.isabs(key_path):
                            # Check common locations
                            from pathlib import Path
                            possible_paths = [
                                Path.cwd() / key_path,
                                Path.home() / key_path,
                                Path.home() / ".ssh" / key_path,
                                Path("/Users/jankootstra") / key_path
                            ]
                            for path in possible_paths:
                                if path.exists():
                                    key_path = str(path)
                                    break
                        action_deployment_config['key_path'] = key_path
                    
                    # Create action request
                    action_request = ActionRequest(
                        action_type=action_type,
                        target_path=target_path,
                        description=request.task,
                        deployment_config=action_deployment_config,
                        parameters={'knowledge': relevant_knowledge} if relevant_knowledge else None
                    )
                    
                    print(f"🚀 Executing action: {action_type.value} on {target_path}")
                    
                    # EXECUTE THE ACTION - THIS IS WHERE REAL WORK HAPPENS!
                    action_result = await action_executor.execute_action(action_request)
                    
                    # Add the ACTUAL EXECUTION RESULTS to the response
                    if action_result.success:
                        result_text += f"\n\n✅ **Action Successfully Executed: {action_result.action_type.value.upper()}**\n"
                        result_text += f"\n{action_result.output}"
                        if action_result.files_affected:
                            result_text += f"\n\n📁 **Files Processed:** {len(action_result.files_affected)} files"
                            for file in action_result.files_affected[:10]:  # Show first 10 files
                                result_text += f"\n  • {file}"
                        if action_result.metadata:
                            result_text += f"\n\n📊 **Additional Info:**"
                            for key, value in action_result.metadata.items():
                                result_text += f"\n  • {key}: {value}"
                    else:
                        result_text += f"\n\n⚠️ **Action Failed:** {action_result.error}\n"
                        result_text += f"\nAction type attempted: {action_result.action_type.value}"
                    
                    print(f"✅ Action completed: success={action_result.success}")
                    
                    # Store action result for later retrieval
                    self.action_results[context.task_id] = action_result
                    
                except Exception as e:
                    print(f"❌ Error executing action: {e}")
                    result_text += f"\n\n⚠️ **Error executing action:** {str(e)}"
            
            response = TaskResponse(
                task_id=context.task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "response": result_text,
                    "agent": agent_name,
                    "knowledge_items_used": len(relevant_knowledge),
                    "task": request.task,
                    "agent_role": agent.specialization if agent else "general",
                    "action_executed": action_result.action_type.value if action_result else None,
                    "action_success": action_result.success if action_result else None
                },
                metadata={"agent_id": agent_name}
            )
            
        except Exception as e:
            # If there's an error, return an error response
            response = TaskResponse(
                task_id=context.task_id,
                status=TaskStatus.FAILED,
                result={
                    "error": str(e),
                    "agent": agent_name,
                    "task": request.task
                },
                error=str(e),
                metadata={"agent_id": agent_name}
            )
        
        # Store the task result for later retrieval
        self.task_results[context.task_id] = response
        
        # Record conversation in shared memory
        if self.memory_system:
            try:
                self.memory_system.add_conversation(
                    agent=agent_name,
                    user_query=request.task,
                    agent_response=response.result.get("response", "") if response.result else str(response.error),
                    task_id=context.task_id,
                    knowledge_used=len(relevant_knowledge),
                    success=(response.status == TaskStatus.COMPLETED),
                    metadata={
                        "deployment": deployment_config,
                        "agent_role": agent.specialization if agent else "general",
                        "action_executed": action_result.action_type.value if action_result else None,
                        "action_files": action_result.files_affected if action_result else []
                    }
                )
            except Exception as e:
                print(f"Error recording to shared memory: {e}")
        
        # Learn from the interaction if successful
        if response.status == TaskStatus.COMPLETED:
            self.knowledge_manager.learn_from_interaction(
                agent_name=agent_name,
                task_id=context.task_id,
                interaction={
                    "successful": True,
                    "problem": request.task,
                    "solution": str(response.result),
                    "confidence": 0.9
                }
            )
        
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
            "blockchain_developer": ["blockchain", "smart contract", "solidity", "web3", "crypto", "defi"]
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
        return "full_stack_developer" if "full_stack_developer" in self.agents else list(self.agents.keys())[0]
    
    def _generate_intelligent_response(self, agent_name: str, task: str, knowledge_items: list) -> str:
        """Generate an intelligent response based on agent expertise and knowledge"""
        agent_config = self.config_manager.get_agent(agent_name)
        
        # Build response based on agent's expertise
        response = f"As a {agent_config.description if agent_config else agent_name}, I'll help you with: {task}\n\n"
        
        if knowledge_items:
            response += "Based on my knowledge base:\n\n"
            for i, item in enumerate(knowledge_items[:3], 1):
                response += f"{i}. From {item.category}:\n"
                response += f"   {item.content[:300]}\n\n"
        
        # Add specific guidance based on agent type
        agent_responses = {
            "full_stack_developer": "I recommend implementing this using modern web technologies with a focus on scalability and maintainability.",
            "mobile_developer": "For mobile implementation, consider cross-platform frameworks for efficiency while maintaining native performance.",
            "security_expert": "Security considerations include authentication, authorization, data encryption, and vulnerability assessment.",
            "devops_engineer": "I suggest containerization, CI/CD pipelines, and infrastructure as code for deployment.",
            "data_scientist": "Data analysis approach should include exploratory analysis, feature engineering, and model validation.",
            "ui_ux_designer": "Focus on user experience with intuitive navigation, responsive design, and accessibility.",
            "blockchain_developer": "Consider smart contract design, gas optimization, and decentralization principles."
        }
        
        response += "\n" + agent_responses.get(agent_name, "I'll provide expert guidance based on best practices.")
        
        return response
    
    async def _execute_analysis(self, agent_name: str, task: str, deployment_config: dict, knowledge_items: list):
        """Execute file reading and analysis using SSH"""
        try:
            from ssh_executor import SSHExecutor, SSHConfig, ServerAnalyzer
            import os
            
            # Use SSH executor for ubuntu_server deployments
            if deployment_config.get("type") == "ubuntu_server":
                # Create SSH config
                ssh_config = SSHConfig(
                    host=deployment_config.get("server_ip"),
                    username=deployment_config.get("username", "ubuntu"),
                    key_path=deployment_config.get("key_path"),
                    working_directory=deployment_config.get("working_directory", "/home/ubuntu")
                )
                
                # Connect and analyze
                executor = SSHExecutor(ssh_config)
                if executor.connect():
                    analyzer = ServerAnalyzer(executor)
                    
                    # Perform full analysis
                    analysis = analyzer.full_analysis()
                    summary = analyzer.generate_summary(analysis)
                    
                    executor.close()
                    
                    return {
                        "success": True,
                        "message": summary,
                        "files_analyzed": list(analysis.get("projects", {}).keys()),
                        "raw_analysis": analysis
                    }
                else:
                    return {
                        "success": False,
                        "message": "Failed to connect to server"
                    }
            
            elif deployment_config.get("type") == "local":
                from code_executor import ExecutionConfig, ExecutionMode, CodeExecutor, FileOperation
                from pathlib import Path
                
                exec_config = ExecutionConfig(
                    mode=ExecutionMode.LOCAL,
                    working_directory=deployment_config.get("path", "/tmp/agent_workspace")
                )
            else:
                return None
            
            executor = CodeExecutor(exec_config)
            files_analyzed = []
            
            # List files in the directory
            try:
                if exec_config.mode == ExecutionMode.LOCAL:
                    path = Path(exec_config.working_directory)
                    if path.exists():
                        # Get list of relevant files
                        files = [f for f in path.rglob("*") if f.is_file() and f.suffix in ['.py', '.js', '.ts', '.java', '.go', '.rb', '.php', '.sql', '.json', '.yaml', '.yml', '.md', '.txt']]
                        files = files[:10]  # Limit to first 10 files for analysis
                        
                        # Read each file
                        file_contents = {}
                        for file_path in files:
                            try:
                                relative_path = file_path.relative_to(path)
                                result = await executor.file_operation(
                                    FileOperation.READ,
                                    str(relative_path),
                                    ""
                                )
                                if result.success:
                                    file_contents[str(relative_path)] = file_path.read_text()[:1000]  # First 1000 chars
                                    files_analyzed.append(str(relative_path))
                            except:
                                pass
                        
                        # Provide summary
                        if files_analyzed:
                            return {
                                "success": True,
                                "message": f"Successfully analyzed {len(files_analyzed)} files in {exec_config.working_directory}",
                                "files_analyzed": files_analyzed,
                                "summary": f"Found {len(files)} code files. Main technologies detected based on file extensions."
                            }
                elif exec_config.mode == ExecutionMode.REMOTE_SSH:
                    # For remote, use ls command
                    result = await executor.execute_command("ls -la")
                    if result.success:
                        files_analyzed.append("Remote directory listing")
                        return {
                            "success": True,
                            "message": f"Successfully accessed remote server at {deployment_config.get('server_ip')}",
                            "files_analyzed": files_analyzed,
                            "directory_listing": result.output[:500]
                        }
                
            finally:
                executor.close()
                
            return {
                "success": False,
                "message": "No files found to analyze"
            }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Analysis error: {str(e)}"
            }
    
    async def _execute_code_deployment(self, agent_name: str, task: str, deployment_config: dict, knowledge_items: list):
        """Execute code generation and deployment"""
        try:
            from agent_code_integration import IntegratedAgentExecutor
            from code_executor import ExecutionConfig, ExecutionMode
            from aws_deployment import get_ubuntu_server_config, AWSConfig, AWSDeploymentManager
            from agent_actions import AgentActionExecutor, ActionRequest, ActionClassifier, ActionType
            from pathlib import Path
            import os
            
            # Determine execution config based on deployment type
            if deployment_config.get("type") == "local":
                exec_config = ExecutionConfig(
                    mode=ExecutionMode.LOCAL,
                    working_directory=deployment_config.get("path", "/tmp/agent_workspace")
                )
            elif deployment_config.get("type") == "ubuntu_server":
                # Handle key path - expand user home and make absolute
                key_path = deployment_config.get("key_path")
                if key_path:
                    # Remove quotes if present
                    key_path = key_path.strip('"').strip("'")
                    # If it's just a filename, check common locations
                    if not os.path.isabs(key_path):
                        possible_paths = [
                            Path.cwd() / key_path,
                            Path.home() / key_path,
                            Path.home() / ".ssh" / key_path,
                        ]
                        for path in possible_paths:
                            if path.exists():
                                key_path = str(path)
                                break
                    # Expand ~ to home directory
                    key_path = os.path.expanduser(key_path)
                
                exec_config = get_ubuntu_server_config(
                    server_ip=deployment_config.get("server_ip"),
                    username=deployment_config.get("username", "ubuntu"),
                    key_path=key_path
                )
            elif deployment_config.get("type") == "aws_ec2":
                # Handle AWS deployment
                aws_config = AWSConfig(
                    region=deployment_config.get("region", "us-east-1"),
                    instance_type=deployment_config.get("instance_type", "t2.micro"),
                    key_name=deployment_config.get("key_name")
                )
                manager = AWSDeploymentManager(aws_config)
                instance_info = manager.get_or_create_instance()
                exec_config = manager.get_executor_config(instance_info)
            else:
                return None
            
            # Create integrated executor
            executor = IntegratedAgentExecutor(agent_name, exec_config)
            
            # Execute the task with deployment
            result = await executor.execute_task(
                task,
                deploy_to_aws=(deployment_config.get("type") == "aws_ec2")
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": f"Code successfully deployed to {deployment_config.get('type')}",
                    "files_created": list(result["generated_files"].keys()),
                    "deployment_path": deployment_config.get("path") or deployment_config.get("working_directory")
                }
            else:
                return {
                    "success": False,
                    "message": f"Deployment failed: {result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Deployment error: {str(e)}"
            }


# Enhanced API with new endpoints
enhanced_app = FastAPI(
    title="Agent Lightning Enhanced API",
    version="2.0.0",
    description="Production API with Knowledge Management"
)

# Include all original routes
enhanced_app.mount("/original", app)

# Initialize enhanced service
enhanced_service = EnhancedAgentService()


@enhanced_app.post("/api/v2/agents/execute")
async def execute_with_knowledge(request: AgentRequest):
    """Execute task with knowledge-enhanced agents"""
    return await enhanced_service.process_task_with_knowledge(request)


@enhanced_app.post("/api/v2/agents/execute-visual")
async def execute_with_visual_planning(
    task: str,
    agent_id: Optional[str] = None,
    use_visual: Optional[bool] = None,  # None=auto, True=force, False=skip
    debug: bool = False,
    deployment_config: Optional[Dict[str, Any]] = None
):
    """Execute task with intelligent visual planning
    
    The system automatically decides whether to use visual planning based on:
    - Task complexity (simple, moderate, complex, architectural)
    - Number of components detected (api, database, auth, etc.)
    - Estimated lines of code
    
    Visual planning is:
    - REQUIRED for architectural tasks (500+ lines, multiple services)
    - RECOMMENDED for complex tasks (200+ lines, multiple modules)
    - OPTIONAL for moderate tasks (50-200 lines)
    - SKIPPED for simple tasks (<50 lines, single function)
    """
    
    # Analyze task complexity first
    analysis = enhanced_service.task_analyzer.analyze_task(task)
    
    # Create enhanced request
    enhanced_request = EnhancedAgentRequest(
        task=task,
        agent_id=agent_id,
        use_visual=use_visual,
        debug_visual=debug,
        deployment_config=deployment_config
    )
    
    # Process with visual planning integration
    response = await enhance_agent_with_visual(enhanced_service, enhanced_request)
    
    # Add analysis info to response
    if hasattr(response, 'result') and isinstance(response.result, dict):
        response.result['task_analysis'] = {
            'complexity': analysis.complexity.value,
            'estimated_lines': analysis.estimated_lines,
            'components': analysis.detected_components,
            'visual_decision': analysis.visual_decision.value,
            'confidence': analysis.confidence,
            'reasoning': analysis.reasoning
        }
    
    return response


@enhanced_app.post("/api/v2/agents/analyze-task")
async def analyze_task_complexity(task: str):
    """Analyze task complexity to determine if visual planning would be used
    
    Returns complexity analysis without executing the task.
    Useful for understanding how the system would handle different tasks.
    """
    analysis = enhanced_service.task_analyzer.analyze_task(task)
    
    return {
        "task": task,
        "complexity": analysis.complexity.value,
        "estimated_lines": analysis.estimated_lines,
        "detected_components": analysis.detected_components,
        "visual_decision": analysis.visual_decision.value,
        "reasoning": analysis.reasoning,
        "confidence": analysis.confidence,
        "recommendation": {
            "use_visual": analysis.visual_decision.value in ["required", "recommended"],
            "explanation": f"For {analysis.complexity.value} tasks with {len(analysis.detected_components)} components, "
                         f"visual planning is {analysis.visual_decision.value}. {analysis.reasoning}"
        }
    }


@enhanced_app.get("/api/v2/visual/statistics")
async def get_visual_planning_statistics():
    """Get statistics about visual planning usage"""
    stats = enhanced_service.visual_integration.get_statistics()
    return {
        "statistics": stats,
        "message": f"Visual planning used for {stats.get('visual_planning_rate', '0%')} of processed tasks"
    }


@enhanced_app.get("/api/v2/agents/list")
async def list_specialized_agents():
    """List all specialized agents with their capabilities"""
    agents = []
    
    for agent_name in enhanced_service.config_manager.list_agents():
        config = enhanced_service.config_manager.get_agent(agent_name)
        if config:
            agents.append({
                "id": agent_name,
                "name": config.description,
                "role": config.role.value,
                "model": config.model,
                "capabilities": [k for k, v in config.capabilities.__dict__.items() if v],
                "knowledge_items": len(enhanced_service.knowledge_manager.knowledge_bases.get(agent_name, [])),
                "domains": config.knowledge_base.domains[:5]  # First 5 domains
            })
    
    return {"agents": agents}


@enhanced_app.post("/api/v2/knowledge/add")
async def add_knowledge(
    agent_id: str,
    category: str,
    content: str,
    source: str = "api"
):
    """Add knowledge to an agent"""
    item = enhanced_service.knowledge_manager.add_knowledge(
        agent_name=agent_id,
        category=category,
        content=content,
        source=source
    )
    
    # Reload the agent with new knowledge
    config = enhanced_service.config_manager.get_agent(agent_id)
    if config:
        enhanced_service.agents[agent_id] = enhanced_service._create_enhanced_agent(config)
    
    return {
        "success": True,
        "item_id": item.id,
        "message": f"Knowledge added to {agent_id}"
    }


@enhanced_app.get("/api/v2/knowledge/search")
async def search_knowledge(
    agent_id: str,
    query: str,
    category: Optional[str] = None,
    limit: int = 10
):
    """Search an agent's knowledge base"""
    results = enhanced_service.knowledge_manager.search_knowledge(
        agent_name=agent_id,
        query=query,
        category=category,
        limit=limit
    )
    
    return {
        "agent": agent_id,
        "query": query,
        "results": [
            {
                "category": item.category,
                "content": item.content[:500],  # Truncate for response
                "source": item.source,
                "relevance": item.relevance_score,
                "usage_count": item.usage_count
            }
            for item in results
        ]
    }


@enhanced_app.get("/api/v2/knowledge/stats/{agent_id}")
async def get_knowledge_stats(agent_id: str):
    """Get knowledge statistics for an agent"""
    stats = enhanced_service.knowledge_manager.get_statistics(agent_id)
    return stats


@enhanced_app.post("/api/v2/knowledge/train/{agent_id}")
async def train_agent(agent_id: str, force_all: bool = False):
    """Make an agent consume and train on new knowledge"""
    try:
        from knowledge_trainer import KnowledgeTrainer
        trainer = KnowledgeTrainer()
        result = trainer.consume_knowledge(agent_id, force_all)
        
        return {
            "success": len(result.errors) == 0,
            "agent": result.agent_name,
            "knowledge_consumed": result.knowledge_consumed,
            "knowledge_integrated": result.knowledge_integrated,
            "processing_time": result.processing_time,
            "improvements": result.improvements,
            "errors": result.errors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@enhanced_app.post("/api/v2/knowledge/train-all")
async def train_all_agents():
    """Train all agents with their new knowledge"""
    try:
        from knowledge_trainer import KnowledgeTrainer
        trainer = KnowledgeTrainer()
        results = trainer.auto_consume_all_agents()
        
        return {
            "success": True,
            "agents_trained": len(results),
            "total_consumed": sum(r.knowledge_consumed for r in results.values()),
            "total_integrated": sum(r.knowledge_integrated for r in results.values()),
            "details": {
                name: {
                    "consumed": r.knowledge_consumed,
                    "integrated": r.knowledge_integrated,
                    "errors": r.errors
                }
                for name, r in results.items()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@enhanced_app.get("/api/v2/memory/context/{agent_id}")
async def get_agent_memory_context(agent_id: str):
    """Get memory context for an agent"""
    if enhanced_service.memory_system:
        context = enhanced_service.memory_system.get_agent_context(agent_id)
        return {"agent": agent_id, "context": context}
    return {"error": "Memory system not available"}


@enhanced_app.get("/api/v2/memory/status")
async def get_memory_status():
    """Get project status report from memory"""
    if enhanced_service.memory_system:
        report = enhanced_service.memory_system.get_project_status_report()
        summary = enhanced_service.memory_system.get_conversation_summary(24)
        return {
            "report": report,
            "summary": summary,
            "conversations_24h": summary.get("total_conversations", 0)
        }
    return {"error": "Memory system not available"}


@enhanced_app.get("/api/v2/actions/{task_id}")
async def get_action_result(task_id: str):
    """Get the actual action execution result for a task"""
    if task_id in enhanced_service.action_results:
        action_result = enhanced_service.action_results[task_id]
        return {
            "task_id": task_id,
            "action_type": action_result.action_type.value,
            "success": action_result.success,
            "output": action_result.output,
            "files_affected": action_result.files_affected,
            "error": action_result.error,
            "metadata": action_result.metadata
        }
    return {"error": f"No action result found for task {task_id}"}


@enhanced_app.post("/api/v2/memory/project/update")
async def update_project_memory(project_name: str, updates: Dict[str, Any]):
    """Update project memory"""
    if enhanced_service.memory_system:
        enhanced_service.memory_system.update_project_memory(project_name, **updates)
        return {"success": True, "project": project_name}
    return {"error": "Memory system not available"}


@enhanced_app.post("/api/v2/memory/share-learning")
async def share_learning():
    """Share learnings across all agents"""
    if enhanced_service.memory_system:
        enhanced_service.memory_system.share_learning_across_agents()
        return {"success": True, "message": "Learnings shared across agents"}
    return {"error": "Memory system not available"}


# Task status endpoint for dashboard compatibility
@enhanced_app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status - returns the actual task result if available"""
    # Check if we have the task result stored
    if task_id in enhanced_service.task_results:
        task_response = enhanced_service.task_results[task_id]
        return {
            "task_id": task_id,
            "status": task_response.status,
            "result": task_response.result,
            "metadata": task_response.metadata
        }
    else:
        # Return a default completed status for unknown tasks
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
    # In production, implement proper authentication
    if username == "admin" and password == "admin":
        return {
            "access_token": "mock-token-12345",
            "token_type": "bearer"
        }
    return {"error": "Invalid credentials"}


# Health check
@enhanced_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "specialized_agents": len(enhanced_service.config_manager.list_agents()),
        "total_knowledge_items": sum(
            len(items) for items in enhanced_service.knowledge_manager.knowledge_bases.values()
        )
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced Agent Lightning API with Knowledge Management")
    print("=" * 60)
    print("\nSpecialized Agents Available:")
    for agent in enhanced_service.config_manager.list_agents():
        config = enhanced_service.config_manager.get_agent(agent)
        if config:
            print(f"  - {agent}: {config.description}")
    
    print("\n\nEndpoints:")
    print("  Original API: http://localhost:8002/original")
    print("  Enhanced API: http://localhost:8002/api/v2")
    print("  Documentation: http://localhost:8002/docs")
    print("\nTo run:")
    print("  uvicorn enhanced_production_api:enhanced_app --reload --port 8002")
    
    uvicorn.run(enhanced_app, host="0.0.0.0", port=8002)