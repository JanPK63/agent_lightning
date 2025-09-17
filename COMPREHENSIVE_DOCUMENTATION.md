# Agent Lightning - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Action-Based Execution System](#action-based-execution-system)
5. [Agent Management](#agent-management)
6. [Knowledge Management](#knowledge-management)
7. [Memory System](#memory-system)
8. [Deployment Configuration](#deployment-configuration)
9. [API Reference](#api-reference)
10. [Dashboard Interface](#dashboard-interface)
11. [SSH Execution](#ssh-execution)
12. [Installation & Setup](#installation-setup)
13. [Usage Examples](#usage-examples)
14. [Troubleshooting](#troubleshooting)

---

## Overview

Agent Lightning is an AI agent factory that enables multiple specialized agents to execute real tasks on local and remote servers. Unlike traditional AI assistants that only describe what they would do, Agent Lightning agents **actually execute** tasks through SSH connections, file operations, and code deployment.

### Key Features
- **Action-Based Execution**: Agents perform real work, not just descriptions
- **Multiple Specialized Agents**: Full-stack, mobile, security, DevOps, data science, UI/UX, blockchain, database specialists
- **Remote Server Operations**: SSH connectivity to Ubuntu servers and AWS EC2 instances
- **Knowledge Management**: Active training and knowledge consumption for agents
- **Shared Memory System**: Agents share context and learn from each other
- **Project Configuration**: Save deployment settings per project
- **Real-Time Monitoring**: Dashboard for tracking agent performance and tasks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard (8501)  │  API Endpoints (8002/8001)   │
├─────────────────────────────────────────────────────────────┤
│                    Agent Management Layer                     │
├──────────────────┬──────────────────┬───────────────────────┤
│ Agent Config     │ Knowledge Manager│ Memory System         │
│ Manager          │                  │                       │
├──────────────────┴──────────────────┴───────────────────────┤
│                 Action Execution Layer                       │
├──────────────────┬──────────────────┬───────────────────────┤
│ Action Classifier│ Action Executor  │ SSH Executor          │
├──────────────────┴──────────────────┴───────────────────────┤
│                  Infrastructure Layer                        │
├──────────────────┬──────────────────┬───────────────────────┤
│ Local Execution  │ Ubuntu Servers   │ AWS EC2              │
└──────────────────┴──────────────────┴───────────────────────┘
```

---

## Core Components

### 1. Enhanced Production API (`enhanced_production_api.py`)
The main API server that orchestrates all agent operations.

**Key Responsibilities:**
- Agent task routing and execution
- Knowledge integration
- Memory management
- Action execution coordination

**Startup:**
```bash
uvicorn enhanced_production_api:enhanced_app --reload --port 8002 --host 0.0.0.0
```

### 2. Monitoring Dashboard (`monitoring_dashboard.py`)
Streamlit-based dashboard for managing and monitoring agents.

**Features:**
- Real-time agent performance metrics
- Task submission interface
- Project configuration management
- Knowledge management UI
- Agent training interface
- System logs and history

**Startup:**
```bash
streamlit run monitoring_dashboard.py --server.port 8501
```

### 3. Agent Configuration (`agent_config.py`)
Manages agent definitions and capabilities.

**Agent Types:**
- `full_stack_developer`: Web development, APIs, databases
- `mobile_developer`: iOS/Android app development
- `security_expert`: Security audits, penetration testing
- `devops_engineer`: Deployment, CI/CD, infrastructure
- `data_scientist`: Data analysis, ML models
- `ui_ux_designer`: Design, prototypes, CSS
- `blockchain_developer`: Smart contracts, Web3
- `database_specialist`: SQL/NoSQL optimization
- `system_architect`: System design, analysis
- `information_analyst`: Data analysis, reporting

---

## Action-Based Execution System

### Core Concept
The action system ensures agents execute real tasks instead of describing hypothetical actions.

### Action Types (`agent_actions.py`)

#### 1. READ_ANALYZE
**Purpose:** Read and analyze files/code on servers
```python
ActionType.READ_ANALYZE
# Reads files, analyzes project structure, identifies technologies
```

**Execution:**
- Connects via SSH
- Navigates to target directory
- Reads file contents
- Analyzes code patterns
- Returns actual file contents and analysis

#### 2. CREATE_CODE
**Purpose:** Generate and write new code files
```python
ActionType.CREATE_CODE
# Creates new files, writes code, sets permissions
```

**Execution:**
- Creates directories if needed
- Writes code to files
- Sets executable permissions
- Returns list of created files

#### 3. TEST
**Purpose:** Run test suites
```python
ActionType.TEST
# Detects test framework, runs tests, reports results
```

**Execution:**
- Detects testing framework (pytest, jest, go test, etc.)
- Executes test commands
- Captures output
- Returns test results

#### 4. IMPLEMENT
**Purpose:** Deploy and implement changes
```python
ActionType.IMPLEMENT
# Deploys code, starts services, runs scripts
```

**Execution:**
- Looks for deployment scripts
- Installs dependencies
- Builds projects
- Starts applications
- Returns deployment status

#### 5. DEBUG
**Purpose:** Debug issues and analyze logs
```python
ActionType.DEBUG
# Checks logs, processes, ports, identifies issues
```

**Execution:**
- Searches log files for errors
- Checks running processes
- Analyzes port usage
- Returns diagnostic information

#### 6. OPTIMIZE
**Purpose:** Performance optimization
```python
ActionType.OPTIMIZE
# Analyzes performance, suggests improvements
```

#### 7. DOCUMENT
**Purpose:** Create documentation
```python
ActionType.DOCUMENT
# Generates README, API docs, comments
```

#### 8. CONFIGURE
**Purpose:** System configuration
```python
ActionType.CONFIGURE
# Sets up environments, configures services
```

### Action Classification
The system automatically classifies user requests into appropriate actions:

```python
from agent_actions import ActionClassifier

# Example
user_query = "analyze my blockchain application"
action_type = ActionClassifier.classify_request(user_query)
# Returns: ActionType.READ_ANALYZE
```

### Action Execution Flow

```python
# 1. Create action request
action_request = ActionRequest(
    action_type=ActionType.READ_ANALYZE,
    target_path="/home/ubuntu/project",
    description="Analyze project",
    deployment_config={...},
    parameters={...}
)

# 2. Execute action
executor = AgentActionExecutor()
result = await executor.execute_action(action_request)

# 3. Process result
if result.success:
    print(f"Output: {result.output}")
    print(f"Files: {result.files_affected}")
```

---

## Agent Management

### Creating Agents
Agents are defined with specific capabilities and knowledge bases:

```python
from agent_config import AgentConfig, AgentRole, AgentCapabilities

config = AgentConfig(
    name="database_specialist",
    role=AgentRole.BACKEND,
    description="Database optimization expert",
    model="gpt-4",
    capabilities=AgentCapabilities(
        can_write_code=True,
        can_debug=True,
        can_analyze_data=True
    ),
    knowledge_base=KnowledgeBase(
        domains=["SQL", "NoSQL", "Performance"],
        examples=[...]
    )
)
```

### Agent Selection
The system automatically selects the best agent for a task:

```python
def _select_best_agent(task: str) -> str:
    # Analyzes task keywords
    # Scores each agent
    # Returns best match
```

---

## Knowledge Management

### Knowledge Structure
```python
@dataclass
class KnowledgeItem:
    id: str
    agent_name: str
    category: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float
    usage_count: int
```

### Adding Knowledge
```python
# Via API
POST /api/v2/knowledge/add
{
    "agent_id": "database_specialist",
    "category": "optimization",
    "content": "Index optimization techniques...",
    "source": "documentation"
}

# Via Code
knowledge_manager.add_knowledge(
    agent_name="database_specialist",
    category="optimization",
    content="...",
    source="training"
)
```

### Knowledge Training
Agents actively consume and learn from knowledge:

```python
from knowledge_trainer import KnowledgeTrainer

trainer = KnowledgeTrainer()
result = trainer.consume_knowledge("database_specialist")
# Agent integrates new knowledge into responses
```

### Knowledge Retrieval
```python
relevant_knowledge = knowledge_manager.search_knowledge(
    agent_name="database_specialist",
    query="optimize PostgreSQL",
    limit=5
)
```

---

## Memory System

### Memory Types

#### 1. Episodic Memory
Specific experiences and interactions:
```python
memory_manager.store_episodic({
    "experience": "Successfully optimized query",
    "context": {"database": "PostgreSQL"},
    "result": "50% performance improvement"
})
```

#### 2. Semantic Memory
General knowledge and concepts:
```python
memory_manager.store_semantic(
    concept="query_optimization",
    content={
        "techniques": ["indexing", "caching"],
        "tools": ["EXPLAIN ANALYZE"]
    }
)
```

#### 3. Procedural Memory
How-to knowledge and skills:
```python
memory_manager.store_procedural(
    skill_name="optimize_slow_query",
    procedure={
        "steps": ["identify", "analyze", "optimize"],
        "tools": ["query profiler"]
    }
)
```

### Shared Memory System
All agents share context and learnings:

```python
# Add conversation to shared memory
memory_system.add_conversation(
    agent="database_specialist",
    user_query="Optimize my database",
    agent_response="...",
    success=True
)

# Get agent context
context = memory_system.get_agent_context("database_specialist")
# Returns recent conversations, project status, memories
```

---

## Deployment Configuration

### Project Configuration System
Save deployment settings per project:

```python
@dataclass
class ProjectConfig:
    project_name: str
    description: str
    deployment_targets: List[DeploymentTarget]
    documentation_paths: List[str]
    active: bool = True
```

### Deployment Targets

#### 1. Local Development
```python
DeploymentTarget(
    name="local",
    type="local",
    server_ip=None,
    username=None,
    ssh_key_path=None,
    working_directory="/Users/jankootstra/project"
)
```

#### 2. Ubuntu Server
```python
DeploymentTarget(
    name="production",
    type="ubuntu_server",
    server_ip="13.38.102.28",
    username="ubuntu",
    ssh_key_path="/Users/jankootstra/blockchain.pem",
    working_directory="/home/ubuntu"
)
```

#### 3. AWS EC2
```python
DeploymentTarget(
    name="aws",
    type="aws_ec2",
    aws_region="us-east-1",
    instance_type="t2.micro",
    key_name="my-key"
)
```

---

## API Reference

### Base URLs
- Enhanced API: `http://localhost:8002`
- Original API: `http://localhost:8001`
- Dashboard: `http://localhost:8501`

### Core Endpoints

#### Execute Task with Agent
```http
POST /api/v2/agents/execute
Content-Type: application/json

{
    "task": "Analyze my blockchain application",
    "agent_id": "full_stack_developer",  # Optional
    "context": {
        "deployment": {
            "type": "ubuntu_server",
            "server_ip": "13.38.102.28",
            "username": "ubuntu",
            "key_path": "/Users/jankootstra/blockchain.pem",
            "working_directory": "/home/ubuntu/fabric-api-gateway-modular"
        }
    }
}
```

**Response:**
```json
{
    "task_id": "uuid",
    "status": "completed",
    "result": {
        "response": "Actual execution results...",
        "agent": "full_stack_developer",
        "knowledge_items_used": 5,
        "action_executed": "read_analyze",
        "action_success": true
    }
}
```

#### Get Action Result
```http
GET /api/v2/actions/{task_id}
```

**Response:**
```json
{
    "task_id": "uuid",
    "action_type": "read_analyze",
    "success": true,
    "output": "Detailed execution output...",
    "files_affected": ["file1.js", "file2.py"],
    "metadata": {...}
}
```

#### List Specialized Agents
```http
GET /api/v2/agents/list
```

#### Add Knowledge
```http
POST /api/v2/knowledge/add
{
    "agent_id": "database_specialist",
    "category": "optimization",
    "content": "Knowledge content...",
    "source": "api"
}
```

#### Train Agent
```http
POST /api/v2/knowledge/train/{agent_id}
```

#### Get Memory Context
```http
GET /api/v2/memory/context/{agent_id}
```

#### Get Project Status
```http
GET /api/v2/memory/status
```

---

## Dashboard Interface

### Main Features

#### 1. Agent Performance Tab
- Real-time metrics
- Task completion rates
- Response times
- Active agents status

#### 2. Submit Task Tab
- Task input field
- Agent selection
- Deployment configuration
- Real-time response display

#### 3. Project Config Tab
- Create/edit projects
- Manage deployment targets
- Save SSH credentials
- Directory structure configuration

#### 4. Knowledge Management Tab
- View knowledge bases
- Add new knowledge
- Search existing knowledge
- Knowledge statistics

#### 5. Train Agent Tab
- Select agent for training
- Add training data
- Test with queries
- View training results

#### 6. System Logs Tab
- Real-time logs
- Error tracking
- Performance monitoring
- Debug information

---

## SSH Execution

### SSH Configuration
```python
from ssh_executor import SSHConfig, SSHExecutor

config = SSHConfig(
    host="13.38.102.28",
    username="ubuntu",
    key_path="/Users/jankootstra/blockchain.pem",
    working_directory="/home/ubuntu"
)

executor = SSHExecutor(config)
if executor.connect():
    result = executor.execute_command("ls -la")
    print(result.stdout)
    executor.close()
```

### Server Analysis
```python
from ssh_executor import ServerAnalyzer

analyzer = ServerAnalyzer(executor)
analysis = analyzer.full_analysis()
# Returns: system info, projects, databases, services
```

### Database Analysis
```python
databases = executor.analyze_databases()
# Returns: PostgreSQL, MySQL, MongoDB, Redis info
```

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### Starting Services

#### 1. Start Enhanced API
```bash
uvicorn enhanced_production_api:enhanced_app --reload --port 8002 --host 0.0.0.0
```

#### 2. Start Dashboard
```bash
streamlit run monitoring_dashboard.py --server.port 8501
```

#### 3. Start Original API (Optional)
```bash
uvicorn production_api:app --port 8001 --host 0.0.0.0
```

---

## Usage Examples

### Example 1: Analyze Remote Application
```python
# Via Dashboard
1. Go to "Submit Task" tab
2. Enter: "Analyze my blockchain application in fabric-api-gateway-modular"
3. Select deployment: Ubuntu Server
4. Submit

# Via API
curl -X POST http://localhost:8002/api/v2/agents/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze blockchain app",
    "context": {
      "deployment": {
        "type": "ubuntu_server",
        "server_ip": "13.38.102.28",
        "username": "ubuntu",
        "key_path": "/Users/jankootstra/blockchain.pem"
      }
    }
  }'
```

### Example 2: Create and Deploy Code
```python
# Task
"Create a REST API for user management with authentication"

# System will:
1. Classify as CREATE_CODE action
2. Generate API code
3. Deploy to specified server
4. Return actual files created
```

### Example 3: Debug Production Issue
```python
# Task
"Debug why my application is not responding on port 3000"

# System will:
1. Classify as DEBUG action
2. Check logs for errors
3. Verify process status
4. Check port bindings
5. Return diagnostic results
```

### Example 4: Run Tests
```python
# Task
"Run all tests for my Node.js application"

# System will:
1. Classify as TEST action
2. Detect test framework (Jest, Mocha, etc.)
3. Execute test suite
4. Return test results
```

---

## Troubleshooting

### Common Issues

#### 1. SSH Connection Failed
**Error:** "No such file or directory: 'blockchain.pem'"

**Solution:**
```python
# Use absolute path
key_path = "/Users/jankootstra/blockchain.pem"

# Or expand user directory
key_path = os.path.expanduser("~/blockchain.pem")
```

#### 2. Agent Not Executing Tasks
**Symptom:** Agent describes actions instead of executing

**Solution:**
- Ensure deployment configuration is provided
- Check action system is integrated
- Verify SSH credentials are correct

#### 3. Memory System Not Available
**Error:** "Memory system not available"

**Solution:**
```python
# Ensure SharedMemorySystem is initialized
from shared_memory_system import SharedMemorySystem
memory_system = SharedMemorySystem()
```

#### 4. Knowledge Not Being Used
**Symptom:** Agents not using added knowledge

**Solution:**
```python
# Train agent after adding knowledge
POST /api/v2/knowledge/train/{agent_id}
```

### Debug Mode
Enable detailed logging:
```python
# In enhanced_production_api.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check
```http
GET /health
```

Returns:
```json
{
    "status": "healthy",
    "version": "2.0.0",
    "specialized_agents": 10,
    "total_knowledge_items": 250
}
```

---

## Best Practices

### 1. Project Configuration
- Always save project configurations for reuse
- Use descriptive project names
- Store SSH keys securely

### 2. Task Submission
- Be specific in task descriptions
- Include target paths when relevant
- Specify deployment target explicitly

### 3. Knowledge Management
- Regularly train agents with new knowledge
- Categorize knowledge appropriately
- Review and update outdated knowledge

### 4. Memory Optimization
- Monitor memory usage statistics
- Allow consolidation of episodic memories
- Review shared learnings periodically

### 5. Security
- Never commit SSH keys to version control
- Use environment variables for sensitive data
- Regularly rotate access credentials

---

## Architecture Decisions

### Why Action-Based System?
- **Problem:** AI agents only described hypothetical actions
- **Solution:** Concrete action types with real execution
- **Benefit:** Reliable, verifiable task completion

### Why Shared Memory?
- **Problem:** Agents operated in isolation
- **Solution:** Unified memory system
- **Benefit:** Agents learn from each other's experiences

### Why Project Configuration?
- **Problem:** Repeated entry of deployment details
- **Solution:** Persistent project configurations
- **Benefit:** One-time setup, multiple uses

### Why Knowledge Training?
- **Problem:** Static agent knowledge
- **Solution:** Active knowledge consumption
- **Benefit:** Continuously improving agents

---

## Future Enhancements

### Planned Features
1. **Multi-agent Collaboration**: Agents working together on complex tasks
2. **Visual Code Generation**: UI-based code builders
3. **Automated Testing**: Continuous testing integration
4. **Performance Monitoring**: Real-time application monitoring
5. **Rollback Capability**: Undo deployed changes
6. **Multi-cloud Support**: GCP, Azure integration
7. **Version Control Integration**: Direct Git operations
8. **Container Management**: Docker/Kubernetes operations

### Contribution Guidelines
- Follow existing code patterns
- Add tests for new features
- Update documentation
- Use type hints
- Follow PEP 8 style guide

---

## Support & Contact

### Resources
- GitHub Issues: Report bugs and request features
- API Documentation: `/docs` endpoint
- Dashboard Help: Built-in help tooltips

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Network access for API calls
- SSH access for remote operations

---

## License

This implementation is part of the Agent Lightning project. All rights reserved.

---

## Conclusion

Agent Lightning represents a paradigm shift from AI assistants that merely describe tasks to agents that **actually execute** them. With its action-based system, specialized agents, and comprehensive infrastructure support, it provides a reliable platform for automated task execution across local and remote environments.

The system's ability to learn, remember, and share knowledge ensures continuous improvement, while the project configuration system and intuitive dashboard make it accessible for both technical and non-technical users.

**Key Takeaway:** When you tell Agent Lightning to do something, it doesn't just tell you how—it actually does it.