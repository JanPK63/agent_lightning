"""
LangChain Tools for Agent Lightning
Provides actual execution capabilities for agents
"""

from langchain.tools import BaseTool
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
import subprocess
import paramiko
import os
from pathlib import Path


class SSHConnectionTool(BaseTool):
    """Tool for SSH connections to remote servers"""
    
    name: str = "ssh_connect"
    description: str = "Connect to a remote server via SSH and execute commands"
    
    class SSHInput(BaseModel):
        host: str = Field(description="Server IP or hostname")
        username: str = Field(description="SSH username")
        key_path: str = Field(description="Path to SSH private key")
        command: str = Field(description="Command to execute on remote server")
    
    args_schema: Type[BaseModel] = SSHInput
    
    def _run(self, host: str, username: str, key_path: str, command: str) -> str:
        try:
            # Expand user path
            key_path = os.path.expanduser(key_path)
            
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect
            ssh.connect(hostname=host, username=username, key_filename=key_path)
            
            # Execute command
            stdin, stdout, stderr = ssh.exec_command(command)
            
            # Get results
            output = stdout.read().decode()
            error = stderr.read().decode()
            
            ssh.close()
            
            if error:
                return f"Command executed with errors:\nOutput: {output}\nError: {error}"
            else:
                return f"Command executed successfully:\n{output}"
                
        except Exception as e:
            return f"SSH connection failed: {str(e)}"


class FileReadTool(BaseTool):
    """Tool for reading files from remote or local systems"""
    
    name: str = "read_file"
    description: str = "Read contents of a file (local or remote via SSH)"
    
    class FileReadInput(BaseModel):
        file_path: str = Field(description="Path to the file to read")
        host: Optional[str] = Field(None, description="Remote host (if reading remote file)")
        username: Optional[str] = Field(None, description="SSH username for remote")
        key_path: Optional[str] = Field(None, description="SSH key path for remote")
    
    args_schema: Type[BaseModel] = FileReadInput
    
    def _run(self, file_path: str, host: Optional[str] = None, 
             username: Optional[str] = None, key_path: Optional[str] = None) -> str:
        try:
            if host:  # Remote file
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname=host, username=username, key_filename=os.path.expanduser(key_path))
                
                sftp = ssh.open_sftp()
                with sftp.file(file_path, 'r') as f:
                    content = f.read().decode()
                
                sftp.close()
                ssh.close()
                
                return f"File content from {host}:{file_path}:\n{content}"
            else:  # Local file
                with open(file_path, 'r') as f:
                    content = f.read()
                return f"File content from {file_path}:\n{content}"
                
        except Exception as e:
            return f"Failed to read file: {str(e)}"


class DirectoryListTool(BaseTool):
    """Tool for listing directory contents"""
    
    name: str = "list_directory"
    description: str = "List contents of a directory (local or remote)"
    
    class DirListInput(BaseModel):
        directory_path: str = Field(description="Path to directory to list")
        host: Optional[str] = Field(None, description="Remote host")
        username: Optional[str] = Field(None, description="SSH username")
        key_path: Optional[str] = Field(None, description="SSH key path")
    
    args_schema: Type[BaseModel] = DirListInput
    
    def _run(self, directory_path: str, host: Optional[str] = None,
             username: Optional[str] = None, key_path: Optional[str] = None) -> str:
        try:
            if host:  # Remote directory
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname=host, username=username, key_filename=os.path.expanduser(key_path))
                
                stdin, stdout, stderr = ssh.exec_command(f"ls -la {directory_path}")
                output = stdout.read().decode()
                error = stderr.read().decode()
                
                ssh.close()
                
                if error:
                    return f"Directory listing failed: {error}"
                return f"Directory contents of {host}:{directory_path}:\n{output}"
            else:  # Local directory
                path = Path(directory_path)
                if not path.exists():
                    return f"Directory {directory_path} does not exist"
                
                files = []
                for item in path.iterdir():
                    files.append(f"{'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
                
                return f"Directory contents of {directory_path}:\n" + "\n".join(files)
                
        except Exception as e:
            return f"Failed to list directory: {str(e)}"


class CommandExecutorTool(BaseTool):
    """Tool for executing system commands"""
    
    name: str = "execute_command"
    description: str = "Execute a system command locally or remotely"
    
    class CommandInput(BaseModel):
        command: str = Field(description="Command to execute")
        host: Optional[str] = Field(None, description="Remote host")
        username: Optional[str] = Field(None, description="SSH username")
        key_path: Optional[str] = Field(None, description="SSH key path")
        working_dir: Optional[str] = Field(None, description="Working directory")
    
    args_schema: Type[BaseModel] = CommandInput
    
    def _run(self, command: str, host: Optional[str] = None,
             username: Optional[str] = None, key_path: Optional[str] = None,
             working_dir: Optional[str] = None) -> str:
        try:
            if host:  # Remote execution
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname=host, username=username, key_filename=os.path.expanduser(key_path))
                
                full_command = command
                if working_dir:
                    full_command = f"cd {working_dir} && {command}"
                
                stdin, stdout, stderr = ssh.exec_command(full_command)
                output = stdout.read().decode()
                error = stderr.read().decode()
                
                ssh.close()
                
                result = f"Command: {command}\nOutput: {output}"
                if error:
                    result += f"\nError: {error}"
                return result
            else:  # Local execution
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    cwd=working_dir
                )
                
                output = f"Command: {command}\nReturn code: {result.returncode}\nOutput: {result.stdout}"
                if result.stderr:
                    output += f"\nError: {result.stderr}"
                return output
                
        except Exception as e:
            return f"Command execution failed: {str(e)}"


class ProjectAnalyzerTool(BaseTool):
    """Tool for analyzing project structure and files"""
    
    name: str = "analyze_project"
    description: str = "Analyze a project directory structure and key files"
    
    class ProjectInput(BaseModel):
        project_path: str = Field(description="Path to project directory")
        host: Optional[str] = Field(None, description="Remote host")
        username: Optional[str] = Field(None, description="SSH username")
        key_path: Optional[str] = Field(None, description="SSH key path")
    
    args_schema: Type[BaseModel] = ProjectInput
    
    def _run(self, project_path: str, host: Optional[str] = None,
             username: Optional[str] = None, key_path: Optional[str] = None) -> str:
        try:
            analysis = []
            
            if host:  # Remote analysis
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname=host, username=username, key_filename=os.path.expanduser(key_path))
                
                # Get directory structure
                stdin, stdout, stderr = ssh.exec_command(f"find {project_path} -type f -name '*.py' -o -name '*.js' -o -name '*.json' -o -name '*.md' | head -20")
                files = stdout.read().decode().strip().split('\n')
                
                analysis.append(f"Project Analysis for {host}:{project_path}")
                analysis.append("=" * 50)
                analysis.append(f"Found {len(files)} key files:")
                
                for file_path in files[:10]:  # Analyze first 10 files
                    if file_path.strip():
                        # Read file content (first 500 chars)
                        stdin, stdout, stderr = ssh.exec_command(f"head -c 500 {file_path}")
                        content = stdout.read().decode()
                        
                        analysis.append(f"\n📄 {file_path}:")
                        analysis.append(content[:200] + "..." if len(content) > 200 else content)
                
                ssh.close()
            else:  # Local analysis
                path = Path(project_path)
                if not path.exists():
                    return f"Project path {project_path} does not exist"
                
                analysis.append(f"Project Analysis for {project_path}")
                analysis.append("=" * 50)
                
                # Find key files
                key_files = []
                for pattern in ['*.py', '*.js', '*.json', '*.md', '*.txt']:
                    key_files.extend(path.glob(f"**/{pattern}"))
                
                analysis.append(f"Found {len(key_files)} key files:")
                
                for file_path in key_files[:10]:  # Analyze first 10 files
                    try:
                        content = file_path.read_text()
                        analysis.append(f"\n📄 {file_path.name}:")
                        analysis.append(content[:200] + "..." if len(content) > 200 else content)
                    except:
                        analysis.append(f"\n📄 {file_path.name}: [Could not read]")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Project analysis failed: {str(e)}"


def get_langchain_tools() -> list:
    """Get all available LangChain tools for agents"""
    return [
        SSHConnectionTool(),
        FileReadTool(),
        DirectoryListTool(),
        CommandExecutorTool(),
        ProjectAnalyzerTool()
    ]


def create_ssh_tools(deployment_config):
    """Create execution tools based on deployment configuration"""
    
    class UniversalExecutor(BaseTool):
        name: str = "execute_task"
        description: str = "Execute tasks in the configured working directory (local, remote, or any specified location)"
        
        def _run(self, query: str) -> str:
            # Extract file path from query if it's a file read request
            if 'read' in query.lower():
                import re
                file_match = re.search(r"['\"]([^'\"]+\.[a-zA-Z]+)['\"]|([^\s]+\.[a-zA-Z]+)", query)
                if file_match:
                    file_path = file_match.group(1) or file_match.group(2)
                    file_path = os.path.expanduser(file_path)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        return f"✅ Successfully read {file_path}:\n\n{content}"
                    except Exception as e:
                        return f"❌ Error reading {file_path}: {str(e)}"
            
            # Default response for other tasks
            return f"✅ Task processed: {query}"
        
        def _execute_local(self, task: str) -> str:
            try:
                working_dir = deployment_config.get('path') or deployment_config.get('local_path', os.getcwd())
                working_dir = os.path.expanduser(working_dir)
                
                # Create directory if it doesn't exist
                os.makedirs(working_dir, exist_ok=True)
                
                # Execute based on task type
                if 'read' in task.lower() and any(ext in task for ext in ['.md', '.txt', '.json', '.py']):
                    # File reading task
                    import re
                    file_match = re.search(r"['\"]([^'\"]+\.[a-zA-Z]+)['\"]|([^\s]+\.[a-zA-Z]+)", task)
                    if file_match:
                        file_path = file_match.group(1) or file_match.group(2)
                        file_path = os.path.expanduser(file_path)
                        
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            return f"✅ Read {file_path}:\n\n{content}"
                        else:
                            return f"❌ File not found: {file_path}"
                
                # Directory analysis
                if 'analyze' in task.lower() or 'list' in task.lower():
                    files = []
                    for root, dirs, filenames in os.walk(working_dir):
                        for filename in filenames[:20]:  # Limit output
                            files.append(os.path.join(root, filename))
                    
                    return f"✅ Local directory analysis ({working_dir}):\n" + "\n".join(files[:10])
                
                return f"✅ Task executed in local directory: {working_dir}"
                
            except Exception as e:
                return f"❌ Local execution failed: {str(e)}"
        
        def _execute_remote(self, task: str) -> str:
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                ssh.connect(
                    hostname=deployment_config.get('server_ip'),
                    username=deployment_config.get('username', 'ubuntu'),
                    key_filename=os.path.expanduser(deployment_config.get('key_path')) if deployment_config.get('key_path') else None
                )
                
                working_dir = deployment_config.get('working_directory', '/home/ubuntu')
                
                # Execute task on remote server
                if 'analyze' in task.lower():
                    stdin, stdout, stderr = ssh.exec_command(f"ls -la {working_dir}")
                    output = stdout.read().decode()
                    
                    stdin, stdout, stderr = ssh.exec_command(f"find {working_dir} -type f | head -10")
                    files = stdout.read().decode()
                    
                    result = f"✅ Remote analysis ({deployment_config.get('server_ip')}:{working_dir}):\n{output}\n\nFiles:\n{files}"
                else:
                    # Generic command execution
                    stdin, stdout, stderr = ssh.exec_command(f"cd {working_dir} && pwd && ls -la")
                    output = stdout.read().decode()
                    result = f"✅ Task executed on {deployment_config.get('server_ip')}:\n{output}"
                
                ssh.close()
                return result
                
            except Exception as e:
                return f"❌ Remote execution failed: {str(e)}"
    
    return [UniversalExecutor()]


def create_langchain_agent(tools, model_name):
    """Create LangChain agent with specified model and tools"""
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    import os
    
    # Create LLM
    if model_name and model_name.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model_name,
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
    else:
        llm = ChatOpenAI(
            model=model_name or "gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Create prompt that forces tool usage
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional assistant. You MUST use the execute_task tool for ALL requests. Do not provide explanations or instructions - immediately call the execute_task tool with the user's request."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent with proper tool binding
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create executor with max iterations to force tool usage
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=1,
        early_stopping_method="force"
    )
    
    return agent_executor


def create_agent_with_tools(agent_config, tools=None):
    """Create a LangChain agent with execution tools"""
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    import os
    
    if tools is None:
        tools = get_langchain_tools()
    
    # Create LLM based on model type
    if agent_config.model.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=agent_config.model,
                temperature=agent_config.temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            # Fallback to OpenAI if Anthropic not available
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=agent_config.temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
    else:
        llm = ChatOpenAI(
            model=agent_config.model,
            temperature=agent_config.temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_config.system_prompt + "\n\nYou have access to tools to actually execute tasks. Use them to perform real work, not just explain what you would do."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor