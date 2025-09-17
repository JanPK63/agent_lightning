#!/usr/bin/env python3
"""
Agent Code Executor
Allows AI agents to execute code and implement solutions on local or remote systems
"""

import os
import subprocess
import tempfile
import json
import paramiko
import git
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    LOCAL = "local"
    REMOTE_SSH = "remote_ssh"
    DOCKER = "docker"
    SANDBOX = "sandbox"


class FileOperation(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    READ = "read"


@dataclass
class ExecutionConfig:
    """Configuration for code execution"""
    mode: ExecutionMode
    working_directory: str
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    docker_image: Optional[str] = None
    sandbox_limits: Optional[Dict[str, Any]] = None
    git_repo: Optional[str] = None
    branch: Optional[str] = "main"


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: str
    error: Optional[str] = None
    files_created: List[str] = None
    files_modified: List[str] = None
    execution_time: float = 0.0


class CodeExecutor:
    """Execute code and manage files on local or remote systems"""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.ssh_client = None
        self.git_repo = None
        
        # Safety checks
        self.allowed_commands = {
            "ls", "pwd", "echo", "cat", "mkdir", "cp", "mv", "rm",
            "python", "pip", "npm", "node", "git", "docker",
            "terraform", "kubectl", "aws", "gcloud", "az"
        }
        
        # File extensions we can work with
        self.allowed_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
            ".rs", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb",
            ".php", ".swift", ".kt", ".scala", ".r", ".m",
            ".html", ".css", ".scss", ".json", ".xml", ".yaml",
            ".yml", ".toml", ".ini", ".conf", ".env", ".md",
            ".txt", ".sh", ".bash", ".zsh", ".dockerfile", ".sql"
        }
        
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup connection based on execution mode"""
        if self.config.mode == ExecutionMode.REMOTE_SSH:
            self._setup_ssh()
        elif self.config.mode == ExecutionMode.LOCAL:
            self._setup_local()
        
        if self.config.git_repo:
            self._setup_git()
    
    def _setup_ssh(self):
        """Setup SSH connection for remote execution"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if self.config.ssh_key_path:
                self.ssh_client.connect(
                    hostname=self.config.ssh_host,
                    username=self.config.ssh_user,
                    key_filename=self.config.ssh_key_path
                )
            else:
                # Use SSH agent
                self.ssh_client.connect(
                    hostname=self.config.ssh_host,
                    username=self.config.ssh_user
                )
            
            logger.info(f"✅ Connected to {self.config.ssh_host} via SSH")
        except Exception as e:
            logger.error(f"Failed to setup SSH: {e}")
            raise
    
    def _setup_local(self):
        """Setup local execution environment"""
        # Ensure working directory exists
        Path(self.config.working_directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Local execution in {self.config.working_directory}")
    
    def _setup_git(self):
        """Setup Git repository"""
        try:
            repo_path = Path(self.config.working_directory)
            
            if repo_path.exists() and (repo_path / '.git').exists():
                self.git_repo = git.Repo(repo_path)
                logger.info(f"✅ Using existing Git repo at {repo_path}")
            else:
                if self.config.git_repo:
                    self.git_repo = git.Repo.clone_from(
                        self.config.git_repo,
                        repo_path
                    )
                    logger.info(f"✅ Cloned Git repo from {self.config.git_repo}")
        except Exception as e:
            logger.warning(f"Git setup failed: {e}")
    
    async def execute_command(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Execute a shell command"""
        
        # Safety check
        command_parts = command.split()
        base_command = command_parts[0] if command_parts else ""
        
        if base_command not in self.allowed_commands and not base_command.startswith("./"):
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command '{base_command}' not in allowed list"
            )
        
        if self.config.mode == ExecutionMode.LOCAL:
            return await self._execute_local(command, timeout)
        elif self.config.mode == ExecutionMode.REMOTE_SSH:
            return await self._execute_remote(command, timeout)
        elif self.config.mode == ExecutionMode.DOCKER:
            return await self._execute_docker(command, timeout)
        else:
            return await self._execute_sandbox(command, timeout)
    
    async def _execute_local(self, command: str, timeout: int) -> ExecutionResult:
        """Execute command locally"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.working_directory
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8') if stderr else None
            )
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
    
    async def _execute_remote(self, command: str, timeout: int) -> ExecutionResult:
        """Execute command on remote server via SSH"""
        if not self.ssh_client:
            return ExecutionResult(
                success=False,
                output="",
                error="SSH connection not established"
            )
        
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"cd {self.config.working_directory} && {command}",
                timeout=timeout
            )
            
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            return ExecutionResult(
                success=stdout.channel.recv_exit_status() == 0,
                output=output,
                error=error if error else None
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
    
    async def _execute_docker(self, command: str, timeout: int) -> ExecutionResult:
        """Execute command in Docker container"""
        docker_command = f"docker run --rm -v {self.config.working_directory}:/workspace {self.config.docker_image} {command}"
        return await self._execute_local(docker_command, timeout)
    
    async def _execute_sandbox(self, command: str, timeout: int) -> ExecutionResult:
        """Execute command in sandboxed environment"""
        # Use firejail or similar for sandboxing on Linux
        sandbox_command = f"firejail --quiet {command}"
        return await self._execute_local(sandbox_command, timeout)
    
    async def manage_file(self, operation: FileOperation, file_path: str, 
                         content: Optional[str] = None) -> ExecutionResult:
        """Manage files (create, modify, delete, read)"""
        
        # Safety check for file extension
        file_ext = Path(file_path).suffix
        if file_ext not in self.allowed_extensions:
            return ExecutionResult(
                success=False,
                output="",
                error=f"File extension '{file_ext}' not allowed"
            )
        
        full_path = Path(self.config.working_directory) / file_path
        
        try:
            if operation == FileOperation.CREATE:
                return await self._create_file(full_path, content)
            elif operation == FileOperation.MODIFY:
                return await self._modify_file(full_path, content)
            elif operation == FileOperation.DELETE:
                return await self._delete_file(full_path)
            elif operation == FileOperation.READ:
                return await self._read_file(full_path)
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
    
    async def _create_file(self, file_path: Path, content: str) -> ExecutionResult:
        """Create a new file"""
        if self.config.mode == ExecutionMode.LOCAL:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return ExecutionResult(
                success=True,
                output=f"Created file: {file_path}",
                files_created=[str(file_path)]
            )
        elif self.config.mode == ExecutionMode.REMOTE_SSH:
            # Use SSH to create file
            command = f"mkdir -p {file_path.parent} && cat > {file_path} << 'EOF'\n{content}\nEOF"
            return await self.execute_command(command)
    
    async def _modify_file(self, file_path: Path, content: str) -> ExecutionResult:
        """Modify an existing file"""
        if self.config.mode == ExecutionMode.LOCAL:
            if not file_path.exists():
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"File {file_path} does not exist"
                )
            
            # Backup original
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            backup_path.write_text(file_path.read_text())
            
            # Write new content
            file_path.write_text(content)
            
            return ExecutionResult(
                success=True,
                output=f"Modified file: {file_path}",
                files_modified=[str(file_path)]
            )
    
    async def _delete_file(self, file_path: Path) -> ExecutionResult:
        """Delete a file"""
        if self.config.mode == ExecutionMode.LOCAL:
            if file_path.exists():
                file_path.unlink()
                return ExecutionResult(
                    success=True,
                    output=f"Deleted file: {file_path}"
                )
            else:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"File {file_path} does not exist"
                )
    
    async def _read_file(self, file_path: Path) -> ExecutionResult:
        """Read a file"""
        if self.config.mode == ExecutionMode.LOCAL:
            if file_path.exists():
                content = file_path.read_text()
                return ExecutionResult(
                    success=True,
                    output=content
                )
            else:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"File {file_path} does not exist"
                )
    
    async def git_commit(self, message: str, files: List[str] = None) -> ExecutionResult:
        """Commit changes to Git"""
        if not self.git_repo:
            return ExecutionResult(
                success=False,
                output="",
                error="Git repository not initialized"
            )
        
        try:
            if files:
                self.git_repo.index.add(files)
            else:
                self.git_repo.git.add(A=True)
            
            self.git_repo.index.commit(message)
            
            return ExecutionResult(
                success=True,
                output=f"Committed: {message}"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
    
    async def git_push(self) -> ExecutionResult:
        """Push changes to remote repository"""
        if not self.git_repo:
            return ExecutionResult(
                success=False,
                output="",
                error="Git repository not initialized"
            )
        
        try:
            origin = self.git_repo.remote(name='origin')
            origin.push()
            
            return ExecutionResult(
                success=True,
                output="Pushed to remote repository"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def close(self):
        """Close connections"""
        if self.ssh_client:
            self.ssh_client.close()


class AgentCodeExecutor:
    """High-level executor for AI agents to implement solutions"""
    
    def __init__(self, executor: CodeExecutor):
        self.executor = executor
        self.implementation_history = []
    
    async def implement_feature(self, description: str, code_snippets: Dict[str, str]) -> Dict[str, Any]:
        """Implement a complete feature based on description and code snippets"""
        
        results = {
            "description": description,
            "files_created": [],
            "files_modified": [],
            "commands_executed": [],
            "success": True,
            "errors": []
        }
        
        # Create or modify files
        for file_path, content in code_snippets.items():
            result = await self.executor.manage_file(
                FileOperation.CREATE if not (Path(self.executor.config.working_directory) / file_path).exists() else FileOperation.MODIFY,
                file_path,
                content
            )
            
            if result.success:
                if result.files_created:
                    results["files_created"].extend(result.files_created)
                if result.files_modified:
                    results["files_modified"].extend(result.files_modified)
            else:
                results["errors"].append(f"Failed to manage {file_path}: {result.error}")
                results["success"] = False
        
        # Commit changes if Git is configured
        if self.executor.git_repo and results["success"]:
            commit_result = await self.executor.git_commit(
                f"Implement: {description[:50]}",
                results["files_created"] + results["files_modified"]
            )
            
            if not commit_result.success:
                results["errors"].append(f"Git commit failed: {commit_result.error}")
        
        self.implementation_history.append(results)
        return results
    
    async def run_tests(self, test_command: str = "pytest") -> ExecutionResult:
        """Run tests to verify implementation"""
        return await self.executor.execute_command(test_command)
    
    async def deploy(self, deploy_script: str) -> ExecutionResult:
        """Deploy the implementation"""
        return await self.executor.execute_command(deploy_script)


# Example usage
async def main():
    # Local execution example
    local_config = ExecutionConfig(
        mode=ExecutionMode.LOCAL,
        working_directory="/tmp/agent_workspace",
        git_repo="https://github.com/user/project.git"
    )
    
    executor = CodeExecutor(local_config)
    agent_executor = AgentCodeExecutor(executor)
    
    # Implement a simple feature
    code_snippets = {
        "hello.py": "#!/usr/bin/env python3\n\ndef hello():\n    print('Hello from Agent!')\n\nif __name__ == '__main__':\n    hello()\n",
        "README.md": "# Agent Implementation\n\nThis code was automatically generated and implemented by an AI agent.\n"
    }
    
    result = await agent_executor.implement_feature(
        "Create a simple hello world Python script",
        code_snippets
    )
    
    print(json.dumps(result, indent=2))
    
    # Run the code
    exec_result = await executor.execute_command("python hello.py")
    print(f"Output: {exec_result.output}")
    
    executor.close()


if __name__ == "__main__":
    asyncio.run(main())