#!/usr/bin/env python3
"""
SSH Executor for Agent Lightning
Provides SSH connectivity and command execution for agents
"""

import paramiko
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    """SSH connection configuration"""
    host: str
    username: str
    key_path: Optional[str] = None
    password: Optional[str] = None
    port: int = 22
    working_directory: str = "/home/ubuntu"


@dataclass
class CommandResult:
    """Result from SSH command execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    command: str


class SSHExecutor:
    """Execute commands on remote servers via SSH"""
    
    def __init__(self, config: SSHConfig):
        self.config = config
        self.client = None
        self.sftp = None
        
    def connect(self) -> bool:
        """Establish SSH connection"""
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect using key or password
            if self.config.key_path:
                # Expand the key path
                key_path = os.path.expanduser(self.config.key_path)
                if not os.path.exists(key_path):
                    logger.error(f"SSH key not found: {key_path}")
                    return False
                    
                self.client.connect(
                    hostname=self.config.host,
                    username=self.config.username,
                    key_filename=key_path,
                    port=self.config.port
                )
            else:
                self.client.connect(
                    hostname=self.config.host,
                    username=self.config.username,
                    password=self.config.password,
                    port=self.config.port
                )
            
            # Setup SFTP for file operations
            self.sftp = self.client.open_sftp()
            logger.info(f"✅ Connected to {self.config.host}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def execute_command(self, command: str, timeout: int = 30) -> CommandResult:
        """Execute a command on the remote server"""
        if not self.client:
            return CommandResult(
                success=False,
                stdout="",
                stderr="Not connected to server",
                exit_code=-1,
                command=command
            )
        
        try:
            # Execute with working directory
            full_command = f"cd {self.config.working_directory} && {command}"
            stdin, stdout, stderr = self.client.exec_command(full_command, timeout=timeout)
            
            # Get results
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            return CommandResult(
                success=(exit_code == 0),
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=exit_code,
                command=command
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                command=command
            )
    
    def analyze_project(self, project_path: str = None) -> Dict[str, Any]:
        """Analyze a project on the remote server"""
        if project_path is None:
            project_path = self.config.working_directory
        
        analysis = {
            "project_path": project_path,
            "structure": {},
            "technologies": [],
            "services": {},
            "recent_activity": []
        }
        
        # Check if path exists
        result = self.execute_command(f"test -d {project_path} && echo 'exists'")
        if not result.success or 'exists' not in result.stdout:
            return {"error": f"Path {project_path} does not exist"}
        
        # Get directory structure
        result = self.execute_command(f"find {project_path} -maxdepth 2 -type d | head -20")
        if result.success:
            analysis["structure"]["directories"] = result.stdout.strip().split('\n')
        
        # Count files by type
        result = self.execute_command(f"""
            find {project_path} -type f -name "*.js" -o -name "*.ts" -o -name "*.py" \
            -o -name "*.go" -o -name "*.java" -o -name "*.rb" | wc -l
        """)
        if result.success:
            analysis["structure"]["code_files"] = int(result.stdout.strip())
        
        # Check for common config files
        configs = ["package.json", "requirements.txt", "go.mod", "pom.xml", "Gemfile", "docker-compose.yml"]
        found_configs = []
        
        for config in configs:
            result = self.execute_command(f"test -f {project_path}/{config} && echo 'found'")
            if result.success and 'found' in result.stdout:
                found_configs.append(config)
        
        analysis["structure"]["config_files"] = found_configs
        
        # Detect technologies
        if "package.json" in found_configs:
            result = self.execute_command(f"cat {project_path}/package.json | grep -E '\"(react|vue|angular|express|fastify)\"'")
            if result.success:
                if "react" in result.stdout:
                    analysis["technologies"].append("React")
                if "express" in result.stdout:
                    analysis["technologies"].append("Express.js")
                if "vue" in result.stdout:
                    analysis["technologies"].append("Vue.js")
        
        if "requirements.txt" in found_configs:
            analysis["technologies"].append("Python")
            result = self.execute_command(f"cat {project_path}/requirements.txt | grep -E '(django|flask|fastapi)'")
            if result.success:
                if "django" in result.stdout:
                    analysis["technologies"].append("Django")
                if "flask" in result.stdout:
                    analysis["technologies"].append("Flask")
                if "fastapi" in result.stdout:
                    analysis["technologies"].append("FastAPI")
        
        if "go.mod" in found_configs:
            analysis["technologies"].append("Go")
        
        if "docker-compose.yml" in found_configs:
            analysis["technologies"].append("Docker")
        
        # Check running services
        result = self.execute_command("docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null | head -10")
        if result.success and result.stdout:
            analysis["services"]["docker_containers"] = result.stdout
        
        # Check Node.js processes
        result = self.execute_command("ps aux | grep node | grep -v grep | head -5")
        if result.success and result.stdout:
            analysis["services"]["node_processes"] = len(result.stdout.strip().split('\n'))
        
        # Check recent file changes
        result = self.execute_command(f"find {project_path} -type f -mtime -7 | head -10")
        if result.success:
            analysis["recent_activity"] = result.stdout.strip().split('\n')
        
        # Check git status if it's a git repo
        result = self.execute_command(f"cd {project_path} && git status --short 2>/dev/null | head -10")
        if result.success and result.stdout:
            analysis["git_status"] = result.stdout
        
        return analysis
    
    def list_projects(self, base_path: str = "/home/ubuntu") -> List[str]:
        """List all projects in a directory"""
        result = self.execute_command(f"find {base_path} -maxdepth 1 -type d | grep -v '^\.' | tail -n +2")
        if result.success:
            return result.stdout.strip().split('\n')
        return []
    
    def check_service_status(self, service_name: str) -> Dict[str, Any]:
        """Check the status of a service"""
        status = {
            "service": service_name,
            "running": False,
            "details": {}
        }
        
        # Check systemctl
        result = self.execute_command(f"systemctl is-active {service_name} 2>/dev/null")
        if result.success and "active" in result.stdout:
            status["running"] = True
            status["details"]["systemctl"] = "active"
        
        # Check docker
        result = self.execute_command(f"docker ps | grep {service_name} 2>/dev/null")
        if result.success and result.stdout:
            status["running"] = True
            status["details"]["docker"] = "running"
        
        # Check process
        result = self.execute_command(f"pgrep -f {service_name} 2>/dev/null")
        if result.success and result.stdout:
            status["running"] = True
            status["details"]["process"] = "found"
        
        return status
    
    def deploy_code(self, local_file: str, remote_path: str) -> bool:
        """Deploy a file to the remote server"""
        if not self.sftp:
            logger.error("SFTP not connected")
            return False
        
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            self.execute_command(f"mkdir -p {remote_dir}")
            
            # Upload file
            self.sftp.put(local_file, remote_path)
            logger.info(f"✅ Deployed {local_file} to {remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def analyze_databases(self) -> Dict[str, Any]:
        """Analyze all databases on the server"""
        db_analysis = {
            "postgresql": {},
            "mysql": {},
            "mongodb": {},
            "redis": {}
        }
        
        # Check PostgreSQL
        result = self.execute_command("sudo -u postgres psql -l 2>/dev/null || psql -U postgres -l 2>/dev/null")
        if result.success and ("List of databases" in result.stdout or "Name" in result.stdout):
            db_analysis["postgresql"]["status"] = "running"
            # Parse database list
            lines = result.stdout.split('\n')
            databases = []
            for line in lines:
                if '|' in line and 'Name' not in line and '---' not in line:
                    parts = line.split('|')
                    if len(parts) > 0:
                        db_name = parts[0].strip()
                        if db_name and db_name not in ['', 'rows)', 'List']:
                            databases.append(db_name)
            db_analysis["postgresql"]["databases"] = databases[:10]  # Limit to 10
            
            # Get PostgreSQL version
            result = self.execute_command("postgres --version 2>/dev/null || psql --version")
            if result.success:
                db_analysis["postgresql"]["version"] = result.stdout.strip()
        
        # Check MySQL/MariaDB
        result = self.execute_command("mysql -e 'SHOW DATABASES;' 2>/dev/null")
        if result.success and "Database" in result.stdout:
            db_analysis["mysql"]["status"] = "running"
            databases = [line.strip() for line in result.stdout.split('\n') 
                        if line.strip() and line.strip() != 'Database']
            db_analysis["mysql"]["databases"] = databases[:10]
            
            # Get MySQL version
            result = self.execute_command("mysql --version")
            if result.success:
                db_analysis["mysql"]["version"] = result.stdout.strip()
        
        # Check MongoDB
        result = self.execute_command("mongosh --eval 'db.adminCommand({listDatabases: 1})' --quiet 2>/dev/null || mongo --eval 'db.adminCommand({listDatabases: 1})' --quiet 2>/dev/null")
        if result.success and ("databases" in result.stdout or "name" in result.stdout):
            db_analysis["mongodb"]["status"] = "running"
            # Try to parse output
            try:
                # Look for database names in the output
                databases = []
                for line in result.stdout.split('\n'):
                    if '"name"' in line:
                        # Extract name from JSON-like output
                        import re
                        match = re.search(r'"name"\s*:\s*"([^"]+)"', line)
                        if match:
                            databases.append(match.group(1))
                db_analysis["mongodb"]["databases"] = databases[:10]
            except:
                db_analysis["mongodb"]["databases"] = []
            
            # Get MongoDB version
            result = self.execute_command("mongod --version 2>/dev/null | head -1")
            if result.success:
                db_analysis["mongodb"]["version"] = result.stdout.strip()
        
        # Check Redis
        result = self.execute_command("redis-cli ping 2>/dev/null")
        if result.success and "PONG" in result.stdout:
            db_analysis["redis"]["status"] = "running"
            
            # Get Redis info
            result = self.execute_command("redis-cli INFO server | grep redis_version")
            if result.success:
                db_analysis["redis"]["version"] = result.stdout.strip().replace("redis_version:", "")
            
            # Get key count
            result = self.execute_command("redis-cli DBSIZE")
            if result.success:
                db_analysis["redis"]["keys"] = result.stdout.strip()
            
            # Get memory usage
            result = self.execute_command("redis-cli INFO memory | grep used_memory_human")
            if result.success:
                db_analysis["redis"]["memory"] = result.stdout.strip().replace("used_memory_human:", "")
        
        return db_analysis
    
    def optimize_query(self, db_type: str, query: str, database: str = None) -> str:
        """Provide query optimization suggestions"""
        if db_type == "postgresql":
            # Run EXPLAIN ANALYZE
            db_flag = f"-d {database}" if database else ""
            result = self.execute_command(f"psql {db_flag} -c 'EXPLAIN ANALYZE {query}' 2>/dev/null")
            if result.success:
                return f"Query Plan:\n{result.stdout}"
        elif db_type == "mysql":
            db_flag = f"-D {database}" if database else ""
            result = self.execute_command(f"mysql {db_flag} -e 'EXPLAIN {query}' 2>/dev/null")
            if result.success:
                return f"Query Plan:\n{result.stdout}"
        elif db_type == "mongodb":
            if database:
                result = self.execute_command(f"mongosh {database} --eval 'db.{query}.explain()' --quiet 2>/dev/null")
                if result.success:
                    return f"Query Plan:\n{result.stdout}"
        return "Unable to analyze query"
    
    def get_database_stats(self, db_type: str, database: str = None) -> Dict[str, Any]:
        """Get detailed statistics for a specific database"""
        stats = {}
        
        if db_type == "postgresql" and database:
            # Get table sizes
            query = f"""
            SELECT schemaname, tablename, 
                   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
            FROM pg_tables 
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 10;
            """
            result = self.execute_command(f"psql -d {database} -c \"{query}\" 2>/dev/null")
            if result.success:
                stats["table_sizes"] = result.stdout
            
            # Get connection count
            result = self.execute_command(f"psql -d {database} -c 'SELECT count(*) FROM pg_stat_activity;' -t 2>/dev/null")
            if result.success:
                stats["connections"] = result.stdout.strip()
                
        elif db_type == "mysql" and database:
            # Get table sizes
            query = f"""
            SELECT table_name, 
                   ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'Size (MB)'
            FROM information_schema.TABLES 
            WHERE table_schema = '{database}'
            ORDER BY (data_length + index_length) DESC
            LIMIT 10;
            """
            result = self.execute_command(f"mysql -D {database} -e \"{query}\" 2>/dev/null")
            if result.success:
                stats["table_sizes"] = result.stdout
                
        elif db_type == "mongodb" and database:
            # Get collection stats
            result = self.execute_command(f"mongosh {database} --eval 'db.stats()' --quiet 2>/dev/null")
            if result.success:
                stats["database_stats"] = result.stdout
                
        elif db_type == "redis":
            # Get detailed Redis stats
            result = self.execute_command("redis-cli INFO stats")
            if result.success:
                stats["redis_stats"] = result.stdout
        
        return stats
    
    def close(self):
        """Close SSH connection"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        logger.info("SSH connection closed")


class ServerAnalyzer:
    """High-level server analysis for agents"""
    
    def __init__(self, ssh_executor: SSHExecutor):
        self.ssh = ssh_executor
    
    def full_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive server analysis"""
        report = {
            "server": self.ssh.config.host,
            "user": self.ssh.config.username,
            "projects": {},
            "services": {},
            "system": {},
            "databases": {}
        }
        
        # Get system info
        result = self.ssh.execute_command("uname -a")
        if result.success:
            report["system"]["kernel"] = result.stdout.strip()
        
        result = self.ssh.execute_command("df -h / | tail -1")
        if result.success:
            report["system"]["disk_usage"] = result.stdout.strip()
        
        result = self.ssh.execute_command("free -h | grep Mem")
        if result.success:
            report["system"]["memory"] = result.stdout.strip()
        
        # List and analyze projects
        projects = self.ssh.list_projects()
        for project_path in projects[:5]:  # Limit to first 5 projects
            project_name = os.path.basename(project_path)
            if project_name and not project_name.startswith('.'):
                logger.info(f"Analyzing project: {project_name}")
                report["projects"][project_name] = self.ssh.analyze_project(project_path)
        
        # Check common services
        services_to_check = ["docker", "nginx", "apache2", "mysql", "postgresql", "redis", "mongodb"]
        for service in services_to_check:
            report["services"][service] = self.ssh.check_service_status(service)
        
        # Analyze databases
        report["databases"] = self.ssh.analyze_databases()
        
        return report
    
    def generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary from analysis"""
        lines = []
        lines.append(f"# Server Analysis Report")
        lines.append(f"\n## Server: {analysis['server']}")
        lines.append(f"User: {analysis['user']}")
        
        if analysis.get('system'):
            lines.append(f"\n## System Information")
            for key, value in analysis['system'].items():
                lines.append(f"- {key}: {value}")
        
        if analysis.get('projects'):
            lines.append(f"\n## Projects Found ({len(analysis['projects'])})")
            for project_name, details in analysis['projects'].items():
                lines.append(f"\n### {project_name}")
                
                if details.get('technologies'):
                    lines.append(f"Technologies: {', '.join(details['technologies'])}")
                
                if details.get('structure'):
                    if details['structure'].get('code_files'):
                        lines.append(f"Code files: {details['structure']['code_files']}")
                    if details['structure'].get('config_files'):
                        lines.append(f"Config files: {', '.join(details['structure']['config_files'])}")
                
                if details.get('git_status'):
                    lines.append(f"Git changes detected")
        
        if analysis.get('services'):
            lines.append(f"\n## Services Status")
            for service, status in analysis['services'].items():
                if status['running']:
                    lines.append(f"✅ {service}: Running")
                else:
                    lines.append(f"❌ {service}: Not running")
        
        if analysis.get('databases'):
            lines.append(f"\n## Database Analysis")
            dbs = analysis['databases']
            
            # PostgreSQL
            if dbs.get('postgresql', {}).get('status') == 'running':
                lines.append(f"\n### PostgreSQL")
                if dbs['postgresql'].get('version'):
                    lines.append(f"Version: {dbs['postgresql']['version']}")
                if dbs['postgresql'].get('databases'):
                    lines.append(f"Databases: {', '.join(dbs['postgresql']['databases'][:5])}")
            
            # MySQL
            if dbs.get('mysql', {}).get('status') == 'running':
                lines.append(f"\n### MySQL/MariaDB")
                if dbs['mysql'].get('version'):
                    lines.append(f"Version: {dbs['mysql']['version']}")
                if dbs['mysql'].get('databases'):
                    lines.append(f"Databases: {', '.join(dbs['mysql']['databases'][:5])}")
            
            # MongoDB
            if dbs.get('mongodb', {}).get('status') == 'running':
                lines.append(f"\n### MongoDB")
                if dbs['mongodb'].get('version'):
                    lines.append(f"Version: {dbs['mongodb']['version']}")
                if dbs['mongodb'].get('databases'):
                    lines.append(f"Databases: {', '.join(dbs['mongodb']['databases'][:5])}")
            
            # Redis
            if dbs.get('redis', {}).get('status') == 'running':
                lines.append(f"\n### Redis")
                if dbs['redis'].get('version'):
                    lines.append(f"Version: {dbs['redis']['version']}")
                if dbs['redis'].get('keys'):
                    lines.append(f"Total keys: {dbs['redis']['keys']}")
                if dbs['redis'].get('memory'):
                    lines.append(f"Memory usage: {dbs['redis']['memory']}")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test with your Ubuntu server
    config = SSHConfig(
        host="13.38.102.28",
        username="ubuntu",
        key_path="/Users/jankootstra/blockchain.pem",
        working_directory="/home/ubuntu"
    )
    
    executor = SSHExecutor(config)
    if executor.connect():
        analyzer = ServerAnalyzer(executor)
        
        # Quick test
        result = executor.execute_command("ls -la | head -5")
        print(f"Command result: {result.stdout}")
        
        # Analyze a specific project
        project_analysis = executor.analyze_project("/home/ubuntu/dims_agents")
        print(f"\nProject Analysis: {json.dumps(project_analysis, indent=2)}")
        
        executor.close()