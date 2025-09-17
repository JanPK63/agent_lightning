#!/usr/bin/env python3
"""
System Executor - Gives AI agents real system access capabilities
Allows agents to execute SSH commands, run system analysis, and perform real operations
"""

import os
import sys
import subprocess
import paramiko
import re
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemExecutor:
    """Execute system commands for AI agents"""
    
    def __init__(self):
        self.ssh_clients = {}
        
    async def execute_system_task(self, task_description: str, agent_id: str) -> Dict[str, Any]:
        """
        Execute system tasks based on task description
        """
        
        # Check for SSH commands
        ssh_pattern = r'ssh\s+(?:-i\s+"([^"]+)"\s+)?([^@]+)@([\d\.]+|[\w\.-]+)'
        ssh_match = re.search(ssh_pattern, task_description)
        
        if ssh_match:
            key_file = ssh_match.group(1)
            username = ssh_match.group(2)
            hostname = ssh_match.group(3)
            
            logger.info(f"Agent {agent_id} requesting SSH access to {username}@{hostname}")
            
            # Perform SSH analysis
            return await self._perform_ssh_analysis(hostname, username, key_file, task_description)
        
        # Check for local system commands
        elif any(cmd in task_description.lower() for cmd in ['analyze', 'check', 'status', 'test']):
            return await self._perform_local_analysis(task_description)
        
        return {
            "error": "Could not determine system task type",
            "description": task_description
        }
    
    async def _perform_ssh_analysis(self, hostname: str, username: str, key_file: str, task: str) -> Dict[str, Any]:
        """
        Perform analysis on remote system via SSH
        """
        results = {
            "type": "ssh_analysis",
            "hostname": hostname,
            "username": username,
            "analysis": {}
        }
        
        try:
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect
            if key_file:
                key_path = os.path.expanduser(key_file)
                if os.path.exists(key_path):
                    ssh.connect(hostname, username=username, key_filename=key_path)
                else:
                    return {"error": f"Key file not found: {key_file}"}
            else:
                # Try with default keys
                ssh.connect(hostname, username=username)
            
            # Perform analysis
            commands = {
                "system_info": "uname -a",
                "memory": "free -h",
                "disk": "df -h",
                "processes": "ps aux | head -20",
                "services": "systemctl list-units --type=service --state=running | head -20",
                "docker": "docker ps 2>/dev/null || echo 'Docker not available'",
                "network": "netstat -tuln | head -20",
                "logs": "tail -n 50 /var/log/syslog 2>/dev/null || tail -n 50 /var/log/messages 2>/dev/null || echo 'No system logs accessible'"
            }
            
            for name, cmd in commands.items():
                stdin, stdout, stderr = ssh.exec_command(cmd)
                output = stdout.read().decode('utf-8')
                error = stderr.read().decode('utf-8')
                
                results["analysis"][name] = {
                    "output": output[:1000],  # Limit output size
                    "error": error if error else None
                }
            
            # Check for specific applications
            if "blockchain" in task.lower() or "ethereum" in task.lower():
                # Check for blockchain nodes
                stdin, stdout, stderr = ssh.exec_command("ps aux | grep -E 'geth|parity|bitcoin|ethereum' | grep -v grep")
                blockchain_processes = stdout.read().decode('utf-8')
                results["analysis"]["blockchain"] = {
                    "processes": blockchain_processes if blockchain_processes else "No blockchain processes found"
                }
            
            ssh.close()
            
            # Generate summary
            results["summary"] = self._generate_analysis_summary(results["analysis"])
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"SSH analysis failed: {e}")
        
        return results
    
    async def _perform_local_analysis(self, task: str) -> Dict[str, Any]:
        """
        Perform analysis on local system
        """
        results = {
            "type": "local_analysis",
            "analysis": {}
        }
        
        try:
            # Basic system information
            commands = {
                "hostname": ["hostname"],
                "system": ["uname", "-a"],
                "memory": ["vm_stat"],  # macOS specific
                "disk": ["df", "-h"],
                "processes": ["ps", "aux"],
                "network": ["netstat", "-an"]
            }
            
            for name, cmd in commands.items():
                try:
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                    results["analysis"][name] = output[:1000]  # Limit output
                except subprocess.CalledProcessError as e:
                    results["analysis"][name] = f"Error: {e.output}"
                except FileNotFoundError:
                    results["analysis"][name] = f"Command not found: {' '.join(cmd)}"
            
            # Check for specific services if mentioned in task
            if "docker" in task.lower():
                try:
                    docker_ps = subprocess.check_output(["docker", "ps"], text=True)
                    results["analysis"]["docker"] = docker_ps
                except:
                    results["analysis"]["docker"] = "Docker not running or not installed"
            
            if "kubernetes" in task.lower() or "k8s" in task.lower():
                try:
                    kubectl_nodes = subprocess.check_output(["kubectl", "get", "nodes"], text=True)
                    kubectl_pods = subprocess.check_output(["kubectl", "get", "pods", "--all-namespaces"], text=True)
                    results["analysis"]["kubernetes"] = {
                        "nodes": kubectl_nodes,
                        "pods": kubectl_pods[:1000]
                    }
                except:
                    results["analysis"]["kubernetes"] = "Kubernetes not configured"
            
            # Generate summary
            results["summary"] = self._generate_analysis_summary(results["analysis"])
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Local analysis failed: {e}")
        
        return results
    
    def _generate_analysis_summary(self, analysis: Dict) -> str:
        """
        Generate a summary of the analysis
        """
        summary_parts = []
        
        # System info
        if "system_info" in analysis or "system" in analysis:
            sys_info = analysis.get("system_info", analysis.get("system", {}))
            if isinstance(sys_info, dict):
                summary_parts.append(f"System: {sys_info.get('output', 'Unknown')[:100]}")
            else:
                summary_parts.append(f"System: {str(sys_info)[:100]}")
        
        # Memory
        if "memory" in analysis:
            mem_info = analysis["memory"]
            if isinstance(mem_info, dict) and "output" in mem_info:
                summary_parts.append("Memory status captured")
            else:
                summary_parts.append("Memory: " + str(mem_info)[:100])
        
        # Services
        if "services" in analysis:
            services = analysis["services"]
            if isinstance(services, dict) and "output" in services:
                service_lines = services["output"].split('\n')
                running_count = len([l for l in service_lines if 'running' in l.lower()])
                summary_parts.append(f"Services: {running_count} running services detected")
        
        # Docker
        if "docker" in analysis:
            docker_info = analysis["docker"]
            if isinstance(docker_info, str):
                if "CONTAINER ID" in docker_info:
                    container_count = len(docker_info.split('\n')) - 1
                    summary_parts.append(f"Docker: {container_count} containers running")
                else:
                    summary_parts.append("Docker: Not available or no containers")
        
        # Blockchain
        if "blockchain" in analysis:
            blockchain = analysis["blockchain"]
            if isinstance(blockchain, dict) and blockchain.get("processes"):
                summary_parts.append("Blockchain: Node processes detected")
            else:
                summary_parts.append("Blockchain: No blockchain nodes detected")
        
        return "\n".join(summary_parts)


# Singleton instance
_system_executor = None

def get_system_executor():
    global _system_executor
    if _system_executor is None:
        _system_executor = SystemExecutor()
    return _system_executor