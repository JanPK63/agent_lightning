#!/usr/bin/env python3
"""
AI Task Executor - Real AI Processing for Agent Framework
Integrates with LLMs to actually execute tasks and return meaningful results
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from system_executor import get_system_executor
import anthropic
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AITaskExecutor:
    """Executes tasks using real AI models"""
    
    def __init__(self):
        self.dal = DataAccessLayer("ai_executor")
        
        # Initialize system executor for real system access
        self.system_executor = get_system_executor()
        
        # Initialize AI clients
        self.claude_client = None
        self.openai_client = None
        
        # Try to initialize Claude
        claude_key = os.getenv('ANTHROPIC_API_KEY')
        if claude_key:
            self.claude_client = anthropic.Anthropic(api_key=claude_key)
            logger.info("Claude AI client initialized")
        
        # Try to initialize OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)
            logger.info("OpenAI client initialized")
            
        # Agent specializations and their prompts
        self.agent_prompts = {
            'data_scientist': """You are an expert Data Scientist specializing in data analysis, 
                machine learning, and statistical modeling. Analyze code, data structures, and architectures 
                to provide comprehensive insights.""",
            
            'security_expert': """You are a Security Expert specializing in cybersecurity, 
                vulnerability assessment, and secure coding practices. Analyze systems for security 
                weaknesses and provide recommendations.""",
            
            'devops_engineer': """You are a DevOps Engineer specializing in CI/CD, infrastructure, 
                monitoring, and deployment. Analyze systems for operational efficiency and provide 
                optimization recommendations.""",
            
            'full_stack_developer': """You are a Full Stack Developer with expertise in frontend, 
                backend, databases, and system architecture. Provide comprehensive technical analysis 
                and implementation recommendations.""",
            
            'blockchain_developer': """You are a Blockchain Developer specializing in distributed 
                ledgers, smart contracts, and decentralized systems. Analyze blockchain implementations 
                and provide architectural insights.""",
            
            'ai_researcher': """You are an AI Researcher specializing in machine learning, 
                neural networks, and AI system design. Analyze AI implementations and provide 
                research-based recommendations.""",
            
            'cloud_architect': """You are a Cloud Architect specializing in cloud infrastructure, 
                scalability, and distributed systems. Analyze architectures and provide cloud 
                optimization strategies.""",
            
            'test_engineer': """You are a Test Engineer specializing in comprehensive software testing.
                When asked to test an application:
                1. Analyze the application structure and identify testable components
                2. Create and execute a comprehensive test plan covering: unit tests, integration tests, 
                   end-to-end tests, performance tests, security tests, and API tests
                3. Report specific test results with pass/fail status, bugs found, performance metrics,
                   security concerns, and actionable recommendations
                Always perform actual testing and provide detailed results, not just analysis.""",
            
            'mobile_developer': """You are a Mobile Developer specializing in iOS and Android 
                development. Analyze mobile applications and provide platform-specific insights.""",
            
            'database_specialist': """You are a Database Specialist with expertise in SQL, NoSQL, 
                and data modeling. Analyze database designs and provide optimization recommendations.""",
            
            'system_architect': """You are a System Architect specializing in enterprise architecture, 
                system design, and integration patterns. Provide high-level architectural analysis 
                and strategic recommendations."""
        }
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task using AI and return real results"""
        
        # Get task from database
        task = self.dal.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        
        agent_id = task['agent_id']
        description = task['description']
        context = task.get('context', {})
        model_preference = context.get('model', 'claude-3-haiku')  # Get model from context
        
        logger.info(f"Executing task {task_id} for agent {agent_id} with model {model_preference}")
        
        # Update task status to started
        self.dal.update_task_status(task_id, "started")
        
        try:
            # Check if this is a system/SSH task
            if 'ssh' in description.lower() or 'analyze' in description.lower() and 'system' in description.lower():
                logger.info(f"Detected system task for agent {agent_id}")
                
                # Execute real system analysis
                system_result = await self.system_executor.execute_system_task(description, agent_id)
                
                if 'error' not in system_result:
                    # Format the real results for the agent to interpret
                    agent_prompt = self.agent_prompts.get(agent_id, 
                        "You are an AI assistant. Help with the following task.")
                    
                    full_prompt = f"""{agent_prompt}

Task: {description}

ACTUAL SYSTEM ANALYSIS RESULTS:
{json.dumps(system_result, indent=2)}

Based on these REAL system analysis results, provide:
1. System Status Summary
2. Key Findings and Issues
3. Performance Metrics
4. Security Concerns
5. Recommendations for Improvements
6. Strategic Architecture Recommendations
"""
                    # Execute with AI to interpret the results
                    result = await self._call_ai(full_prompt, agent_id, model_preference)
                    result['system_analysis'] = system_result
                else:
                    result = system_result
                
                # Update task with result
                self.dal.update_task_status(task_id, "completed", result)
                logger.info(f"System task {task_id} completed")
                return result
            
            # Get agent specialization prompt
            agent_prompt = self.agent_prompts.get(agent_id, 
                "You are an AI assistant. Help with the following task.")
            
            # Special handling for test_engineer - ALWAYS execute actual tests
            if agent_id == 'test_engineer':
                # Extract path from description if present
                import re
                path_match = re.search(r'/Users/[^\s]+', description)
                if path_match:
                    project_path = path_match.group(0)
                    
                    # Execute actual tests FIRST
                    logger.info(f"Executing actual tests for {project_path}")
                    test_results = await self._execute_tests(project_path)
                    
                    # If we got real test results, return them directly with AI summary
                    if test_results and "Error: Path" not in test_results:
                        # Add test results to the prompt for AI to summarize
                        full_prompt = f"""{agent_prompt}

Task: {description}

ACTUAL TEST EXECUTION RESULTS (THIS IS REAL OUTPUT FROM RUNNING TESTS):
================================================================================
{test_results}
================================================================================

Based on these ACTUAL test execution results above, provide:
1. Test Summary with specific counts (e.g., "5 tests passed, 2 failed")
2. List the specific test failures and their error messages
3. Performance metrics extracted from the test output
4. Security issues found during testing
5. Specific recommendations for fixing the failures
6. Overall assessment of the application's test coverage

IMPORTANT: Base your response ONLY on the actual test results shown above, not on general analysis.
"""
                    else:
                        # No valid path, but still try to run tests if possible
                        full_prompt = f"""{agent_prompt}

Task: {description}

Could not find valid project path. Attempting to analyze testing requirements instead.
To run actual tests, please provide a valid project path like: /Users/username/project
"""
                else:
                    full_prompt = f"""{agent_prompt}

Task: {description}

Note: No project path found. Please provide a path to test, e.g., /Users/username/project
To execute actual tests, I need a valid project directory."""
            
            # Check if task involves file/project analysis (for other agents)
            elif '/Users/' in description or 'analyze' in description.lower():
                # Extract path from description if present
                import re
                path_match = re.search(r'/Users/[^\s]+', description)
                if path_match:
                    project_path = path_match.group(0)
                    
                    # Read project structure and files
                    project_context = await self._analyze_project(project_path)
                    
                    # Add project context to the task
                    full_prompt = f"""{agent_prompt}

Task: {description}

Project Analysis:
{project_context}

Please provide a comprehensive analysis including:
1. Architecture Overview
2. Key Components and Technologies
3. Design Patterns Used
4. Recommendations for Improvement
5. Security Considerations
6. Scalability Analysis
"""
                else:
                    full_prompt = f"{agent_prompt}\n\nTask: {description}"
            else:
                full_prompt = f"{agent_prompt}\n\nTask: {description}"
            
            # Execute with AI - pass model preference
            result = await self._call_ai(full_prompt, agent_id, model_preference)
            
            # Update task with real result
            self.dal.update_task_status(task_id, "completed", result)
            
            logger.info(f"Task {task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            error_result = {"error": str(e)}
            self.dal.update_task_status(task_id, "failed", error_result)
            return error_result
    
    async def _execute_tests(self, project_path: str) -> str:
        """Execute actual tests on the project"""
        import subprocess
        import os
        
        test_results = []
        
        # Check if path exists
        if not os.path.exists(project_path):
            return f"Error: Path {project_path} does not exist"
        
        test_results.append(f"=== ACTUAL TEST EXECUTION FOR: {project_path} ===")
        test_results.append(f"Started at: {datetime.now().isoformat()}\n")
        
        # Detect project type and run appropriate tests
        try:
            # Check for Swift/iOS project files first
            if any(f.endswith('.swift') for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f))):
                test_results.append("=== Swift/iOS Project Detected ===\n")
                test_results.append("Running Swift tests...\n")
                
                # Try to run swift test
                try:
                    result = subprocess.run(
                        ['swift', 'test'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        test_results.append("✅ Swift Test Results:\n")
                    else:
                        test_results.append("❌ Swift Test Results (some failures):\n")
                    test_results.append(result.stdout if result.stdout else "No output")
                    if result.stderr:
                        test_results.append(f"\nErrors:\n{result.stderr}")
                except subprocess.TimeoutExpired:
                    test_results.append("⏱️ Swift tests timed out after 60 seconds")
                except FileNotFoundError:
                    test_results.append("⚠️ Swift not installed or not in PATH")
                except Exception as e:
                    test_results.append(f"⚠️ Could not run swift test: {e}")
            
            # Check for package.json (JavaScript/Node.js project)
            elif os.path.exists(os.path.join(project_path, 'package.json')):
                test_results.append("=== JavaScript/Node.js Project Detected ===\n")
                
                # Try to run npm test
                try:
                    result = subprocess.run(
                        ['npm', 'test'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    test_results.append(f"NPM Test Results:\n{result.stdout}\n{result.stderr}")
                except subprocess.TimeoutExpired:
                    test_results.append("NPM tests timed out after 30 seconds")
                except Exception as e:
                    test_results.append(f"Could not run npm test: {e}")
            
            # Check for requirements.txt or setup.py (Python project)
            elif os.path.exists(os.path.join(project_path, 'requirements.txt')) or \
                 os.path.exists(os.path.join(project_path, 'setup.py')):
                test_results.append("=== Python Project Detected ===\n")
                
                # Try to run pytest
                try:
                    result = subprocess.run(
                        ['python', '-m', 'pytest', '-v'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    test_results.append(f"Pytest Results:\n{result.stdout}\n{result.stderr}")
                except subprocess.TimeoutExpired:
                    test_results.append("Python tests timed out after 30 seconds")
                except Exception as e:
                    test_results.append(f"Could not run pytest: {e}")
            
            # Check for Gemfile (Ruby project)
            elif os.path.exists(os.path.join(project_path, 'Gemfile')):
                test_results.append("=== Ruby Project Detected ===\n")
                
                # Try to run rspec
                try:
                    result = subprocess.run(
                        ['bundle', 'exec', 'rspec'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    test_results.append(f"RSpec Results:\n{result.stdout}\n{result.stderr}")
                except Exception as e:
                    test_results.append(f"Could not run rspec: {e}")
            
            # Check for pom.xml or build.gradle (Java project)
            elif os.path.exists(os.path.join(project_path, 'pom.xml')):
                test_results.append("=== Java/Maven Project Detected ===\n")
                
                # Try to run Maven tests
                try:
                    result = subprocess.run(
                        ['mvn', 'test'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    test_results.append(f"Maven Test Results:\n{result.stdout}\n{result.stderr}")
                except Exception as e:
                    test_results.append(f"Could not run Maven tests: {e}")
            
            # Check for .xcodeproj or .xcworkspace (iOS project)
            elif any(f.endswith(('.xcodeproj', '.xcworkspace')) for f in os.listdir(project_path)):
                test_results.append("=== iOS/Swift Project Detected ===\n")
                
                # Try to run xcodebuild test
                try:
                    result = subprocess.run(
                        ['xcodebuild', 'test', '-scheme', 'YourScheme', '-destination', 
                         'platform=iOS Simulator,name=iPhone 14'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    test_results.append(f"XCode Test Results:\n{result.stdout[-5000:]}")  # Last 5000 chars
                except Exception as e:
                    test_results.append(f"Could not run XCode tests: {e}")
            
            # Generic file analysis as fallback
            if not test_results or len(test_results) == 1:
                test_results.append("\n=== No standard test framework detected ===")
                test_results.append("Performing static analysis...")
                
                # Count files by type
                file_counts = {}
                for root, dirs, files in os.walk(project_path):
                    for file in files:
                        ext = os.path.splitext(file)[1]
                        file_counts[ext] = file_counts.get(ext, 0) + 1
                
                test_results.append(f"File statistics: {file_counts}")
                
                # Look for test files
                test_files = []
                for root, dirs, files in os.walk(project_path):
                    for file in files:
                        if 'test' in file.lower() or 'spec' in file.lower():
                            test_files.append(os.path.join(root, file))
                
                if test_files:
                    test_results.append(f"Found {len(test_files)} test files:")
                    for tf in test_files[:10]:  # Show first 10
                        test_results.append(f"  - {tf}")
                else:
                    test_results.append("No test files found in project")
            
        except Exception as e:
            test_results.append(f"Error during test execution: {e}")
        
        return "\n".join(test_results)
    
    async def _analyze_project(self, project_path: str) -> str:
        """Analyze a project directory and return context"""
        context = []
        
        try:
            path = Path(project_path)
            if not path.exists():
                return f"Project path not found: {project_path}"
            
            context.append(f"Project: {path.name}")
            context.append(f"Location: {project_path}\n")
            
            # Get project structure
            context.append("Project Structure:")
            for item in self._get_tree_structure(path, max_depth=3):
                context.append(item)
            
            # Identify project type and key files
            if (path / "Package.swift").exists():
                context.append("\nProject Type: Swift/iOS Project")
                # Read Package.swift for dependencies
                try:
                    with open(path / "Package.swift", 'r') as f:
                        content = f.read()[:1000]
                        context.append(f"\nPackage.swift (excerpt):\n{content}")
                except:
                    pass
            
            # Look for key configuration files
            key_files = ['README.md', 'Info.plist', 'Podfile', 'Cartfile', 
                        'Package.json', 'requirements.txt', 'pom.xml', 'build.gradle']
            
            for filename in key_files:
                file_path = self._find_file(path, filename)
                if file_path:
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()[:500]
                            context.append(f"\n{filename}:\n{content}")
                    except:
                        pass
            
            # Analyze code files
            code_files = list(path.rglob("*.swift"))[:5]  # Sample Swift files
            if code_files:
                context.append("\nSample Code Files:")
                for code_file in code_files:
                    try:
                        with open(code_file, 'r') as f:
                            content = f.read()[:300]
                            context.append(f"\n{code_file.name}:\n{content}...")
                    except:
                        pass
            
            return "\n".join(context)
            
        except Exception as e:
            return f"Error analyzing project: {e}"
    
    def _get_tree_structure(self, path: Path, prefix: str = "", max_depth: int = 3, 
                           current_depth: int = 0) -> List[str]:
        """Get directory tree structure"""
        if current_depth >= max_depth:
            return []
        
        items = []
        try:
            contents = list(path.iterdir())
            for i, item in enumerate(sorted(contents)[:20]):  # Limit items per level
                if item.name.startswith('.'):
                    continue
                    
                is_last = i == len(contents) - 1
                current_prefix = "└── " if is_last else "├── "
                items.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    items.extend(self._get_tree_structure(
                        item, next_prefix, max_depth, current_depth + 1
                    ))
        except PermissionError:
            pass
        
        return items
    
    def _find_file(self, path: Path, filename: str) -> Optional[Path]:
        """Find a file in the project directory"""
        try:
            for file_path in path.rglob(filename):
                return file_path
        except:
            pass
        return None
    
    async def _call_ai(self, prompt: str, agent_id: str, model_preference: str = "claude-3-haiku") -> Dict[str, Any]:
        """Call AI model to process the task with specified model preference"""
        
        # Map model preferences to actual model names
        model_mapping = {
            "gpt-4o": "gpt-4",
            "gpt-4o-mini": "gpt-4-0613",
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307"
        }
        
        actual_model = model_mapping.get(model_preference, model_preference)
        
        # Determine which client to use based on model preference
        use_openai = model_preference.startswith('gpt')
        use_claude = model_preference.startswith('claude')
        
        logger.info(f"Using model {actual_model} for agent {agent_id}")
        
        # Try preferred model first
        if use_openai and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "system", "content": self.agent_prompts.get(agent_id, "")},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.7
                )
                
                return {
                    "agent": agent_id,
                    "model": model_preference,
                    "analysis": response.choices[0].message.content,
                    "execution_time": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"OpenAI API error with {actual_model}: {e}")
                # Fall back to GPT-3.5 if GPT-4 fails
                if "gpt-4" in actual_model:
                    try:
                        response = self.openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": self.agent_prompts.get(agent_id, "")},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=2000,
                            temperature=0.7
                        )
                        return {
                            "agent": agent_id,
                            "model": "gpt-3.5-turbo (fallback from " + model_preference + ")",
                            "analysis": response.choices[0].message.content,
                            "execution_time": datetime.now().isoformat()
                        }
                    except Exception as e2:
                        logger.error(f"Fallback to GPT-3.5 also failed: {e2}")
        
        elif use_claude and self.claude_client:
            try:
                response = self.claude_client.messages.create(
                    model=actual_model,
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return {
                    "agent": agent_id,
                    "model": model_preference,
                    "analysis": response.content[0].text,
                    "execution_time": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Claude API error with {actual_model}: {e}")
                # Fall back to Haiku if other models fail
                if actual_model != "claude-3-haiku-20240307":
                    try:
                        response = self.claude_client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=4000,
                            temperature=0.7,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        return {
                            "agent": agent_id,
                            "model": "claude-3-haiku (fallback from " + model_preference + ")",
                            "analysis": response.content[0].text,
                            "execution_time": datetime.now().isoformat()
                        }
                    except Exception as e2:
                        logger.error(f"Fallback to Claude Haiku also failed: {e2}")
        
        # Try any available client as fallback
        if self.claude_client:
            try:
                response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return {
                    "agent": agent_id,
                    "model": "claude-3-haiku (fallback)",
                    "analysis": response.content[0].text,
                    "execution_time": datetime.now().isoformat(),
                    "note": f"Requested model {model_preference} not available"
                }
            except Exception as e:
                logger.error(f"Final Claude fallback error: {e}")
        
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.agent_prompts.get(agent_id, "")},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
                
                return {
                    "agent": agent_id,
                    "model": "gpt-3.5-turbo (fallback)",
                    "analysis": response.choices[0].message.content,
                    "execution_time": datetime.now().isoformat(),
                    "note": f"Requested model {model_preference} not available"
                }
            except Exception as e:
                logger.error(f"Final OpenAI fallback error: {e}")
        
        # Fallback to local analysis (without AI)
        task_desc = prompt.split('Task:')[1].split('\\n')[0] if 'Task:' in prompt else 'Unknown'
        
        return {
            "agent": agent_id,
            "model": "local-analysis",
            "analysis": f"""
Based on the task description, here's a basic analysis:

Task: {task_desc}

Without API keys configured for {model_preference}, I can provide a structural analysis:
1. The project appears to be an iOS application
2. Key technologies likely include Swift, UIKit/SwiftUI
3. Consider implementing MVVM or MVC architecture
4. Security: Implement keychain for sensitive data
5. Performance: Use lazy loading and caching strategies

Note: For detailed AI analysis, please configure ANTHROPIC_API_KEY or OPENAI_API_KEY
            """,
            "execution_time": datetime.now().isoformat(),
            "note": f"API keys not configured - requested model: {model_preference}"
        }


async def process_pending_tasks():
    """Process all pending tasks"""
    executor = AITaskExecutor()
    
    # Get pending tasks
    tasks = executor.dal.list_tasks(status="pending")
    
    for task in tasks:
        logger.info(f"Processing task {task['id']}")
        await executor.execute_task(task['id'])
        await asyncio.sleep(1)  # Rate limiting


async def main():
    """Main service loop"""
    executor = AITaskExecutor()
    logger.info("AI Task Executor Service started")
    
    while True:
        try:
            # Get pending tasks
            tasks = executor.dal.list_tasks(status="pending")
            
            for task in tasks:
                logger.info(f"Processing task {task['id']}")
                await executor.execute_task(task['id'])
                
            # Also check for started tasks that might be stuck
            started_tasks = executor.dal.list_tasks(status="started")
            for task in started_tasks:
                # If task has been started for more than 5 minutes, retry it
                if task.get('started_at'):
                    # Handle timezone-aware datetime
                    from dateutil import parser
                    started_time = parser.parse(task['started_at'])
                    # Make current time timezone-aware if needed
                    now = datetime.now(started_time.tzinfo) if started_time.tzinfo else datetime.now()
                    if (now - started_time).total_seconds() > 300:
                        logger.info(f"Retrying stuck task {task['id']}")
                        await executor.execute_task(task['id'])
            
            # Sleep for a bit before checking again
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    # Run task processor service
    asyncio.run(main())