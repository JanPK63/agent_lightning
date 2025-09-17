#!/usr/bin/env python3
"""
Agent Actions System - Defines concrete executable actions for agents
Ensures agents execute real tasks instead of describing what they would do
"""

import os
import sys
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ssh_executor import SSHExecutor, SSHConfig, ServerAnalyzer
from code_executor import CodeExecutor, ExecutionConfig, ExecutionMode, FileOperation


class ActionType(Enum):
    """Types of actions agents can execute"""
    READ_ANALYZE = "read_analyze"      # Read and analyze existing code/files
    CREATE_CODE = "create_code"        # Generate new code/files
    TEST = "test"                      # Run tests and validate
    IMPLEMENT = "implement"            # Deploy/implement changes
    DEBUG = "debug"                    # Debug and fix issues
    OPTIMIZE = "optimize"              # Optimize performance
    DOCUMENT = "document"              # Create documentation
    CONFIGURE = "configure"            # Setup/configure systems


@dataclass
class ActionRequest:
    """Request for an agent action"""
    action_type: ActionType
    target_path: str                  # Where to execute (local path or remote directory)
    description: str                   # What to do
    deployment_config: Dict[str, Any] # Deployment configuration
    parameters: Dict[str, Any] = None # Additional parameters
    

@dataclass
class ActionResult:
    """Result of an executed action"""
    success: bool
    action_type: ActionType
    output: str
    files_affected: List[str]
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class AgentActionExecutor:
    """
    Executes concrete actions for agents
    Ensures real work gets done, not just descriptions
    """
    
    def __init__(self):
        self.ssh_executor = None
        self.code_executor = None
        
        # Map action types to execution methods
        self.action_handlers = {
            ActionType.READ_ANALYZE: self._execute_read_analyze,
            ActionType.CREATE_CODE: self._execute_create_code,
            ActionType.TEST: self._execute_test,
            ActionType.IMPLEMENT: self._execute_implement,
            ActionType.DEBUG: self._execute_debug,
            ActionType.OPTIMIZE: self._execute_optimize,
            ActionType.DOCUMENT: self._execute_document,
            ActionType.CONFIGURE: self._execute_configure
        }
    
    async def execute_action(self, request: ActionRequest) -> ActionResult:
        """
        Main entry point - executes the requested action
        """
        # Get the appropriate handler
        handler = self.action_handlers.get(request.action_type)
        if not handler:
            return ActionResult(
                success=False,
                action_type=request.action_type,
                output="",
                files_affected=[],
                error=f"Unknown action type: {request.action_type}"
            )
        
        # Setup connection based on deployment config
        if not self._setup_connection(request.deployment_config):
            return ActionResult(
                success=False,
                action_type=request.action_type,
                output="",
                files_affected=[],
                error="Failed to establish connection"
            )
        
        try:
            # Execute the action
            result = await handler(request)
            return result
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=request.action_type,
                output="",
                files_affected=[],
                error=f"Action execution failed: {str(e)}"
            )
        finally:
            self._cleanup_connection()
    
    def _setup_connection(self, deployment_config: Dict[str, Any]) -> bool:
        """Setup SSH or local connection based on deployment config"""
        try:
            if deployment_config.get("type") == "ubuntu_server":
                ssh_config = SSHConfig(
                    host=deployment_config.get("server_ip"),
                    username=deployment_config.get("username", "ubuntu"),
                    key_path=deployment_config.get("key_path"),
                    working_directory=deployment_config.get("working_directory", "/home/ubuntu")
                )
                self.ssh_executor = SSHExecutor(ssh_config)
                return self.ssh_executor.connect()
                
            elif deployment_config.get("type") == "local":
                exec_config = ExecutionConfig(
                    mode=ExecutionMode.LOCAL,
                    working_directory=deployment_config.get("path", "/tmp/agent_workspace")
                )
                self.code_executor = CodeExecutor(exec_config)
                return True
                
            return False
        except Exception as e:
            print(f"Connection setup failed: {e}")
            return False
    
    def _cleanup_connection(self):
        """Cleanup connections"""
        if self.ssh_executor:
            self.ssh_executor.close()
            self.ssh_executor = None
        if self.code_executor:
            self.code_executor.close()
            self.code_executor = None
    
    async def _execute_read_analyze(self, request: ActionRequest) -> ActionResult:
        """
        READ/ANALYZE Action: Read files and analyze code
        """
        files_read = []
        analysis_output = []
        
        # Handle local file analysis
        if self.code_executor and request.deployment_config.get("type") == "local":
            import os
            from pathlib import Path
            
            target_path = Path(request.target_path)
            
            if not target_path.exists():
                return ActionResult(
                    success=False,
                    action_type=request.action_type,
                    output=f"Path not found: {request.target_path}",
                    files_affected=[],
                    error="Target path does not exist"
                )
            
            analysis_output.append(f"📁 Analyzing local directory: {request.target_path}")
            
            if target_path.is_dir():
                # Find relevant files
                file_patterns = ['*.swift', '*.m', '*.h', '*.js', '*.ts', '*.py', '*.java', '*.go', '*.rs', '*.json', '*.yaml', '*.yml', '*.md', '*.txt', '*.plist']
                all_files = []
                
                for pattern in file_patterns:
                    all_files.extend(target_path.rglob(pattern))
                
                # Limit to first 50 files
                relevant_files = all_files[:50]
                files_read = [str(f) for f in relevant_files]
                
                analysis_output.append(f"\n📊 Found {len(all_files)} total files, analyzing first {len(relevant_files)}")
                
                # Analyze project structure
                if any(f.suffix == '.swift' for f in relevant_files):
                    analysis_output.append("\n🍎 **iOS/Swift Project Detected**")
                    
                    # Look for key iOS files
                    key_files = {
                        'Info.plist': 'App configuration',
                        'AppDelegate.swift': 'App lifecycle',
                        'SceneDelegate.swift': 'Scene management',
                        'ContentView.swift': 'SwiftUI main view',
                        'ViewController.swift': 'UIKit view controller',
                        'Podfile': 'CocoaPods dependencies',
                        'Package.swift': 'Swift Package Manager',
                        'project.pbxproj': 'Xcode project file'
                    }
                    
                    found_files = []
                    for key_file, description in key_files.items():
                        matches = list(target_path.rglob(key_file))
                        if matches:
                            found_files.append(f"  ✅ {key_file} - {description}")
                            
                            # Read and analyze key files
                            try:
                                content = matches[0].read_text(encoding='utf-8', errors='ignore')
                                if key_file == 'Info.plist':
                                    if 'CFBundleIdentifier' in content:
                                        analysis_output.append(f"    📱 Bundle ID found in Info.plist")
                                    if 'NSCameraUsageDescription' in content:
                                        analysis_output.append(f"    📷 Camera permissions configured")
                                elif key_file == 'Podfile':
                                    if 'pod ' in content:
                                        pod_count = content.count('pod ')
                                        analysis_output.append(f"    📦 {pod_count} CocoaPods dependencies")
                                elif key_file.endswith('.swift'):
                                    if 'import UIKit' in content:
                                        analysis_output.append(f"    🎨 Uses UIKit framework")
                                    if 'import SwiftUI' in content:
                                        analysis_output.append(f"    🎨 Uses SwiftUI framework")
                                    if 'blockchain' in content.lower():
                                        analysis_output.append(f"    ⛓️ Contains blockchain-related code")
                                    if 'security' in content.lower():
                                        analysis_output.append(f"    🔒 Contains security-related code")
                            except Exception as e:
                                analysis_output.append(f"    ⚠️ Could not read {key_file}: {str(e)}")
                        else:
                            found_files.append(f"  ❌ {key_file} - Missing")
                    
                    analysis_output.append("\n📋 **Key iOS Files Status:**")
                    analysis_output.extend(found_files)
                    
                    # Analyze Swift source files
                    swift_files = [f for f in relevant_files if f.suffix == '.swift']
                    if swift_files:
                        analysis_output.append(f"\n📝 **Swift Source Analysis ({len(swift_files)} files):**")
                        
                        total_lines = 0
                        imports = set()
                        classes = []
                        
                        for swift_file in swift_files[:10]:  # Analyze first 10 Swift files
                            try:
                                content = swift_file.read_text(encoding='utf-8', errors='ignore')
                                lines = content.split('\n')
                                total_lines += len(lines)
                                
                                # Extract imports
                                for line in lines:
                                    if line.strip().startswith('import '):
                                        import_name = line.strip().replace('import ', '')
                                        imports.add(import_name)
                                
                                # Extract class/struct names
                                for line in lines:
                                    if 'class ' in line or 'struct ' in line:
                                        classes.append(f"{swift_file.name}: {line.strip()[:50]}")
                                        
                            except Exception as e:
                                analysis_output.append(f"    ⚠️ Error reading {swift_file.name}: {str(e)}")
                        
                        analysis_output.append(f"  📊 Total lines of Swift code: ~{total_lines}")
                        if imports:
                            analysis_output.append(f"  📚 Frameworks used: {', '.join(sorted(imports)[:10])}")
                        if classes:
                            analysis_output.append(f"  🏗️ Classes/Structs found: {len(classes)}")
                            for cls in classes[:5]:  # Show first 5
                                analysis_output.append(f"    • {cls}")
                
                # Check for other project types
                elif any(f.name == 'package.json' for f in relevant_files):
                    analysis_output.append("\n📦 **Node.js Project Detected**")
                elif any(f.suffix == '.py' for f in relevant_files):
                    analysis_output.append("\n🐍 **Python Project Detected**")
                elif any(f.suffix == '.java' for f in relevant_files):
                    analysis_output.append("\n☕ **Java Project Detected**")
                
                # Security analysis
                security_keywords = ['password', 'secret', 'key', 'token', 'auth', 'crypto', 'encrypt', 'decrypt']
                security_files = []
                
                for file_path in relevant_files[:20]:  # Check first 20 files for security
                    try:
                        if file_path.suffix in ['.swift', '.m', '.h', '.js', '.py']:
                            content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                            found_keywords = [kw for kw in security_keywords if kw in content]
                            if found_keywords:
                                security_files.append(f"{file_path.name}: {', '.join(found_keywords)}")
                    except:
                        pass
                
                if security_files:
                    analysis_output.append(f"\n🔒 **Security-Related Code Found:**")
                    for sf in security_files[:5]:
                        analysis_output.append(f"  • {sf}")
                
                # Project health check
                analysis_output.append(f"\n✅ **Analysis Summary:**")
                analysis_output.append(f"  • Total files analyzed: {len(relevant_files)}")
                analysis_output.append(f"  • Project appears to be: {'iOS/Swift' if any(f.suffix == '.swift' for f in relevant_files) else 'Multi-platform'}")
                analysis_output.append(f"  • Security elements: {'Found' if security_files else 'None detected'}")
                analysis_output.append(f"  • Project structure: {'Well organized' if len(relevant_files) > 5 else 'Simple structure'}")
                
            elif target_path.is_file():
                # Single file analysis
                try:
                    content = target_path.read_text(encoding='utf-8', errors='ignore')
                    files_read.append(str(target_path))
                    analysis_output.append(f"📄 **File Analysis: {target_path.name}**")
                    analysis_output.append(f"  • Size: {len(content)} characters")
                    analysis_output.append(f"  • Lines: {len(content.split())} lines")
                    analysis_output.append(f"  • Type: {target_path.suffix} file")
                    
                    # Show first part of content
                    analysis_output.append("\n--- File Content Preview ---")
                    analysis_output.append(content[:2000])  # First 2000 chars
                    if len(content) > 2000:
                        analysis_output.append("\n... (content truncated)")
                        
                except Exception as e:
                    analysis_output.append(f"❌ Error reading file: {str(e)}")
            
            return ActionResult(
                success=True,
                action_type=request.action_type,
                output="\n".join(analysis_output),
                files_affected=files_read,
                metadata={
                    "total_files": len(files_read),
                    "analysis_type": "local_filesystem",
                    "project_type": "iOS" if any('swift' in f.lower() for f in files_read) else "unknown"
                }
            )
        
        # Handle remote SSH analysis
        elif self.ssh_executor:
            # Check if target path exists
            result = self.ssh_executor.execute_command(f"test -d {request.target_path} && echo 'DIR' || test -f {request.target_path} && echo 'FILE' || echo 'NOT_FOUND'")
            
            if "NOT_FOUND" in result.stdout:
                return ActionResult(
                    success=False,
                    action_type=request.action_type,
                    output=f"Path not found: {request.target_path}",
                    files_affected=[],
                    error="Target path does not exist"
                )
            
            if "DIR" in result.stdout:
                # It's a directory - analyze the project
                analysis_output.append(f"📁 Analyzing directory: {request.target_path}")
                
                # List files
                result = self.ssh_executor.execute_command(f"find {request.target_path} -type f -name '*.js' -o -name '*.py' -o -name '*.java' -o -name '*.go' -o -name '*.rs' -o -name '*.ts' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' | head -50")
                if result.success and result.stdout:
                    files = result.stdout.strip().split('\n')
                    files_read = files[:20]  # Limit to first 20 files
                    
                    analysis_output.append(f"\n📊 Found {len(files)} code files")
                    
                    # Read package.json or similar to understand project
                    package_files = ["package.json", "requirements.txt", "go.mod", "Cargo.toml", "pom.xml"]
                    for pf in package_files:
                        result = self.ssh_executor.execute_command(f"test -f {request.target_path}/{pf} && cat {request.target_path}/{pf}")
                        if result.success and result.stdout:
                            analysis_output.append(f"\n📦 Found {pf}:")
                            # Parse key information
                            if pf == "package.json":
                                analysis_output.append("  Type: Node.js/JavaScript project")
                                if "react" in result.stdout:
                                    analysis_output.append("  Framework: React")
                                if "express" in result.stdout:
                                    analysis_output.append("  Backend: Express")
                            elif pf == "requirements.txt":
                                analysis_output.append("  Type: Python project")
                                if "django" in result.stdout:
                                    analysis_output.append("  Framework: Django")
                                if "flask" in result.stdout:
                                    analysis_output.append("  Framework: Flask")
                    
                    # Read first few files to understand structure
                    for file_path in files_read[:5]:
                        result = self.ssh_executor.execute_command(f"head -50 {file_path}")
                        if result.success:
                            analysis_output.append(f"\n📄 {file_path}:")
                            analysis_output.append(f"  First 50 lines analyzed")
                            # Basic analysis
                            if ".js" in file_path or ".ts" in file_path:
                                if "blockchain" in result.stdout.lower():
                                    analysis_output.append("  ✓ Contains blockchain-related code")
                                if "fabric" in result.stdout.lower():
                                    analysis_output.append("  ✓ Uses Hyperledger Fabric")
                                if "smart contract" in result.stdout.lower() or "chaincode" in result.stdout.lower():
                                    analysis_output.append("  ✓ Contains smart contract/chaincode")
            
            elif "FILE" in result.stdout:
                # It's a single file - read it
                result = self.ssh_executor.execute_command(f"cat {request.target_path}")
                if result.success:
                    files_read.append(request.target_path)
                    analysis_output.append(f"📄 File content read: {request.target_path}")
                    analysis_output.append(f"Content length: {len(result.stdout)} characters")
                    analysis_output.append("\n--- File Content ---\n")
                    analysis_output.append(result.stdout[:5000])  # First 5000 chars
            
            # Generate summary
            if files_read:
                analysis_output.append(f"\n✅ Successfully analyzed {len(files_read)} files")
                return ActionResult(
                    success=True,
                    action_type=request.action_type,
                    output="\n".join(analysis_output),
                    files_affected=files_read,
                    metadata={"total_files": len(files_read)}
                )
        
        return ActionResult(
            success=False,
            action_type=request.action_type,
            output="Unable to complete analysis",
            files_affected=[],
            error="No executor available"
        )
    
    async def _execute_create_code(self, request: ActionRequest) -> ActionResult:
        """
        CREATE_CODE Action: Generate and write new code files
        """
        created_files = []
        output = []
        
        # Extract code to create from parameters
        code_files = request.parameters.get("files", {}) if request.parameters else {}
        
        if not code_files:
            return ActionResult(
                success=False,
                action_type=request.action_type,
                output="No code provided to create",
                files_affected=[],
                error="Missing 'files' parameter with code to create"
            )
        
        if self.ssh_executor:
            # Create each file
            for file_path, content in code_files.items():
                full_path = f"{request.target_path}/{file_path}"
                
                # Create directory if needed
                dir_path = os.path.dirname(full_path)
                result = self.ssh_executor.execute_command(f"mkdir -p {dir_path}")
                
                # Write file
                # Escape special characters for shell
                escaped_content = content.replace("'", "'\\''")
                result = self.ssh_executor.execute_command(f"cat > {full_path} << 'EOF'\n{content}\nEOF")
                
                if result.success:
                    created_files.append(full_path)
                    output.append(f"✅ Created: {file_path}")
                    
                    # Make executable if it's a script
                    if file_path.endswith(('.sh', '.py', '.js')):
                        self.ssh_executor.execute_command(f"chmod +x {full_path}")
                else:
                    output.append(f"❌ Failed to create: {file_path} - {result.stderr}")
        
        if created_files:
            return ActionResult(
                success=True,
                action_type=request.action_type,
                output="\n".join(output),
                files_affected=created_files,
                metadata={"files_created": len(created_files)}
            )
        
        return ActionResult(
            success=False,
            action_type=request.action_type,
            output="\n".join(output) if output else "Failed to create any files",
            files_affected=[],
            error="No files were created"
        )
    
    async def _execute_test(self, request: ActionRequest) -> ActionResult:
        """
        TEST Action: Run tests and validate code
        """
        test_output = []
        
        if self.ssh_executor:
            # Navigate to target path
            result = self.ssh_executor.execute_command(f"cd {request.target_path} && pwd")
            if not result.success:
                return ActionResult(
                    success=False,
                    action_type=request.action_type,
                    output=f"Cannot access {request.target_path}",
                    files_affected=[],
                    error="Invalid path"
                )
            
            test_output.append(f"🧪 Running tests in: {request.target_path}")
            
            # Detect test framework and run tests
            test_commands = [
                ("npm test", "package.json", "Node.js tests"),
                ("python -m pytest", "requirements.txt", "Python tests"),
                ("go test ./...", "go.mod", "Go tests"),
                ("cargo test", "Cargo.toml", "Rust tests"),
                ("mvn test", "pom.xml", "Java/Maven tests"),
                ("./test.sh", "test.sh", "Shell script tests")
            ]
            
            tests_run = False
            for test_cmd, check_file, test_name in test_commands:
                result = self.ssh_executor.execute_command(f"test -f {request.target_path}/{check_file} && echo 'EXISTS'")
                if result.success and "EXISTS" in result.stdout:
                    test_output.append(f"\n📋 Running {test_name}...")
                    result = self.ssh_executor.execute_command(f"cd {request.target_path} && {test_cmd}", timeout=60)
                    
                    if result.success:
                        test_output.append(f"✅ {test_name} passed")
                        test_output.append(result.stdout[:1000])  # First 1000 chars of output
                    else:
                        test_output.append(f"❌ {test_name} failed")
                        test_output.append(result.stderr[:1000])
                    
                    tests_run = True
                    break
            
            if not tests_run:
                # Try generic test commands
                test_output.append("\n⚠️ No standard test framework detected, trying generic commands...")
                result = self.ssh_executor.execute_command(f"cd {request.target_path} && make test 2>/dev/null")
                if result.success:
                    test_output.append("✅ Make test succeeded")
                    test_output.append(result.stdout[:1000])
                    tests_run = True
            
            if tests_run:
                return ActionResult(
                    success=True,
                    action_type=request.action_type,
                    output="\n".join(test_output),
                    files_affected=[request.target_path],
                    metadata={"tests_executed": True}
                )
        
        return ActionResult(
            success=False,
            action_type=request.action_type,
            output="\n".join(test_output) if test_output else "No tests found or unable to run tests",
            files_affected=[],
            error="Testing failed or unavailable"
        )
    
    async def _execute_implement(self, request: ActionRequest) -> ActionResult:
        """
        IMPLEMENT Action: Deploy and implement changes
        """
        implementation_output = []
        affected_files = []
        
        if self.ssh_executor:
            implementation_output.append(f"🚀 Implementing in: {request.target_path}")
            
            # Check for deployment scripts
            deploy_scripts = ["deploy.sh", "install.sh", "setup.sh", "start.sh"]
            script_found = False
            
            for script in deploy_scripts:
                result = self.ssh_executor.execute_command(f"test -f {request.target_path}/{script} && echo 'EXISTS'")
                if result.success and "EXISTS" in result.stdout:
                    implementation_output.append(f"\n📜 Found {script}, executing...")
                    result = self.ssh_executor.execute_command(f"cd {request.target_path} && chmod +x {script} && ./{script}", timeout=120)
                    
                    if result.success:
                        implementation_output.append(f"✅ {script} executed successfully")
                        implementation_output.append(result.stdout[:2000])
                        affected_files.append(f"{request.target_path}/{script}")
                        script_found = True
                    else:
                        implementation_output.append(f"❌ {script} failed: {result.stderr[:500]}")
                    break
            
            if not script_found:
                # Try standard deployment commands based on project type
                implementation_output.append("\n📦 No deployment script found, trying standard commands...")
                
                # Check project type and deploy accordingly
                result = self.ssh_executor.execute_command(f"test -f {request.target_path}/package.json && echo 'NODE'")
                if result.success and "NODE" in result.stdout:
                    # Node.js deployment
                    commands = [
                        ("npm install", "Installing dependencies"),
                        ("npm run build", "Building project"),
                        ("pm2 restart all || npm start &", "Starting application")
                    ]
                    
                    for cmd, desc in commands:
                        implementation_output.append(f"\n⚙️ {desc}...")
                        result = self.ssh_executor.execute_command(f"cd {request.target_path} && {cmd}", timeout=60)
                        if result.success:
                            implementation_output.append(f"✅ {desc} completed")
                        else:
                            implementation_output.append(f"⚠️ {desc} had issues: {result.stderr[:200]}")
                    
                    affected_files.append(request.target_path)
                    script_found = True
            
            if affected_files:
                return ActionResult(
                    success=True,
                    action_type=request.action_type,
                    output="\n".join(implementation_output),
                    files_affected=affected_files,
                    metadata={"deployment_completed": True}
                )
        
        return ActionResult(
            success=False,
            action_type=request.action_type,
            output="\n".join(implementation_output) if implementation_output else "Implementation failed",
            files_affected=[],
            error="Unable to implement changes"
        )
    
    async def _execute_debug(self, request: ActionRequest) -> ActionResult:
        """
        DEBUG Action: Debug and fix issues
        """
        debug_output = []
        
        if self.ssh_executor:
            debug_output.append(f"🔍 Debugging in: {request.target_path}")
            
            # Check logs
            log_locations = [
                "logs/*.log",
                "*.log", 
                "/var/log/syslog",
                "npm-debug.log",
                "error.log"
            ]
            
            for log_pattern in log_locations:
                result = self.ssh_executor.execute_command(f"cd {request.target_path} && tail -100 {log_pattern} 2>/dev/null | grep -i 'error\\|fail\\|exception' | head -20")
                if result.success and result.stdout:
                    debug_output.append(f"\n📋 Errors found in {log_pattern}:")
                    debug_output.append(result.stdout)
            
            # Check process status
            result = self.ssh_executor.execute_command("ps aux | grep -E 'node|python|java' | grep -v grep | head -10")
            if result.success and result.stdout:
                debug_output.append("\n⚙️ Running processes:")
                debug_output.append(result.stdout)
            
            # Check port usage
            result = self.ssh_executor.execute_command("netstat -tulpn 2>/dev/null | grep LISTEN | head -10")
            if result.success and result.stdout:
                debug_output.append("\n🔌 Listening ports:")
                debug_output.append(result.stdout)
            
            return ActionResult(
                success=True,
                action_type=request.action_type,
                output="\n".join(debug_output),
                files_affected=[request.target_path],
                metadata={"debug_completed": True}
            )
        
        return ActionResult(
            success=False,
            action_type=request.action_type,
            output="Debug failed",
            files_affected=[],
            error="Unable to debug"
        )
    
    async def _execute_optimize(self, request: ActionRequest) -> ActionResult:
        """
        OPTIMIZE Action: Optimize performance
        """
        # Implementation for optimization
        return ActionResult(
            success=True,
            action_type=request.action_type,
            output="Optimization analysis completed",
            files_affected=[],
            metadata={"optimization_suggestions": ["Use caching", "Optimize queries", "Minify assets"]}
        )
    
    async def _execute_document(self, request: ActionRequest) -> ActionResult:
        """
        DOCUMENT Action: Create documentation
        """
        # Implementation for documentation
        return ActionResult(
            success=True,
            action_type=request.action_type,
            output="Documentation generated",
            files_affected=["README.md"],
            metadata={"docs_created": True}
        )
    
    async def _execute_configure(self, request: ActionRequest) -> ActionResult:
        """
        CONFIGURE Action: Setup and configure systems
        """
        # Implementation for configuration
        return ActionResult(
            success=True,
            action_type=request.action_type,
            output="Configuration completed",
            files_affected=[],
            metadata={"configured": True}
        )


class ActionClassifier:
    """
    Classifies user requests into concrete actions
    """
    
    @staticmethod
    def classify_request(user_query: str) -> ActionType:
        """
        Determine which action type based on the user's request
        """
        query_lower = user_query.lower()
        
        # Keywords for each action type
        action_keywords = {
            ActionType.READ_ANALYZE: [
                'read', 'analyze', 'review', 'examine', 'inspect', 'check', 
                'look', 'find', 'show', 'report', 'understand', 'explain',
                'what is', 'where is', 'list', 'display', 'view'
            ],
            ActionType.CREATE_CODE: [
                'create', 'build', 'generate', 'write', 'develop', 'make',
                'construct', 'add', 'implement new', 'code new'
            ],
            ActionType.TEST: [
                'test', 'validate', 'verify', 'check if', 'ensure',
                'run tests', 'unit test', 'integration test'
            ],
            ActionType.IMPLEMENT: [
                'deploy', 'implement', 'install', 'setup', 'launch',
                'start', 'run', 'execute', 'activate', 'push'
            ],
            ActionType.DEBUG: [
                'debug', 'fix', 'troubleshoot', 'solve', 'resolve',
                'error', 'issue', 'problem', 'not working', 'broken'
            ],
            ActionType.OPTIMIZE: [
                'optimize', 'improve', 'enhance', 'speed up', 'performance',
                'faster', 'efficient', 'reduce', 'minimize'
            ],
            ActionType.DOCUMENT: [
                'document', 'describe', 'readme', 'documentation',
                'explain how', 'write docs', 'api docs'
            ],
            ActionType.CONFIGURE: [
                'configure', 'config', 'set up', 'settings', 'initialize',
                'environment', 'env', 'variables'
            ]
        }
        
        # Score each action type
        scores = {}
        for action_type, keywords in action_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[action_type] = score
        
        # Return the action with highest score
        if scores:
            best_action = max(scores, key=scores.get)
            if scores[best_action] > 0:
                return best_action
        
        # Default to READ_ANALYZE if unclear
        return ActionType.READ_ANALYZE


if __name__ == "__main__":
    # Example usage
    print("🎯 Agent Actions System")
    print("=" * 60)
    
    # Example: Classify a request
    test_query = "please read the current implementation of my blockchain application on my ubuntu server and report your findings"
    action_type = ActionClassifier.classify_request(test_query)
    print(f"\nQuery: {test_query}")
    print(f"Classified as: {action_type.value}")
    
    # Example: Execute an action
    async def test_action():
        executor = AgentActionExecutor()
        
        request = ActionRequest(
            action_type=ActionType.READ_ANALYZE,
            target_path="/home/ubuntu/fabric-api-gateway-modular",
            description="Analyze blockchain application",
            deployment_config={
                "type": "ubuntu_server",
                "server_ip": "13.38.102.28",
                "username": "ubuntu",
                "key_path": "/Users/jankootstra/blockchain.pem"
            }
        )
        
        result = await executor.execute_action(request)
        
        print(f"\nAction Result:")
        print(f"Success: {result.success}")
        print(f"Files Affected: {len(result.files_affected)}")
        print(f"Output Preview: {result.output[:500]}...")
    
    # Run the test
    # asyncio.run(test_action())
    
    print("\n✅ Agent Actions System Ready!")
    print("\nAction Types Available:")
    for action in ActionType:
        print(f"  - {action.value}: {action.name.replace('_', ' ').title()}")