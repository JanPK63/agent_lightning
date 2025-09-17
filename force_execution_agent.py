"""
Force Execution Agent - Makes agents DO work instead of just talking about it
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import time

class ForceExecutionAgent:
    """Forces agents to actually execute tasks instead of giving approaches"""
    
    def __init__(self):
        self.execution_results = {}
    
    def force_execute_analysis(self, task: str, target_path: str) -> Dict[str, Any]:
        """Force actual execution of analysis tasks"""
        
        # Extract the actual path from task
        if not target_path and '/' in task:
            import re
            path_match = re.search(r'/[^\s]+', task)
            if path_match:
                target_path = path_match.group(0)
        
        if not target_path:
            return {"error": "No valid path found in task"}
        
        # Expand path
        target_path = os.path.expanduser(target_path)
        
        if not os.path.exists(target_path):
            return {"error": f"Path does not exist: {target_path}"}
        
        results = {
            "path_analyzed": target_path,
            "analysis_type": "comprehensive",
            "findings": {}
        }
        
        try:
            # 1. DIRECTORY STRUCTURE ANALYSIS
            if os.path.isdir(target_path):
                results["findings"]["directory_structure"] = self._analyze_directory_structure(target_path)
                results["findings"]["file_count"] = self._count_files(target_path)
                results["findings"]["project_type"] = self._detect_project_type(target_path)
                
                # 2. CODE ANALYSIS
                if any(keyword in task.lower() for keyword in ['ios', 'swift', 'xcode']):
                    results["findings"]["ios_analysis"] = self._analyze_ios_project(target_path)
                
                # 3. CONFIGURATION FILES
                results["findings"]["config_files"] = self._find_config_files(target_path)
                
                # 4. DEPENDENCIES
                results["findings"]["dependencies"] = self._analyze_dependencies(target_path)
                
                # 5. BUILD STATUS
                results["findings"]["build_status"] = self._check_build_status(target_path)
                
                # 6. SERVER CONNECTION TEST (if mentioned)
                if any(keyword in task.lower() for keyword in ['server', 'ubuntu', 'connection']):
                    results["findings"]["server_connection"] = self._test_server_connections(target_path)
            
            else:
                # Single file analysis
                results["findings"]["file_analysis"] = self._analyze_single_file(target_path)
            
            # 7. GENERATE SUMMARY
            results["summary"] = self._generate_analysis_summary(results["findings"])
            results["status"] = "completed"
            
        except Exception as e:
            results["error"] = str(e)
            results["status"] = "failed"
        
        return results
    
    def _analyze_directory_structure(self, path: str) -> Dict[str, Any]:
        """Analyze directory structure"""
        structure = {}
        
        try:
            path_obj = Path(path)
            
            # Get immediate subdirectories
            subdirs = [d.name for d in path_obj.iterdir() if d.is_dir() and not d.name.startswith('.')]
            structure["subdirectories"] = subdirs[:20]  # Limit to 20
            
            # Get file types
            file_extensions = {}
            for file_path in path_obj.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext:
                        file_extensions[ext] = file_extensions.get(ext, 0) + 1
            
            structure["file_types"] = dict(sorted(file_extensions.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Check for common project indicators
            indicators = {}
            common_files = [
                'package.json', 'requirements.txt', 'Podfile', 'Gemfile', 
                'pom.xml', 'build.gradle', 'Cargo.toml', 'go.mod',
                'Dockerfile', 'docker-compose.yml', '.gitignore', 'README.md'
            ]
            
            for file_name in common_files:
                file_path = path_obj / file_name
                if file_path.exists():
                    indicators[file_name] = "present"
            
            structure["project_indicators"] = indicators
            
        except Exception as e:
            structure["error"] = str(e)
        
        return structure
    
    def _count_files(self, path: str) -> Dict[str, int]:
        """Count files by type"""
        counts = {"total_files": 0, "total_directories": 0}
        
        try:
            path_obj = Path(path)
            for item in path_obj.rglob("*"):
                if item.is_file():
                    counts["total_files"] += 1
                elif item.is_dir():
                    counts["total_directories"] += 1
        except Exception as e:
            counts["error"] = str(e)
        
        return counts
    
    def _detect_project_type(self, path: str) -> Dict[str, Any]:
        """Detect project type based on files"""
        path_obj = Path(path)
        project_types = []
        
        # iOS Project
        if any(path_obj.rglob("*.xcodeproj")) or any(path_obj.rglob("*.xcworkspace")):
            project_types.append("iOS/Xcode")
        
        # Android Project
        if (path_obj / "build.gradle").exists() or (path_obj / "app" / "build.gradle").exists():
            project_types.append("Android")
        
        # Web Projects
        if (path_obj / "package.json").exists():
            project_types.append("Node.js/JavaScript")
        
        # Python Projects
        if (path_obj / "requirements.txt").exists() or (path_obj / "setup.py").exists():
            project_types.append("Python")
        
        # Docker
        if (path_obj / "Dockerfile").exists():
            project_types.append("Docker")
        
        return {"detected_types": project_types}
    
    def _analyze_ios_project(self, path: str) -> Dict[str, Any]:
        """Specific iOS project analysis"""
        ios_analysis = {}\n        path_obj = Path(path)\n        \n        # Find Xcode project files\n        xcodeproj_files = list(path_obj.rglob(\"*.xcodeproj\"))\n        xcworkspace_files = list(path_obj.rglob(\"*.xcworkspace\"))\n        \n        ios_analysis[\"xcode_projects\"] = [str(f.name) for f in xcodeproj_files]\n        ios_analysis[\"xcode_workspaces\"] = [str(f.name) for f in xcworkspace_files]\n        \n        # Check for Swift files\n        swift_files = list(path_obj.rglob(\"*.swift\"))\n        ios_analysis[\"swift_files_count\"] = len(swift_files)\n        \n        # Check for Podfile\n        podfile = path_obj / \"Podfile\"\n        if podfile.exists():\n            ios_analysis[\"cocoapods\"] = \"present\"\n            try:\n                podfile_content = podfile.read_text()\n                ios_analysis[\"podfile_preview\"] = podfile_content[:500]\n            except:\n                pass\n        \n        # Check for key iOS directories\n        ios_dirs = []\n        for dir_name in [\"Security_Agent_New\", \"Backend\", \"Resources\", \"Core\", \"UI\", \"Models\"]:\n            if (path_obj / dir_name).exists():\n                ios_dirs.append(dir_name)\n        \n        ios_analysis[\"key_directories\"] = ios_dirs\n        \n        return ios_analysis\n    \n    def _find_config_files(self, path: str) -> Dict[str, Any]:\n        \"\"\"Find configuration files\"\"\"\n        config_files = {}\n        path_obj = Path(path)\n        \n        config_patterns = [\n            \"*.json\", \"*.yml\", \"*.yaml\", \"*.toml\", \"*.ini\", \n            \"*.conf\", \"*.config\", \"*.plist\", \"*.env\"\n        ]\n        \n        for pattern in config_patterns:\n            files = list(path_obj.rglob(pattern))\n            if files:\n                config_files[pattern] = [str(f.relative_to(path_obj)) for f in files[:5]]\n        \n        return config_files\n    \n    def _analyze_dependencies(self, path: str) -> Dict[str, Any]:\n        \"\"\"Analyze project dependencies\"\"\"\n        dependencies = {}\n        path_obj = Path(path)\n        \n        # Package.json\n        package_json = path_obj / \"package.json\"\n        if package_json.exists():\n            try:\n                with open(package_json) as f:\n                    data = json.load(f)\n                    dependencies[\"npm\"] = {\n                        \"dependencies\": len(data.get(\"dependencies\", {})),\n                        \"devDependencies\": len(data.get(\"devDependencies\", {}))\n                    }\n            except:\n                dependencies[\"npm\"] = \"error reading\"\n        \n        # Requirements.txt\n        requirements = path_obj / \"requirements.txt\"\n        if requirements.exists():\n            try:\n                lines = requirements.read_text().strip().split('\\n')\n                dependencies[\"python\"] = {\"packages\": len([l for l in lines if l.strip() and not l.startswith('#')])}\n            except:\n                dependencies[\"python\"] = \"error reading\"\n        \n        # Podfile.lock\n        podfile_lock = path_obj / \"Podfile.lock\"\n        if podfile_lock.exists():\n            try:\n                content = podfile_lock.read_text()\n                pod_count = content.count('- ')\n                dependencies[\"cocoapods\"] = {\"pods\": pod_count}\n            except:\n                dependencies[\"cocoapods\"] = \"error reading\"\n        \n        return dependencies\n    \n    def _check_build_status(self, path: str) -> Dict[str, Any]:\n        \"\"\"Check build status and build files\"\"\"\n        build_status = {}\n        path_obj = Path(path)\n        \n        # Check for build directories\n        build_dirs = [\"build\", \"dist\", \"target\", \".build\", \"DerivedData\"]\n        for build_dir in build_dirs:\n            if (path_obj / build_dir).exists():\n                build_status[f\"{build_dir}_exists\"] = True\n        \n        # Check for build files\n        build_files = [\"Makefile\", \"build.sh\", \"compile.sh\", \"CMakeLists.txt\"]\n        for build_file in build_files:\n            if (path_obj / build_file).exists():\n                build_status[f\"{build_file}_exists\"] = True\n        \n        return build_status\n    \n    def _test_server_connections(self, path: str) -> Dict[str, Any]:\n        \"\"\"Test server connections if SSH keys are found\"\"\"\n        connection_results = {}\n        path_obj = Path(path)\n        \n        # Look for SSH keys\n        ssh_keys = list(path_obj.rglob(\"*.pem\")) + list(path_obj.rglob(\"*key*\"))\n        if ssh_keys:\n            connection_results[\"ssh_keys_found\"] = [str(k.name) for k in ssh_keys[:3]]\n            \n            # Look for server IPs in files\n            server_ips = self._extract_server_ips(path)\n            if server_ips:\n                connection_results[\"server_ips_found\"] = server_ips\n                \n                # Try to test connection (safely)\n                for ip in server_ips[:1]:  # Only test first IP\n                    try:\n                        # Simple ping test\n                        result = subprocess.run(\n                            [\"ping\", \"-c\", \"1\", \"-W\", \"2000\", ip], \n                            capture_output=True, \n                            text=True, \n                            timeout=5\n                        )\n                        connection_results[f\"ping_{ip}\"] = \"success\" if result.returncode == 0 else \"failed\"\n                    except:\n                        connection_results[f\"ping_{ip}\"] = \"timeout\"\n        \n        return connection_results\n    \n    def _extract_server_ips(self, path: str) -> List[str]:\n        \"\"\"Extract server IPs from configuration files\"\"\"\n        import re\n        ips = []\n        path_obj = Path(path)\n        \n        # Look in common config files\n        config_files = list(path_obj.rglob(\"*.py\")) + list(path_obj.rglob(\"*.json\")) + list(path_obj.rglob(\"*.yml\"))\n        \n        ip_pattern = r'\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b'\n        \n        for config_file in config_files[:10]:  # Limit to 10 files\n            try:\n                content = config_file.read_text()\n                found_ips = re.findall(ip_pattern, content)\n                for ip in found_ips:\n                    if ip not in ['127.0.0.1', '0.0.0.0'] and ip not in ips:\n                        ips.append(ip)\n            except:\n                continue\n        \n        return ips[:5]  # Return max 5 IPs\n    \n    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:\n        \"\"\"Analyze a single file\"\"\"\n        analysis = {}\n        \n        try:\n            path_obj = Path(file_path)\n            analysis[\"file_name\"] = path_obj.name\n            analysis[\"file_size\"] = path_obj.stat().st_size\n            analysis[\"file_extension\"] = path_obj.suffix\n            \n            # Read content preview\n            if path_obj.suffix.lower() in ['.txt', '.py', '.js', '.json', '.yml', '.yaml', '.md']:\n                content = path_obj.read_text()\n                analysis[\"content_preview\"] = content[:1000]\n                analysis[\"line_count\"] = len(content.split('\\n'))\n            \n        except Exception as e:\n            analysis[\"error\"] = str(e)\n        \n        return analysis\n    \n    def _generate_analysis_summary(self, findings: Dict[str, Any]) -> str:\n        \"\"\"Generate a comprehensive analysis summary\"\"\"\n        summary_parts = []\n        \n        # Directory structure summary\n        if \"directory_structure\" in findings:\n            structure = findings[\"directory_structure\"]\n            if \"subdirectories\" in structure:\n                summary_parts.append(f\"📁 Found {len(structure['subdirectories'])} main directories\")\n            if \"file_types\" in structure:\n                top_types = list(structure[\"file_types\"].items())[:3]\n                summary_parts.append(f\"📄 Main file types: {', '.join([f'{ext}({count})' for ext, count in top_types])}\")\n        \n        # Project type summary\n        if \"project_type\" in findings:\n            types = findings[\"project_type\"].get(\"detected_types\", [])\n            if types:\n                summary_parts.append(f\"🔧 Project types: {', '.join(types)}\")\n        \n        # iOS specific summary\n        if \"ios_analysis\" in findings:\n            ios = findings[\"ios_analysis\"]\n            if ios.get(\"xcode_projects\"):\n                summary_parts.append(f\"📱 iOS Project: {len(ios['xcode_projects'])} Xcode projects, {ios.get('swift_files_count', 0)} Swift files\")\n        \n        # Dependencies summary\n        if \"dependencies\" in findings:\n            deps = findings[\"dependencies\"]\n            dep_summary = []\n            for dep_type, info in deps.items():\n                if isinstance(info, dict):\n                    if \"dependencies\" in info:\n                        dep_summary.append(f\"{dep_type}: {info['dependencies']} packages\")\n                    elif \"packages\" in info:\n                        dep_summary.append(f\"{dep_type}: {info['packages']} packages\")\n                    elif \"pods\" in info:\n                        dep_summary.append(f\"{dep_type}: {info['pods']} pods\")\n            if dep_summary:\n                summary_parts.append(f\"📦 Dependencies: {', '.join(dep_summary)}\")\n        \n        # Server connection summary\n        if \"server_connection\" in findings:\n            conn = findings[\"server_connection\"]\n            if conn.get(\"ssh_keys_found\"):\n                summary_parts.append(f\"🔑 SSH keys found: {len(conn['ssh_keys_found'])}\")\n            if conn.get(\"server_ips_found\"):\n                summary_parts.append(f\"🌐 Server IPs detected: {len(conn['server_ips_found'])}\")\n        \n        # File count summary\n        if \"file_count\" in findings:\n            count = findings[\"file_count\"]\n            summary_parts.append(f\"📊 Total: {count.get('total_files', 0)} files, {count.get('total_directories', 0)} directories\")\n        \n        return \"\\n\".join(summary_parts) if summary_parts else \"Analysis completed - no specific findings to report\"\n\n\n# Integration function for the main API\ndef execute_forced_analysis(task: str, target_path: str = None) -> Dict[str, Any]:\n    \"\"\"Execute forced analysis - called by main API\"\"\"\n    executor = ForceExecutionAgent()\n    return executor.force_execute_analysis(task, target_path)\n