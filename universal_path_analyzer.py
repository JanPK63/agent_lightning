#!/usr/bin/env python3
"""
Universal Path Analyzer
Analyzes ANY path - local directories, remote servers, or any location
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

class UniversalPathAnalyzer:
    """Analyzes any path universally"""
    
    def __init__(self):
        self.project_config = None
        self._load_project_config()
    
    def _load_project_config(self):
        """Load project configuration for SSH access"""
        try:
            from project_config import ProjectConfigManager
            config_manager = ProjectConfigManager()
            self.project_config = config_manager.get_active_project()
        except:
            pass
    
    def analyze_path(self, path_or_task: str) -> Dict[str, Any]:
        """Analyze any path from task description"""
        
        # Extract path from task
        target_path = self._extract_path(path_or_task)
        
        # Try local analysis first
        local_result = self._analyze_local(target_path)
        if local_result['found']:
            return local_result
        
        # Try remote analysis
        remote_result = self._analyze_remote(target_path)
        if remote_result['found']:
            return remote_result
        
        return {
            'found': False,
            'path': target_path,
            'error': f"Path '{target_path}' not found locally or remotely"
        }
    
    def _extract_path(self, text: str) -> str:
        """Extract path from any text"""
        # First try explicit paths
        path_patterns = [
            r'/[^\s]+',  # /Users/... or /home/...
            r'~[^\s]*',  # ~/project
            r'[a-zA-Z]:[^\s]*',  # C:\Windows
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        # Then try directory names with context
        directory_patterns = [
            r'directory[:\s]+([a-zA-Z0-9_-]{3,}(?:-[a-zA-Z0-9_-]+)*)',
            r'analyze[\s]+([a-zA-Z0-9_-]{3,}(?:-[a-zA-Z0-9_-]+)*)',
            r'check[\s]+(?:the[\s]+)?([a-zA-Z0-9_-]{3,}(?:-[a-zA-Z0-9_-]+)*)',
            r'([a-zA-Z0-9_-]{3,}(?:-[a-zA-Z0-9_-]+)*)[\s]+(?:directory|project)',
            r'\b([a-zA-Z0-9_-]{5,}(?:-[a-zA-Z0-9_-]+)+)\b'  # multi-part names
        ]
        
        for pattern in directory_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'fabric-api-gateway-modular'  # Default
    
    def _analyze_local(self, path: str) -> Dict[str, Any]:
        """Analyze local path"""
        try:
            if not (path.startswith('/') or path.startswith('~') or ':' in path):
                return {'found': False}
            
            local_path = Path(path).expanduser()
            if not local_path.exists():
                return {'found': False}
            
            result = {
                'found': True,
                'type': 'local',
                'path': str(local_path),
                'analysis': []
            }
            
            if local_path.is_dir():
                # Count files by type
                js_files = list(local_path.rglob('*.js'))
                py_files = list(local_path.rglob('*.py'))
                json_files = list(local_path.rglob('*.json'))
                
                result['analysis'].append(f"📁 LOCAL: {local_path.name}")
                result['analysis'].append(f"Files: {len(js_files)} JS, {len(py_files)} Python, {len(json_files)} JSON")
                
                # Detect project type
                if (local_path / 'package.json').exists():
                    result['analysis'].append("🟢 Node.js Project")
                elif py_files:
                    result['analysis'].append("🟢 Python Project")
                
                # Check for key files
                key_files = ['package.json', 'requirements.txt', 'Dockerfile', 'README.md']
                found_files = [f for f in key_files if (local_path / f).exists()]
                if found_files:
                    result['analysis'].append(f"Key files: {', '.join(found_files)}")
            
            return result
            
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def _analyze_remote(self, path: str) -> Dict[str, Any]:
        """Analyze remote path via SSH"""
        try:
            if not self.project_config:
                return {'found': False, 'error': 'No project config'}
            
            # Find Ubuntu server target
            ubuntu_target = None
            for target in self.project_config.deployment_targets:
                if target.type == 'ubuntu_server':
                    ubuntu_target = target
                    break
            
            if not ubuntu_target:
                return {'found': False, 'error': 'No Ubuntu server configured'}
            
            ssh_cmd = f"ssh -i {ubuntu_target.ssh_key_path} -o StrictHostKeyChecking=no {ubuntu_target.username}@{ubuntu_target.server_ip}"
            
            # Try multiple path variations
            path_variations = [
                path,
                f"~/{path}",
                f"/home/{ubuntu_target.username}/{path}",
                f"/home/ubuntu/{path}"
            ]
            
            for test_path in path_variations:
                # Check if path exists
                result = subprocess.run(f"{ssh_cmd} 'test -d {test_path} && echo EXISTS'", 
                                      shell=True, capture_output=True, text=True, timeout=10)
                
                if 'EXISTS' in result.stdout:
                    # Found it! Analyze
                    analysis_result = {
                        'found': True,
                        'type': 'remote',
                        'path': test_path,
                        'server': ubuntu_target.server_ip,
                        'analysis': []
                    }
                    
                    # Execute analysis commands
                    commands = [
                        f"echo '=== {test_path} ===' && ls -la {test_path} | head -5",
                        f"find {test_path} -name '*.js' -o -name '*.py' -o -name '*.json' | wc -l",
                        f"test -f {test_path}/package.json && echo 'Node.js Project' || echo 'Other Project'",
                        f"du -sh {test_path} 2>/dev/null || echo 'Size unknown'"
                    ]
                    
                    for cmd in commands:
                        try:
                            res = subprocess.run(f"{ssh_cmd} '{cmd}'", 
                                              shell=True, capture_output=True, text=True, timeout=10)
                            if res.stdout.strip():
                                analysis_result['analysis'].append(res.stdout.strip())
                        except:
                            pass
                    
                    return analysis_result
            
            return {'found': False, 'error': f"Path '{path}' not found on server"}
            
        except Exception as e:
            return {'found': False, 'error': f"Remote analysis failed: {str(e)}"}


# Integration with enhanced API
def integrate_universal_analyzer():
    """Replace the existing analysis code with universal analyzer"""
    
    analyzer_code = '''
            # UNIVERSAL PATH ANALYSIS - ANY PATH, ANY LOCATION
            if needs_file_access or any(keyword in request.task.lower() for keyword in [
                'analyze', 'examine', 'check', 'review', 'status', 'report', 'investigate', 'assess'
            ]):
                try:
                    from universal_path_analyzer import UniversalPathAnalyzer
                    
                    analyzer = UniversalPathAnalyzer()
                    analysis_result = analyzer.analyze_path(request.task)
                    
                    if analysis_result['found']:
                        result_text += f"\\n\\n✅ **{analysis_result['type'].upper()} ANALYSIS EXECUTED**\\n"
                        if analysis_result['type'] == 'remote':
                            result_text += f"Server: {analysis_result['server']}\\n"
                        result_text += f"Path: {analysis_result['path']}\\n\\n"
                        result_text += "\\n".join(analysis_result['analysis'])
                    else:
                        result_text += f"\\n\\n❌ **Analysis failed:** {analysis_result.get('error', 'Unknown error')}"
                        
                except Exception as e:
                    result_text += f"\\n\\n❌ **Analysis error:** {str(e)}"
    '''
    
    return analyzer_code


if __name__ == "__main__":
    # Test the analyzer
    analyzer = UniversalPathAnalyzer()
    
    test_paths = [
        "/Users/jankootstra/agent-lightning-main",
        "fabric-api-gateway-modular",
        "~/Documents",
        "analyze this directory: fabric-api-gateway-modular"
    ]
    
    for test_path in test_paths:
        print(f"\n=== Testing: {test_path} ===")
        result = analyzer.analyze_path(test_path)
        print(f"Found: {result['found']}")
        if result['found']:
            print(f"Type: {result['type']}")
            print(f"Path: {result['path']}")
            for line in result.get('analysis', []):
                print(f"  {line}")
        else:
            print(f"Error: {result.get('error', 'Not found')}")