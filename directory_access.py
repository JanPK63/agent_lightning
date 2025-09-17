"""
Directory Access Authorization System
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class DirectoryAccessManager:
    def __init__(self):
        self.auth_file = Path.home() / ".agent_lightning" / "directory_auth.json"
        self.auth_file.parent.mkdir(exist_ok=True)
        self.authorized_dirs = self._load_authorizations()
    
    def _load_authorizations(self) -> Dict[str, Dict]:
        """Load authorized directories from file"""
        if self.auth_file.exists():
            try:
                return json.loads(self.auth_file.read_text())
            except:
                return {}
        return {}
    
    def _save_authorizations(self):
        """Save authorizations to file"""
        self.auth_file.write_text(json.dumps(self.authorized_dirs, indent=2))
    
    def request_access(self, directory: str, agent_id: str, reason: str) -> Dict:
        """Request access to a directory"""
        abs_path = os.path.abspath(directory)
        
        # Check if already authorized
        if abs_path in self.authorized_dirs:
            auth = self.authorized_dirs[abs_path]
            if agent_id in auth.get("agents", []):
                return {"status": "granted", "message": "Access already authorized"}
        
        # Return authorization request
        return {
            "status": "authorization_required",
            "directory": abs_path,
            "agent_id": agent_id,
            "reason": reason,
            "message": f"Agent {agent_id} requests access to {abs_path}. Reason: {reason}",
            "grant_command": f"grant_access('{abs_path}', '{agent_id}')"
        }
    
    def grant_access(self, directory: str, agent_id: str) -> Dict:
        """Grant access to a directory"""
        abs_path = os.path.abspath(directory)
        
        if abs_path not in self.authorized_dirs:
            self.authorized_dirs[abs_path] = {
                "agents": [],
                "granted_at": datetime.now().isoformat()
            }
        
        if agent_id not in self.authorized_dirs[abs_path]["agents"]:
            self.authorized_dirs[abs_path]["agents"].append(agent_id)
        
        self._save_authorizations()
        return {"status": "granted", "message": f"Access granted to {agent_id} for {abs_path}"}
    
    def check_access(self, directory: str, agent_id: str) -> bool:
        """Check if agent has access to directory"""
        abs_path = os.path.abspath(directory)
        return (abs_path in self.authorized_dirs and 
                agent_id in self.authorized_dirs[abs_path].get("agents", []))
    
    def list_authorizations(self) -> Dict:
        """List all authorizations"""
        return self.authorized_dirs

# Global instance
access_manager = DirectoryAccessManager()

def analyze_directory_with_auth(directory: str, agent_id: str, reason: str = "Project analysis") -> Dict:
    """Analyze directory with authorization check"""
    
    # Check authorization
    if not access_manager.check_access(directory, agent_id):
        return access_manager.request_access(directory, agent_id, reason)
    
    # Perform analysis
    try:
        path_obj = Path(directory)
        if not path_obj.exists():
            return {"error": f"Directory {directory} does not exist"}
        
        files = []
        for item in path_obj.rglob("*"):
            if item.is_file() and item.suffix in ['.swift', '.m', '.h', '.py', '.js', '.json', '.plist', '.md', '.txt', '.yml', '.yaml']:
                try:
                    content = item.read_text(encoding='utf-8', errors='ignore')[:500]
                    files.append({
                        "path": str(item.relative_to(path_obj)),
                        "size": item.stat().st_size,
                        "type": item.suffix,
                        "content_preview": content
                    })
                except:
                    files.append({
                        "path": str(item.relative_to(path_obj)),
                        "size": item.stat().st_size,
                        "type": item.suffix,
                        "content_preview": "Could not read file"
                    })
        
        # Generate analysis
        analysis = f"Directory Analysis: {directory}\n\n"
        analysis += f"Files found: {len(files)}\n\n"
        
        # Group by file type
        by_type = {}
        for f in files:
            ftype = f["type"] or "no extension"
            by_type[ftype] = by_type.get(ftype, 0) + 1
        
        analysis += "File types:\n"
        for ftype, count in sorted(by_type.items()):
            analysis += f"  {ftype}: {count} files\n"
        
        analysis += f"\nFirst 10 files:\n"
        for f in files[:10]:
            analysis += f"  - {f['path']} ({f['size']} bytes)\n"
        
        return {
            "status": "success",
            "analysis": analysis,
            "files": files,
            "directory": directory
        }
        
    except Exception as e:
        return {"error": str(e)}