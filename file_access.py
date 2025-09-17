"""
Simple file system access for agents
"""
import os
from pathlib import Path
from typing import List, Dict, Any

def read_directory(path: str) -> Dict[str, Any]:
    """Read directory contents with file analysis"""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return {"error": f"Directory {path} does not exist"}
        
        files = []
        for item in path_obj.rglob("*"):
            if item.is_file() and item.suffix in ['.swift', '.m', '.h', '.py', '.js', '.json', '.plist', '.md']:
                try:
                    content = item.read_text(encoding='utf-8', errors='ignore')[:1000]
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
        
        return {
            "success": True,
            "path": path,
            "files_found": len(files),
            "files": files[:20]  # Limit to first 20 files
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_ios_project(path: str) -> str:
    """Analyze iOS project structure"""
    result = read_directory(path)
    
    if "error" in result:
        return f"Cannot access directory: {result['error']}"
    
    swift_files = [f for f in result["files"] if f["type"] == ".swift"]
    objc_files = [f for f in result["files"] if f["type"] in [".m", ".h"]]
    config_files = [f for f in result["files"] if f["type"] in [".plist", ".json"]]
    
    analysis = f"""
iOS Project Analysis for {path}:

📱 Project Structure:
- Swift files: {len(swift_files)}
- Objective-C files: {len(objc_files)}
- Configuration files: {len(config_files)}
- Total files analyzed: {result['files_found']}

🔍 Key Files Found:
"""
    
    for file in result["files"][:10]:
        analysis += f"\n- {file['path']} ({file['size']} bytes)"
    
    return analysis