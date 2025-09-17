#!/usr/bin/env python3
"""
Quick script to grant directory access to agents
Usage: python grant_access.py /path/to/directory agent_name
"""
import sys
from directory_access import access_manager

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python grant_access.py /path/to/directory agent_name")
        sys.exit(1)
    
    directory = sys.argv[1]
    agent_name = sys.argv[2]
    
    result = access_manager.grant_access(directory, agent_name)
    print(f"✅ {result['message']}")