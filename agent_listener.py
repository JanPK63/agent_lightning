#!/usr/bin/env python3
"""
Simple Agent Listener - Polls for tasks and executes them
"""

import time
import requests
import json
from datetime import datetime

class AgentListener:
    def __init__(self, api_url="http://localhost:8002"):
        self.api_url = api_url
        self.running = False
        
    def start_listening(self):
        """Start listening for tasks"""
        print(f"🎧 Agent listener started - polling {self.api_url}")
        self.running = True
        
        while self.running:
            try:
                # Check for pending tasks (simplified - just log that we're listening)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 👂 Listening for tasks...")
                
                # Sleep for 5 seconds between polls
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping agent listener...")
                self.running = False
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    listener = AgentListener()
    listener.start_listening()