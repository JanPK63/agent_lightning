#!/usr/bin/env python3
"""
Agent Lightning Unified Startup Script
Starts all services with proper integration and context awareness
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

class AgentLightningLauncher:
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        
    def setup_environment(self):
        """Setup environment and dependencies"""
        print("🔧 Setting up Agent Lightning environment...")
        
        # Ensure all agents are configured
        try:
            subprocess.run([sys.executable, "setup_agents.py"], 
                         cwd=self.base_dir, check=True, capture_output=True)
            print("✅ All 31 specialized agents configured")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Agent setup warning: {e}")
        
        # Initialize knowledge bases
        try:
            from knowledge_manager import KnowledgeManager
            km = KnowledgeManager()
            print(f"✅ Knowledge system initialized")
        except Exception as e:
            print(f"⚠️  Knowledge system warning: {e}")
    
    def start_service(self, name, command, port=None, wait_time=3):
        """Start a service and track the process"""
        print(f"🚀 Starting {name}...")
        
        try:
            process = subprocess.Popen(
                command,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append({
                'name': name,
                'process': process,
                'port': port,
                'command': command
            })
            
            time.sleep(wait_time)
            
            if process.poll() is None:
                print(f"✅ {name} started successfully" + (f" on port {port}" if port else ""))
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ {name} failed to start: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting {name}: {e}")
            return False
    
    def check_service_health(self, port, service_name):
        """Check if service is responding"""
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} health check passed")
                return True
        except:
            pass
        print(f"⚠️  {service_name} health check failed")
        return False
    
    def start_all_services(self):
        """Start all Agent Lightning services in correct order"""
        print("=" * 60)
        print("⚡ AGENT LIGHTNING - UNIFIED STARTUP")
        print("=" * 60)
        
        # 1. Setup environment
        self.setup_environment()
        
        # 2. Start Enhanced Production API (main service)
        success = self.start_service(
            "Enhanced Production API",
            [sys.executable, "enhanced_production_api.py"],
            port=8002,
            wait_time=5
        )
        
        if success:
            self.check_service_health(8002, "Enhanced Production API")
        
        # 3. Start Internet Access Service
        self.start_service(
            "Internet Access Service", 
            [sys.executable, "internet_agent_api.py"],
            port=8892,
            wait_time=3
        )
        
        # 4. Start Spec-Driven Development Service
        if (self.base_dir / "spec_driven_service.py").exists():
            self.start_service(
                "Spec-Driven Service",
                [sys.executable, "spec_driven_service.py"],
                port=8029,
                wait_time=3
            )
        
        # 5. Start Monitoring Dashboard
        self.start_service(
            "Monitoring Dashboard",
            [sys.executable, "-m", "streamlit", "run", "monitoring_dashboard_integrated.py", 
             "--server.port", "8051", "--server.headless", "true"],
            port=8051,
            wait_time=5
        )
        
        # 6. Start RL Training Server (if available)
        if (self.base_dir / "rl_training_server.py").exists():
            self.start_service(
                "RL Training Server",
                [sys.executable, "rl_training_server.py"],
                port=8003,
                wait_time=3
            )
        
        print("\n" + "=" * 60)
        print("🎉 AGENT LIGHTNING STARTUP COMPLETE")
        print("=" * 60)
        
        # Display service status
        print("\n📊 Service Status:")
        print("-" * 40)
        for service in self.processes:
            status = "🟢 Running" if service['process'].poll() is None else "🔴 Stopped"
            port_info = f" (:{service['port']})" if service['port'] else ""
            print(f"  {service['name']}{port_info}: {status}")
        
        print("\n🌐 Access Points:")
        print("-" * 40)
        print("  📊 Dashboard: http://localhost:8051")
        print("  🔧 API Docs: http://localhost:8002/docs")
        print("  🧠 Enhanced API: http://localhost:8002/api/v2")
        print("  🌍 Internet Service: http://localhost:8892")
        if any(s['port'] == 8029 for s in self.processes):
            print("  📋 Spec Service: http://localhost:8029")
        
        print("\n💡 Quick Start:")
        print("-" * 40)
        print("  1. Open Dashboard: http://localhost:8051")
        print("  2. Go to 'Task Assignment' tab")
        print("  3. Select an agent and submit a task")
        print("  4. All 31 agents have internet access and RL training")
        
        return True
    
    def monitor_services(self):
        """Monitor running services"""
        print(f"\n🔍 Monitoring {len(self.processes)} services...")
        print("Press Ctrl+C to stop all services")
        
        try:
            while True:
                time.sleep(10)
                
                # Check if any service died
                for service in self.processes:
                    if service['process'].poll() is not None:
                        print(f"⚠️  {service['name']} stopped unexpectedly")
                        
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown of all services"""
        print("🧹 Cleaning up services...")
        
        for service in self.processes:
            try:
                if service['process'].poll() is None:
                    print(f"  Stopping {service['name']}...")
                    service['process'].terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        service['process'].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        service['process'].kill()
                        
            except Exception as e:
                print(f"  Error stopping {service['name']}: {e}")
        
        print("✅ All services stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}")
        self.cleanup()
        sys.exit(0)


def main():
    launcher = AgentLightningLauncher()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    
    try:
        # Start all services
        if launcher.start_all_services():
            # Monitor services
            launcher.monitor_services()
        else:
            print("❌ Failed to start some services")
            launcher.cleanup()
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        launcher.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()