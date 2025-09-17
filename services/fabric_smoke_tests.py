#!/usr/bin/env python3
"""
Fabric Smoke Tests
Basic validation suite to ensure all services are running and integrated
Tests health, connectivity, and basic operations for the entire framework
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import psycopg2
import redis
from tabulate import tabulate
from colorama import init, Fore, Back, Style

# Initialize colorama for colored output
init(autoreset=True)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status enum"""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⏭️ SKIPPED"
    WARNING = "⚠️ WARNING"


@dataclass
class TestResult:
    """Test result data class"""
    name: str
    category: str
    status: TestStatus
    message: str
    duration: float
    details: Optional[Dict] = None


class ServiceEndpoint:
    """Service endpoint configuration"""
    def __init__(self, name: str, port: int, health_path: str = "/health"):
        self.name = name
        self.port = port
        self.health_path = health_path
        self.base_url = f"http://localhost:{port}"
        self.health_url = f"{self.base_url}{health_path}"


class FabricSmokeTests:
    """Comprehensive smoke test suite for the AI Agent Framework"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
        # Define all service endpoints
        self.services = [
            ServiceEndpoint("API Gateway", 8000),
            ServiceEndpoint("Auth Service", 8001),
            ServiceEndpoint("Agent Designer", 8002),
            ServiceEndpoint("Workflow Engine", 8003),
            ServiceEndpoint("AI Model Service", 8105),  # Updated port
            ServiceEndpoint("Service Discovery", 8005),
            ServiceEndpoint("Integration Hub", 8006),
            ServiceEndpoint("Monitoring Service", 8007),
            ServiceEndpoint("WebSocket Service", 8008),
            ServiceEndpoint("RL Server", 8010),
            ServiceEndpoint("RL Orchestrator", 8011),
            ServiceEndpoint("Memory Service", 8012),
            ServiceEndpoint("Checkpoint Service", 8013),
            ServiceEndpoint("Batch Accumulator", 8014),
            ServiceEndpoint("AutoGen Integration", 8015),
            ServiceEndpoint("LangGraph Integration", 8016),
            ServiceEndpoint("Code RAG Service", 8017),
            ServiceEndpoint("Polyglot Test Runner", 8018),
            ServiceEndpoint("Git Integration", 8019),
            ServiceEndpoint("Security Gates", 8020),
            ServiceEndpoint("CI/CD Matrix", 8022),
        ]
        
        # Database configuration
        self.db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "agent_lightning"),
            "user": os.getenv("POSTGRES_USER", "agent_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "agent_password")
        }
        
        # Redis configuration
        self.redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "db": 0
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all smoke tests"""
        self.start_time = time.time()
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🔥 FABRIC SMOKE TESTS - AI AGENT FRAMEWORK")
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.WHITE}Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Run test categories
        await self.test_infrastructure()
        await self.test_services()
        await self.test_integrations()
        await self.test_workflows()
        await self.test_agents()
        await self.test_performance()
        
        self.end_time = time.time()
        
        # Generate and display report
        report = self.generate_report()
        self.display_report(report)
        
        return report
    
    async def test_infrastructure(self):
        """Test core infrastructure components"""
        print(f"\n{Fore.YELLOW}🏗️  Testing Infrastructure Components...")
        print(f"{Fore.WHITE}{'-'*60}")
        
        # Test PostgreSQL
        start = time.time()
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            
            # Check critical tables
            tables_to_check = ["agents", "tasks", "workflows", "agent_memories", "checkpoints"]
            missing_tables = []
            
            for table in tables_to_check:
                cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')")
                if not cur.fetchone()[0]:
                    missing_tables.append(table)
            
            conn.close()
            
            if missing_tables:
                self.add_result("PostgreSQL Database", "Infrastructure", TestStatus.WARNING,
                              f"Connected but missing tables: {missing_tables}", time.time() - start)
            else:
                self.add_result("PostgreSQL Database", "Infrastructure", TestStatus.PASSED,
                              f"Connected successfully", time.time() - start,
                              {"version": version[:50]})
        except Exception as e:
            self.add_result("PostgreSQL Database", "Infrastructure", TestStatus.FAILED,
                          f"Connection failed: {str(e)}", time.time() - start)
        
        # Test Redis
        start = time.time()
        try:
            r = redis.Redis(**self.redis_config)
            r.ping()
            info = r.info()
            
            self.add_result("Redis Cache", "Infrastructure", TestStatus.PASSED,
                          f"Connected successfully", time.time() - start,
                          {"version": info.get("redis_version", "unknown"),
                           "used_memory": info.get("used_memory_human", "unknown")})
        except Exception as e:
            self.add_result("Redis Cache", "Infrastructure", TestStatus.FAILED,
                          f"Connection failed: {str(e)}", time.time() - start)
    
    async def test_services(self):
        """Test all microservices health"""
        print(f"\n{Fore.YELLOW}🚀 Testing Microservices...")
        print(f"{Fore.WHITE}{'-'*60}")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for service in self.services:
                tasks.append(self.check_service_health(session, service))
            
            await asyncio.gather(*tasks)
    
    async def check_service_health(self, session: aiohttp.ClientSession, service: ServiceEndpoint):
        """Check individual service health"""
        start = time.time()
        try:
            async with session.get(service.health_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.add_result(service.name, "Services", TestStatus.PASSED,
                                  f"Service healthy", time.time() - start,
                                  {"status": data.get("status", "unknown")})
                else:
                    self.add_result(service.name, "Services", TestStatus.WARNING,
                                  f"Unhealthy (HTTP {resp.status})", time.time() - start)
        except asyncio.TimeoutError:
            self.add_result(service.name, "Services", TestStatus.FAILED,
                          "Timeout - service not responding", time.time() - start)
        except Exception as e:
            self.add_result(service.name, "Services", TestStatus.FAILED,
                          f"Not running", time.time() - start)
    
    async def test_integrations(self):
        """Test service integrations"""
        print(f"\n{Fore.YELLOW}🔗 Testing Service Integrations...")
        print(f"{Fore.WHITE}{'-'*60}")
        
        # Test Agent Registration Integration
        await self.test_agent_registration()
        
        # Test Workflow Creation Integration
        await self.test_workflow_creation()
        
        # Test Task Execution Integration
        await self.test_task_execution()
        
        # Test Memory Persistence Integration
        await self.test_memory_persistence()
    
    async def test_agent_registration(self):
        """Test agent registration flow"""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                # Try to register a test agent
                agent_data = {
                    "name": "smoke-test-agent",
                    "model": "test-model",
                    "capabilities": ["test"]
                }
                
                async with session.post(
                    "http://localhost:8002/agents",
                    json=agent_data,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status in [200, 201]:
                        self.add_result("Agent Registration", "Integration", TestStatus.PASSED,
                                      "Agent registration working", time.time() - start)
                    else:
                        self.add_result("Agent Registration", "Integration", TestStatus.WARNING,
                                      f"Registration returned {resp.status}", time.time() - start)
        except Exception as e:
            self.add_result("Agent Registration", "Integration", TestStatus.FAILED,
                          f"Registration failed: {str(e)[:50]}", time.time() - start)
    
    async def test_workflow_creation(self):
        """Test workflow creation"""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                workflow_data = {
                    "name": "smoke-test-workflow",
                    "nodes": ["start", "end"],
                    "edges": [["start", "end"]]
                }
                
                async with session.post(
                    "http://localhost:8016/workflows",
                    json=workflow_data,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status in [200, 201]:
                        self.add_result("Workflow Creation", "Integration", TestStatus.PASSED,
                                      "Workflow creation working", time.time() - start)
                    else:
                        self.add_result("Workflow Creation", "Integration", TestStatus.WARNING,
                                      f"Creation returned {resp.status}", time.time() - start)
        except Exception as e:
            self.add_result("Workflow Creation", "Integration", TestStatus.FAILED,
                          f"Creation failed: {str(e)[:50]}", time.time() - start)
    
    async def test_task_execution(self):
        """Test task execution flow"""
        start = time.time()
        try:
            # Check if task execution is available
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
            completed_tasks = cur.fetchone()[0]
            conn.close()
            
            if completed_tasks > 0:
                self.add_result("Task Execution", "Integration", TestStatus.PASSED,
                              f"{completed_tasks} tasks completed", time.time() - start)
            else:
                self.add_result("Task Execution", "Integration", TestStatus.WARNING,
                              "No completed tasks found", time.time() - start)
        except Exception as e:
            self.add_result("Task Execution", "Integration", TestStatus.FAILED,
                          f"Check failed: {str(e)[:50]}", time.time() - start)
    
    async def test_memory_persistence(self):
        """Test memory persistence"""
        start = time.time()
        try:
            # Check Redis for memory keys
            r = redis.Redis(**self.redis_config)
            memory_keys = r.keys("memory:*")
            
            if memory_keys:
                self.add_result("Memory Persistence", "Integration", TestStatus.PASSED,
                              f"{len(memory_keys)} memories cached", time.time() - start)
            else:
                self.add_result("Memory Persistence", "Integration", TestStatus.WARNING,
                              "No memories found in cache", time.time() - start)
        except Exception as e:
            self.add_result("Memory Persistence", "Integration", TestStatus.FAILED,
                          f"Check failed: {str(e)[:50]}", time.time() - start)
    
    async def test_workflows(self):
        """Test workflow operations"""
        print(f"\n{Fore.YELLOW}⚙️  Testing Workflow Operations...")
        print(f"{Fore.WHITE}{'-'*60}")
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                # Check LangGraph workflows
                async with session.get(
                    "http://localhost:8016/workflows",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        workflow_count = len(data.get("workflows", []))
                        
                        if workflow_count > 0:
                            self.add_result("Workflow Registry", "Workflows", TestStatus.PASSED,
                                          f"{workflow_count} workflows registered", time.time() - start)
                        else:
                            self.add_result("Workflow Registry", "Workflows", TestStatus.WARNING,
                                          "No workflows registered", time.time() - start)
                    else:
                        self.add_result("Workflow Registry", "Workflows", TestStatus.FAILED,
                                      f"Registry check failed", time.time() - start)
        except Exception as e:
            self.add_result("Workflow Registry", "Workflows", TestStatus.FAILED,
                          f"Check failed: {str(e)[:50]}", time.time() - start)
    
    async def test_agents(self):
        """Test agent operations"""
        print(f"\n{Fore.YELLOW}🤖 Testing Agent Operations...")
        print(f"{Fore.WHITE}{'-'*60}")
        
        start = time.time()
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Check development agents
            dev_agents = [
                'router-agent', 'planner-agent', 'retriever-agent',
                'coder-agent', 'tester-agent', 'reviewer-agent', 'integrator-agent'
            ]
            
            cur.execute("SELECT id FROM agents WHERE id = ANY(%s)", (dev_agents,))
            found_agents = [row[0] for row in cur.fetchall()]
            
            if len(found_agents) == len(dev_agents):
                self.add_result("Development Agents", "Agents", TestStatus.PASSED,
                              f"All {len(dev_agents)} agents registered", time.time() - start)
            elif found_agents:
                self.add_result("Development Agents", "Agents", TestStatus.WARNING,
                              f"Only {len(found_agents)}/{len(dev_agents)} agents found", time.time() - start,
                              {"found": found_agents})
            else:
                self.add_result("Development Agents", "Agents", TestStatus.FAILED,
                              "No development agents found", time.time() - start)
            
            conn.close()
        except Exception as e:
            self.add_result("Development Agents", "Agents", TestStatus.FAILED,
                          f"Check failed: {str(e)[:50]}", time.time() - start)
    
    async def test_performance(self):
        """Test basic performance metrics"""
        print(f"\n{Fore.YELLOW}⚡ Testing Performance Metrics...")
        print(f"{Fore.WHITE}{'-'*60}")
        
        # Test service response times
        async with aiohttp.ClientSession() as session:
            for service in self.services[:5]:  # Test first 5 services
                start = time.time()
                try:
                    async with session.get(
                        service.health_url,
                        timeout=aiohttp.ClientTimeout(total=1)
                    ) as resp:
                        response_time = (time.time() - start) * 1000  # Convert to ms
                        
                        if response_time < 100:
                            status = TestStatus.PASSED
                            message = f"Response time: {response_time:.1f}ms"
                        elif response_time < 500:
                            status = TestStatus.WARNING
                            message = f"Slow response: {response_time:.1f}ms"
                        else:
                            status = TestStatus.FAILED
                            message = f"Very slow: {response_time:.1f}ms"
                        
                        self.add_result(f"{service.name} Response", "Performance", status,
                                      message, response_time / 1000)
                except:
                    pass  # Skip if service is down
    
    def add_result(self, name: str, category: str, status: TestStatus, 
                   message: str, duration: float, details: Optional[Dict] = None):
        """Add a test result"""
        result = TestResult(name, category, status, message, duration, details)
        self.results.append(result)
        
        # Print real-time result
        status_color = {
            TestStatus.PASSED: Fore.GREEN,
            TestStatus.FAILED: Fore.RED,
            TestStatus.WARNING: Fore.YELLOW,
            TestStatus.SKIPPED: Fore.BLUE
        }.get(status, Fore.WHITE)
        
        print(f"  {status_color}{status.value} {Fore.WHITE}{name}: {message}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total_duration = self.end_time - self.start_time if self.end_time else 0
        
        # Count results by status
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        warnings = sum(1 for r in self.results if r.status == TestStatus.WARNING)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        return {
            "summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped,
                "duration": total_duration,
                "pass_rate": (passed / len(self.results) * 100) if self.results else 0
            },
            "categories": categories,
            "results": self.results,
            "timestamp": datetime.now().isoformat()
        }
    
    def display_report(self, report: Dict[str, Any]):
        """Display the test report"""
        summary = report["summary"]
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}📊 TEST RESULTS SUMMARY")
        print(f"{Fore.CYAN}{'='*80}")
        
        # Summary table
        summary_data = [
            ["Total Tests", summary["total_tests"]],
            ["Passed", f"{Fore.GREEN}{summary['passed']}{Fore.WHITE}"],
            ["Failed", f"{Fore.RED}{summary['failed']}{Fore.WHITE}"],
            ["Warnings", f"{Fore.YELLOW}{summary['warnings']}{Fore.WHITE}"],
            ["Skipped", f"{Fore.BLUE}{summary['skipped']}{Fore.WHITE}"],
            ["Pass Rate", f"{summary['pass_rate']:.1f}%"],
            ["Duration", f"{summary['duration']:.2f}s"]
        ]
        
        print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Category breakdown
        print(f"\n{Fore.CYAN}📋 RESULTS BY CATEGORY")
        print(f"{Fore.CYAN}{'-'*80}")
        
        for category, results in report["categories"].items():
            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in results if r.status == TestStatus.FAILED)
            warnings = sum(1 for r in results if r.status == TestStatus.WARNING)
            
            print(f"\n{Fore.YELLOW}{category}:")
            print(f"  ✅ Passed: {passed} | ❌ Failed: {failed} | ⚠️ Warnings: {warnings}")
            
            # Show failed tests in detail
            for result in results:
                if result.status == TestStatus.FAILED:
                    print(f"    {Fore.RED}❌ {result.name}: {result.message}")
        
        # Overall status
        print(f"\n{Fore.CYAN}{'='*80}")
        if summary["failed"] == 0:
            if summary["warnings"] == 0:
                print(f"{Fore.GREEN}✅ ALL TESTS PASSED! The framework is fully operational.")
            else:
                print(f"{Fore.YELLOW}⚠️ TESTS PASSED WITH WARNINGS. Review warnings above.")
        else:
            print(f"{Fore.RED}❌ SOME TESTS FAILED. Review failures above.")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def export_report(self, report: Dict[str, Any], filename: str = "smoke_test_report.json"):
        """Export report to JSON file"""
        # Convert TestResult objects to dictionaries
        export_data = {
            "summary": report["summary"],
            "timestamp": report["timestamp"],
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "status": r.status.value,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in report["results"]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"{Fore.GREEN}Report exported to {filename}")


async def main():
    """Main entry point"""
    tester = FabricSmokeTests()
    report = await tester.run_all_tests()
    
    # Export report
    tester.export_report(report)
    
    # Return exit code based on results
    if report["summary"]["failed"] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)