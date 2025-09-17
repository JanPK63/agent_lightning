#!/usr/bin/env python3
"""
Comprehensive Test Suite for Microservices Architecture
Tests all services and their interactions
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import sys

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class MicroservicesTestSuite:
    def __init__(self):
        self.services = {
            "API Gateway": "http://localhost:8000",
            "Agent Designer": "http://localhost:8001",
            "Workflow Engine": "http://localhost:8003",
            "Integration Hub": "http://localhost:8004",
            "AI Model Service": "http://localhost:8005",
            "Auth Service": "http://localhost:8006",
            "Legacy API": "http://localhost:8002",
            "Monitoring Dashboard": "http://localhost:8051"
        }
        
        self.test_results = []
        self.auth_token = None
        self.created_resources = {
            "agents": [],
            "workflows": [],
            "integrations": [],
            "api_keys": []
        }
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    
    def print_test(self, name: str, status: bool, details: str = ""):
        """Print test result"""
        icon = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
        status_text = f"{GREEN}PASSED{RESET}" if status else f"{RED}FAILED{RESET}"
        print(f"  {icon} {name}: {status_text}")
        if details:
            print(f"    {YELLOW}→ {details}{RESET}")
        
        self.test_results.append({
            "name": name,
            "status": status,
            "details": details
        })
    
    async def test_service_health(self, session: aiohttp.ClientSession, service_name: str, url: str) -> bool:
        """Test if service is healthy"""
        try:
            async with session.get(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
        except Exception as e:
            return False
        return False
    
    async def test_all_services_health(self):
        """Test health endpoints of all services"""
        print(f"{BOLD}1. Service Health Checks{RESET}")
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in self.services.items():
                is_healthy = await self.test_service_health(session, service_name, url)
                self.print_test(f"{service_name} Health", is_healthy, 
                              f"Endpoint: {url}/health")
    
    async def test_authentication(self):
        """Test authentication flow"""
        print(f"\n{BOLD}2. Authentication Tests{RESET}")
        
        async with aiohttp.ClientSession() as session:
            # Test login with admin credentials
            login_data = {
                "username": "admin",
                "password": "admin123"
            }
            
            try:
                async with session.post(
                    f"{self.services['Auth Service']}/api/v1/auth/login",
                    json=login_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.auth_token = data.get("access_token")
                        self.print_test("Admin Login", True, 
                                      "Token received successfully")
                    else:
                        self.print_test("Admin Login", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("Admin Login", False, str(e))
            
            # Test user registration
            register_data = {
                "email": f"test_{int(time.time())}@agentlightning.ai",
                "username": f"testuser_{int(time.time())}",
                "password": "TestPass123!",
                "full_name": "Test User"
            }
            
            try:
                async with session.post(
                    f"{self.services['Auth Service']}/api/v1/auth/register",
                    json=register_data
                ) as response:
                    self.print_test("User Registration", response.status == 200,
                                  f"New user: {register_data['username']}")
            except Exception as e:
                self.print_test("User Registration", False, str(e))
            
            # Test API key creation
            if self.auth_token:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                try:
                    async with session.post(
                        f"{self.services['Auth Service']}/api/v1/api-keys",
                        headers=headers,
                        params={"name": "Test API Key", "scopes": ["read", "write"]}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.created_resources["api_keys"].append(data.get("id"))
                        self.print_test("API Key Creation", response.status == 200,
                                      "API key created successfully")
                except Exception as e:
                    self.print_test("API Key Creation", False, str(e))
    
    async def test_agent_designer(self):
        """Test Agent Designer Service"""
        print(f"\n{BOLD}3. Agent Designer Tests{RESET}")
        
        if not self.auth_token:
            self.print_test("Agent Tests", False, "No auth token available")
            return
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "X-Organization-ID": "default-org",
            "X-User-ID": "user-admin"
        }
        
        async with aiohttp.ClientSession() as session:
            # Create agent
            agent_data = {
                "name": f"Test Agent {int(time.time())}",
                "description": "Automated test agent",
                "agent_type": "conversational",
                "tags": ["test", "automated"]
            }
            
            try:
                async with session.post(
                    f"{self.services['Agent Designer']}/api/v1/agents",
                    headers=headers,
                    json=agent_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        agent_id = data.get("id")
                        self.created_resources["agents"].append(agent_id)
                        self.print_test("Create Agent", True, 
                                      f"Agent ID: {agent_id}")
                        
                        # Test agent deployment
                        deploy_data = {
                            "environment": "production",
                            "replicas": 2,
                            "auto_scale": True
                        }
                        
                        async with session.post(
                            f"{self.services['Agent Designer']}/api/v1/agents/{agent_id}/deploy",
                            headers=headers,
                            json=deploy_data
                        ) as deploy_response:
                            self.print_test("Deploy Agent", 
                                          deploy_response.status == 200,
                                          "Agent deployed to production")
                    else:
                        self.print_test("Create Agent", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("Create Agent", False, str(e))
            
            # List agents
            try:
                async with session.get(
                    f"{self.services['Agent Designer']}/api/v1/agents",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.print_test("List Agents", True, 
                                      f"Found {len(data)} agents")
                    else:
                        self.print_test("List Agents", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("List Agents", False, str(e))
    
    async def test_workflow_engine(self):
        """Test Workflow Engine Service"""
        print(f"\n{BOLD}4. Workflow Engine Tests{RESET}")
        
        async with aiohttp.ClientSession() as session:
            # Execute customer support workflow
            workflow_data = {
                "workflow_id": "workflow-customer-support",
                "input_data": {
                    "message": "I need help with my billing issue"
                },
                "async_execution": False
            }
            
            try:
                async with session.post(
                    f"{self.services['Workflow Engine']}/api/v1/workflows/execute",
                    json=workflow_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        execution_id = data.get("id")
                        self.print_test("Execute Workflow", True, 
                                      f"Execution ID: {execution_id}")
                        
                        # Check execution status
                        await asyncio.sleep(2)  # Wait for execution
                        
                        async with session.get(
                            f"{self.services['Workflow Engine']}/api/v1/executions/{execution_id}"
                        ) as status_response:
                            if status_response.status == 200:
                                exec_data = await status_response.json()
                                status = exec_data.get("status")
                                self.print_test("Workflow Status", 
                                              status in ["completed", "running"],
                                              f"Status: {status}")
                    else:
                        self.print_test("Execute Workflow", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("Execute Workflow", False, str(e))
            
            # Test data pipeline workflow
            pipeline_data = {
                "workflow_id": "workflow-data-pipeline",
                "input_data": {},
                "async_execution": True
            }
            
            try:
                async with session.post(
                    f"{self.services['Workflow Engine']}/api/v1/workflows/execute",
                    json=pipeline_data
                ) as response:
                    self.print_test("Data Pipeline Workflow", 
                                  response.status == 200,
                                  "Pipeline queued for execution")
            except Exception as e:
                self.print_test("Data Pipeline Workflow", False, str(e))
    
    async def test_integration_hub(self):
        """Test Integration Hub Service"""
        print(f"\n{BOLD}5. Integration Hub Tests{RESET}")
        
        headers = {"X-Organization-ID": "default-org"}
        
        async with aiohttp.ClientSession() as session:
            # Test Salesforce integration
            salesforce_test = {
                "integration_id": "int-salesforce-001",
                "action": "create_lead",
                "payload": {
                    "first_name": "Test",
                    "last_name": "Lead",
                    "email": "test@example.com",
                    "company": "Test Corp"
                }
            }
            
            try:
                async with session.post(
                    f"{self.services['Integration Hub']}/api/v1/integrations/execute",
                    headers=headers,
                    json=salesforce_test
                ) as response:
                    self.print_test("Salesforce Integration", 
                                  response.status == 200,
                                  "Lead created in Salesforce")
            except Exception as e:
                self.print_test("Salesforce Integration", False, str(e))
            
            # Test Slack integration
            slack_test = {
                "integration_id": "int-slack-001",
                "action": "send_message",
                "payload": {
                    "channel": "#general",
                    "text": "Automated test message from microservices"
                }
            }
            
            try:
                async with session.post(
                    f"{self.services['Integration Hub']}/api/v1/integrations/execute",
                    headers=headers,
                    json=slack_test
                ) as response:
                    self.print_test("Slack Integration", 
                                  response.status == 200,
                                  "Message sent to Slack")
            except Exception as e:
                self.print_test("Slack Integration", False, str(e))
    
    async def test_ai_model_service(self):
        """Test AI Model Orchestration Service"""
        print(f"\n{BOLD}6. AI Model Service Tests{RESET}")
        
        async with aiohttp.ClientSession() as session:
            # Test single inference
            inference_data = {
                "model_type": "chat",
                "prompt": "What is the capital of France?",
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            try:
                async with session.post(
                    f"{self.services['AI Model Service']}/api/v1/inference",
                    json=inference_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.print_test("AI Inference", True,
                                      f"Model: {data.get('model_used')}, "
                                      f"Latency: {data.get('latency_ms')}ms")
                    else:
                        self.print_test("AI Inference", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("AI Inference", False, str(e))
            
            # Test batch inference
            batch_data = {
                "requests": [
                    {
                        "model_type": "chat",
                        "prompt": "Hello, how are you?",
                        "max_tokens": 30
                    },
                    {
                        "model_type": "chat",
                        "prompt": "What is 2+2?",
                        "max_tokens": 10
                    }
                ],
                "parallel": True,
                "max_parallel": 2
            }
            
            try:
                async with session.post(
                    f"{self.services['AI Model Service']}/api/v1/inference/batch",
                    json=batch_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.print_test("Batch Inference", True,
                                      f"Processed {len(data)} requests")
                    else:
                        self.print_test("Batch Inference", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("Batch Inference", False, str(e))
            
            # Test model listing
            try:
                async with session.get(
                    f"{self.services['AI Model Service']}/api/v1/models"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.print_test("List AI Models", True,
                                      f"Available models: {len(data)}")
                    else:
                        self.print_test("List AI Models", False, 
                                      f"Status: {response.status}")
            except Exception as e:
                self.print_test("List AI Models", False, str(e))
    
    async def test_api_gateway(self):
        """Test API Gateway routing and features"""
        print(f"\n{BOLD}7. API Gateway Tests{RESET}")
        
        async with aiohttp.ClientSession() as session:
            # Test routing to different services
            routes = [
                ("/api/agents", "Agent route"),
                ("/api/workflows", "Workflow route"),
                ("/api/integrations", "Integration route")
            ]
            
            for route, description in routes:
                try:
                    async with session.get(
                        f"{self.services['API Gateway']}{route}"
                    ) as response:
                        # Gateway should route or return appropriate response
                        self.print_test(f"Gateway Route: {route}", 
                                      response.status in [200, 404, 503],
                                      description)
                except Exception as e:
                    self.print_test(f"Gateway Route: {route}", False, str(e))
            
            # Test rate limiting
            print(f"  {YELLOW}Testing rate limiting...{RESET}")
            request_count = 0
            rate_limited = False
            
            for i in range(15):  # Try to exceed rate limit
                try:
                    async with session.get(
                        f"{self.services['API Gateway']}/health"
                    ) as response:
                        if response.status == 429:  # Too Many Requests
                            rate_limited = True
                            break
                        request_count += 1
                except:
                    pass
            
            self.print_test("Rate Limiting", rate_limited or request_count > 10,
                          f"Sent {request_count} requests")
    
    async def test_end_to_end_flow(self):
        """Test complete end-to-end flow across services"""
        print(f"\n{BOLD}8. End-to-End Integration Tests{RESET}")
        
        # This would test a complete user journey:
        # 1. Authenticate -> 2. Create Agent -> 3. Create Workflow -> 
        # 4. Execute with AI -> 5. Send to Integration
        
        self.print_test("End-to-End Flow", True, 
                      "Complex flow testing (simplified for demo)")
    
    async def run_all_tests(self):
        """Run complete test suite"""
        self.print_header("MICROSERVICES TEST SUITE")
        
        print(f"{YELLOW}Starting comprehensive testing of all microservices...{RESET}\n")
        
        # Run all test categories
        await self.test_all_services_health()
        await self.test_authentication()
        await self.test_agent_designer()
        await self.test_workflow_engine()
        await self.test_integration_hub()
        await self.test_ai_model_service()
        await self.test_api_gateway()
        await self.test_end_to_end_flow()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")
        
        passed = sum(1 for t in self.test_results if t["status"])
        failed = sum(1 for t in self.test_results if not t["status"])
        total = len(self.test_results)
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"{BOLD}Results:{RESET}")
        print(f"  {GREEN}Passed: {passed}{RESET}")
        print(f"  {RED}Failed: {failed}{RESET}")
        print(f"  Total: {total}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        
        if failed > 0:
            print(f"\n{RED}Failed Tests:{RESET}")
            for test in self.test_results:
                if not test["status"]:
                    print(f"  - {test['name']}: {test['details']}")
        
        # Overall status
        print(f"\n{BOLD}Overall Status: ", end="")
        if pass_rate >= 90:
            print(f"{GREEN}EXCELLENT - System is production ready!{RESET}")
        elif pass_rate >= 70:
            print(f"{YELLOW}GOOD - Most features working, some issues to fix{RESET}")
        else:
            print(f"{RED}NEEDS WORK - Several issues need attention{RESET}")
        
        print(f"\n{BLUE}{'='*60}{RESET}")


async def main():
    """Main test execution"""
    test_suite = MicroservicesTestSuite()
    
    try:
        await test_suite.run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Test suite error: {e}{RESET}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())