"""
Task assignment interface for Agent Lightning Monitoring Dashboard
"""

import streamlit as st
import requests
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from .models import (
    TaskAssignmentRequest, TaskAssignmentResponse,
    TaskFeedbackRequest, APIConnectionRequest,
    ChatRequest, ChatResponse, DashboardConfig
)


class TaskAssignmentInterface:
    """Handles task assignment and agent interaction"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def _make_resilient_request(self, url: str, method: str = "GET",
                               json_data: Optional[Dict] = None,
                               headers: Optional[Dict] = None,
                               timeout: float = 10.0) -> Optional[requests.Response]:
        """
        Make HTTP request with circuit breaker and exponential backoff
        Returns response object or None on failure
        """
        service_key = url.split("://")[1].split("/")[0]  # Extract service identifier

        # Initialize circuit breaker tracking if not exists
        if not hasattr(self, 'circuit_breaker_failures'):
            self.circuit_breaker_failures = {}
            self.circuit_breaker_threshold = 3
            self.circuit_breaker_timeout = 60  # seconds

        # Check circuit breaker
        if service_key in self.circuit_breaker_failures:
            failure_count, last_failure = self.circuit_breaker_failures[service_key]
            if failure_count >= self.circuit_breaker_threshold:
                time_since_failure = (datetime.now() - last_failure).seconds
                if time_since_failure < self.circuit_breaker_timeout:
                    st.warning(f"Circuit breaker OPEN for {service_key}, skipping request")
                    return None
                else:
                    # Half-open: reset failure count
                    self.circuit_breaker_failures[service_key] = (0, datetime.now())

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    json=json_data,
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    # Success: reset circuit breaker
                    if service_key in self.circuit_breaker_failures:
                        self.circuit_breaker_failures[service_key] = (0, datetime.now())
                    return response
                else:
                    st.warning(f"Request to {url} failed with status {response.status_code}")
                    # Record failure for circuit breaker
                    if service_key not in self.circuit_breaker_failures:
                        self.circuit_breaker_failures[service_key] = (1, datetime.now())
                    else:
                        count, _ = self.circuit_breaker_failures[service_key]
                        self.circuit_breaker_failures[service_key] = (count + 1, datetime.now())
                    return response  # Return response even on error for proper error handling

            except requests.exceptions.Timeout:
                st.warning(f"Request to {url} timed out (attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                st.warning(f"Connection error to {url} (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                st.error(f"Request to {url} failed: {e}")

            # Record failure for circuit breaker
            if service_key not in self.circuit_breaker_failures:
                self.circuit_breaker_failures[service_key] = (1, datetime.now())
            else:
                count, _ = self.circuit_breaker_failures[service_key]
                self.circuit_breaker_failures[service_key] = (count + 1, datetime.now())

            # Exponential backoff with jitter
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + np.random.random() * 0.1
                time.sleep(delay)

        # All retries failed
        st.error(f"All {max_retries} attempts to {url} failed")
        return None

    def _validate_request_payload(self, payload: Dict[str, Any], schema_class) -> Dict[str, Any]:
        """Validate request payload using Pydantic schema"""
        try:
            validated = schema_class(**payload)
            return validated.dict()
        except Exception as e:
            st.error(f"Request validation failed: {str(e)}")
            raise ValueError(f"Request validation failed: {str(e)}")

    def _validate_response_payload(self, response_data: Dict[str, Any], schema_class) -> Dict[str, Any]:
        """Validate response payload using Pydantic schema"""
        try:
            validated = schema_class(**response_data)
            return validated.dict()
        except Exception as e:
            st.warning(f"Response validation failed: {str(e)}")
            # Don't show error to user for response validation failures, just log
            return response_data  # Return original data if validation fails

    def _check_service_health(self, service_url: str, service_name: str) -> Dict[str, Any]:
        """
        Check health of a dependent service
        Returns: {"healthy": bool, "response_time": float, "status_code": int, "error": str}
        """
        try:
            start_time = time.time()
            response = self._make_resilient_request(
                f"{service_url}/health",
                method="GET",
                timeout=3.0
            )
            response_time = time.time() - start_time

            if response and response.status_code == 200:
                return {
                    "healthy": True,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "error": None
                }
            else:
                status_code = response.status_code if response else "TIMEOUT"
                return {
                    "healthy": False,
                    "response_time": response_time,
                    "status_code": status_code,
                    "error": f"Service {service_name} returned status {status_code}"
                }
        except Exception as e:
            return {
                "healthy": False,
                "response_time": float('inf'),
                "status_code": "ERROR",
                "error": str(e)
            }

    def _check_sla_compliance(self, priority: str) -> Dict[str, Any]:
        """
        Check if dependent services meet SLA requirements for given priority
        Returns: {"compliant": bool, "unhealthy_services": List[str], "warnings": List[str]}
        """
        # Define SLA requirements by priority
        sla_requirements = {
            "urgent": {"max_response_time": 1.0, "require_all_healthy": True},
            "high": {"max_response_time": 2.0, "require_all_healthy": True},
            "normal": {"max_response_time": 5.0, "require_all_healthy": False},
            "low": {"max_response_time": 10.0, "require_all_healthy": False}
        }

        requirements = sla_requirements.get(priority, sla_requirements["normal"])

        # Check dependent services
        services_to_check = [
            ("http://localhost:8002", "Enhanced API"),
            ("http://localhost:8025", "RL Orchestrator"),
            ("http://localhost:5432", "PostgreSQL")  # Note: This would need a different health check
        ]

        unhealthy_services = []
        warnings = []

        for service_url, service_name in services_to_check:
            if service_name == "PostgreSQL":
                # Skip PostgreSQL health check for now (would need different implementation)
                continue

            health = self._check_service_health(service_url, service_name)

            if not health["healthy"]:
                unhealthy_services.append(service_name)
            elif health["response_time"] > requirements["max_response_time"]:
                warnings.append(f"{service_name} response time ({health['response_time']:.2f}s) exceeds SLA ({requirements['max_response_time']}s)")

        # Determine compliance
        if requirements["require_all_healthy"] and unhealthy_services:
            compliant = False
        elif unhealthy_services:
            # For non-urgent priorities, allow some services to be unhealthy
            compliant = len(unhealthy_services) <= 1
        else:
            compliant = True

        return {
            "compliant": compliant,
            "unhealthy_services": unhealthy_services,
            "warnings": warnings
        }

    def _emit_telemetry(self, event_type: str, data: Dict[str, Any]):
        """
        Emit telemetry data for monitoring and analytics
        """
        try:
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "source": "dashboard",
                **data
            }

            # For now, just log the telemetry data
            st.info(f"TELEMETRY: {event_type} - {telemetry_data}")

        except Exception as e:
            st.warning(f"Failed to emit telemetry: {e}")

    def get_available_agents(self) -> List[str]:
        """Get list of available agents from the API"""
        try:
            # Try to get agents from the enhanced API
            api_url = "http://localhost:8002/api/v2/agents/list"
            response = requests.get(api_url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                # Extract agent IDs from the response
                agents = [agent['id'] for agent in data.get('agents', [])]
                if agents:
                    return agents
        except Exception as e:
            st.warning(f"API connection failed: {e}")

        # Force refresh from API
        try:
            from agent_config import AgentConfigManager
            config_manager = AgentConfigManager()
            agents = config_manager.list_agents()
            if agents:
                st.info(f"✅ Loaded {len(agents)} agents from config manager")
                return agents
        except Exception as e:
            st.warning(f"Config manager failed: {e}")

        # All 31 specialized agents fallback
        return [
            "full_stack_developer", "mobile_developer", "frontend_developer", "backend_developer", "game_developer",
            "devops_engineer", "cloud_architect", "site_reliability_engineer", "platform_engineer", "network_engineer",
            "security_expert", "compliance_officer", "data_scientist", "ai_ml_engineer", "database_specialist",
            "qa_engineer", "test_automation_engineer", "performance_engineer", "system_architect", "solution_architect",
            "ui_ux_designer", "blockchain_developer", "embedded_systems_engineer", "integration_specialist",
            "technical_lead", "project_manager", "product_manager", "business_analyst", "systems_analyst",
            "technical_writer", "research_scientist"
        ]

    def assign_task(self, task_description: str, agent_type: str, priority: str,
                   ai_model: str, reference_task_id: Optional[str] = None,
                   deployment_config: Optional[Dict] = None) -> Optional[Dict]:
        """Assign a task to an agent"""
        try:
            # SLA Check for high-priority tasks
            if priority in ["urgent", "high"]:
                sla_check = self._check_sla_compliance(priority)
                if not sla_check["compliant"]:
                    st.error(f"🚫 **SLA Violation**: Cannot submit {priority} priority task. Unhealthy services: {', '.join(sla_check['unhealthy_services'])}")
                    if sla_check["warnings"]:
                        for warning in sla_check["warnings"]:
                            st.warning(f"⚠️ {warning}")
                    st.info("💡 Wait for services to recover or reduce task priority to proceed.")
                    return None

            headers = {
                "Authorization": f"Bearer {st.session_state.get('api_token', '')}",
                "Content-Type": "application/json"
            }

            # Build context with reference task if provided
            context = {
                "deployment": deployment_config if deployment_config else None,
                "model": ai_model
            }

            if reference_task_id:
                context["reference_task_id"] = reference_task_id

            # Validate request payload before sending
            request_payload = {
                "task_description": task_description,
                "agent_type": agent_type,
                "priority": priority,
                "ai_model": ai_model,
                "reference_task_id": reference_task_id,
                "deployment_config": deployment_config
            }
            validated_payload = self._validate_request_payload(request_payload, TaskAssignmentRequest)

            # Emit telemetry for task validation
            self._emit_telemetry("task_validation", {
                "priority": priority,
                "agent_type": agent_type,
                "ai_model": ai_model,
                "has_deployment": deployment_config is not None,
                "has_reference": bool(reference_task_id),
                "task_description_length": len(task_description)
            })

            # Track task submission latency
            submission_start_time = time.time()

            # Execute task using agent service orchestrator
            if st.session_state.get('orchestrator'):
                try:
                    import asyncio
                    # Run orchestrator in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    task_result = loop.run_until_complete(
                        st.session_state.orchestrator.execute_task(
                            task_description=validated_payload["task_description"],
                            agent_id=validated_payload["agent_type"] if validated_payload["agent_type"] != "auto" else None,
                            context={
                                "deployment": validated_payload["deployment_config"],
                                "model": validated_payload["ai_model"],
                                "reference_task_id": validated_payload["reference_task_id"]
                            }
                        )
                    )

                    # Convert orchestrator result to expected format
                    task_response = type('MockResponse', (), {
                        'status_code': 200,
                        'json': lambda: {
                            "task_id": f"orchestrator_{int(time.time())}",
                            "metadata": {
                                "agent_id": task_result.get("agent_id", validated_payload["agent_type"]),
                                "confidence": 0.9,
                                "assignment_reason": "orchestrator_selection"
                            },
                            "result": task_result.get("result", {}),
                            "status": task_result.get("status", "completed")
                        }
                    })()

                except Exception as e:
                    st.warning(f"Orchestrator execution failed: {e}")
                    # Fallback to direct API call
                    task_response = self._make_resilient_request(
                        "http://localhost:8002/api/v2/agents/execute",
                        method="POST",
                        json_data={
                            "task": validated_payload["task_description"],
                            "agent_id": validated_payload["agent_type"] if validated_payload["agent_type"] != "auto" else None,
                            "context": {
                                "deployment": validated_payload["deployment_config"],
                                "model": validated_payload["ai_model"],
                                "reference_task_id": validated_payload["reference_task_id"]
                            },
                            "timeout": 60
                        },
                        headers=headers,
                        timeout=30
                    )
            else:
                # Fallback to direct API call if orchestrator not available
                task_response = self._make_resilient_request(
                    "http://localhost:8002/api/v2/agents/execute",
                    method="POST",
                    json_data={
                        "task": validated_payload["task_description"],
                        "agent_id": validated_payload["agent_type"] if validated_payload["agent_type"] != "auto" else None,
                        "context": {
                            "deployment": validated_payload["deployment_config"],
                            "model": validated_payload["ai_model"],
                            "reference_task_id": validated_payload["reference_task_id"]
                        },
                        "timeout": 60
                    },
                    headers=headers,
                    timeout=30
                )

            submission_latency = time.time() - submission_start_time

            if task_response and task_response.status_code == 200:
                task_data = task_response.json()
                task_id = task_data.get("task_id")

                # Determine assignment details for telemetry
                actual_agent = task_data.get("metadata", {}).get("agent_id", agent_type)
                confidence = task_data.get("metadata", {}).get("confidence", 0.0)
                assignment_reason = task_data.get("metadata", {}).get("assignment_reason", "auto")

                # Emit telemetry for successful task assignment
                self._emit_telemetry("task_assigned", {
                    "task_id": task_id,
                    "requested_agent": agent_type,
                    "assigned_agent": actual_agent,
                    "confidence": confidence,
                    "assignment_reason": assignment_reason,
                    "priority": priority,
                    "ai_model": ai_model,
                    "submission_latency": submission_latency,
                    "has_deployment": deployment_config is not None,
                    "has_reference": bool(reference_task_id)
                })

                return task_data
            else:
                # Emit telemetry for task submission failure
                self._emit_telemetry("task_submission_failed", {
                    "priority": priority,
                    "agent_type": agent_type,
                    "ai_model": ai_model,
                    "submission_latency": submission_latency,
                    "http_status": task_response.status_code if task_response else "TIMEOUT",
                    "task_description_length": len(task_description)
                })

                error_msg = f"Task submission failed"
                if task_response:
                    error_msg += f" (HTTP {task_response.status_code})"
                st.error(error_msg)
                return None

        except Exception as e:
            st.error(f"Error assigning task: {str(e)}")
            return None

    def send_feedback(self, task_id: str, agent_id: str, success: bool, quality_score: int):
        """Send feedback for a completed task"""
        try:
            # Validate feedback request
            feedback_payload = {
                "task_id": task_id,
                "agent_id": agent_id,
                "success": success,
                "quality_score": quality_score
            }
            validated_feedback = self._validate_request_payload(feedback_payload, TaskFeedbackRequest)

            feedback_response = requests.post(
                "http://localhost:8025/task-feedback",
                json=validated_feedback,
                timeout=5
            )
            if feedback_response.status_code == 200:
                st.success("✅ Feedback recorded - agent will improve!")
            else:
                st.warning("Feedback sent but not processed")
        except Exception as e:
            st.error(f"Error sending feedback: {str(e)}")

    def chat_with_agent(self, message: str, agent_id: str, context: Optional[Dict] = None):
        """Send a chat message to an agent"""
        try:
            headers = {
                "Authorization": f"Bearer {st.session_state.get('api_token', '')}",
                "Content-Type": "application/json"
            }

            # Validate chat request
            chat_payload = {
                "message": message,
                "agent_id": agent_id,
                "context": context or {}
            }
            validated_chat = self._validate_request_payload(chat_payload, ChatRequest)

            response = requests.post(
                f"http://localhost:8002/agents/{validated_chat['agent_id']}/chat",
                json=validated_chat,
                headers=headers,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                # Validate response
                validated_response = self._validate_response_payload(result, ChatResponse)
                st.markdown(f"**{validated_response['agent_id'].title()}:** {validated_response.get('response', 'No response')}")
            else:
                st.error(f"Chat failed: {response.text}")

        except Exception as e:
            st.error(f"Error in chat: {str(e)}")

    def get_task_history(self):
        """Get task history from database"""
        # Direct PostgreSQL query for task history
        task_history = []

        # Check if psycopg2 is available at module level
        try:
            import psycopg2
            PSYCOPG2_AVAILABLE = True
            st.success("✅ PostgreSQL driver loaded successfully")
        except ImportError as e:
            PSYCOPG2_AVAILABLE = False
            st.error(f"❌ Import error: {e}")
        except Exception as e:
            PSYCOPG2_AVAILABLE = False
            st.error(f"❌ Unexpected error: {e}")

        if PSYCOPG2_AVAILABLE:
            try:
                from psycopg2.extras import RealDictCursor
                import os

                conn = psycopg2.connect(
                    host="localhost",
                    database="agent_lightning_memory",
                    user=os.getenv('USER'),
                    port=5432
                )

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id as task_id, agent_id, task_description, result, execution_time, tokens_used, cost, created_at
                        FROM task_history
                        ORDER BY created_at DESC
                        LIMIT 20
                    """)
                    task_history = [dict(row) for row in cur.fetchall()]
                conn.close()

                if task_history:
                    st.success(f"Found {len(task_history)} completed tasks")
                    for task in task_history:
                        timestamp = task['created_at'].strftime("%H:%M:%S") if task['created_at'] else "Unknown"
                        task_desc = task['task_description'][:50] + "..." if len(task['task_description']) > 50 else task['task_description']

                        with st.expander(f"{timestamp} - {task_desc} (Agent: {task['agent_id']})"):
                            # Show task ID for copying
                            st.code(f"Task ID: {task['task_id']}", language="text")

                            result = task.get('result', {})
                            if isinstance(result, dict):
                                st.write(result.get('response', 'No response'))
                            else:
                                st.write(str(result))
                            if task.get('execution_time'):
                                st.caption(f"⏱️ Execution time: {task['execution_time']:.2f}s")
                            if task.get('tokens_used'):
                                st.caption(f"🔤 Tokens used: {task['tokens_used']}")
                            if task.get('cost'):
                                st.caption(f"💰 Cost: ${task['cost']:.4f}")
                else:
                    st.info("No task history found")
            except Exception as e:
                st.error(f"Database connection error: {e}")
        else:
            st.warning("⚠️ PostgreSQL driver not installed. Install with: pip install psycopg2-binary")
            st.info("Task history requires PostgreSQL connection")