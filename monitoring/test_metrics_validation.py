#!/usr/bin/env python3
"""
Agent Lightning Metrics Validation Test

This script validates that Prometheus metrics are being collected correctly
from all services in the Agent Lightning system.

Usage:
    python test_metrics_validation.py

Requirements:
    - All services must be running with metrics enabled
    - Prometheus server must be running and configured
    - requests library for HTTP calls
"""

import time
import requests
import json
import subprocess
import sys
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsValidator:
    """Validates Prometheus metrics collection from Agent Lightning services"""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.services = {
            "agent-coordinator": "http://localhost:8001",
            "agent-designer": "http://localhost:8002",
            "ai-model-service": "http://localhost:8003",
            "langchain-integration": "http://localhost:8004",
            "memory-manager": "http://localhost:8005",
            "knowledge-manager": "http://localhost:8006",
            "workflow-engine": "http://localhost:8007",
            "rl-orchestrator": "http://localhost:8008",
            "monitoring-dashboard": "http://localhost:8009",
            "performance-metrics": "http://localhost:8010",
            "websocket-service": "http://localhost:8011",
            "event-replay-debugger": "http://localhost:8012"
        }

    def query_prometheus(self, query: str) -> Dict:
        """Query Prometheus and return results"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return {"status": "error", "error": str(e)}

    def check_service_health(self, service_name: str, service_url: str) -> bool:
        """Check if a service is healthy and responding"""
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate_service_traffic(self, service_name: str, service_url: str) -> bool:
        """Generate some traffic to a service to create metrics"""
        try:
            # Try health endpoint first
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Generated health check traffic for {service_name}")
                return True

            # Try root endpoint
            response = requests.get(service_url, timeout=5)
            if response.status_code in [200, 404]:  # 404 is ok, means service is responding
                logger.info(f"Generated root endpoint traffic for {service_name}")
                return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to generate traffic for {service_name}: {e}")

        return False

    def validate_service_metrics(self, service_name: str) -> Dict:
        """Validate metrics for a specific service"""
        results = {
            "service": service_name,
            "metrics_found": False,
            "request_count": 0,
            "error_count": 0,
            "up_status": False,
            "issues": []
        }

        # Check if service is up
        service_url = self.services.get(service_name)
        if service_url:
            results["up_status"] = self.check_service_health(service_name, service_url)

        # Query for request metrics
        request_query = f'agent_lightning_requests_total{{job="{service_name}"}}'
        request_data = self.query_prometheus(request_query)

        if request_data.get("status") == "success" and request_data.get("data", {}).get("result"):
            results["metrics_found"] = True
            # Sum all request counts
            total_requests = 0
            for result in request_data["data"]["result"]:
                if "value" in result and len(result["value"]) > 1:
                    total_requests += float(result["value"][1])
            results["request_count"] = total_requests
        else:
            results["issues"].append(f"No request metrics found for {service_name}")

        # Query for error metrics
        error_query = f'agent_lightning_requests_total{{job="{service_name}", status=~"5.."}}'
        error_data = self.query_prometheus(error_query)

        if error_data.get("status") == "success" and error_data.get("data", {}).get("result"):
            total_errors = 0
            for result in error_data["data"]["result"]:
                if "value" in result and len(result["value"]) > 1:
                    total_errors += float(result["value"][1])
            results["error_count"] = total_errors

        # Check for up metric
        up_query = f'up{{job="{service_name}"}}'
        up_data = self.query_prometheus(up_query)

        if up_data.get("status") == "success" and up_data.get("data", {}).get("result"):
            for result in up_data["data"]["result"]:
                if "value" in result and len(result["value"]) > 1:
                    up_value = float(result["value"][1])
                    if up_value == 1.0:
                        results["up_status"] = True
                    elif up_value == 0.0:
                        results["issues"].append(f"Service {service_name} is reported as down by Prometheus")

        return results

    def validate_alerting_rules(self) -> Dict:
        """Validate that alerting rules are working"""
        results = {
            "alerts_configured": False,
            "active_alerts": [],
            "issues": []
        }

        try:
            # Query for active alerts
            response = requests.get(f"{self.prometheus_url}/api/v1/alerts")
            response.raise_for_status()
            alert_data = response.json()

            if alert_data.get("status") == "success":
                results["alerts_configured"] = True
                alerts = alert_data.get("data", {}).get("alerts", [])
                results["active_alerts"] = [alert["labels"]["alertname"] for alert in alerts if alert["state"] == "firing"]

        except requests.exceptions.RequestException as e:
            results["issues"].append(f"Failed to query alerts: {e}")

        return results

    def run_validation(self) -> Dict:
        """Run complete metrics validation"""
        logger.info("Starting Agent Lightning Metrics Validation")
        logger.info("=" * 50)

        results = {
            "timestamp": time.time(),
            "services_tested": 0,
            "services_with_metrics": 0,
            "services_healthy": 0,
            "total_requests": 0,
            "total_errors": 0,
            "service_results": [],
            "alerting_results": {},
            "overall_status": "unknown",
            "recommendations": []
        }

        # Test each service
        for service_name, service_url in self.services.items():
            logger.info(f"Testing service: {service_name}")

            # Generate some traffic first
            self.generate_service_traffic(service_name, service_url)

            # Wait a moment for metrics to be scraped
            time.sleep(2)

            # Validate metrics
            service_result = self.validate_service_metrics(service_name)
            results["service_results"].append(service_result)
            results["services_tested"] += 1

            if service_result["metrics_found"]:
                results["services_with_metrics"] += 1
                results["total_requests"] += service_result["request_count"]
                results["total_errors"] += service_result["error_count"]

            if service_result["up_status"]:
                results["services_healthy"] += 1

            # Log results
            status = "✓" if service_result["metrics_found"] and service_result["up_status"] else "✗"
            logger.info(f"  {status} {service_name}: metrics={service_result['metrics_found']}, healthy={service_result['up_status']}, requests={service_result['request_count']}")

            if service_result["issues"]:
                for issue in service_result["issues"]:
                    logger.warning(f"    Issue: {issue}")

        # Test alerting
        logger.info("Testing alerting rules...")
        results["alerting_results"] = self.validate_alerting_rules()

        # Calculate overall status
        if results["services_with_metrics"] == results["services_tested"] and results["services_healthy"] >= results["services_tested"] * 0.8:
            results["overall_status"] = "success"
        elif results["services_with_metrics"] > 0:
            results["overall_status"] = "partial"
        else:
            results["overall_status"] = "failed"

        # Generate recommendations
        if results["services_with_metrics"] < results["services_tested"]:
            results["recommendations"].append("Some services are not exposing metrics. Check service logs and ensure metrics endpoints are accessible.")

        if results["services_healthy"] < results["services_tested"]:
            results["recommendations"].append("Some services are not healthy. Check service status and restart if necessary.")

        if not results["alerting_results"].get("alerts_configured"):
            results["recommendations"].append("Alerting rules are not configured or accessible. Check Prometheus configuration.")

        # Summary
        logger.info("=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Services tested: {results['services_tested']}")
        logger.info(f"Services with metrics: {results['services_with_metrics']}")
        logger.info(f"Healthy services: {results['services_healthy']}")
        logger.info(f"Total requests recorded: {results['total_requests']}")
        logger.info(f"Total errors recorded: {results['total_errors']}")
        logger.info(f"Overall status: {results['overall_status'].upper()}")

        if results["alerting_results"].get("active_alerts"):
            logger.info(f"Active alerts: {', '.join(results['alerting_results']['active_alerts'])}")

        if results["recommendations"]:
            logger.info("RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                logger.info(f"  - {rec}")

        return results

def main():
    """Main entry point"""
    validator = MetricsValidator()

    # Check if Prometheus is accessible
    try:
        response = requests.get(f"{validator.prometheus_url}/api/v1/status/buildinfo", timeout=5)
        response.raise_for_status()
        logger.info("Prometheus server is accessible")
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to Prometheus at {validator.prometheus_url}: {e}")
        logger.error("Please ensure Prometheus is running and accessible")
        sys.exit(1)

    # Run validation
    results = validator.run_validation()

    # Save results to file
    output_file = "monitoring/metrics_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")

    # Exit with appropriate code
    if results["overall_status"] == "success":
        logger.info("✅ Metrics validation PASSED")
        sys.exit(0)
    elif results["overall_status"] == "partial":
        logger.warning("⚠️  Metrics validation PARTIAL - some issues found")
        sys.exit(1)
    else:
        logger.error("❌ Metrics validation FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()