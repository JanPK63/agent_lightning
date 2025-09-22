#!/usr/bin/env python3
"""
Test script to discover agents and probe their health via the RL Orchestrator.

Usage:
  python tests/test_agents_discovery.py --base-url http://localhost:8025
"""

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests

DEFAULT_URL = "http://localhost:8025"


def list_agents(base_url: str) -> Dict[str, Any]:
    resp = requests.get(f"{base_url}/agents", timeout=10)
    resp.raise_for_status()
    return resp.json()


def ping_agent(base_url: str, agent_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{base_url}/agents/{agent_id}/ping", timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Agents discovery test")
    parser.add_argument("--base-url", type=str, default=DEFAULT_URL, help="Orchestrator base URL")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    try:
        print(f"[main] Querying orchestrator at {base_url}/agents ...")
        data = list_agents(base_url)
        agents = data.get("agents", [])
        print(f"[main] Found {len(agents)} agents")
        for a in agents:
            print("-" * 60)
            print(f"ID: {a.get('agent_id')}")
            print(f"Name: {a.get('name')}")
            print(f"Capabilities: {a.get('capabilities')}")
            print(f"Base confidence: {a.get('base_confidence')}")
            print(f"Health: {a.get('health')}")
            print(f"Health details: {a.get('health_details')}")
        print("-" * 60)

        # Ping any agents that are not OK
        for a in agents:
            if a.get("health") != "ok" and a.get("health_url"):
                print(f"[main] Re-pinging {a.get('agent_id')} via direct endpoint...")
                ping = ping_agent(base_url, a.get("agent_id"))
                print(json.dumps(ping, indent=2))

    except Exception as e:
        print(f"[main] ERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
