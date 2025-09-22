#!/usr/bin/env python3
"""
Test helper to register multiple agents with the orchestrator and then probe them.

Usage:
  - Provide a JSON file with array of agents:
      [
        {"agent_id":"data_analyst", "health_url":"http://host:9002/health", "execute_url":"http://host:9002/execute"},
        ...
      ]
  - Run:
      python tests/test_agents_register_and_probe.py --base-url http://localhost:8025 --agents-file ./agents_list.json

If no file provided, the script will prompt to register a small built-in example set.
"""

import argparse
import json
import requests
import sys
from typing import List, Dict

DEFAULT_URL = "http://localhost:8025"


def register_agent(base_url: str, agent: Dict):
    resp = requests.post(f"{base_url}/agents/register", json=agent, timeout=10)
    resp.raise_for_status()
    return resp.json()


def list_agents(base_url: str):
    resp = requests.get(f"{base_url}/agents", timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=DEFAULT_URL)
    parser.add_argument("--agents-file", type=str, help="JSON file with agents array")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.agents_file:
        try:
            with open(args.agents_file, "r") as f:
                agents = json.load(f)
                if not isinstance(agents, list):
                    print("Agents file must contain a JSON array")
                    sys.exit(2)
        except Exception as e:
            print(f"Failed to read agents file: {e}")
            sys.exit(2)
    else:
        # Example placeholder entries - replace with your actual 32 agent definitions
        agents = [
            {"agent_id": f"agent_{i+1}", "health_url": f"http://localhost:91{10+i}/health", "execute_url": f"http://localhost:91{10+i}/execute"}
            for i in range(4)  # change or replace with your 32 entries
        ]
        print("No agents file provided; using small example set. Replace with real agent definitions.")

    print(f"[main] Registering {len(agents)} agents with orchestrator at {base_url}/agents/register ...")
    for a in agents:
        try:
            r = register_agent(base_url, a)
            print(f"Registered {a['agent_id']}: {r}")
        except Exception as e:
            print(f"Failed to register {a.get('agent_id')}: {e}")

    print("\n[main] Probing agents via /agents ...")
    try:
        report = list_agents(base_url)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Failed to list/probe agents: {e}")


if __name__ == "__main__":
    main()
