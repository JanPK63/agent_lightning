#!/usr/bin/env python3
"""
Mock Agent service for local testing.

Run multiple instances with different --agent-id and --port to simulate available agents.
Each instance exposes:
  GET  /health         -> returns simple JSON health
  POST /execute        -> accepts { "task_id", "task_description", "context" } and returns result

Example:
  python services/mock_agent.py --agent-id data_analyst --port 9002
"""

import argparse
import json
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()


def make_agent_state(agent_id: str) -> Dict[str, Any]:
    return {
        "agent_id": agent_id,
        "name": f"Mock-{agent_id}",
        "start_time": time.time(),
        "cwd": os.getcwd()
    }


AGENT_STATE: Dict[str, Any] = {}


@app.get("/health")
async def health():
    """Return agent health and basic metadata"""
    return {
        "status": "healthy",
        "agent_id": AGENT_STATE.get("agent_id"),
        "name": AGENT_STATE.get("name"),
        "cwd": AGENT_STATE.get("cwd"),
        "uptime_seconds": time.time() - AGENT_STATE.get("start_time", time.time())
    }


@app.post("/execute")
async def execute(request: Request):
    """
    Execute a task. Expected JSON body:
      { "task_id": "...", "task_description": "...", "context": {...} }
    This mock will:
     - simulate a brief execution delay
     - if description contains 'Security_Agent_New' and working_directory provided, check for directory existence
    """
    payload = await request.json()
    task_id = payload.get("task_id")
    desc = payload.get("task_description", "")
    context = payload.get("context", {}) or {}

    # Simulate some work
    time.sleep(0.5)

    # Check for Security_Agent_New directory if present
    working_dir = context.get("working_directory") or context.get("execution_environment", {}).get("working_directory")
    if "Security_Agent_New" in desc:
        if working_dir and os.path.exists(working_dir):
            analysis = {
                "status": "completed",
                "message": f"Mock analysis of Security_Agent_New at {working_dir} completed",
                "task_id": task_id
            }
            return analysis
        else:
            return {"status": "failed", "error": f"Directory not found: {working_dir}", "task_id": task_id}

    # Default success
    return {
        "status": "completed",
        "message": f"Mock agent {AGENT_STATE.get('agent_id')} executed task",
        "task_id": task_id,
        "context": context
    }


def run_mock_agent(agent_id: str, port: int, host: str = "0.0.0.0"):
    global AGENT_STATE
    AGENT_STATE = make_agent_state(agent_id)
    print(f"[mock_agent] Starting mock agent {agent_id} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", type=str, required=True, help="Agent id to simulate (e.g. data_analyst)")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on (e.g. 9002)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    run_mock_agent(args.agent_id, args.port, args.host)
