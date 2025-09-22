#!/usr/bin/env python3
"""
Small local test to trigger and monitor task execution in the RL Orchestrator.

Usage:
  - Run against an existing task id:
      python tests/test_local_execution.py --task-id 4392dc02-83eb-4d70-a7ae-6429f02c7d1a

  - Create a new test task and execute it:
      python tests/test_local_execution.py --create-new --description "Local test: analyze Security_Agent_New" \
           --working-dir "/Users/jankootstra/Security/Security/Security_Agent_New"

Default orchestrator URL: http://localhost:8025
"""

import argparse
import json
import sys
import time
import uuid
from typing import Optional

import requests

DEFAULT_URL = "http://localhost:8025"


def post_assign_task(base_url: str, task_id: str, description: str, working_dir: Optional[str] = None) -> dict:
    payload = {
        "task_id": task_id,
        "description": description,
        "priority": 5,
        "metadata": {},
        "idempotency_key": f"local-test-{task_id}",
        # execution-related fields if supported by the orchestrator
        "target_environment": "local",
        "working_directory": working_dir,
        "search_scope": "local",
        "platform": "macos",
        "environment_vars": {},
        "execution_mode": "standard"
    }
    resp = requests.post(f"{base_url}/assign-task", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def post_execute_now(base_url: str, task_id: str) -> dict:
    payload = {"task_id": task_id}
    resp = requests.post(f"{base_url}/execute-now", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_task_status(base_url: str, task_id: str) -> dict:
    resp = requests.get(f"{base_url}/tasks/{task_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def poll_task_until_done(base_url: str, task_id: str, timeout: int = 60, interval: float = 2.0) -> dict:
    start = time.time()
    while True:
        try:
            status = get_task_status(base_url, task_id)
        except requests.HTTPError as e:
            print(f"[poll] HTTP error fetching task status: {e}")
            raise
        except Exception as e:
            print(f"[poll] Error fetching task status: {e}")
            raise

        state = status.get("status", "").lower()
        now = time.time()
        elapsed = now - start

        print(f"[poll] {task_id} status={state} (elapsed: {elapsed:.1f}s)")

        if state in ("completed", "failed"):
            return status

        if elapsed > timeout:
            raise TimeoutError(f"Task {task_id} did not finish within {timeout} seconds")

        time.sleep(interval)


def pretty_print_result(task_status: dict):
    print("\n" + "=" * 80)
    print("FINAL TASK REPORT")
    print("=" * 80)
    print(f"Task ID: {task_status.get('task_id') or task_status.get('id')}")
    print(f"Assigned Agent: {task_status.get('assigned_agent') or task_status.get('agent_id')}")
    print(f"Status: {task_status.get('status')}")
    print(f"Confidence: {task_status.get('confidence')}")
    print(f"Validation Passed: {task_status.get('validation_passed')}")
    print(f"Warnings: {task_status.get('warnings')}")
    print(f"Created: {task_status.get('created_at')}")
    print(f"Started: {task_status.get('started_at') or task_status.get('execution_started')}")
    print(f"Ended: {task_status.get('completed_at') or task_status.get('execution_end')}")
    print("\nResult payload (truncated):")
    result = task_status.get("result") or {}
    try:
        print(json.dumps(result, indent=2)[:2000])
    except Exception:
        print(str(result)[:2000])

    print("\nDebug Info (if any):")
    debug_info = task_status.get("debug_info") or {}
    if debug_info:
        print(json.dumps(debug_info, indent=2))
    else:
        print("No debug_info present")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Local execution test for RL Orchestrator")
    parser.add_argument("--base-url", type=str, default=DEFAULT_URL, help="Base URL of the orchestrator")
    parser.add_argument("--task-id", type=str, help="Existing task id to execute")
    parser.add_argument("--create-new", action="store_true", help="Create a new test task before executing")
    parser.add_argument("--description", type=str, default="Local execution test", help="Description for new task")
    parser.add_argument("--working-dir", type=str, default=None, help="Working directory to request for execution")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for polling task completion")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    try:
        if args.create_new:
            task_id = str(uuid.uuid4())
            print(f"[main] Creating test task {task_id} ...")
            assign_resp = post_assign_task(base_url, task_id, args.description, args.working_dir)
            print(f"[main] Assign response: {assign_resp}")
        elif args.task_id:
            task_id = args.task_id
            print(f"[main] Using existing task id: {task_id}")
        else:
            print("[main] You must provide --task-id or --create-new")
            sys.exit(2)

        print(f"[main] Requesting immediate execution of task {task_id} ...")
        exec_resp = post_execute_now(base_url, task_id)
        print(f"[main] execute-now response: {exec_resp}")

        print(f"[main] Polling task {task_id} until done (timeout {args.timeout}s) ...")
        final_status = poll_task_until_done(base_url, task_id, timeout=args.timeout)
        pretty_print_result(final_status)

    except Exception as e:
        print(f"[main] ERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
