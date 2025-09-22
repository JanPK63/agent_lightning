#!/usr/bin/env python3
"""
Patch script to add force_execute support to RL Orchestrator
"""

import re

def patch_rl_orchestrator():
    """Add force_execute bypass logic to RL Orchestrator"""

    # Read the file
    with open('services/rl_orchestrator_improved.py', 'r') as f:
        content = f.read()

    # Find the confidence check and add force_execute bypass
    # Look for the pattern where confidence is checked against 0.6
    pattern = r'if best_agent\[1\] < 0\.6:'
    replacement = '''if best_agent[1] < 0.6 and not request.force_execute:
        # Allow force_execute to bypass confidence check
        logger.warning(f"Low confidence match: {best_agent[0]} for task: {task_description[:100]}")
        if not request.force_execute:
            # Only reject if not force_execute'''

    # Apply the patch
    if pattern in content:
        # Find the confidence check block
        confidence_check_pattern = r'(if best_agent\[1\] < 0\.6:.*?)(?=if best_agent\[1\] < 0\.6:)'
        match = re.search(confidence_check_pattern, content, re.DOTALL)
        if match:
            # Replace with force_execute check
            new_content = content.replace(
                'if best_agent[1] < 0.6:',
                'if best_agent[1] < 0.6 and not request.force_execute:'
            )
            # Add logging for force_execute
            new_content = new_content.replace(
                'logger.warning(f"Low confidence match: {best_agent[0]} for task: {task_description[:100]}")',
                '''if request.force_execute:
    logger.info(f"Force executing task despite low confidence {best_agent[1]:.2f}: {best_agent[0]} for task: {task_description[:100]}")
else:
    logger.warning(f"Low confidence match: {best_agent[0]} for task: {task_description[:100]}")'''
            )

            # Write back
            with open('services/rl_orchestrator_improved.py', 'w') as f:
                f.write(new_content)

            print("✅ Successfully patched RL Orchestrator with force_execute support")
            return True

    print("❌ Could not find confidence check pattern to patch")
    return False

if __name__ == "__main__":
    patch_rl_orchestrator()