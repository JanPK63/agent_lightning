"""
Simple Agent Workflow Enhancement
Minimal multi-agent coordination
"""

from langchain_agent_wrapper import LangChainAgentManager
import asyncio


class SimpleWorkflow:
    def __init__(self):
        self.agents = LangChainAgentManager()
    
    def execute_task_chain(self, tasks, session_id="workflow"):
        """Execute tasks in sequence, passing results between agents"""
        results = []
        context = ""
        
        for i, (task_desc, agent_name) in enumerate(tasks):
            agent = self.agents.get_agent(agent_name)
            if not agent:
                agent = list(self.agents.agents.values())[0]  # Use first available
            
            # Add context from previous tasks
            full_prompt = f"{task_desc}\n\nPrevious context: {context}" if context else task_desc
            
            result = agent.invoke(full_prompt, f"{session_id}_{i}")
            results.append({"task": task_desc, "agent": agent_name, "result": result})
            
            # Update context for next task
            context = result[:200] + "..." if len(result) > 200 else result
        
        return results


# Test
if __name__ == "__main__":
    workflow = SimpleWorkflow()
    
    tasks = [
        ("Calculate 2+2 using Python", "full_stack_developer"),
        ("Explain the calculation result", "writer")
    ]
    
    results = workflow.execute_task_chain(tasks)
    for r in results:
        print(f"Task: {r['task']}")
        print(f"Result: {r['result'][:100]}...")
        print("---")