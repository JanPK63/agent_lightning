"""
Spec Workflow System Prompt Loader Agent

This agent is called FIRST when a user wants to start a spec process.
It returns the file path to the appropriate workflow system prompt.

Inputs: type of spec workflow requested
Outputs: file path to the appropriate workflow prompt file
"""

import os
from pathlib import Path
from typing import Dict, Optional


class SpecWorkflowLoader:
    """
    Agent to load appropriate spec workflow system prompts.
    
    This is the entry point for all spec-driven development workflows.
    It determines which workflow type the user needs and returns the
    corresponding system prompt file path.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the workflow loader.
        
        Args:
            base_path: Base path to the spec_driven_workflow directory.
                      If None, uses the directory containing this file.
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent
        self.base_path = Path(base_path)
        self.workflows_path = self.base_path / "workflows"
        
        # Workflow type mappings
        self.workflow_types = {
            "new_spec": "main_spec_workflow.md",
            "requirements": "spec_requirements_workflow.md", 
            "design": "spec_design_workflow.md",
            "tasks": "spec_tasks_workflow.md",
            "judge": "spec_judge_workflow.md",
            "implementation": "spec_implementation_workflow.md",
            "test": "spec_test_workflow.md",
            "full_workflow": "main_spec_workflow.md"
        }
        
    def get_workflow_prompt(self, workflow_type: str) -> Dict[str, str]:
        """
        Get the system prompt file path for the specified workflow type.
        
        Args:
            workflow_type: Type of workflow requested. Can be:
                - "new_spec": Start a completely new specification
                - "requirements": Work on requirements gathering
                - "design": Work on technical design
                - "tasks": Break down implementation tasks
                - "judge": Review and validate specs
                - "implementation": Execute coding tasks
                - "test": Create test documentation and code
                - "full_workflow": Complete end-to-end workflow
                
        Returns:
            Dictionary with:
                - "success": Boolean indicating if prompt was found
                - "prompt_path": Absolute path to the prompt file
                - "workflow_type": The normalized workflow type
                - "description": Description of the workflow
                - "error": Error message if prompt not found
        """
        # Normalize workflow type
        workflow_type = workflow_type.lower().replace("-", "_").replace(" ", "_")
        
        if workflow_type not in self.workflow_types:
            return {
                "success": False,
                "error": f"Unknown workflow type: {workflow_type}",
                "available_types": list(self.workflow_types.keys()),
                "prompt_path": None,
                "workflow_type": workflow_type,
                "description": None
            }
            
        prompt_filename = self.workflow_types[workflow_type]
        prompt_path = self.workflows_path / prompt_filename
        
        # Check if prompt file exists
        if not prompt_path.exists():
            return {
                "success": False,
                "error": f"Workflow prompt file not found: {prompt_path}",
                "prompt_path": str(prompt_path),
                "workflow_type": workflow_type,
                "description": None
            }
            
        # Return success with full information
        return {
            "success": True,
            "prompt_path": str(prompt_path.absolute()),
            "workflow_type": workflow_type,
            "description": self._get_workflow_description(workflow_type),
            "error": None
        }
        
    def _get_workflow_description(self, workflow_type: str) -> str:
        """Get description for the workflow type."""
        descriptions = {
            "new_spec": "Complete specification workflow from idea to implementation plan",
            "requirements": "Gather and document user requirements and acceptance criteria",
            "design": "Create technical design and architecture from requirements",
            "tasks": "Break down design into concrete implementation tasks",
            "judge": "Review and validate specification documents for quality",
            "implementation": "Execute specific coding tasks following TDD",
            "test": "Create comprehensive test documentation and code",
            "full_workflow": "Complete end-to-end specification-driven development"
        }
        return descriptions.get(workflow_type, "Unknown workflow")
        
    def list_available_workflows(self) -> Dict[str, str]:
        """
        List all available workflow types and their descriptions.
        
        Returns:
            Dictionary mapping workflow types to descriptions
        """
        return {
            workflow_type: self._get_workflow_description(workflow_type)
            for workflow_type in self.workflow_types.keys()
        }
        
    def validate_workflow_setup(self) -> Dict[str, any]:
        """
        Validate that all required workflow files exist.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "missing_files": [],
            "existing_files": [],
            "base_path": str(self.base_path),
            "workflows_path": str(self.workflows_path)
        }
        
        for workflow_type, filename in self.workflow_types.items():
            file_path = self.workflows_path / filename
            if file_path.exists():
                results["existing_files"].append({
                    "workflow": workflow_type,
                    "file": filename,
                    "path": str(file_path)
                })
            else:
                results["missing_files"].append({
                    "workflow": workflow_type,
                    "file": filename,
                    "expected_path": str(file_path)
                })
                results["valid"] = False
                
        return results


def load_spec_workflow(workflow_type: str, base_path: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to load a spec workflow prompt.
    
    Args:
        workflow_type: Type of workflow requested
        base_path: Optional base path to spec_driven_workflow directory
        
    Returns:
        Dictionary with workflow prompt information
    """
    loader = SpecWorkflowLoader(base_path)
    return loader.get_workflow_prompt(workflow_type)


# CLI interface for testing
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python spec_workflow_loader.py <workflow_type>")
        print("Available workflow types:")
        loader = SpecWorkflowLoader()
        for wf_type, desc in loader.list_available_workflows().items():
            print(f"  {wf_type}: {desc}")
        sys.exit(1)
        
    workflow_type = sys.argv[1]
    result = load_spec_workflow(workflow_type)
    print(json.dumps(result, indent=2))