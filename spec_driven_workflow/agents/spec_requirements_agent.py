"""
Spec Requirements Agent

This agent creates and refines specification requirements documents.
It focuses on WHAT users need and WHY, avoiding HOW implementation details.

Key responsibilities:
- Gather user stories and acceptance criteria
- Define clear scope boundaries
- Identify stakeholders and personas
- Create measurable success criteria
- Mark ambiguities for clarification
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class SpecRequirementsAgent:
    """
    Agent for creating and refining specification requirements documents.
    
    This agent follows the SDD principle of executable specifications by
    creating precise, complete, and unambiguous requirements that can
    drive technical implementation.
    """
    
    def __init__(self, project_path: Optional[str] = None):
        """
        Initialize the requirements agent.
        
        Args:
            project_path: Path to the project root. If None, uses current directory.
        """
        if project_path is None:
            project_path = os.getcwd()
        self.project_path = Path(project_path)
        self.specs_path = self.project_path / "specs"
        self.templates_path = Path(__file__).parent.parent / "templates"
        
    def create_requirements_document(self, 
                                   spec_name: str, 
                                   user_input: str,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new requirements document from user input.
        
        Args:
            spec_name: Name for the specification (used for directory naming)
            user_input: Raw user description of what they want
            context: Optional context including project info, constraints, etc.
            
        Returns:
            Dictionary with creation results and next steps
        """
        # Normalize spec name for directory
        normalized_name = self._normalize_spec_name(spec_name)
        spec_dir = self.specs_path / normalized_name
        
        # Create spec directory if it doesn't exist
        spec_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze user input to extract requirements
        analysis = self._analyze_user_input(user_input, context)
        
        # Generate requirements document
        requirements_doc = self._generate_requirements_document(
            spec_name, user_input, analysis, context
        )
        
        # Write requirements document
        requirements_path = spec_dir / "requirements.md"
        with open(requirements_path, 'w') as f:
            f.write(requirements_doc)
            
        # Create metadata file
        metadata = {
            "spec_name": spec_name,
            "normalized_name": normalized_name,
            "created_at": datetime.now().isoformat(),
            "status": "requirements_draft",
            "phase": "requirements_gathering",
            "user_input": user_input,
            "analysis": analysis,
            "files": {
                "requirements": str(requirements_path)
            }
        }
        
        metadata_path = spec_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "success": True,
            "spec_directory": str(spec_dir),
            "requirements_path": str(requirements_path),
            "metadata_path": str(metadata_path),
            "next_steps": self._get_next_steps(analysis),
            "clarifications_needed": analysis.get("clarifications_needed", []),
            "metadata": metadata
        }
        
    def refine_requirements(self, 
                           spec_directory: str, 
                           clarifications: Dict[str, str]) -> Dict[str, Any]:
        """
        Refine an existing requirements document with clarifications.
        
        Args:
            spec_directory: Path to the spec directory
            clarifications: Dictionary of clarification responses
            
        Returns:
            Dictionary with refinement results
        """
        spec_dir = Path(spec_directory)
        
        # Load existing metadata
        metadata_path = spec_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Load existing requirements
        requirements_path = spec_dir / "requirements.md"
        with open(requirements_path, 'r') as f:
            current_requirements = f.read()
            
        # Apply clarifications
        refined_requirements = self._apply_clarifications(
            current_requirements, clarifications
        )
        
        # Write updated requirements
        with open(requirements_path, 'w') as f:
            f.write(refined_requirements)
            
        # Update metadata
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["clarifications_applied"] = clarifications
        metadata["status"] = "requirements_refined"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "success": True,
            "requirements_path": str(requirements_path),
            "metadata": metadata,
            "ready_for_judge": self._check_requirements_completeness(refined_requirements)
        }
        
    def validate_requirements(self, spec_directory: str) -> Dict[str, Any]:
        """
        Validate requirements document for completeness and clarity.
        
        Args:
            spec_directory: Path to the spec directory
            
        Returns:
            Validation results with issues and recommendations
        """
        spec_dir = Path(spec_directory)
        requirements_path = spec_dir / "requirements.md"
        
        if not requirements_path.exists():
            return {
                "success": False,
                "error": "Requirements document not found"
            }
            
        with open(requirements_path, 'r') as f:
            requirements_content = f.read()
            
        validation = self._validate_requirements_content(requirements_content)
        
        return {
            "success": True,
            "validation": validation,
            "ready_for_design": validation["is_complete"],
            "issues": validation.get("issues", []),
            "recommendations": validation.get("recommendations", [])
        }
        
    def _normalize_spec_name(self, spec_name: str) -> str:
        """Normalize spec name for directory naming."""
        import re
        # Convert to lowercase, replace spaces and special chars with hyphens
        normalized = re.sub(r'[^a-zA-Z0-9\-]', '-', spec_name.lower())
        # Remove multiple consecutive hyphens
        normalized = re.sub(r'-+', '-', normalized)
        # Remove leading/trailing hyphens
        return normalized.strip('-')
        
    def _analyze_user_input(self, user_input: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze user input to extract key requirements information.
        
        This is a simplified analysis. In a real implementation, this would
        use NLP techniques or LLM analysis to extract:
        - Core functionality
        - User personas
        - Success criteria
        - Constraints
        - Ambiguities requiring clarification
        """
        analysis = {
            "core_functionality": [],
            "user_personas": [],
            "success_criteria": [],
            "constraints": [],
            "clarifications_needed": [],
            "estimated_complexity": "medium"
        }
        
        # Simple keyword-based analysis (would be more sophisticated in reality)
        input_lower = user_input.lower()
        
        # Identify potential clarifications needed
        ambiguous_terms = ["better", "improved", "easier", "faster", "more", "some", "many"]
        for term in ambiguous_terms:
            if term in input_lower:
                analysis["clarifications_needed"].append(
                    f"Define what '{term}' means specifically in the context: '{user_input}'"
                )
                
        # Look for missing details
        if "user" not in input_lower and "customer" not in input_lower:
            analysis["clarifications_needed"].append("Who are the target users for this feature?")
            
        if "why" not in input_lower and "problem" not in input_lower:
            analysis["clarifications_needed"].append("What problem does this solve for users?")
            
        return analysis
        
    def _generate_requirements_document(self, 
                                      spec_name: str, 
                                      user_input: str, 
                                      analysis: Dict[str, Any],
                                      context: Optional[Dict[str, Any]]) -> str:
        """Generate the requirements document content."""
        
        template = f"""# Requirements Specification: {spec_name}

> Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Status: Draft - Needs Review
> Phase: Requirements Gathering

## Overview

**Original Request:**
{user_input}

## Purpose and Goals

### Problem Statement
[NEEDS CLARIFICATION: What specific problem does this solve for users?]

### Goals
[NEEDS CLARIFICATION: What are the measurable goals for this feature?]

## User Stories

### Primary User Story
As a [NEEDS CLARIFICATION: user type],
I want [NEEDS CLARIFICATION: specific capability],
So that [NEEDS CLARIFICATION: benefit/outcome].

### Additional User Stories
[NEEDS CLARIFICATION: Are there additional user scenarios to consider?]

## Acceptance Criteria

### Must Have
- [NEEDS CLARIFICATION: What are the essential requirements?]

### Should Have
- [NEEDS CLARIFICATION: What are important but not critical requirements?]

### Could Have
- [NEEDS CLARIFICATION: What are nice-to-have features?]

### Won't Have (This Release)
- [NEEDS CLARIFICATION: What is explicitly out of scope?]

## Success Criteria

### Functional Success
- [NEEDS CLARIFICATION: How will we know the feature works correctly?]

### Business Success
- [NEEDS CLARIFICATION: How will we measure business impact?]

### User Experience Success
- [NEEDS CLARIFICATION: How will we measure user satisfaction?]

## Constraints and Dependencies

### Technical Constraints
- [NEEDS CLARIFICATION: Any technical limitations or requirements?]

### Business Constraints
- [NEEDS CLARIFICATION: Any business rules or policies that apply?]

### Timeline Constraints
- [NEEDS CLARIFICATION: Any deadline or timeline requirements?]

## Risk Assessment

### High Risk
- [NEEDS CLARIFICATION: What could prevent success?]

### Medium Risk
- [NEEDS CLARIFICATION: What might cause delays or issues?]

### Mitigation Strategies
- [NEEDS CLARIFICATION: How can we address identified risks?]

## Clarifications Needed

The following items need clarification before proceeding to design:

"""
        
        # Add identified clarifications
        for i, clarification in enumerate(analysis.get("clarifications_needed", []), 1):
            template += f"{i}. {clarification}\n"
            
        template += """
## Review Checklist

- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] All user stories have clear acceptance criteria
- [ ] Success criteria are measurable
- [ ] Constraints are clearly identified
- [ ] Scope is well-defined (in-scope and out-of-scope)
- [ ] Stakeholders have reviewed and approved

## Next Steps

1. Address all clarifications needed
2. Review with stakeholders
3. Get approval from Spec Judge Agent
4. Proceed to Design phase

---

*This document follows SDD principles: specifications drive implementation, not the other way around.*
"""
        
        return template
        
    def _apply_clarifications(self, requirements: str, clarifications: Dict[str, str]) -> str:
        """Apply clarifications to replace [NEEDS CLARIFICATION] markers."""
        updated_requirements = requirements
        
        for question, answer in clarifications.items():
            # Find and replace clarification markers
            # This is simplified - would need more sophisticated parsing
            if question in updated_requirements:
                updated_requirements = updated_requirements.replace(
                    f"[NEEDS CLARIFICATION: {question}]", 
                    answer
                )
                
        return updated_requirements
        
    def _check_requirements_completeness(self, requirements: str) -> bool:
        """Check if requirements document is complete (no clarification markers)."""
        return "[NEEDS CLARIFICATION" not in requirements
        
    def _validate_requirements_content(self, content: str) -> Dict[str, Any]:
        """Validate requirements document content."""
        validation = {
            "is_complete": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for remaining clarification markers
        if "[NEEDS CLARIFICATION" in content:
            validation["is_complete"] = False
            validation["issues"].append("Document contains unresolved clarification markers")
            
        # Check for essential sections
        required_sections = ["Purpose and Goals", "User Stories", "Acceptance Criteria", "Success Criteria"]
        for section in required_sections:
            if section not in content:
                validation["issues"].append(f"Missing required section: {section}")
                validation["is_complete"] = False
                
        return validation
        
    def _get_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Get recommended next steps based on analysis."""
        next_steps = []
        
        if analysis.get("clarifications_needed"):
            next_steps.append("Address clarifications needed before proceeding")
            
        next_steps.extend([
            "Review requirements with stakeholders",
            "Submit to Spec Judge Agent for validation", 
            "Once approved, proceed to Design phase"
        ])
        
        return next_steps


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python spec_requirements_agent.py <command> <args>")
        print("Commands:")
        print("  create <spec_name> '<user_input>'")
        print("  validate <spec_directory>")
        sys.exit(1)
        
    command = sys.argv[1]
    agent = SpecRequirementsAgent()
    
    if command == "create":
        spec_name = sys.argv[2]
        user_input = sys.argv[3]
        result = agent.create_requirements_document(spec_name, user_input)
        print(json.dumps(result, indent=2))
        
    elif command == "validate":
        spec_directory = sys.argv[2]
        result = agent.validate_requirements(spec_directory)
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)