"""
Spec Design Agent

This agent creates and refines specification design documents.
It translates requirements into technical architecture and system design.

Key responsibilities:
- Convert requirements into technical specifications
- Design data models and system architecture
- Define API contracts and interfaces
- Create detailed technical plans
- Ensure alignment with constitutional principles
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class SpecDesignAgent:
    """
    Agent for creating and refining specification design documents.
    
    This agent operates after requirements approval and focuses on
    translating business requirements into technical architecture
    following SDD and constitutional principles.
    """
    
    def __init__(self, project_path: Optional[str] = None):
        """
        Initialize the design agent.
        
        Args:
            project_path: Path to the project root. If None, uses current directory.
        """
        if project_path is None:
            project_path = os.getcwd()
        self.project_path = Path(project_path)
        self.specs_path = self.project_path / "specs"
        self.constitution_path = Path(__file__).parent.parent / "constitution"
        
    def create_design_document(self, 
                              spec_directory: str,
                              design_approach: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a design document from approved requirements.
        
        Args:
            spec_directory: Path to the spec directory with requirements
            design_approach: Optional specific design approach or constraints
            
        Returns:
            Dictionary with creation results and next steps
        """
        spec_dir = Path(spec_directory)
        
        # Load and validate requirements
        requirements_result = self._load_requirements(spec_dir)
        if not requirements_result["success"]:
            return requirements_result
            
        requirements = requirements_result["requirements"]
        metadata = requirements_result["metadata"]
        
        # Check constitutional principles
        constitutional_check = self._check_constitutional_compliance(requirements)
        
        # Generate design document
        design_doc = self._generate_design_document(
            requirements, metadata, constitutional_check, design_approach
        )
        
        # Write design document
        design_path = spec_dir / "design.md"
        with open(design_path, 'w') as f:
            f.write(design_doc)
            
        # Update metadata
        metadata["design_created_at"] = datetime.now().isoformat()
        metadata["status"] = "design_draft"
        metadata["phase"] = "design"
        metadata["files"]["design"] = str(design_path)
        metadata["constitutional_check"] = constitutional_check
        
        metadata_path = spec_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "success": True,
            "design_path": str(design_path),
            "metadata": metadata,
            "constitutional_issues": constitutional_check.get("issues", []),
            "next_steps": self._get_design_next_steps(constitutional_check)
        }
        
    def refine_design(self, 
                     spec_directory: str, 
                     refinements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine an existing design document.
        
        Args:
            spec_directory: Path to the spec directory
            refinements: Dictionary of refinement instructions
            
        Returns:
            Dictionary with refinement results
        """
        spec_dir = Path(spec_directory)
        design_path = spec_dir / "design.md"
        
        if not design_path.exists():
            return {
                "success": False,
                "error": "Design document not found"
            }
            
        # Load current design
        with open(design_path, 'r') as f:
            current_design = f.read()
            
        # Apply refinements
        refined_design = self._apply_design_refinements(current_design, refinements)
        
        # Re-check constitutional compliance
        constitutional_check = self._check_constitutional_compliance_design(refined_design)
        
        # Write updated design
        with open(design_path, 'w') as f:
            f.write(refined_design)
            
        # Update metadata
        metadata_path = spec_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        metadata["design_updated_at"] = datetime.now().isoformat()
        metadata["refinements_applied"] = refinements
        metadata["constitutional_check"] = constitutional_check
        metadata["status"] = "design_refined"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "success": True,
            "design_path": str(design_path),
            "metadata": metadata,
            "constitutional_issues": constitutional_check.get("issues", []),
            "ready_for_judge": constitutional_check.get("compliant", False)
        }
        
    def validate_design(self, spec_directory: str) -> Dict[str, Any]:
        """
        Validate design document for completeness and constitutional compliance.
        
        Args:
            spec_directory: Path to the spec directory
            
        Returns:
            Validation results with issues and recommendations
        """
        spec_dir = Path(spec_directory)
        design_path = spec_dir / "design.md"
        
        if not design_path.exists():
            return {
                "success": False,
                "error": "Design document not found"
            }
            
        with open(design_path, 'r') as f:
            design_content = f.read()
            
        validation = self._validate_design_content(design_content)
        constitutional_check = self._check_constitutional_compliance_design(design_content)
        
        return {
            "success": True,
            "validation": validation,
            "constitutional_compliance": constitutional_check,
            "ready_for_tasks": validation["is_complete"] and constitutional_check.get("compliant", False),
            "issues": validation.get("issues", []) + constitutional_check.get("issues", []),
            "recommendations": validation.get("recommendations", [])
        }
        
    def _load_requirements(self, spec_dir: Path) -> Dict[str, Any]:
        """Load and validate requirements document."""
        requirements_path = spec_dir / "requirements.md"
        metadata_path = spec_dir / "metadata.json"
        
        if not requirements_path.exists():
            return {
                "success": False,
                "error": "Requirements document not found"
            }
            
        if not metadata_path.exists():
            return {
                "success": False,
                "error": "Metadata file not found"
            }
            
        with open(requirements_path, 'r') as f:
            requirements = f.read()
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Check if requirements are approved
        if "[NEEDS CLARIFICATION" in requirements:
            return {
                "success": False,
                "error": "Requirements contain unresolved clarifications"
            }
            
        return {
            "success": True,
            "requirements": requirements,
            "metadata": metadata
        }
        
    def _check_constitutional_compliance(self, requirements: str) -> Dict[str, Any]:
        """Check requirements against constitutional principles."""
        compliance = {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Article I: Library-First Principle
        if "library" not in requirements.lower():
            compliance["recommendations"].append(
                "Consider how this feature can be implemented as a standalone library (Article I)"
            )
            
        # Article VII: Simplicity
        complex_terms = ["microservice", "distributed", "scalable", "enterprise"]
        if any(term in requirements.lower() for term in complex_terms):
            compliance["issues"].append(
                "Requirements suggest complex architecture - ensure justification per Article VII"
            )
            compliance["compliant"] = False
            
        return compliance
        
    def _check_constitutional_compliance_design(self, design: str) -> Dict[str, Any]:
        """Check design against constitutional principles."""
        compliance = {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Article I: Library-First Principle
        if "library" not in design.lower():
            compliance["issues"].append(
                "Design must implement feature as standalone library (Article I)"
            )
            compliance["compliant"] = False
            
        # Article II: CLI Interface Mandate  
        if "cli" not in design.lower() and "command" not in design.lower():
            compliance["issues"].append(
                "Design must include CLI interface (Article II)"
            )
            compliance["compliant"] = False
            
        # Article III: Test-First Imperative
        if "test" not in design.lower() or "tdd" not in design.lower():
            compliance["issues"].append(
                "Design must specify test-first approach (Article III)"
            )
            compliance["compliant"] = False
            
        # Article VII: Simplicity (≤3 projects)
        project_count = design.lower().count("project")
        if project_count > 3:
            compliance["issues"].append(
                f"Design suggests {project_count} projects, maximum 3 allowed (Article VII)"
            )
            compliance["compliant"] = False
            
        return compliance
        
    def _generate_design_document(self, 
                                 requirements: str, 
                                 metadata: Dict[str, Any],
                                 constitutional_check: Dict[str, Any],
                                 design_approach: Optional[str]) -> str:
        """Generate the design document content."""
        
        spec_name = metadata.get("spec_name", "Unknown Spec")
        
        template = f"""# Technical Design: {spec_name}

> Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Status: Draft - Needs Review
> Phase: Design
> Based on: requirements.md

## Design Overview

### Architecture Summary
This feature will be implemented as a standalone library following constitutional principles.

### Constitutional Compliance Gates

#### Article I: Library-First Principle ✓
- Feature implemented as standalone library
- Clear module boundaries and interfaces
- Reusable across different applications

#### Article II: CLI Interface Mandate
- [ ] Text-based input/output interface
- [ ] JSON format support for structured data
- [ ] Command-line accessibility

#### Article III: Test-First Imperative  
- [ ] Unit tests written before implementation
- [ ] Tests validated and approved
- [ ] TDD Red-Green-Refactor cycle

#### Article VII: Simplicity Gate
- [ ] Using ≤3 projects total
- [ ] No premature optimization
- [ ] Minimal necessary complexity

#### Article VIII: Anti-Abstraction Gate
- [ ] Using framework features directly
- [ ] Single model representation
- [ ] No unnecessary abstractions

#### Article IX: Integration-First Testing
- [ ] Real database/service integration
- [ ] Contract tests defined
- [ ] End-to-end scenarios

## System Architecture

### High-Level Design
```
[User Input] → [CLI Interface] → [Core Library] → [Data Layer] → [Output]
```

### Component Structure
1. **Core Library Module**
   - Primary business logic
   - Domain models and rules
   - Public API interfaces

2. **CLI Interface Module**  
   - Command-line argument parsing
   - Input/output formatting
   - Error handling and reporting

3. **Data Layer Module**
   - Data persistence (if needed)
   - External service integration
   - Configuration management

### Technology Stack

#### Programming Language
- Primary: [SPECIFY BASED ON PROJECT]
- Testing: [SPECIFY TEST FRAMEWORK]

#### Dependencies
- Minimal external dependencies
- Well-established, maintained libraries only
- Version pinning for stability

#### Data Storage
- [SPECIFY IF NEEDED: File-based, Database, etc.]
- [JUSTIFY CHOICE BASED ON REQUIREMENTS]

## Data Models

### Core Domain Models
```
[DEFINE KEY DATA STRUCTURES]
```

### Data Flow
1. Input validation and parsing
2. Business logic processing  
3. Output generation and formatting

## API Contracts

### CLI Interface Contract
```bash
# Primary command structure
command-name [options] [arguments]

# Input: Text via stdin, args, or files
# Output: Text via stdout (structured as JSON when appropriate)
```

### Library Interface Contract
```
[DEFINE PUBLIC API METHODS]
```

## Integration Points

### External Dependencies
- [LIST ANY EXTERNAL SERVICES OR APIS]
- [JUSTIFY EACH DEPENDENCY]

### Internal Integration
- [DESCRIBE HOW THIS INTEGRATES WITH EXISTING SYSTEM]

## Testing Strategy

### Test-First Development Plan
1. **Unit Tests**: Core business logic
2. **Integration Tests**: Data layer and external services  
3. **Contract Tests**: API interface compliance
4. **End-to-End Tests**: Complete user scenarios

### Test Environment Requirements
- Real database instances (no mocks for integration)
- Actual service endpoints where possible
- Isolated test data and cleanup

## Implementation Phases

### Phase 1: Core Library
- [ ] Define data models
- [ ] Implement core business logic
- [ ] Unit test coverage

### Phase 2: CLI Interface
- [ ] Command-line argument parsing
- [ ] Input/output formatting
- [ ] Integration with core library

### Phase 3: Integration
- [ ] Data layer implementation
- [ ] External service integration
- [ ] End-to-end testing

## Security Considerations

### Input Validation
- All user inputs validated and sanitized
- Error handling for malformed data
- Resource limits and timeouts

### Data Protection
- [SPECIFY ANY SENSITIVE DATA HANDLING]
- [DEFINE ACCESS CONTROLS IF NEEDED]

## Performance Requirements

### Scalability Targets
- [DEFINE BASED ON REQUIREMENTS]
- [AVOID PREMATURE OPTIMIZATION]

### Resource Constraints
- Memory usage limits
- Processing time expectations
- Storage requirements

## Deployment Strategy

### Distribution Method
- Library package distribution
- CLI tool installation
- Configuration management

### Environment Requirements
- Runtime dependencies
- System requirements
- Configuration needs

## Risk Assessment

### Technical Risks
- [IDENTIFY POTENTIAL TECHNICAL CHALLENGES]
- [DEFINE MITIGATION STRATEGIES]

### Integration Risks
- [CONSIDER EXTERNAL DEPENDENCIES]
- [PLAN FOR SERVICE UNAVAILABILITY]

## Constitutional Compliance Review

"""
        
        # Add constitutional compliance status
        if constitutional_check.get("issues"):
            template += "### Issues to Address\n"
            for issue in constitutional_check["issues"]:
                template += f"- ❌ {issue}\n"
                
        if constitutional_check.get("recommendations"):
            template += "\n### Recommendations\n"
            for rec in constitutional_check["recommendations"]:
                template += f"- 💡 {rec}\n"
                
        template += """
## Review Checklist

- [ ] All constitutional gates pass
- [ ] Architecture follows library-first principle
- [ ] CLI interface properly defined
- [ ] Test strategy includes TDD approach
- [ ] Complexity justified (≤3 projects)
- [ ] No unnecessary abstractions
- [ ] Integration testing planned

## Next Steps

1. Address any constitutional compliance issues
2. Review design with stakeholders
3. Get approval from Spec Judge Agent
4. Proceed to Task breakdown phase

---

*This design follows SDD principles and constitutional requirements.*
"""
        
        return template
        
    def _apply_design_refinements(self, design: str, refinements: Dict[str, Any]) -> str:
        """Apply refinements to the design document."""
        # This is simplified - would implement sophisticated refinement logic
        updated_design = design
        
        for section, updates in refinements.items():
            if isinstance(updates, str):
                # Simple text replacement
                updated_design = updated_design.replace(f"[{section}]", updates)
                
        return updated_design
        
    def _validate_design_content(self, content: str) -> Dict[str, Any]:
        """Validate design document content."""
        validation = {
            "is_complete": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for essential sections
        required_sections = [
            "Architecture Summary", "System Architecture", "Data Models", 
            "API Contracts", "Testing Strategy", "Constitutional Compliance"
        ]
        
        for section in required_sections:
            if section not in content:
                validation["issues"].append(f"Missing required section: {section}")
                validation["is_complete"] = False
                
        # Check for TODO markers
        if "[SPECIFY" in content or "[DEFINE" in content:
            validation["issues"].append("Design contains unresolved placeholders")
            validation["is_complete"] = False
            
        return validation
        
    def _get_design_next_steps(self, constitutional_check: Dict[str, Any]) -> List[str]:
        """Get recommended next steps based on constitutional compliance."""
        next_steps = []
        
        if constitutional_check.get("issues"):
            next_steps.append("Address constitutional compliance issues")
            
        next_steps.extend([
            "Complete any missing design sections",
            "Review design with technical stakeholders",
            "Submit to Spec Judge Agent for validation",
            "Once approved, proceed to Task breakdown phase"
        ])
        
        return next_steps


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python spec_design_agent.py <command> <args>")
        print("Commands:")
        print("  create <spec_directory> [design_approach]")
        print("  validate <spec_directory>")
        sys.exit(1)
        
    command = sys.argv[1]
    agent = SpecDesignAgent()
    
    if command == "create":
        spec_directory = sys.argv[2]
        design_approach = sys.argv[3] if len(sys.argv) > 3 else None
        result = agent.create_design_document(spec_directory, design_approach)
        print(json.dumps(result, indent=2))
        
    elif command == "validate":
        spec_directory = sys.argv[2]
        result = agent.validate_design(spec_directory)
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)