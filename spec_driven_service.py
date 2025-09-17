#!/usr/bin/env python3
"""
Spec-Driven Development Microservice
Implements GitHub's Spec-Kit methodology for generating comprehensive project specifications
"""

import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

class SpecRequest(BaseModel):
    """Specification generation request"""
    description: str
    project_type: str = "Web Application"
    spec_depth: str = "Detailed"
    include_sections: List[str] = []
    additional_requirements: str = ""
    output_directory: str

class FeatureRequest(BaseModel):
    """New feature specification request"""
    feature_description: str
    output_directory: str
    project_name: str = ""

class PlanRequest(BaseModel):
    """Implementation plan generation request"""
    feature_spec_path: str
    technical_approach: str
    output_directory: str

class SpecDrivenService:
    """Spec-Driven Development microservice"""
    
    def __init__(self):
        self.app = FastAPI(title="Spec-Driven Development Service", version="1.0.0")
        self.agent_api_url = "http://localhost:8002"
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Spec-Driven Development Service",
                "status": "operational",
                "methodology": "GitHub Spec-Kit SDD",
                "endpoints": [
                    "/new_feature - Create new feature specification",
                    "/generate_plan - Generate implementation plan",
                    "/generate_spec - Generate comprehensive specification",
                    "/list_specs - List existing specifications"
                ]
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/new_feature")
        async def new_feature(request: FeatureRequest):
            """Create new feature specification using SDD methodology"""
            try:
                # Create output directory
                os.makedirs(request.output_directory, exist_ok=True)
                
                # Get next feature number
                feature_number = self._get_next_feature_number(request.output_directory)
                
                # Generate branch name
                branch_name = self._generate_branch_name(feature_number, request.feature_description)
                
                # Create feature directory
                feature_dir = os.path.join(request.output_directory, f"specs/{branch_name}")
                os.makedirs(feature_dir, exist_ok=True)
                
                # Generate feature specification
                spec_content = await self._generate_feature_spec(
                    request.feature_description,
                    feature_number,
                    request.project_name
                )
                
                # Save feature specification
                spec_file = os.path.join(feature_dir, "feature-spec.md")
                with open(spec_file, 'w') as f:
                    f.write(spec_content)
                
                return {
                    "status": "success",
                    "feature_number": feature_number,
                    "branch_name": branch_name,
                    "spec_file": spec_file,
                    "feature_directory": feature_dir,
                    "message": f"Feature {feature_number} specification created"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate_plan")
        async def generate_plan(request: PlanRequest):
            """Generate implementation plan from feature specification"""
            try:
                # Read feature specification
                if not os.path.exists(request.feature_spec_path):
                    raise HTTPException(status_code=404, detail="Feature specification not found")
                
                with open(request.feature_spec_path, 'r') as f:
                    feature_spec = f.read()
                
                # Generate implementation plan
                plan_content = await self._generate_implementation_plan(
                    feature_spec,
                    request.technical_approach
                )
                
                # Create implementation details directory
                spec_dir = os.path.dirname(request.feature_spec_path)
                impl_dir = os.path.join(spec_dir, "implementation-details")
                os.makedirs(impl_dir, exist_ok=True)
                
                # Save implementation plan
                plan_file = os.path.join(spec_dir, "implementation-plan.md")
                with open(plan_file, 'w') as f:
                    f.write(plan_content)
                
                # Generate supporting documents
                supporting_docs = await self._generate_supporting_docs(
                    feature_spec,
                    request.technical_approach,
                    impl_dir
                )
                
                return {
                    "status": "success",
                    "plan_file": plan_file,
                    "implementation_directory": impl_dir,
                    "supporting_documents": supporting_docs,
                    "message": "Implementation plan generated successfully"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate_spec")
        async def generate_spec(request: SpecRequest):
            """Generate comprehensive project specification"""
            try:
                # Create output directory
                os.makedirs(request.output_directory, exist_ok=True)
                
                # Generate specification
                spec_content = await self._generate_comprehensive_spec(request)
                
                # Save specification
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                project_name = request.project_type.lower().replace(' ', '_')
                spec_file = os.path.join(request.output_directory, f"{project_name}_{timestamp}_specification.md")
                
                with open(spec_file, 'w') as f:
                    f.write(spec_content)
                
                # Extract and save sections
                sections = self._extract_sections(spec_content)
                section_files = []
                
                for section_name, section_content in sections.items():
                    if section_content.strip():
                        section_filename = f"{project_name}_{timestamp}_{section_name.lower().replace(' ', '_')}.md"
                        section_file = os.path.join(request.output_directory, section_filename)
                        
                        with open(section_file, 'w') as f:
                            f.write(f"# {section_name}\n\n{section_content}")
                        
                        section_files.append(section_file)
                
                return {
                    "status": "success",
                    "specification_file": spec_file,
                    "section_files": section_files,
                    "output_directory": request.output_directory,
                    "message": "Comprehensive specification generated"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/list_specs")
        async def list_specs(output_dir: str):
            """List existing specifications"""
            try:
                if not os.path.exists(output_dir):
                    return {"specifications": [], "message": "Output directory does not exist"}
                
                specs = []
                for file in os.listdir(output_dir):
                    if file.endswith('.md'):
                        file_path = os.path.join(output_dir, file)
                        file_time = os.path.getmtime(file_path)
                        specs.append({
                            "filename": file,
                            "path": file_path,
                            "modified": datetime.fromtimestamp(file_time).isoformat(),
                            "size": os.path.getsize(file_path)
                        })
                
                # Sort by modification time (newest first)
                specs.sort(key=lambda x: x['modified'], reverse=True)
                
                return {"specifications": specs}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_next_feature_number(self, output_dir: str) -> str:
        """Get next feature number by scanning existing specs"""
        specs_dir = os.path.join(output_dir, "specs")
        if not os.path.exists(specs_dir):
            return "001"
        
        max_num = 0
        for item in os.listdir(specs_dir):
            if os.path.isdir(os.path.join(specs_dir, item)):
                # Extract number from directory name (e.g., "001-chat-system" -> 1)
                parts = item.split('-')
                if parts and parts[0].isdigit():
                    max_num = max(max_num, int(parts[0]))
        
        return f"{max_num + 1:03d}"
    
    def _generate_branch_name(self, feature_number: str, description: str) -> str:
        """Generate semantic branch name"""
        # Clean description for branch name
        clean_desc = description.lower()
        clean_desc = ''.join(c if c.isalnum() or c.isspace() else '' for c in clean_desc)
        clean_desc = '-'.join(clean_desc.split()[:4])  # Max 4 words
        
        return f"{feature_number}-{clean_desc}"
    
    async def _generate_feature_spec(self, description: str, feature_number: str, project_name: str) -> str:
        """Generate feature specification using SDD methodology"""
        
        task = f"""Generate a feature specification following GitHub's Spec-Kit methodology.

Feature Description: {description}
Feature Number: {feature_number}
Project: {project_name}

Create a comprehensive feature specification that includes:

# Feature {feature_number}: [Feature Name]

## Overview
Brief description of what this feature does and why it's needed.

## User Stories
Detailed user stories with acceptance criteria:
- As a [user type], I want [goal] so that [benefit]
- Acceptance Criteria:
  - [ ] Specific, testable criteria
  - [ ] Edge cases covered
  - [ ] Error handling defined

## Functional Requirements
What the system must do:
- REQ-{feature_number}-001: [Specific requirement]
- REQ-{feature_number}-002: [Another requirement]

## Non-Functional Requirements
Performance, security, usability requirements:
- NFR-{feature_number}-001: [Performance requirement]
- NFR-{feature_number}-002: [Security requirement]

## Success Metrics
How we'll measure success:
- Metric 1: [Measurable outcome]
- Metric 2: [Another measurable outcome]

## Dependencies
What this feature depends on:
- Internal: [Other features/systems]
- External: [Third-party services]

## Risks and Mitigations
Potential risks and how to address them:
- Risk 1: [Description] → Mitigation: [Solution]

## Out of Scope
What this feature explicitly does NOT include:
- [Item 1]
- [Item 2]

IMPORTANT: 
- Focus on WHAT users need and WHY, not HOW to implement
- Use [NEEDS CLARIFICATION] markers for any ambiguities
- Make requirements testable and unambiguous
- No technical implementation details in this specification
"""

        try:
            response = requests.post(
                f"{self.agent_api_url}/execute",
                json={
                    "task": task,
                    "agent_id": "system_architect",
                    "model": "gpt-4o"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", "Failed to generate specification")
            else:
                return f"Error generating specification: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _generate_implementation_plan(self, feature_spec: str, technical_approach: str) -> str:
        """Generate implementation plan from feature specification"""
        
        task = f"""Generate a detailed implementation plan following SDD methodology.

FEATURE SPECIFICATION:
{feature_spec}

TECHNICAL APPROACH:
{technical_approach}

Create a comprehensive implementation plan that includes:

# Implementation Plan

## Technical Architecture
High-level technical approach and architecture decisions.

## Phase -1: Pre-Implementation Gates
### Simplicity Gate (Article VII)
- [ ] Using ≤3 projects?
- [ ] No future-proofing?
- [ ] Minimal project structure?

### Anti-Abstraction Gate (Article VIII)  
- [ ] Using framework directly?
- [ ] Single model representation?
- [ ] No unnecessary wrappers?

### Integration-First Gate (Article IX)
- [ ] Contracts defined?
- [ ] Contract tests written?
- [ ] Real environment testing?

## Implementation Phases

### Phase 0: Foundation
- Library structure setup
- CLI interface definition
- Basic project scaffolding

### Phase 1: Core Implementation
- Core functionality implementation
- Unit tests (TDD approach)
- Integration tests

### Phase 2: Integration
- API endpoints
- Database integration
- External service connections

### Phase 3: Validation
- End-to-end testing
- Performance validation
- Security review

## Technology Decisions
Document all technology choices with rationale:
- Framework: [Choice] - Reason: [Why]
- Database: [Choice] - Reason: [Why]
- Testing: [Choice] - Reason: [Why]

## File Structure
```
library-name/
├── src/
├── tests/
├── cli/
└── docs/
```

## Test Strategy
Following TDD principles:
1. Write tests first
2. Implement to make tests pass
3. Refactor for quality

## Acceptance Criteria Mapping
Map each user story to implementation components.

IMPORTANT:
- Follow constitutional principles (library-first, CLI interface, test-first)
- Keep high-level, put detailed specs in implementation-details/
- Every technical decision must trace back to requirements
- Use [NEEDS CLARIFICATION] for any ambiguities
"""

        try:
            response = requests.post(
                f"{self.agent_api_url}/execute",
                json={
                    "task": task,
                    "agent_id": "system_architect", 
                    "model": "gpt-4o"
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", "Failed to generate implementation plan")
            else:
                return f"Error generating plan: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _generate_supporting_docs(self, feature_spec: str, technical_approach: str, impl_dir: str) -> List[str]:
        """Generate supporting implementation documents"""
        
        documents = [
            ("00-research.md", "Research findings and technology comparisons"),
            ("02-data-model.md", "Database schema and data structures"),
            ("03-api-contracts.md", "API endpoints and contracts"),
            ("06-contract-tests.md", "Contract test scenarios"),
            ("08-inter-library-tests.md", "Integration test specifications")
        ]
        
        generated_files = []
        
        for filename, description in documents:
            task = f"""Generate {description} for the following feature:

FEATURE SPECIFICATION:
{feature_spec}

TECHNICAL APPROACH:
{technical_approach}

Create detailed {description} following SDD methodology. Include:
- Specific technical details
- Code examples where appropriate
- Test scenarios
- Implementation guidelines

Focus on {description} specifically."""

            try:
                response = requests.post(
                    f"{self.agent_api_url}/execute",
                    json={
                        "task": task,
                        "agent_id": "system_architect",
                        "model": "gpt-4o"
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("result", f"# {description}\n\nContent generation failed")
                    
                    file_path = os.path.join(impl_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(content)
                    
                    generated_files.append(file_path)
                
            except Exception as e:
                print(f"Error generating {filename}: {e}")
        
        return generated_files
    
    async def _generate_comprehensive_spec(self, request: SpecRequest) -> str:
        """Generate comprehensive project specification"""
        
        sections_text = ", ".join(request.include_sections) if request.include_sections else "standard sections"
        
        task = f"""Generate a comprehensive {request.spec_depth.lower()} project specification for a {request.project_type}.

PROJECT DESCRIPTION:
{request.description}

ADDITIONAL REQUIREMENTS:
{request.additional_requirements}

INCLUDE SECTIONS: {sections_text}

Create a complete specification following SDD methodology that includes:

# {request.project_type} Specification

## Executive Summary
Project overview, objectives, and success criteria.

## Technical Architecture  
System design, components, and architectural decisions.

## Functional Requirements
Detailed feature specifications with acceptance criteria.

## Non-Functional Requirements
Performance, security, scalability, and usability requirements.

## API Specification
(If requested) Endpoints, request/response formats, authentication.

## Data Model
(If requested) Database schema, relationships, constraints.

## User Stories
(If requested) Detailed user scenarios with acceptance criteria.

## Security Requirements
(If requested) Authentication, authorization, data protection.

## Testing Strategy
(If requested) Unit, integration, and end-to-end testing plans.

## Deployment Specification
(If requested) Infrastructure, CI/CD, environment setup.

## Monitoring & Observability
(If requested) Logging, metrics, alerting strategies.

## Implementation Guidelines
Development approach and best practices.

## Acceptance Criteria
Measurable success criteria for the project.

IMPORTANT:
- Make specifications precise and unambiguous
- Include code examples where appropriate
- Focus on executable specifications
- Use [NEEDS CLARIFICATION] for ambiguities
- Ensure specifications can generate working systems
"""

        try:
            response = requests.post(
                f"{self.agent_api_url}/execute",
                json={
                    "task": task,
                    "agent_id": "system_architect",
                    "model": "gpt-4o"
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", "Failed to generate specification")
            else:
                return f"Error generating specification: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('# ') or line.startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.lstrip('# ').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

def main():
    """Run the Spec-Driven Development service"""
    import uvicorn
    
    service = SpecDrivenService()
    
    print("\n" + "="*60)
    print("📋 SPEC-DRIVEN DEVELOPMENT SERVICE")
    print("="*60)
    print("\n🎯 GitHub Spec-Kit Methodology Implementation")
    print("\n📍 Endpoints:")
    print("  • http://localhost:8029/ - Service info")
    print("  • http://localhost:8029/new_feature - Create feature spec")
    print("  • http://localhost:8029/generate_plan - Generate implementation plan")
    print("  • http://localhost:8029/generate_spec - Generate comprehensive spec")
    print("  • http://localhost:8029/list_specs - List existing specifications")
    print("\n✨ Features:")
    print("  • Executable specifications")
    print("  • Constitutional compliance")
    print("  • Test-first development")
    print("  • Library-first architecture")
    print("  • Continuous refinement")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8029)

if __name__ == "__main__":
    main()