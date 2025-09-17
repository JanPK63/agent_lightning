#!/usr/bin/env python3
"""
Agent Visual Builder Integration
Intelligently uses Visual Code Builder for complex tasks
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import VisualProgram, BlockFactory, BlockType
from visual_component_library import ComponentLibrary, ComponentCategory
from visual_to_code_translator import VisualToCodeTranslator, TargetLanguage
from visual_debugger import VisualDebugger, DebugState


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"        # < 50 lines, single function
    MODERATE = "moderate"    # 50-200 lines, multiple functions
    COMPLEX = "complex"      # 200+ lines, multiple modules
    ARCHITECTURAL = "architectural"  # System design, multiple services


class VisualPlanningDecision(Enum):
    """Decision on whether to use visual planning"""
    REQUIRED = "required"        # Must use visual planning
    RECOMMENDED = "recommended"  # Should use visual planning
    OPTIONAL = "optional"        # Can use if beneficial
    SKIP = "skip"               # Too simple, skip visual


@dataclass
class TaskAnalysis:
    """Analysis of a task to determine complexity"""
    task_description: str
    detected_components: List[str] = field(default_factory=list)
    estimated_lines: int = 0
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    visual_decision: VisualPlanningDecision = VisualPlanningDecision.SKIP
    confidence: float = 0.0
    reasoning: str = ""


class TaskComplexityAnalyzer:
    """Analyzes tasks to determine if visual planning is needed"""
    
    # Keywords that indicate complexity
    COMPLEXITY_INDICATORS = {
        "simple": [
            "print", "log", "display", "show", "get", "fetch", "return",
            "calculate", "add", "subtract", "multiply", "divide"
        ],
        "moderate": [
            "process", "transform", "filter", "aggregate", "combine",
            "validate", "parse", "format", "convert", "map"
        ],
        "complex": [
            "pipeline", "workflow", "orchestrate", "integrate", "synchronize",
            "distribute", "optimize", "refactor", "architect", "scale"
        ],
        "architectural": [
            "microservice", "api gateway", "event-driven", "message queue",
            "load balancer", "database cluster", "kubernetes", "docker"
        ]
    }
    
    # Component patterns
    COMPONENT_PATTERNS = {
        "api": r"(api|endpoint|rest|graphql|webhook)",
        "database": r"(database|db|sql|nosql|mongo|postgres|mysql)",
        "auth": r"(auth|authentication|authorization|oauth|jwt|token)",
        "file": r"(file|read|write|csv|json|xml|parse)",
        "network": r"(http|request|fetch|download|upload|stream)",
        "async": r"(async|await|concurrent|parallel|thread|queue)",
        "ml": r"(machine learning|ml|ai|model|predict|train)",
        "ui": r"(ui|interface|frontend|react|vue|angular|component)"
    }
    
    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Analyze a task to determine its complexity and visual planning needs"""
        analysis = TaskAnalysis(task_description=task_description)
        
        # Convert to lowercase for analysis
        task_lower = task_description.lower()
        
        # Detect components
        for component, pattern in self.COMPONENT_PATTERNS.items():
            if re.search(pattern, task_lower, re.IGNORECASE):
                analysis.detected_components.append(component)
        
        # Estimate complexity based on keywords
        complexity_scores = {
            TaskComplexity.SIMPLE: 0,
            TaskComplexity.MODERATE: 0,
            TaskComplexity.COMPLEX: 0,
            TaskComplexity.ARCHITECTURAL: 0
        }
        
        for level, keywords in self.COMPLEXITY_INDICATORS.items():
            for keyword in keywords:
                if keyword in task_lower:
                    if level == "simple":
                        complexity_scores[TaskComplexity.SIMPLE] += 1
                    elif level == "moderate":
                        complexity_scores[TaskComplexity.MODERATE] += 2
                    elif level == "complex":
                        complexity_scores[TaskComplexity.COMPLEX] += 3
                    elif level == "architectural":
                        complexity_scores[TaskComplexity.ARCHITECTURAL] += 4
        
        # Factor in number of components
        component_count = len(analysis.detected_components)
        if component_count > 5:
            complexity_scores[TaskComplexity.ARCHITECTURAL] += 5
        elif component_count > 3:
            complexity_scores[TaskComplexity.COMPLEX] += 3
        elif component_count > 1:
            complexity_scores[TaskComplexity.MODERATE] += 2
        
        # Determine complexity
        max_score = max(complexity_scores.values())
        if max_score == 0:
            analysis.complexity = TaskComplexity.SIMPLE
        else:
            for complexity, score in complexity_scores.items():
                if score == max_score:
                    analysis.complexity = complexity
                    break
        
        # Estimate lines of code
        if analysis.complexity == TaskComplexity.SIMPLE:
            analysis.estimated_lines = 20 + (component_count * 10)
        elif analysis.complexity == TaskComplexity.MODERATE:
            analysis.estimated_lines = 100 + (component_count * 30)
        elif analysis.complexity == TaskComplexity.COMPLEX:
            analysis.estimated_lines = 300 + (component_count * 50)
        else:  # ARCHITECTURAL
            analysis.estimated_lines = 500 + (component_count * 100)
        
        # Determine visual planning decision
        if analysis.complexity == TaskComplexity.ARCHITECTURAL:
            analysis.visual_decision = VisualPlanningDecision.REQUIRED
            analysis.reasoning = "Architectural complexity requires visual planning for reliability"
        elif analysis.complexity == TaskComplexity.COMPLEX:
            analysis.visual_decision = VisualPlanningDecision.RECOMMENDED
            analysis.reasoning = "Complex task would benefit from visual planning"
        elif analysis.complexity == TaskComplexity.MODERATE and component_count > 2:
            analysis.visual_decision = VisualPlanningDecision.OPTIONAL
            analysis.reasoning = "Moderate complexity with multiple components"
        else:
            analysis.visual_decision = VisualPlanningDecision.SKIP
            analysis.reasoning = "Simple task - direct code generation is sufficient"
        
        # Calculate confidence
        if max_score > 10:
            analysis.confidence = 0.9
        elif max_score > 5:
            analysis.confidence = 0.7
        elif max_score > 0:
            analysis.confidence = 0.5
        else:
            analysis.confidence = 0.3
        
        return analysis


class VisualProgramBuilder:
    """Builds visual programs from task descriptions"""
    
    def __init__(self):
        self.factory = BlockFactory()
        self.component_library = ComponentLibrary()
        self.translator = VisualToCodeTranslator()
    
    def build_from_task(self, task: str, components: List[str]) -> VisualProgram:
        """Build a visual program from task description and detected components"""
        program = VisualProgram(name="Agent Task")
        
        # Add main function block
        main_func = self.factory.create_function_block()
        main_func.properties["function_name"] = "execute_task"
        main_func.properties["parameters"] = ["input_data"]
        program.add_block(main_func)
        
        # Add blocks based on detected components
        for component in components:
            blocks = self._create_component_blocks(component)
            for block in blocks:
                program.add_block(block)
                # Connect to previous block
                if len(program.blocks) > 1:
                    prev_block = program.blocks[-2]
                    if prev_block.output_ports and block.input_ports:
                        program.connect_blocks(
                            prev_block.block_id,
                            prev_block.output_ports[0].name,
                            block.block_id,
                            block.input_ports[0].name
                        )
        
        # Add return block
        return_block = self.factory.create_return_block()
        return_block.properties["value"] = "result"
        program.add_block(return_block)
        
        return program
    
    def _create_component_blocks(self, component: str) -> List:
        """Create blocks for a specific component type"""
        blocks = []
        
        if component == "api":
            # API call block
            api_block = self.factory.create_api_call_block()
            api_block.properties["method"] = "GET"
            blocks.append(api_block)
            
            # Error handling
            try_block = self.factory.create_try_catch_block()
            blocks.append(try_block)
            
        elif component == "database":
            # Database query block
            db_block = self.factory.create_database_query_block()
            db_block.properties["query_type"] = "SELECT"
            blocks.append(db_block)
            
        elif component == "file":
            # File operations
            read_block = self.factory.create_file_read_block()
            blocks.append(read_block)
            
            # Process data
            process_block = self.factory.create_expression_block()
            process_block.properties["expression"] = "process_data(content)"
            blocks.append(process_block)
            
        elif component == "auth":
            # Authentication check
            if_block = self.factory.create_if_block()
            if_block.properties["condition_expression"] = "is_authenticated"
            blocks.append(if_block)
            
        elif component in ["ml", "ai"]:
            # ML model block
            var_block = self.factory.create_variable_block()
            var_block.properties["variable_name"] = "model"
            blocks.append(var_block)
            
            # Prediction block
            expr_block = self.factory.create_expression_block()
            expr_block.properties["expression"] = "model.predict(input_data)"
            blocks.append(expr_block)
        
        return blocks


class AgentVisualIntegration:
    """Integrates Visual Code Builder with AI agents"""
    
    def __init__(self):
        self.analyzer = TaskComplexityAnalyzer()
        self.builder = VisualProgramBuilder()
        self.translator = VisualToCodeTranslator()
        self.debugger = None
        
        # Statistics
        self.stats = {
            "tasks_analyzed": 0,
            "visual_plans_created": 0,
            "visual_plans_skipped": 0,
            "lines_generated": 0,
            "errors_prevented": 0
        }
    
    async def should_use_visual_planning(self, task: str) -> Tuple[bool, TaskAnalysis]:
        """Determine if visual planning should be used for a task"""
        analysis = self.analyzer.analyze_task(task)
        self.stats["tasks_analyzed"] += 1
        
        # Decision logic
        should_use = analysis.visual_decision in [
            VisualPlanningDecision.REQUIRED,
            VisualPlanningDecision.RECOMMENDED
        ]
        
        # Override: Always use for certain keywords
        force_visual_keywords = ["architecture", "design", "blueprint", "visual", "diagram"]
        if any(keyword in task.lower() for keyword in force_visual_keywords):
            should_use = True
            analysis.reasoning += " (Forced: visual keyword detected)"
        
        return should_use, analysis
    
    async def create_visual_plan(self, task: str, analysis: TaskAnalysis) -> Optional[VisualProgram]:
        """Create a visual plan for the task"""
        try:
            # Build visual program
            program = self.builder.build_from_task(task, analysis.detected_components)
            
            # Validate the program
            if program.validate():
                self.stats["visual_plans_created"] += 1
                return program
            else:
                print(f"Visual plan validation failed for task: {task}")
                return None
                
        except Exception as e:
            print(f"Error creating visual plan: {e}")
            return None
    
    async def generate_code_from_visual(
        self,
        program: VisualProgram,
        language: TargetLanguage = TargetLanguage.PYTHON
    ) -> Tuple[bool, str, List[str]]:
        """Generate code from visual program"""
        try:
            # Translate to code
            code = self.translator.translate_program(program, language)
            
            # Validate generated code
            valid, errors = self.translator.validate_translation(code, language)
            
            if valid:
                self.stats["lines_generated"] += len(code.split('\n'))
            else:
                self.stats["errors_prevented"] += len(errors)
            
            return valid, code, errors
            
        except Exception as e:
            return False, "", [str(e)]
    
    async def process_task_with_visual_planning(
        self,
        task: str,
        force_visual: bool = False,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Process a task with optional visual planning"""
        
        # Analyze task
        should_use, analysis = await self.should_use_visual_planning(task)
        
        # Force visual if requested
        if force_visual:
            should_use = True
            analysis.reasoning += " (Forced by user)"
        
        result = {
            "task": task,
            "analysis": analysis,
            "used_visual": False,
            "visual_program": None,
            "generated_code": None,
            "errors": [],
            "success": False
        }
        
        if should_use:
            print(f"📊 Using Visual Planning for task (Complexity: {analysis.complexity.value})")
            print(f"   Reasoning: {analysis.reasoning}")
            
            # Create visual plan
            program = await self.create_visual_plan(task, analysis)
            
            if program:
                result["used_visual"] = True
                result["visual_program"] = program
                
                # Generate code
                valid, code, errors = await self.generate_code_from_visual(program)
                
                if valid:
                    result["generated_code"] = code
                    result["success"] = True
                    print(f"✅ Generated {len(code.split(chr(10)))} lines of code from visual plan")
                    
                    # Debug if requested
                    if debug:
                        self.debugger = VisualDebugger(program)
                        await self.debugger.start_debugging()
                        result["debug_session"] = self.debugger.export_debug_session()
                else:
                    result["errors"] = errors
                    print(f"❌ Code generation failed: {errors}")
            else:
                self.stats["visual_plans_skipped"] += 1
                print("⚠️ Failed to create visual plan, falling back to direct generation")
        else:
            self.stats["visual_plans_skipped"] += 1
            print(f"⏩ Skipping visual planning (Task complexity: {analysis.complexity.value})")
            result["reasoning"] = f"Direct generation recommended: {analysis.reasoning}"
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total = self.stats["visual_plans_created"] + self.stats["visual_plans_skipped"]
        visual_rate = (self.stats["visual_plans_created"] / total * 100) if total > 0 else 0
        
        return {
            **self.stats,
            "visual_planning_rate": f"{visual_rate:.1f}%",
            "avg_lines_per_task": self.stats["lines_generated"] / max(self.stats["visual_plans_created"], 1)
        }


# Enhanced agent request with visual planning option
@dataclass
class EnhancedAgentRequest:
    """Enhanced request with visual planning options"""
    task: str
    agent_id: Optional[str] = None
    use_visual: Optional[bool] = None  # None = auto-decide, True = force, False = skip
    debug_visual: bool = False
    target_language: str = "python"
    deployment_config: Optional[Dict[str, Any]] = None


# Integration function for existing agents
async def enhance_agent_with_visual(agent_service, request: EnhancedAgentRequest):
    """Enhance existing agent service with visual planning"""
    
    # Create visual integration
    visual_integration = AgentVisualIntegration()
    
    # Determine if visual planning should be used
    if request.use_visual is None:
        # Auto-decide based on task complexity
        result = await visual_integration.process_task_with_visual_planning(
            request.task,
            force_visual=False,
            debug=request.debug_visual
        )
    elif request.use_visual:
        # Force visual planning
        result = await visual_integration.process_task_with_visual_planning(
            request.task,
            force_visual=True,
            debug=request.debug_visual
        )
    else:
        # Skip visual planning
        result = {
            "used_visual": False,
            "reasoning": "Visual planning disabled by user"
        }
    
    # If visual planning was successful, use the generated code
    if result.get("success") and result.get("generated_code"):
        # Create a modified request with the visual-generated code
        enhanced_task = f"""
Execute the following professionally planned and validated code:

```{request.target_language}
{result['generated_code']}
```

This code was generated through visual planning and validation.
Task: {request.task}
"""
        
        # Update the original request
        original_request = type('AgentRequest', (), {
            'task': enhanced_task,
            'agent_id': request.agent_id,
            'context': {'deployment': request.deployment_config} if request.deployment_config else {}
        })()
        
        # Process with the enhanced task
        response = await agent_service.process_task_with_knowledge(original_request)
        
        # Add visual planning metadata to response
        if hasattr(response, 'result') and response.result:
            response.result['visual_planning'] = {
                'used': True,
                'complexity': result['analysis'].complexity.value,
                'components': result['analysis'].detected_components,
                'estimated_lines': result['analysis'].estimated_lines,
                'confidence': result['analysis'].confidence
            }
    else:
        # Fallback to normal processing
        original_request = type('AgentRequest', (), {
            'task': request.task,
            'agent_id': request.agent_id,
            'context': {'deployment': request.deployment_config} if request.deployment_config else {}
        })()
        response = await agent_service.process_task_with_knowledge(original_request)
        
        if hasattr(response, 'result') and response.result:
            response.result['visual_planning'] = {
                'used': False,
                'reasoning': result.get('reasoning', 'Visual planning not applicable')
            }
    
    # Log statistics
    print(f"\n📊 Visual Planning Statistics:")
    stats = visual_integration.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return response


# Test the integration
async def test_visual_integration():
    """Test the visual integration system"""
    print("\n" + "="*60)
    print("Testing Agent Visual Integration")
    print("="*60)
    
    integration = AgentVisualIntegration()
    
    # Test cases with different complexity levels
    test_cases = [
        "Print hello world",  # Simple
        "Create a function to calculate fibonacci",  # Simple-Moderate
        "Build a REST API with authentication",  # Complex
        "Design a microservice architecture with message queue",  # Architectural
        "Process CSV file and save to database",  # Moderate with components
    ]
    
    for task in test_cases:
        print(f"\n📝 Task: {task}")
        print("-" * 40)
        
        # Analyze and process
        result = await integration.process_task_with_visual_planning(task)
        
        print(f"   Complexity: {result['analysis'].complexity.value}")
        print(f"   Components: {result['analysis'].detected_components}")
        print(f"   Visual Used: {result['used_visual']}")
        print(f"   Success: {result['success']}")
        
        if result.get('generated_code'):
            lines = result['generated_code'].split('\n')
            print(f"   Generated: {len(lines)} lines of code")
    
    # Show final statistics
    print("\n" + "="*60)
    print("Final Statistics:")
    stats = integration.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    print("Agent Visual Builder Integration")
    print("="*60)
    
    # Run test
    asyncio.run(test_visual_integration())
    
    print("\n✅ Visual Integration ready for agents!")