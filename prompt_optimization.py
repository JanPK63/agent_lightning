"""
Prompt Optimization for Agent Lightning
Automatically improves prompts using reinforcement learning and evolutionary strategies
Implements DSPy-inspired and constitutional AI approaches
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict
import random
import time
from pathlib import Path

# Import Agent Lightning components  
from mdp_agents import MDPAgent, AgentState
from reward_functions import RewardCalculator, RewardConfig


@dataclass
class PromptTemplate:
    """Represents a prompt template with variables"""
    template_id: str
    template: str
    variables: List[str]
    category: str
    performance_score: float = 0.0
    usage_count: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class PromptVariation:
    """A variation of a prompt with specific values"""
    variation_id: str
    base_template: str
    variation: str
    performance: float = 0.0
    reward_history: List[float] = field(default_factory=list)
    generation: int = 0


@dataclass 
class OptimizationConfig:
    """Configuration for prompt optimization"""
    population_size: int = 20
    num_generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    temperature: float = 1.0
    exploration_weight: float = 0.2
    num_evaluations: int = 3


class PromptOptimizer:
    """
    Main prompt optimization system for Agent Lightning
    Uses RL and evolutionary strategies to improve prompts
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize prompt optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Prompt components library
        self.prompt_components = self._build_component_library()
        
        # Template database
        self.templates = self._initialize_templates()
        
        # Optimization history
        self.optimization_history = []
        self.best_prompts = {}
        self.prompt_performances = defaultdict(list)
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Constitutional principles for prompt improvement
        self.principles = self._define_principles()
        
        print(f"🎯 Prompt Optimizer initialized")
        print(f"   Population size: {self.config.population_size}")
        print(f"   Generations: {self.config.num_generations}")
        print(f"   Templates: {len(self.templates)}")
    
    def _build_component_library(self) -> Dict[str, List[str]]:
        """Build library of prompt components"""
        return {
            "instructions": [
                "Think step by step",
                "Let's work through this systematically",
                "Break down the problem",
                "Consider all aspects",
                "Be precise and clear",
                "Explain your reasoning"
            ],
            "roles": [
                "You are a helpful assistant",
                "You are an expert",
                "You are a problem solver",
                "You are a teacher",
                "You are a researcher",
                "You are an analyst"
            ],
            "constraints": [
                "Be concise",
                "Provide detailed explanation",
                "Use examples",
                "Focus on accuracy",
                "Prioritize clarity",
                "Consider edge cases"
            ],
            "output_formats": [
                "Provide your answer in the following format:",
                "Structure your response as:",
                "Output format:",
                "Please format as:",
                "Return the result as:",
                "Present your answer:"
            ],
            "thinking_patterns": [
                "First, identify the key elements",
                "Start by understanding the context",
                "Begin with the main objective",
                "Consider the requirements",
                "Analyze the input",
                "Review the constraints"
            ],
            "validation": [
                "Double-check your answer",
                "Verify the solution",
                "Ensure correctness",
                "Validate your approach",
                "Confirm the result",
                "Review for accuracy"
            ]
        }
    
    def _initialize_templates(self) -> List[PromptTemplate]:
        """Initialize base prompt templates"""
        templates = [
            PromptTemplate(
                template_id="math_solver",
                template="{role}. {instruction}. Given: {problem}. {constraint}. {output_format}",
                variables=["role", "instruction", "problem", "constraint", "output_format"],
                category="math"
            ),
            PromptTemplate(
                template_id="text_analyzer",
                template="{role}. {thinking_pattern}. Text: {text}. Task: {task}. {validation}",
                variables=["role", "thinking_pattern", "text", "task", "validation"],
                category="text"
            ),
            PromptTemplate(
                template_id="code_generator",
                template="{instruction}. Requirements: {requirements}. {constraint}. {output_format}",
                variables=["instruction", "requirements", "constraint", "output_format"],
                category="code"
            ),
            PromptTemplate(
                template_id="reasoning_chain",
                template="{role}. {thinking_pattern}. Problem: {problem}. {instruction}. {validation}",
                variables=["role", "thinking_pattern", "problem", "instruction", "validation"],
                category="reasoning"
            ),
            PromptTemplate(
                template_id="multi_step",
                template="Step 1: {step1}\nStep 2: {step2}\nStep 3: {step3}\n{validation}",
                variables=["step1", "step2", "step3", "validation"],
                category="complex"
            )
        ]
        return templates
    
    def _define_principles(self) -> List[str]:
        """Define constitutional principles for prompt improvement"""
        return [
            "Be helpful and accurate",
            "Avoid ambiguity and confusion",
            "Use clear and precise language",
            "Include necessary context",
            "Follow logical structure",
            "Provide actionable guidance",
            "Maintain consistency",
            "Be appropriately detailed"
        ]
    
    def generate_prompt_variation(self, template: PromptTemplate) -> PromptVariation:
        """
        Generate a variation of a prompt template
        
        Args:
            template: Base template to vary
            
        Returns:
            Prompt variation
        """
        variation_values = {}
        
        for variable in template.variables:
            # Map variable to component category
            if "role" in variable:
                options = self.prompt_components["roles"]
            elif "instruction" in variable:
                options = self.prompt_components["instructions"]
            elif "constraint" in variable:
                options = self.prompt_components["constraints"]
            elif "output_format" in variable:
                options = self.prompt_components["output_formats"]
            elif "thinking" in variable:
                options = self.prompt_components["thinking_patterns"]
            elif "validation" in variable:
                options = self.prompt_components["validation"]
            elif "step" in variable:
                # For multi-step, combine components
                options = (self.prompt_components["thinking_patterns"] + 
                          self.prompt_components["instructions"])
            else:
                # Default to instruction
                options = self.prompt_components["instructions"]
            
            variation_values[variable] = random.choice(options)
        
        # Create variation
        variation_text = template.template
        for var, value in variation_values.items():
            if var in ["problem", "text", "requirements", "task"]:
                # These are task-specific, use placeholder
                variation_text = variation_text.replace(f"{{{var}}}", f"[{var.upper()}]")
            else:
                variation_text = variation_text.replace(f"{{{var}}}", value)
        
        return PromptVariation(
            variation_id=f"{template.template_id}_var_{np.random.randint(10000)}",
            base_template=template.template_id,
            variation=variation_text,
            generation=0
        )
    
    def evaluate_prompt(self,
                       prompt: str,
                       task: Dict[str, Any],
                       agent: Optional[MDPAgent] = None) -> float:
        """
        Evaluate a prompt's performance on a task
        
        Args:
            prompt: Prompt to evaluate
            task: Task to perform
            agent: Agent to use (optional)
            
        Returns:
            Performance score
        """
        if agent is None:
            agent = MDPAgent(role="Evaluator")
        
        # Fill in task-specific variables
        filled_prompt = prompt
        for key, value in task.items():
            placeholder = f"[{key.upper()}]"
            if placeholder in filled_prompt:
                filled_prompt = filled_prompt.replace(placeholder, str(value))
        
        # Execute with agent
        state = agent.observe({
            "input": filled_prompt,
            "context": task,
            "semantic_variables": {}
        })
        
        action, _ = agent.act(state)
        
        # Calculate reward
        ground_truth = task.get("expected_output", "")
        task_type = task.get("type", "general")
        
        reward = self.reward_calculator.calculate_reward(
            action=action.content,
            ground_truth=ground_truth,
            task_type=task_type,
            metadata={"prompt": prompt}
        )
        
        # Additional quality metrics
        quality_score = self._assess_prompt_quality(prompt)
        
        # Combined score
        performance = 0.7 * reward + 0.3 * quality_score
        
        return performance
    
    def _assess_prompt_quality(self, prompt: str) -> float:
        """Assess intrinsic quality of a prompt"""
        score = 0.5  # Base score
        
        # Check for clarity indicators
        if any(word in prompt.lower() for word in ["step", "first", "then", "finally"]):
            score += 0.1
        
        # Check for specificity
        if any(word in prompt.lower() for word in ["specifically", "exactly", "precisely"]):
            score += 0.1
        
        # Check for structure
        if "\n" in prompt or ":" in prompt:
            score += 0.05
        
        # Penalize for being too long
        if len(prompt) > 500:
            score -= 0.1
        elif len(prompt) < 50:
            score -= 0.05
        
        # Check for validation
        if any(word in prompt.lower() for word in ["check", "verify", "ensure"]):
            score += 0.1
        
        # Check against principles
        principle_adherence = sum(
            1 for principle in self.principles
            if any(word in prompt.lower() for word in principle.lower().split())
        ) / len(self.principles)
        
        score += 0.15 * principle_adherence
        
        return min(1.0, max(0.0, score))
    
    def optimize_prompt_evolutionary(self,
                                    template: PromptTemplate,
                                    tasks: List[Dict[str, Any]]) -> PromptVariation:
        """
        Optimize prompt using evolutionary algorithm
        
        Args:
            template: Base template to optimize
            tasks: Tasks to optimize for
            
        Returns:
            Best prompt variation
        """
        print(f"\n🧬 Optimizing prompt: {template.template_id}")
        
        # Initialize population
        population = [
            self.generate_prompt_variation(template)
            for _ in range(self.config.population_size)
        ]
        
        best_overall = None
        
        for generation in range(self.config.num_generations):
            # Evaluate population
            for variation in population:
                if len(variation.reward_history) == 0:  # Not evaluated yet
                    scores = []
                    for task in tasks[:self.config.num_evaluations]:
                        score = self.evaluate_prompt(variation.variation, task)
                        scores.append(score)
                    
                    variation.performance = np.mean(scores)
                    variation.reward_history = scores
                    variation.generation = generation
            
            # Sort by performance
            population.sort(key=lambda x: x.performance, reverse=True)
            
            # Track best
            if best_overall is None or population[0].performance > best_overall.performance:
                best_overall = population[0]
            
            print(f"  Generation {generation}: Best = {population[0].performance:.3f}")
            
            # Selection and reproduction
            if generation < self.config.num_generations - 1:
                # Keep top performers
                elite_size = self.config.population_size // 4
                new_population = population[:elite_size]
                
                # Generate offspring
                while len(new_population) < self.config.population_size:
                    # Tournament selection
                    parent1 = self._tournament_select(population)
                    parent2 = self._tournament_select(population)
                    
                    # Crossover
                    if random.random() < self.config.crossover_rate:
                        child = self._crossover(parent1, parent2, template)
                    else:
                        child = parent1
                    
                    # Mutation
                    if random.random() < self.config.mutation_rate:
                        child = self._mutate(child, template)
                    
                    new_population.append(child)
                
                population = new_population
        
        # Store in history
        self.optimization_history.append({
            "template_id": template.template_id,
            "best_variation": best_overall,
            "final_performance": best_overall.performance,
            "generations": self.config.num_generations
        })
        
        return best_overall
    
    def _tournament_select(self, population: List[PromptVariation], 
                          tournament_size: int = 3) -> PromptVariation:
        """Tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.performance)
    
    def _crossover(self, parent1: PromptVariation, parent2: PromptVariation,
                  template: PromptTemplate) -> PromptVariation:
        """Crossover two prompt variations"""
        # Extract components from parents
        components1 = self._extract_components(parent1.variation)
        components2 = self._extract_components(parent2.variation)
        
        # Mix components
        child_components = []
        for c1, c2 in zip(components1, components2):
            if random.random() < 0.5:
                child_components.append(c1)
            else:
                child_components.append(c2)
        
        # Reconstruct prompt
        child_text = ". ".join(child_components)
        
        return PromptVariation(
            variation_id=f"{template.template_id}_child_{np.random.randint(10000)}",
            base_template=template.template_id,
            variation=child_text,
            generation=max(parent1.generation, parent2.generation) + 1
        )
    
    def _mutate(self, variation: PromptVariation, 
               template: PromptTemplate) -> PromptVariation:
        """Mutate a prompt variation"""
        components = self._extract_components(variation.variation)
        
        if components and random.random() < 0.5:
            # Replace a component
            idx = random.randint(0, len(components) - 1)
            component_type = random.choice(list(self.prompt_components.keys()))
            components[idx] = random.choice(self.prompt_components[component_type])
        else:
            # Add or remove a component
            if random.random() < 0.5 and len(components) > 2:
                # Remove
                components.pop(random.randint(0, len(components) - 1))
            else:
                # Add
                component_type = random.choice(list(self.prompt_components.keys()))
                new_component = random.choice(self.prompt_components[component_type])
                components.insert(random.randint(0, len(components)), new_component)
        
        mutated_text = ". ".join(components)
        
        return PromptVariation(
            variation_id=f"{variation.variation_id}_mut",
            base_template=variation.base_template,
            variation=mutated_text,
            generation=variation.generation
        )
    
    def _extract_components(self, prompt: str) -> List[str]:
        """Extract components from a prompt"""
        # Split by sentences and major separators
        components = re.split(r'[.!?\n]+', prompt)
        return [c.strip() for c in components if c.strip()]
    
    def optimize_with_reinforcement_learning(self,
                                            template: PromptTemplate,
                                            tasks: List[Dict[str, Any]],
                                            num_iterations: int = 50) -> PromptVariation:
        """
        Optimize prompt using reinforcement learning
        
        Args:
            template: Base template
            tasks: Tasks to optimize for
            num_iterations: Number of RL iterations
            
        Returns:
            Optimized prompt variation
        """
        print(f"\n🤖 RL Optimization for: {template.template_id}")
        
        # Initialize Q-values for prompt components
        q_values = defaultdict(lambda: defaultdict(float))
        
        # Current best
        best_variation = self.generate_prompt_variation(template)
        best_reward = 0
        
        # Exploration parameters
        epsilon = 1.0
        epsilon_decay = 0.95
        alpha = 0.1  # Learning rate
        
        for iteration in range(num_iterations):
            # Epsilon-greedy selection
            if random.random() < epsilon:
                # Explore: generate random variation
                variation = self.generate_prompt_variation(template)
            else:
                # Exploit: use best known components
                variation = self._generate_from_q_values(template, q_values)
            
            # Evaluate on tasks
            rewards = []
            for task in random.sample(tasks, min(3, len(tasks))):
                reward = self.evaluate_prompt(variation.variation, task)
                rewards.append(reward)
            
            avg_reward = np.mean(rewards)
            variation.performance = avg_reward
            variation.reward_history = rewards
            
            # Update Q-values
            components = self._extract_components(variation.variation)
            for i, component in enumerate(components):
                old_q = q_values[i][component]
                q_values[i][component] = old_q + alpha * (avg_reward - old_q)
            
            # Track best
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_variation = variation
            
            # Decay exploration
            epsilon *= epsilon_decay
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Best reward = {best_reward:.3f}")
        
        return best_variation
    
    def _generate_from_q_values(self, template: PromptTemplate,
                               q_values: Dict) -> PromptVariation:
        """Generate variation using Q-values"""
        components = []
        
        for i in range(len(template.variables)):
            if i in q_values and q_values[i]:
                # Select component with highest Q-value
                best_component = max(q_values[i].items(), key=lambda x: x[1])[0]
                components.append(best_component)
            else:
                # Random selection if no Q-values
                component_type = random.choice(list(self.prompt_components.keys()))
                components.append(random.choice(self.prompt_components[component_type]))
        
        variation_text = ". ".join(components)
        
        return PromptVariation(
            variation_id=f"{template.template_id}_rl_{np.random.randint(10000)}",
            base_template=template.template_id,
            variation=variation_text
        )
    
    def chain_of_thought_optimization(self, 
                                     base_prompt: str,
                                     task: Dict[str, Any]) -> str:
        """
        Optimize prompt by adding chain-of-thought reasoning
        
        Args:
            base_prompt: Original prompt
            task: Task to optimize for
            
        Returns:
            Optimized prompt with CoT
        """
        cot_templates = [
            "Let's think step by step:\n{base}\nShow your reasoning process.",
            "Break this down:\n1. Understand the problem\n2. Plan the approach\n3. Execute\n{base}",
            "Before answering, consider:\n- What is being asked?\n- What information is given?\n- What method should be used?\n{base}",
            "{base}\nExplain your thought process before providing the final answer.",
            "Step-by-step solution:\n{base}\nShow all intermediate steps."
        ]
        
        best_prompt = base_prompt
        best_score = self.evaluate_prompt(base_prompt, task)
        
        for cot_template in cot_templates:
            cot_prompt = cot_template.replace("{base}", base_prompt)
            score = self.evaluate_prompt(cot_prompt, task)
            
            if score > best_score:
                best_score = score
                best_prompt = cot_prompt
        
        return best_prompt
    
    def constitutional_refinement(self, prompt: str) -> str:
        """
        Refine prompt using constitutional AI principles
        
        Args:
            prompt: Prompt to refine
            
        Returns:
            Refined prompt
        """
        refinements = []
        
        for principle in self.principles:
            if "helpful" in principle.lower() and "helpful" not in prompt.lower():
                refinements.append("Be helpful and provide useful information.")
            elif "accurate" in principle.lower() and "accurate" not in prompt.lower():
                refinements.append("Ensure accuracy in your response.")
            elif "clear" in principle.lower() and "clear" not in prompt.lower():
                refinements.append("Use clear and precise language.")
            elif "logical" in principle.lower() and "logical" not in prompt.lower():
                refinements.append("Follow a logical structure.")
        
        if refinements:
            refined_prompt = prompt + "\n\nAdditional guidelines:\n" + "\n".join(refinements)
        else:
            refined_prompt = prompt
        
        return refined_prompt
    
    def auto_prompt_engineer(self,
                            task_description: str,
                            examples: List[Dict[str, str]] = None) -> str:
        """
        Automatically engineer a prompt for a task
        
        Args:
            task_description: Description of the task
            examples: Optional examples
            
        Returns:
            Engineered prompt
        """
        # Analyze task
        task_type = self._classify_task(task_description)
        
        # Select appropriate template
        template = None
        for t in self.templates:
            if t.category == task_type:
                template = t
                break
        
        if template is None:
            template = self.templates[0]  # Default
        
        # Generate base prompt
        base_variation = self.generate_prompt_variation(template)
        
        # Add examples if provided
        if examples:
            example_text = "\n\nExamples:\n"
            for i, ex in enumerate(examples[:3], 1):
                example_text += f"{i}. Input: {ex.get('input', '')}\n   Output: {ex.get('output', '')}\n"
            
            enhanced_prompt = base_variation.variation + example_text
        else:
            enhanced_prompt = base_variation.variation
        
        # Add task description
        final_prompt = enhanced_prompt.replace("[TASK]", task_description)
        
        # Constitutional refinement
        final_prompt = self.constitutional_refinement(final_prompt)
        
        return final_prompt
    
    def _classify_task(self, description: str) -> str:
        """Classify task type from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["math", "calculate", "solve", "equation"]):
            return "math"
        elif any(word in description_lower for word in ["code", "program", "function", "implement"]):
            return "code"
        elif any(word in description_lower for word in ["text", "write", "summarize", "explain"]):
            return "text"
        elif any(word in description_lower for word in ["reason", "logic", "deduce", "infer"]):
            return "reasoning"
        else:
            return "complex"
    
    def batch_optimize(self, tasks: List[Dict[str, Any]]) -> Dict[str, PromptVariation]:
        """
        Optimize prompts for multiple task types
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dictionary of optimized prompts by category
        """
        optimized = {}
        
        # Group tasks by type
        tasks_by_type = defaultdict(list)
        for task in tasks:
            task_type = task.get("type", "general")
            tasks_by_type[task_type].append(task)
        
        # Optimize for each type
        for task_type, type_tasks in tasks_by_type.items():
            # Find matching template
            template = None
            for t in self.templates:
                if t.category == task_type:
                    template = t
                    break
            
            if template:
                # Optimize using both evolutionary and RL
                evo_best = self.optimize_prompt_evolutionary(template, type_tasks)
                rl_best = self.optimize_with_reinforcement_learning(template, type_tasks)
                
                # Choose better one
                if evo_best.performance > rl_best.performance:
                    optimized[task_type] = evo_best
                else:
                    optimized[task_type] = rl_best
                
                self.best_prompts[task_type] = optimized[task_type]
        
        return optimized
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            "total_optimizations": len(self.optimization_history),
            "best_prompts_by_category": {},
            "average_improvements": {},
            "optimization_methods_used": ["evolutionary", "reinforcement_learning", "chain_of_thought"],
            "principles_applied": self.principles
        }
        
        # Best prompts
        for category, prompt_var in self.best_prompts.items():
            report["best_prompts_by_category"][category] = {
                "prompt": prompt_var.variation,
                "performance": prompt_var.performance,
                "generation": prompt_var.generation
            }
        
        # Calculate improvements
        for opt in self.optimization_history:
            template_id = opt["template_id"]
            improvement = opt["final_performance"]
            
            if template_id not in report["average_improvements"]:
                report["average_improvements"][template_id] = []
            
            report["average_improvements"][template_id].append(improvement)
        
        # Average the improvements
        for template_id, improvements in report["average_improvements"].items():
            report["average_improvements"][template_id] = np.mean(improvements)
        
        return report


# Example usage
if __name__ == "__main__":
    print("🎯 Testing Prompt Optimization System")
    print("=" * 60)
    
    # Initialize optimizer
    config = OptimizationConfig(
        population_size=10,
        num_generations=5,
        mutation_rate=0.15,
        num_evaluations=3
    )
    
    optimizer = PromptOptimizer(config)
    
    # Define test tasks
    test_tasks = [
        {
            "type": "math",
            "problem": "Solve: 2x + 5 = 13",
            "expected_output": "x = 4"
        },
        {
            "type": "text",
            "text": "The quick brown fox jumps over the lazy dog",
            "task": "Count the words",
            "expected_output": "9"
        },
        {
            "type": "code",
            "requirements": "Write a function to check if a number is even",
            "expected_output": "def is_even(n): return n % 2 == 0"
        }
    ]
    
    # Test evolutionary optimization
    print("\n🧬 Testing Evolutionary Optimization...")
    math_template = optimizer.templates[0]  # Math solver template
    best_evo = optimizer.optimize_prompt_evolutionary(math_template, test_tasks)
    
    print(f"\nBest Evolutionary Prompt:")
    print(f"  Performance: {best_evo.performance:.3f}")
    print(f"  Prompt: {best_evo.variation[:200]}...")
    
    # Test RL optimization
    print("\n🤖 Testing RL Optimization...")
    best_rl = optimizer.optimize_with_reinforcement_learning(
        math_template, test_tasks, num_iterations=20
    )
    
    print(f"\nBest RL Prompt:")
    print(f"  Performance: {best_rl.performance:.3f}")
    print(f"  Prompt: {best_rl.variation[:200]}...")
    
    # Test auto prompt engineering
    print("\n🔧 Testing Auto Prompt Engineering...")
    task_desc = "Solve mathematical equations step by step"
    examples = [
        {"input": "x + 3 = 7", "output": "x = 4"},
        {"input": "2y = 10", "output": "y = 5"}
    ]
    
    auto_prompt = optimizer.auto_prompt_engineer(task_desc, examples)
    print(f"\nAuto-Engineered Prompt:")
    print(auto_prompt[:300] + "...")
    
    # Test chain-of-thought optimization
    print("\n💭 Testing Chain-of-Thought Optimization...")
    base = "Solve the problem"
    cot_prompt = optimizer.chain_of_thought_optimization(base, test_tasks[0])
    print(f"\nCoT-Optimized Prompt:")
    print(cot_prompt)
    
    # Batch optimization
    print("\n📦 Testing Batch Optimization...")
    optimized_prompts = optimizer.batch_optimize(test_tasks)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION REPORT")
    print("=" * 60)
    
    report = optimizer.generate_optimization_report()
    
    print(f"\nTotal Optimizations: {report['total_optimizations']}")
    print(f"Methods Used: {', '.join(report['optimization_methods_used'])}")
    
    if report['best_prompts_by_category']:
        print("\nBest Prompts by Category:")
        for category, info in report['best_prompts_by_category'].items():
            print(f"\n  {category}:")
            print(f"    Performance: {info['performance']:.3f}")
            print(f"    Generation: {info['generation']}")
            print(f"    Prompt: {info['prompt'][:100]}...")
    
    if report['average_improvements']:
        print("\nAverage Improvements:")
        for template_id, improvement in report['average_improvements'].items():
            print(f"  {template_id}: {improvement:.3f}")
    
    print("\n✅ Prompt optimization test complete!")