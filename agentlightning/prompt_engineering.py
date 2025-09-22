#!/usr/bin/env python3
"""
Dynamic Prompt Engineering for Agent Lightning

This module provides advanced prompt engineering capabilities including:
- Dynamic prompt generation based on task analysis
- Chain-of-thought prompting
- Few-shot learning with dynamic examples
- Prompt optimization and A/B testing
- Context-aware prompt adaptation
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Different prompt engineering strategies"""
    DIRECT = "direct"  # Simple direct instruction
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    FEW_SHOT = "few_shot"  # Learning from examples
    ZERO_SHOT_COT = "zero_shot_cot"  # Zero-shot chain of thought
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # Multiple reasoning paths
    SELF_CONSISTENCY = "self_consistency"  # Multiple attempts for consistency


@dataclass
class PromptTemplate:
    """A dynamic prompt template"""
    template_id: str
    name: str
    strategy: PromptStrategy
    base_template: str
    variables: List[str]
    examples: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PromptContext:
    """Context information for prompt generation"""
    task_description: str
    input_data: Any
    domain: Optional[str] = None
    complexity: Optional[str] = None  # "simple", "medium", "complex"
    constraints: List[str] = None
    previous_attempts: List[Dict[str, Any]] = None
    performance_history: List[float] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.previous_attempts is None:
            self.previous_attempts = []
        if self.performance_history is None:
            self.performance_history = []


class PromptGenerator(ABC):
    """Abstract base class for prompt generators"""

    @abstractmethod
    def generate_prompt(self, context: PromptContext) -> str:
        """Generate a prompt based on context"""
        pass

    @abstractmethod
    def get_strategy(self) -> PromptStrategy:
        """Get the prompt strategy used by this generator"""
        pass


class DirectPromptGenerator(PromptGenerator):
    """Simple direct instruction prompt generator"""

    def get_strategy(self) -> PromptStrategy:
        return PromptStrategy.DIRECT

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate a direct instruction prompt"""
        prompt = f"Task: {context.task_description}\n\n"

        if context.input_data:
            prompt += f"Input: {context.input_data}\n\n"

        if context.constraints:
            prompt += f"Constraints: {'; '.join(context.constraints)}\n\n"

        prompt += "Please provide a helpful and accurate response."

        return prompt


class ChainOfThoughtGenerator(PromptGenerator):
    """Chain-of-thought prompting generator"""

    def get_strategy(self) -> PromptStrategy:
        return PromptStrategy.CHAIN_OF_THOUGHT

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate a chain-of-thought prompt"""
        prompt = f"Task: {context.task_description}\n\n"

        if context.input_data:
            prompt += f"Input: {context.input_data}\n\n"

        # Add chain-of-thought instruction
        prompt += "Let's solve this step by step:\n\n"

        # Add complexity-based guidance
        if context.complexity == "complex":
            prompt += "1. First, break down the problem into smaller components.\n"
            prompt += "2. Analyze each component systematically.\n"
            prompt += "3. Consider edge cases and constraints.\n"
            prompt += "4. Synthesize the solution from the analysis.\n\n"
        elif context.complexity == "medium":
            prompt += "1. Understand the key requirements.\n"
            prompt += "2. Identify the main approach.\n"
            prompt += "3. Work through the solution step by step.\n\n"
        else:
            prompt += "1. Read and understand the input.\n"
            prompt += "2. Apply the required logic.\n"
            prompt += "3. Provide the final answer.\n\n"

        if context.constraints:
            prompt += f"Remember these constraints: {'; '.join(context.constraints)}\n\n"

        prompt += "Final Answer:"

        return prompt


class FewShotGenerator(PromptGenerator):
    """Few-shot learning prompt generator"""

    def __init__(self, example_database: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.example_database = example_database or {}
        self.max_examples = 3

    def get_strategy(self) -> PromptStrategy:
        return PromptStrategy.FEW_SHOT

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate a few-shot prompt with relevant examples"""
        prompt = f"Task: {context.task_description}\n\n"

        # Find relevant examples
        relevant_examples = self._find_relevant_examples(context)

        if relevant_examples:
            prompt += "Examples:\n\n"
            for i, example in enumerate(relevant_examples[:self.max_examples], 1):
                prompt += f"Example {i}:\n"
                prompt += f"Input: {example.get('input', '')}\n"
                prompt += f"Output: {example.get('output', '')}\n\n"

        prompt += "Now solve this:\n\n"
        if context.input_data:
            prompt += f"Input: {context.input_data}\n\n"

        if context.constraints:
            prompt += f"Constraints: {'; '.join(context.constraints)}\n\n"

        prompt += "Output:"

        return prompt

    def _find_relevant_examples(self, context: PromptContext) -> List[Dict[str, Any]]:
        """Find examples relevant to the current context"""
        # Simple keyword matching - can be enhanced with embeddings
        task_keywords = self._extract_keywords(context.task_description)
        relevant_examples = []

        for domain_examples in self.example_database.values():
            for example in domain_examples:
                example_text = f"{example.get('input', '')} {example.get('output', '')}"
                example_keywords = self._extract_keywords(example_text)

                # Check for keyword overlap
                if set(task_keywords) & set(example_keywords):
                    relevant_examples.append(example)

        return relevant_examples

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - remove common words and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates


class ZeroShotCoTGenerator(PromptGenerator):
    """Zero-shot chain-of-thought generator"""

    def get_strategy(self) -> PromptStrategy:
        return PromptStrategy.ZERO_SHOT_COT

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate a zero-shot chain-of-thought prompt"""
        prompt = f"Task: {context.task_description}\n\n"

        if context.input_data:
            prompt += f"Input: {context.input_data}\n\n"

        # Zero-shot CoT instruction
        prompt += "Let's think step by step to solve this problem.\n\n"

        if context.constraints:
            prompt += f"Consider these constraints: {'; '.join(context.constraints)}\n\n"

        prompt += "Step-by-step reasoning:"

        return prompt


class DynamicPromptEngineer:
    """Main dynamic prompt engineering orchestrator"""

    def __init__(self):
        self.generators: Dict[PromptStrategy, PromptGenerator] = {}
        self.templates: Dict[str, PromptTemplate] = {}
        self.performance_tracker: Dict[str, List[float]] = {}
        self.ab_tests: Dict[str, Dict[str, Any]] = {}

        # Initialize default generators
        self._register_generator(DirectPromptGenerator())
        self._register_generator(ChainOfThoughtGenerator())
        self._register_generator(FewShotGenerator())
        self._register_generator(ZeroShotCoTGenerator())

    def _register_generator(self, generator: PromptGenerator):
        """Register a prompt generator"""
        self.generators[generator.get_strategy()] = generator

    def add_template(self, template: PromptTemplate):
        """Add a custom prompt template"""
        self.templates[template.template_id] = template

    def analyze_task(self, task_description: str, input_data: Any = None) -> PromptContext:
        """Analyze a task to determine optimal prompt strategy"""
        context = PromptContext(
            task_description=task_description,
            input_data=input_data
        )

        # Determine domain
        if any(keyword in task_description.lower() for keyword in ['code', 'programming', 'function']):
            context.domain = 'programming'
        elif any(keyword in task_description.lower() for keyword in ['math', 'calculate', 'equation']):
            context.domain = 'mathematics'
        elif any(keyword in task_description.lower() for keyword in ['analyze', 'review', 'evaluate']):
            context.domain = 'analysis'
        else:
            context.domain = 'general'

        # Determine complexity
        word_count = len(task_description.split())
        if word_count > 50 or any(keyword in task_description.lower() for keyword in ['complex', 'advanced', 'multiple steps']):
            context.complexity = 'complex'
        elif word_count > 20:
            context.complexity = 'medium'
        else:
            context.complexity = 'simple'

        # Extract constraints
        constraint_patterns = [
            r'requirements?:?\s*(.*?)(?:\n|$)',
            r'constraints?:?\s*(.*?)(?:\n|$)',
            r'must\s+(.*?)(?:\n|$)',
            r'should\s+(.*?)(?:\n|$)',
        ]

        for pattern in constraint_patterns:
            matches = re.findall(pattern, task_description, re.IGNORECASE)
            context.constraints.extend(matches)

        return context

    def select_strategy(self, context: PromptContext) -> PromptStrategy:
        """Select the best prompt strategy for a given context"""
        # Strategy selection logic
        if context.complexity == 'complex':
            if context.domain in ['mathematics', 'programming']:
                return PromptStrategy.CHAIN_OF_THOUGHT
            else:
                return PromptStrategy.ZERO_SHOT_COT
        elif context.complexity == 'medium':
            return PromptStrategy.CHAIN_OF_THOUGHT
        else:
            # For simple tasks, check if we have relevant examples
            few_shot_gen = self.generators.get(PromptStrategy.FEW_SHOT)
            if isinstance(few_shot_gen, FewShotGenerator):
                examples = few_shot_gen._find_relevant_examples(context)
                if len(examples) >= 2:
                    return PromptStrategy.FEW_SHOT

            return PromptStrategy.DIRECT

    def generate_dynamic_prompt(self, task_description: str, input_data: Any = None,
                               strategy: Optional[PromptStrategy] = None) -> Tuple[str, PromptStrategy]:
        """Generate a dynamic prompt based on task analysis"""
        context = self.analyze_task(task_description, input_data)

        if strategy is None:
            strategy = self.select_strategy(context)

        generator = self.generators.get(strategy)
        if not generator:
            # Fallback to direct
            generator = self.generators[PromptStrategy.DIRECT]

        prompt = generator.generate_prompt(context)

        logger.info(f"Generated {strategy.value} prompt for {context.domain} task (complexity: {context.complexity})")

        return prompt, strategy

    def optimize_prompt(self, base_prompt: str, task_context: PromptContext,
                       iterations: int = 3) -> List[str]:
        """Generate multiple optimized variations of a prompt"""
        variations = [base_prompt]

        # Add chain-of-thought variation
        if "step by step" not in base_prompt.lower():
            cot_variation = base_prompt + "\n\nLet's think step by step:"
            variations.append(cot_variation)

        # Add few-shot variation if examples available
        few_shot_gen = self.generators.get(PromptStrategy.FEW_SHOT)
        if isinstance(few_shot_gen, FewShotGenerator):
            examples = few_shot_gen._find_relevant_examples(task_context)
            if examples:
                few_shot_prompt = few_shot_gen.generate_prompt(task_context)
                variations.append(few_shot_prompt)

        # Add constraint emphasis variation
        if task_context.constraints:
            constraint_variation = base_prompt + f"\n\nImportant constraints to follow: {'; '.join(task_context.constraints)}"
            variations.append(constraint_variation)

        return variations[:iterations]

    def start_ab_test(self, test_id: str, prompts: List[str], task_description: str):
        """Start A/B testing for different prompts"""
        self.ab_tests[test_id] = {
            'prompts': prompts,
            'task_description': task_description,
            'results': [[] for _ in prompts],  # Performance scores for each prompt
            'active': True
        }
        logger.info(f"Started A/B test {test_id} with {len(prompts)} prompt variations")

    def record_ab_test_result(self, test_id: str, prompt_index: int, performance: float):
        """Record performance result for A/B test"""
        if test_id in self.ab_tests:
            test_data = self.ab_tests[test_id]
            if 0 <= prompt_index < len(test_data['results']):
                test_data['results'][prompt_index].append(performance)

    def get_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get results from A/B test"""
        if test_id not in self.ab_tests:
            return None

        test_data = self.ab_tests[test_id]
        results = []

        for i, performances in enumerate(test_data['results']):
            if performances:
                avg_performance = sum(performances) / len(performances)
                results.append({
                    'prompt_index': i,
                    'average_performance': avg_performance,
                    'sample_count': len(performances)
                })

        return {
            'test_id': test_id,
            'task_description': test_data['task_description'],
            'results': results,
            'best_prompt_index': max(range(len(results)), key=lambda i: results[i]['average_performance']) if results else None
        }

    def add_examples(self, domain: str, examples: List[Dict[str, Any]]):
        """Add examples to the few-shot generator"""
        few_shot_gen = self.generators.get(PromptStrategy.FEW_SHOT)
        if isinstance(few_shot_gen, FewShotGenerator):
            if domain not in few_shot_gen.example_database:
                few_shot_gen.example_database[domain] = []
            few_shot_gen.example_database[domain].extend(examples)
            logger.info(f"Added {len(examples)} examples to {domain} domain")


# Global prompt engineer instance
prompt_engineer = DynamicPromptEngineer()

# Initialize with some example data
prompt_engineer.add_examples('programming', [
    {'input': 'Write a function to reverse a string', 'output': 'def reverse_string(s): return s[::-1]'},
    {'input': 'Calculate factorial of n', 'output': 'def factorial(n): return 1 if n <= 1 else n * factorial(n-1)'},
])

prompt_engineer.add_examples('mathematics', [
    {'input': 'Solve 2x + 3 = 7', 'output': '2x = 4, x = 2'},
    {'input': 'Area of circle with radius r', 'output': 'A = πr²'},
])


def generate_optimal_prompt(task: str, input_data: Any = None,
                           strategy: Optional[PromptStrategy] = None) -> str:
    """Convenience function to generate an optimal prompt"""
    prompt, _ = prompt_engineer.generate_dynamic_prompt(task, input_data, strategy)
    return prompt


def create_prompt_variations(base_prompt: str, context: PromptContext,
                           count: int = 3) -> List[str]:
    """Create multiple variations of a prompt for testing"""
    return prompt_engineer.optimize_prompt(base_prompt, context, count)