"""
Training Dataset Creation for Agent Lightning
Creates JSONL formatted datasets with ground truth for different task types
Following Agent Lightning's data interface specifications
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class TrainingDatasetCreator:
    """Creates diverse training datasets for multi-agent RL training"""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize dataset creator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Task type generators
        self.task_generators = {
            "math": self.generate_math_tasks,
            "text": self.generate_text_tasks,
            "code": self.generate_code_tasks,
            "multi_agent": self.generate_multi_agent_tasks,
            "rag": self.generate_rag_tasks,
            "tool_use": self.generate_tool_use_tasks
        }
        
        print(f"📁 Dataset creator initialized. Output directory: {self.output_dir}")
    
    def create_comprehensive_dataset(self, samples_per_type: int = 100) -> Path:
        """
        Create a comprehensive training dataset with multiple task types
        Following Agent Lightning's approach to diverse task training
        """
        all_samples = []
        
        # Generate samples for each task type
        for task_type, generator in self.task_generators.items():
            print(f"\n🔨 Generating {samples_per_type} {task_type} samples...")
            samples = generator(samples_per_type)
            all_samples.extend(samples)
            print(f"   ✓ Generated {len(samples)} {task_type} samples")
        
        # Shuffle for better training distribution
        random.shuffle(all_samples)
        
        # Save to JSONL format
        train_file = self.output_dir / "train.jsonl"
        val_file = self.output_dir / "val.jsonl"
        
        # Split 80/20 for train/val
        split_idx = int(len(all_samples) * 0.8)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        # Write training data
        with open(train_file, 'w') as f:
            for sample in train_samples:
                f.write(json.dumps(sample) + '\n')
        
        # Write validation data
        with open(val_file, 'w') as f:
            for sample in val_samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"\n✅ Dataset creation complete!")
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Validation samples: {len(val_samples)}")
        print(f"   Files: {train_file}, {val_file}")
        
        return train_file
    
    def generate_math_tasks(self, n: int) -> List[Dict]:
        """Generate mathematical reasoning tasks with hierarchical structure"""
        samples = []
        
        for i in range(n):
            # Generate problem parameters
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = random.randint(1, 50)
            
            problem_types = [
                {
                    "type": "algebra",
                    "problem": f"Solve for x: {a}x + {b} = {c}",
                    "solution": f"x = {(c - b) / a:.2f}",
                    "steps": [
                        f"Subtract {b} from both sides: {a}x = {c - b}",
                        f"Divide by {a}: x = {(c - b) / a:.2f}"
                    ]
                },
                {
                    "type": "arithmetic",
                    "problem": f"Calculate: ({a} + {b}) * {c}",
                    "solution": str((a + b) * c),
                    "steps": [
                        f"First add: {a} + {b} = {a + b}",
                        f"Then multiply: {a + b} * {c} = {(a + b) * c}"
                    ]
                },
                {
                    "type": "word_problem",
                    "problem": f"A store has {a} apples. They receive {b} more apples and sell {c}. How many are left?",
                    "solution": str(a + b - c),
                    "steps": [
                        f"Initial: {a} apples",
                        f"After receiving: {a} + {b} = {a + b}",
                        f"After selling: {a + b} - {c} = {a + b - c}"
                    ]
                }
            ]
            
            task = random.choice(problem_types)
            
            sample = {
                "task_id": f"math_{i:04d}",
                "task_type": "solve_math",
                "hierarchy_level": "high" if task["type"] == "word_problem" else "low",
                "messages": [
                    {"role": "user", "content": task["problem"]},
                    {"role": "assistant", "content": f"Let me solve this step by step:\n" + "\n".join(task["steps"]) + f"\n\nAnswer: {task['solution']}"}
                ],
                "ground_truth": task["solution"],
                "reward": 1.0,
                "subtasks": ["understand", "calculate", "verify"] if task["type"] == "word_problem" else ["calculate"],
                "quality_metrics": {
                    "accuracy": 1.0,
                    "completeness": 0.9,
                    "clarity": 0.85
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_text_tasks(self, n: int) -> List[Dict]:
        """Generate text generation and summarization tasks"""
        samples = []
        
        topics = [
            "artificial intelligence", "climate change", "space exploration",
            "quantum computing", "renewable energy", "biotechnology",
            "machine learning", "robotics", "nanotechnology", "blockchain"
        ]
        
        task_types = ["summarize", "explain", "compare", "analyze"]
        
        for i in range(n):
            topic = random.choice(topics)
            task_type = random.choice(task_types)
            
            if task_type == "summarize":
                prompt = f"Summarize the key concepts of {topic} in 3-5 sentences"
                response = f"{topic.capitalize()} is a transformative field that... [comprehensive summary]"
            elif task_type == "explain":
                prompt = f"Explain how {topic} works to a general audience"
                response = f"{topic.capitalize()} works by... [detailed explanation]"
            elif task_type == "compare":
                topic2 = random.choice([t for t in topics if t != topic])
                prompt = f"Compare and contrast {topic} with {topic2}"
                response = f"Both {topic} and {topic2} are important technologies... [comparison]"
            else:  # analyze
                prompt = f"Analyze the potential impact of {topic} on society"
                response = f"The impact of {topic} on society could be profound... [analysis]"
            
            sample = {
                "task_id": f"text_{i:04d}",
                "task_type": f"text_{task_type}",
                "hierarchy_level": "high",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "ground_truth": f"High-quality {task_type} about {topic}",
                "reward": 0.85 + random.random() * 0.15,
                "quality_metrics": {
                    "completeness": 0.8 + random.random() * 0.2,
                    "accuracy": 0.85 + random.random() * 0.15,
                    "coherence": 0.9 + random.random() * 0.1,
                    "conciseness": 0.7 + random.random() * 0.3
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_code_tasks(self, n: int) -> List[Dict]:
        """Generate code generation and debugging tasks"""
        samples = []
        
        code_tasks = [
            {
                "description": "Write a function to check if a number is prime",
                "function_name": "is_prime",
                "test_cases": [(2, True), (4, False), (17, True), (1, False)],
                "difficulty": "easy"
            },
            {
                "description": "Implement a function to reverse a linked list",
                "function_name": "reverse_linked_list",
                "test_cases": [([1,2,3], [3,2,1]), ([1], [1])],
                "difficulty": "medium"
            },
            {
                "description": "Create a function to find the longest palindrome in a string",
                "function_name": "longest_palindrome",
                "test_cases": [("babad", "bab"), ("cbbd", "bb")],
                "difficulty": "hard"
            }
        ]
        
        for i in range(n):
            task = random.choice(code_tasks)
            
            sample = {
                "task_id": f"code_{i:04d}",
                "task_type": "code_generation",
                "hierarchy_level": "low" if task["difficulty"] == "easy" else "high",
                "messages": [
                    {"role": "user", "content": task["description"]},
                    {"role": "assistant", "content": f"```python\ndef {task['function_name']}(...):\n    # Implementation\n    pass\n```"}
                ],
                "ground_truth": f"Working {task['function_name']} implementation",
                "reward": 0.9,
                "test_cases": task["test_cases"],
                "quality_metrics": {
                    "correctness": 1.0,
                    "efficiency": 0.8 + random.random() * 0.2,
                    "readability": 0.85 + random.random() * 0.15
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_multi_agent_tasks(self, n: int) -> List[Dict]:
        """Generate tasks requiring multi-agent collaboration"""
        samples = []
        
        collaboration_scenarios = [
            {
                "scenario": "Research and write a technical article",
                "agents": ["researcher", "writer", "reviewer"],
                "topic": "emerging technology trends"
            },
            {
                "scenario": "Analyze and solve a complex business problem",
                "agents": ["analyst", "strategist", "optimizer"],
                "topic": "market expansion strategy"
            },
            {
                "scenario": "Design and implement a software feature",
                "agents": ["designer", "developer", "tester"],
                "topic": "user authentication system"
            }
        ]
        
        for i in range(n):
            scenario = random.choice(collaboration_scenarios)
            
            # Generate multi-turn conversation
            messages = []
            agent_rewards = {}
            
            for agent in scenario["agents"]:
                messages.append({
                    "role": agent,
                    "content": f"{agent.capitalize()} contribution to {scenario['topic']}..."
                })
                agent_rewards[agent] = 0.7 + random.random() * 0.3
            
            sample = {
                "task_id": f"multi_agent_{i:04d}",
                "task_type": "multi_agent_collaboration",
                "hierarchy_level": "high",
                "agents_involved": scenario["agents"],
                "messages": messages,
                "ground_truth": f"Successful {scenario['scenario']}",
                "reward": sum(agent_rewards.values()) / len(agent_rewards),
                "agent_rewards": agent_rewards,
                "scenario": scenario["scenario"],
                "coordination_type": "cooperative",
                "quality_metrics": {
                    "collaboration": 0.85 + random.random() * 0.15,
                    "completeness": 0.8 + random.random() * 0.2,
                    "coherence": 0.9 + random.random() * 0.1
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_rag_tasks(self, n: int) -> List[Dict]:
        """Generate retrieval-augmented generation tasks"""
        samples = []
        
        questions = [
            "What are the main causes of climate change?",
            "How does quantum computing differ from classical computing?",
            "What are the applications of CRISPR technology?",
            "Explain the concept of neural networks",
            "What are the benefits of renewable energy?",
            "How does blockchain technology work?",
            "What is the role of AI in healthcare?",
            "Describe the process of photosynthesis",
            "What are the challenges in space exploration?",
            "How do vaccines work?"
        ]
        
        for i in range(n):
            question = random.choice(questions)
            
            sample = {
                "task_id": f"rag_{i:04d}",
                "task_type": "retrieval_augmented_generation",
                "hierarchy_level": "high",
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "system", "content": "Searching for relevant information..."},
                    {"role": "retriever", "content": "[Retrieved passages about the topic]"},
                    {"role": "assistant", "content": f"Based on the retrieved information, {question.lower()} [comprehensive answer]"}
                ],
                "ground_truth": f"Accurate answer to: {question}",
                "reward": 0.85 + random.random() * 0.15,
                "retrieval_steps": random.randint(1, 3),
                "quality_metrics": {
                    "relevance": 0.9 + random.random() * 0.1,
                    "accuracy": 0.85 + random.random() * 0.15,
                    "completeness": 0.8 + random.random() * 0.2,
                    "grounding": 0.9  # How well grounded in retrieved docs
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def generate_tool_use_tasks(self, n: int) -> List[Dict]:
        """Generate tasks requiring tool usage"""
        samples = []
        
        tool_scenarios = [
            {
                "tool": "calculator",
                "task": "Calculate compound interest",
                "params": {"principal": 1000, "rate": 0.05, "time": 10}
            },
            {
                "tool": "web_search",
                "task": "Find recent news about a topic",
                "params": {"query": "AI breakthroughs 2024"}
            },
            {
                "tool": "code_executor",
                "task": "Run Python code to analyze data",
                "params": {"code": "import pandas as pd; df.describe()"}
            },
            {
                "tool": "database_query",
                "task": "Query database for user information",
                "params": {"sql": "SELECT * FROM users WHERE active = true"}
            }
        ]
        
        for i in range(n):
            scenario = random.choice(tool_scenarios)
            
            sample = {
                "task_id": f"tool_use_{i:04d}",
                "task_type": "tool_usage",
                "hierarchy_level": "low",
                "tool_used": scenario["tool"],
                "messages": [
                    {"role": "user", "content": scenario["task"]},
                    {"role": "assistant", "content": f"I'll use the {scenario['tool']} tool to help with this task."},
                    {"role": "tool", "content": f"[{scenario['tool']} output]"},
                    {"role": "assistant", "content": f"Based on the {scenario['tool']} results, [final answer]"}
                ],
                "ground_truth": f"Correct usage of {scenario['tool']}",
                "reward": 0.9 + random.random() * 0.1,
                "tool_parameters": scenario["params"],
                "quality_metrics": {
                    "tool_selection": 1.0,
                    "parameter_accuracy": 0.95,
                    "result_interpretation": 0.9 + random.random() * 0.1
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def create_specialized_dataset(self, task_type: str, n_samples: int) -> Path:
        """Create a specialized dataset for a specific task type"""
        if task_type not in self.task_generators:
            raise ValueError(f"Unknown task type: {task_type}. Available: {list(self.task_generators.keys())}")
        
        print(f"🔨 Generating {n_samples} {task_type} samples...")
        samples = self.task_generators[task_type](n_samples)
        
        # Save to file
        output_file = self.output_dir / f"{task_type}_train.jsonl"
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Created {task_type} dataset: {output_file}")
        print(f"   Samples: {len(samples)}")
        
        return output_file
    
    def load_dataset(self, file_path: Path) -> List[Dict]:
        """Load a JSONL dataset"""
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        return samples
    
    def get_dataset_statistics(self, file_path: Path) -> Dict:
        """Get statistics about a dataset"""
        samples = self.load_dataset(file_path)
        
        stats = {
            "total_samples": len(samples),
            "task_types": {},
            "hierarchy_levels": {"high": 0, "low": 0},
            "average_reward": 0,
            "agent_types": set()
        }
        
        total_reward = 0
        for sample in samples:
            # Count task types
            task_type = sample.get("task_type", "unknown")
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
            
            # Count hierarchy levels
            level = sample.get("hierarchy_level", "low")
            stats["hierarchy_levels"][level] += 1
            
            # Sum rewards
            total_reward += sample.get("reward", 0)
            
            # Collect agent types
            if "agents_involved" in sample:
                stats["agent_types"].update(sample["agents_involved"])
        
        stats["average_reward"] = total_reward / len(samples) if samples else 0
        stats["agent_types"] = list(stats["agent_types"])
        
        return stats


# Main execution
if __name__ == "__main__":
    print("🚀 Agent Lightning Training Dataset Creator")
    print("=" * 60)
    
    # Create dataset creator
    creator = TrainingDatasetCreator()
    
    # Create comprehensive dataset
    print("\n📊 Creating comprehensive training dataset...")
    train_file = creator.create_comprehensive_dataset(samples_per_type=50)
    
    # Create specialized datasets
    print("\n📊 Creating specialized datasets...")
    math_file = creator.create_specialized_dataset("math", 100)
    rag_file = creator.create_specialized_dataset("rag", 100)
    multi_agent_file = creator.create_specialized_dataset("multi_agent", 100)
    
    # Display statistics
    print("\n📈 Dataset Statistics:")
    print("-" * 40)
    
    for file_path in [train_file, math_file, rag_file, multi_agent_file]:
        if file_path.exists():
            stats = creator.get_dataset_statistics(file_path)
            print(f"\n{file_path.name}:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Average reward: {stats['average_reward']:.3f}")
            print(f"  Task types: {list(stats['task_types'].keys())[:3]}...")
            print(f"  Hierarchy: High={stats['hierarchy_levels']['high']}, Low={stats['hierarchy_levels']['low']}")
    
    print("\n✅ Dataset creation complete!")
    print(f"📁 All datasets saved to: {creator.output_dir}")
    print("\nReady for training with Agent Lightning!")