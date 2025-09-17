"""
Test Dataset Validation for Agent Lightning
Tests the complete system using Calc-X and Spider datasets
Validates agents, rewards, and training pipeline
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path

# Import Agent Lightning components
from mdp_agents import MDPAgent, AgentState, AgentAction, MDPTransition
from multi_agent_system import MultiAgentSystem, SystemConfig
from orchestration_workflows import (
    create_workflow, WorkflowType, WorkflowTask,
    SequentialWorkflow, ParallelWorkflow, HierarchicalWorkflow
)
from reward_functions import RewardCalculator, RewardConfig, RewardType
from memory_manager import MemoryManager
from observability_setup import AgentLightningObservability
from create_training_dataset import TrainingDataGenerator


class DatasetValidator:
    """
    Validates Agent Lightning system using real datasets
    Tests end-to-end functionality with Calc-X and Spider
    """
    
    def __init__(self):
        """Initialize dataset validator"""
        self.calc_x_path = Path("examples/calc_x_data.jsonl")
        self.spider_path = Path("examples/spider_data.jsonl")
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.observability = AgentLightningObservability(
            service_name="agent-lightning-validation",
            prometheus_port=8002,
            enable_console_export=False
        )
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            RewardConfig(reward_type=RewardType.SHAPED)
        )
        
        # Initialize multi-agent system
        self.multi_agent_system = MultiAgentSystem(
            SystemConfig(num_agents=4)
        )
        
        # Metrics tracking
        self.validation_results = {
            "calc_x": {},
            "spider": {}
        }
        
        print("🧪 Dataset Validator initialized")
    
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load dataset from file"""
        if dataset_name == "calc_x":
            path = self.calc_x_path
        elif dataset_name == "spider":
            path = self.spider_path
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not path.exists():
            print(f"⚠️ Dataset file not found: {path}")
            print("  Generating sample data...")
            self.generate_sample_data(dataset_name, path)
        
        # Load data
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"📂 Loaded {len(data)} examples from {dataset_name}")
        return data
    
    def generate_sample_data(self, dataset_name: str, path: Path):
        """Generate sample data if dataset doesn't exist"""
        generator = TrainingDataGenerator()
        
        if dataset_name == "calc_x":
            # Generate Calc-X style math problems
            samples = []
            for i in range(20):
                problem = generator.generate_math_problem()
                samples.append({
                    "id": f"calc_x_{i}",
                    "question": problem["question"],
                    "answer": problem["answer"],
                    "difficulty": problem.get("difficulty", "medium"),
                    "type": "math"
                })
        
        elif dataset_name == "spider":
            # Generate Spider style SQL problems
            samples = []
            for i in range(20):
                problem = generator.generate_sql_problem()
                samples.append({
                    "id": f"spider_{i}",
                    "question": problem["question"],
                    "sql": problem["answer"],
                    "database": problem.get("database", "example_db"),
                    "difficulty": problem.get("difficulty", "medium"),
                    "type": "sql"
                })
        
        # Save to file
        path.parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Generated {len(samples)} sample examples for {dataset_name}")
    
    async def validate_calc_x(self) -> Dict[str, Any]:
        """Validate system using Calc-X dataset"""
        print("\n📐 Validating with Calc-X Dataset...")
        print("=" * 60)
        
        # Load dataset
        data = self.load_dataset("calc_x")
        
        # Test different workflow types
        workflows_to_test = [
            WorkflowType.SEQUENTIAL,
            WorkflowType.PARALLEL,
            WorkflowType.HIERARCHICAL
        ]
        
        results = {}
        
        for workflow_type in workflows_to_test:
            print(f"\n🔄 Testing {workflow_type.value} workflow...")
            
            # Create workflow
            agents = {
                "math_solver": MDPAgent(role="MathSolver"),
                "verifier": MDPAgent(role="Verifier")
            }
            
            workflow = create_workflow(
                workflow_type,
                agents,
                memory_manager=self.memory_manager,
                observability=self.observability
            )
            
            # Test on subset of data
            test_data = data[:5]  # Use first 5 examples
            correct = 0
            total_reward = 0.0
            
            for example in test_data:
                # Create task
                task = WorkflowTask(
                    task_id=example["id"],
                    task_type="math",
                    input_data={"question": example["question"]},
                    required_agents=list(agents.keys()),
                    metadata={"difficulty": example.get("difficulty", "medium")}
                )
                
                # Execute workflow
                with self.observability.trace_agent_execution(
                    agent_id=f"calc_x_{workflow_type.value}",
                    task_type="math_validation"
                ):
                    result = await workflow.execute(task)
                
                # Extract answer from results
                predicted_answer = self.extract_answer(result.results)
                
                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    action=predicted_answer,
                    ground_truth=str(example["answer"]),
                    task_type="math",
                    metadata={"example_id": example["id"]}
                )
                
                total_reward += reward
                
                # Check correctness
                if self.check_math_answer(predicted_answer, str(example["answer"])):
                    correct += 1
                    print(f"  ✅ {example['id']}: Correct (reward: {reward:.3f})")
                else:
                    print(f"  ❌ {example['id']}: Incorrect (reward: {reward:.3f})")
                    print(f"     Expected: {example['answer']}, Got: {predicted_answer}")
            
            # Calculate metrics
            accuracy = correct / len(test_data)
            avg_reward = total_reward / len(test_data)
            
            results[workflow_type.value] = {
                "accuracy": accuracy,
                "avg_reward": avg_reward,
                "correct": correct,
                "total": len(test_data)
            }
            
            print(f"\n  📊 {workflow_type.value} Results:")
            print(f"     Accuracy: {accuracy:.2%}")
            print(f"     Average Reward: {avg_reward:.3f}")
        
        self.validation_results["calc_x"] = results
        return results
    
    async def validate_spider(self) -> Dict[str, Any]:
        """Validate system using Spider dataset"""
        print("\n🕷️ Validating with Spider Dataset...")
        print("=" * 60)
        
        # Load dataset
        data = self.load_dataset("spider")
        
        # Create specialized agents for SQL
        agents = {
            "sql_generator": MDPAgent(role="SQLGenerator"),
            "sql_optimizer": MDPAgent(role="SQLOptimizer"),
            "sql_validator": MDPAgent(role="SQLValidator")
        }
        
        # Test with hierarchical workflow (best for SQL)
        workflow = create_workflow(
            WorkflowType.HIERARCHICAL,
            agents,
            memory_manager=self.memory_manager,
            observability=self.observability
        )
        
        # Test on subset
        test_data = data[:5]
        results = []
        
        for example in test_data:
            # Create task
            task = WorkflowTask(
                task_id=example["id"],
                task_type="sql",
                input_data={
                    "question": example["question"],
                    "database": example.get("database", "example_db")
                },
                required_agents=list(agents.keys()),
                metadata={"difficulty": example.get("difficulty", "medium")}
            )
            
            # Execute workflow
            with self.observability.trace_agent_execution(
                agent_id="spider_validation",
                task_type="sql_validation"
            ):
                result = await workflow.execute(task)
            
            # Extract SQL from results
            predicted_sql = self.extract_sql(result.results)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(
                action=predicted_sql,
                ground_truth=example["sql"],
                task_type="sql",
                metadata={
                    "example_id": example["id"],
                    "execution_success": self.validate_sql_syntax(predicted_sql)
                }
            )
            
            # Store result
            results.append({
                "id": example["id"],
                "predicted": predicted_sql,
                "ground_truth": example["sql"],
                "reward": reward,
                "valid_syntax": self.validate_sql_syntax(predicted_sql)
            })
            
            print(f"  📝 {example['id']}:")
            print(f"     Reward: {reward:.3f}")
            print(f"     Valid SQL: {results[-1]['valid_syntax']}")
        
        # Calculate metrics
        avg_reward = np.mean([r["reward"] for r in results])
        valid_syntax_rate = sum(1 for r in results if r["valid_syntax"]) / len(results)
        
        spider_results = {
            "avg_reward": avg_reward,
            "valid_syntax_rate": valid_syntax_rate,
            "num_examples": len(results),
            "examples": results
        }
        
        print(f"\n  📊 Spider Results:")
        print(f"     Average Reward: {avg_reward:.3f}")
        print(f"     Valid SQL Rate: {valid_syntax_rate:.2%}")
        
        self.validation_results["spider"] = spider_results
        return spider_results
    
    async def test_multi_agent_coordination(self) -> Dict[str, Any]:
        """Test multi-agent coordination on mixed tasks"""
        print("\n👥 Testing Multi-Agent Coordination...")
        print("=" * 60)
        
        # Load both datasets
        calc_x_data = self.load_dataset("calc_x")[:3]
        spider_data = self.load_dataset("spider")[:3]
        
        # Mix tasks
        mixed_tasks = []
        for calc_example in calc_x_data:
            mixed_tasks.append({
                "type": "math",
                "data": calc_example
            })
        for spider_example in spider_data:
            mixed_tasks.append({
                "type": "sql",
                "data": spider_example
            })
        
        # Execute with multi-agent system
        results = []
        
        for task_info in mixed_tasks:
            task_type = task_info["type"]
            example = task_info["data"]
            
            print(f"\n  Processing {task_type} task: {example['id']}")
            
            # Execute through multi-agent system
            task_input = {
                "type": task_type,
                "question": example.get("question", ""),
                "context": example
            }
            
            with self.observability.trace_agent_execution(
                agent_id="multi_agent_coordinator",
                task_type=f"mixed_{task_type}"
            ):
                result = await self.multi_agent_system.execute(task_input)
            
            # Calculate reward based on task type
            if task_type == "math":
                predicted = self.extract_answer({"result": result})
                ground_truth = str(example["answer"])
            else:  # sql
                predicted = self.extract_sql({"result": result})
                ground_truth = example["sql"]
            
            reward = self.reward_calculator.calculate_reward(
                action=predicted,
                ground_truth=ground_truth,
                task_type=task_type,
                metadata={
                    "multi_agent": True,
                    "example_id": example["id"]
                }
            )
            
            results.append({
                "task_type": task_type,
                "example_id": example["id"],
                "reward": reward,
                "execution_time": result.get("execution_time", 0)
            })
            
            print(f"    Reward: {reward:.3f}")
            print(f"    Execution time: {result.get('execution_time', 0):.2f}s")
        
        # Calculate metrics
        avg_reward = np.mean([r["reward"] for r in results])
        avg_time = np.mean([r["execution_time"] for r in results])
        
        coordination_results = {
            "avg_reward": avg_reward,
            "avg_execution_time": avg_time,
            "num_tasks": len(results),
            "task_breakdown": {
                "math": sum(1 for r in results if r["task_type"] == "math"),
                "sql": sum(1 for r in results if r["task_type"] == "sql")
            }
        }
        
        print(f"\n  📊 Multi-Agent Coordination Results:")
        print(f"     Average Reward: {avg_reward:.3f}")
        print(f"     Average Execution Time: {avg_time:.2f}s")
        print(f"     Tasks Processed: {len(results)}")
        
        return coordination_results
    
    async def test_reward_shaping(self) -> Dict[str, Any]:
        """Test different reward shaping strategies"""
        print("\n💰 Testing Reward Shaping Strategies...")
        print("=" * 60)
        
        # Load test data
        calc_x_data = self.load_dataset("calc_x")[:3]
        
        # Test different reward configurations
        reward_configs = [
            RewardConfig(reward_type=RewardType.SPARSE),
            RewardConfig(reward_type=RewardType.DENSE),
            RewardConfig(reward_type=RewardType.SHAPED),
            RewardConfig(reward_type=RewardType.HIERARCHICAL)
        ]
        
        results = {}
        
        for config in reward_configs:
            print(f"\n  Testing {config.reward_type.value} rewards...")
            
            calculator = RewardCalculator(config)
            rewards = []
            
            for example in calc_x_data:
                # Simulate agent response
                predicted = f"The answer is {example['answer']}"
                
                reward = calculator.calculate_reward(
                    action=predicted,
                    ground_truth=str(example["answer"]),
                    task_type="math",
                    metadata={
                        "num_steps": 5,
                        "confidence": 0.8,
                        "time_taken": 2.5,
                        "max_time": 10
                    }
                )
                
                rewards.append(reward)
                print(f"    {example['id']}: {reward:.3f}")
            
            avg_reward = np.mean(rewards)
            reward_variance = np.var(rewards)
            
            results[config.reward_type.value] = {
                "avg_reward": avg_reward,
                "variance": reward_variance,
                "min": min(rewards),
                "max": max(rewards)
            }
            
            print(f"  Average: {avg_reward:.3f}, Variance: {reward_variance:.3f}")
        
        return results
    
    # Utility methods
    def extract_answer(self, results: Dict) -> str:
        """Extract math answer from workflow results"""
        # Simplified extraction logic
        for key, value in results.items():
            if isinstance(value, dict):
                if "action" in value and isinstance(value["action"], dict):
                    content = value["action"].get("content", "")
                    # Look for numbers in the content
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', content)
                    if numbers:
                        return numbers[-1]  # Return last number found
        return "0"
    
    def extract_sql(self, results: Dict) -> str:
        """Extract SQL from workflow results"""
        # Simplified extraction logic
        for key, value in results.items():
            if isinstance(value, dict):
                if "action" in value and isinstance(value["action"], dict):
                    content = value["action"].get("content", "")
                    if "SELECT" in content.upper():
                        return content
        return "SELECT * FROM table"
    
    def check_math_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if math answer is correct"""
        try:
            pred_num = float(predicted)
            truth_num = float(ground_truth)
            return abs(pred_num - truth_num) < 1e-6
        except:
            return False
    
    def validate_sql_syntax(self, sql: str) -> bool:
        """Basic SQL syntax validation"""
        sql_upper = sql.upper().strip()
        valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"]
        return any(sql_upper.startswith(start) for start in valid_starts)
    
    async def run_full_validation(self):
        """Run complete validation suite"""
        print("\n🚀 Starting Full System Validation")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Validate with Calc-X
        calc_x_results = await self.validate_calc_x()
        
        # 2. Validate with Spider
        spider_results = await self.validate_spider()
        
        # 3. Test multi-agent coordination
        coordination_results = await self.test_multi_agent_coordination()
        
        # 4. Test reward shaping
        reward_results = await self.test_reward_shaping()
        
        # Generate summary report
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY REPORT")
        print("=" * 60)
        
        print("\n📐 Calc-X Results:")
        for workflow, metrics in calc_x_results.items():
            print(f"  {workflow}:")
            print(f"    - Accuracy: {metrics['accuracy']:.2%}")
            print(f"    - Avg Reward: {metrics['avg_reward']:.3f}")
        
        print("\n🕷️ Spider Results:")
        print(f"  - Avg Reward: {spider_results['avg_reward']:.3f}")
        print(f"  - Valid SQL Rate: {spider_results['valid_syntax_rate']:.2%}")
        
        print("\n👥 Multi-Agent Coordination:")
        print(f"  - Avg Reward: {coordination_results['avg_reward']:.3f}")
        print(f"  - Avg Execution Time: {coordination_results['avg_execution_time']:.2f}s")
        
        print("\n💰 Reward Shaping Comparison:")
        for reward_type, metrics in reward_results.items():
            print(f"  {reward_type}: avg={metrics['avg_reward']:.3f}, var={metrics['variance']:.3f}")
        
        print(f"\n⏱️ Total Validation Time: {elapsed_time:.2f} seconds")
        
        # Save results to file
        results_file = Path("validation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "calc_x": calc_x_results,
                "spider": spider_results,
                "coordination": coordination_results,
                "reward_shaping": reward_results,
                "elapsed_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        print("\n✅ Full system validation complete!")
        
        return self.validation_results


# Main execution
if __name__ == "__main__":
    print("🧪 Agent Lightning Dataset Validation")
    print("=" * 60)
    
    async def main():
        validator = DatasetValidator()
        await validator.run_full_validation()
    
    # Run validation
    asyncio.run(main())