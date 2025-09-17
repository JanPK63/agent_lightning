"""
Meta-Learning for Agent Lightning
Implements learning-to-learn capabilities for faster adaptation
Following MAML (Model-Agnostic Meta-Learning) and Reptile approaches
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import copy
import time
from pathlib import Path

# Import Agent Lightning components
from mdp_agents import MDPAgent, AgentState, AgentAction, MDPTransition
from reward_functions import RewardCalculator
from curriculum_learning import CurriculumTask, TaskCategory


@dataclass
class MetaTask:
    """Represents a task for meta-learning"""
    task_id: str
    task_type: str
    support_set: List[Dict]  # Few examples for learning
    query_set: List[Dict]    # Examples for evaluation
    metadata: Dict = field(default_factory=dict)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning"""
    inner_lr: float = 0.01          # Learning rate for task adaptation
    outer_lr: float = 0.001         # Learning rate for meta-update
    inner_steps: int = 5            # Gradient steps for task adaptation
    meta_batch_size: int = 4        # Number of tasks per meta-update
    adaptation_steps: int = 3       # Steps for fine-tuning
    first_order: bool = True        # Use first-order approximation
    use_reptile: bool = False       # Use Reptile instead of MAML


class MetaLearningModel(nn.Module):
    """
    Neural network model for meta-learning
    Adapts quickly to new tasks with few examples
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Task-specific heads (can be adapted quickly)
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Meta-knowledge parameters
        self.meta_embeddings = nn.Parameter(torch.randn(1, hidden_dim))
        
    def forward(self, x: torch.Tensor, task_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional task embedding"""
        features = self.feature_extractor(x)
        
        # Add meta-knowledge
        if task_embedding is not None:
            features = features + task_embedding
        else:
            features = features + self.meta_embeddings
        
        output = self.task_head(features)
        return output
    
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor, 
              steps: int = 5, lr: float = 0.01) -> 'MetaLearningModel':
        """
        Adapt model to new task using support set
        Returns adapted model
        """
        # Create a copy for adaptation
        adapted_model = copy.deepcopy(self)
        optimizer = optim.SGD(adapted_model.parameters(), lr=lr)
        
        # Adapt on support set
        for _ in range(steps):
            pred = adapted_model(support_x)
            loss = nn.functional.mse_loss(pred, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model


class MetaLearner:
    """
    Main meta-learning system for Agent Lightning
    Implements MAML and Reptile algorithms for fast adaptation
    """
    
    def __init__(self, config: MetaLearningConfig = None):
        """
        Initialize meta-learner
        
        Args:
            config: Meta-learning configuration
        """
        self.config = config or MetaLearningConfig()
        
        # Initialize model
        self.model = MetaLearningModel()
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
        
        # Task distribution
        self.task_distribution = self._build_task_distribution()
        
        # Performance tracking
        self.meta_losses = []
        self.task_performances = defaultdict(list)
        self.adaptation_trajectories = []
        
        # Knowledge base for meta-learning
        self.meta_knowledge = {
            "task_embeddings": {},
            "successful_adaptations": [],
            "task_similarities": {}
        }
        
        print(f"🧠 Meta-Learner initialized")
        print(f"   Algorithm: {'Reptile' if self.config.use_reptile else 'MAML'}")
        print(f"   Inner LR: {self.config.inner_lr}, Outer LR: {self.config.outer_lr}")
        print(f"   Inner steps: {self.config.inner_steps}")
    
    def _build_task_distribution(self) -> Dict[str, List[MetaTask]]:
        """Build distribution of meta-tasks"""
        distribution = {}
        
        # Math tasks
        distribution["math"] = [
            MetaTask(
                task_id="meta_math_arithmetic",
                task_type="arithmetic",
                support_set=[
                    {"input": "2 + 3", "output": "5"},
                    {"input": "7 - 4", "output": "3"},
                    {"input": "5 * 2", "output": "10"}
                ],
                query_set=[
                    {"input": "8 + 6", "output": "14"},
                    {"input": "9 - 3", "output": "6"}
                ]
            ),
            MetaTask(
                task_id="meta_math_algebra",
                task_type="algebra",
                support_set=[
                    {"input": "x + 2 = 5", "output": "x = 3"},
                    {"input": "2x = 8", "output": "x = 4"},
                    {"input": "x - 3 = 7", "output": "x = 10"}
                ],
                query_set=[
                    {"input": "3x = 15", "output": "x = 5"},
                    {"input": "x + 4 = 9", "output": "x = 5"}
                ]
            )
        ]
        
        # Text tasks
        distribution["text"] = [
            MetaTask(
                task_id="meta_text_sentiment",
                task_type="sentiment",
                support_set=[
                    {"input": "I love this!", "output": "positive"},
                    {"input": "This is terrible", "output": "negative"},
                    {"input": "It's okay", "output": "neutral"}
                ],
                query_set=[
                    {"input": "Amazing work!", "output": "positive"},
                    {"input": "I hate it", "output": "negative"}
                ]
            ),
            MetaTask(
                task_id="meta_text_classification",
                task_type="classification",
                support_set=[
                    {"input": "The cat sat on the mat", "output": "statement"},
                    {"input": "What time is it?", "output": "question"},
                    {"input": "Please help me", "output": "request"}
                ],
                query_set=[
                    {"input": "The sun is shining", "output": "statement"},
                    {"input": "Can you assist?", "output": "question"}
                ]
            )
        ]
        
        # Code tasks
        distribution["code"] = [
            MetaTask(
                task_id="meta_code_function",
                task_type="function_prediction",
                support_set=[
                    {"input": "def add(a, b):", "output": "return a + b"},
                    {"input": "def multiply(x, y):", "output": "return x * y"},
                    {"input": "def subtract(m, n):", "output": "return m - n"}
                ],
                query_set=[
                    {"input": "def divide(p, q):", "output": "return p / q"},
                    {"input": "def power(base, exp):", "output": "return base ** exp"}
                ]
            )
        ]
        
        # Reasoning tasks
        distribution["reasoning"] = [
            MetaTask(
                task_id="meta_reasoning_pattern",
                task_type="pattern_completion",
                support_set=[
                    {"input": "2, 4, 6, ?", "output": "8"},
                    {"input": "1, 3, 5, ?", "output": "7"},
                    {"input": "10, 20, 30, ?", "output": "40"}
                ],
                query_set=[
                    {"input": "5, 10, 15, ?", "output": "20"},
                    {"input": "3, 6, 9, ?", "output": "12"}
                ]
            )
        ]
        
        return distribution
    
    def meta_train_step(self, tasks: List[MetaTask]) -> float:
        """
        Perform one meta-training step
        
        Args:
            tasks: Batch of meta-tasks
            
        Returns:
            Meta-loss value
        """
        if self.config.use_reptile:
            return self._reptile_step(tasks)
        else:
            return self._maml_step(tasks)
    
    def _maml_step(self, tasks: List[MetaTask]) -> float:
        """MAML meta-training step"""
        meta_loss = 0.0
        
        for task in tasks:
            # Convert task data to tensors
            support_x, support_y = self._task_to_tensors(task.support_set)
            query_x, query_y = self._task_to_tensors(task.query_set)
            
            # Inner loop: Adapt to task
            adapted_model = self.model.adapt(
                support_x, support_y,
                steps=self.config.inner_steps,
                lr=self.config.inner_lr
            )
            
            # Compute loss on query set with adapted model
            query_pred = adapted_model(query_x)
            task_loss = nn.functional.mse_loss(query_pred, query_y)
            
            # Accumulate gradients for meta-update
            if self.config.first_order:
                # First-order approximation (FOMAML)
                task_loss.backward()
            else:
                # Full second-order gradients
                grads = torch.autograd.grad(task_loss, adapted_model.parameters())
                for param, grad in zip(self.model.parameters(), grads):
                    param.grad = grad if param.grad is None else param.grad + grad
            
            meta_loss += task_loss.item()
        
        # Meta-update
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        
        avg_meta_loss = meta_loss / len(tasks)
        self.meta_losses.append(avg_meta_loss)
        
        return avg_meta_loss
    
    def _reptile_step(self, tasks: List[MetaTask]) -> float:
        """Reptile meta-training step"""
        meta_loss = 0.0
        
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        for task in tasks:
            # Convert task data to tensors
            support_x, support_y = self._task_to_tensors(task.support_set)
            query_x, query_y = self._task_to_tensors(task.query_set)
            
            # Reset to initial parameters
            for p, initial_p in zip(self.model.parameters(), initial_params):
                p.data.copy_(initial_p.data)
            
            # Inner loop: Train on task
            task_optimizer = optim.SGD(self.model.parameters(), lr=self.config.inner_lr)
            
            for _ in range(self.config.inner_steps):
                pred = self.model(support_x)
                loss = nn.functional.mse_loss(pred, support_y)
                
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
            
            # Evaluate on query set
            with torch.no_grad():
                query_pred = self.model(query_x)
                task_loss = nn.functional.mse_loss(query_pred, query_y)
                meta_loss += task_loss.item()
        
        # Reptile meta-update: Move towards average of task-adapted parameters
        for p, initial_p in zip(self.model.parameters(), initial_params):
            p.data = initial_p.data + self.config.outer_lr * (p.data - initial_p.data)
        
        avg_meta_loss = meta_loss / len(tasks)
        self.meta_losses.append(avg_meta_loss)
        
        return avg_meta_loss
    
    def _task_to_tensors(self, examples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert task examples to tensors"""
        # Simplified: Use random embeddings for demonstration
        # In practice, would use actual text/code embeddings
        x = torch.randn(len(examples), 768)
        y = torch.randn(len(examples), 10)
        return x, y
    
    def adapt_to_new_task(self, task: MetaTask, num_steps: int = None) -> Dict[str, Any]:
        """
        Adapt to a new task using meta-learned initialization
        
        Args:
            task: New task to adapt to
            num_steps: Number of adaptation steps
            
        Returns:
            Adaptation results
        """
        num_steps = num_steps or self.config.adaptation_steps
        
        # Convert task data
        support_x, support_y = self._task_to_tensors(task.support_set)
        query_x, query_y = self._task_to_tensors(task.query_set)
        
        # Track adaptation trajectory
        trajectory = []
        
        # Initial performance
        with torch.no_grad():
            initial_pred = self.model(query_x)
            initial_loss = nn.functional.mse_loss(initial_pred, query_y).item()
            trajectory.append(initial_loss)
        
        # Adapt model
        adapted_model = copy.deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)
        
        for step in range(num_steps):
            # Train on support set
            support_pred = adapted_model(support_x)
            support_loss = nn.functional.mse_loss(support_pred, support_y)
            
            optimizer.zero_grad()
            support_loss.backward()
            optimizer.step()
            
            # Evaluate on query set
            with torch.no_grad():
                query_pred = adapted_model(query_x)
                query_loss = nn.functional.mse_loss(query_pred, query_y).item()
                trajectory.append(query_loss)
        
        # Calculate improvement
        improvement = initial_loss - trajectory[-1]
        adaptation_efficiency = improvement / num_steps if num_steps > 0 else 0
        
        # Store successful adaptation
        if improvement > 0:
            self.meta_knowledge["successful_adaptations"].append({
                "task_id": task.task_id,
                "initial_loss": initial_loss,
                "final_loss": trajectory[-1],
                "improvement": improvement,
                "steps": num_steps
            })
        
        results = {
            "task_id": task.task_id,
            "initial_loss": initial_loss,
            "final_loss": trajectory[-1],
            "improvement": improvement,
            "adaptation_efficiency": adaptation_efficiency,
            "trajectory": trajectory,
            "adapted_model": adapted_model
        }
        
        # Update performance tracking
        self.task_performances[task.task_type].append(results["final_loss"])
        self.adaptation_trajectories.append(trajectory)
        
        return results
    
    def learn_task_embeddings(self, tasks: List[MetaTask]) -> Dict[str, torch.Tensor]:
        """
        Learn embeddings for different task types
        
        Args:
            tasks: List of tasks to learn embeddings for
            
        Returns:
            Dictionary of task embeddings
        """
        embeddings = {}
        
        for task in tasks:
            # Extract features from support set
            support_x, _ = self._task_to_tensors(task.support_set)
            
            with torch.no_grad():
                # Get features from model
                features = self.model.feature_extractor(support_x)
                # Average pooling to get task embedding
                task_embedding = features.mean(dim=0, keepdim=True)
                
                embeddings[task.task_id] = task_embedding
                self.meta_knowledge["task_embeddings"][task.task_id] = task_embedding
        
        # Compute task similarities
        self._compute_task_similarities(embeddings)
        
        return embeddings
    
    def _compute_task_similarities(self, embeddings: Dict[str, torch.Tensor]):
        """Compute similarities between task embeddings"""
        task_ids = list(embeddings.keys())
        
        for i, task_i in enumerate(task_ids):
            for j, task_j in enumerate(task_ids):
                if i < j:
                    # Cosine similarity
                    emb_i = embeddings[task_i]
                    emb_j = embeddings[task_j]
                    
                    similarity = torch.cosine_similarity(emb_i, emb_j).item()
                    
                    self.meta_knowledge["task_similarities"][f"{task_i}_{task_j}"] = similarity
    
    def meta_train(self, num_iterations: int = 100, save_interval: int = 10):
        """
        Full meta-training loop
        
        Args:
            num_iterations: Number of meta-training iterations
            save_interval: Interval for saving checkpoints
        """
        print(f"\n🚀 Starting meta-training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Sample tasks from distribution
            task_batch = self._sample_task_batch()
            
            # Meta-training step
            meta_loss = self.meta_train_step(task_batch)
            
            # Logging
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Meta-loss = {meta_loss:.4f}")
            
            # Save checkpoint
            if iteration % save_interval == 0 and iteration > 0:
                self.save_checkpoint(f"meta_checkpoint_{iteration}.pt")
        
        print("✅ Meta-training complete!")
        
        # Final evaluation
        self.evaluate_meta_learning()
    
    def _sample_task_batch(self) -> List[MetaTask]:
        """Sample a batch of tasks for meta-training"""
        batch = []
        
        for _ in range(self.config.meta_batch_size):
            # Sample task category
            category = np.random.choice(list(self.task_distribution.keys()))
            # Sample task from category
            task = np.random.choice(self.task_distribution[category])
            batch.append(task)
        
        return batch
    
    def evaluate_meta_learning(self) -> Dict[str, float]:
        """Evaluate meta-learning performance"""
        print("\n📊 Evaluating meta-learning performance...")
        
        results = {}
        
        # Test on each task category
        for category, tasks in self.task_distribution.items():
            category_losses = []
            
            for task in tasks[:2]:  # Test on first 2 tasks per category
                adaptation_result = self.adapt_to_new_task(task)
                category_losses.append(adaptation_result["final_loss"])
            
            avg_loss = np.mean(category_losses)
            results[category] = avg_loss
            
            print(f"  {category}: {avg_loss:.4f}")
        
        # Overall metrics
        overall_loss = np.mean(list(results.values()))
        avg_improvement = np.mean([
            adapt["improvement"] 
            for adapt in self.meta_knowledge["successful_adaptations"]
        ]) if self.meta_knowledge["successful_adaptations"] else 0
        
        results["overall"] = overall_loss
        results["avg_improvement"] = avg_improvement
        
        print(f"\n  Overall: {overall_loss:.4f}")
        print(f"  Average Improvement: {avg_improvement:.4f}")
        
        return results
    
    def transfer_knowledge(self, source_task: str, target_task: str) -> float:
        """
        Transfer knowledge from source to target task
        
        Args:
            source_task: Source task ID
            target_task: Target task ID
            
        Returns:
            Transfer effectiveness score
        """
        # Get task embeddings
        source_emb = self.meta_knowledge["task_embeddings"].get(source_task)
        
        if source_emb is None:
            print(f"⚠️ No embedding found for source task: {source_task}")
            return 0.0
        
        # Find target task
        target = None
        for tasks in self.task_distribution.values():
            for task in tasks:
                if task.task_id == target_task:
                    target = task
                    break
        
        if target is None:
            print(f"⚠️ Target task not found: {target_task}")
            return 0.0
        
        # Adapt using source task embedding
        adapted_model = copy.deepcopy(self.model)
        
        # Use source embedding as initialization hint
        support_x, support_y = self._task_to_tensors(target.support_set)
        query_x, query_y = self._task_to_tensors(target.query_set)
        
        # Initial performance with transfer
        with torch.no_grad():
            # Add source embedding to influence adaptation
            query_pred = adapted_model(query_x, task_embedding=source_emb)
            transfer_loss = nn.functional.mse_loss(query_pred, query_y).item()
        
        # Baseline without transfer
        with torch.no_grad():
            baseline_pred = self.model(query_x)
            baseline_loss = nn.functional.mse_loss(baseline_pred, query_y).item()
        
        # Calculate transfer effectiveness
        transfer_score = (baseline_loss - transfer_loss) / baseline_loss if baseline_loss > 0 else 0
        
        print(f"📤 Transfer from {source_task} to {target_task}")
        print(f"   Baseline loss: {baseline_loss:.4f}")
        print(f"   Transfer loss: {transfer_loss:.4f}")
        print(f"   Transfer effectiveness: {transfer_score:.2%}")
        
        return transfer_score
    
    def save_checkpoint(self, filepath: str):
        """Save meta-learning checkpoint"""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.meta_optimizer.state_dict(),
            "config": self.config,
            "meta_losses": self.meta_losses,
            "meta_knowledge": self.meta_knowledge,
            "timestamp": time.time()
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load meta-learning checkpoint"""
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.meta_optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.config = checkpoint["config"]
        self.meta_losses = checkpoint["meta_losses"]
        self.meta_knowledge = checkpoint["meta_knowledge"]
        
        print(f"📂 Checkpoint loaded from {filepath}")
    
    def generate_meta_report(self) -> Dict[str, Any]:
        """Generate comprehensive meta-learning report"""
        report = {
            "training_iterations": len(self.meta_losses),
            "final_meta_loss": self.meta_losses[-1] if self.meta_losses else None,
            "avg_meta_loss": np.mean(self.meta_losses) if self.meta_losses else None,
            "num_successful_adaptations": len(self.meta_knowledge["successful_adaptations"]),
            "avg_adaptation_improvement": np.mean([
                a["improvement"] for a in self.meta_knowledge["successful_adaptations"]
            ]) if self.meta_knowledge["successful_adaptations"] else 0,
            "task_embeddings_learned": len(self.meta_knowledge["task_embeddings"]),
            "task_performance": {
                task_type: {
                    "avg_loss": np.mean(losses),
                    "min_loss": np.min(losses),
                    "num_evaluations": len(losses)
                }
                for task_type, losses in self.task_performances.items()
                if losses
            }
        }
        
        return report


# Example usage
if __name__ == "__main__":
    print("🧠 Testing Meta-Learning System")
    print("=" * 60)
    
    # Initialize meta-learner
    config = MetaLearningConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        meta_batch_size=4,
        use_reptile=False  # Use MAML
    )
    
    meta_learner = MetaLearner(config)
    
    # Meta-training
    print("\n📚 Meta-Training Phase...")
    meta_learner.meta_train(num_iterations=20)
    
    # Test adaptation to new task
    print("\n🎯 Testing Adaptation to New Task...")
    test_task = meta_learner.task_distribution["math"][0]
    adaptation_result = meta_learner.adapt_to_new_task(test_task, num_steps=10)
    
    print(f"\nAdaptation Results for {test_task.task_id}:")
    print(f"  Initial loss: {adaptation_result['initial_loss']:.4f}")
    print(f"  Final loss: {adaptation_result['final_loss']:.4f}")
    print(f"  Improvement: {adaptation_result['improvement']:.4f}")
    print(f"  Efficiency: {adaptation_result['adaptation_efficiency']:.4f}")
    
    # Learn task embeddings
    print("\n🔍 Learning Task Embeddings...")
    all_tasks = []
    for tasks in meta_learner.task_distribution.values():
        all_tasks.extend(tasks)
    
    embeddings = meta_learner.learn_task_embeddings(all_tasks[:5])
    print(f"  Learned embeddings for {len(embeddings)} tasks")
    
    # Test knowledge transfer
    print("\n🔄 Testing Knowledge Transfer...")
    if len(all_tasks) >= 2:
        transfer_score = meta_learner.transfer_knowledge(
            all_tasks[0].task_id,
            all_tasks[1].task_id
        )
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 META-LEARNING REPORT")
    print("=" * 60)
    
    report = meta_learner.generate_meta_report()
    print(f"\nTraining Iterations: {report['training_iterations']}")
    print(f"Final Meta-Loss: {report['final_meta_loss']:.4f}" if report['final_meta_loss'] else "N/A")
    print(f"Successful Adaptations: {report['num_successful_adaptations']}")
    print(f"Avg Adaptation Improvement: {report['avg_adaptation_improvement']:.4f}")
    print(f"Task Embeddings Learned: {report['task_embeddings_learned']}")
    
    if report['task_performance']:
        print("\nTask Performance:")
        for task_type, metrics in report['task_performance'].items():
            print(f"  {task_type}:")
            print(f"    Avg Loss: {metrics['avg_loss']:.4f}")
            print(f"    Min Loss: {metrics['min_loss']:.4f}")
    
    print("\n✅ Meta-learning test complete!")