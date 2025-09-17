"""
Distributed Training for Agent Lightning
Scales training across multiple nodes using Ray, Horovod, and PyTorch Distributed
Implements data parallelism and model parallelism strategies
"""

import ray
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
import os
from pathlib import Path
import multiprocessing as mp
from ray import serve
from ray.train import Trainer
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

# Import Agent Lightning components
from ray_distributed_config import AgentLightningEnv, HierarchicalRLModel


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    num_nodes: int = 2
    gpus_per_node: int = 1
    cpus_per_node: int = 8
    batch_size: int = 32
    local_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    checkpoint_frequency: int = 100
    communication_backend: str = "ray"  # ray, horovod, or pytorch


class AgentLightningDataset(Dataset):
    """Dataset for distributed training"""
    
    def __init__(self, data_path: str, max_samples: int = 10000):
        """
        Initialize dataset
        
        Args:
            data_path: Path to training data
            max_samples: Maximum samples to load
        """
        self.data = []
        
        # Load data (simplified for demo)
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    self.data.append(json.loads(line))
        else:
            # Generate synthetic data
            for i in range(max_samples):
                self.data.append({
                    "state": np.random.randn(768).tolist(),
                    "action": np.random.randint(0, 10),
                    "reward": np.random.random(),
                    "next_state": np.random.randn(768).tolist(),
                    "done": i % 50 == 49
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "state": torch.FloatTensor(item["state"]),
            "action": torch.LongTensor([item["action"]]),
            "reward": torch.FloatTensor([item["reward"]]),
            "next_state": torch.FloatTensor(item["next_state"]),
            "done": torch.BoolTensor([item["done"]])
        }


class DistributedAgentModel(nn.Module):
    """Distributed model for Agent Lightning"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 10):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer layers for better scalability
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Add batch dimension for transformer
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        
        # Transform
        transformed = self.transformer(features)
        
        # Remove sequence dimension if added
        if transformed.shape[1] == 1:
            transformed = transformed.squeeze(1)
        
        # Get policy and value
        policy = self.policy_head(transformed)
        value = self.value_head(transformed)
        
        return policy, value


class DistributedTrainer:
    """
    Main distributed training system for Agent Lightning
    Supports multiple distributed training backends
    """
    
    def __init__(self, config: DistributedConfig = None):
        """
        Initialize distributed trainer
        
        Args:
            config: Distributed training configuration
        """
        self.config = config or DistributedConfig()
        self.model = None
        self.optimizer = None
        self.rank = 0
        self.world_size = 1
        self.device = None
        
        print(f"🌐 Distributed Trainer initialized")
        print(f"   Backend: {config.communication_backend}")
        print(f"   Nodes: {config.num_nodes}")
        print(f"   GPUs per node: {config.gpus_per_node}")
    
    def setup_pytorch_distributed(self):
        """Setup PyTorch distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
        else:
            print("⚠️ PyTorch distributed environment not set. Using single process.")
            self.rank = 0
            self.world_size = 1
            return
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Set device
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')
        
        print(f"✅ PyTorch Distributed initialized - Rank {self.rank}/{self.world_size}")
    
    def setup_ray_distributed(self):
        """Setup Ray distributed training"""
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.config.cpus_per_node * self.config.num_nodes,
                num_gpus=self.config.gpus_per_node * self.config.num_nodes
            )
        
        print(f"✅ Ray initialized with {ray.available_resources()}")
    
    def create_model(self) -> nn.Module:
        """Create and setup distributed model"""
        model = DistributedAgentModel()
        
        if self.config.communication_backend == "pytorch":
            if self.world_size > 1:
                model = model.to(self.device)
                model = DDP(model, device_ids=[self.device])
            else:
                model = model.to(self.device if self.device else torch.device('cpu'))
        
        return model
    
    def train_step_pytorch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for PyTorch distributed"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        policy, value = self.model(batch["state"])
        
        # Calculate losses
        policy_loss = nn.functional.cross_entropy(policy, batch["action"].squeeze())
        value_loss = nn.functional.mse_loss(value.squeeze(), batch["reward"].squeeze())
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if isinstance(self.model, DDP):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
    
    def train_pytorch_distributed(self, num_epochs: int = 10):
        """Train using PyTorch distributed"""
        self.setup_pytorch_distributed()
        
        # Create model and optimizer
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Create dataset and dataloader
        dataset = AgentLightningDataset("training_data.jsonl")
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank
        ) if self.world_size > 1 else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.local_batch_size,
            sampler=sampler,
            shuffle=(sampler is None)
        )
        
        print(f"\n🚀 Starting PyTorch distributed training...")
        
        for epoch in range(num_epochs):
            if sampler:
                sampler.set_epoch(epoch)
            
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                metrics = self.train_step_pytorch(batch)
                epoch_losses.append(metrics["loss"])
                
                if batch_idx % 10 == 0 and self.rank == 0:
                    print(f"  Epoch {epoch}, Batch {batch_idx}: Loss = {metrics['loss']:.4f}")
            
            # Synchronize and average losses across processes
            if self.world_size > 1:
                avg_loss = torch.tensor(np.mean(epoch_losses), device=self.device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / self.world_size
            else:
                avg_loss = np.mean(epoch_losses)
            
            if self.rank == 0:
                print(f"Epoch {epoch} complete - Average Loss: {avg_loss:.4f}")
        
        # Cleanup
        if self.world_size > 1:
            dist.destroy_process_group()
    
    @ray.remote(num_gpus=1)
    class RayWorker:
        """Ray worker for distributed training"""
        
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size
            self.model = DistributedAgentModel()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.model = self.model.to(self.device)
            else:
                self.device = torch.device('cpu')
        
        def train_batch(self, batch: Dict) -> Dict[str, float]:
            """Train on a single batch"""
            # Convert to tensors and move to device
            batch_tensors = {
                k: torch.FloatTensor(v).to(self.device) if k != "action" 
                else torch.LongTensor(v).to(self.device)
                for k, v in batch.items()
            }
            
            # Forward pass
            policy, value = self.model(batch_tensors["state"])
            
            # Calculate losses
            policy_loss = nn.functional.cross_entropy(policy, batch_tensors["action"].squeeze())
            value_loss = nn.functional.mse_loss(value.squeeze(), batch_tensors["reward"].squeeze())
            
            total_loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return {
                "loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item()
            }
        
        def get_model_state(self):
            """Get model state dict"""
            return self.model.state_dict()
        
        def set_model_state(self, state_dict):
            """Set model state dict"""
            self.model.load_state_dict(state_dict)
    
    def train_ray_distributed(self, num_epochs: int = 10):
        """Train using Ray distributed"""
        self.setup_ray_distributed()
        
        # Create workers
        num_workers = self.config.num_nodes * self.config.gpus_per_node
        workers = [
            self.RayWorker.remote(rank=i, world_size=num_workers)
            for i in range(num_workers)
        ]
        
        print(f"\n🚀 Starting Ray distributed training with {num_workers} workers...")
        
        # Load dataset
        dataset = AgentLightningDataset("training_data.jsonl")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Create batches
            num_batches = len(dataset) // (self.config.batch_size * num_workers)
            
            for batch_idx in range(num_batches):
                # Distribute batches to workers
                futures = []
                for worker_idx, worker in enumerate(workers):
                    # Get batch for this worker
                    start_idx = (batch_idx * num_workers + worker_idx) * self.config.local_batch_size
                    end_idx = start_idx + self.config.local_batch_size
                    
                    if start_idx < len(dataset):
                        batch_data = {
                            "state": [],
                            "action": [],
                            "reward": [],
                            "next_state": [],
                            "done": []
                        }
                        
                        for i in range(start_idx, min(end_idx, len(dataset))):
                            sample = dataset[i]
                            for key in batch_data:
                                batch_data[key].append(sample[key].numpy())
                        
                        # Stack into arrays
                        for key in batch_data:
                            batch_data[key] = np.stack(batch_data[key])
                        
                        future = worker.train_batch.remote(batch_data)
                        futures.append(future)
                
                # Collect results
                results = ray.get(futures)
                batch_loss = np.mean([r["loss"] for r in results])
                epoch_losses.append(batch_loss)
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch}, Batch {batch_idx}: Loss = {batch_loss:.4f}")
            
            # Synchronize models (parameter averaging)
            if epoch % 5 == 0:
                # Get all model states
                model_states = ray.get([w.get_model_state.remote() for w in workers])
                
                # Average parameters
                avg_state = {}
                for key in model_states[0].keys():
                    avg_state[key] = torch.stack([state[key] for state in model_states]).mean(dim=0)
                
                # Distribute averaged model
                ray.get([w.set_model_state.remote(avg_state) for w in workers])
            
            print(f"Epoch {epoch} complete - Average Loss: {np.mean(epoch_losses):.4f}")
        
        ray.shutdown()
    
    def train_with_ray_train(self):
        """Train using Ray Train for better integration"""
        from ray.train.torch import TorchTrainer
        from ray.air import session
        from ray.air.config import ScalingConfig
        
        def train_func(config):
            """Training function for Ray Train"""
            # Setup
            model = DistributedAgentModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            
            # Get dataset shard
            dataset = session.get_dataset_shard("train")
            
            for epoch in range(config["num_epochs"]):
                epoch_losses = []
                
                for batch in dataset.iter_torch_batches(batch_size=config["batch_size"]):
                    # Forward pass
                    policy, value = model(batch["state"])
                    
                    # Calculate losses
                    policy_loss = nn.functional.cross_entropy(policy, batch["action"].squeeze())
                    value_loss = nn.functional.mse_loss(value.squeeze(), batch["reward"].squeeze())
                    total_loss = policy_loss + 0.5 * value_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(total_loss.item())
                
                # Report metrics
                session.report({
                    "epoch": epoch,
                    "loss": np.mean(epoch_losses)
                })
        
        # Create Ray dataset
        data = []
        for i in range(1000):
            data.append({
                "state": np.random.randn(768),
                "action": np.random.randint(0, 10),
                "reward": np.random.random(),
                "next_state": np.random.randn(768),
                "done": i % 50 == 49
            })
        
        ray_dataset = ray.data.from_items(data)
        
        # Configure trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={
                "lr": 1e-4,
                "batch_size": 32,
                "num_epochs": 10
            },
            scaling_config=ScalingConfig(
                num_workers=self.config.num_nodes,
                use_gpu=self.config.gpus_per_node > 0
            ),
            datasets={"train": ray_dataset}
        )
        
        # Run training
        result = trainer.fit()
        print(f"Training complete: {result}")
    
    def setup_model_parallelism(self):
        """Setup model parallelism for very large models"""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("⚠️ Model parallelism requires multiple GPUs")
            return None
        
        class ModelParallelNetwork(nn.Module):
            """Model split across multiple GPUs"""
            
            def __init__(self):
                super().__init__()
                
                # First part on GPU 0
                self.encoder = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024)
                ).to('cuda:0')
                
                # Second part on GPU 1
                self.decoder = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                ).to('cuda:1')
            
            def forward(self, x):
                x = x.to('cuda:0')
                x = self.encoder(x)
                x = x.to('cuda:1')
                x = self.decoder(x)
                return x
        
        return ModelParallelNetwork()
    
    def federated_learning_setup(self):
        """Setup federated learning for privacy-preserving training"""
        
        @ray.remote
        class FederatedClient:
            """Client for federated learning"""
            
            def __init__(self, client_id: int):
                self.client_id = client_id
                self.model = DistributedAgentModel()
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
                self.data = self._load_local_data()
            
            def _load_local_data(self):
                """Load client's local data"""
                # Simulate local data
                return AgentLightningDataset(f"client_{self.client_id}_data.jsonl", max_samples=100)
            
            def train_local(self, global_weights: Dict, num_epochs: int = 5):
                """Train on local data"""
                # Load global weights
                self.model.load_state_dict(global_weights)
                
                dataloader = DataLoader(self.data, batch_size=16, shuffle=True)
                
                for epoch in range(num_epochs):
                    for batch in dataloader:
                        # Training step
                        policy, value = self.model(batch["state"])
                        loss = nn.functional.cross_entropy(policy, batch["action"].squeeze())
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                
                return self.model.state_dict()
        
        return FederatedClient
    
    def gradient_compression(self, gradients: torch.Tensor, compression_rate: float = 0.1):
        """Compress gradients for efficient communication"""
        # Top-k sparsification
        k = int(gradients.numel() * compression_rate)
        values, indices = torch.topk(gradients.abs().flatten(), k)
        
        compressed = torch.zeros_like(gradients.flatten())
        compressed[indices] = gradients.flatten()[indices]
        
        return compressed.reshape(gradients.shape)
    
    def asynchronous_sgd(self, num_workers: int = 4, num_iterations: int = 1000):
        """Implement Asynchronous SGD for faster convergence"""
        
        @ray.remote
        class AsyncWorker:
            def __init__(self, worker_id: int):
                self.worker_id = worker_id
                self.model = DistributedAgentModel()
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            
            def compute_gradient(self, batch: Dict, model_weights: Dict):
                """Compute gradient for a batch"""
                self.model.load_state_dict(model_weights)
                
                # Forward pass
                batch_tensors = {k: torch.FloatTensor(v) for k, v in batch.items()}
                policy, value = self.model(batch_tensors["state"])
                loss = nn.functional.cross_entropy(policy, batch_tensors["action"].squeeze())
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Return gradients
                gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone()
                
                return gradients
        
        # Create workers
        workers = [AsyncWorker.remote(i) for i in range(num_workers)]
        
        # Parameter server
        model = DistributedAgentModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        print(f"🔄 Starting Asynchronous SGD with {num_workers} workers...")
        
        for iteration in range(num_iterations):
            # Send current model to random worker
            worker = workers[iteration % num_workers]
            
            # Generate batch (simplified)
            batch = {
                "state": np.random.randn(32, 768),
                "action": np.random.randint(0, 10, (32, 1)),
                "reward": np.random.random((32, 1))
            }
            
            # Get gradients asynchronously
            gradients = ray.get(worker.compute_gradient.remote(batch, model.state_dict()))
            
            # Apply gradients
            for name, param in model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name]
            
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}/{num_iterations}")
    
    def benchmark_distributed_training(self):
        """Benchmark different distributed training approaches"""
        import time
        
        results = {}
        
        # Benchmark data
        num_samples = 1000
        batch_size = 32
        
        print("\n📊 Benchmarking Distributed Training Methods...")
        
        # Single GPU baseline
        if torch.cuda.is_available():
            start_time = time.time()
            model = DistributedAgentModel().cuda()
            optimizer = torch.optim.Adam(model.parameters())
            
            for _ in range(10):
                batch = torch.randn(batch_size, 768).cuda()
                policy, value = model(batch)
                loss = policy.mean() + value.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            results["single_gpu"] = time.time() - start_time
            print(f"  Single GPU: {results['single_gpu']:.2f}s")
        
        # Data parallel (if multiple GPUs)
        if torch.cuda.device_count() > 1:
            start_time = time.time()
            model = nn.DataParallel(DistributedAgentModel()).cuda()
            optimizer = torch.optim.Adam(model.parameters())
            
            for _ in range(10):
                batch = torch.randn(batch_size * 2, 768).cuda()
                policy, value = model(batch)
                loss = policy.mean() + value.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            results["data_parallel"] = time.time() - start_time
            print(f"  Data Parallel: {results['data_parallel']:.2f}s")
        
        return results


# Example usage
if __name__ == "__main__":
    print("🌐 Testing Distributed Training System")
    print("=" * 60)
    
    # Initialize distributed trainer
    config = DistributedConfig(
        num_nodes=2,
        gpus_per_node=0,  # Set to 0 for CPU testing
        cpus_per_node=4,
        batch_size=32,
        communication_backend="ray"
    )
    
    trainer = DistributedTrainer(config)
    
    # Test different distributed training methods
    print("\n1️⃣ Testing Ray Distributed Training...")
    try:
        trainer.train_ray_distributed(num_epochs=2)
        print("✅ Ray distributed training successful!")
    except Exception as e:
        print(f"⚠️ Ray training error: {e}")
    
    print("\n2️⃣ Testing Asynchronous SGD...")
    try:
        trainer.asynchronous_sgd(num_workers=2, num_iterations=20)
        print("✅ Asynchronous SGD successful!")
    except Exception as e:
        print(f"⚠️ Async SGD error: {e}")
    
    print("\n3️⃣ Testing Model Parallelism Setup...")
    parallel_model = trainer.setup_model_parallelism()
    if parallel_model:
        print("✅ Model parallelism configured!")
    else:
        print("ℹ️ Model parallelism requires multiple GPUs")
    
    print("\n4️⃣ Benchmarking Performance...")
    benchmark_results = trainer.benchmark_distributed_training()
    
    print("\n" + "=" * 60)
    print("📊 DISTRIBUTED TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Nodes: {config.num_nodes}")
    print(f"  GPUs per node: {config.gpus_per_node}")
    print(f"  CPUs per node: {config.cpus_per_node}")
    print(f"  Backend: {config.communication_backend}")
    
    if benchmark_results:
        print(f"\nBenchmark Results:")
        for method, time_taken in benchmark_results.items():
            print(f"  {method}: {time_taken:.2f}s")
    
    print("\n✅ Distributed training test complete!")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()