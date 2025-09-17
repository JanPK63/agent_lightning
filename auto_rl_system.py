#!/usr/bin/env python3
"""
Intelligent Auto-RL System
Automatically decides when to trigger RL training based on task analysis
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RLDecision(Enum):
    SKIP = "skip"
    LIGHT = "light"  # 2-3 epochs
    STANDARD = "standard"  # 5-7 epochs  
    INTENSIVE = "intensive"  # 10+ epochs

@dataclass
class RLRecommendation:
    decision: RLDecision
    algorithm: str
    epochs: int
    confidence: float
    reasoning: str

class AutoRLAnalyzer:
    """Analyzes tasks to automatically determine RL training needs"""
    
    def __init__(self):
        self.rl_keywords = {
            "optimization": ["optimize", "improve", "enhance", "better", "faster", "efficient"],
            "learning": ["learn", "adapt", "train", "smart", "intelligent", "ai"],
            "performance": ["performance", "speed", "accuracy", "quality", "results"],
            "complex": ["complex", "advanced", "sophisticated", "multi-step", "workflow"],
            "repetitive": ["repeat", "batch", "multiple", "series", "automate", "routine"]
        }
        
        self.task_patterns = {
            "code_optimization": r"(optimize|improve|enhance).*(code|algorithm|performance)",
            "learning_task": r"(learn|adapt|train).*(from|pattern|behavior)",
            "multi_agent": r"(multi|multiple|several).*(agent|task|step)",
            "performance_critical": r"(fast|speed|performance|efficient|optimize)",
            "repetitive_work": r"(batch|multiple|series|repeat|automate)"
        }
    
    def analyze_task(self, task: str, agent_id: str, context: Dict[str, Any] = None) -> RLRecommendation:
        """Analyze task and return RL recommendation"""
        
        task_lower = task.lower()
        score = 0
        reasons = []
        
        # Check for RL-beneficial patterns
        for category, keywords in self.rl_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in task_lower)
            if matches > 0:
                score += matches * 2
                reasons.append(f"{category} indicators ({matches})")
        
        # Check regex patterns
        for pattern_name, pattern in self.task_patterns.items():
            if re.search(pattern, task_lower):
                score += 3
                reasons.append(f"{pattern_name} pattern")
        
        # Agent-specific scoring
        agent_multipliers = {
            "full_stack_developer": 1.2,
            "data_scientist": 1.5,  # High benefit from RL
            "devops_engineer": 1.1,
            "security_expert": 1.3,
            "system_architect": 1.4
        }
        
        if agent_id in agent_multipliers:
            score *= agent_multipliers[agent_id]
            reasons.append(f"agent {agent_id} benefits from RL")
        
        # Task complexity scoring
        complexity_indicators = len(re.findall(r'\b(and|then|after|before|while|if|when)\b', task_lower))
        if complexity_indicators > 2:
            score += complexity_indicators
            reasons.append(f"complex task structure ({complexity_indicators} steps)")
        
        # Length-based scoring
        if len(task.split()) > 20:
            score += 2
            reasons.append("detailed task description")
        
        # Context-based scoring
        if context:
            if context.get("deployment"):
                score += 3
                reasons.append("deployment context")
            if context.get("priority") == "high":
                score += 2
                reasons.append("high priority task")
        
        # Make decision based on score
        if score >= 12:
            decision = RLDecision.INTENSIVE
            algorithm = "ppo"
            epochs = 10
            confidence = min(0.95, score / 15)
        elif score >= 8:
            decision = RLDecision.STANDARD
            algorithm = "ppo"
            epochs = 5
            confidence = min(0.85, score / 12)
        elif score >= 4:
            decision = RLDecision.LIGHT
            algorithm = "ppo"
            epochs = 2
            confidence = min(0.75, score / 8)
        else:
            decision = RLDecision.SKIP
            algorithm = None
            epochs = 0
            confidence = 0.6
        
        reasoning = f"Score: {score:.1f}. " + "; ".join(reasons[:3])
        
        return RLRecommendation(decision, algorithm, epochs, confidence, reasoning)

class AutoRLManager:
    """Manages automatic RL training decisions and execution"""
    
    def __init__(self):
        self.analyzer = AutoRLAnalyzer()
        self.rl_history = {}  # Track RL usage per agent
        self.performance_metrics = {}  # Track performance improvements
    
    def should_use_rl(self, task: str, agent_id: str, context: Dict[str, Any] = None) -> RLRecommendation:
        """Determine if RL should be used for this task"""
        
        recommendation = self.analyzer.analyze_task(task, agent_id, context)
        
        # Check recent RL usage to avoid over-training
        recent_rl = self.rl_history.get(agent_id, 0)
        if recent_rl > 3:  # More than 3 RL sessions recently
            if recommendation.decision != RLDecision.SKIP:
                recommendation.epochs = max(1, recommendation.epochs // 2)
                recommendation.reasoning += "; reduced due to recent RL usage"
        
        # Update history
        if recommendation.decision != RLDecision.SKIP:
            self.rl_history[agent_id] = recent_rl + 1
        
        return recommendation
    
    def create_rl_context(self, recommendation: RLRecommendation) -> Dict[str, Any]:
        """Create RL context for API request"""
        
        if recommendation.decision == RLDecision.SKIP:
            return {}
        
        return {
            "use_rl_training": True,
            "rl_training": {
                "algorithm": recommendation.algorithm,
                "epochs": recommendation.epochs,
                "auto_triggered": True,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning
            }
        }
    
    def get_rl_stats(self) -> Dict[str, Any]:
        """Get RL usage statistics"""
        total_sessions = sum(self.rl_history.values())
        active_agents = len([k for k, v in self.rl_history.items() if v > 0])
        
        return {
            "total_rl_sessions": total_sessions,
            "active_rl_agents": active_agents,
            "rl_history": dict(self.rl_history),
            "performance_metrics": dict(self.performance_metrics)
        }

# Global auto-RL manager instance
auto_rl_manager = AutoRLManager()

def auto_enhance_request(task: str, agent_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Automatically enhance request with RL context if beneficial"""
    
    # Get RL recommendation
    recommendation = auto_rl_manager.should_use_rl(task, agent_id, context)
    
    # Create enhanced context
    enhanced_context = dict(context) if context else {}
    
    if recommendation.decision != RLDecision.SKIP:
        rl_context = auto_rl_manager.create_rl_context(recommendation)
        enhanced_context.update(rl_context)
        
        print(f"🧠 Auto-RL: {recommendation.decision.value} training recommended")
        print(f"   Algorithm: {recommendation.algorithm}, Epochs: {recommendation.epochs}")
        print(f"   Confidence: {recommendation.confidence:.2f}")
        print(f"   Reasoning: {recommendation.reasoning}")
    else:
        print(f"⚪ Auto-RL: Skipping RL training for this task")
    
    return enhanced_context

if __name__ == "__main__":
    # Test the auto-RL system
    test_cases = [
        ("Fix a simple bug in the login function", "full_stack_developer"),
        ("Optimize the database query performance for the user analytics dashboard", "data_scientist"),
        ("Create a comprehensive multi-service architecture with authentication, caching, and monitoring", "system_architect"),
        ("Deploy the application to AWS with auto-scaling and load balancing", "devops_engineer"),
        ("Write a hello world function", "full_stack_developer")
    ]
    
    print("🧪 Testing Auto-RL System...")
    print("=" * 60)
    
    for task, agent in test_cases:
        print(f"\n📋 Task: {task}")
        print(f"🤖 Agent: {agent}")
        
        enhanced_context = auto_enhance_request(task, agent)
        
        if enhanced_context.get("use_rl_training"):
            rl_info = enhanced_context["rl_training"]
            print(f"✅ RL Training: {rl_info['algorithm']} for {rl_info['epochs']} epochs")
        else:
            print("⚪ No RL training needed")
    
    print(f"\n📊 RL Stats: {auto_rl_manager.get_rl_stats()}")