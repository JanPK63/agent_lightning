#!/usr/bin/env python3
"""
Visual AI Assistant for Agent Lightning
AI-powered pattern learning, optimization suggestions, auto-completion, and block prediction
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import uuid
from collections import defaultdict, Counter
from pathlib import Path
import hashlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_code_builder import (
    VisualBlock,
    BlockType,
    VisualProgram,
    BlockFactory
)


class PatternType(Enum):
    """Types of patterns the AI can learn"""
    SEQUENCE = "sequence"           # Common block sequences
    STRUCTURE = "structure"          # Program structures
    CONNECTION = "connection"        # Connection patterns
    PROPERTY = "property"            # Property configurations
    ERROR = "error"                  # Common error patterns
    OPTIMIZATION = "optimization"    # Optimization opportunities


@dataclass
class Pattern:
    """Represents a learned pattern"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType = PatternType.SEQUENCE
    frequency: int = 0
    confidence: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def update_frequency(self):
        """Update pattern frequency and last seen time"""
        self.frequency += 1
        self.last_seen = datetime.now()
        self.confidence = min(1.0, self.frequency / 10.0)  # Simple confidence calculation


@dataclass
class Suggestion:
    """Represents an optimization or improvement suggestion"""
    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    severity: str = "info"  # info, warning, error, optimization
    affected_blocks: List[str] = field(default_factory=list)
    suggested_action: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    pattern_id: Optional[str] = None


@dataclass
class PredictionResult:
    """Result of block prediction"""
    predicted_block: Optional[VisualBlock] = None
    confidence: float = 0.0
    alternatives: List[Tuple[VisualBlock, float]] = field(default_factory=list)
    reasoning: str = ""


class PatternLearner:
    """Learns patterns from visual programs"""
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.sequence_patterns: List[List[BlockType]] = []
        self.connection_patterns: Dict[str, List[str]] = defaultdict(list)
        self.property_patterns: Dict[BlockType, Dict[str, Any]] = defaultdict(dict)
        self.pattern_db_path = Path("ai_patterns.db")
        self.load_patterns()
    
    def learn_from_program(self, program: VisualProgram):
        """Learn patterns from a visual program"""
        # Learn sequence patterns
        self._learn_sequences(program)
        
        # Learn structure patterns
        self._learn_structures(program)
        
        # Learn connection patterns
        self._learn_connections(program)
        
        # Learn property patterns
        self._learn_properties(program)
        
        # Save learned patterns
        self.save_patterns()
    
    def _learn_sequences(self, program: VisualProgram):
        """Learn common block sequences"""
        blocks = program.get_execution_order()
        
        # Extract sequences of 2-5 blocks
        for length in range(2, min(6, len(blocks) + 1)):
            for i in range(len(blocks) - length + 1):
                sequence = [blocks[j].block_type for j in range(i, i + length)]
                sequence_key = self._sequence_to_key(sequence)
                
                if sequence_key in self.patterns:
                    self.patterns[sequence_key].update_frequency()
                else:
                    pattern = Pattern(
                        pattern_type=PatternType.SEQUENCE,
                        data={"sequence": sequence, "length": length},
                        frequency=1
                    )
                    self.patterns[sequence_key] = pattern
    
    def _learn_structures(self, program: VisualProgram):
        """Learn program structure patterns"""
        structure = {
            "block_count": len(program.blocks),
            "connection_count": len(program.connections),
            "block_types": Counter([b.block_type.value for b in program.blocks]),
            "has_loops": any(b.block_type in [BlockType.FOR_LOOP, BlockType.WHILE_LOOP] for b in program.blocks),
            "has_conditionals": any(b.block_type == BlockType.IF_CONDITION for b in program.blocks),
            "has_error_handling": any(b.block_type == BlockType.TRY_CATCH for b in program.blocks)
        }
        
        structure_key = self._structure_to_key(structure)
        
        if structure_key in self.patterns:
            self.patterns[structure_key].update_frequency()
        else:
            pattern = Pattern(
                pattern_type=PatternType.STRUCTURE,
                data=structure,
                frequency=1
            )
            self.patterns[structure_key] = pattern
    
    def _learn_connections(self, program: VisualProgram):
        """Learn connection patterns between blocks"""
        for connection in program.connections:
            source_block = program.get_block(connection.source_id)
            target_block = program.get_block(connection.target_id)
            
            if source_block and target_block:
                connection_key = f"{source_block.block_type.value}->{target_block.block_type.value}"
                
                if connection_key in self.patterns:
                    self.patterns[connection_key].update_frequency()
                else:
                    pattern = Pattern(
                        pattern_type=PatternType.CONNECTION,
                        data={
                            "source_type": source_block.block_type.value,
                            "target_type": target_block.block_type.value,
                            "port_types": (connection.source_port, connection.target_port)
                        },
                        frequency=1
                    )
                    self.patterns[connection_key] = pattern
    
    def _learn_properties(self, program: VisualProgram):
        """Learn common property configurations"""
        for block in program.blocks:
            for prop_name, prop_value in block.properties.items():
                prop_key = f"{block.block_type.value}:{prop_name}"
                
                if prop_key not in self.property_patterns[block.block_type]:
                    self.property_patterns[block.block_type][prop_key] = []
                
                # Store property values for pattern analysis
                self.property_patterns[block.block_type][prop_key].append(prop_value)
    
    def _sequence_to_key(self, sequence: List[BlockType]) -> str:
        """Convert sequence to unique key"""
        return "SEQ:" + "->".join([bt.value if isinstance(bt, BlockType) else str(bt) for bt in sequence])
    
    def _structure_to_key(self, structure: Dict[str, Any]) -> str:
        """Convert structure to unique key"""
        # Create a deterministic hash of the structure
        structure_str = json.dumps(structure, sort_keys=True)
        return "STRUCT:" + hashlib.md5(structure_str.encode()).hexdigest()[:16]
    
    def save_patterns(self):
        """Save learned patterns to disk"""
        try:
            with open(self.pattern_db_path, 'wb') as f:
                pickle.dump({
                    'patterns': self.patterns,
                    'property_patterns': dict(self.property_patterns)
                }, f)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    def load_patterns(self):
        """Load patterns from disk"""
        if self.pattern_db_path.exists():
            try:
                with open(self.pattern_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data.get('patterns', {})
                    self.property_patterns = defaultdict(dict, data.get('property_patterns', {}))
            except Exception as e:
                print(f"Error loading patterns: {e}")
    
    def get_relevant_patterns(self, context: Dict[str, Any]) -> List[Pattern]:
        """Get patterns relevant to current context"""
        relevant = []
        
        # Filter patterns by confidence and recency
        for pattern in self.patterns.values():
            if pattern.confidence > 0.3:  # Minimum confidence threshold
                days_old = (datetime.now() - pattern.last_seen).days
                if days_old < 30:  # Recent patterns only
                    relevant.append(pattern)
        
        # Sort by confidence and frequency
        relevant.sort(key=lambda p: (p.confidence, p.frequency), reverse=True)
        return relevant[:20]  # Return top 20 patterns


class OptimizationAnalyzer:
    """Analyzes programs for optimization opportunities"""
    
    def __init__(self, pattern_learner: PatternLearner):
        self.pattern_learner = pattern_learner
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load optimization rules"""
        return [
            {
                "name": "Remove Redundant Loops",
                "condition": lambda p: self._has_redundant_loops(p),
                "suggestion": "Combine multiple sequential loops over the same data",
                "severity": "optimization"
            },
            {
                "name": "Optimize Database Queries",
                "condition": lambda p: self._has_multiple_db_queries(p),
                "suggestion": "Batch database queries to reduce round trips",
                "severity": "optimization"
            },
            {
                "name": "Add Error Handling",
                "condition": lambda p: self._lacks_error_handling(p),
                "suggestion": "Add try-catch blocks for error-prone operations",
                "severity": "warning"
            },
            {
                "name": "Parallelize Independent Operations",
                "condition": lambda p: self._has_parallelizable_operations(p),
                "suggestion": "These operations can run in parallel for better performance",
                "severity": "optimization"
            },
            {
                "name": "Cache Repeated Calculations",
                "condition": lambda p: self._has_repeated_calculations(p),
                "suggestion": "Cache the results of expensive repeated calculations",
                "severity": "optimization"
            },
            {
                "name": "Simplify Nested Conditions",
                "condition": lambda p: self._has_complex_conditions(p),
                "suggestion": "Simplify nested if-else blocks for better readability",
                "severity": "info"
            }
        ]
    
    def analyze_program(self, program: VisualProgram) -> List[Suggestion]:
        """Analyze program for optimization opportunities"""
        suggestions = []
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            if rule["condition"](program):
                suggestion = Suggestion(
                    title=rule["name"],
                    description=rule["suggestion"],
                    severity=rule["severity"],
                    confidence=0.8
                )
                suggestions.append(suggestion)
        
        # Pattern-based optimizations
        patterns = self.pattern_learner.get_relevant_patterns({"program": program})
        for pattern in patterns:
            if pattern.pattern_type == PatternType.OPTIMIZATION:
                suggestion = Suggestion(
                    title="Pattern-based Optimization",
                    description=f"Based on learned patterns: {pattern.data.get('description', '')}",
                    severity="optimization",
                    pattern_id=pattern.pattern_id,
                    confidence=pattern.confidence
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _has_redundant_loops(self, program: VisualProgram) -> bool:
        """Check for redundant loops"""
        loops = [b for b in program.blocks if b.block_type in [BlockType.FOR_LOOP, BlockType.WHILE_LOOP]]
        if len(loops) < 2:
            return False
        
        # Check if loops iterate over same data
        for i in range(len(loops) - 1):
            if loops[i].properties.get("items_expression") == loops[i+1].properties.get("items_expression"):
                return True
        return False
    
    def _has_multiple_db_queries(self, program: VisualProgram) -> bool:
        """Check for multiple database queries that could be batched"""
        db_blocks = [b for b in program.blocks if b.block_type == BlockType.DATABASE_QUERY]
        return len(db_blocks) > 2
    
    def _lacks_error_handling(self, program: VisualProgram) -> bool:
        """Check if program lacks error handling"""
        has_try_catch = any(b.block_type == BlockType.TRY_CATCH for b in program.blocks)
        has_risky_ops = any(b.block_type in [BlockType.API_CALL, BlockType.DATABASE_QUERY, BlockType.FILE_READ] 
                            for b in program.blocks)
        return has_risky_ops and not has_try_catch
    
    def _has_parallelizable_operations(self, program: VisualProgram) -> bool:
        """Check for operations that could run in parallel"""
        api_calls = [b for b in program.blocks if b.block_type == BlockType.API_CALL]
        # Multiple API calls to different endpoints could be parallelized
        if len(api_calls) > 1:
            urls = [b.properties.get("url") for b in api_calls]
            return len(set(urls)) > 1
        return False
    
    def _has_repeated_calculations(self, program: VisualProgram) -> bool:
        """Check for repeated calculations"""
        expressions = [b for b in program.blocks if b.block_type == BlockType.EXPRESSION]
        if len(expressions) < 2:
            return False
        
        # Check for duplicate expressions
        expr_values = [b.properties.get("expression") for b in expressions]
        return len(expr_values) != len(set(expr_values))
    
    def _has_complex_conditions(self, program: VisualProgram) -> bool:
        """Check for overly complex conditional logic"""
        if_blocks = [b for b in program.blocks if b.block_type == BlockType.IF_CONDITION]
        # More than 3 nested conditions is considered complex
        return len(if_blocks) > 3


class AutoCompleter:
    """Auto-completes visual flows based on patterns"""
    
    def __init__(self, pattern_learner: PatternLearner):
        self.pattern_learner = pattern_learner
        self.factory = BlockFactory()
    
    def suggest_completions(self, program: VisualProgram, 
                           current_block: Optional[VisualBlock] = None) -> List[VisualBlock]:
        """Suggest blocks to complete the current flow"""
        suggestions = []
        
        if not current_block and program.blocks:
            current_block = program.blocks[-1]  # Use last block if not specified
        
        if not current_block:
            # Suggest starting blocks
            return self._suggest_starting_blocks()
        
        # Get relevant sequence patterns
        current_sequence = self._get_current_sequence(program, current_block)
        matching_patterns = self._find_matching_sequences(current_sequence)
        
        # Generate suggestions based on patterns
        for pattern in matching_patterns[:5]:  # Top 5 matches
            next_block_type = self._get_next_in_pattern(pattern, current_sequence)
            if next_block_type:
                suggested_block = self._create_block_from_type(next_block_type)
                if suggested_block:
                    suggestions.append(suggested_block)
        
        # Add context-aware suggestions
        context_suggestions = self._get_context_suggestions(program, current_block)
        suggestions.extend(context_suggestions)
        
        # Remove duplicates and return
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        return unique_suggestions[:10]  # Return top 10 suggestions
    
    def _suggest_starting_blocks(self) -> List[VisualBlock]:
        """Suggest blocks to start a program"""
        suggestions = [
            self.factory.create_function_block(),
            self.factory.create_variable_block(),
            self.factory.create_file_read_block(),
            self.factory.create_api_call_block(),
            self.factory.create_database_query_block()
        ]
        return suggestions
    
    def _get_current_sequence(self, program: VisualProgram, 
                             current_block: VisualBlock, 
                             max_length: int = 5) -> List[BlockType]:
        """Get the sequence of blocks leading to current block"""
        sequence = []
        blocks = program.get_execution_order()
        
        current_idx = None
        for i, block in enumerate(blocks):
            if block.block_id == current_block.block_id:
                current_idx = i
                break
        
        if current_idx is not None:
            start_idx = max(0, current_idx - max_length + 1)
            sequence = [blocks[i].block_type for i in range(start_idx, current_idx + 1)]
        
        return sequence
    
    def _find_matching_sequences(self, current_sequence: List[BlockType]) -> List[Pattern]:
        """Find patterns matching the current sequence"""
        matching = []
        
        for pattern_key, pattern in self.pattern_learner.patterns.items():
            if pattern.pattern_type == PatternType.SEQUENCE:
                pattern_seq = pattern.data.get("sequence", [])
                
                # Check if current sequence matches start of pattern
                if len(pattern_seq) > len(current_sequence):
                    if pattern_seq[:len(current_sequence)] == current_sequence:
                        matching.append(pattern)
        
        # Sort by confidence and frequency
        matching.sort(key=lambda p: (p.confidence, p.frequency), reverse=True)
        return matching
    
    def _get_next_in_pattern(self, pattern: Pattern, 
                           current_sequence: List[BlockType]) -> Optional[BlockType]:
        """Get the next block type in a pattern"""
        pattern_seq = pattern.data.get("sequence", [])
        
        if len(pattern_seq) > len(current_sequence):
            next_idx = len(current_sequence)
            if next_idx < len(pattern_seq):
                return pattern_seq[next_idx]
        
        return None
    
    def _create_block_from_type(self, block_type: BlockType) -> Optional[VisualBlock]:
        """Create a visual block from block type"""
        type_to_factory = {
            BlockType.FUNCTION_DEF: self.factory.create_function_block,
            BlockType.VARIABLE: self.factory.create_variable_block,
            BlockType.IF_CONDITION: self.factory.create_if_block,
            BlockType.FOR_LOOP: self.factory.create_for_loop_block,
            BlockType.WHILE_LOOP: self.factory.create_while_loop_block,
            BlockType.API_CALL: self.factory.create_api_call_block,
            BlockType.DATABASE_QUERY: self.factory.create_database_query_block,
            BlockType.FILE_READ: self.factory.create_file_read_block,
            BlockType.EXPRESSION: self.factory.create_expression_block,
            BlockType.TRY_CATCH: self.factory.create_try_catch_block,
            BlockType.RETURN: self.factory.create_return_block,
            BlockType.OUTPUT: self.factory.create_output_block
        }
        
        factory_method = type_to_factory.get(block_type)
        if factory_method:
            return factory_method()
        return None
    
    def _get_context_suggestions(self, program: VisualProgram, 
                                current_block: VisualBlock) -> List[VisualBlock]:
        """Get context-aware suggestions"""
        suggestions = []
        
        # If current block is a function, suggest return
        if current_block.block_type == BlockType.FUNCTION_DEF:
            suggestions.append(self.factory.create_return_block())
        
        # If current block reads data, suggest processing
        if current_block.block_type in [BlockType.FILE_READ, BlockType.DATABASE_QUERY]:
            suggestions.append(self.factory.create_for_loop_block())
            suggestions.append(self.factory.create_expression_block())
        
        # If current block is API call, suggest error handling
        if current_block.block_type == BlockType.API_CALL:
            suggestions.append(self.factory.create_try_catch_block())
        
        # If no error handling exists, suggest it
        if not any(b.block_type == BlockType.TRY_CATCH for b in program.blocks):
            if current_block.block_type in [BlockType.API_CALL, BlockType.DATABASE_QUERY]:
                suggestions.append(self.factory.create_try_catch_block())
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[VisualBlock]) -> List[VisualBlock]:
        """Remove duplicate suggestions"""
        seen_types = set()
        unique = []
        
        for block in suggestions:
            if block.block_type not in seen_types:
                unique.append(block)
                seen_types.add(block.block_type)
        
        return unique


class BlockPredictor:
    """Predicts next blocks based on context"""
    
    def __init__(self, pattern_learner: PatternLearner, auto_completer: AutoCompleter):
        self.pattern_learner = pattern_learner
        self.auto_completer = auto_completer
        self.factory = BlockFactory()
    
    def predict_next_block(self, program: VisualProgram) -> PredictionResult:
        """Predict the most likely next block"""
        
        if not program.blocks:
            # Predict first block
            predicted = self.factory.create_function_block()
            return PredictionResult(
                predicted_block=predicted,
                confidence=0.9,
                reasoning="Starting with a function block is the most common pattern"
            )
        
        # Get current context
        last_block = program.blocks[-1]
        sequence = self.auto_completer._get_current_sequence(program, last_block)
        
        # Find matching patterns
        patterns = self.auto_completer._find_matching_sequences(sequence)
        
        if patterns:
            # Use the best matching pattern
            best_pattern = patterns[0]
            next_type = self.auto_completer._get_next_in_pattern(best_pattern, sequence)
            
            if next_type:
                predicted = self.auto_completer._create_block_from_type(next_type)
                
                # Get alternatives
                alternatives = []
                for pattern in patterns[1:4]:  # Next 3 best patterns
                    alt_type = self.auto_completer._get_next_in_pattern(pattern, sequence)
                    if alt_type and alt_type != next_type:
                        alt_block = self.auto_completer._create_block_from_type(alt_type)
                        if alt_block:
                            alternatives.append((alt_block, pattern.confidence))
                
                return PredictionResult(
                    predicted_block=predicted,
                    confidence=best_pattern.confidence,
                    alternatives=alternatives,
                    reasoning=f"Based on pattern seen {best_pattern.frequency} times"
                )
        
        # Fallback to context-based prediction
        context_suggestions = self.auto_completer._get_context_suggestions(program, last_block)
        
        if context_suggestions:
            return PredictionResult(
                predicted_block=context_suggestions[0],
                confidence=0.6,
                alternatives=[(b, 0.5) for b in context_suggestions[1:3]],
                reasoning="Context-based prediction"
            )
        
        # Default prediction
        return PredictionResult(
            predicted_block=self.factory.create_expression_block(),
            confidence=0.3,
            reasoning="Default suggestion when no patterns match"
        )
    
    def predict_connections(self, program: VisualProgram, 
                          new_block: VisualBlock) -> List[Tuple[str, str, float]]:
        """Predict connections for a new block"""
        predictions = []
        
        # Look for connection patterns
        for pattern_key, pattern in self.pattern_learner.patterns.items():
            if pattern.pattern_type == PatternType.CONNECTION:
                source_type = pattern.data.get("source_type")
                target_type = pattern.data.get("target_type")
                
                # Check if new block matches pattern
                for existing_block in program.blocks:
                    if existing_block.block_type.value == source_type and \
                       new_block.block_type.value == target_type:
                        predictions.append((
                            existing_block.block_id,
                            new_block.block_id,
                            pattern.confidence
                        ))
                    elif existing_block.block_type.value == target_type and \
                         new_block.block_type.value == source_type:
                        predictions.append((
                            new_block.block_id,
                            existing_block.block_id,
                            pattern.confidence
                        ))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions[:5]  # Return top 5 predictions


class VisualAIAssistant:
    """Main AI assistant for visual code building"""
    
    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.optimization_analyzer = OptimizationAnalyzer(self.pattern_learner)
        self.auto_completer = AutoCompleter(self.pattern_learner)
        self.block_predictor = BlockPredictor(self.pattern_learner, self.auto_completer)
        self.learning_enabled = True
        self.suggestion_history: List[Suggestion] = []
    
    def learn_from_program(self, program: VisualProgram):
        """Learn patterns from a program"""
        if self.learning_enabled:
            self.pattern_learner.learn_from_program(program)
            print(f"🧠 Learned patterns from program: {program.name}")
    
    def suggest_optimizations(self, program: VisualProgram) -> List[Suggestion]:
        """Suggest optimizations for a program"""
        suggestions = self.optimization_analyzer.analyze_program(program)
        self.suggestion_history.extend(suggestions)
        return suggestions
    
    def auto_complete(self, program: VisualProgram, 
                     current_block: Optional[VisualBlock] = None) -> List[VisualBlock]:
        """Auto-complete the visual flow"""
        return self.auto_completer.suggest_completions(program, current_block)
    
    def predict_next(self, program: VisualProgram) -> PredictionResult:
        """Predict the next block"""
        return self.block_predictor.predict_next_block(program)
    
    def predict_connections(self, program: VisualProgram, 
                          new_block: VisualBlock) -> List[Tuple[str, str, float]]:
        """Predict connections for a new block"""
        return self.block_predictor.predict_connections(program, new_block)
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        stats = {
            "total_patterns": len(self.pattern_learner.patterns),
            "pattern_types": Counter([p.pattern_type.value for p in self.pattern_learner.patterns.values()]),
            "high_confidence_patterns": sum(1 for p in self.pattern_learner.patterns.values() if p.confidence > 0.7),
            "recent_patterns": sum(1 for p in self.pattern_learner.patterns.values() 
                                 if (datetime.now() - p.last_seen).days < 7)
        }
        return stats
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable pattern learning"""
        self.learning_enabled = enabled
        print(f"📚 Learning {'enabled' if enabled else 'disabled'}")
    
    def reset_patterns(self):
        """Reset all learned patterns"""
        self.pattern_learner.patterns.clear()
        self.pattern_learner.property_patterns.clear()
        self.pattern_learner.save_patterns()
        print("🔄 All patterns reset")


def test_ai_assistant():
    """Test the AI assistant"""
    print("\n" + "="*60)
    print("Visual AI Assistant Test")
    print("="*60)
    
    assistant = VisualAIAssistant()
    
    # Create sample programs to learn from
    factory = BlockFactory()
    
    # Program 1: API to Database
    program1 = VisualProgram(name="API to Database")
    
    func = factory.create_function_block()
    program1.add_block(func)
    
    api_call = factory.create_api_call_block()
    program1.add_block(api_call)
    
    try_catch = factory.create_try_catch_block()
    program1.add_block(try_catch)
    
    db_query = factory.create_database_query_block()
    program1.add_block(db_query)
    
    program1.connect_blocks(func.block_id, "output", api_call.block_id, "input")
    program1.connect_blocks(api_call.block_id, "output", try_catch.block_id, "input")
    program1.connect_blocks(try_catch.block_id, "output", db_query.block_id, "input")
    
    # Learn from program
    print("\n📚 Learning from sample programs...")
    assistant.learn_from_program(program1)
    
    # Program 2: File Processing
    program2 = VisualProgram(name="File Processing")
    
    file_read = factory.create_file_read_block()
    program2.add_block(file_read)
    
    for_loop = factory.create_for_loop_block()
    program2.add_block(for_loop)
    
    expression = factory.create_expression_block()
    program2.add_block(expression)
    
    output = factory.create_output_block()
    program2.add_block(output)
    
    program2.connect_blocks(file_read.block_id, "output", for_loop.block_id, "input")
    program2.connect_blocks(for_loop.block_id, "output", expression.block_id, "input")
    program2.connect_blocks(expression.block_id, "output", output.block_id, "input")
    
    assistant.learn_from_program(program2)
    
    # Get pattern statistics
    stats = assistant.get_pattern_statistics()
    print(f"\n📊 Pattern Statistics:")
    print(f"   Total Patterns: {stats['total_patterns']}")
    print(f"   Pattern Types: {dict(stats['pattern_types'])}")
    print(f"   High Confidence: {stats['high_confidence_patterns']}")
    
    # Test optimization suggestions
    print(f"\n🔍 Analyzing for optimizations...")
    suggestions = assistant.suggest_optimizations(program1)
    for suggestion in suggestions:
        print(f"   💡 {suggestion.title}: {suggestion.description}")
    
    # Test auto-completion
    print(f"\n🤖 Testing auto-completion...")
    new_program = VisualProgram(name="Test Program")
    start_block = factory.create_function_block()
    new_program.add_block(start_block)
    
    completions = assistant.auto_complete(new_program, start_block)
    print(f"   Suggested {len(completions)} completions:")
    for block in completions[:3]:
        print(f"     • {block.block_type.value}: {block.title}")
    
    # Test prediction
    print(f"\n🔮 Testing block prediction...")
    prediction = assistant.predict_next(new_program)
    if prediction.predicted_block:
        print(f"   Predicted: {prediction.predicted_block.block_type.value}")
        print(f"   Confidence: {prediction.confidence:.2%}")
        print(f"   Reasoning: {prediction.reasoning}")
    
    # Test connection prediction
    if prediction.predicted_block:
        new_program.add_block(prediction.predicted_block)
        connections = assistant.predict_connections(new_program, prediction.predicted_block)
        if connections:
            print(f"\n🔗 Predicted connections:")
            for source_id, target_id, confidence in connections[:3]:
                print(f"   • Connection with {confidence:.2%} confidence")
    
    return assistant


if __name__ == "__main__":
    print("Visual AI Assistant for Agent Lightning")
    print("="*60)
    
    assistant = test_ai_assistant()
    
    print("\n✅ AI Assistant ready!")
    print("\nFeatures:")
    print("  • Pattern learning from visual programs")
    print("  • Optimization suggestions")
    print("  • Auto-completion of visual flows")
    print("  • Next block prediction")
    print("  • Connection prediction")
    print("  • Adaptive learning over time")