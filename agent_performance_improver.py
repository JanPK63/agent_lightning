#!/usr/bin/env python3
"""
Agent Performance Improver Module

This module provides actionable steps to improve agent performance
when confidence scores are too low. It analyzes knowledge gaps,
suggests specific improvements, and provides implementation guidance.

Author: Agent Lightning Team
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ImprovementType(Enum):
    """Types of performance improvements"""
    KNOWLEDGE_ADDITION = "knowledge_addition"
    CAPABILITY_ENHANCEMENT = "capability_enhancement"
    TRAINING_DATA_EXPANSION = "training_data_expansion"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    SYSTEM_INTEGRATION = "system_integration"


@dataclass
class KnowledgeGap:
    """Represents a specific knowledge gap identified in an agent"""
    domain: str
    missing_concepts: List[str]
    related_keywords: List[str]
    confidence_impact: float
    priority: str = "medium"


@dataclass
class ImprovementSuggestion:
    """Represents a specific improvement suggestion"""
    improvement_type: ImprovementType
    title: str
    description: str
    implementation_steps: List[str]
    expected_confidence_gain: float
    effort_estimate: str
    prerequisites: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)


@dataclass
class PerformanceAnalysis:
    """Complete analysis of agent performance for a task"""
    task_description: str
    current_confidence: float
    assigned_agent: str
    knowledge_gaps: List[KnowledgeGap]
    improvement_suggestions: List[ImprovementSuggestion]
    overall_recommendation: str
    estimated_improvement_time: str


class AgentPerformanceImprover:
    """
    Analyzes low-confidence agent assignments and provides
    actionable improvement steps.
    """

    def __init__(self):
        """Initialize the performance improver with templates"""
        self.improvement_templates = self._initialize_templates()
        self.knowledge_domains = self._initialize_domains()
        logger.info("Initialized Agent Performance Improver")

    def _initialize_templates(self) -> Dict[str, List[ImprovementSuggestion]]:
        """Initialize improvement templates"""
        return {
            "web_development": [
                ImprovementSuggestion(
                    improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                    title="Add Modern Frontend Framework Knowledge",
                    description="Enhance agent knowledge of React, Vue frameworks",
                    implementation_steps=[
                        "Add framework-specific keywords to expertise_keywords",
                        "Include framework documentation in training data",
                        "Add code examples for common framework patterns",
                        "Update capability descriptions to include framework support"
                    ],
                    expected_confidence_gain=0.15,
                    effort_estimate="medium",
                    code_examples=[
                        "agent.capabilities.append('react', 'vue', 'angular')",
                        "agent.expertise_keywords.extend(['component', 'routing'])"
                    ]
                )
            ]
        }

    def _initialize_domains(self) -> Dict[str, List[str]]:
        """Initialize knowledge domains and keywords"""
        return {
            "web_development": [
                "html", "css", "javascript", "frontend", "backend",
                "api", "database", "responsive", "framework"
            ],
            "security": [
                "encryption", "authentication", "authorization",
                "vulnerability", "penetration", "audit", "compliance"
            ],
            "data_science": [
                "machine learning", "statistics", "visualization",
                "pandas", "numpy", "scikit-learn", "tensorflow"
            ]
        }

    def analyze_performance_gap(self, task_description: str,
                               agent_id: str, current_confidence: float
                               ) -> PerformanceAnalysis:
        """
        Analyze performance gap for a low-confidence agent assignment.
        """
        logger.info(f"Analyzing performance gap for agent {agent_id}")

        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(
            task_description, agent_id)

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            task_description, agent_id, knowledge_gaps)

        # Create overall recommendation
        overall_recommendation = self._create_overall_recommendation(
            current_confidence, knowledge_gaps, improvement_suggestions)

        # Estimate improvement time
        estimated_time = self._estimate_improvement_time(
            improvement_suggestions)

        return PerformanceAnalysis(
            task_description=task_description,
            current_confidence=current_confidence,
            assigned_agent=agent_id,
            knowledge_gaps=knowledge_gaps,
            improvement_suggestions=improvement_suggestions,
            overall_recommendation=overall_recommendation,
            estimated_improvement_time=estimated_time
        )

    def _identify_knowledge_gaps(self, task_description: str,
                                agent_id: str) -> List[KnowledgeGap]:
        """Identify specific knowledge gaps based on task analysis."""
        gaps = []
        task_lower = task_description.lower()

        # Analyze each knowledge domain
        for domain, keywords in self.knowledge_domains.items():
            missing_concepts = []
            found_keywords = []

            for keyword in keywords:
                if keyword in task_lower:
                    found_keywords.append(keyword)
                else:
                    if self._is_related_concept_missing(
                            task_lower, keyword, domain):
                        missing_concepts.append(keyword)

            if missing_concepts:
                confidence_impact = min(0.3, len(missing_concepts) * 0.05)
                gap = KnowledgeGap(
                    domain=domain,
                    missing_concepts=missing_concepts,
                    related_keywords=found_keywords,
                    confidence_impact=confidence_impact,
                    priority=self._determine_gap_priority(
                        domain, missing_concepts)
                )
                gaps.append(gap)

        return gaps

    def _is_related_concept_missing(self, task_text: str,
                                   keyword: str, domain: str) -> bool:
        """Check if a related concept is missing from the task context"""
        domain_indicators = {
            "web_development": ["website", "web", "application"],
            "security": ["secure", "protect", "vulnerable", "threat"],
            "data_science": ["data", "analyze", "predict", "model"]
        }

        domain_keywords = domain_indicators.get(domain, [])
        has_domain_context = any(indicator in task_text
                                for indicator in domain_keywords)

        return has_domain_context and keyword not in task_text

    def _determine_gap_priority(self, domain: str,
                               missing_concepts: List[str]) -> str:
        """Determine priority of a knowledge gap"""
        if len(missing_concepts) > 3:
            return "high"
        elif len(missing_concepts) > 1:
            return "medium"
        else:
            return "low"

    def _generate_improvement_suggestions(self, task_description: str,
                                         agent_id: str,
                                         knowledge_gaps: List[KnowledgeGap]
                                         ) -> List[ImprovementSuggestion]:
        """Generate specific improvement suggestions"""
        suggestions = []

        # Get domain-specific suggestions
        for gap in knowledge_gaps:
            if gap.domain in self.improvement_templates:
                domain_suggestions = self.improvement_templates[gap.domain]
                suggestions.extend(domain_suggestions)

        # Add general improvements if needed
        if not suggestions:
            suggestions.extend(self._get_general_improvements(
                task_description))

        # Sort by expected confidence gain
        suggestions.sort(key=lambda x: x.expected_confidence_gain,
                        reverse=True)

        return suggestions[:5]  # Return top 5 suggestions

    def _get_general_improvements(self, task_description: str
                                 ) -> List[ImprovementSuggestion]:
        """Get general improvement suggestions"""
        return [
            ImprovementSuggestion(
                improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                title="Expand Task-Specific Knowledge Base",
                description="Add domain-specific knowledge relevant to task",
                implementation_steps=[
                    "Analyze task requirements and identify key concepts",
                    "Add relevant keywords to agent expertise_keywords",
                    "Include task-related documentation in training data",
                    "Update agent capabilities to reflect new knowledge"
                ],
                expected_confidence_gain=0.10,
                effort_estimate="medium"
            )
        ]

    def _create_overall_recommendation(self, current_confidence: float,
                                      knowledge_gaps: List[KnowledgeGap],
                                      suggestions: List[ImprovementSuggestion]
                                      ) -> str:
        """Create an overall recommendation based on analysis"""
        if current_confidence < 0.2:
            priority_gaps = [gap for gap in knowledge_gaps
                           if gap.priority == "high"]
            if priority_gaps:
                domains = ', '.join([g.domain for g in priority_gaps])
                return f"Critical knowledge gaps in {domains}. " \
                       "Immediate improvement recommended."
            else:
                return "Very low confidence detected. " \
                       "Consider reassigning or implementing quick fixes."

        elif current_confidence < 0.4:
            top_suggestion = suggestions[0] if suggestions else None
            if top_suggestion:
                return f"Moderate confidence. " \
                       f"Focus on '{top_suggestion.title}' for best results."
            else:
                return "Moderate confidence gap. " \
                       "Implement suggested improvements."

        else:
            return "Minor confidence gap. " \
                   "Optional improvements available for optimization."

    def _estimate_improvement_time(self,
                                  suggestions: List[ImprovementSuggestion]
                                  ) -> str:
        """Estimate time required for implementing improvements"""
        if not suggestions:
            return "Unknown"

        effort_levels = [s.effort_estimate for s in suggestions]
        high_count = effort_levels.count("high")
        medium_count = effort_levels.count("medium")

        if high_count > 0:
            return "2-4 weeks"
        elif medium_count > 1:
            return "1-2 weeks"
        elif medium_count == 1:
            return "3-7 days"
        else:
            return "1-3 days"

    def get_improvement_plan(self, analysis: PerformanceAnalysis
                            ) -> Dict[str, Any]:
        """Generate a detailed improvement plan"""
        plan = {
            "task_summary": {
                "description": analysis.task_description[:100] + "...",
                "current_confidence": f"{analysis.current_confidence:.2f}",
                "assigned_agent": analysis.assigned_agent
            },
            "knowledge_gaps": [
                {
                    "domain": gap.domain,
                    "missing_concepts": gap.missing_concepts,
                    "confidence_impact": f"{gap.confidence_impact:.2f}",
                    "priority": gap.priority
                }
                for gap in analysis.knowledge_gaps
            ],
            "improvement_suggestions": [
                {
                    "type": suggestion.improvement_type.value,
                    "title": suggestion.title,
                    "description": suggestion.description,
                    "implementation_steps": suggestion.implementation_steps,
                    "expected_gain": f"{suggestion.expected_confidence_gain:.2f}",
                    "effort": suggestion.effort_estimate,
                    "prerequisites": suggestion.prerequisites,
                    "code_examples": suggestion.code_examples
                }
                for suggestion in analysis.improvement_suggestions
            ],
            "overall_recommendation": analysis.overall_recommendation,
            "estimated_time": analysis.estimated_improvement_time,
            "expected_total_gain": f"{sum(s.expected_confidence_gain
                                        for s in analysis.improvement_suggestions):.2f}"
        }

        return plan


# Global instance for easy access
_performance_improver = None

def get_performance_improver() -> AgentPerformanceImprover:
    """Get the global performance improver instance"""
    global _performance_improver
    if _performance_improver is None:
        _performance_improver = AgentPerformanceImprover()
    return _performance_improver


if __name__ == "__main__":
    # Test the performance improver
    improver = AgentPerformanceImprover()

    test_cases = [
        ("Create a React-based dashboard", "web_developer", 0.35),
        ("Perform security audit", "security_expert", 0.25),
        ("Build ML model for segmentation", "data_analyst", 0.40)
    ]

    print("Agent Performance Improver Test Results:")
    print("=" * 50)

    for task, agent, confidence in test_cases:
        print(f"\nTask: {task}")
        print(f"Agent: {agent}")
        print(f"Current Confidence: {confidence:.2f}")
        print("-" * 40)

        analysis = improver.analyze_performance_gap(
            task, agent, confidence)
        plan = improver.get_improvement_plan(analysis)

        print(f"Knowledge Gaps: {len(plan['knowledge_gaps'])}")
        for gap in plan['knowledge_gaps'][:2]:
            print(f"  - {gap['domain']}: {', '.join(gap['missing_concepts'][:3])}")

        print(f"Improvement Suggestions: {len(plan['improvement_suggestions'])}")
        for suggestion in plan['improvement_suggestions'][:2]:
            print(f"  - {suggestion['title']} (Gain: {suggestion['expected_gain']}, Effort: {suggestion['effort']})")

        print(f"Overall Recommendation: {plan['overall_recommendation']}")
        print(f"Estimated Time: {plan['estimated_time']}")
        print(f"Expected Total Gain: {plan['expected_total_gain']}")