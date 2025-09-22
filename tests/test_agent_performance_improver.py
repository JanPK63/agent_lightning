#!/usr/bin/env python3
"""
Unit tests for Agent Performance Improver Module

Tests the functionality of the agent performance improvement system,
including knowledge gap analysis, improvement suggestions, and
integration with the capability matcher.

Author: Agent Lightning Team
"""

import pytest
from unittest.mock import Mock, patch
from agent_performance_improver import (
    AgentPerformanceImprover,
    KnowledgeGap,
    ImprovementSuggestion,
    PerformanceAnalysis,
    ImprovementType,
    get_performance_improver
)


class TestAgentPerformanceImprover:
    """Test cases for the AgentPerformanceImprover class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.improver = AgentPerformanceImprover()

    def test_initialization(self):
        """Test that the improver initializes correctly"""
        assert self.improver.improvement_templates is not None
        assert self.improver.knowledge_domains is not None
        assert len(self.improver.improvement_templates) > 0
        assert len(self.improver.knowledge_domains) > 0

    def test_analyze_performance_gap_high_confidence(self):
        """Test analysis of a high-confidence task"""
        task = "Create a simple hello world web page"
        agent_id = "web_developer"
        confidence = 0.85

        analysis = self.improver.analyze_performance_gap(
            task, agent_id, confidence)

        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.task_description == task
        assert analysis.current_confidence == confidence
        assert analysis.assigned_agent == agent_id
        assert isinstance(analysis.knowledge_gaps, list)
        assert isinstance(analysis.improvement_suggestions, list)
        assert analysis.overall_recommendation is not None

    def test_analyze_performance_gap_low_confidence(self):
        """Test analysis of a low-confidence task"""
        task = "Implement quantum computing algorithms"
        agent_id = "data_analyst"
        confidence = 0.25

        analysis = self.improver.analyze_performance_gap(
            task, agent_id, confidence)

        assert analysis.current_confidence == confidence
        assert len(analysis.knowledge_gaps) > 0
        assert len(analysis.improvement_suggestions) > 0
        assert "quantum" in analysis.overall_recommendation.lower()

    def test_identify_knowledge_gaps_web_task(self):
        """Test knowledge gap identification for web development task"""
        task = "Build a React dashboard with authentication"
        agent_id = "web_developer"

        gaps = self.improver._identify_knowledge_gaps(task, agent_id)

        # Should identify gaps in React and authentication
        gap_domains = [gap.domain for gap in gaps]
        assert "web_development" in gap_domains

        # Check that missing concepts are identified
        web_gap = next((gap for gap in gaps if gap.domain == "web_development"), None)
        if web_gap:
            assert len(web_gap.missing_concepts) > 0

    def test_identify_knowledge_gaps_security_task(self):
        """Test knowledge gap identification for security task"""
        task = "Perform penetration testing on API"
        agent_id = "security_expert"

        gaps = self.improver._identify_knowledge_gaps(task, agent_id)

        # Should identify security-related gaps
        gap_domains = [gap.domain for gap in gaps]
        assert "security" in gap_domains

    def test_generate_improvement_suggestions(self):
        """Test generation of improvement suggestions"""
        task = "Create a machine learning model"
        agent_id = "data_analyst"
        knowledge_gaps = [
            KnowledgeGap(
                domain="data_science",
                missing_concepts=["tensorflow", "neural networks"],
                related_keywords=["machine learning", "model"],
                confidence_impact=0.2,
                priority="high"
            )
        ]

        suggestions = self.improver._generate_improvement_suggestions(
            task, agent_id, knowledge_gaps)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Check that suggestions have required fields
        for suggestion in suggestions:
            assert isinstance(suggestion, ImprovementSuggestion)
            assert suggestion.title is not None
            assert suggestion.description is not None
            assert len(suggestion.implementation_steps) > 0
            assert suggestion.expected_confidence_gain > 0

    def test_create_overall_recommendation_critical_gaps(self):
        """Test recommendation creation for critical knowledge gaps"""
        confidence = 0.15
        knowledge_gaps = [
            KnowledgeGap(
                domain="security",
                missing_concepts=["encryption", "authentication"],
                related_keywords=[],
                confidence_impact=0.3,
                priority="high"
            )
        ]
        suggestions = [
            ImprovementSuggestion(
                improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                title="Add Security Knowledge",
                description="Add security concepts",
                implementation_steps=["Step 1", "Step 2"],
                expected_confidence_gain=0.2,
                effort_estimate="high"
            )
        ]

        recommendation = self.improver._create_overall_recommendation(
            confidence, knowledge_gaps, suggestions)

        assert "critical" in recommendation.lower()
        assert "security" in recommendation.lower()

    def test_create_overall_recommendation_moderate_gaps(self):
        """Test recommendation creation for moderate knowledge gaps"""
        confidence = 0.45
        knowledge_gaps = [
            KnowledgeGap(
                domain="web_development",
                missing_concepts=["react", "api"],
                related_keywords=[],
                confidence_impact=0.15,
                priority="medium"
            )
        ]
        suggestions = [
            ImprovementSuggestion(
                improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                title="Add Framework Knowledge",
                description="Add framework concepts",
                implementation_steps=["Step 1"],
                expected_confidence_gain=0.15,
                effort_estimate="medium"
            )
        ]

        recommendation = self.improver._create_overall_recommendation(
            confidence, knowledge_gaps, suggestions)

        assert "moderate" in recommendation.lower()

    def test_estimate_improvement_time(self):
        """Test improvement time estimation"""
        # Test high effort suggestions
        high_effort_suggestions = [
            ImprovementSuggestion(
                improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                title="Test",
                description="Test",
                implementation_steps=["Step"],
                expected_confidence_gain=0.1,
                effort_estimate="high"
            )
        ]

        time_estimate = self.improver._estimate_improvement_time(
            high_effort_suggestions)
        assert "2-4 weeks" in time_estimate

        # Test medium effort suggestions
        medium_effort_suggestions = [
            ImprovementSuggestion(
                improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                title="Test",
                description="Test",
                implementation_steps=["Step"],
                expected_confidence_gain=0.1,
                effort_estimate="medium"
            )
        ]

        time_estimate = self.improver._estimate_improvement_time(
            medium_effort_suggestions)
        assert "1-2 weeks" in time_estimate

    def test_get_improvement_plan(self):
        """Test improvement plan generation"""
        analysis = PerformanceAnalysis(
            task_description="Test task",
            current_confidence=0.5,
            assigned_agent="test_agent",
            knowledge_gaps=[
                KnowledgeGap(
                    domain="test_domain",
                    missing_concepts=["concept1", "concept2"],
                    related_keywords=["keyword1"],
                    confidence_impact=0.2,
                    priority="high"
                )
            ],
            improvement_suggestions=[
                ImprovementSuggestion(
                    improvement_type=ImprovementType.KNOWLEDGE_ADDITION,
                    title="Test Improvement",
                    description="Test description",
                    implementation_steps=["Step 1", "Step 2"],
                    expected_confidence_gain=0.15,
                    effort_estimate="medium",
                    prerequisites=["Prerequisite 1"],
                    code_examples=["print('example')"]
                )
            ],
            overall_recommendation="Test recommendation",
            estimated_improvement_time="1 week"
        )

        plan = self.improver.get_improvement_plan(analysis)

        assert isinstance(plan, dict)
        assert "task_summary" in plan
        assert "knowledge_gaps" in plan
        assert "improvement_suggestions" in plan
        assert "overall_recommendation" in plan
        assert "estimated_time" in plan
        assert "expected_total_gain" in plan

        # Check task summary
        assert plan["task_summary"]["description"].startswith("Test task")
        assert plan["task_summary"]["current_confidence"] == "0.50"
        assert plan["task_summary"]["assigned_agent"] == "test_agent"

        # Check knowledge gaps
        assert len(plan["knowledge_gaps"]) == 1
        gap = plan["knowledge_gaps"][0]
        assert gap["domain"] == "test_domain"
        assert gap["missing_concepts"] == ["concept1", "concept2"]
        assert gap["priority"] == "high"

        # Check improvement suggestions
        assert len(plan["improvement_suggestions"]) == 1
        suggestion = plan["improvement_suggestions"][0]
        assert suggestion["title"] == "Test Improvement"
        assert suggestion["expected_gain"] == "0.15"
        assert suggestion["effort"] == "medium"

    def test_determine_gap_priority(self):
        """Test gap priority determination"""
        # High priority - many missing concepts
        priority = self.improver._determine_gap_priority(
            "test_domain", ["concept1", "concept2", "concept3", "concept4"])
        assert priority == "high"

        # Medium priority - some missing concepts
        priority = self.improver._determine_gap_priority(
            "test_domain", ["concept1", "concept2"])
        assert priority == "medium"

        # Low priority - few missing concepts
        priority = self.improver._determine_gap_priority(
            "test_domain", ["concept1"])
        assert priority == "low"

    def test_is_related_concept_missing(self):
        """Test related concept missing detection"""
        # Should detect missing concept when domain context exists
        task_text = "build a website with modern features"
        keyword = "react"
        domain = "web_development"

        is_missing = self.improver._is_related_concept_missing(
            task_text, keyword, domain)
        assert is_missing is True

        # Should not detect when no domain context
        task_text = "analyze sales data"
        keyword = "react"
        domain = "web_development"

        is_missing = self.improver._is_related_concept_missing(
            task_text, keyword, domain)
        assert is_missing is False

    @patch('agent_performance_improver.logger')
    def test_error_handling_in_analysis(self, mock_logger):
        """Test error handling in performance analysis"""
        # Test with invalid inputs
        analysis = self.improver.analyze_performance_gap(
            "", "invalid_agent", -0.1)

        assert isinstance(analysis, PerformanceAnalysis)
        # Should handle gracefully and return valid analysis

    def test_get_general_improvements(self):
        """Test general improvement suggestions"""
        task = "Perform unknown task type"
        suggestions = self.improver._get_general_improvements(task)

        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert isinstance(suggestion, ImprovementSuggestion)
            assert suggestion.improvement_type == ImprovementType.KNOWLEDGE_ADDITION
            assert len(suggestion.implementation_steps) > 0


class TestGlobalFunctions:
    """Test cases for global functions"""

    def test_get_performance_improver_singleton(self):
        """Test that get_performance_improver returns singleton"""
        improver1 = get_performance_improver()
        improver2 = get_performance_improver()

        assert improver1 is improver2
        assert isinstance(improver1, AgentPerformanceImprover)


class TestIntegrationWithCapabilityMatcher:
    """Integration tests with capability matcher"""

    @patch('agent_performance_improver.get_performance_improver')
    def test_low_confidence_integration(self, mock_get_improver):
        """Test integration when confidence is low"""
        # Mock the performance improver
        mock_improver = Mock()
        mock_analysis = Mock()
        mock_analysis.improvement_suggestions = [
            Mock(title="Test Improvement", expected_confidence_gain=0.15)
        ]
        mock_plan = {
            "improvement_suggestions": [
                {"title": "Test Improvement", "expected_gain": "0.15"}
            ]
        }

        mock_improver.analyze_performance_gap.return_value = mock_analysis
        mock_improver.get_improvement_plan.return_value = mock_plan
        mock_get_improver.return_value = mock_improver

        # Import and test the capability matcher integration
        from agent_capability_matcher import AgentCapabilityMatcher

        matcher = AgentCapabilityMatcher()

        # This would normally trigger the performance improver
        # when confidence is below threshold
        agent, confidence, reason = matcher.find_best_agent(
            "Implement complex quantum algorithms")

        # Verify the mock was called (if integration is working)
        if hasattr(matcher, '_get_performance_improvements'):
            mock_improver.analyze_performance_gap.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])