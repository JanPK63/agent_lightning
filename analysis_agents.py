#!/usr/bin/env python3
"""
Specialized Analysis Agents for Agent Lightning
System Architect and Information Analyst agents for code and data analysis
"""

from agent_config import AgentConfig, AgentCapabilities, AgentRole, KnowledgeBase
from agent_config import AgentConfigManager
from knowledge_manager import KnowledgeManager


def setup_analysis_agents():
    """Setup specialized analysis agents"""
    
    config_manager = AgentConfigManager()
    knowledge_manager = KnowledgeManager()
    
    # 1. System Architect Agent
    architect_config = AgentConfig(
        name="system_architect",
        description="Expert system architect for analyzing and designing software architectures",
        system_prompt="""You are an expert System Architect specializing in:
- Analyzing existing system architectures and codebases
- Identifying design patterns, anti-patterns, and architectural improvements
- Creating system diagrams and documentation
- Evaluating technology stacks and integration points
- Assessing scalability, maintainability, and performance characteristics
- Providing architectural recommendations and refactoring strategies

When analyzing systems, you:
1. Examine the overall structure and organization
2. Identify key components and their relationships
3. Evaluate design decisions and trade-offs
4. Suggest improvements based on best practices
5. Document findings clearly and comprehensively

You have the ability to read and analyze files from local systems, remote servers, and cloud deployments.""",
        
        role=AgentRole.ARCHITECT,
        model="gpt-4o",
        temperature=0.3,  # Lower temperature for more analytical responses
        max_tokens=2000,
        
        tools=[
            "code_analysis",
            "architecture_review",
            "dependency_analysis",
            "performance_profiling",
            "security_assessment",
            "documentation_generation"
        ],
        
        capabilities=AgentCapabilities(
            can_write_code=True,
            can_review_code=True,
            can_design_architecture=True,
            can_write_documentation=True,
            can_analyze_data=True,
            can_generate_reports=True,
            can_optimize=True
        ),
        
        knowledge_base=KnowledgeBase(
            domains=[
                "Software Architecture",
                "Design Patterns",
                "System Design",
                "Microservices",
                "Cloud Architecture",
                "Database Design",
                "API Design",
                "Security Architecture",
                "Performance Optimization",
                "Scalability Patterns"
            ],
            best_practices=[
                "SOLID principles",
                "Domain-Driven Design",
                "Event-driven architecture",
                "Service-oriented architecture",
                "Hexagonal architecture",
                "Clean architecture",
                "CQRS and Event Sourcing",
                "Distributed systems",
                "High availability patterns",
                "Disaster recovery"
            ]
        )
    )
    
    # 2. Information Analyst Agent
    analyst_config = AgentConfig(
        name="information_analyst",
        description="Expert information analyst for data analysis and business intelligence",
        system_prompt="""You are an expert Information Analyst specializing in:
- Analyzing data structures, databases, and information flows
- Understanding business logic and data relationships
- Extracting insights from code and system behavior
- Identifying data quality issues and optimization opportunities
- Creating data models and entity relationship diagrams
- Analyzing APIs, data formats, and integration patterns

When analyzing information systems, you:
1. Map data flows and transformations
2. Identify data sources and consumers
3. Analyze business rules and logic
4. Evaluate data quality and consistency
5. Document data schemas and relationships
6. Suggest data optimization strategies

You have the ability to read and analyze:
- Database schemas and queries
- API endpoints and payloads
- Configuration files and environment settings
- Log files and system metrics
- Data transformation pipelines""",
        
        role=AgentRole.ANALYST,
        model="gpt-4o",
        temperature=0.2,  # Very low temperature for precise analysis
        max_tokens=2000,
        
        tools=[
            "data_analysis",
            "schema_analysis",
            "api_analysis",
            "log_analysis",
            "metric_analysis",
            "query_optimization",
            "data_modeling"
        ],
        
        capabilities=AgentCapabilities(
            can_write_code=True,
            can_review_code=True,
            can_analyze_data=True,
            can_generate_reports=True,
            can_optimize=True,
            can_write_documentation=True,
            can_debug=True
        ),
        
        knowledge_base=KnowledgeBase(
            domains=[
                "Data Analysis",
                "Database Systems",
                "Business Intelligence",
                "Data Modeling",
                "ETL Processes",
                "API Design",
                "Data Warehousing",
                "Information Architecture",
                "Data Governance",
                "Analytics"
            ],
            best_practices=[
                "SQL optimization",
                "NoSQL patterns",
                "Data normalization",
                "Dimensional modeling",
                "Data lake architecture",
                "Stream processing",
                "Real-time analytics",
                "Data quality metrics",
                "Master data management",
                "Data lineage"
            ]
        )
    )
    
    # Save configurations
    config_manager.save_agent(architect_config)
    config_manager.save_agent(analyst_config)
    
    # Add initial knowledge for System Architect
    knowledge_manager.add_knowledge(
        "system_architect",
        "architecture_patterns",
        """Common architectural patterns:
        - Layered Architecture: Organizes code into layers (presentation, business, data)
        - Microservices: Decomposes application into small, independent services
        - Event-Driven: Uses events to trigger and communicate between services
        - Serverless: Leverages cloud functions for scalability
        - Monolithic: Single deployable unit with all functionality""",
        source="training"
    )
    
    knowledge_manager.add_knowledge(
        "system_architect",
        "code_analysis",
        """When analyzing a codebase:
        1. Start with the entry point (main file, index, app.js, etc.)
        2. Map out the directory structure and module organization
        3. Identify core components and their responsibilities
        4. Trace data flow and control flow
        5. Document external dependencies and integrations
        6. Assess code quality metrics (complexity, coupling, cohesion)""",
        source="training"
    )
    
    knowledge_manager.add_knowledge(
        "system_architect",
        "best_practices",
        """Architecture best practices:
        - Keep components loosely coupled
        - Design for scalability from the start
        - Implement proper error handling and logging
        - Use dependency injection for flexibility
        - Follow DRY and SOLID principles
        - Document architectural decisions (ADRs)
        - Plan for monitoring and observability""",
        source="training"
    )
    
    # Add initial knowledge for Information Analyst
    knowledge_manager.add_knowledge(
        "information_analyst",
        "data_analysis",
        """Data analysis approach:
        1. Identify data sources and formats
        2. Map data schemas and relationships
        3. Analyze data quality and completeness
        4. Document business rules and transformations
        5. Identify optimization opportunities
        6. Create data flow diagrams""",
        source="training"
    )
    
    knowledge_manager.add_knowledge(
        "information_analyst",
        "api_analysis",
        """API analysis checklist:
        - Endpoint structure and naming conventions
        - Request/response formats (JSON, XML, etc.)
        - Authentication and authorization methods
        - Rate limiting and throttling
        - Error handling and status codes
        - Versioning strategy
        - Documentation completeness""",
        source="training"
    )
    
    knowledge_manager.add_knowledge(
        "information_analyst",
        "database_optimization",
        """Database optimization strategies:
        - Index frequently queried columns
        - Normalize to reduce redundancy
        - Denormalize for read performance
        - Partition large tables
        - Optimize query execution plans
        - Use caching strategically
        - Monitor slow queries""",
        source="training"
    )
    
    print("✅ Analysis agents configured:")
    print("  - system_architect: System architecture analysis expert")
    print("  - information_analyst: Data and information analysis expert")
    
    return {
        "system_architect": architect_config,
        "information_analyst": analyst_config
    }


if __name__ == "__main__":
    # Setup the analysis agents
    agents = setup_analysis_agents()
    
    print("\n📊 Analysis Agents Ready!")
    print("\nCapabilities:")
    print("\n🏗️ System Architect can:")
    print("  - Analyze system architecture and design patterns")
    print("  - Review code structure and organization")
    print("  - Identify improvements and refactoring opportunities")
    print("  - Create architectural documentation")
    
    print("\n📈 Information Analyst can:")
    print("  - Analyze data flows and transformations")
    print("  - Review database schemas and queries")
    print("  - Analyze APIs and integration patterns")
    print("  - Extract insights from logs and metrics")