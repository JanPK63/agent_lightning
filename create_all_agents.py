#!/usr/bin/env python3
"""
Create ALL 31 Specialized Agents with RL Integration
"""

from agent_config import AgentConfigManager, AgentConfig, AgentRole, KnowledgeBase, AgentCapabilities

def create_all_31_agents():
    """Create all 31 specialized agents"""
    config_manager = AgentConfigManager()
    
    agents = [
        # Core Development
        ("full_stack_developer", "Expert full-stack developer", AgentRole.FULL_STACK_DEVELOPER),
        ("mobile_developer", "Mobile development expert", AgentRole.CUSTOM),
        ("frontend_developer", "Frontend specialist", AgentRole.CUSTOM),
        ("backend_developer", "Backend specialist", AgentRole.CUSTOM),
        ("game_developer", "Game development expert", AgentRole.CUSTOM),
        
        # Infrastructure & Operations
        ("devops_engineer", "DevOps specialist", AgentRole.DEVOPS_ENGINEER),
        ("cloud_architect", "Cloud architecture expert", AgentRole.CUSTOM),
        ("site_reliability_engineer", "SRE specialist", AgentRole.CUSTOM),
        ("platform_engineer", "Platform engineering expert", AgentRole.CUSTOM),
        ("network_engineer", "Network specialist", AgentRole.CUSTOM),
        
        # Security & Compliance
        ("security_expert", "Cybersecurity specialist", AgentRole.CUSTOM),
        ("compliance_officer", "Compliance and governance expert", AgentRole.CUSTOM),
        
        # Data & AI
        ("data_scientist", "Data science expert", AgentRole.DATA_SCIENTIST),
        ("ai_ml_engineer", "AI/ML engineering specialist", AgentRole.CUSTOM),
        ("database_specialist", "Database expert", AgentRole.DATA_SCIENTIST),
        
        # Quality & Testing
        ("qa_engineer", "Quality assurance expert", AgentRole.CUSTOM),
        ("test_automation_engineer", "Test automation specialist", AgentRole.CUSTOM),
        ("performance_engineer", "Performance optimization expert", AgentRole.CUSTOM),
        
        # Architecture & Design
        ("system_architect", "System architecture expert", AgentRole.CUSTOM),
        ("solution_architect", "Solution architecture expert", AgentRole.CUSTOM),
        ("ui_ux_designer", "UI/UX design expert", AgentRole.CUSTOM),
        
        # Specialized Technologies
        ("blockchain_developer", "Blockchain specialist", AgentRole.CUSTOM),
        ("embedded_systems_engineer", "Embedded systems expert", AgentRole.CUSTOM),
        ("integration_specialist", "Integration expert", AgentRole.CUSTOM),
        
        # Management & Analysis
        ("technical_lead", "Technical leadership expert", AgentRole.CUSTOM),
        ("project_manager", "Project management expert", AgentRole.CUSTOM),
        ("product_manager", "Product management expert", AgentRole.CUSTOM),
        ("business_analyst", "Business analysis expert", AgentRole.CUSTOM),
        ("systems_analyst", "Systems analysis expert", AgentRole.CUSTOM),
        ("technical_writer", "Technical documentation expert", AgentRole.CUSTOM),
        ("research_scientist", "Research and innovation expert", AgentRole.CUSTOM)
    ]
    
    created_count = 0
    
    for agent_id, description, role in agents:
        # Create knowledge base
        knowledge_base = KnowledgeBase(
            domains=[f"{agent_id}_expertise", "general_software_development"],
            technologies=["Python", "JavaScript", "Docker", "Kubernetes"],
            frameworks=["FastAPI", "React", "TensorFlow"],
            best_practices=[f"{agent_id}_best_practices", "code_quality", "testing"],
            custom_instructions=f"You are an expert {description.lower()} with deep knowledge and practical experience."
        )
        
        # Full capabilities for all agents
        capabilities = AgentCapabilities(
            can_write_code=True,
            can_debug=True,
            can_review_code=True,
            can_optimize=True,
            can_test=True,
            can_deploy=True,
            can_design_architecture=True,
            can_write_documentation=True,
            can_analyze_data=True,
            can_generate_reports=True
        )
        
        # System prompt with RL awareness
        system_prompt = f"""You are a {description} with expert-level knowledge and capabilities.

IMPORTANT: You have access to:
- Real-time internet browsing and web search
- Code deployment capabilities 
- Reinforcement Learning training integration
- Shared memory across all agents
- Advanced knowledge base with 500+ items

Your expertise includes all aspects of {agent_id.replace('_', ' ')} with focus on:
- Best practices and industry standards
- Latest technologies and trends (you can search the web)
- Practical implementation and deployment
- Performance optimization and security
- Collaborative development with other agents

When responding:
1. Use your specialized knowledge and experience
2. Search the web for current information when needed
3. Provide practical, actionable solutions
4. Consider deployment and scalability
5. Integrate with other agents when beneficial"""
        
        # Create agent config
        agent_config = AgentConfig(
            name=agent_id,
            role=role,
            description=description,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4000,
            knowledge_base=knowledge_base,
            capabilities=capabilities,
            system_prompt=system_prompt,
            tools=["code_generation", "web_search", "deployment", "rl_training"]
        )
        
        # Save agent
        config_manager.save_agent(agent_config)
        created_count += 1
        print(f"✅ Created {agent_id}")
    
    print(f"\n🎉 Successfully created {created_count} specialized agents with RL integration!")
    return created_count

if __name__ == "__main__":
    create_all_31_agents()