#!/usr/bin/env python3
"""
Set up specialized agents with specific knowledge domains
"""

from agent_config import AgentConfigManager, AgentConfig, AgentRole, KnowledgeBase, AgentCapabilities
from knowledge_manager import KnowledgeManager
import json


def setup_full_stack_developer():
    """Set up a full-stack developer agent with comprehensive knowledge"""
    
    print("🚀 Setting up Full-Stack Developer Agent...")
    
    # Initialize managers
    config_manager = AgentConfigManager()
    knowledge_manager = KnowledgeManager()
    
    # Create full-stack developer configuration
    full_stack = config_manager.create_full_stack_developer()
    config_manager.save_agent(full_stack)
    
    print(f"✅ Created agent configuration: {full_stack.name}")
    
    # Add specific knowledge to the agent's knowledge base
    knowledge_items = [
        # React knowledge
        {
            "category": "code_examples",
            "content": """React functional component with hooks:
            
import React, { useState, useEffect } from 'react';

const UserProfile = ({ userId }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchUser(userId).then(data => {
            setUser(data);
            setLoading(false);
        });
    }, [userId]);
    
    if (loading) return <div>Loading...</div>;
    return <div>{user?.name}</div>;
};""",
            "source": "react_patterns"
        },
        
        # Python/FastAPI knowledge
        {
            "category": "code_examples",
            "content": """FastAPI endpoint with authentication:
            
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = await get_current_user(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user""",
            "source": "fastapi_patterns"
        },
        
        # Database optimization
        {
            "category": "best_practices",
            "content": """Database Query Optimization Tips:
1. Use indexes on columns used in WHERE, JOIN, and ORDER BY clauses
2. Avoid SELECT * - only fetch needed columns
3. Use EXPLAIN to analyze query execution plans
4. Batch operations when possible
5. Use connection pooling for better performance
6. Implement caching for frequently accessed data
7. Consider denormalization for read-heavy workloads""",
            "source": "database_optimization"
        },
        
        # Security best practices
        {
            "category": "best_practices",
            "content": """Web Application Security Checklist:
1. Always validate and sanitize user input
2. Use parameterized queries to prevent SQL injection
3. Implement proper authentication and authorization
4. Use HTTPS for all communications
5. Store passwords using bcrypt or argon2
6. Implement CSRF protection
7. Set security headers (CSP, X-Frame-Options, etc.)
8. Keep dependencies updated
9. Implement rate limiting
10. Log security events for monitoring""",
            "source": "security_guide"
        },
        
        # Docker deployment
        {
            "category": "code_examples",
            "content": """Multi-stage Docker build for Node.js app:
            
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Production stage
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]""",
            "source": "docker_patterns"
        },
        
        # Testing patterns
        {
            "category": "code_examples",
            "content": """Jest testing example for React component:
            
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UserForm from './UserForm';

describe('UserForm', () => {
    test('submits form with user data', async () => {
        const handleSubmit = jest.fn();
        render(<UserForm onSubmit={handleSubmit} />);
        
        await userEvent.type(screen.getByLabelText(/name/i), 'John Doe');
        await userEvent.type(screen.getByLabelText(/email/i), 'john@example.com');
        await userEvent.click(screen.getByRole('button', { name: /submit/i }));
        
        await waitFor(() => {
            expect(handleSubmit).toHaveBeenCalledWith({
                name: 'John Doe',
                email: 'john@example.com'
            });
        });
    });
});""",
            "source": "testing_patterns"
        },
        
        # Architecture patterns
        {
            "category": "architecture_patterns",
            "content": """Microservices Architecture Best Practices:
1. Design services around business capabilities
2. Implement API Gateway for client communication
3. Use service discovery for dynamic service location
4. Implement circuit breakers for fault tolerance
5. Use event-driven communication where appropriate
6. Implement distributed tracing for debugging
7. Use containerization for deployment consistency
8. Implement health checks and monitoring
9. Design for eventual consistency
10. Use API versioning for backward compatibility""",
            "source": "architecture_guide"
        },
        
        # Performance optimization
        {
            "category": "best_practices",
            "content": """Frontend Performance Optimization:
1. Implement code splitting and lazy loading
2. Optimize images (WebP, lazy loading, responsive images)
3. Minimize and compress CSS/JS bundles
4. Use CDN for static assets
5. Implement browser caching strategies
6. Reduce initial bundle size
7. Use virtual scrolling for long lists
8. Debounce/throttle event handlers
9. Optimize React re-renders with memo and useMemo
10. Implement Progressive Web App features""",
            "source": "performance_guide"
        }
    ]
    
    # Add all knowledge items
    for item in knowledge_items:
        knowledge_manager.add_knowledge(
            agent_name="full_stack_developer",
            category=item["category"],
            content=item["content"],
            source=item["source"]
        )
    
    print(f"✅ Added {len(knowledge_items)} knowledge items to the agent")
    
    # Get statistics
    stats = knowledge_manager.get_statistics("full_stack_developer")
    print(f"\n📊 Knowledge Base Statistics:")
    print(f"   Total items: {stats['total_items']}")
    print(f"   Categories: {stats['categories']}")
    
    return full_stack


def setup_data_scientist():
    """Set up a data scientist agent"""
    
    print("\n🔬 Setting up Data Scientist Agent...")
    
    config_manager = AgentConfigManager()
    knowledge_manager = KnowledgeManager()
    
    # Create data scientist configuration
    data_scientist = config_manager.create_data_scientist()
    config_manager.save_agent(data_scientist)
    
    # Add specific knowledge
    knowledge_items = [
        {
            "category": "code_examples",
            "content": """Pandas data preprocessing:
            
import pandas as pd
import numpy as np

# Handle missing values
df['column'].fillna(df['column'].median(), inplace=True)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# One-hot encoding
df = pd.get_dummies(df, columns=['categorical_column'])""",
            "source": "data_preprocessing"
        },
        {
            "category": "code_examples",
            "content": """Machine Learning pipeline:
            
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)""",
            "source": "ml_patterns"
        }
    ]
    
    for item in knowledge_items:
        knowledge_manager.add_knowledge(
            agent_name="data_scientist",
            category=item["category"],
            content=item["content"],
            source=item["source"]
        )
    
    print(f"✅ Created data scientist agent with {len(knowledge_items)} knowledge items")
    
    return data_scientist


def list_all_agents():
    """List all configured agents"""
    config_manager = AgentConfigManager()
    agents = config_manager.list_agents()
    
    print("\n📋 Available Specialized Agents:")
    print("=" * 50)
    
    for agent_name in agents:
        agent = config_manager.get_agent(agent_name)
        if agent:
            print(f"\n🤖 {agent.name}")
            print(f"   Role: {agent.role.value}")
            print(f"   Description: {agent.description}")
            print(f"   Model: {agent.model}")
            print(f"   Capabilities: {sum(1 for k, v in agent.capabilities.__dict__.items() if v)} enabled")
            print(f"   Knowledge domains: {', '.join(agent.knowledge_base.domains[:3])}...")


def test_agent_knowledge(agent_name: str, query: str):
    """Test an agent's knowledge retrieval"""
    knowledge_manager = KnowledgeManager()
    
    print(f"\n🔍 Searching knowledge for '{query}' in {agent_name}:")
    results = knowledge_manager.search_knowledge(agent_name, query, limit=3)
    
    for i, item in enumerate(results, 1):
        print(f"\n{i}. [{item.category}] from {item.source}")
        print(f"   {item.content[:200]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("⚡ AGENT LIGHTNING - Specialized Agent Setup")
    print("=" * 60)
    
    # Set up agents
    full_stack = setup_full_stack_developer()
    data_scientist = setup_data_scientist()
    
    # List all agents
    list_all_agents()
    
    # Test knowledge retrieval
    print("\n" + "=" * 60)
    print("🧪 Testing Knowledge Retrieval")
    print("=" * 60)
    
    test_agent_knowledge("full_stack_developer", "React hooks")
    test_agent_knowledge("full_stack_developer", "security")
    test_agent_knowledge("full_stack_developer", "Docker")
    
    print("\n✅ Agent setup complete! Agents are ready to use.")
    print("\nTo use these agents:")
    print("1. They will appear in the Task Assignment tab")
    print("2. Select the agent when submitting a task")
    print("3. The agent will use its specialized knowledge to provide better responses")