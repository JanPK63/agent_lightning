# Solution Architecture: Database Persistence Layer

**Document ID:** SA-001  
**Date:** 2025-09-05  
**Status:** For Review  
**Priority:** Critical  
**Author:** System Architect  

## Executive Summary

This solution architecture addresses the critical data persistence issue in Agent Lightning. Currently, all data is stored in-memory using Python dictionaries, resulting in complete data loss on system restart. This document outlines the implementation of a PostgreSQL-based persistence layer that will provide data durability, consistency, and scalability.

## Problem Statement

### Current Issues
- **Data Loss:** All data is lost when services restart
- **No Consistency:** Each service maintains separate in-memory storage
- **Limited Scale:** Memory constraints limit data volume
- **No History:** Cannot track changes or audit actions
- **No Backup:** Cannot recover from failures

### Business Impact
- System unreliable for production use
- Cannot scale beyond single instance
- No disaster recovery capability
- Compliance issues (no audit trail)

## Proposed Solution

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Main     │  │   Agent    │  │  Workflow  │            │
│  │ Dashboard  │  │  Designer  │  │   Engine   │  ...        │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │               │               │                     │
└────────┼───────────────┼───────────────┼─────────────────────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Database Access Layer       │
         │  ┌─────────────────────────┐ │
         │  │   SQLAlchemy ORM        │ │
         │  ├─────────────────────────┤ │
         │  │   Connection Pool       │ │
         │  │   (20 connections)      │ │
         │  └──────────┬──────────────┘ │
         └─────────────┼─────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │     PostgreSQL 15           │
         │  ┌────────────────────┐    │
         │  │   Agents Table     │    │
         │  ├────────────────────┤    │
         │  │   Tasks Table      │    │
         │  ├────────────────────┤    │
         │  │   Knowledge Table  │    │
         │  ├────────────────────┤    │
         │  │   Workflows Table  │    │
         │  ├────────────────────┤    │
         │  │   Sessions Table   │    │
         │  ├────────────────────┤    │
         │  │   Metrics Table    │    │
         │  └────────────────────┘    │
         └─────────────────────────────┘
```

### Database Schema Design

```sql
-- Agents table: Core agent definitions
CREATE TABLE agents (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model VARCHAR(50) NOT NULL,
    specialization VARCHAR(50),
    status VARCHAR(20) DEFAULT 'idle',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table: Agent task tracking
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(50) REFERENCES agents(id),
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    priority VARCHAR(20) DEFAULT 'normal',
    context JSONB,
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Knowledge base: Agent learning storage
CREATE TABLE knowledge (
    id VARCHAR(100) PRIMARY KEY,
    agent_id VARCHAR(50) REFERENCES agents(id),
    category VARCHAR(50),
    content TEXT,
    source VARCHAR(100),
    metadata JSONB,
    embedding VECTOR(1536), -- For similarity search
    usage_count INTEGER DEFAULT 0,
    relevance_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

-- Workflows: Complex task orchestration
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100),
    description TEXT,
    steps JSONB,
    status VARCHAR(20),
    created_by VARCHAR(50),
    assigned_to VARCHAR(50),
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions: User session management
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100),
    token VARCHAR(500) UNIQUE,
    data JSONB,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metrics: Performance tracking
CREATE TABLE metrics (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(50),
    metric_name VARCHAR(100),
    value FLOAT,
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_knowledge_agent_id ON knowledge(agent_id);
CREATE INDEX idx_knowledge_category ON knowledge(agent_id, category);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX idx_metrics_service ON metrics(service_name, metric_name, timestamp);

-- Enable Row Level Security for multi-tenancy (future)
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
```

### Connection Management

```python
# shared/database_config.py
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "agent_lightning"),
    "user": os.getenv("DB_USER", "agent_user"),
    "password": os.getenv("DB_PASSWORD", "secure_password"),
    
    # Connection pool settings
    "pool_size": 20,
    "max_overflow": 40,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
    
    # Performance settings
    "echo": False,
    "echo_pool": False,
    "connect_args": {
        "connect_timeout": 10,
        "application_name": "agent_lightning",
        "options": "-c statement_timeout=30000"
    }
}
```

## Implementation Plan

### Phase 1: Database Setup (Day 1)
1. Install PostgreSQL 15 with pgvector extension
2. Create database and user accounts
3. Run schema creation scripts
4. Set up connection pooling
5. Configure SSL/TLS

### Phase 2: ORM Layer (Day 2)
1. Create SQLAlchemy models
2. Implement repository pattern
3. Add transaction management
4. Create migration scripts (Alembic)
5. Add query optimization

### Phase 3: Data Migration (Day 3)
1. Export existing in-memory data
2. Transform to database format
3. Load into PostgreSQL
4. Verify data integrity
5. Create backup

### Phase 4: Service Integration (Days 4-5)
1. Update monitoring_dashboard.py
2. Update all agent classes
3. Update microservices
4. Remove in-memory storage
5. Test end-to-end

## Technical Specifications

### Dependencies
```python
# requirements.txt additions
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
alembic==1.13.0
pgvector==0.2.3
asyncpg==0.29.0  # For async operations
```

### Performance Requirements
- Query response time: < 100ms for simple queries
- Connection pool: 20-60 concurrent connections
- Transaction throughput: 1000 TPS minimum
- Storage: 100GB initial, scalable to 1TB
- Backup: Daily incremental, weekly full

### Security Considerations
- Encrypted connections (SSL/TLS)
- Connection string in environment variables
- Prepared statements to prevent SQL injection
- Row-level security for multi-tenancy
- Audit logging for all modifications
- Regular security patches

## Rollback Plan

### Rollback Strategy
1. Keep in-memory storage code (disabled)
2. Create feature flag for database usage
3. Maintain data export functionality
4. Test rollback procedure
5. Document rollback steps

### Rollback Procedure
```bash
# 1. Stop all services
systemctl stop agent-lightning-*

# 2. Export database data
pg_dump -h localhost -U agent_user agent_lightning > backup.sql

# 3. Switch feature flag
export USE_DATABASE=false

# 4. Restart services with in-memory mode
systemctl start agent-lightning-*

# 5. Import critical data if needed
python scripts/import_critical_data.py
```

## Success Metrics

### Technical Metrics
- ✅ Zero data loss on restart
- ✅ All services using shared database
- ✅ < 100ms query latency
- ✅ 99.9% uptime
- ✅ Successful daily backups

### Business Metrics
- ✅ Production-ready persistence
- ✅ Horizontal scalability enabled
- ✅ Audit trail available
- ✅ Disaster recovery capability
- ✅ Compliance requirements met

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Data corruption during migration | Low | High | Extensive testing, backups |
| Performance degradation | Medium | Medium | Query optimization, caching |
| Connection pool exhaustion | Low | High | Monitoring, auto-scaling |
| Database failure | Low | Critical | HA setup, replicas |
| Security breach | Low | Critical | Encryption, access control |

## Approval

**Review Checklist:**
- [ ] Solution addresses all identified problems
- [ ] Performance requirements are realistic
- [ ] Security measures are adequate
- [ ] Rollback plan is comprehensive
- [ ] Resource requirements are available

**Sign-off Required From:**
- [ ] Technical Lead
- [ ] DevOps Team
- [ ] Security Team
- [ ] Product Owner

---

**Next Steps After Approval:**
1. Provision PostgreSQL instance
2. Create development database
3. Begin ORM implementation
4. Schedule migration window

**Related Documents:**
- SYSTEM_ARCHITECTURE_ANALYSIS.md
- INTEGRATION_PLAN.md
- Next: SA-002 Redis Cache & Event Bus