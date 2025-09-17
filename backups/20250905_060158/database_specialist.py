#!/usr/bin/env python3
"""
Database Specialist Agent for Agent Lightning
Handles both relational (SQL/PostgreSQL) and non-relational (MongoDB, Redis) databases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_config import AgentConfigManager, AgentConfig, AgentRole, AgentCapabilities, KnowledgeBase


def setup_database_specialist():
    """Setup database specialist agent with comprehensive database knowledge"""
    
    manager = AgentConfigManager()
    
    # Create database specialist agent
    db_config = AgentConfig(
        name="database_specialist",
        description="Database Expert - SQL, PostgreSQL, MongoDB, Redis",
        role=AgentRole.DATA_SCIENTIST,  # Using data scientist role for database specialist
        model="gpt-4o",
        temperature=0.3,  # Lower temperature for precise database operations
        max_tokens=2000,
        system_prompt="""You are an expert database specialist with deep knowledge of:

## Relational Databases (SQL/PostgreSQL):
- Schema design and normalization (1NF, 2NF, 3NF, BCNF)
- Query optimization and performance tuning
- Indexing strategies (B-tree, Hash, GiST, GIN)
- Transaction management (ACID properties)
- Stored procedures, triggers, and views
- Replication and clustering
- PostgreSQL-specific features (JSONB, arrays, full-text search)
- SQL query writing and optimization
- Database migration strategies
- Connection pooling and resource management

## Non-Relational Databases:

### MongoDB:
- Document modeling and schema design
- Aggregation pipelines
- Indexing and sharding strategies
- Replica sets and high availability
- Change streams and real-time updates
- GridFS for large files
- MongoDB Atlas cloud deployment

### Redis:
- Data structures (Strings, Lists, Sets, Sorted Sets, Hashes, Streams)
- Caching strategies and TTL management
- Pub/Sub messaging patterns
- Redis Cluster and Sentinel for HA
- Lua scripting
- Persistence options (RDB, AOF)
- Memory optimization

## General Database Skills:
- Database security and access control
- Backup and recovery strategies
- Performance monitoring and tuning
- Data migration between different database systems
- CAP theorem and consistency models
- Database containerization with Docker
- Connection to application layers
- ORM/ODM usage and optimization

You can analyze existing databases, optimize queries, design schemas, troubleshoot performance issues, 
and provide best practices for database management. You understand trade-offs between different 
database systems and can recommend the right database for specific use cases.

When analyzing databases on servers, you can:
- Check database status and health
- Analyze table/collection structures
- Review indexes and performance
- Identify optimization opportunities
- Suggest security improvements
- Help with data migrations""",
        
        tools=["file_read", "file_write", "code_generation", "web_search"],
        
        capabilities=AgentCapabilities(
            can_write_code=True,
            can_debug=True,
            can_review_code=True,
            can_optimize=True,
            can_test=True,
            can_deploy=False,
            can_design_architecture=True,
            can_write_documentation=True,
            can_analyze_data=True,
            can_generate_reports=True
        ),
        
        knowledge_base=KnowledgeBase(
            domains=[
                "postgresql", "mysql", "mariadb", "oracle", "sql-server",
                "mongodb", "redis", "cassandra", "dynamodb", "couchdb",
                "database-design", "query-optimization", "indexing",
                "replication", "sharding", "clustering", "high-availability",
                "backup-recovery", "migration", "performance-tuning"
            ],
            
            technologies=[
                "SQL", "NoSQL", "NewSQL", "PostgreSQL", "MySQL", "MariaDB",
                "MongoDB", "Redis", "Cassandra", "DynamoDB", "CouchDB",
                "Elasticsearch", "InfluxDB", "TimescaleDB", "Neo4j"
            ],
            
            frameworks=[
                "SQLAlchemy", "Django ORM", "Sequelize", "TypeORM", "Prisma",
                "Mongoose", "Redis OM", "Spring Data", "Hibernate",
                "Entity Framework", "ActiveRecord", "Eloquent"
            ],
            
            best_practices=[
                "Always use parameterized queries to prevent SQL injection",
                "Create indexes on frequently queried columns",
                "Use connection pooling for production applications",
                "Implement proper backup and recovery procedures",
                "Monitor slow query logs regularly",
                "Use appropriate isolation levels for transactions",
                "Normalize relational data to at least 3NF unless denormalization is justified",
                "For MongoDB, embed related data that's queried together",
                "Use Redis for session storage and caching, not as primary database",
                "Implement database monitoring and alerting",
                "Use read replicas to distribute read load",
                "Test database migrations in staging environment first"
            ],
            
            custom_instructions="""You are a database specialist with expertise in both SQL and NoSQL databases.
You can design schemas, optimize queries, troubleshoot performance issues, and implement migrations.
Provide code examples and best practices for:
- PostgreSQL: Advanced features like JSONB, arrays, CTEs, window functions
- MongoDB: Aggregation pipelines, sharding strategies, indexing
- Redis: Data structures, caching patterns, pub/sub
- MySQL: Replication, partitioning, query optimization
Always consider ACID properties, CAP theorem, and appropriate use cases for each database type."""
        ),
        
        examples=[
            {
                "input": "Optimize a slow PostgreSQL query",
                "output": "I'll analyze the query execution plan using EXPLAIN ANALYZE, identify bottlenecks, and suggest appropriate indexes or query restructuring..."
            },
            {
                "input": "Design a MongoDB schema for e-commerce",
                "output": "I'll design a document structure with embedded vs referenced relationships based on access patterns, including products, orders, and user collections..."
            },
            {
                "input": "Setup Redis caching strategy",
                "output": "I'll implement a multi-layer caching strategy using Redis data structures, with appropriate TTL, cache invalidation, and fallback mechanisms..."
            }
        ]
    )
    
    # Save the configuration
    try:
        manager.save_agent(db_config)
        print("✅ Created database_specialist agent")
        
        # Add specialized database knowledge
        from knowledge_manager import KnowledgeManager
        km = KnowledgeManager()
        
        # Add PostgreSQL-specific knowledge
        km.add_knowledge(
            agent_name="database_specialist",
            category="postgresql_optimization",
            content="""PostgreSQL Performance Tuning:
1. Configuration tuning:
   - shared_buffers: 25% of RAM
   - effective_cache_size: 50-75% of RAM
   - work_mem: RAM / (max_connections * 2)
   - maintenance_work_mem: RAM / 16
   
2. Query optimization:
   - Use EXPLAIN ANALYZE for execution plans
   - Check for sequential scans on large tables
   - Ensure statistics are up-to-date (ANALYZE)
   - Use CTEs and window functions for complex queries
   
3. Index strategies:
   - B-tree for equality and range queries
   - Hash for equality only
   - GIN for full-text search and JSONB
   - Partial indexes for filtered queries
   - Covering indexes to avoid heap lookups""",
            source="postgresql_expertise"
        )
        
        # Add MongoDB-specific knowledge
        km.add_knowledge(
            agent_name="database_specialist",
            category="mongodb_patterns",
            content="""MongoDB Design Patterns:
1. Embedding vs Referencing:
   - Embed: 1-to-few relationships, data accessed together
   - Reference: 1-to-many or many-to-many, large documents
   
2. Bucketing Pattern:
   - Group time-series data into buckets
   - Reduces index size and improves query performance
   
3. Computed Pattern:
   - Pre-calculate and store frequently computed values
   - Trade write performance for read performance
   
4. Schema Versioning:
   - Add schema_version field to documents
   - Handle multiple versions in application code
   
5. Aggregation Pipeline Optimization:
   - Use $match early to reduce dataset
   - $project to limit fields before $group
   - Use indexes with $match and $sort""",
            source="mongodb_expertise"
        )
        
        # Add Redis patterns
        km.add_knowledge(
            agent_name="database_specialist",
            category="redis_patterns",
            content="""Redis Advanced Patterns:
1. Cache Patterns:
   - Cache-aside: Read from cache, fallback to DB
   - Write-through: Write to cache and DB simultaneously
   - Write-behind: Write to cache, async to DB
   
2. Distributed Locking:
   SET resource_lock unique_id NX PX 30000
   
3. Rate Limiting:
   - Sliding window with sorted sets
   - Token bucket with Lua scripts
   
4. Real-time Analytics:
   - HyperLogLog for unique counts
   - Streams for event processing
   - Sorted sets for rankings
   
5. Pub/Sub Patterns:
   - Fan-out messaging
   - Event-driven architectures
   - Real-time notifications""",
            source="redis_expertise"
        )
        
        print("📚 Added specialized database knowledge")
        return db_config
    except Exception as e:
        print(f"❌ Error creating database specialist: {e}")
        return None


def add_database_analysis_to_ssh():
    """Add database analysis capabilities to SSH executor"""
    
    additional_code = '''
    
    def analyze_databases(self) -> Dict[str, Any]:
        """Analyze all databases on the server"""
        db_analysis = {
            "postgresql": {},
            "mysql": {},
            "mongodb": {},
            "redis": {}
        }
        
        # Check PostgreSQL
        result = self.execute_command("sudo -u postgres psql -l 2>/dev/null || psql -U postgres -l 2>/dev/null")
        if result.success and "List of databases" in result.stdout:
            db_analysis["postgresql"]["status"] = "running"
            # Parse database list
            lines = result.stdout.split('\\n')
            databases = []
            for line in lines[3:]:  # Skip header
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) > 0:
                        db_name = parts[0].strip()
                        if db_name and db_name not in ['', 'Name']:
                            databases.append(db_name)
            db_analysis["postgresql"]["databases"] = databases[:10]  # Limit to 10
            
            # Get PostgreSQL version
            result = self.execute_command("postgres --version 2>/dev/null || psql --version")
            if result.success:
                db_analysis["postgresql"]["version"] = result.stdout.strip()
        
        # Check MySQL/MariaDB
        result = self.execute_command("mysql -e 'SHOW DATABASES;' 2>/dev/null")
        if result.success:
            db_analysis["mysql"]["status"] = "running"
            databases = [line.strip() for line in result.stdout.split('\\n') if line.strip() and line.strip() != 'Database']
            db_analysis["mysql"]["databases"] = databases[:10]
            
            # Get MySQL version
            result = self.execute_command("mysql --version")
            if result.success:
                db_analysis["mysql"]["version"] = result.stdout.strip()
        
        # Check MongoDB
        result = self.execute_command("mongosh --eval 'db.adminCommand({listDatabases: 1})' --quiet 2>/dev/null || mongo --eval 'db.adminCommand({listDatabases: 1})' --quiet 2>/dev/null")
        if result.success and "databases" in result.stdout:
            db_analysis["mongodb"]["status"] = "running"
            # Try to parse JSON output
            try:
                import json
                data = json.loads(result.stdout)
                db_analysis["mongodb"]["databases"] = [db["name"] for db in data.get("databases", [])][:10]
            except:
                db_analysis["mongodb"]["databases"] = ["parsing_error"]
            
            # Get MongoDB version
            result = self.execute_command("mongod --version 2>/dev/null | head -1")
            if result.success:
                db_analysis["mongodb"]["version"] = result.stdout.strip()
        
        # Check Redis
        result = self.execute_command("redis-cli ping 2>/dev/null")
        if result.success and "PONG" in result.stdout:
            db_analysis["redis"]["status"] = "running"
            
            # Get Redis info
            result = self.execute_command("redis-cli INFO server | grep redis_version")
            if result.success:
                db_analysis["redis"]["version"] = result.stdout.strip().replace("redis_version:", "")
            
            # Get key count
            result = self.execute_command("redis-cli DBSIZE")
            if result.success:
                db_analysis["redis"]["keys"] = result.stdout.strip()
        
        return db_analysis
    
    def optimize_query(self, db_type: str, query: str) -> str:
        """Provide query optimization suggestions"""
        if db_type == "postgresql":
            # Run EXPLAIN ANALYZE
            result = self.execute_command(f"psql -c 'EXPLAIN ANALYZE {query}' 2>/dev/null")
            if result.success:
                return f"Query Plan:\\n{result.stdout}"
        elif db_type == "mysql":
            result = self.execute_command(f"mysql -e 'EXPLAIN {query}' 2>/dev/null")
            if result.success:
                return f"Query Plan:\\n{result.stdout}"
        return "Unable to analyze query"
'''
    
    print("📝 Database analysis methods added to SSH executor documentation")
    return additional_code


if __name__ == "__main__":
    # Setup the database specialist
    config = setup_database_specialist()
    if config:
        print("\n✅ Database Specialist Agent Created Successfully!")
        print(f"   Model: {config.model}")
        print(f"   Role: {config.role.value}")
        print(f"   Capabilities: Database design, optimization, migration")
        
        # Show the additional SSH methods
        print("\n📚 Additional SSH Database Methods:")
        print(add_database_analysis_to_ssh())