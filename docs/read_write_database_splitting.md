# Read/Write Database Splitting

This document describes the read/write database splitting feature in Agent Lightning, which allows you to route read operations to read replicas and write operations to the primary database for improved performance and scalability.

## Overview

Read/write database splitting is a common database architecture pattern that improves application performance by:

- Directing read operations to read replicas (which can be scaled horizontally)
- Directing write operations to the primary database (ensuring data consistency)
- Reducing load on the primary database
- Improving overall system throughput

## Configuration

### Environment Variables

Configure the following environment variables to enable read/write database splitting:

```bash
# Enable database splitting
ENABLE_DB_SPLITTING=true

# Read database (replica) URL
READ_DATABASE_URL=postgresql://user:pass@read-replica:5432/agentlightning

# Write database (primary) URL
WRITE_DATABASE_URL=postgresql://user:pass@write-primary:5432/agentlightning
```

### Default Behavior

If `ENABLE_DB_SPLITTING` is not set or is `false`, the system uses single database mode where all operations use the `DATABASE_URL`.

If read/write URLs are not specified, they default to the `DATABASE_URL`.

## Supported Databases

The feature supports all databases that Agent Lightning supports:

- PostgreSQL
- SQLite
- MongoDB (with limitations - see below)

## Usage

### Automatic Routing

When database splitting is enabled, the system automatically routes operations:

```python
from shared.database import db_manager

# Read operations automatically use the read database
read_session = db_manager.get_read_db()

# Write operations automatically use the write database
write_session = db_manager.get_write_db()

# Legacy method defaults to write database for backward compatibility
default_session = db_manager.get_db()
```

### Direct Session Access

You can also access sessions directly:

```python
from shared.database import get_read_session, get_write_session

# Get a read session
read_session = get_read_session()

# Get a write session
write_session = get_write_session()
```

## Monitoring

### Connection Information

Get detailed information about database connections:

```python
from shared.database import db_manager

info = db_manager.get_connection_info()
print(info)
```

When splitting is enabled, this returns:

```json
{
  "type": "sql",
  "db_splitting_enabled": true,
  "read_database": {
    "url": "postgresql://user:***@read-replica:5432/agentlightning",
    "pool_class": "QueuePool",
    "pool_size": 10,
    "max_overflow": 20
  },
  "write_database": {
    "url": "postgresql://user:***@write-primary:5432/agentlightning",
    "pool_class": "QueuePool",
    "pool_size": 10,
    "max_overflow": 20
  }
}
```

### Pool Statistics

Monitor connection pool statistics for both databases:

```python
from shared.database import db_manager

# Read database stats
read_stats = db_manager.get_pool_stats("read")

# Write database stats
write_stats = db_manager.get_pool_stats("write")
```

## Best Practices

### 1. Connection Pooling

Configure appropriate connection pool sizes for your read and write databases:

```bash
# Read database can have more connections (typically more replicas)
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# You may need separate pool configs for read/write in advanced setups
```

### 2. Health Checks

Always implement proper health checks for both read and write databases:

```python
from shared.database import db_manager

# Check overall database health
is_healthy = db_manager.health_check()
```

### 3. Transaction Management

Be careful with transactions that span read and write operations. Consider using the write database for all operations within a transaction to ensure consistency.

### 4. Failover Handling

Implement proper failover logic in your application layer when read replicas become unavailable.

## Limitations

### MongoDB Support

MongoDB has limited support for read/write splitting:

- MongoDB's native driver handles connection pooling internally
- Read/write splitting is not as critical since MongoDB replicas can serve both read and write operations
- The feature will still work but provides less benefit compared to SQL databases

### Cross-Database Transactions

The current implementation does not support distributed transactions across different databases. All operations within a transaction should use the same database.

## Migration Guide

### From Single Database to Read/Write Splitting

1. **Setup read replicas**: Configure your database to have read replicas
2. **Test configuration**: Enable splitting in a test environment first
3. **Gradual rollout**: Start with `ENABLE_DB_SPLITTING=true` and monitor performance
4. **Update application code**: Use `get_read_db()` and `get_write_db()` where appropriate

### Environment Variables Example

```bash
# Single database (existing setup)
DATABASE_URL=postgresql://user:pass@localhost/agentlightning

# Read/write splitting (new setup)
ENABLE_DB_SPLITTING=true
READ_DATABASE_URL=postgresql://user:pass@read-replica:5432/agentlightning
WRITE_DATABASE_URL=postgresql://user:pass@write-primary:5432/agentlightning
```

## Troubleshooting

### Common Issues

1. **Connection failures**: Check that read replica URLs are correct and accessible
2. **Performance issues**: Monitor pool statistics and adjust pool sizes
3. **Data consistency**: Ensure write operations complete before expecting to read updated data

### Debugging

Enable SQLAlchemy echo logging to see which database operations are using:

```python
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Testing

The feature includes comprehensive tests in `tests/test_database_splitting.py`. Run tests to verify functionality:

```bash
python -m pytest tests/test_database_splitting.py -v
```

## Future Enhancements

- **Automatic query routing**: Route queries based on SQL analysis rather than explicit session selection
- **Load balancing**: Distribute reads across multiple replicas
- **Connection failover**: Automatic failover from read replicas to primary
- **Metrics integration**: Enhanced monitoring and alerting for split databases