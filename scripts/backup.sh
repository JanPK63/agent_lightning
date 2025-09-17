#!/bin/bash

# Backup PostgreSQL
docker exec agent-lightning-postgres-1 pg_dump -U agent_user agent_lightning > backup_$(date +%Y%m%d).sql

# Backup Redis
docker exec agent-lightning-redis-1 redis-cli BGSAVE
docker cp agent-lightning-redis-1:/data/dump.rdb backup_redis_$(date +%Y%m%d).rdb

# Backup InfluxDB
docker exec agent-lightning-influxdb-1 influx backup /tmp/backup
docker cp agent-lightning-influxdb-1:/tmp/backup ./backup_influx_$(date +%Y%m%d)/