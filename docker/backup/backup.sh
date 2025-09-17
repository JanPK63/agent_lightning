#!/bin/bash

set -e

BACKUP_DIR="/tmp/agent-lightning-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "💾 Creating backup in $BACKUP_DIR..."

# Backup PostgreSQL
echo "🗄️ Backing up PostgreSQL..."
docker exec agent-postgres pg_dump -U agent_user agent_lightning_memory > "$BACKUP_DIR/postgres.sql"

# Backup Redis
echo "🔴 Backing up Redis..."
docker exec agent-redis redis-cli --rdb /tmp/dump.rdb
docker cp agent-redis:/tmp/dump.rdb "$BACKUP_DIR/redis.rdb"

# Backup volumes
echo "📦 Backing up Docker volumes..."
docker run --rm -v agent-lightning-main_postgres_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .
docker run --rm -v agent-lightning-main_grafana_data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/grafana_data.tar.gz -C /data .

# Backup configurations
echo "⚙️ Backing up configurations..."
cp -r docker/configs "$BACKUP_DIR/"
cp docker-compose*.yml "$BACKUP_DIR/"

# Create archive
echo "📦 Creating backup archive..."
tar czf "agent-lightning-backup-$(date +%Y%m%d-%H%M%S).tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"

echo "✅ Backup complete: agent-lightning-backup-$(date +%Y%m%d-%H%M%S).tar.gz"