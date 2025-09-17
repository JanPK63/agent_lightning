# Agent Lightning - Production Deployment Guide

## 🚀 Quick Start

### Development Environment
```bash
./deploy.sh dev
```

### Production Environment (Standard)
```bash
./deploy.sh production
```

### Production Environment (Secure)
```bash
./deploy.sh production secure
```

## 🔧 Configuration Options

### Environment Variables
- `ENV`: `dev` | `production` (default: production)
- `MODE`: `standard` | `secure` (default: secure)

### Access Points

**Development Mode:**
- Application: http://localhost:8051
- API: http://localhost:8001
- Grafana: http://localhost:3000

**Production Mode (Standard):**
- Application: http://localhost
- API: http://localhost:8001
- Grafana: http://localhost:3000

**Production Mode (Secure):**
- Application: https://localhost
- Grafana: https://localhost:3000
- All HTTP traffic redirected to HTTPS

## 🛡️ Security Features

### Container Security
- Non-root users in all containers
- Read-only filesystems where possible
- No new privileges flag
- Resource limits enforced

### Network Security
- SSL/TLS encryption (secure mode)
- Internal network isolation
- Localhost-only database access

### Secrets Management
- Docker secrets for sensitive data
- Auto-generated secure passwords
- No hardcoded credentials

## 📊 Monitoring & Observability

### Metrics Collection
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Alertmanager: http://localhost:9093

### Log Aggregation
- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601

### Distributed Tracing
- Jaeger: http://localhost:16686

## 💾 Backup & Recovery

### Create Backup
```bash
./docker/backup/backup.sh
```

### Restore from Backup
```bash
# Extract backup
tar xzf agent-lightning-backup-YYYYMMDD-HHMMSS.tar.gz

# Restore volumes
docker run --rm -v agent-lightning-main_postgres_data:/data -v ./backup:/backup alpine tar xzf /backup/postgres_data.tar.gz -C /data

# Restore database
docker exec -i agent-postgres psql -U agent_user agent_lightning_memory < backup/postgres.sql
```

## 🔧 Troubleshooting

### Common Issues

**Port Conflicts:**
```bash
# Check what's using ports
lsof -i :80 -i :443 -i :5432 -i :6379

# Stop conflicting services
docker stop $(docker ps -q --filter "publish=80")
```

**Permission Issues:**
```bash
# Fix SSL certificate permissions
chmod 600 docker/configs/ssl/nginx.key
chmod 644 docker/configs/ssl/nginx.crt
```

**Health Check Failures:**
```bash
# Check service logs
docker-compose logs [service-name]

# Check service health
docker-compose ps
```

### Performance Optimization
```bash
# Run system optimization
sudo ./docker/performance/optimize.sh

# Monitor resource usage
docker stats
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale specific services
docker-compose up -d --scale api=3 --scale worker=2
```

### Resource Adjustment
Edit resource limits in `docker-compose.production-secure.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

## 🔄 Updates & Maintenance

### Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild and deploy
./deploy.sh production secure
```

### Update Dependencies
```bash
# Update base images
docker-compose pull

# Rebuild with new images
docker-compose up --build -d
```

### Maintenance Mode
```bash
# Stop all services
docker-compose down

# Start only essential services
docker-compose up -d postgres redis
```