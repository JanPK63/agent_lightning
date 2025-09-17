# Agent Lightning System Status & Services

## 🚀 Active Services and Ports

### Core Services

| Service | Port | URL | Purpose | Start Command |
|---------|------|-----|---------|---------------|
| **Enhanced Production API** | 8002 | http://localhost:8002 | Main Agent Lightning API with Knowledge Management | `python enhanced_production_api.py --port 8002` |
| **Monitoring Dashboard** | 8051 | http://localhost:8051 | Streamlit real-time monitoring dashboard | `streamlit run monitoring_dashboard.py --server.port 8051` |
| **Grafana** | 3000 | http://localhost:3000 | Metrics visualization with 4 pre-configured dashboards | `docker-compose up -d` (part of stack) |
| **InfluxDB** | 8086 | http://localhost:8086 | Time-series database for metrics storage | `docker-compose up -d` (part of stack) |

### API Endpoints (Port 8002)

- **Original API**: http://localhost:8002/original
- **Enhanced API v2**: http://localhost:8002/api/v2
- **API Documentation**: http://localhost:8002/docs
- **Agent List**: http://localhost:8002/api/v2/agents/list

### Available Specialized Agents

The API has 10+ specialized agents loaded:
- `data_scientist` - Data analysis, ML, statistical modeling
- `devops_engineer` - Infrastructure, automation, deployment
- `system_architect` - Software architecture design
- `ui_ux_designer` - Interface and user experience design
- `blockchain_developer` - Web3 and blockchain development
- `security_expert` - Cybersecurity and secure coding
- `database_specialist` - SQL, PostgreSQL, MongoDB, Redis
- `full_stack_developer` - Complete web application development
- `mobile_developer` - iOS and Android development
- `information_analyst` - Data analysis and business intelligence

## 🔑 Credentials

### InfluxDB
- **URL**: http://localhost:8086
- **Username**: `admin`
- **Password**: `supersecret123`
- **Organization**: `agent-system`
- **API Token**: `agent-system-token-supersecret-12345678`
- **Primary Bucket**: `performance_metrics`

### Grafana
- **URL**: http://localhost:3000
- **Username**: `admin`
- **Password**: `admin123`

## 📊 Configured Dashboards (Grafana)

1. **System Performance Monitoring** - CPU, Memory, Disk, Network metrics
2. **Agent Metrics Dashboard** - Agent tasks, completion rates, resource usage
3. **System Alerts Dashboard** - Real-time alerts and notifications
4. **Test Metrics Dashboard** - Test coverage, execution results

## 🗄️ InfluxDB Buckets Configuration

| Bucket | Retention | Purpose |
|--------|-----------|---------|
| `performance_metrics` | 30 days | High-resolution system metrics |
| `performance_metrics_long` | 1 year | Downsampled long-term metrics |
| `agent_metrics` | 90 days | Agent-specific metrics |
| `alerts` | 180 days | System alerts and notifications |
| `test_metrics` | 60 days | Test execution and coverage |
| `deployment_metrics` | Infinite | Deployment and release metrics |
| `debug_metrics` | 3 days | High-frequency debug data |

## 🚦 Service Management

### Start All Services

```bash
# 1. Start Docker (required for InfluxDB and Grafana)
open -a Docker

# 2. Start InfluxDB and Grafana
./influxdb_manager.sh start

# 3. Start the API
python enhanced_production_api.py --port 8002 &

# 4. Start the Monitoring Dashboard
streamlit run monitoring_dashboard.py --server.port 8051 &
```

### Stop Services

```bash
# Stop InfluxDB and Grafana
./influxdb_manager.sh stop

# Stop API and Dashboard (find and kill processes)
pkill -f "enhanced_production_api.py"
pkill -f "streamlit run monitoring_dashboard.py"
```

### Check Service Status

```bash
# Check Docker containers
docker ps | grep -E "influx|grafana"

# Check InfluxDB and Grafana
./influxdb_manager.sh status

# Check if API is running
curl http://localhost:8002/api/v2/agents/list

# Check if Dashboard is running
curl http://localhost:8051
```

### Test Services

```bash
# Test complete monitoring pipeline
python test_monitoring_pipeline.py

# Test InfluxDB connection
./influxdb_manager.sh test

# Send sample metrics
./influxdb_manager.sh sample
```

## 📁 Key Configuration Files

- `.env.influxdb` - InfluxDB credentials and tokens
- `docker-compose.yml` - Docker services configuration
- `influxdb_manager.sh` - Service management script
- `influxdb_configurator.py` - Bucket and retention setup
- `grafana_dashboard_creator.py` - Dashboard creation/upload
- `enhanced_production_api.py` - Main API server
- `monitoring_dashboard.py` - Streamlit dashboard

## 🔄 Environment Variables

Required environment variables (in `.env.influxdb`):
```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=agent-system-token-supersecret-12345678
INFLUXDB_ORG=agent-system
INFLUXDB_BUCKET=performance_metrics
```

## 📈 Current Implementation Status

### ✅ Completed
- InfluxDB setup with 7 configured buckets
- Grafana with 4 pre-built dashboards
- Data retention policies (30 days to infinite)
- Metrics collection system (CPU, Memory, Disk, Network)
- Alert system with configurable thresholds
- API with 10+ specialized agents
- Streamlit monitoring dashboard
- Docker-based infrastructure

### 🔄 In Progress
- Data aggregation jobs
- Automated backup system
- Data compression
- Export functionality
- Migration tools

## 🆘 Troubleshooting

### InfluxDB Connection Issues
```bash
# Check if InfluxDB is running
docker ps | grep influx

# Restart InfluxDB
./influxdb_manager.sh restart

# Re-run initial setup if needed
python influxdb_initial_setup.py
```

### API Not Starting
```bash
# Check if port 8002 is in use
lsof -i :8002

# Kill existing process if needed
kill -9 $(lsof -t -i:8002)

# Start with explicit port
uvicorn enhanced_production_api:enhanced_app --reload --port 8002
```

### Dashboard Issues
```bash
# Check if Streamlit is installed
pip install streamlit

# Check if port 8051 is in use
lsof -i :8051

# Start with different port if needed
streamlit run monitoring_dashboard.py --server.port 8052
```

## 📝 Notes

- All services use Docker except the API and Streamlit dashboard
- Data is persisted in Docker volumes (survives container restarts)
- The system is designed for local development but can be deployed to production
- Monitoring data flows: Agents → API → InfluxDB → Grafana/Dashboard

---

**Last Updated**: September 4, 2025
**System Version**: Agent Lightning v0.1.2
**Infrastructure**: Docker, InfluxDB 2.7, Grafana latest, Streamlit