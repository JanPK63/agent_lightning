# InfluxDB and Grafana Setup Guide

## Overview
This system uses InfluxDB for time-series metrics storage and Grafana for visualization.

## Quick Start

### 1. Start Services
```bash
./influxdb_manager.sh start
```

### 2. Access Web Interfaces

#### InfluxDB
- URL: http://localhost:8086
- Username: admin
- Password: supersecret123

#### Grafana
- URL: http://localhost:3000
- Username: admin  
- Password: admin123

## Initial Setup (First Time Only)

### InfluxDB Setup
1. Open http://localhost:8086 in your browser
2. Click "Get Started"
3. Enter the following:
   - Username: `admin`
   - Password: `supersecret123`
   - Organization: `agent-system`
   - Bucket: `performance_metrics`
4. Click "Continue"
5. Copy the generated API token and save it

### Configure Environment
1. Update `.env.influxdb` with your API token:
```bash
INFLUXDB_TOKEN=your-actual-token-here
```

2. Test the connection:
```bash
./influxdb_manager.sh test
```

## Management Commands

```bash
# Start services
./influxdb_manager.sh start

# Stop services
./influxdb_manager.sh stop

# Check status
./influxdb_manager.sh status

# View logs
./influxdb_manager.sh logs

# Test connection
./influxdb_manager.sh test

# Insert sample data
./influxdb_manager.sh sample

# Clean all data (careful!)
./influxdb_manager.sh clean
```

## Python Integration

### Send Metrics
```python
import os
from influxdb_metrics import PerformanceMonitorWithInfluxDB

# Load environment
os.environ['INFLUXDB_URL'] = 'http://localhost:8086'
os.environ['INFLUXDB_TOKEN'] = 'your-token-here'
os.environ['INFLUXDB_ORG'] = 'agent-system'
os.environ['INFLUXDB_BUCKET'] = 'performance_metrics'

# Start monitoring
monitor = PerformanceMonitorWithInfluxDB()
monitor.start()

# Metrics are automatically sent to InfluxDB
```

### Query Metrics
```python
from influxdb_metrics import InfluxDBMetricsWriter

writer = InfluxDBMetricsWriter()
metrics = writer.query_metrics(
    metric_type=MetricType.CPU_USAGE,
    start_time="-1h"
)
```

## Grafana Dashboard Setup

### Add InfluxDB Data Source
1. Open Grafana: http://localhost:3000
2. Go to Configuration > Data Sources
3. Click "Add data source"
4. Choose "InfluxDB"
5. Configure:
   - Query Language: Flux
   - URL: http://influxdb:8086
   - Organization: agent-system
   - Token: (your API token)
   - Default Bucket: performance_metrics
6. Click "Save & Test"

### Import Dashboard
1. Go to Dashboards > Import
2. Upload `grafana-dashboard.json` (if available)
3. Or create a new dashboard with these panels:
   - CPU Usage (line graph)
   - Memory Usage (gauge)
   - Disk I/O (area chart)
   - Network Traffic (line graph)
   - Process Count (stat)
   - Alerts (table)

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ Performance     │────▶│   InfluxDB   │◀────│   Grafana   │
│ Monitor         │     │  (Port 8086) │     │ (Port 3000) │
└─────────────────┘     └──────────────┘     └─────────────┘
        │                       │                      │
        │                       ▼                      ▼
        │               ┌──────────────┐      ┌─────────────┐
        └──────────────▶│ Time Series  │      │ Dashboards  │
                        │   Storage    │      │   & Alerts  │
                        └──────────────┘      └─────────────┘
```

## Benefits

1. **Historical Data**: Track trends over time
2. **Real-time Monitoring**: See metrics as they happen
3. **Alerting**: Set up complex alert rules
4. **Visualization**: Professional dashboards
5. **Analysis**: Query and analyze performance data
6. **Scalability**: Handles millions of data points

## Troubleshooting

### InfluxDB Not Starting
```bash
# Check logs
docker logs agent-influxdb

# Restart
./influxdb_manager.sh restart
```

### Connection Issues
```bash
# Check if running
docker ps | grep influx

# Test health
curl http://localhost:8086/health
```

### Reset Everything
```bash
# WARNING: This deletes all data!
./influxdb_manager.sh clean
./influxdb_manager.sh start
```

## Next Steps

1. Complete initial setup via web UI
2. Configure Grafana dashboards
3. Set up alerts for critical metrics
4. Create retention policies for data
5. Integrate with production monitoring

## Security Notes

⚠️ **Important**: The default passwords in `docker-compose.yml` should be changed for production use!

Update these values:
- `INFLUXDB_ADMIN_PASSWORD`
- `INFLUXDB_INIT_PASSWORD`
- `INFLUXDB_INIT_ADMIN_TOKEN`
- `GF_SECURITY_ADMIN_PASSWORD`