# Agent Lightning Monitoring Setup

Welcome to the Agent Lightning monitoring system! This guide will help you get started with monitoring your AI agents using Prometheus, Grafana, and Alertmanager.

## 🚀 Quick Start (For Non-IT Users)

### Step 1: Prerequisites
Make sure you have Docker installed on your system. That's it!

### Step 2: Start Monitoring
Open a terminal and navigate to the monitoring folder:

```bash
cd monitoring
./start_monitoring.sh
```

That's it! The monitoring stack will start automatically and show you the access URLs.

### Step 3: Access Your Dashboards

Once started, you'll see output like this:

```
🎉 Agent Lightning Monitoring Stack is Running!

📊 Access your monitoring dashboards:

   🌐 Grafana (Main Dashboard):
      http://localhost:3000
      Username: admin
      Password: admin123

   📈 Prometheus (Metrics):
      http://localhost:9090

   🚨 Alertmanager (Alerts):
      http://localhost:9093
```

### Step 4: View Your Metrics

1. **Open Grafana**: Go to http://localhost:3000
2. **Login** with username `admin` and password `admin123`
3. **Change Password**: Please change the default password when prompted
4. **View Dashboard**: The "Agent Lightning - System Overview" dashboard will show all your metrics

## 📊 What You'll See

### System Health Overview
- Real-time status of all 12 services
- Service up/down indicators
- Overall system health

### Request Monitoring
- Total requests per service
- Request rates over time
- Error rates and trends

### Service-Specific Metrics
- **Agent Coordination**: Coordinator and Designer performance
- **AI Models**: Model inference and LangChain operations
- **Memory Systems**: Memory management and retrieval
- **Workflow Engines**: Orchestration and RL training
- **Communication**: WebSocket connections and events

### System Resources
- CPU usage by service
- Memory consumption
- System performance metrics

## 🎛️ Available Commands

The monitoring script supports several commands:

```bash
# Start monitoring (default)
./start_monitoring.sh

# Check status of services
./start_monitoring.sh status

# View service logs
./start_monitoring.sh logs

# Stop monitoring
./start_monitoring.sh stop

# Restart monitoring
./start_monitoring.sh restart

# Show help
./start_monitoring.sh help
```

## 🔧 Troubleshooting

### Monitoring Won't Start
- Make sure Docker is running
- Check if ports 3000, 9090, 9093, 9100 are available
- Try: `./start_monitoring.sh stop` then `./start_monitoring.sh start`

### Can't Access Grafana
- Wait a moment for services to fully start
- Check if port 3000 is accessible
- Try refreshing the browser

### No Metrics Showing
- Make sure your Agent Lightning services are running
- Services need to be accessible on their configured ports
- Check service logs for errors

### Services Not Healthy
- Verify your services are running and responding
- Check service endpoints with `/health`
- Review service logs for issues

## 📈 Understanding Your Metrics

### Key Metrics to Monitor

1. **Service Health**: Green = healthy, Red = issues
2. **Request Rates**: How busy your services are
3. **Error Rates**: Should stay below 5%
4. **Response Times**: Keep under 1 second for good UX
5. **Resource Usage**: Monitor CPU/memory trends

### Common Alerts

- **Service Down**: Critical - service stopped responding
- **High Error Rate**: Warning - too many failed requests
- **High CPU/Memory**: Warning - resource constraints
- **WebSocket Issues**: Info - connection problems

## 🛠️ Advanced Configuration

### Custom Dashboards
- Import additional dashboards via Grafana UI
- Create custom panels for specific metrics
- Set up custom alerts and notifications

### Alert Configuration
- Modify `alert_rules.yml` for custom alerts
- Configure email notifications in `alertmanager.yml`
- Set up Slack/webhook integrations

### Scaling
- Add more services to `working_prometheus.yml`
- Configure service discovery for dynamic environments
- Set up high availability with multiple Prometheus instances

## 📚 Additional Resources

- **Full Documentation**: See `docs/PROMETHEUS_METRICS_IMPLEMENTATION.md`
- **Grafana Docs**: https://grafana.com/docs/
- **Prometheus Docs**: https://prometheus.io/docs/
- **Alertmanager Docs**: https://prometheus.io/docs/alerting/latest/alertmanager/

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review service logs: `./start_monitoring.sh logs`
3. Check the full documentation
4. Contact the Agent Lightning team

---

**Happy Monitoring! 🎉**

Your Agent Lightning system is now fully observable with enterprise-grade monitoring capabilities.