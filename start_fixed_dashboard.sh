#!/bin/bash
# Start Fixed Agent Dashboard

echo "🚀 Starting Fixed Agent Lightning Dashboard..."

# Kill any existing Streamlit processes on port 8051
echo "Stopping existing dashboard..."
lsof -ti:8051 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 2

# Start the updated dashboard
echo "Starting updated dashboard on port 8051..."
streamlit run update_dashboard.py --server.port 8051 --server.address localhost &

# Wait for startup
sleep 3

# Check if it's running
if curl -s http://localhost:8051 > /dev/null; then
    echo "✅ Dashboard is running!"
    echo "   🌐 URL: http://localhost:8051"
    echo "   📱 Dashboard: Fixed Agent System Integration"
    echo ""
    echo "🎯 Next Steps:"
    echo "1. Open http://localhost:8051 in your browser"
    echo "2. Go to 'Fixed Task Assignment' tab"
    echo "3. Click 'Connect' to connect to fixed agents"
    echo "4. Submit tasks and see agents actually work!"
else
    echo "❌ Dashboard failed to start"
    echo "Check for errors above"
fi