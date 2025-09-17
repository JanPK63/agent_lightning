#!/bin/bash
# InfluxDB Backup Scheduler
# Runs automated backups using cron or as a background service

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function setup_cron() {
    echo -e "${GREEN}Setting up automated backup schedule...${NC}"
    
    # Create cron entries
    CRON_DAILY="0 2 * * * cd $SCRIPT_DIR && /usr/bin/python3 influxdb_backup_system.py backup --type daily >> $SCRIPT_DIR/backups/backup.log 2>&1"
    CRON_WEEKLY="0 3 * * 0 cd $SCRIPT_DIR && /usr/bin/python3 influxdb_backup_system.py backup --type weekly >> $SCRIPT_DIR/backups/backup.log 2>&1"
    CRON_MONTHLY="0 4 1 * * cd $SCRIPT_DIR && /usr/bin/python3 influxdb_backup_system.py backup --type monthly >> $SCRIPT_DIR/backups/backup.log 2>&1"
    CRON_CLEANUP="0 5 * * * cd $SCRIPT_DIR && /usr/bin/python3 influxdb_backup_system.py cleanup >> $SCRIPT_DIR/backups/backup.log 2>&1"
    
    # Check if cron entries already exist
    crontab -l 2>/dev/null | grep -q "influxdb_backup_system.py"
    if [ $? -eq 0 ]; then
        echo -e "${YELLOW}Backup cron jobs already exist. Updating...${NC}"
        # Remove existing entries
        crontab -l | grep -v "influxdb_backup_system.py" | crontab -
    fi
    
    # Add new cron entries
    (crontab -l 2>/dev/null; echo "$CRON_DAILY") | crontab -
    (crontab -l 2>/dev/null; echo "$CRON_WEEKLY") | crontab -
    (crontab -l 2>/dev/null; echo "$CRON_MONTHLY") | crontab -
    (crontab -l 2>/dev/null; echo "$CRON_CLEANUP") | crontab -
    
    echo -e "${GREEN}✅ Automated backup schedule configured:${NC}"
    echo "   • Daily backups at 2:00 AM"
    echo "   • Weekly backups on Sunday at 3:00 AM"
    echo "   • Monthly backups on 1st at 4:00 AM"
    echo "   • Cleanup old backups daily at 5:00 AM"
    echo ""
    echo "View cron jobs with: crontab -l"
    echo "Remove cron jobs with: crontab -l | grep -v influxdb_backup_system.py | crontab -"
}

function manual_backup() {
    echo -e "${GREEN}Creating manual backup...${NC}"
    python3 "$SCRIPT_DIR/influxdb_backup_system.py" backup --type manual
}

function restore_backup() {
    if [ -z "$1" ]; then
        echo -e "${RED}Please specify a backup file to restore${NC}"
        echo "Usage: $0 restore <backup_file>"
        python3 "$SCRIPT_DIR/influxdb_backup_system.py" list
        exit 1
    fi
    
    echo -e "${YELLOW}⚠️  WARNING: This will restore InfluxDB data from backup${NC}"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        python3 "$SCRIPT_DIR/influxdb_backup_system.py" restore --file "$1"
    else
        echo "Restore cancelled"
    fi
}

function list_backups() {
    python3 "$SCRIPT_DIR/influxdb_backup_system.py" list
}

function run_scheduler() {
    echo -e "${GREEN}Starting backup scheduler service...${NC}"
    echo "Press Ctrl+C to stop"
    python3 "$SCRIPT_DIR/influxdb_backup_system.py" schedule
}

function show_usage() {
    echo "InfluxDB Backup Management"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup      - Set up automated backup schedule (cron)"
    echo "  backup     - Create manual backup now"
    echo "  restore    - Restore from backup file"
    echo "  list       - List available backups"
    echo "  scheduler  - Run backup scheduler (foreground)"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Set up automated backups"
    echo "  $0 backup                   # Create manual backup"
    echo "  $0 restore backup_file.tar.gz  # Restore from backup"
    echo "  $0 list                     # List all backups"
}

# Main script logic
case "$1" in
    setup)
        setup_cron
        ;;
    backup)
        manual_backup
        ;;
    restore)
        restore_backup "$2"
        ;;
    list)
        list_backups
        ;;
    scheduler)
        run_scheduler
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        if [ -z "$1" ]; then
            manual_backup
        else
            echo -e "${RED}Unknown command: $1${NC}"
            show_usage
            exit 1
        fi
        ;;
esac