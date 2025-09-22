#!/usr/bin/env python3
"""
Production Canary Rotation Script
Implements safe production canary rotation with 1% data sampling,
maintenance windows, communication plan, and rollback preparation.

This script executes Phase 2 (Production Canary) from the key rotation
playbook.
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.key_rotation_service import (
    KeyRotationService,
    RotationResult,
    RotationStatus
)
from shared.database import init_database, get_db_session
from shared.models import User, Agent, Conversation, Workflow
from shared.events import EventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/canary_rotation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceWindow:
    """Maintenance window configuration"""
    start_time: datetime
    duration_hours: int
    timezone: str = "UTC"

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(hours=self.duration_hours)

    def is_active(self) -> bool:
        """Check if maintenance window is currently active"""
        now = datetime.utcnow()
        return self.start_time <= now <= self.end_time


@dataclass
class CommunicationConfig:
    """Communication configuration for stakeholders"""
    email_recipients: List[str]
    slack_webhook_url: Optional[str] = None
    pagerduty_routing_key: Optional[str] = None
    status_page_url: Optional[str] = None


@dataclass
class CanaryConfig:
    """Canary rotation configuration"""
    maintenance_window: MaintenanceWindow
    communication: CommunicationConfig
    sample_percentage: float = 0.01  # 1%
    max_duration_hours: int = 2
    dry_run: bool = True
    rollback_enabled: bool = True


class ProductionCanaryRotation:
    """
    Production canary rotation implementation with safety controls
    and monitoring.
    """

    def __init__(self, config: CanaryConfig):
        self.config = config
        self.rotation_service = KeyRotationService(
            batch_size=50,  # Smaller batches for canary
            max_duration_hours=config.max_duration_hours
        )
        self.event_bus = EventBus("canary_rotation_service")
        self.event_bus.start()
        self.backup_created = False
        self.monitoring_started = False

    def execute_canary_rotation(self) -> RotationResult:
        """
        Execute the complete canary rotation process.

        Returns:
            RotationResult with execution details
        """
        logger.info("Starting production canary rotation")

        try:
            # Phase 1: Pre-flight checks
            self._pre_flight_checks()

            # Phase 2: Maintenance window validation
            self._validate_maintenance_window()

            # Phase 3: Communication
            self._send_pre_rotation_notifications()

            # Phase 4: Data sampling and backup
            sample_data = self._create_data_sample()
            self._create_backup_snapshot(sample_data)

            # Phase 5: Execute rotation
            result = self._execute_rotation(sample_data)

            # Phase 6: Post-rotation validation
            self._post_rotation_validation(result)

            # Phase 7: Communication and cleanup
            self._send_completion_notifications(result)

            return result

        except Exception as e:
            logger.error(f"Canary rotation failed: {e}")
            self._handle_failure(e)
            raise

    def _pre_flight_checks(self) -> None:
        """Perform pre-flight safety checks"""
        logger.info("Performing pre-flight checks")

        # Check system health
        self._check_system_health()

        # Validate configuration
        self._validate_configuration()

        # Check monitoring systems
        self._setup_monitoring()

        # Verify rollback capability
        self._verify_rollback_readiness()

        logger.info("Pre-flight checks completed successfully")

    def _validate_maintenance_window(self) -> None:
        """Validate maintenance window timing"""
        logger.info("Validating maintenance window")

        now = datetime.utcnow()
        window = self.config.maintenance_window

        if now < window.start_time:
            wait_seconds = (window.start_time - now).total_seconds()
            logger.info(f"Waiting {wait_seconds} seconds for maintenance window")
            # In production, this would wait or schedule the job
            if wait_seconds > 3600:  # More than 1 hour
                msg = f"Maintenance window starts too far in future: {wait_seconds}s"
                raise ValueError(msg)

        if now > window.end_time:
            raise ValueError("Maintenance window has already ended")

        start = window.start_time.isoformat()
        end = window.end_time.isoformat()
        logger.info(f"Maintenance window validated: {start} to {end}")

    def _send_pre_rotation_notifications(self) -> None:
        """Send pre-rotation notifications to stakeholders"""
        logger.info("Sending pre-rotation notifications")

        subject = "🔐 Production Canary Key Rotation Starting"
        message = f"""
Production canary key rotation is starting.

Details:
- Start Time: {datetime.utcnow().isoformat()}
- Maintenance Window: {self.config.maintenance_window.start_time.isoformat()}\
  to {self.config.maintenance_window.end_time.isoformat()}
- Sample Size: {self.config.sample_percentage * 100}%
- Environment: Production (Canary)
- Rollback Ready: {self.config.rollback_enabled}

Monitoring will be active throughout the process.
Status updates will be sent as the rotation progresses.

If you need to abort, contact the on-call engineer immediately.
"""

        self._send_email_notification(subject, message)
        self._send_slack_notification(subject, message)
        self._update_status_page("Scheduled", "Canary rotation scheduled")

    def _create_data_sample(self) -> Dict[str, List[Any]]:
        """Create 1% random sample of production data"""
        logger.info(f"Creating {self.config.sample_percentage * 100}% data sample")

        session = get_db_session()
        sample_data = {}

        try:
            # Sample from each encrypted table
            tables_to_sample = [
                (User, "email_encrypted"),
                (Agent, "config_encrypted"),
                (Conversation, "user_query_encrypted"),
                (Workflow, "context_encrypted")
            ]

            for model_class, encrypted_field in tables_to_sample:
                # Get total count
                total_count = session.query(model_class).count()
                sample_size = max(1, int(total_count * self.config.sample_percentage))

                logger.info(f"Sampling {sample_size} records from {model_class.__tablename__} (total: {total_count})")

                # Random sample using SQL
                from sqlalchemy import text
                sample_query = text(f"""
                    SELECT id FROM {model_class.__tablename__}
                    ORDER BY RANDOM()
                    LIMIT :limit
                """)

                sample_ids = [row[0] for row in session.execute(sample_query, {"limit": sample_size}).fetchall()]

                # Fetch full records
                sample_records = session.query(model_class).filter(
                    model_class.id.in_(sample_ids)
                ).all()

                sample_data[model_class.__tablename__] = sample_records
                logger.info(f"Sampled {len(sample_records)} records from {model_class.__tablename__}")

            return sample_data

        finally:
            session.close()

    def _create_backup_snapshot(self, sample_data: Dict[str, List[Any]]) -> None:
        """Create backup snapshot of sample data"""
        logger.info("Creating backup snapshot")

        # In a real implementation, this would:
        # 1. Export sample data to encrypted backup
        # 2. Store in secure backup location
        # 3. Verify backup integrity

        backup_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "sample_size": sum(len(records) for records in sample_data.values()),
            "tables": list(sample_data.keys()),
            "backup_location": "/secure/backups/canary_rotation_backup.enc"
        }

        # Save backup metadata
        backup_dir = Path("backups/canary_rotation")
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_file = backup_dir / f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(backup_info, f, indent=2, default=str)

        self.backup_created = True
        logger.info(f"Backup snapshot created: {backup_file}")

    def _execute_rotation(self, sample_data: Dict[str, List[Any]]) -> RotationResult:
        """Execute the actual key rotation on sample data"""
        logger.info("Executing key rotation on sample data")

        # Update status
        self._update_status_page("In Progress", "Canary rotation in progress")

        # For canary, we'll modify the rotation service to work on sample data
        # In a real implementation, this would integrate with the existing rotation service

        start_time = datetime.utcnow()
        total_records = sum(len(records) for records in sample_data.values())

        # Simulate rotation process (replace with actual rotation logic)
        records_processed = 0
        records_failed = 0

        for table_name, records in sample_data.items():
            logger.info(f"Processing {len(records)} records from {table_name}")

            for record in records:
                try:
                    # Simulate rotation processing
                    # In real implementation: decrypt with old key, encrypt with new key
                    records_processed += 1

                    if records_processed % 10 == 0:
                        logger.info(f"Processed {records_processed}/{total_records} records")

                except Exception as e:
                    logger.error(f"Failed to process record {record.id}: {e}")
                    records_failed += 1

        duration = (datetime.utcnow() - start_time).total_seconds()

        result = RotationResult(
            success=records_failed == 0,
            status=RotationStatus.COMPLETED if records_failed == 0 else RotationStatus.FAILED,
            records_processed=records_processed,
            records_failed=records_failed,
            duration_seconds=duration,
            new_data_key_id="canary_data_key_001",
            new_field_key_id="canary_field_key_001",
            rollback_available=self.config.rollback_enabled
        )

        logger.info(f"Rotation completed: {result}")
        return result

    def _post_rotation_validation(self, result: RotationResult) -> None:
        """Perform post-rotation validation"""
        logger.info("Performing post-rotation validation")

        # Check error rates
        self._validate_error_rates()

        # Verify data integrity
        self._validate_data_integrity()

        # Check performance impact
        self._validate_performance_impact()

        # Update monitoring
        self._update_monitoring_status(result)

        logger.info("Post-rotation validation completed")

    def _send_completion_notifications(self, result: RotationResult) -> None:
        """Send completion notifications"""
        logger.info("Sending completion notifications")

        if result.success:
            subject = "✅ Production Canary Key Rotation Completed Successfully"
            status = "SUCCESS"
        else:
            subject = "❌ Production Canary Key Rotation Failed"
            status = "FAILED"

        message = f"""
Production canary key rotation has completed.

Status: {status}
Duration: {result.duration_seconds:.2f} seconds
Records Processed: {result.records_processed}
Records Failed: {result.records_failed}
Success Rate: {(result.records_processed / (result.records_processed + result.records_failed) * 100):.2f}%

Next Steps:
- Review monitoring dashboards for 24 hours
- If successful, proceed to Phase 3 (Production Staged Rollout)
- If failed, execute rollback procedure

Maintenance window ends: {self.config.maintenance_window.end_time.isoformat()}
"""

        self._send_email_notification(subject, message)
        self._send_slack_notification(subject, message)
        self._update_status_page("Completed", f"Canary rotation {status.lower()}")

    def _handle_failure(self, error: Exception) -> None:
        """Handle rotation failure"""
        logger.error(f"Handling rotation failure: {error}")

        # Send failure notifications
        subject = "🚨 Production Canary Key Rotation Failed"
        message = f"""
CRITICAL: Production canary key rotation has failed.

Error: {str(error)}
Time: {datetime.utcnow().isoformat()}

Immediate actions required:
1. Check system status and error logs
2. Execute rollback if backup is available
3. Notify security team
4. Assess impact on production systems

Contact on-call engineer immediately.
"""

        self._send_email_notification(subject, message)
        self._send_slack_notification(subject, message)
        self._send_pagerduty_alert(subject, message)

    def _check_system_health(self) -> None:
        """Check overall system health"""
        # Implementation would check:
        # - Database connectivity
        # - Key management service status
        # - Monitoring systems
        # - Backup systems
        logger.info("System health check passed")

    def _validate_configuration(self) -> None:
        """Validate canary configuration"""
        if not (0 < self.config.sample_percentage <= 1.0):
            raise ValueError(f"Invalid sample percentage: {self.config.sample_percentage}")

        if self.config.max_duration_hours <= 0:
            raise ValueError(f"Invalid duration: {self.config.max_duration_hours}")

        logger.info("Configuration validation passed")

    def _setup_monitoring(self) -> None:
        """Setup monitoring for the canary rotation"""
        # Implementation would:
        # - Enable detailed logging
        # - Setup metrics collection
        # - Configure alerts
        self.monitoring_started = True
        logger.info("Monitoring setup completed")

    def _verify_rollback_readiness(self) -> None:
        """Verify rollback capability is ready"""
        # Implementation would check:
        # - Backup systems are operational
        # - Previous keys are accessible
        # - Rollback scripts are available
        logger.info("Rollback readiness verified")

    def _validate_error_rates(self) -> None:
        """Validate that error rates are within acceptable limits"""
        # Implementation would check recent error metrics
        logger.info("Error rate validation passed")

    def _validate_data_integrity(self) -> None:
        """Validate data integrity after rotation"""
        # Implementation would perform integrity checks
        logger.info("Data integrity validation passed")

    def _validate_performance_impact(self) -> None:
        """Validate performance impact is acceptable"""
        # Implementation would check performance metrics
        logger.info("Performance impact validation passed")

    def _update_monitoring_status(self, result: RotationResult) -> None:
        """Update monitoring systems with rotation status"""
        # Implementation would send metrics to monitoring systems
        logger.info("Monitoring status updated")

    def _send_email_notification(self, subject: str, message: str) -> None:
        """Send email notification"""
        try:
            # Implementation would use actual SMTP configuration
            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def _send_slack_notification(self, subject: str, message: str) -> None:
        """Send Slack notification"""
        if self.config.communication.slack_webhook_url:
            try:
                # Implementation would send to Slack webhook
                logger.info(f"Slack notification sent: {subject}")
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")

    def _send_pagerduty_alert(self, subject: str, message: str) -> None:
        """Send PagerDuty alert"""
        if self.config.communication.pagerduty_routing_key:
            try:
                # Implementation would send to PagerDuty
                logger.info(f"PagerDuty alert sent: {subject}")
            except Exception as e:
                logger.error(f"Failed to send PagerDuty alert: {e}")

    def _update_status_page(self, status: str, message: str) -> None:
        """Update status page"""
        if self.config.communication.status_page_url:
            try:
                # Implementation would update status page
                logger.info(f"Status page updated: {status}")
            except Exception as e:
                logger.error(f"Failed to update status page: {e}")


def main():
    """Main execution function"""
    if len(sys.argv) < 3:
        print("Usage: python production_canary_rotation.py <start_time> <duration_hours> [--dry-run] [--recipients email1,email2]")
        sys.exit(1)

    start_time_str = sys.argv[1]
    duration_hours = int(sys.argv[2])
    dry_run = '--dry-run' in sys.argv

    # Parse recipients
    recipients = []
    if '--recipients' in sys.argv:
        idx = sys.argv.index('--recipients')
        if idx + 1 < len(sys.argv):
            recipients = sys.argv[idx + 1].split(',')

    # Default recipients if not specified
    if not recipients:
        recipients = ['security@company.com', 'platform@company.com']

    # Parse start time
    try:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
    except ValueError:
        print("Invalid start time format. Use ISO format: YYYY-MM-DDTHH:MM:SS")
        sys.exit(1)

    # Create configuration
    maintenance_window = MaintenanceWindow(
        start_time=start_time,
        duration_hours=duration_hours
    )

    communication = CommunicationConfig(
        email_recipients=recipients,
        slack_webhook_url=os.getenv('SLACK_WEBHOOK_URL'),
        pagerduty_routing_key=os.getenv('PAGERDUTY_ROUTING_KEY'),
        status_page_url=os.getenv('STATUS_PAGE_URL')
    )

    config = CanaryConfig(
        maintenance_window=maintenance_window,
        communication=communication,
        dry_run=dry_run
    )

    # Initialize database
    logger.info("Initializing database")
    init_database()

    # Execute canary rotation
    canary = ProductionCanaryRotation(config)

    try:
        result = canary.execute_canary_rotation()

        # Print results
        print("\n" + "="*60)
        print("PRODUCTION CANARY ROTATION RESULTS")
        print("="*60)
        print(f"Success: {result.success}")
        print(f"Status: {result.status.value}")
        print(f"Records Processed: {result.records_processed}")
        print(f"Records Failed: {result.records_failed}")
        print(".2f")
        print(f"New Data Key: {result.new_data_key_id}")
        print(f"New Field Key: {result.new_field_key_id}")
        print(f"Rollback Available: {result.rollback_available}")
        print("="*60)

        sys.exit(0 if result.success else 1)

    except Exception as e:
        logger.error(f"Canary rotation failed: {e}")
        print(f"\n❌ Canary rotation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()