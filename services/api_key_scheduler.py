"""
API Key Rotation Scheduler
Background service for automatic API key rotation and maintenance
"""

import time
import logging
import schedule
from datetime import datetime, timedelta
from typing import Optional, Callable
import threading
import signal
import sys

from api_key_rotation_service import api_key_rotation_service, NotificationInfo
from shared.events import EventBus, EventChannel

logger = logging.getLogger(__name__)


class ApiKeyScheduler:
    """Scheduler for automatic API key rotation and maintenance tasks"""

    def __init__(self, check_interval_minutes: int = 60):
        """Initialize the scheduler

        Args:
            check_interval_minutes: How often to check for keys due for rotation
        """
        self.check_interval_minutes = check_interval_minutes
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.event_bus = EventBus("api_key_scheduler")
        self.event_bus.start()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"API Key Scheduler initialized with {check_interval_minutes} minute intervals")

    def start(self):
        """Start the scheduler in a background thread"""
        if self.running:
            logger.warning("Scheduler is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()

        logger.info("API Key Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        try:
            self.event_bus.stop()
        except:
            pass

        logger.info("API Key Scheduler stopped")

    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Starting scheduler loop")

        # Schedule tasks
        schedule.every(self.check_interval_minutes).minutes.do(self._check_and_rotate_keys)
        schedule.every(24).hours.do(self._cleanup_expired_keys)
        schedule.every(6).hours.do(self._send_rotation_notifications)

        # Run initial check
        self._check_and_rotate_keys()
        self._cleanup_expired_keys()
        self._send_rotation_notifications()

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for pending tasks
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)

        logger.info("Scheduler loop ended")

    def _check_and_rotate_keys(self):
        """Check for keys due for rotation and rotate them"""
        try:
            logger.info("Checking for API keys due for rotation")

            # Get keys due for rotation (overdue only)
            due_keys = api_key_rotation_service.get_keys_due_for_rotation(days_ahead=0)

            if not due_keys:
                logger.info("No API keys due for rotation")
                return

            logger.info(f"Found {len(due_keys)} API keys due for rotation")

            # Rotate each key
            rotated_count = 0
            failed_count = 0

            for key_info in due_keys:
                try:
                    result = api_key_rotation_service.rotate_api_key(
                        api_key_id=key_info['id'],
                        reason="scheduled"
                    )

                    if result.success:
                        rotated_count += 1
                        logger.info(f"Successfully rotated API key {key_info['id']} ({key_info['name']})")

                        # Emit success event
                        self.event_bus.emit(
                            EventChannel.SYSTEM_ALERT,
                            {
                                "type": "scheduled_rotation_success",
                                "api_key_id": key_info['id'],
                                "key_name": key_info['name'],
                                "user_id": key_info['user_id']
                            }
                        )
                    else:
                        failed_count += 1
                        logger.error(f"Failed to rotate API key {key_info['id']}: {result.error_message}")

                        # Emit failure event
                        self.event_bus.emit(
                            EventChannel.SYSTEM_ALERT,
                            {
                                "type": "scheduled_rotation_failed",
                                "api_key_id": key_info['id'],
                                "key_name": key_info['name'],
                                "error": result.error_message
                            }
                        )

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Unexpected error rotating key {key_info['id']}: {e}")

            logger.info(f"Rotation check completed: {rotated_count} rotated, {failed_count} failed")

        except Exception as e:
            logger.error(f"Error in rotation check: {e}")

    def _cleanup_expired_keys(self):
        """Clean up expired API keys"""
        try:
            logger.info("Starting cleanup of expired API keys")

            cleaned_count = api_key_rotation_service.cleanup_expired_keys()

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired API keys")

                # Emit cleanup event
                self.event_bus.emit(
                    EventChannel.SYSTEM_ALERT,
                    {
                        "type": "expired_keys_cleanup",
                        "cleaned_count": cleaned_count
                    }
                )
            else:
                logger.info("No expired API keys to clean up")

        except Exception as e:
            logger.error(f"Error in expired key cleanup: {e}")

    def _send_rotation_notifications(self):
        """Send notifications for upcoming key rotations"""
        try:
            logger.info("Checking for API keys requiring rotation notifications")

            notifications = api_key_rotation_service.get_pending_notifications()

            if not notifications:
                logger.info("No API keys require rotation notifications")
                return

            logger.info(f"Sending notifications for {len(notifications)} API keys")

            sent_count = 0
            failed_count = 0

            for notification in notifications:
                try:
                    success = self._send_notification(notification)
                    if success:
                        sent_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error sending notification for key {notification.api_key_id}: {e}")

            logger.info(f"Notification check completed: {sent_count} sent, {failed_count} failed")

        except Exception as e:
            logger.error(f"Error in notification check: {e}")

    def _send_notification(self, notification: NotificationInfo) -> bool:
        """Send a rotation notification

        Args:
            notification: Notification information

        Returns:
            True if notification was sent successfully
        """
        try:
            # For now, just log the notification
            # In a real implementation, this would send email/SMS/webhook notifications

            message = f"""
API Key Rotation Notification

Your API key '{notification.key_name}' is scheduled for rotation in {notification.days_until_rotation} days.

Rotation Date: {notification.rotation_date.strftime('%Y-%m-%d %H:%M:%S UTC')}

Please ensure your applications are updated to use the new key after rotation.
The old key will remain valid for a grace period after rotation.

If you have any questions, please contact support.
"""

            logger.info(f"NOTIFICATION for user {notification.user_id}: {message.strip()}")

            # TODO: Implement actual notification sending (email, SMS, webhook, etc.)
            # For now, we'll just mark it as sent in the database

            # In a real implementation, you would:
            # 1. Send email to user's email address
            # 2. Send webhook notification if configured
            # 3. Update notification status in database

            return True

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down scheduler")
        self.stop()

    def run_once(self):
        """Run all scheduled tasks once (for testing/manual execution)"""
        logger.info("Running scheduler tasks once")

        self._check_and_rotate_keys()
        self._cleanup_expired_keys()
        self._send_rotation_notifications()

        logger.info("One-time scheduler run completed")


def main():
    """Main entry point for running the scheduler as a service"""
    import argparse

    parser = argparse.ArgumentParser(description="API Key Rotation Scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in minutes (default: 60)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run tasks once and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    scheduler = ApiKeyScheduler(check_interval_minutes=args.interval)

    if args.once:
        scheduler.run_once()
    else:
        logger.info("Starting API Key Scheduler service")
        scheduler.start()

        try:
            # Keep the main thread alive
            while scheduler.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            scheduler.stop()

        logger.info("API Key Scheduler service stopped")


if __name__ == "__main__":
    main()