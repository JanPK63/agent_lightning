#!/usr/bin/env python3
"""
Execute Key Rotation Script
Safe execution of encryption key rotation with monitoring and rollback.
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.key_rotation_service import KeyRotationService
from shared.database import init_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/key_rotation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python execute_key_rotation.py <environment> [--dry-run] [--data-key-id KEY] [--field-key-id KEY]")
        sys.exit(1)

    environment = sys.argv[1]
    dry_run = '--dry-run' in sys.argv

    # Parse optional key IDs
    data_key_id = None
    field_key_id = None

    if '--data-key-id' in sys.argv:
        idx = sys.argv.index('--data-key-id')
        if idx + 1 < len(sys.argv):
            data_key_id = sys.argv[idx + 1]

    if '--field-key-id' in sys.argv:
        idx = sys.argv.index('--field-key-id')
        if idx + 1 < len(sys.argv):
            field_key_id = sys.argv[idx + 1]

    # Initialize database
    logger.info(f"Initializing database for {environment}")
    init_database()

    # Create rotation service
    rotation_service = KeyRotationService(batch_size=100, max_duration_hours=2)

    # Execute rotation
    logger.info(f"Starting key rotation in {environment} (dry_run={dry_run})")

    result = rotation_service.rotate_keys(
        data_key_id=data_key_id,
        field_key_id=field_key_id,
        environment=environment,
        dry_run=dry_run
    )

    # Log results
    logger.info(f"Rotation completed: {result}")

    # Save results to file
    results_dir = Path('exports/rotation_reports')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'rotation_{environment}_{timestamp}.json'

    results_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'environment': environment,
        'dry_run': dry_run,
        'success': result.success,
        'status': result.status.value,
        'records_processed': result.records_processed,
        'records_failed': result.records_failed,
        'duration_seconds': result.duration_seconds,
        'new_data_key_id': result.new_data_key_id,
        'new_field_key_id': result.new_field_key_id,
        'error_message': result.error_message,
        'rollback_available': result.rollback_available
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"Results saved to {results_file}")

    # Print summary
    print("\n" + "="*50)
    print("KEY ROTATION SUMMARY")
    print("="*50)
    print(f"Environment: {environment}")
    print(f"Dry Run: {dry_run}")
    print(f"Success: {result.success}")
    print(f"Status: {result.status.value}")
    print(f"Records Processed: {result.records_processed}")
    print(f"Records Failed: {result.records_failed}")
    print(".2f")
    if result.new_data_key_id:
        print(f"New Data Key: {result.new_data_key_id}")
    if result.new_field_key_id:
        print(f"New Field Key: {result.new_field_key_id}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    print(f"Rollback Available: {result.rollback_available}")
    print(f"Results File: {results_file}")
    print("="*50)

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == '__main__':
    main()