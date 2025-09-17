#!/usr/bin/env python3
"""
InfluxDB Data Migration Tools
Migrate data between InfluxDB instances or to other database systems
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import pymongo
from tqdm import tqdm

# Load environment
load_dotenv('.env.influxdb')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TargetDatabase(Enum):
    """Supported target databases for migration"""
    INFLUXDB = "influxdb"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    CSV = "csv"
    PARQUET = "parquet"


@dataclass
class MigrationConfig:
    """Configuration for data migration"""
    source_bucket: str
    target_type: TargetDatabase
    target_config: Dict[str, Any]
    time_range: Tuple[str, str] = ("-30d", "now()")
    batch_size: int = 1000
    measurement_filter: Optional[str] = None
    tag_filters: Optional[Dict[str, str]] = None
    transform_function: Optional[callable] = None


class DataMigrator:
    """Main data migration orchestrator"""
    
    def __init__(self):
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN', 'agent-system-token-supersecret-12345678')
        self.org = os.getenv('INFLUXDB_ORG', 'agent-system')
        
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        
        self.migration_dir = Path('migrations')
        self.migration_dir.mkdir(exist_ok=True)
        
        # Migration statistics
        self.stats = {
            'records_read': 0,
            'records_written': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def migrate(self, config: MigrationConfig) -> bool:
        """Execute data migration based on configuration"""
        
        logger.info(f"Starting migration from {config.source_bucket} to {config.target_type.value}")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Read data from InfluxDB
            data = self._read_source_data(config)
            
            if data.empty:
                logger.warning("No data to migrate")
                return False
            
            # Apply transformation if specified
            if config.transform_function:
                data = config.transform_function(data)
            
            # Write to target database
            success = self._write_to_target(data, config)
            
            self.stats['end_time'] = datetime.now()
            self._save_migration_report(config)
            
            return success
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.stats['errors'] += 1
            return False
    
    def _read_source_data(self, config: MigrationConfig) -> pd.DataFrame:
        """Read data from InfluxDB source"""
        
        logger.info(f"Reading data from {config.source_bucket}...")
        
        # Build Flux query
        query = f'''
from(bucket: "{config.source_bucket}")
    |> range(start: {config.time_range[0]}, stop: {config.time_range[1]})'''
        
        if config.measurement_filter:
            query += f'''
    |> filter(fn: (r) => r["_measurement"] == "{config.measurement_filter}")'''
        
        if config.tag_filters:
            for tag, value in config.tag_filters.items():
                query += f'''
    |> filter(fn: (r) => r["{tag}"] == "{value}")'''
        
        # Execute query
        query_api = self.client.query_api()
        tables = query_api.query(query, org=self.org)
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                row = {
                    'time': record.get_time(),
                    'measurement': record.get_measurement(),
                    'field': record.get_field(),
                    'value': record.get_value()
                }
                # Add tags
                for key, val in record.values.items():
                    if not key.startswith('_') and key not in row:
                        row[key] = val
                records.append(row)
        
        df = pd.DataFrame(records)
        self.stats['records_read'] = len(df)
        
        logger.info(f"  Read {len(df):,} records")
        return df
    
    def _write_to_target(self, data: pd.DataFrame, config: MigrationConfig) -> bool:
        """Write data to target database"""
        
        if config.target_type == TargetDatabase.INFLUXDB:
            return self._write_to_influxdb(data, config.target_config)
        elif config.target_type == TargetDatabase.POSTGRESQL:
            return self._write_to_postgresql(data, config.target_config)
        elif config.target_type == TargetDatabase.SQLITE:
            return self._write_to_sqlite(data, config.target_config)
        elif config.target_type == TargetDatabase.MONGODB:
            return self._write_to_mongodb(data, config.target_config)
        elif config.target_type == TargetDatabase.CSV:
            return self._write_to_csv(data, config.target_config)
        elif config.target_type == TargetDatabase.PARQUET:
            return self._write_to_parquet(data, config.target_config)
        else:
            logger.error(f"Unsupported target type: {config.target_type}")
            return False
    
    def _write_to_influxdb(self, data: pd.DataFrame, config: Dict) -> bool:
        """Migrate to another InfluxDB instance"""
        
        logger.info("Writing to InfluxDB...")
        
        # Connect to target InfluxDB
        target_client = InfluxDBClient(
            url=config.get('url', 'http://localhost:8086'),
            token=config.get('token'),
            org=config.get('org', 'agent-system')
        )
        
        write_api = target_client.write_api(write_options=SYNCHRONOUS)
        bucket = config.get('bucket', 'migrated_data')
        
        # Write data in batches
        batch_size = config.get('batch_size', 1000)
        points = []
        
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Migrating"):
            point = Point(row['measurement'])
            
            # Add tags
            for col in data.columns:
                if col not in ['time', 'measurement', 'field', 'value']:
                    if pd.notna(row[col]):
                        point = point.tag(col, str(row[col]))
            
            # Add field
            if pd.notna(row.get('field')) and pd.notna(row.get('value')):
                point = point.field(row['field'], row['value'])
            
            # Add timestamp
            if pd.notna(row.get('time')):
                point = point.time(row['time'])
            
            points.append(point)
            
            # Write batch
            if len(points) >= batch_size:
                write_api.write(bucket=bucket, org=config.get('org'), record=points)
                self.stats['records_written'] += len(points)
                points = []
        
        # Write remaining points
        if points:
            write_api.write(bucket=bucket, org=config.get('org'), record=points)
            self.stats['records_written'] += len(points)
        
        target_client.close()
        logger.info(f"  Wrote {self.stats['records_written']:,} records")
        return True
    
    def _write_to_postgresql(self, data: pd.DataFrame, config: Dict) -> bool:
        """Migrate to PostgreSQL"""
        
        logger.info("Writing to PostgreSQL...")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=config.get('host', 'localhost'),
            port=config.get('port', 5432),
            database=config.get('database', 'metrics'),
            user=config.get('user', 'postgres'),
            password=config.get('password', '')
        )
        
        cur = conn.cursor()
        
        # Create table if not exists
        table_name = config.get('table', 'influx_metrics')
        
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP,
                measurement VARCHAR(255),
                field VARCHAR(255),
                value FLOAT,
                tags JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Prepare data for insertion
        records = []
        for _, row in data.iterrows():
            # Collect tags
            tags = {}
            for col in data.columns:
                if col not in ['time', 'measurement', 'field', 'value']:
                    if pd.notna(row[col]):
                        tags[col] = str(row[col])
            
            records.append((
                row.get('time'),
                row.get('measurement'),
                row.get('field'),
                float(row.get('value', 0)) if pd.notna(row.get('value')) else None,
                json.dumps(tags) if tags else None
            ))
        
        # Batch insert
        execute_batch(
            cur,
            f'''INSERT INTO {table_name} (time, measurement, field, value, tags)
                VALUES (%s, %s, %s, %s, %s)''',
            records,
            page_size=1000
        )
        
        self.stats['records_written'] = len(records)
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info(f"  Wrote {self.stats['records_written']:,} records")
        return True
    
    def _write_to_sqlite(self, data: pd.DataFrame, config: Dict) -> bool:
        """Migrate to SQLite"""
        
        logger.info("Writing to SQLite...")
        
        db_path = config.get('path', 'metrics.db')
        conn = sqlite3.connect(db_path)
        
        # Create table
        table_name = config.get('table', 'metrics')
        
        # Write DataFrame to SQLite
        data.to_sql(table_name, conn, if_exists='append', index=False)
        
        self.stats['records_written'] = len(data)
        
        conn.commit()
        conn.close()
        
        logger.info(f"  Wrote {self.stats['records_written']:,} records to {db_path}")
        return True
    
    def _write_to_mongodb(self, data: pd.DataFrame, config: Dict) -> bool:
        """Migrate to MongoDB"""
        
        logger.info("Writing to MongoDB...")
        
        # Connect to MongoDB
        client = pymongo.MongoClient(
            config.get('connection_string', 'mongodb://localhost:27017/')
        )
        
        db = client[config.get('database', 'metrics')]
        collection = db[config.get('collection', 'influx_data')]
        
        # Convert DataFrame to documents
        documents = []
        for _, row in data.iterrows():
            doc = row.to_dict()
            # Convert timestamp to datetime for MongoDB
            if 'time' in doc and pd.notna(doc['time']):
                doc['time'] = pd.Timestamp(doc['time']).to_pydatetime()
            documents.append(doc)
        
        # Batch insert
        if documents:
            result = collection.insert_many(documents)
            self.stats['records_written'] = len(result.inserted_ids)
        
        client.close()
        
        logger.info(f"  Wrote {self.stats['records_written']:,} documents")
        return True
    
    def _write_to_csv(self, data: pd.DataFrame, config: Dict) -> bool:
        """Export to CSV file"""
        
        logger.info("Writing to CSV...")
        
        filename = config.get('filename', f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filepath = self.migration_dir / filename
        
        data.to_csv(filepath, index=False)
        self.stats['records_written'] = len(data)
        
        logger.info(f"  Wrote {self.stats['records_written']:,} records to {filepath}")
        return True
    
    def _write_to_parquet(self, data: pd.DataFrame, config: Dict) -> bool:
        """Export to Parquet file"""
        
        logger.info("Writing to Parquet...")
        
        filename = config.get('filename', f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
        filepath = self.migration_dir / filename
        
        data.to_parquet(filepath, index=False, compression='snappy')
        self.stats['records_written'] = len(data)
        
        logger.info(f"  Wrote {self.stats['records_written']:,} records to {filepath}")
        return True
    
    def _save_migration_report(self, config: MigrationConfig):
        """Save migration report"""
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'source_bucket': config.source_bucket,
            'target_type': config.target_type.value,
            'target_config': {k: v for k, v in config.target_config.items() if k != 'password'},
            'time_range': config.time_range,
            'records_read': self.stats['records_read'],
            'records_written': self.stats['records_written'],
            'errors': self.stats['errors'],
            'duration_seconds': duration,
            'rate': self.stats['records_written'] / duration if duration > 0 else 0
        }
        
        report_file = self.migration_dir / 'migration_history.json'
        
        # Load existing history
        history = []
        if report_file.exists():
            with open(report_file, 'r') as f:
                history = json.load(f)
        
        # Add new report
        history.append(report)
        
        # Save updated history
        with open(report_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\n📊 Migration Report:")
        logger.info(f"   Source: {config.source_bucket}")
        logger.info(f"   Target: {config.target_type.value}")
        logger.info(f"   Records: {self.stats['records_read']:,} → {self.stats['records_written']:,}")
        logger.info(f"   Duration: {duration:.2f} seconds")
        logger.info(f"   Rate: {report['rate']:.0f} records/second")


class MigrationWizard:
    """Interactive migration wizard"""
    
    def __init__(self):
        self.migrator = DataMigrator()
    
    def run(self):
        """Run interactive migration wizard"""
        
        print("\n🔄 InfluxDB Migration Wizard")
        print("=" * 60)
        
        # Select source bucket
        print("\n1. Select source bucket:")
        buckets = self._list_buckets()
        for i, bucket in enumerate(buckets, 1):
            print(f"   {i}. {bucket}")
        
        bucket_idx = int(input("\nEnter bucket number: ")) - 1
        source_bucket = buckets[bucket_idx]
        
        # Select target database
        print("\n2. Select target database:")
        targets = list(TargetDatabase)
        for i, target in enumerate(targets, 1):
            print(f"   {i}. {target.value}")
        
        target_idx = int(input("\nEnter target number: ")) - 1
        target_type = targets[target_idx]
        
        # Get target configuration
        target_config = self._get_target_config(target_type)
        
        # Time range
        print("\n3. Time range (default: last 30 days)")
        custom_range = input("   Enter custom range (e.g., -7d) or press Enter for default: ")
        time_range = (custom_range, "now()") if custom_range else ("-30d", "now()")
        
        # Create migration config
        config = MigrationConfig(
            source_bucket=source_bucket,
            target_type=target_type,
            target_config=target_config,
            time_range=time_range
        )
        
        # Confirm
        print(f"\n📋 Migration Summary:")
        print(f"   Source: {source_bucket}")
        print(f"   Target: {target_type.value}")
        print(f"   Time Range: {time_range[0]} to {time_range[1]}")
        
        confirm = input("\nProceed with migration? (yes/no): ")
        
        if confirm.lower() == 'yes':
            success = self.migrator.migrate(config)
            if success:
                print("\n✅ Migration completed successfully!")
            else:
                print("\n❌ Migration failed. Check logs for details.")
        else:
            print("\nMigration cancelled.")
    
    def _list_buckets(self) -> List[str]:
        """List available buckets"""
        buckets_api = self.migrator.client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        return [b.name for b in buckets if not b.name.startswith('_')]
    
    def _get_target_config(self, target_type: TargetDatabase) -> Dict:
        """Get configuration for target database"""
        
        config = {}
        
        if target_type == TargetDatabase.INFLUXDB:
            config['url'] = input("   InfluxDB URL (default: http://localhost:8086): ") or "http://localhost:8086"
            config['token'] = input("   Token: ")
            config['org'] = input("   Organization (default: agent-system): ") or "agent-system"
            config['bucket'] = input("   Bucket name: ")
            
        elif target_type == TargetDatabase.POSTGRESQL:
            config['host'] = input("   Host (default: localhost): ") or "localhost"
            config['port'] = int(input("   Port (default: 5432): ") or "5432")
            config['database'] = input("   Database: ")
            config['user'] = input("   User: ")
            config['password'] = input("   Password: ")
            config['table'] = input("   Table name (default: metrics): ") or "metrics"
            
        elif target_type == TargetDatabase.SQLITE:
            config['path'] = input("   Database path (default: metrics.db): ") or "metrics.db"
            config['table'] = input("   Table name (default: metrics): ") or "metrics"
            
        elif target_type == TargetDatabase.MONGODB:
            config['connection_string'] = input("   Connection string: ")
            config['database'] = input("   Database name: ")
            config['collection'] = input("   Collection name: ")
            
        elif target_type in [TargetDatabase.CSV, TargetDatabase.PARQUET]:
            config['filename'] = input("   Filename (or press Enter for auto): ") or None
        
        return config


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InfluxDB Data Migration Tools")
    parser.add_argument("action", 
                        choices=["wizard", "migrate", "history"],
                        help="Action to perform")
    parser.add_argument("--source", help="Source bucket")
    parser.add_argument("--target", help="Target type (influxdb, postgresql, sqlite, mongodb, csv, parquet)")
    parser.add_argument("--config", help="Target configuration as JSON")
    
    args = parser.parse_args()
    
    if args.action == "wizard":
        wizard = MigrationWizard()
        wizard.run()
        
    elif args.action == "migrate":
        if not all([args.source, args.target, args.config]):
            print("Error: --source, --target, and --config are required for migrate")
            sys.exit(1)
        
        migrator = DataMigrator()
        config = MigrationConfig(
            source_bucket=args.source,
            target_type=TargetDatabase(args.target),
            target_config=json.loads(args.config)
        )
        
        success = migrator.migrate(config)
        sys.exit(0 if success else 1)
        
    elif args.action == "history":
        history_file = Path('migrations/migration_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            print("\n📋 Migration History:")
            print("-" * 60)
            for entry in history[-10:]:  # Show last 10
                print(f"\n{entry['timestamp']}")
                print(f"  {entry['source_bucket']} → {entry['target_type']}")
                print(f"  Records: {entry['records_read']:,} → {entry['records_written']:,}")
                print(f"  Duration: {entry['duration_seconds']:.2f}s")
        else:
            print("No migration history found")


if __name__ == "__main__":
    # If no arguments, run wizard
    if len(sys.argv) == 1:
        print("🔄 InfluxDB Data Migration Tools")
        print("=" * 60)
        print("\nOptions:")
        print("  1. Run Migration Wizard")
        print("  2. View Migration History")
        print("  3. Exit")
        
        choice = input("\nSelect option (1-3): ")
        
        if choice == '1':
            wizard = MigrationWizard()
            wizard.run()
        elif choice == '2':
            main_args = ['', 'history']
            sys.argv = main_args
            main()
        else:
            print("Goodbye!")
    else:
        main()