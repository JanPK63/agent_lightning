#!/usr/bin/env python3
"""
InfluxDB Data Export System
Export data from InfluxDB in multiple formats (CSV, JSON, Parquet, Excel)
"""

import os
import sys
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
import argparse

# Load environment
load_dotenv('.env.influxdb')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InfluxDBDataExporter:
    """Export data from InfluxDB in various formats"""
    
    def __init__(self):
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN', 'agent-system-token-supersecret-12345678')
        self.org = os.getenv('INFLUXDB_ORG', 'agent-system')
        
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        
        self.export_dir = Path('exports')
        self.export_dir.mkdir(exist_ok=True)
        
        # Supported export formats
        self.formats = ['csv', 'json', 'parquet', 'excel', 'html', 'sql']
    
    def export_bucket(self, 
                      bucket_name: str,
                      format: str = 'csv',
                      start_time: str = '-30d',
                      end_time: str = 'now()',
                      measurement: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> Optional[Path]:
        """Export data from a bucket in specified format"""
        
        if format not in self.formats:
            logger.error(f"Unsupported format: {format}. Supported: {', '.join(self.formats)}")
            return None
        
        logger.info(f"Exporting {bucket_name} to {format.upper()}...")
        
        # Build query
        query = self._build_query(bucket_name, start_time, end_time, measurement, tags)
        
        try:
            # Query data
            query_api = self.client.query_api()
            tables = query_api.query(query, org=self.org)
            
            # Convert to pandas DataFrame
            df = self._tables_to_dataframe(tables)
            
            if df.empty:
                logger.warning(f"No data found for export from {bucket_name}")
                return None
            
            # Export based on format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{bucket_name}_{timestamp}.{format}"
            filepath = self.export_dir / filename
            
            if format == 'csv':
                filepath = self._export_csv(df, filepath)
            elif format == 'json':
                filepath = self._export_json(df, filepath)
            elif format == 'parquet':
                filepath = self._export_parquet(df, filepath)
            elif format == 'excel':
                filepath = self._export_excel(df, filepath, bucket_name)
            elif format == 'html':
                filepath = self._export_html(df, filepath, bucket_name)
            elif format == 'sql':
                filepath = self._export_sql(df, filepath, bucket_name)
            
            if filepath and filepath.exists():
                file_size = filepath.stat().st_size
                logger.info(f"✅ Exported to: {filepath}")
                logger.info(f"   Records: {len(df):,}")
                logger.info(f"   Size: {file_size:,} bytes")
                
                # Save export metadata
                self._save_export_metadata(filepath, bucket_name, format, len(df), file_size)
                
                return filepath
            else:
                logger.error(f"Export failed for {bucket_name}")
                return None
                
        except Exception as e:
            logger.error(f"Export error: {e}")
            return None
    
    def _build_query(self, bucket_name: str, start_time: str, end_time: str,
                     measurement: Optional[str] = None, 
                     tags: Optional[Dict[str, str]] = None) -> str:
        """Build Flux query for data export"""
        
        query = f'''
from(bucket: "{bucket_name}")
    |> range(start: {start_time}, stop: {end_time})'''
        
        if measurement:
            query += f'''
    |> filter(fn: (r) => r["_measurement"] == "{measurement}")'''
        
        if tags:
            for tag_key, tag_value in tags.items():
                query += f'''
    |> filter(fn: (r) => r["{tag_key}"] == "{tag_value}")'''
        
        return query
    
    def _tables_to_dataframe(self, tables) -> pd.DataFrame:
        """Convert InfluxDB tables to pandas DataFrame"""
        
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
                for tag in record.values:
                    if not tag.startswith('_') and tag not in ['time', 'measurement', 'field', 'value']:
                        row[tag] = record.values[tag]
                records.append(row)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Convert time to datetime if present
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
        
        return df
    
    def _export_csv(self, df: pd.DataFrame, filepath: Path) -> Path:
        """Export DataFrame to CSV"""
        df.to_csv(filepath, index=False)
        return filepath
    
    def _export_json(self, df: pd.DataFrame, filepath: Path) -> Path:
        """Export DataFrame to JSON"""
        # Convert datetime to string for JSON serialization
        df_copy = df.copy()
        if 'time' in df_copy.columns:
            df_copy['time'] = df_copy['time'].astype(str)
        
        df_copy.to_json(filepath, orient='records', indent=2)
        return filepath
    
    def _export_parquet(self, df: pd.DataFrame, filepath: Path) -> Path:
        """Export DataFrame to Parquet"""
        df.to_parquet(filepath, index=False, compression='snappy')
        return filepath
    
    def _export_excel(self, df: pd.DataFrame, filepath: Path, sheet_name: str) -> Path:
        """Export DataFrame to Excel with formatting"""
        filepath = filepath.with_suffix('.xlsx')
        
        # Remove timezone from datetime columns for Excel compatibility
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.tz_localize(None)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_copy.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            
            # Get the worksheet
            worksheet = writer.sheets[sheet_name[:31]]
            
            # Auto-adjust column widths
            for column in df:
                column_length = max(df[column].astype(str).map(len).max(), len(str(column)))
                col_idx = df.columns.get_loc(column) + 1
                worksheet.column_dimensions[chr(64 + col_idx)].width = min(column_length + 2, 50)
        
        return filepath
    
    def _export_html(self, df: pd.DataFrame, filepath: Path, title: str) -> Path:
        """Export DataFrame to HTML with styling"""
        filepath = filepath.with_suffix('.html')
        
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>{title} Export</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metadata {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{title} Data Export</h1>
    <div class="metadata">
        <p><strong>Exported:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Records:</strong> {len(df):,}</p>
        <p><strong>Time Range:</strong> {df['time'].min() if 'time' in df.columns else 'N/A'} to {df['time'].max() if 'time' in df.columns else 'N/A'}</p>
    </div>
    {df.to_html(index=False, classes='data-table')}
</body>
</html>
'''
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        return filepath
    
    def _export_sql(self, df: pd.DataFrame, filepath: Path, table_name: str) -> Path:
        """Export DataFrame to SQL INSERT statements"""
        filepath = filepath.with_suffix('.sql')
        
        # Clean table name for SQL
        table_name = table_name.replace('-', '_')
        
        with open(filepath, 'w') as f:
            # Create table statement
            f.write(f"-- SQL export for {table_name}\n")
            f.write(f"-- Generated: {datetime.now()}\n\n")
            
            f.write(f"CREATE TABLE IF NOT EXISTS {table_name} (\n")
            
            # Define columns based on DataFrame
            columns = []
            for col in df.columns:
                if col == 'time':
                    columns.append(f"    {col} TIMESTAMP")
                elif df[col].dtype == 'float64':
                    columns.append(f"    {col} FLOAT")
                elif df[col].dtype == 'int64':
                    columns.append(f"    {col} INTEGER")
                else:
                    columns.append(f"    {col} TEXT")
            
            f.write(',\n'.join(columns))
            f.write("\n);\n\n")
            
            # Insert statements
            for _, row in df.iterrows():
                values = []
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        values.append('NULL')
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    else:
                        # Escape single quotes
                        val = str(val).replace("'", "''")
                        values.append(f"'{val}'")
                
                f.write(f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({', '.join(values)});\n")
        
        return filepath
    
    def _save_export_metadata(self, filepath: Path, bucket: str, format: str, 
                              records: int, size: int):
        """Save metadata about the export"""
        metadata_file = self.export_dir / 'export_history.json'
        
        # Load existing history
        history = []
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                history = json.load(f)
        
        # Add new export
        history.append({
            'timestamp': datetime.now().isoformat(),
            'file': str(filepath.name),
            'bucket': bucket,
            'format': format,
            'records': records,
            'size': size,
            'path': str(filepath)
        })
        
        # Keep last 100 exports
        history = history[-100:]
        
        # Save updated history
        with open(metadata_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def export_all_buckets(self, format: str = 'csv', start_time: str = '-30d'):
        """Export all buckets"""
        logger.info(f"Exporting all buckets to {format.upper()}...")
        logger.info("=" * 60)
        
        # Get all buckets
        buckets_api = self.client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        
        exported = []
        failed = []
        
        for bucket in buckets:
            if bucket.name.startswith('_'):
                continue  # Skip system buckets
            
            filepath = self.export_bucket(bucket.name, format, start_time)
            if filepath:
                exported.append(bucket.name)
            else:
                failed.append(bucket.name)
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"Export Summary:")
        logger.info(f"  Successful: {len(exported)}")
        logger.info(f"  Failed: {len(failed)}")
        
        if exported:
            logger.info(f"  Exported buckets: {', '.join(exported)}")
        if failed:
            logger.info(f"  Failed buckets: {', '.join(failed)}")
    
    def list_exports(self, limit: int = 20):
        """List recent exports"""
        metadata_file = self.export_dir / 'export_history.json'
        
        if not metadata_file.exists():
            logger.info("No export history found")
            return
        
        with open(metadata_file, 'r') as f:
            history = json.load(f)
        
        if not history:
            logger.info("No exports found")
            return
        
        logger.info("\nRecent Exports:")
        logger.info("-" * 70)
        
        for export in reversed(history[-limit:]):
            size_mb = export['size'] / (1024 * 1024)
            logger.info(f"📁 {export['file']}")
            logger.info(f"   Bucket: {export['bucket']}")
            logger.info(f"   Format: {export['format'].upper()}")
            logger.info(f"   Records: {export['records']:,}")
            logger.info(f"   Size: {size_mb:.2f} MB")
            logger.info(f"   Date: {export['timestamp']}")
            logger.info("")
    
    def schedule_export(self, bucket: str, format: str = 'csv', 
                        interval: str = 'daily', time_of_day: str = '02:00'):
        """Schedule regular exports (creates cron entry)"""
        
        # Create export script
        script_path = self.export_dir / f"scheduled_export_{bucket}.sh"
        
        with open(script_path, 'w') as f:
            f.write(f'''#!/bin/bash
# Scheduled export for {bucket}
cd {os.getcwd()}
python influxdb_data_export.py export --bucket {bucket} --format {format} >> exports/export.log 2>&1
''')
        
        script_path.chmod(0o755)
        
        # Cron schedule based on interval
        if interval == 'hourly':
            cron_schedule = '0 * * * *'
        elif interval == 'daily':
            hour = time_of_day.split(':')[0]
            cron_schedule = f'0 {hour} * * *'
        elif interval == 'weekly':
            hour = time_of_day.split(':')[0]
            cron_schedule = f'0 {hour} * * 0'
        elif interval == 'monthly':
            hour = time_of_day.split(':')[0]
            cron_schedule = f'0 {hour} 1 * *'
        else:
            logger.error(f"Invalid interval: {interval}")
            return
        
        cron_entry = f"{cron_schedule} {script_path}"
        
        logger.info(f"✅ Export scheduled for {bucket}")
        logger.info(f"   Format: {format}")
        logger.info(f"   Interval: {interval}")
        logger.info(f"   Cron: {cron_entry}")
        logger.info(f"\nTo enable, add to crontab:")
        logger.info(f"   crontab -e")
        logger.info(f"   {cron_entry}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="InfluxDB Data Export System")
    parser.add_argument("action", 
                        choices=["export", "export-all", "list", "schedule"],
                        help="Action to perform")
    parser.add_argument("--bucket", help="Bucket name to export")
    parser.add_argument("--format", default="csv",
                        choices=["csv", "json", "parquet", "excel", "html", "sql"],
                        help="Export format")
    parser.add_argument("--start", default="-30d", help="Start time (e.g., -30d, -1h)")
    parser.add_argument("--end", default="now()", help="End time")
    parser.add_argument("--measurement", help="Specific measurement to export")
    parser.add_argument("--interval", default="daily",
                        choices=["hourly", "daily", "weekly", "monthly"],
                        help="Schedule interval")
    
    args = parser.parse_args()
    
    exporter = InfluxDBDataExporter()
    
    print("📤 InfluxDB Data Export System")
    print("=" * 60)
    
    if args.action == "export":
        if not args.bucket:
            print("❌ Please specify --bucket")
            sys.exit(1)
        
        exporter.export_bucket(
            args.bucket,
            format=args.format,
            start_time=args.start,
            end_time=args.end,
            measurement=args.measurement
        )
    
    elif args.action == "export-all":
        exporter.export_all_buckets(
            format=args.format,
            start_time=args.start
        )
    
    elif args.action == "list":
        exporter.list_exports()
    
    elif args.action == "schedule":
        if not args.bucket:
            print("❌ Please specify --bucket")
            sys.exit(1)
        
        exporter.schedule_export(
            args.bucket,
            format=args.format,
            interval=args.interval
        )


if __name__ == "__main__":
    # If no arguments, export all buckets to CSV
    if len(sys.argv) == 1:
        exporter = InfluxDBDataExporter()
        print("📤 InfluxDB Data Export System")
        print("=" * 60)
        exporter.export_all_buckets(format='csv')
        print("\nUsage:")
        print("  python influxdb_data_export.py export --bucket <name> --format <type>")
        print("  python influxdb_data_export.py export-all --format <type>")
        print("  python influxdb_data_export.py list")
        print("  python influxdb_data_export.py schedule --bucket <name> --interval daily")
    else:
        main()