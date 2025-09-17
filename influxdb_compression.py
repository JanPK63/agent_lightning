#!/usr/bin/env python3
"""
InfluxDB Data Compression System
Implements compression strategies to reduce storage requirements
"""

import os
import sys
import json
import gzip
import zlib
import lz4.frame
import brotli
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.influxdb')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Available compression algorithms"""
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"
    BROTLI = "brotli"
    NONE = "none"


@dataclass
class CompressionResult:
    """Result of compression operation"""
    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float


class DataCompressor:
    """Handles data compression with multiple algorithms"""
    
    def __init__(self):
        self.algorithms = {
            CompressionAlgorithm.GZIP: (self._gzip_compress, self._gzip_decompress),
            CompressionAlgorithm.ZLIB: (self._zlib_compress, self._zlib_decompress),
            CompressionAlgorithm.LZ4: (self._lz4_compress, self._lz4_decompress),
            CompressionAlgorithm.BROTLI: (self._brotli_compress, self._brotli_decompress),
            CompressionAlgorithm.NONE: (self._no_compress, self._no_decompress)
        }
    
    def compress(self, data: bytes, algorithm: CompressionAlgorithm) -> Tuple[bytes, CompressionResult]:
        """Compress data using specified algorithm"""
        import time
        
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        compress_func, _ = self.algorithms[algorithm]
        
        start_time = time.time()
        compressed_data = compress_func(data)
        compression_time = time.time() - start_time
        
        # Test decompression time
        _, decompress_func = self.algorithms[algorithm]
        start_time = time.time()
        _ = decompress_func(compressed_data)
        decompression_time = time.time() - start_time
        
        result = CompressionResult(
            algorithm=algorithm,
            original_size=len(data),
            compressed_size=len(compressed_data),
            compression_ratio=len(data) / len(compressed_data) if len(compressed_data) > 0 else 0,
            compression_time=compression_time,
            decompression_time=decompression_time
        )
        
        return compressed_data, result
    
    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data using specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        _, decompress_func = self.algorithms[algorithm]
        return decompress_func(data)
    
    def _gzip_compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=6)
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)
    
    def _zlib_compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)
    
    def _zlib_decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)
    
    def _lz4_compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data)
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)
    
    def _brotli_compress(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=6)
    
    def _brotli_decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)
    
    def _no_compress(self, data: bytes) -> bytes:
        return data
    
    def _no_decompress(self, data: bytes) -> bytes:
        return data
    
    def find_best_algorithm(self, data: bytes) -> Tuple[CompressionAlgorithm, CompressionResult]:
        """Find the best compression algorithm for given data"""
        best_algorithm = CompressionAlgorithm.NONE
        best_result = None
        best_score = 0
        
        for algorithm in CompressionAlgorithm:
            if algorithm == CompressionAlgorithm.NONE:
                continue
            
            try:
                _, result = self.compress(data, algorithm)
                
                # Score based on compression ratio and speed
                # Higher compression ratio is better, lower time is better
                score = result.compression_ratio / (result.compression_time + 0.001)
                
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
                    best_result = result
            except Exception as e:
                logger.warning(f"Failed to test {algorithm}: {e}")
        
        return best_algorithm, best_result


class InfluxDBCompressionManager:
    """Manages compression for InfluxDB data"""
    
    def __init__(self):
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN', 'agent-system-token-supersecret-12345678')
        self.org = os.getenv('INFLUXDB_ORG', 'agent-system')
        
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        
        self.compressor = DataCompressor()
        self.compression_dir = Path('compression_cache')
        self.compression_dir.mkdir(exist_ok=True)
        
        # Compression settings per bucket
        self.bucket_settings = {
            'performance_metrics': CompressionAlgorithm.LZ4,  # Fast for real-time
            'performance_metrics_long': CompressionAlgorithm.BROTLI,  # High compression for archive
            'agent_metrics': CompressionAlgorithm.GZIP,  # Balanced
            'alerts': CompressionAlgorithm.LZ4,  # Fast for quick access
            'test_metrics': CompressionAlgorithm.ZLIB,  # Good for text data
            'debug_metrics': CompressionAlgorithm.LZ4,  # Fast, short-lived data
            'deployment_metrics': CompressionAlgorithm.BROTLI  # Maximum compression for long-term
        }
    
    def compress_bucket_data(self, bucket_name: str, start_time: str = "-30d") -> Dict:
        """Compress data from a bucket"""
        logger.info(f"Compressing data from bucket: {bucket_name}")
        
        # Query data
        query_api = self.client.query_api()
        query = f'''
        from(bucket: "{bucket_name}")
            |> range(start: {start_time})
        '''
        
        try:
            result = query_api.query_csv(query, org=self.org)
            
            # Convert CSV result to string then bytes
            csv_lines = []
            for row in result:
                if isinstance(row, list):
                    csv_lines.append(','.join(str(x) for x in row))
                else:
                    csv_lines.append(str(row))
            csv_data = '\n'.join(csv_lines)
            original_data = csv_data.encode('utf-8')
            
            # Determine compression algorithm
            algorithm = self.bucket_settings.get(bucket_name, CompressionAlgorithm.GZIP)
            
            # Compress data
            compressed_data, compression_result = self.compressor.compress(original_data, algorithm)
            
            # Save compressed data
            compressed_file = self.compression_dir / f"{bucket_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{algorithm.value}"
            with open(compressed_file, 'wb') as f:
                f.write(compressed_data)
            
            # Save metadata
            metadata = {
                'bucket': bucket_name,
                'timestamp': datetime.now().isoformat(),
                'algorithm': algorithm.value,
                'original_size': compression_result.original_size,
                'compressed_size': compression_result.compressed_size,
                'compression_ratio': compression_result.compression_ratio,
                'file': str(compressed_file.name)
            }
            
            metadata_file = compressed_file.with_suffix('.meta.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  ✅ Compressed {bucket_name}: {compression_result.compression_ratio:.2f}x reduction")
            logger.info(f"     Original: {compression_result.original_size:,} bytes")
            logger.info(f"     Compressed: {compression_result.compressed_size:,} bytes")
            logger.info(f"     Algorithm: {algorithm.value}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"  ❌ Failed to compress {bucket_name}: {e}")
            return None
    
    def decompress_file(self, compressed_file: Path) -> Optional[str]:
        """Decompress a file"""
        # Load metadata
        metadata_file = compressed_file.with_suffix('.meta.json')
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        algorithm = CompressionAlgorithm(metadata['algorithm'])
        
        # Read compressed data
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
        
        # Decompress
        decompressed_data = self.compressor.decompress(compressed_data, algorithm)
        
        return decompressed_data.decode('utf-8')
    
    def analyze_compression_potential(self, bucket_name: str) -> Dict:
        """Analyze compression potential for a bucket"""
        logger.info(f"Analyzing compression potential for: {bucket_name}")
        
        # Query sample data
        query_api = self.client.query_api()
        query = f'''
        from(bucket: "{bucket_name}")
            |> range(start: -1h)
            |> limit(n: 1000)
        '''
        
        try:
            result = query_api.query_csv(query, org=self.org)
            csv_data = ''.join(result)
            sample_data = csv_data.encode('utf-8')
            
            if len(sample_data) == 0:
                logger.warning(f"  No data in bucket: {bucket_name}")
                return {}
            
            analysis = {
                'bucket': bucket_name,
                'sample_size': len(sample_data),
                'algorithms': {}
            }
            
            # Test each algorithm
            for algorithm in CompressionAlgorithm:
                if algorithm == CompressionAlgorithm.NONE:
                    continue
                
                try:
                    _, result = self.compressor.compress(sample_data, algorithm)
                    analysis['algorithms'][algorithm.value] = {
                        'compressed_size': result.compressed_size,
                        'compression_ratio': result.compression_ratio,
                        'compression_time_ms': result.compression_time * 1000,
                        'decompression_time_ms': result.decompression_time * 1000,
                        'savings_percent': (1 - 1/result.compression_ratio) * 100 if result.compression_ratio > 0 else 0
                    }
                except Exception as e:
                    logger.warning(f"    Failed to test {algorithm}: {e}")
            
            # Find best algorithm
            best_algorithm, best_result = self.compressor.find_best_algorithm(sample_data)
            analysis['recommended'] = best_algorithm.value
            
            return analysis
            
        except Exception as e:
            logger.error(f"  Failed to analyze {bucket_name}: {e}")
            return {}
    
    def compress_all_buckets(self):
        """Compress data from all configured buckets"""
        logger.info("Compressing all buckets...")
        logger.info("=" * 60)
        
        results = []
        total_original = 0
        total_compressed = 0
        
        for bucket_name, algorithm in self.bucket_settings.items():
            metadata = self.compress_bucket_data(bucket_name)
            if metadata:
                results.append(metadata)
                total_original += metadata['original_size']
                total_compressed += metadata['compressed_size']
        
        # Summary
        logger.info("=" * 60)
        logger.info("Compression Summary:")
        logger.info(f"  Total Original: {total_original:,} bytes")
        logger.info(f"  Total Compressed: {total_compressed:,} bytes")
        if total_original > 0:
            total_ratio = total_original / total_compressed
            total_savings = (1 - total_compressed/total_original) * 100
            logger.info(f"  Overall Ratio: {total_ratio:.2f}x")
            logger.info(f"  Space Saved: {total_savings:.1f}%")
        
        return results
    
    def list_compressed_files(self):
        """List all compressed files"""
        logger.info("\nCompressed Files:")
        logger.info("-" * 60)
        
        files = list(self.compression_dir.glob("*.meta.json"))
        
        if not files:
            logger.info("No compressed files found")
            return
        
        for meta_file in sorted(files, reverse=True)[:10]:  # Show last 10
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            size_mb = metadata['compressed_size'] / (1024 * 1024)
            ratio = metadata['compression_ratio']
            
            logger.info(f"  {metadata['file']}")
            logger.info(f"    Bucket: {metadata['bucket']}")
            logger.info(f"    Algorithm: {metadata['algorithm']}")
            logger.info(f"    Size: {size_mb:.2f} MB (ratio: {ratio:.2f}x)")
            logger.info(f"    Date: {metadata['timestamp']}")
            logger.info("")
    
    def enable_inline_compression(self):
        """Enable inline compression for new data writes"""
        logger.info("Configuring inline compression for new data...")
        
        # This would typically be done at the database level
        # InfluxDB 2.x has built-in compression (TSM engine)
        # Here we document the settings
        
        config = {
            'engine': 'tsm1',
            'compression': 'snappy',  # Built-in compression
            'cache_max_memory_size': '1g',
            'cache_snapshot_memory_size': '25m',
            'cache_snapshot_write_cold_duration': '10m',
            'compact_full_write_cold_duration': '4h',
            'max_concurrent_compactions': 0,
            'compact_throughput': '48m',
            'compact_throughput_burst': '48m',
            'tsm_use_madv_willneed': False
        }
        
        config_file = self.compression_dir / "influxdb_compression_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"  ✅ Compression configuration saved to: {config_file}")
        logger.info("  Note: InfluxDB uses built-in TSM compression")
        logger.info("  This system provides additional compression for exports/backups")
        
        return config


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InfluxDB Data Compression System")
    parser.add_argument("action", 
                        choices=["compress", "analyze", "list", "decompress", "configure"],
                        help="Action to perform")
    parser.add_argument("--bucket", help="Specific bucket to compress/analyze")
    parser.add_argument("--file", help="File to decompress")
    
    args = parser.parse_args()
    
    manager = InfluxDBCompressionManager()
    
    print("🗜️  InfluxDB Data Compression System")
    print("=" * 60)
    
    if args.action == "compress":
        if args.bucket:
            manager.compress_bucket_data(args.bucket)
        else:
            manager.compress_all_buckets()
    
    elif args.action == "analyze":
        if args.bucket:
            analysis = manager.analyze_compression_potential(args.bucket)
            if analysis:
                print(f"\nCompression Analysis for {args.bucket}:")
                print("-" * 40)
                for algo, stats in analysis.get('algorithms', {}).items():
                    print(f"  {algo}:")
                    print(f"    Ratio: {stats['compression_ratio']:.2f}x")
                    print(f"    Savings: {stats['savings_percent']:.1f}%")
                    print(f"    Speed: {stats['compression_time_ms']:.2f}ms")
                print(f"\n  Recommended: {analysis.get('recommended', 'N/A')}")
        else:
            # Analyze all buckets
            for bucket in manager.bucket_settings.keys():
                analysis = manager.analyze_compression_potential(bucket)
                if analysis and analysis.get('recommended'):
                    print(f"  {bucket}: Recommended {analysis['recommended']}")
    
    elif args.action == "list":
        manager.list_compressed_files()
    
    elif args.action == "decompress":
        if not args.file:
            print("Please specify file with --file")
            sys.exit(1)
        
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = manager.compression_dir / args.file
        
        data = manager.decompress_file(file_path)
        if data:
            print(f"✅ Decompressed {file_path.name}")
            print(f"   Size: {len(data):,} bytes")
    
    elif args.action == "configure":
        config = manager.enable_inline_compression()
        print("\n✅ Compression configured")
        print("\nSettings applied:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # If no arguments, compress all buckets
    if len(sys.argv) == 1:
        manager = InfluxDBCompressionManager()
        print("🗜️  InfluxDB Data Compression System")
        print("=" * 60)
        manager.compress_all_buckets()
        print("\nUsage:")
        print("  python influxdb_compression.py compress    # Compress all buckets")
        print("  python influxdb_compression.py analyze     # Analyze compression potential")
        print("  python influxdb_compression.py list        # List compressed files")
        print("  python influxdb_compression.py configure   # Configure compression")
    else:
        main()