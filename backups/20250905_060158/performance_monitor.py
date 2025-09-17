#!/usr/bin/env python3
"""
Performance Monitor
Real-time system performance monitoring with metrics collection and analysis
"""

import os
import sys
import time
import psutil
import threading
import queue
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import platform
import subprocess
import socket

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MetricType(Enum):
    """Types of performance metrics"""
    CPU_USAGE = "cpu_usage"
    CPU_FREQUENCY = "cpu_frequency"
    CPU_TEMPERATURE = "cpu_temperature"
    MEMORY_USAGE = "memory_usage"
    MEMORY_SWAP = "memory_swap"
    DISK_IO = "disk_io"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    NETWORK_CONNECTIONS = "network_connections"
    PROCESS_COUNT = "process_count"
    THREAD_COUNT = "thread_count"
    LOAD_AVERAGE = "load_average"
    UPTIME = "uptime"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricSample:
    """Single metric sample"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata
        }


@dataclass
class MetricStats:
    """Statistics for a metric over time"""
    metric_type: MetricType
    current: float = 0.0
    average: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    std_dev: float = 0.0
    percentile_50: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    sample_count: int = 0
    window_seconds: int = 60
    
    def calculate(self, samples: List[float]):
        """Calculate statistics from samples"""
        if not samples:
            return
        
        self.sample_count = len(samples)
        self.current = samples[-1] if samples else 0
        self.average = statistics.mean(samples)
        self.minimum = min(samples)
        self.maximum = max(samples)
        
        if len(samples) > 1:
            self.std_dev = statistics.stdev(samples)
        
        if len(samples) >= 2:
            sorted_samples = sorted(samples)
            self.percentile_50 = sorted_samples[len(sorted_samples) // 2]
            self.percentile_95 = sorted_samples[int(len(sorted_samples) * 0.95)]
            self.percentile_99 = sorted_samples[int(len(sorted_samples) * 0.99)]


@dataclass
class Alert:
    """Performance alert"""
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    duration: Optional[float] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "metric_type": self.metric_type.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "duration": self.duration,
            "resolved": self.resolved
        }


@dataclass
class PerformanceThresholds:
    """Configurable performance thresholds"""
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    network_warning_mbps: float = 800.0
    network_critical_mbps: float = 950.0
    load_warning: float = 2.0
    load_critical: float = 4.0
    process_warning: int = 500
    process_critical: int = 1000
    response_time_warning_ms: float = 1000.0
    response_time_critical_ms: float = 3000.0


class MetricCollector:
    """Base class for metric collectors"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.samples = deque(maxlen=3600)  # Keep 1 hour of samples at 1s intervals
        self.last_collection = None
        
    def start(self):
        """Start collecting metrics"""
        self.running = True
        thread = threading.Thread(target=self._collection_loop, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop collecting metrics"""
        self.running = False
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                sample = self.collect()
                if sample:
                    self.samples.append(sample)
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def collect(self) -> Optional[MetricSample]:
        """Collect a single metric sample (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_recent_samples(self, seconds: int = 60) -> List[MetricSample]:
        """Get samples from the last N seconds"""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return [s for s in self.samples if s.timestamp >= cutoff]
    
    def get_stats(self, window_seconds: int = 60) -> MetricStats:
        """Get statistics for recent samples"""
        recent = self.get_recent_samples(window_seconds)
        stats = MetricStats(
            metric_type=self.get_metric_type(),
            window_seconds=window_seconds
        )
        
        if recent:
            values = [s.value for s in recent]
            stats.calculate(values)
        
        return stats
    
    def get_metric_type(self) -> MetricType:
        """Get the metric type (to be implemented by subclasses)"""
        raise NotImplementedError


class CPUCollector(MetricCollector):
    """CPU metrics collector"""
    
    def __init__(self, collection_interval: float = 1.0):
        super().__init__(collection_interval)
        self.cpu_count = psutil.cpu_count()
        self.last_per_cpu = None
    
    def collect(self) -> Optional[MetricSample]:
        """Collect CPU usage"""
        try:
            # Get overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get per-CPU usage
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Get CPU frequency if available
            freq = psutil.cpu_freq()
            
            metadata = {
                "cpu_count": self.cpu_count,
                "per_cpu": per_cpu,
            }
            
            if freq:
                metadata["frequency_mhz"] = freq.current
                metadata["frequency_min"] = freq.min
                metadata["frequency_max"] = freq.max
            
            # Get CPU times
            cpu_times = psutil.cpu_times()
            metadata["user_time"] = cpu_times.user
            metadata["system_time"] = cpu_times.system
            metadata["idle_time"] = cpu_times.idle
            
            return MetricSample(
                timestamp=datetime.now(),
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit="%",
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error collecting CPU metrics: {e}")
            return None
    
    def get_metric_type(self) -> MetricType:
        return MetricType.CPU_USAGE
    
    def get_load_average(self) -> Tuple[float, float, float]:
        """Get system load average (1, 5, 15 minutes)"""
        return psutil.getloadavg()


class MemoryCollector(MetricCollector):
    """Memory metrics collector"""
    
    def collect(self) -> Optional[MetricSample]:
        """Collect memory usage"""
        try:
            # Get virtual memory stats
            vm = psutil.virtual_memory()
            
            # Get swap memory stats
            swap = psutil.swap_memory()
            
            metadata = {
                "total_gb": vm.total / (1024**3),
                "available_gb": vm.available / (1024**3),
                "used_gb": vm.used / (1024**3),
                "free_gb": vm.free / (1024**3),
                "cached_gb": getattr(vm, 'cached', 0) / (1024**3),
                "buffers_gb": getattr(vm, 'buffers', 0) / (1024**3),
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent
            }
            
            return MetricSample(
                timestamp=datetime.now(),
                metric_type=MetricType.MEMORY_USAGE,
                value=vm.percent,
                unit="%",
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error collecting memory metrics: {e}")
            return None
    
    def get_metric_type(self) -> MetricType:
        return MetricType.MEMORY_USAGE


class DiskCollector(MetricCollector):
    """Disk I/O and usage metrics collector"""
    
    def __init__(self, collection_interval: float = 5.0):
        super().__init__(collection_interval)
        self.last_io_counters = None
    
    def collect(self) -> Optional[MetricSample]:
        """Collect disk metrics"""
        try:
            # Get disk usage for all partitions
            partitions = psutil.disk_partitions()
            usage_data = {}
            total_percent = 0
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    usage_data[partition.mountpoint] = {
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent
                    }
                    total_percent = max(total_percent, usage.percent)
                except:
                    continue
            
            # Get disk I/O statistics
            io_counters = psutil.disk_io_counters()
            io_data = {}
            
            if io_counters:
                if self.last_io_counters:
                    time_delta = self.collection_interval
                    
                    # Calculate rates
                    read_rate = (io_counters.read_bytes - self.last_io_counters.read_bytes) / time_delta
                    write_rate = (io_counters.write_bytes - self.last_io_counters.write_bytes) / time_delta
                    
                    io_data = {
                        "read_mb_per_sec": read_rate / (1024**2),
                        "write_mb_per_sec": write_rate / (1024**2),
                        "read_count_per_sec": (io_counters.read_count - self.last_io_counters.read_count) / time_delta,
                        "write_count_per_sec": (io_counters.write_count - self.last_io_counters.write_count) / time_delta
                    }
                
                self.last_io_counters = io_counters
            
            metadata = {
                "partitions": usage_data,
                "io": io_data
            }
            
            return MetricSample(
                timestamp=datetime.now(),
                metric_type=MetricType.DISK_USAGE,
                value=total_percent,
                unit="%",
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error collecting disk metrics: {e}")
            return None
    
    def get_metric_type(self) -> MetricType:
        return MetricType.DISK_USAGE


class NetworkCollector(MetricCollector):
    """Network I/O metrics collector"""
    
    def __init__(self, collection_interval: float = 2.0):
        super().__init__(collection_interval)
        self.last_io_counters = None
    
    def collect(self) -> Optional[MetricSample]:
        """Collect network metrics"""
        try:
            # Get network I/O statistics
            io_counters = psutil.net_io_counters()
            
            if not io_counters:
                return None
            
            io_data = {
                "bytes_sent_total": io_counters.bytes_sent,
                "bytes_recv_total": io_counters.bytes_recv,
                "packets_sent_total": io_counters.packets_sent,
                "packets_recv_total": io_counters.packets_recv,
                "errors_in": io_counters.errin,
                "errors_out": io_counters.errout,
                "drops_in": io_counters.dropin,
                "drops_out": io_counters.dropout
            }
            
            throughput = 0
            
            if self.last_io_counters:
                time_delta = self.collection_interval
                
                # Calculate rates
                send_rate = (io_counters.bytes_sent - self.last_io_counters.bytes_sent) / time_delta
                recv_rate = (io_counters.bytes_recv - self.last_io_counters.bytes_recv) / time_delta
                
                io_data["send_mbps"] = (send_rate * 8) / (1024**2)  # Convert to Mbps
                io_data["recv_mbps"] = (recv_rate * 8) / (1024**2)
                io_data["total_mbps"] = io_data["send_mbps"] + io_data["recv_mbps"]
                
                throughput = io_data["total_mbps"]
            
            self.last_io_counters = io_counters
            
            # Get connection count
            connections = psutil.net_connections()
            connection_stats = defaultdict(int)
            for conn in connections:
                connection_stats[conn.status] += 1
            
            io_data["connections"] = dict(connection_stats)
            io_data["total_connections"] = len(connections)
            
            return MetricSample(
                timestamp=datetime.now(),
                metric_type=MetricType.NETWORK_IO,
                value=throughput,
                unit="Mbps",
                metadata=io_data
            )
            
        except Exception as e:
            print(f"Error collecting network metrics: {e}")
            return None
    
    def get_metric_type(self) -> MetricType:
        return MetricType.NETWORK_IO


class ProcessCollector(MetricCollector):
    """Process and thread metrics collector"""
    
    def __init__(self, collection_interval: float = 5.0):
        super().__init__(collection_interval)
        self.top_processes_count = 10
    
    def collect(self) -> Optional[MetricSample]:
        """Collect process metrics"""
        try:
            # Get process count
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            process_count = len(processes)
            
            # Get top processes by CPU and memory
            top_cpu = sorted(processes, key=lambda p: p.info.get('cpu_percent', 0) or 0, reverse=True)[:self.top_processes_count]
            top_memory = sorted(processes, key=lambda p: p.info.get('memory_percent', 0) or 0, reverse=True)[:self.top_processes_count]
            
            # Count threads
            total_threads = sum(p.num_threads() for p in psutil.process_iter() if hasattr(p, 'num_threads'))
            
            metadata = {
                "total_processes": process_count,
                "total_threads": total_threads,
                "top_cpu": [
                    {
                        "pid": p.info['pid'],
                        "name": p.info['name'],
                        "cpu_percent": p.info.get('cpu_percent', 0)
                    } for p in top_cpu
                ],
                "top_memory": [
                    {
                        "pid": p.info['pid'],
                        "name": p.info['name'],
                        "memory_percent": p.info.get('memory_percent', 0)
                    } for p in top_memory
                ]
            }
            
            return MetricSample(
                timestamp=datetime.now(),
                metric_type=MetricType.PROCESS_COUNT,
                value=process_count,
                unit="processes",
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error collecting process metrics: {e}")
            return None
    
    def get_metric_type(self) -> MetricType:
        return MetricType.PROCESS_COUNT


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, thresholds: PerformanceThresholds = None):
        self.thresholds = thresholds or PerformanceThresholds()
        self.collectors = {}
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        self.running = False
        self.alert_thread = None
        
        # Initialize collectors
        self._init_collectors()
    
    def _init_collectors(self):
        """Initialize metric collectors"""
        self.collectors[MetricType.CPU_USAGE] = CPUCollector()
        self.collectors[MetricType.MEMORY_USAGE] = MemoryCollector()
        self.collectors[MetricType.DISK_USAGE] = DiskCollector()
        self.collectors[MetricType.NETWORK_IO] = NetworkCollector()
        self.collectors[MetricType.PROCESS_COUNT] = ProcessCollector()
    
    def start(self):
        """Start performance monitoring"""
        self.running = True
        
        # Start all collectors
        for collector in self.collectors.values():
            collector.start()
        
        # Start alert monitoring
        self.alert_thread = threading.Thread(target=self._alert_monitor, daemon=True)
        self.alert_thread.start()
        
        print("🚀 Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        
        # Stop all collectors
        for collector in self.collectors.values():
            collector.stop()
        
        print("✋ Performance monitoring stopped")
    
    def _alert_monitor(self):
        """Monitor metrics and generate alerts"""
        while self.running:
            try:
                # Check CPU
                cpu_stats = self.collectors[MetricType.CPU_USAGE].get_stats()
                self._check_threshold(
                    cpu_stats.current,
                    self.thresholds.cpu_warning,
                    self.thresholds.cpu_critical,
                    MetricType.CPU_USAGE,
                    "CPU usage"
                )
                
                # Check memory
                mem_stats = self.collectors[MetricType.MEMORY_USAGE].get_stats()
                self._check_threshold(
                    mem_stats.current,
                    self.thresholds.memory_warning,
                    self.thresholds.memory_critical,
                    MetricType.MEMORY_USAGE,
                    "Memory usage"
                )
                
                # Check disk
                disk_stats = self.collectors[MetricType.DISK_USAGE].get_stats()
                self._check_threshold(
                    disk_stats.current,
                    self.thresholds.disk_warning,
                    self.thresholds.disk_critical,
                    MetricType.DISK_USAGE,
                    "Disk usage"
                )
                
                # Check network
                net_stats = self.collectors[MetricType.NETWORK_IO].get_stats()
                if net_stats.current > 0:
                    self._check_threshold(
                        net_stats.current,
                        self.thresholds.network_warning_mbps,
                        self.thresholds.network_critical_mbps,
                        MetricType.NETWORK_IO,
                        "Network throughput"
                    )
                
                # Check process count
                proc_stats = self.collectors[MetricType.PROCESS_COUNT].get_stats()
                self._check_threshold(
                    proc_stats.current,
                    self.thresholds.process_warning,
                    self.thresholds.process_critical,
                    MetricType.PROCESS_COUNT,
                    "Process count"
                )
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error in alert monitor: {e}")
                time.sleep(5)
    
    def _check_threshold(
        self,
        value: float,
        warning_threshold: float,
        critical_threshold: float,
        metric_type: MetricType,
        metric_name: str
    ):
        """Check if a metric exceeds thresholds"""
        if value >= critical_threshold:
            self._create_alert(
                AlertLevel.CRITICAL,
                metric_type,
                f"{metric_name} is critically high: {value:.1f}",
                value,
                critical_threshold
            )
        elif value >= warning_threshold:
            # Check if we already have a warning for this metric
            recent_alerts = [a for a in self.alerts if 
                           a.metric_type == metric_type and 
                           a.level == AlertLevel.WARNING and
                           not a.resolved]
            if not recent_alerts:
                self._create_alert(
                    AlertLevel.WARNING,
                    metric_type,
                    f"{metric_name} is above warning threshold: {value:.1f}",
                    value,
                    warning_threshold
                )
    
    def _create_alert(
        self,
        level: AlertLevel,
        metric_type: MetricType,
        message: str,
        value: float,
        threshold: float
    ):
        """Create and notify alert"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            metric_type=metric_type,
            message=message,
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_stats(self) -> Dict[str, MetricStats]:
        """Get current statistics for all metrics"""
        stats = {}
        for metric_type, collector in self.collectors.items():
            stats[metric_type.value] = collector.get_stats()
        return stats
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "hostname": socket.gethostname(),
            "python_version": platform.python_version()
        }
    
    def export_metrics(self, filepath: str = None) -> str:
        """Export current metrics to JSON"""
        # Convert stats to dict with proper enum handling
        stats_dict = {}
        for k, v in self.get_current_stats().items():
            stat_dict = asdict(v)
            stat_dict['metric_type'] = stat_dict['metric_type'].value if isinstance(stat_dict['metric_type'], MetricType) else stat_dict['metric_type']
            stats_dict[k] = stat_dict
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "current_stats": stats_dict,
            "recent_alerts": [a.to_dict() for a in list(self.alerts)[-20:]],
            "thresholds": asdict(self.thresholds)
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str


# Example usage
def test_performance_monitor():
    """Test the performance monitor"""
    print("\n" + "="*60)
    print("Testing Performance Monitor")
    print("="*60)
    
    # Create monitor with custom thresholds
    thresholds = PerformanceThresholds(
        cpu_warning=50.0,  # Lower thresholds for testing
        cpu_critical=70.0,
        memory_warning=60.0,
        memory_critical=80.0
    )
    
    monitor = PerformanceMonitor(thresholds)
    
    # Register alert callback
    def alert_handler(alert: Alert):
        icon = "⚠️" if alert.level == AlertLevel.WARNING else "🔴"
        print(f"\n{icon} ALERT: {alert.message}")
    
    monitor.register_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start()
    
    print("\n📊 Monitoring system performance...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Run for 10 seconds and display stats
        for i in range(10):
            time.sleep(1)
            
            # Get current stats
            stats = monitor.get_current_stats()
            
            # Display key metrics
            cpu = stats.get(MetricType.CPU_USAGE.value)
            mem = stats.get(MetricType.MEMORY_USAGE.value)
            disk = stats.get(MetricType.DISK_USAGE.value)
            net = stats.get(MetricType.NETWORK_IO.value)
            
            print(f"\r[{i+1}/10] CPU: {cpu.current:.1f}% | "
                  f"Memory: {mem.current:.1f}% | "
                  f"Disk: {disk.current:.1f}% | "
                  f"Network: {net.current:.1f} Mbps", end="")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Export metrics
        print("\n\n📁 Exporting metrics...")
        monitor.export_metrics("performance_metrics.json")
        print("   Saved to performance_metrics.json")
        
        # Show system info
        print("\n💻 System Information:")
        info = monitor.get_system_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    return monitor


if __name__ == "__main__":
    print("Performance Monitor")
    print("="*60)
    
    monitor = test_performance_monitor()
    
    print("\n✅ Performance Monitor ready!")