"""
Database Connection Pool Metrics for Agent Lightning
Monitors PostgreSQL connection pools and database operations
"""

import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from .metrics import get_metrics

logger = logging.getLogger(__name__)


class DatabaseMetricsCollector:
    """
    Collects metrics for database connection pools and operations.

    This class provides decorators and context managers to monitor:
    - Connection pool utilization
    - Connection acquisition times
    - Query execution times
    - Transaction metrics
    - Connection errors
    """

    def __init__(self, service_name: str = "database"):
        self.service_name = service_name
        self.metrics = get_metrics(service_name)
        self.active_connections = 0
        self.connection_pool_size = 0

    def set_connection_pool_info(self, pool_size: int, min_conn: int = 1, max_conn: int = 20):
        """Set connection pool configuration for metrics."""
        self.connection_pool_size = pool_size
        self.min_connections = min_conn
        self.max_connections = max_conn

        # Record pool configuration as gauge
        self.metrics.db_connection_pool_size.set(pool_size)
        self.metrics.db_connection_pool_min.set(min_conn)
        self.metrics.db_connection_pool_max.set(max_conn)

    def record_connection_acquired(self, pool_name: str = "default"):
        """Record when a database connection is acquired."""
        self.active_connections += 1
        self.metrics.db_connections_active.inc()
        self.metrics.db_connection_acquires_total.labels(pool=pool_name).inc()

        logger.debug(f"Database connection acquired from pool '{pool_name}'. Active: {self.active_connections}")

    def record_connection_released(self, pool_name: str = "default"):
        """Record when a database connection is released."""
        if self.active_connections > 0:
            self.active_connections -= 1
        self.metrics.db_connections_active.dec()
        self.metrics.db_connection_releases_total.labels(pool=pool_name).inc()

        logger.debug(f"Database connection released to pool '{pool_name}'. Active: {self.active_connections}")

    def record_connection_error(self, error_type: str, pool_name: str = "default"):
        """Record database connection errors."""
        self.metrics.db_connection_errors_total.labels(
            error_type=error_type,
            pool=pool_name
        ).inc()

        logger.warning(f"Database connection error in pool '{pool_name}': {error_type}")

    @contextmanager
    def monitor_connection(self, pool_name: str = "default"):
        """
        Context manager to monitor database connection lifecycle.

        Usage:
            with db_metrics.monitor_connection("main_pool"):
                conn = pool.getconn()
                try:
                    # Use connection
                    yield conn
                finally:
                    pool.putconn(conn)
        """
        start_time = time.time()
        self.record_connection_acquired(pool_name)

        try:
            yield
        except Exception as e:
            # Record connection error
            error_type = type(e).__name__
            self.record_connection_error(error_type, pool_name)
            raise
        finally:
            # Record connection acquisition time
            duration = time.time() - start_time
            self.metrics.db_connection_acquisition_duration_seconds.labels(pool=pool_name).observe(duration)

            # Record connection release
            self.record_connection_released(pool_name)

    def record_query_execution(self, query_type: str, table: str, duration: float, rows_affected: int = 0):
        """Record database query execution metrics."""
        self.metrics.db_query_duration_seconds.labels(
            query_type=query_type,
            table=table
        ).observe(duration)

        self.metrics.db_query_total.labels(
            query_type=query_type,
            table=table
        ).inc()

        if rows_affected > 0:
            self.metrics.db_query_rows_affected.labels(
                query_type=query_type,
                table=table
            ).observe(rows_affected)

        logger.debug(f"Query executed: {query_type} on {table} in {duration:.3f}s, rows: {rows_affected}")

    def record_transaction(self, transaction_type: str, duration: float, success: bool = True):
        """Record database transaction metrics."""
        self.metrics.db_transaction_duration_seconds.labels(
            transaction_type=transaction_type
        ).observe(duration)

        if success:
            self.metrics.db_transaction_success_total.labels(transaction_type=transaction_type).inc()
        else:
            self.metrics.db_transaction_failure_total.labels(transaction_type=transaction_type).inc()

        logger.debug(f"Transaction {transaction_type}: {'success' if success else 'failure'} in {duration:.3f}s")

    @contextmanager
    def monitor_query(self, query_type: str, table: str):
        """
        Context manager to monitor database query execution.

        Usage:
            with db_metrics.monitor_query("SELECT", "users"):
                cursor.execute("SELECT * FROM users")
                results = cursor.fetchall()
        """
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_query_execution(query_type, table, duration)

    @contextmanager
    def monitor_transaction(self, transaction_type: str):
        """
        Context manager to monitor database transactions.

        Usage:
            with db_metrics.monitor_transaction("user_creation"):
                with db_manager.get_db() as session:
                    # Perform transaction
                    session.commit()
        """
        start_time = time.time()
        success = False

        try:
            yield
            success = True
        finally:
            duration = time.time() - start_time
            self.record_transaction(transaction_type, duration, success)

    def record_connection_pool_stats(self, pool_name: str, active: int, idle: int, waiting: int = 0):
        """Record detailed connection pool statistics."""
        self.metrics.db_connection_pool_active.labels(pool=pool_name).set(active)
        self.metrics.db_connection_pool_idle.labels(pool=pool_name).set(idle)
        self.metrics.db_connection_pool_waiting.labels(pool=pool_name).set(waiting)

        # Calculate utilization percentage
        if self.connection_pool_size > 0:
            utilization = (active / self.connection_pool_size) * 100
            self.metrics.db_connection_pool_utilization_percent.labels(pool=pool_name).set(utilization)

        logger.debug(f"Pool '{pool_name}' stats: active={active}, idle={idle}, waiting={waiting}")


# Global database metrics collector
_db_metrics_collector = None


def get_database_metrics(service_name: str = "database") -> DatabaseMetricsCollector:
    """Get or create database metrics collector instance."""
    global _db_metrics_collector
    if _db_metrics_collector is None:
        _db_metrics_collector = DatabaseMetricsCollector(service_name)
    return _db_metrics_collector


# Convenience functions for easy integration
def record_db_connection_acquired(pool_name: str = "default"):
    """Convenience function to record connection acquisition."""
    get_database_metrics().record_connection_acquired(pool_name)


def record_db_connection_released(pool_name: str = "default"):
    """Convenience function to record connection release."""
    get_database_metrics().record_connection_released(pool_name)


def record_db_query(query_type: str, table: str, duration: float, rows_affected: int = 0):
    """Convenience function to record query execution."""
    get_database_metrics().record_query_execution(query_type, table, duration, rows_affected)


def record_db_transaction(transaction_type: str, duration: float, success: bool = True):
    """Convenience function to record transaction."""
    get_database_metrics().record_transaction(transaction_type, duration, success)


# Context managers for easy integration
@contextmanager
def monitor_db_connection(pool_name: str = "default"):
    """Context manager for monitoring database connections."""
    with get_database_metrics().monitor_connection(pool_name):
        yield


@contextmanager
def monitor_db_query(query_type: str, table: str):
    """Context manager for monitoring database queries."""
    with get_database_metrics().monitor_query(query_type, table):
        yield


@contextmanager
def monitor_db_transaction(transaction_type: str):
    """Context manager for monitoring database transactions."""
    with get_database_metrics().monitor_transaction(transaction_type):
        yield


# Example usage in database operations:
"""
# 1. Initialize metrics collector
db_metrics = get_database_metrics("memory_service")

# 2. Set pool configuration
db_metrics.set_connection_pool_info(pool_size=20, min_conn=5, max_conn=50)

# 3. Monitor connection usage
with monitor_db_connection("memory_pool"):
    conn = pool.getconn()
    try:
        # Use connection
        with monitor_db_query("SELECT", "agent_memories"):
            cursor.execute("SELECT * FROM agent_memories WHERE id = %s", (memory_id,))
            result = cursor.fetchone()

        with monitor_db_transaction("memory_update"):
            # Update memory
            cursor.execute("UPDATE agent_memories SET ...")
            conn.commit()
    finally:
        pool.putconn(conn)

# 4. Record pool statistics periodically
db_metrics.record_connection_pool_stats("memory_pool", active=5, idle=10, waiting=2)
"""