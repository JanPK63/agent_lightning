"""
Tests for comprehensive health check system
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from shared.health_check import (
    HealthCheckService,
    HealthStatus,
    HealthCheckResult,
    DatabaseHealthCheck,
    SystemHealthCheck,
    ExternalServiceHealthCheck,
    init_health_checks,
    health_check_service
)


class TestHealthCheckService:
    """Test the health check service"""

    def test_health_status_enum(self):
        """Test health status enumeration"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_health_check_result(self):
        """Test health check result dataclass"""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="Test passed",
            details={"key": "value"},
            response_time=1.5
        )

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.response_time == 1.5

        # Test to_dict
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["message"] == "Test passed"

    def test_health_check_service_initialization(self):
        """Test health check service initialization"""
        service = HealthCheckService("test-service")
        assert service.service_name == "test-service"
        assert len(service.checks) == 0

    @pytest.mark.asyncio
    async def test_liveness_check(self):
        """Test liveness check"""
        service = HealthCheckService("test-service")
        result = await service.liveness_check()

        assert result["status"] == "alive"
        assert result["service"] == "test-service"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_readiness_check_no_checks(self):
        """Test readiness check with no health checks"""
        service = HealthCheckService("test-service")
        result = await service.readiness_check()

        assert result["status"] == "unknown"
        assert result["service"] == "test-service"
        assert result["ready"] is False
        assert result["checks"] == 0

    @pytest.mark.asyncio
    async def test_readiness_check_with_checks(self):
        """Test readiness check with health checks"""
        service = HealthCheckService("test-service")

        # Add a mock check that returns healthy
        mock_check = MagicMock()
        mock_check.name = "mock_check"
        mock_check.check = AsyncMock(return_value=HealthCheckResult(
            name="mock_check",
            status=HealthStatus.HEALTHY,
            message="Mock check passed"
        ))
        service.add_check(mock_check)

        result = await service.readiness_check()

        assert result["status"] == "healthy"
        assert result["service"] == "test-service"
        assert result["ready"] is True
        assert result["checks"] == 1

    @pytest.mark.asyncio
    async def test_deep_health_check(self):
        """Test deep health check"""
        service = HealthCheckService("test-service")

        # Add a mock check
        mock_check = MagicMock()
        mock_check.name = "mock_check"
        mock_check.check = AsyncMock(return_value=HealthCheckResult(
            name="mock_check",
            status=HealthStatus.HEALTHY,
            message="Mock check passed",
            details={"detail": "value"}
        ))
        service.add_check(mock_check)

        result = await service.deep_health_check()

        assert result["status"] == "healthy"
        assert result["service"] == "test-service"
        assert result["total_checks"] == 1
        assert result["healthy_checks"] == 1
        assert len(result["checks"]) == 1
        assert result["checks"][0]["name"] == "mock_check"


class TestDatabaseHealthCheck:
    """Test database health check"""

    @pytest.mark.asyncio
    async def test_database_health_check_success(self):
        """Test successful database health check"""
        check = DatabaseHealthCheck("test_db", "write")

        with patch('shared.database.db_manager') as mock_manager:
            mock_session = MagicMock()
            mock_manager.get_write_db.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_manager.get_write_db.return_value.__exit__ = MagicMock(return_value=None)

            result = await check.check()

            assert result.name == "test_db"
            assert result.status == HealthStatus.HEALTHY
            assert "successful" in result.message
            assert result.details["db_type"] == "write"

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self):
        """Test failed database health check"""
        check = DatabaseHealthCheck("test_db", "read")

        with patch('shared.database.db_manager') as mock_manager:
            mock_manager.get_read_db.side_effect = Exception("Connection failed")

            result = await check.check()

            assert result.name == "test_db"
            assert result.status == HealthStatus.UNHEALTHY
            assert "failed" in result.message
            assert "Connection failed" in result.details["error"]


class TestSystemHealthCheck:
    """Test system health check"""

    @pytest.mark.asyncio
    async def test_system_health_check_healthy(self):
        """Test system health check with healthy system"""
        check = SystemHealthCheck("system", 80.0, 80.0)

        with patch('shared.health_check.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_memory = MagicMock()
            mock_memory.percent = 60.0
            mock_memory.used = 8 * (1024**3)  # 8GB
            mock_memory.total = 16 * (1024**3)  # 16GB
            mock_psutil.virtual_memory.return_value = mock_memory

            mock_disk = MagicMock()
            mock_disk.percent = 70.0
            mock_disk.used = 70 * (1024**3)  # 70GB
            mock_disk.total = 100 * (1024**3)  # 100GB
            mock_psutil.disk_usage.return_value = mock_disk

            result = await check.check()

            assert result.name == "system"
            assert result.status == HealthStatus.HEALTHY
            assert "normal" in result.message
            assert result.details["cpu_percent"] == 50.0
            assert result.details["memory_percent"] == 60.0

    @pytest.mark.asyncio
    async def test_system_health_check_degraded(self):
        """Test system health check with degraded system"""
        check = SystemHealthCheck("system", 80.0, 80.0)

        with patch('shared.health_check.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 85.0  # Above threshold
            mock_memory = MagicMock()
            mock_memory.percent = 60.0
            mock_memory.used = 8 * (1024**3)
            mock_memory.total = 16 * (1024**3)
            mock_psutil.virtual_memory.return_value = mock_memory

            result = await check.check()

            assert result.name == "system"
            assert result.status == HealthStatus.DEGRADED
            assert "elevated" in result.message


class TestExternalServiceHealthCheck:
    """Test external service health check"""

    @pytest.mark.asyncio
    async def test_external_service_health_check_success(self):
        """Test successful external service health check"""
        check = ExternalServiceHealthCheck("external_api", "http://api.example.com")

        with patch('shared.health_check.aiohttp') as mock_aiohttp:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            mock_session.get.return_value.__aenter__ = mock_response
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_aiohttp.ClientSession.return_value.__aenter__ = mock_session
            mock_aiohttp.ClientSession.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_aiohttp.ClientTimeout.return_value = MagicMock()

            result = await check.check()

            assert result.name == "external_api"
            assert result.status == HealthStatus.HEALTHY
            assert "healthy" in result.message

    @pytest.mark.asyncio
    async def test_external_service_health_check_failure(self):
        """Test failed external service health check"""
        check = ExternalServiceHealthCheck("external_api", "http://api.example.com")

        with patch('shared.health_check.aiohttp') as mock_aiohttp:
            mock_aiohttp.ClientSession.side_effect = Exception("Connection timeout")

            result = await check.check()

            assert result.name == "external_api"
            assert result.status == HealthStatus.UNHEALTHY
            assert "failed" in result.message


class TestHealthCheckIntegration:
    """Integration tests for health check system"""

    def test_init_health_checks(self):
        """Test health check initialization"""
        service = init_health_checks("test-service")

        assert service.service_name == "test-service"
        # Should have at least system check
        assert len(service.checks) >= 1

    @pytest.mark.asyncio
    async def test_multiple_checks_concurrent(self):
        """Test running multiple health checks concurrently"""
        service = HealthCheckService("test-service")

        # Add multiple mock checks
        for i in range(3):
            mock_check = MagicMock()
            mock_check.name = f"check_{i}"
            mock_check.check = AsyncMock(return_value=HealthCheckResult(
                name=f"check_{i}",
                status=HealthStatus.HEALTHY,
                message=f"Check {i} passed"
            ))
            service.add_check(mock_check)

        summary = await service.run_checks()

        assert summary.status == HealthStatus.HEALTHY
        assert summary.total_checks == 3
        assert summary.healthy_checks == 3
        assert len(summary.checks) == 3

    @pytest.mark.asyncio
    async def test_mixed_check_results(self):
        """Test handling mixed healthy/unhealthy check results"""
        service = HealthCheckService("test-service")

        # Add healthy check
        healthy_check = MagicMock()
        healthy_check.name = "healthy_check"
        healthy_check.check = AsyncMock(return_value=HealthCheckResult(
            name="healthy_check",
            status=HealthStatus.HEALTHY,
            message="Healthy"
        ))
        service.add_check(healthy_check)

        # Add unhealthy check
        unhealthy_check = MagicMock()
        unhealthy_check.name = "unhealthy_check"
        unhealthy_check.check = AsyncMock(return_value=HealthCheckResult(
            name="unhealthy_check",
            status=HealthStatus.UNHEALTHY,
            message="Unhealthy"
        ))
        service.add_check(unhealthy_check)

        summary = await service.run_checks()

        assert summary.status == HealthStatus.UNHEALTHY
        assert summary.total_checks == 2
        assert summary.healthy_checks == 1
        assert summary.unhealthy_checks == 1


if __name__ == "__main__":
    pytest.main([__file__])