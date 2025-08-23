from typing import Dict, Any
import time
import psutil
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()

class HealthStatus(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float


# Track startup time for uptime calculation
_startup_time = time.time()

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns system status and basic metrics. This endpoint should be
    fast (<50ms) and used by load balancers for health checks.
    """
    
    uptime = time.time() - _startup_time
    
    # Basic system checks
    checks = {
        "api": "healthy",  # If we're responding, API is healthy
        "uptime_seconds": uptime,
    }
    
    # Determine overall status
    overall_status = "healthy"
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",  # TODO: Get from config or environment
        uptime_seconds=uptime,
        checks=checks
    )


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """
    Detailed health check with system metrics and dependency checks.
    
    Used for monitoring dashboards and detailed system analysis.
    May take longer than basic health check.
    """
    
    uptime = time.time() - _startup_time
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_metrics = SystemMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_available_mb=memory.available / 1024 / 1024,
        disk_usage_percent=disk.percent
    )
    
    # Dependency checks
    dependency_checks = {}
    
    # TODO: Add database connection check
    # dependency_checks["database"] = await check_database_connection()
    
    # TODO: Add Redis connection check
    # dependency_checks["redis"] = await check_redis_connection()
    
    # TODO: Add external API checks
    # dependency_checks["sleeper_api"] = await check_sleeper_api()
    
    # Determine overall health
    failed_checks = [name for name, status in dependency_checks.items() 
                    if status != "healthy"]
    
    if failed_checks:
        if len(failed_checks) == len(dependency_checks):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "system_metrics": system_metrics.dict(),
        "dependency_checks": dependency_checks,
        "failed_checks": failed_checks,
    }


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if the service is ready to handle requests,
    503 if it's not ready (e.g., still loading ML models).
    """
    
    # TODO: Check if ML models are loaded
    # if not ml_models_loaded:
    #     raise HTTPException(status_code=503, detail="ML models not loaded")
    
    # TODO: Check if database is accessible
    # if not database_accessible:
    #     raise HTTPException(status_code=503, detail="Database not accessible")
    
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if the service is alive (even if not ready).
    Should only return 503 if the service needs to be restarted.
    """
    
    # Basic liveness check - if we can respond, we're alive
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


# Utility functions for future use
async def check_database_connection() -> str:
    """Check database connectivity."""
    # TODO: Implement database ping
    try:
        # await database.ping()
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_redis_connection() -> str:
    """Check Redis connectivity."""
    # TODO: Implement Redis ping
    try:
        # await redis.ping()
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_sleeper_api() -> str:
    """Check Sleeper API connectivity."""
    # TODO: Implement lightweight Sleeper API check
    try:
        # Make a simple API call to Sleeper
        return "healthy"
    except Exception:
        return "unhealthy"