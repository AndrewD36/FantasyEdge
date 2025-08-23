"""
FastAPI application factory and configuration.

This follows the application factory pattern, making testing easier
and allowing for different configurations (dev, test, prod).
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routes.health import router as health_router
from .routes.sleeper import router as sleeper_router
from .middleware.cors import setup_cors
from .routes.simulation import router as simulation_router

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Fantasy Draft API...")

    # NEW: Initialize simulation resources
    
    # app.state.player_database = await load_player_database()
    # app.state.simulation_cache = await initialize_simulation_cache()
    # app.state.monte_carlo_simulator = MonteCarloSimulator(
    #     app.state.player_database,
    #     thread_pool_size=4
    # )
    
    logger.info("Simulation engine initialized")

    # TODO: Initialize database connections
    # TODO: Load ML models
    # TODO: Initialize external API clients
    # TODO: Warm up caches
    
    logger.info("Startup complete")
    
    yield
    
    # Shutdown tasks
    logger.info("Shutting down Fantasy Draft AI API...")

    if hasattr(app.state, 'monte_carlo_simulator'):
        await app.state.monte_carlo_simulator.close()
    
    # TODO: Close database connections
    # TODO: Save any necessary state
    # TODO: Clean up resources
    
    logger.info("Shutdown complete")


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Application factory function.
    
    Creates and configures a FastAPI application instance.
    This pattern makes testing easier and allows for different configurations.
    """
    
    # Default configuration
    default_config = {
        "title": "Fantasy Draft AI",
        "description": "AI-powered fantasy football draft assistance",
        "version": "1.0.0",
        "debug": False,
        "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Create FastAPI app
    app = FastAPI(
        title=default_config["title"],
        description=default_config["description"],
        version=default_config["version"],
        debug=default_config["debug"],
        lifespan=lifespan,
        docs_url="/docs" if default_config["debug"] else None,  # Disable docs in prod
        redoc_url="/redoc" if default_config["debug"] else None,
    )

    app.include_router(simulation_router, prefix="/api/v1", tags=["simulation"])
    
    # Setup middleware
    setup_cors(app, default_config["cors_origins"])
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add response time header for monitoring."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests for monitoring and debugging."""
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s - {request.method} {request.url.path}"
        )
        
        return response
    
    # Exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions with consistent format."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "message": exc.detail,
                    "status_code": exc.status_code,
                }
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "message": "An unexpected error occurred",
                    # Don't leak error details in production
                    "details": str(exc) if default_config["debug"] else None,
                }
            }
        )
    
    # Include routers
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(sleeper_router, prefix="/api/v1", tags=["sleeper"])
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic API information."""
        return {
            "name": default_config["title"],
            "version": default_config["version"],
            "description": default_config["description"],
            "docs_url": "/docs" if default_config["debug"] else None,
            "health_check": "/api/v1/health",
        }
    
    return app


# Create default app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )