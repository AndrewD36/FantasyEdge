"""
CORS (Cross-Origin Resource Sharing) middleware configuration.

Essential for frontend applications running on different domains/ports.
Configured securely with proper origin validation.
"""

from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def setup_cors(app: FastAPI, allowed_origins: List[str] = None):
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins. Defaults to common dev origins.
    """
    
    if allowed_origins is None:
        # Default origins for development
        allowed_origins = [
            "http://localhost:3000",    # React dev server
            "http://localhost:8080",    # Vue dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"],  # Expose our custom timing header
    )


def setup_production_cors(app: FastAPI, production_origins: List[str]):
    """
    Configure CORS for production with stricter settings.
    
    Args:
        app: FastAPI application instance
        production_origins: List of production domain origins
    """
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=production_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],  # More restrictive methods
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
        ],
        expose_headers=["X-Process-Time"],
    )