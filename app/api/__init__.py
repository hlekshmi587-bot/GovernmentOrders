"""
FastAPI Backend Package for Kerala Government Orders Multi-Agent System.
"""

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name in ("app", "create_app"):
        from app.api.main import app as _app, create_app as _create_app
        if name == "app":
            return _app
        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

