import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def async_route(f):
    """Decorator to handle async routes in Flask-RESTX"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped 