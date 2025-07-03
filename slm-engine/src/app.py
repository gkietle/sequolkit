import logging
import os
import atexit
import socket
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace

# Import application components
from config.app_config import app_config, initialize_workflows, llm_config, preload_llm
from api.models import create_api_models
from api.routes import initialize_routes
from exceptions.global_exception_handler import register_error_handlers
from services.observability import observability_service

# Configure logging
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app)
    logger.info("Flask app initialized with CORS support")

    # Initialize Flask-RESTX
    api = Api(
        app,
        version='1.0',
        title='SLM Engine API',
        description='A RESTful API for SQL Language Model Engine',
        doc='/docs'
    )

    # Create API models for documentation
    api_models = create_api_models(api)
    
    # Initialize workflows
    workflows = initialize_workflows()
    
    # Register error handlers
    register_error_handlers(app)
    logger.info("Error handlers registered")
    
    # Initialize routes
    initialize_routes(api, api_models, workflows)
    logger.info("API routes initialized")

    # Pre-load LLM
    is_healthy, response = preload_llm()
    if not is_healthy:
        logger.error(f"Failed to pre-load LLM: {response}")
    
    return app

# Create the application
app = create_app()

if __name__ == '__main__':
    app_config.print_banner()
    logger.info(f"Starting Flask application on port {app_config.SERVICE_PORT}")
    app.run(host='0.0.0.0', port=app_config.SERVICE_PORT, debug=True)