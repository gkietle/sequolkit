import os
import atexit
import socket
from flask import Flask, request, jsonify
from flask_cors import CORS
from core.workflow import SQLAgentWorkflow, SchemaEnrichmentWorkflow
from core.baseline_workflow import BaselineWorkflow
from core.templates import text2sql_prompt_routing, TABLE_RETRIEVAL_TMPL
from core.services import get_schema, get_sample_data
from core.llm import llm_config
from core.services import validate_connection_payload
from exceptions.global_exception_handler import register_error_handlers
from exceptions.app_exception import AppException
from response.app_response import ResponseWrapper
from dotenv import load_dotenv
import logging
from pathlib import Path

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting application initialization...")

# Load environment variables
env_path = Path('.env')
load_dotenv(dotenv_path=env_path)
logger.info(f"Looking for .env file at: {env_path.absolute()}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logger.info("Flask app initialized with CORS support")

# Register error handlers
register_error_handlers(app)
logger.info("Error handlers registered")

# Initialize workflows with the centralized LLM
TEXT_TO_SQL_PROMPT_TMPL = text2sql_prompt_routing(llm_config.settings["prompt_routing"])
workflow = SQLAgentWorkflow(
    text2sql_prompt=TEXT_TO_SQL_PROMPT_TMPL,
    table_retrieval_prompt=TABLE_RETRIEVAL_TMPL,
    llm=llm_config.get_llm(),
    verbose=True
)
logger.info("SQL Agent Workflow initialized successfully")

schema_workflow = SchemaEnrichmentWorkflow(
    llm=llm_config.get_llm(),
    verbose=True
)
logger.info("Schema Enrichment Workflow initialized successfully")

baseline_workflow = BaselineWorkflow(
    llm=llm_config.get_llm(),
    text2sql_prompt=TEXT_TO_SQL_PROMPT_TMPL,
    verbose=True
)
logger.info("Baseline Workflow initialized successfully")

SERVICE_PORT = int(os.getenv("SERVICE_PORT", 5000))

# Default values for new settings
PROMPT_ROUTING = int(os.getenv("PROMPT_ROUTING", 0))
ENRICH_SCHEMA = os.getenv("ENRICH_SCHEMA", "True").lower() in ["true", "1", "yes", "y"]

# Log all configuration values
logger.info("Configuration loaded:")
logger.info(f"PROMPT_ROUTING: {PROMPT_ROUTING}")
logger.info(f"ENRICH_SCHEMA: {ENRICH_SCHEMA}")

def print_banner(banner_file='banner.txt'):
    """Print a banner from a file when the application starts if it exists"""
    try:
        # Check if the banner file exists
        if os.path.exists(banner_file):
            # Read the banner from the file
            with open(banner_file, 'r') as f:
                banner = f.read()
            logger.info("\n" + banner)
    except Exception:
        # Silently ignore any errors reading the banner file
        pass
        
    # Always log server started message
    logger.info("Server started successfully!")

@app.route('/')
def home():
    return "SLM Engine is running!"

@app.route('/query', methods=['POST'])
async def query():
    """Handle regular query endpoint"""
    logger.info("Received request to /query endpoint")
    try:
        data = request.json
        query = data.get("query")
        connection_payload = data.get("connection_payload")
        logger.info(f"Processing query: {query}")

        if not query or not connection_payload:
            logger.warning("Missing required parameters in request")
            return jsonify({"error": "Missing 'query' or 'connection_payload'"}), 400
        
        is_valid, error_message = validate_connection_payload(connection_payload)
        if not is_valid:
            logger.warning(f"Invalid connection payload: {error_message}")
            return jsonify({"error": error_message}), 400

        logger.info("Retrieving schema from database")

        table_details = get_schema(connection_payload)
        for table in table_details:
                table["sample_data"] = get_sample_data(
                    connection_payload=connection_payload, 
                    table_details=table
                )
        database_description = ""
        # Only process schema enrichment if enrich_schema setting is True
        if ENRICH_SCHEMA and "schema_enrich_info" in connection_payload and connection_payload["schema_enrich_info"] is not None:
            logger.info(f"Retrieved schema with {len(table_details)} tables and enrichment information")
            
            # Tạo mapping từ tableIdentifier đến enriched table để tìm kiếm nhanh hơn
            enriched_tables_map = {
                table["tableIdentifier"]: table 
                for table in connection_payload["schema_enrich_info"]["enriched_schema"]
            }
            
            # Add database description to response data if available
            if "database_description" in connection_payload["schema_enrich_info"]:
                database_description = connection_payload["schema_enrich_info"]["database_description"]
            
            for table in table_details:
                table_id = table["tableIdentifier"]
                
                # Kiểm tra xem bảng có trong mapping không
                if table_id in enriched_tables_map:
                    enriched_table = enriched_tables_map[table_id]
                    
                    # Cập nhật mô tả bảng
                    table["tableDescription"] = enriched_table.get("tableDescription", "")
                    
                    # Tạo mapping từ columnIdentifier đến enriched column
                    enriched_columns_map = {
                        col["columnIdentifier"]: col 
                        for col in enriched_table["columns"]
                    }
                    
                    # Cập nhật mô tả cột
                    for column in table["columns"]:
                        column_id = column["columnIdentifier"]
                        
                        # Kiểm tra điều kiện cập nhật và xem cột có trong mapping không
                        is_empty_description = (
                            column.get("columnDescription") in ["", "NULL", "''"] or
                            column.get("columnDescription") is None or
                            len(str(column.get("columnDescription", ""))) <= 1
                        )
                        
                        if is_empty_description and column_id in enriched_columns_map:
                            enriched_column = enriched_columns_map[column_id]
                            column["columnDescription"] = enriched_column.get("columnDescription", "")
        else:
            logger.info(f"Schema enrichment is disabled or no enrichment info available")
                
        logger.info(f"Retrieved schema with {len(table_details)} tables")

        logger.info("Executing workflow")
        response = await workflow.run(
            query=query,
            table_details=table_details,
            database_description=database_description,
            connection_payload=connection_payload
        )
        logger.info("Workflow completed successfully")
        
        return ResponseWrapper.success(response)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise AppException(str(e), 500)
    
@app.route('/query-baseline', methods=['POST'])
async def query_baseline():
    """Handle regular query endpoint"""
    logger.info("Received request to /query-baseline endpoint")
    try:
        data = request.json
        query = data.get("query")
        connection_payload = data.get("connection_payload")
        logger.info(f"Processing query: {query}")
        
        if not query or not connection_payload:
            logger.warning("Missing required parameters in request")
            return jsonify({"error": "Missing 'query' or 'connection_payload'"}), 400
        
        is_valid, error_message = validate_connection_payload(connection_payload)
        if not is_valid:
            logger.warning(f"Invalid connection payload: {error_message}")
            return jsonify({"error": error_message}), 400
        
        logger.info("Retrieving schema from database")
        
        table_details = get_schema(connection_payload)
        for table in table_details:
                table["sample_data"] = get_sample_data(
                    connection_payload=connection_payload, 
                    table_details=table
                )
        database_description = ""
        # Only process schema enrichment if enrich_schema setting is True
        if ENRICH_SCHEMA and "schema_enrich_info" in connection_payload and connection_payload["schema_enrich_info"] is not None:
            logger.info(f"Retrieved schema with {len(table_details)} tables and enrichment information")
            
            # Tạo mapping từ tableIdentifier đến enriched table để tìm kiếm nhanh hơn
            enriched_tables_map = {
                table["tableIdentifier"]: table 
                for table in connection_payload["schema_enrich_info"]["enriched_schema"]
            }
            
            # Add database description to response data if available
            if "database_description" in connection_payload["schema_enrich_info"]:
                database_description = connection_payload["schema_enrich_info"]["database_description"]
            
            for table in table_details:
                table_id = table["tableIdentifier"]
                
                # Kiểm tra xem bảng có trong mapping không
                if table_id in enriched_tables_map:
                    enriched_table = enriched_tables_map[table_id]
                    
                    # Cập nhật mô tả bảng
                    table["tableDescription"] = enriched_table.get("tableDescription", "")
                    
                    # Tạo mapping từ columnIdentifier đến enriched column
                    enriched_columns_map = {
                        col["columnIdentifier"]: col 
                        for col in enriched_table["columns"]
                    }
                    
                    # Cập nhật mô tả cột
                    for column in table["columns"]:
                        column_id = column["columnIdentifier"]
                        
                        # Kiểm tra điều kiện cập nhật và xem cột có trong mapping không
                        is_empty_description = (
                            column.get("columnDescription") in ["", "NULL", "''"] or
                            column.get("columnDescription") is None or
                            len(str(column.get("columnDescription", ""))) <= 1
                        )
                        
                        if is_empty_description and column_id in enriched_columns_map:
                            enriched_column = enriched_columns_map[column_id]
                            column["columnDescription"] = enriched_column.get("columnDescription", "")
        else:
            logger.info(f"Schema enrichment is disabled or no enrichment info available")

        logger.info(f"Retrieved schema with {len(table_details)} tables")
        logger.info("Executing workflow")
        response = await baseline_workflow.run(
            query=query,
            table_details=table_details,
            connection_payload=connection_payload,
            database_description=database_description
        )

        logger.info("Workflow completed successfully")
        
        return ResponseWrapper.success(response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise AppException(str(e), 500)


@app.route('/schema-enrichment', methods=['POST'])
async def schema_enrichment():
    """Handle schema enrichment endpoint"""
    logger.info("Received request to /schema-enrichment endpoint")
    try:
        data = request.json
        connection_payload = data.get("connection_payload")     

        if not connection_payload:
            logger.warning("Missing required parameters in request")
            return jsonify({"error": "Missing 'connection_payload' or 'database_schema'"}), 400
        
        is_valid, error_message = validate_connection_payload(connection_payload)
        if not is_valid:
            logger.warning(f"Invalid connection payload: {error_message}")
            return jsonify({"error": error_message}), 400

        logger.info("Retrieving database schema...")
        table_details = get_schema(connection_payload)
           
        logger.info(f"Retrieved schema with {len(table_details)} tables")

        logger.info("Executing workflow")
        response = await schema_workflow.run(
            connection_payload=connection_payload,
            database_schema=table_details
        )
        logger.info("Workflow completed successfully")
        response["original_schema"] = get_schema(connection_payload)
        return ResponseWrapper.success(response)

    except Exception as e:
        logger.error(f"Error processing schema enrichment: {str(e)}", exc_info=True)
        raise AppException(str(e), 500) 

@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current LLM settings"""
    logger.info("Received request to view current LLM settings")
    try:
        return ResponseWrapper.success(llm_config.get_settings())
    except Exception as e:
        logger.error(f"Error retrieving settings: {str(e)}", exc_info=True)
        raise AppException(str(e), 500)

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update LLM settings"""
    logger.info("Received request to update LLM settings")
    try:
        data = request.json
        
        # Extract settings from request
        host = data.get("ollama_host")
        model = data.get("ollama_model")
        additional_kwargs = data.get("additional_kwargs")
        prompt_routing = data.get("prompt_routing")
        enrich_schema = data.get("enrich_schema")
        
        # Validate that at least one setting is provided
        if host is None and model is None and additional_kwargs is None and prompt_routing is None and enrich_schema is None:
            logger.warning("No settings provided in request")
            return jsonify({"error": "At least one setting must be provided"}), 400
        
        # Update settings using the centralized LLM config
        llm_config.update_settings(
            host=host,
            model=model,
            additional_kwargs=additional_kwargs,
            prompt_routing=prompt_routing,
            enrich_schema=enrich_schema
        )
        
        # Reinitialize workflows with new LLM
        TEXT_TO_SQL_PROMPT_TMPL = text2sql_prompt_routing(llm_config.settings["prompt_routing"])
        workflow.llm = llm_config.get_llm()
        schema_workflow.llm = llm_config.get_llm()
        baseline_workflow.llm = llm_config.get_llm()
        
        logger.info("LLM settings updated successfully")
        return ResponseWrapper.success({
            "message": "Settings updated successfully",
            "current_settings": llm_config.get_settings()
        })
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}", exc_info=True)
        raise AppException(str(e), 500)
    
@app.route('/health-check', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.debug("Health check request received")
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print_banner()
    logger.info(f"Starting Flask application on port {SERVICE_PORT}")
    app.run(host='0.0.0.0', port=SERVICE_PORT, debug=True)