import os
import logging
from flask import request, jsonify
from flask_restx import Resource
from config.app_config import llm_config
from core.services import get_schema, get_sample_data_improved, validate_connection_payload
from core.utils import enrich_schema_with_info
from exceptions.app_exception import AppException
from response.app_response import ResponseWrapper
from middleware.async_handler import async_route
from services.observability import observability_service
from enums.response_enum import ResponseEnum

logger = logging.getLogger(__name__)

# Store workflow instances
workflow = None
schema_workflow = None
baseline_workflow = None
question_workflow = None

def initialize_routes(api, api_models, workflows):
    """Initialize API routes with the given Flask-RESTX API"""
    global workflow, schema_workflow, baseline_workflow, question_workflow
    workflow, schema_workflow, baseline_workflow, question_workflow = workflows
    
    connection_payload_model = api_models['connection_payload_model']
    query_request_model = api_models['query_request_model']
    question_request_model = api_models['question_request_model']
    settings_model = api_models['settings_model']
    schema_enrich_request_model = api_models['schema_enrich_request_model'] 

    @api.route('/')
    class Home(Resource):
        @api.doc('home',
            responses={
                200: 'Success'
            }
        )
        def get(self):
            """Check if the service is running"""
            return "SLM Engine is running!"

    @api.route('/query')
    class Query(Resource):
        @api.expect(query_request_model)
        @api.doc('query',
            responses={
                200: 'Success',
                400: 'Bad Request - Missing or invalid parameters',
                500: 'Internal Server Error'
            }
        )
        @async_route
        async def post(self):
            """Process a natural language query and return SQL results"""
            logger.info("Received request to /query endpoint")
            try:
                data = request.json
                query = data.get("query")
                connection_payload = data.get("connection_payload")
                logger.info(f"Processing query: {query}")
                session_information = data.get("session_information")
                schema_enrich_info = connection_payload.get("schema_enrich_info")
                
                schema = []
                if 'enrich_schema' in schema_enrich_info:
                    schema = schema_enrich_info['enrich_schema']
                list_available_tables = [table["tableIdentifier"] for table in schema]
                logger.info(f"Available tables: {list_available_tables}")
                
                # Create a Langfuse trace for this query
                trace = observability_service.langfuse.trace(
                    name="text_to_sql_query",
                    user_id=connection_payload.get("username", "unknown"),
                    metadata={
                        "query": query,
                        "db_type": connection_payload.get("dbType", "unknown")
                    }
                )

                if not query or not connection_payload:
                    logger.warning("Missing required parameters in request")
                    trace.update(
                        status="failed",
                        metadata={"error": "Missing 'query' or 'connection_payload'"}
                    )
                    return jsonify({"error": "Missing 'query' or 'connection_payload'"}), 400
                
                is_valid, error_message = validate_connection_payload(connection_payload)
                if not is_valid:
                    logger.warning(f"Invalid connection payload: {error_message}")
                    trace.update(status="failed", metadata={"error": error_message})
                    return jsonify({"error": error_message}), 400

                # Create a span for schema retrieval
                schema_span = observability_service.langfuse.span(
                    name="schema_retrieval",
                    parent_id=trace.id
                )
                
                from config.app_config import app_config
                
                logger.info("Retrieving schema from database")
                table_details = get_schema(connection_payload)
                for table in table_details:
                    table["sample_data"] = []
                
                logger.info(f"Enrich schema is enabled: {app_config.ENRICH_SCHEMA}")
                # List all keys in connection_payload
                logger.info(f"Connection payload keys: {connection_payload.keys()}")

                # Enrich schema with additional information
                if app_config.ENRICH_SCHEMA == False:
                    logger.info("Enrich schema is disabled, skipping enrichment")
                    table_details, database_description = table_details, None
                else:
                    table_details, database_description = enrich_schema_with_info(table_details, connection_payload)
                    logger.info(f"Retrieved schema with {len(table_details)} tables")

                # Filter tables that are not in the list of available tables
                if len(list_available_tables) > 0:
                    table_details = [table for table in table_details if table["tableIdentifier"] in list_available_tables]
                logger.info(f"Filtered tables: {[table['tableIdentifier'] for table in table_details]}")
                schema_span.update(
                    metadata={
                        "table_count": len(table_details),
                        "has_database_description": bool(database_description)
                    }
                )

                # Trace the query processing using LlamaIndex instrumentor
                logger.info("Executing workflow")
                with observability_service.llama_index_instrumentor.observe(trace_id=trace.id):
                    response = await workflow.run(
                        query=query,
                        table_details=table_details,
                        database_description=database_description,
                        connection_payload=connection_payload,
                        session_information=session_information
                    )
                    logger.info(f"Trace ID: {trace.id}")
                
                # Score the query and update the trace with the result
                if response:
                    trace.span(output=response)
                    trace.update(
                        status="success",
                        metadata={"generated_sql": response}
                    )
                else:
                    trace.update(
                        status="failed",
                        metadata={"error": "Failed to generate SQL or execute query"}
                    )
                
                logger.info("Workflow completed successfully")
                # Log response wrapper content
                logger.info(f"Response wrapper content: {response}")
                if "SELECT" in response.upper():
                    return ResponseWrapper.success(response)
                else:
                    return ResponseWrapper.error_with_code(ResponseEnum.WORKFLOW_FAILED.code, response)

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                if 'trace' in locals():
                    trace.update(
                        status="failed",
                        metadata={"error": str(e)}
                    )
                raise AppException(str(e), 500)

    @api.route('/query-baseline')
    class QueryBaseline(Resource):
        @api.expect(query_request_model)
        @api.doc('query_baseline',
            responses={
                200: 'Success',
                400: 'Bad Request - Missing or invalid parameters',
                500: 'Internal Server Error'
            }
        )
        @async_route
        async def post(self):
            """Process a natural language query using baseline workflow"""
            logger.info("Received request to /query-baseline endpoint")
            try:
                data = request.json
                query = data.get("query")
                connection_payload = data.get("connection_payload")
                logger.info(f"Processing query: {query}")
                
                # Create a Langfuse trace for this baseline query
                trace = observability_service.langfuse.trace(
                    name="baseline_text_to_sql_query",
                    user_id=connection_payload.get("username", "unknown"),
                    metadata={
                        "query": query,
                        "db_type": connection_payload.get("dbType", "unknown"),
                        "workflow": "baseline"
                    }
                )
                
                if not query or not connection_payload:
                    logger.warning("Missing required parameters in request")
                    trace.update(
                        status="failed",
                        metadata={"error": "Missing 'query' or 'connection_payload'"}
                    )
                    return jsonify({"error": "Missing 'query' or 'connection_payload'"}), 400
                
                is_valid, error_message = validate_connection_payload(connection_payload)
                if not is_valid:
                    logger.warning(f"Invalid connection payload: {error_message}")
                    trace.update(
                        status="failed",
                        metadata={"error": error_message}
                    )
                    return jsonify({"error": error_message}), 400
                
                # Create a span for schema retrieval
                schema_span = observability_service.langfuse.span(
                    name="schema_retrieval",
                    parent_id=trace.id
                )
                
                logger.info("Retrieving schema from database")
                table_details = get_schema(connection_payload)
                for table in table_details:
                    table["sample_data"] = get_sample_data_improved(
                        connection_payload=connection_payload, 
                        table_details=table
                    )
                
                # Enrich schema with additional information
                table_details, database_description = enrich_schema_with_info(table_details, connection_payload)
                logger.info(f"Retrieved schema with {len(table_details)} tables")
                
                schema_span.update(
                    metadata={
                        "table_count": len(table_details),
                        "has_database_description": bool(database_description)
                    }
                )
                
                # Trace the query processing using LlamaIndex instrumentor
                logger.info("Executing workflow")
                with observability_service.llama_index_instrumentor.observe(trace_id=trace.id):
                    response = await baseline_workflow.run(
                        query=query,
                        table_details=table_details,
                        database_description=database_description,
                        connection_payload=connection_payload
                    )


                if response:
                    trace.update(
                        status="success",
                        metadata={
                            "generated_sql": response
                        }
                    )
                else:
                    trace.update(
                        status="failed",
                        metadata={"error": "Failed to generate SQL or execute query"}
                    )
                
                logger.info("Workflow completed successfully")
                
                return ResponseWrapper.success(response)

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                if 'trace' in locals():
                    trace.update(
                        status="failed",
                        metadata={"error": str(e)}
                    )
                raise AppException(str(e), 500)

    @api.route('/schema-enrichment')
    class SchemaEnrichment(Resource):
        @api.expect(schema_enrich_request_model)
        @api.doc('schema_enrichment',
            responses={
                200: 'Success',
                400: 'Bad Request - Missing or invalid parameters',
                500: 'Internal Server Error'
            }
        )
        @async_route
        async def post(self):
            """Enrich database schema with additional information"""
            logger.info("Received request to /schema-enrichment endpoint")
            try:
                data = request.json
                connection_payload = data.get("connection_payload")
                
                # Create a Langfuse trace for schema enrichment
                trace = observability_service.langfuse.trace(
                    name="schema_enrichment",
                    user_id=connection_payload.get("username", "unknown"),
                    metadata={
                        "db_type": connection_payload.get("dbType", "unknown")
                    }
                )

                if not connection_payload:
                    logger.warning("Missing required parameters in request")
                    trace.update(
                        status="failed",
                        metadata={"error": "Missing 'connection_payload' or 'database_schema'"}
                    )
                    return jsonify({"error": "Missing 'connection_payload' or 'database_schema'"}), 400
                
                is_valid, error_message = validate_connection_payload(connection_payload)
                if not is_valid:
                    logger.warning(f"Invalid connection payload: {error_message}")
                    trace.update(
                        status="failed",
                        metadata={"error": error_message}
                    )
                    return jsonify({"error": error_message}), 400
                
                # Create a span for schema retrieval
                schema_span = observability_service.langfuse.span(
                    name="schema_retrieval",
                    parent_id=trace.id
                )
                
                logger.info("Retrieving database schema...")
                table_details = get_schema(connection_payload)
                
                schema_span.update(
                    metadata={
                        "table_count": len(table_details)
                    }
                )
                
                logger.info(f"Retrieved schema with {len(table_details)} tables")

                # Trace the enrichment workflow using LlamaIndex instrumentor
                logger.info("Executing workflow")
                with observability_service.llama_index_instrumentor.observe(trace_id=trace.id):
                    response = await schema_workflow.run(
                        connection_payload=connection_payload,
                        database_schema=table_details
                    )
                
                response["original_schema"] = get_schema(connection_payload)
                
                # Update trace with enrichment results
                if response:
                    trace.update(
                        status="success",
                        metadata={
                            "data": response
                        }
                    )
                else:
                    trace.update(
                        status="failed",
                        metadata={"error": "Failed to enrich schema"}
                    )
                
                return ResponseWrapper.success(response)

            except Exception as e:
                logger.error(f"Error processing schema enrichment: {str(e)}", exc_info=True)
                if 'trace' in locals():
                    trace.update(
                        status="failed",
                        metadata={"error": str(e)}
                    )
                raise AppException(str(e), 500)

    @api.route('/schema-enrichment-lite')
    class SchemaEnrichmentLite(Resource):
        @api.expect(schema_enrich_request_model)
        @api.doc('schema_enrichment_lite',
            responses={
                200: 'Success',
                400: 'Bad Request - Missing or invalid parameters',
                500: 'Internal Server Error'
            }
        )
        @async_route
        async def post(self):
            """Enrich database schema with additional information (lite version)"""
            logger.info("Received request to /schema-enrichment-lite endpoint")
            try:
                data = request.json
                connection_payload = data.get("connection_payload")
                
                # Create a Langfuse trace for schema enrichment lite
                trace = observability_service.langfuse.trace(
                    name="schema_enrichment_lite",
                    user_id=connection_payload.get("username", "unknown"),
                    metadata={
                        "db_type": connection_payload.get("dbType", "unknown")
                    }
                )

                if not connection_payload:
                    logger.warning("Missing required parameters in request")
                    trace.update(
                        status="failed",
                        metadata={"error": "Missing 'connection_payload'"}
                    )
                    return jsonify({"error": "Missing 'connection_payload'"}), 400
                
                is_valid, error_message = validate_connection_payload(connection_payload)
                if not is_valid:
                    logger.warning(f"Invalid connection payload: {error_message}")
                    trace.update(
                        status="failed",
                        metadata={"error": error_message}
                    )
                    return jsonify({"error": error_message}), 400
                
                # Create a span for schema retrieval
                schema_span = observability_service.langfuse.span(
                    name="schema_retrieval",
                    parent_id=trace.id
                )
                
                logger.info("Retrieving database schema...")
                table_details = get_schema(connection_payload)
                
                schema_span.update(
                    metadata={
                        "table_count": len(table_details)
                    }
                )
                
                logger.info(f"Retrieved schema with {len(table_details)} tables")

                # Trace the enrichment workflow using LlamaIndex instrumentor
                logger.info("Executing workflow")
                with observability_service.llama_index_instrumentor.observe(trace_id=trace.id):
                    response = await schema_workflow.run(
                        connection_payload=connection_payload,
                        database_schema=table_details
                    )
                
                # Transform the response to lite format
                lite_response = {
                    "database_description": response.get("database_description", ""),
                    "tables": []
                }
                
                # Extract enriched schema and transform to lite format
                enriched_schema = response.get("enriched_schema", [])
                for table in enriched_schema:
                    lite_table = {
                        "tableIdentifier": table.get("tableIdentifier", ""),
                        "tableDescription": table.get("tableDescription", ""),
                        "columns": []
                    }
                    
                    # Transform columns to lite format
                    for column in table.get("columns", []):
                        lite_column = {
                            "columnIdentifier": column.get("columnIdentifier", ""),
                            "columnDescription": column.get("columnDescription", "")
                        }
                        lite_table["columns"].append(lite_column)
                    
                    lite_response["tables"].append(lite_table)
                
                # Update trace with enrichment results
                if lite_response:
                    trace.update(
                        status="success",
                        metadata={
                            "data": lite_response
                        }
                    )
                else:
                    trace.update(
                        status="failed",
                        metadata={"error": "Failed to enrich schema"}
                    )
                
                return ResponseWrapper.success(lite_response)

            except Exception as e:
                logger.error(f"Error processing schema enrichment lite: {str(e)}", exc_info=True)
                if 'trace' in locals():
                    trace.update(
                        status="failed",
                        metadata={"error": str(e)}
                    )
                raise AppException(str(e), 500)

    @api.route('/settings')
    class Settings(Resource):
        @api.doc('get_settings',
            responses={
                200: 'Success',
                500: 'Internal Server Error'
            }
        )
        def get(self):
            """Get current LLM settings"""
            logger.info("Received request to view current LLM settings")
            try:
                settings = llm_config.get_settings()
                # Add current provider to the response
                settings["provider"] = os.getenv("LLM_PROVIDER", "ollama").lower()
                return ResponseWrapper.success(settings)
            except Exception as e:
                logger.error(f"Error retrieving settings: {str(e)}", exc_info=True)
                raise AppException(str(e), 500)

        @api.expect(settings_model)
        @api.doc('update_settings',
            responses={
                200: 'Success',
                400: 'Bad Request - Invalid settings',
                500: 'Internal Server Error'
            }
        )
        def post(self):
            """Update LLM settings"""
            logger.info("Received request to update LLM settings")
            try:
                data = request.json
                
                # Create a Langfuse trace for settings update
                trace = observability_service.langfuse.trace(
                    name="settings_update",
                    metadata={
                        "settings": data
                    }
                )
                
                # Get current provider and check if it's being changed
                current_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
                new_provider = data.get("provider", current_provider).lower()
                
                # If provider is changing, validate the new provider
                if new_provider != current_provider:
                    if new_provider not in ["ollama", "google"]:
                        trace.update(
                            status="failed",
                            metadata={"error": f"Unsupported LLM provider: {new_provider}"}
                        )
                        raise ValueError(f"Unsupported LLM provider: {new_provider}")
                    
                    # Update the environment variable
                    os.environ["LLM_PROVIDER"] = new_provider
                    logger.info(f"LLM provider changed from {current_provider} to {new_provider}")
                    
                    # Create new LLM config with the new provider
                    global llm_config
                    from core.llm import LLMFactory
                    llm_config = LLMFactory.create_llm_config(new_provider)
                
                # Extract common settings
                prompt_routing = data.get("prompt_routing")
                enrich_schema = data.get("enrich_schema")
                
                # Extract provider-specific settings
                if new_provider == "ollama":
                    settings = {
                        "host": data.get("ollama_host"),
                        "model": data.get("ollama_model"),
                        "additional_kwargs": data.get("additional_kwargs"),
                        "prompt_routing": prompt_routing,
                        "enrich_schema": enrich_schema
                    }
                elif new_provider == "google":
                    settings = {
                        "model": data.get("model"),
                        "api_key": data.get("api_key"),
                        "temperature": data.get("temperature"),
                        "max_tokens": data.get("max_tokens"),
                        "thinking_budget": data.get("thinking_budget"),
                        "prompt_routing": prompt_routing,
                        "enrich_schema": enrich_schema
                    }
                
                # Validate that at least one setting is provided
                if all(v is None for v in settings.values()):
                    logger.warning("No settings provided in request")
                    trace.update(
                        status="failed",
                        metadata={"error": "At least one setting must be provided"}
                    )
                    return jsonify({"error": "At least one setting must be provided"}), 400
                
                # Update settings using the centralized LLM config
                llm_config.update_settings(**settings)
                
                # Reinitialize workflows with new LLM
                from core.templates import text2sql_prompt_routing
                TEXT_TO_SQL_PROMPT_TMPL = text2sql_prompt_routing(llm_config.settings["prompt_routing"])
                
                global workflow, schema_workflow, baseline_workflow
                workflow.llm = llm_config.get_llm()
                schema_workflow.llm = llm_config.get_llm()
                baseline_workflow.llm = llm_config.get_llm()
                
                logger.info("LLM settings updated successfully")
                
                trace.update(
                    status="success",
                    metadata={
                        "provider": new_provider,
                        "updated_settings": settings
                    }
                )
                
                return ResponseWrapper.success({
                    "message": "Settings updated successfully",
                    "current_settings": llm_config.get_settings(),
                    "provider": new_provider
                })
            except Exception as e:
                logger.error(f"Error updating settings: {str(e)}", exc_info=True)
                if 'trace' in locals():
                    trace.update(
                        status="failed",
                        metadata={"error": str(e)}
                    )
                raise AppException(str(e), 500)

    @api.route('/health-check')
    class HealthCheck(Resource):
        @api.doc('health_check',
            responses={
                200: 'Success'
            }
        )
        def get(self):
            """Health check endpoint"""
            logger.debug("Health check request received")
            return jsonify({"status": "ok"})   

    @api.route('/suggest-questions')
    class SuggestQuestions(Resource):
        @api.expect(question_request_model)
        @api.doc('suggest_questions',
            responses={
                200: 'Success',
                400: 'Bad Request - Missing or invalid parameters',
                500: 'Internal Server Error'
            }
        )
        @async_route
        async def post(self):
            """Generate insightful questions based on database schema"""
            logger.info("Received request to /suggest-questions endpoint")
            try:
                data = request.json
                top_k = data.get("top_k", 5)  # Default to 5 questi
                tables = data.get("tables", [])
                database_description = data.get("database_description", None)
                # Create a Langfuse trace for this query
                trace = observability_service.langfuse.trace(
                    name="question_suggestions",
                    metadata={
                        "top_k": top_k,
                        "tables": tables
                    }
                )

                if not tables:
                    logger.warning("Missing required tables in request")
                    trace.update(
                        status="failed",
                        metadata={"error": "Missing 'tables'"}
                    )
                    return jsonify({"error": "Missing 'tables'"}), 400
                

                # Create a span for schema retrieval
                schema_span = observability_service.langfuse.span(
                    name="schema_retrieval",
                    parent_id=trace.id
                )

                logger.info("Retrieving schema from database")
                table_details = tables
                
                # Enrich schema with additional information
                logger.info(f"Retrieved schema with {len(table_details)} tables")
                
                schema_span.update(
                    metadata={
                        "table_count": len(table_details),
                    }
                )

                # Trace the query processing using LlamaIndex instrumentor
                logger.info("Executing workflow")
                with observability_service.llama_index_instrumentor.observe(trace_id=trace.id):
                    from core.events import QuestionSuggestionEvent
                    response = await question_workflow.run(
                        table_details=table_details,
                        top_k=top_k
                    )
                
                # Update trace with result status
                if response and "suggestions" in response:
                    trace.update(
                        status="success",
                        metadata={
                            "question_count": len(response["suggestions"])
                        }
                    )
                else:
                    trace.update(
                        status="failed",
                        metadata={"error": "Failed to generate question suggestions"}
                    )
                
                logger.info("Workflow completed successfully")
                
                return ResponseWrapper.success(response)

            except Exception as e:
                logger.error(f"Error generating question suggestions: {str(e)}", exc_info=True)
                if 'trace' in locals():
                    trace.update(
                        status="failed",
                        metadata={"error": str(e)}
                    )
                raise AppException(str(e), 500)
            
    return api 