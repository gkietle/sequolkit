from flask_restx import fields

# Define models for request/response documentation
def create_api_models(api):
    """Create and return Flask-RESTX models for API documentation"""
    
    connection_payload_model = api.model('ConnectionPayload', {
        'url': fields.String(required=False, description='Database host (required for postgresql, mysql)'),
        'username': fields.String(required=False, description='Database username (required for postgresql, mysql)'),
        'password': fields.String(required=False, description='Database password (required for postgresql, mysql)'),
        'file': fields.String(required=False, description='Base64 encoded SQLite file (required for sqlite file upload)'),
        'dbType': fields.String(required=True, description='Database type (postgresql, mysql, sqlite)'),
        'schema_enrich_info': fields.Raw(required=False, description='Schema enrichment information')
    })

    query_request_model = api.model('QueryRequest', {
        'query': fields.String(required=True, description='Natural language query'),
        'connection_payload': fields.Nested(connection_payload_model, required=True)
    })

    schema_enrich_request_model = api.model('SchemaEnrichRequest', {
        'connection_payload': fields.Nested(connection_payload_model, required=True)
    })
    
    question_request_model = api.model('QuestionRequest', {
        'top_k': fields.Integer(required=False, description='Number of question suggestions to generate', default=5),
        'tables': fields.List(fields.String, required=True, description='List of table names')
    })

    settings_model = api.model('Settings', {
        'provider': fields.String(required=False, description='LLM provider (ollama or google)'),
        'ollama_host': fields.String(required=False, description='Ollama host URL'),
        'ollama_model': fields.String(required=False, description='Ollama model name'),
        'additional_kwargs': fields.Raw(required=False, description='Additional Ollama parameters'),
        'model': fields.String(required=False, description='Google model name'),
        'api_key': fields.String(required=False, description='Google API key'),
        'temperature': fields.Float(required=False, description='Model temperature'),
        'max_tokens': fields.Integer(required=False, description='Maximum tokens'),
        'thinking_budget': fields.Integer(required=False, description='Thinking budget for Google Gemini'),
        'prompt_routing': fields.Integer(required=False, description='Prompt routing setting (0 or 1)'),
        'enrich_schema': fields.Boolean(required=False, description='Schema enrichment setting')
    })
    
    return {
        'connection_payload_model': connection_payload_model,
        'query_request_model': query_request_model,
        'question_request_model': question_request_model,
        'settings_model': settings_model,
        'schema_enrich_request_model': schema_enrich_request_model
    } 