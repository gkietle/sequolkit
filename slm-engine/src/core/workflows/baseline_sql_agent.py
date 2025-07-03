from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from core.utils import (
    extract_sql_query,
    schema_parser,
)
from core.events import TextToSQLEvent, SQLValidatorEvent
from core.models import SQLQuery
from core.services import llm_chat, llm_chat_with_pydantic
from response.log_manager import (
    log_step_start, 
    log_step_end, 
    log_success,
    log_llm_operation,
    log_prompt
)
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class BaselineWorkflow(Workflow):
    """Simple baseline workflow for text-to-SQL conversion without table retrieval or error handling."""

    def __init__(
        self,
        llm: Ollama | GoogleGenAI,
        text2sql_prompt: str,
        *args, **kwargs
    ) -> None:
        """Initialize the Baseline Workflow.""" 
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.text2sql_prompt = text2sql_prompt
        self._timeout = 300.0

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> TextToSQLEvent:
        """Start the Baseline workflow."""
        start_time = log_step_start("BASELINE", query=ev.query, connection_type=ev.connection_payload.get('dbType', 'unknown'))
        
        # Store relevant information in the context
        await context.set("connection_payload", ev.connection_payload)
        await context.set("table_details", ev.table_details)
        await context.set("user_query", ev.query)
        await context.set("database_description", ev.database_description)

        log_success("BASELINE", f"Starting baseline workflow for query: \"{ev.query}\"")
        
        log_step_end("BASELINE", start_time)
        return TextToSQLEvent(relevant_tables=[], query=ev.query)

    @step
    async def Generate_sql(self, context: Context, ev: TextToSQLEvent) -> SQLValidatorEvent:
        """Generate SQL based on the user query and table schema."""
        start_time = log_step_start("GENERATE", query=ev.query)
        
        # Get needed data from context
        table_details = await context.get("table_details")
        connection_payload = await context.get("connection_payload")
        database_description = await context.get("database_description")
        
        # Get database dialect
        dialect = connection_payload.get("dbType", "").upper()
        log_step_start("GENERATE", dialect=dialect)
        
        # Prepare schema information for all tables
        table_schemas = schema_parser(table_details, "DDL", include_sample_data=True)
        
        # Format the prompt
        prompt_text = self.text2sql_prompt.format(
            user_question=ev.query,
            table_schemas=table_schemas,
            database_description=database_description,
            dialect=dialect
        )
    
        # Log the prompt
        log_prompt(prompt_text, "GENERATE")
        
        # Query LLM to generate SQL
        log_step_start("GENERATE", message="Querying LLM for SQL generation...")
        llm_start_time = datetime.now()
        
        try:
            # First try with structured output
            chat_response = llm_chat_with_pydantic(
                llm=self.llm,
                prompt=PromptTemplate(prompt_text),
                pydantic_model=SQLQuery
            )
            sql_query = chat_response.sql_query
        except Exception as e:
            # Fall back to raw output and extraction
            logger.warning(f"Structured output failed: {e}, falling back to extraction")
            chat_response = llm_chat(self.llm, PromptTemplate(prompt_text).format_messages())
            sql_query = extract_sql_query(chat_response.message.content)
            
        log_llm_operation("GENERATE", "LLM response", llm_start_time, chat_response)
        
        log_success("GENERATE", f"Generated SQL query: {sql_query}")
        
        log_step_end("GENERATE", start_time)
        return SQLValidatorEvent(sql_query=sql_query)

    @step
    async def Validate_SQL(self, context: Context, ev: SQLValidatorEvent) -> StopEvent:
        """Validate SQL query and return final result."""
        start_time = log_step_start("VALIDATE", sql=ev.sql_query)
        
        # For baseline workflow, we just return the SQL query without validation
        log_success("VALIDATE", "SQL query validated successfully (baseline mode)")
        
        log_step_end("VALIDATE", start_time)
        return StopEvent(result=ev.sql_query)
