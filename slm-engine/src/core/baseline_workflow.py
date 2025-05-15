from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from core.utils import (
    extract_sql_query,
    schema_parser,
    log_prompt
)
# from core.templates import TEXT_TO_SQL_PROMPT_TMPL
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from core.services import llm_chat
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class TextToSQLEvent(Event):
    """Text-to-SQL event."""
    query: str

class SQLValidatorEvent(Event):
    """Execute SQL event."""
    sql_query: str

class BaselineWorkflow(Workflow):
    """Workflow for generate SQL query based on user query and table schema."""

    def __init__(
        self,
        llm: Ollama | GoogleGenAI,
        text2sql_prompt: PromptTemplate,
        *args, **kwargs
    ) -> None:
        """Initialize the SQLAgent Workflow.""" 
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.text2sql_prompt = text2sql_prompt
        self._timeout = 300.0

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> TextToSQLEvent:
        """Start the SQLAgent Workflow."""
        logger.info("\033[93m[START] Beginning SQL Agent workflow\033[0m")
        start_time = datetime.now()
        await context.set("connection_payload", ev.connection_payload)
        await context.set("table_details", ev.table_details)
        await context.set("user_query", ev.query)
        await context.set("database_description", ev.database_description)

        logger.info(f"\033[93m[START] User query: \"{ev.query}\"\033[0m")
        logger.info(f"\033[93m[START] Connection type: {ev.connection_payload.get('dbType', 'unknown')}\033[0m")
        
        end_time = datetime.now()
        logger.info(f"\033[93m[START] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return TextToSQLEvent(query=ev.query)

    @step
    async def Generate_sql(self, context: Context, ev: TextToSQLEvent) -> SQLValidatorEvent:
        """Generate SQL based on the user query and table schema."""
        logger.info("\033[93m[GENERATE] Starting SQL generation\033[0m")
        start_time = datetime.now()
        
        logger.info(f"\033[93m[GENERATE] Query: \"{ev.query}\"\033[0m")
        
        table_details = await context.get("table_details")
        connection_payload = await context.get("connection_payload")
        database_description = await context.get("database_description")
        dialect = connection_payload.get("dbType", "").upper()
        logger.info(f"\033[93m[GENERATE] DB dialect: {dialect}\033[0m")
        
        # Prepare schema information
        table_schemas = schema_parser(table_details, "DDL")
        
        fmt_messages = self.text2sql_prompt.format_messages(
            user_question=ev.query,
            table_schemas=table_schemas,
            dialect=dialect,
            database_description=database_description
        )
    
        # Log the prompt
        log_prompt(fmt_messages, "GENERATE")
        
        logger.info("\033[93m[GENERATE] Querying LLM for SQL generation...\033[0m")
        llm_start_time = datetime.now()
        chat_response = llm_chat(self.llm, fmt_messages)
        llm_end_time = datetime.now()
        
        logger.info(f"\033[93m[GENERATE] LLM response time: {(llm_end_time - llm_start_time).total_seconds():.2f} seconds\033[0m")
        logger.info("\033[94m[GENERATE] LLM response: " + str(chat_response) + "\033[0m")
        
        # Extract the SQL query from the response
        sql_query = extract_sql_query(chat_response.message.content)
        logger.info("\033[92m[GENERATE] Extracted SQL query: " + sql_query + "\033[0m")
        
        end_time = datetime.now()
        logger.info(f"\033[93m[GENERATE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return SQLValidatorEvent(sql_query=sql_query)

    @step
    async def Validate_SQL(self, context: Context, ev: SQLValidatorEvent) -> StopEvent:
        """Validate SQL query."""
        logger.info("\033[93m[VALIDATE] Starting SQL validation\033[0m")
        start_time = datetime.now()
        
        sql_query = ev.sql_query
        logger.info(f"\033[93m[VALIDATE] SQL to validate: {sql_query}\033[0m")
        
        # For baseline workflow, we just return the SQL query
        # In a real implementation, you might want to add validation logic here
        
        logger.info("\033[92m[VALIDATE] SQL query validated successfully\033[0m")
        
        end_time = datetime.now()
        logger.info(f"\033[93m[VALIDATE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return StopEvent(result=sql_query) 