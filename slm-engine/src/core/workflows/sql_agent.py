from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from core.utils import (
    extract_tables_from_sql,
    schema_parser,
    extract_sql_query,
    is_valid_sql_query
)
from core.events import (
    TableRetrieveEvent,
    TextToSQLEvent,
    SQLValidatorEvent,
    ExecuteSQLEvent,
    SQLReflectionEvent,
)
from core.models import (
    ListOfRelevantTables,
    SQLQuery,
    TranslatedQuery
)
from core.services import execute_sql, llm_chat, llm_chat_with_pydantic, get_sample_data_improved   
from response.log_manager import (
    log_step_start,
    log_step_end,
    log_success,
    log_error,
    log_warning,
    log_llm_operation,
    log_prompt
)
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from exceptions.app_exception import AppException
from config.app_config import app_config
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

class SQLAgentWorkflow(Workflow):
    """SQLAgent Workflow for text-to-SQL conversion with table retrieval and error handling."""

    def __init__(
        self,
        text2sql_prompt: str,
        llm: Ollama | GoogleGenAI,
        *args, **kwargs
    ) -> None:
        """Initialize the SQLAgent Workflow."""
        super().__init__(*args, **kwargs)
        self.text2sql_prompt = text2sql_prompt
        self.num_tables_threshold = 0  # Configurable threshold
        self.max_sql_retries = 3  # Reduced from 5 to avoid infinite loops
        self.llm = llm
        self._timeout = 300.0

    def _normalize_table_name(self, table_name: str) -> str:
        """Normalize table name for consistent comparison."""
        return table_name.lower().strip()

    def _find_tables_by_names(self, table_names: List[str], all_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find table details by names efficiently."""
        normalized_names = {self._normalize_table_name(name): name for name in table_names}
        table_map = {self._normalize_table_name(table['tableIdentifier']): table for table in all_tables}
        
        selected_tables = []
        missing_tables = []
        
        for normalized_name, original_name in normalized_names.items():
            if normalized_name in table_map:
                selected_tables.append(table_map[normalized_name])
            else:
                missing_tables.append(original_name)
        
        if missing_tables:
            log_warning("TABLE_LOOKUP", f"Tables not found: {missing_tables}")
        
        return selected_tables

    def _get_dialect(self, connection_payload: Dict[str, Any]) -> str:
        """Get normalized database dialect."""
        db_type = connection_payload.get("dbType", "").lower()
        dialect_mapping = {
            'postgresql': 'postgres',
            'mysql': 'mysql',
            'sqlite': 'sqlite'
        }
        return dialect_mapping.get(db_type, db_type)

    def _normalize_sql_formatting(self, sql: str) -> str:
        """Safely normalize SQL query formatting while preserving string literals and spacing."""
        # First, protect string literals by temporarily replacing them
        literals = []
        def replace_literal(match):
            literals.append(match.group(0))
            return f"__STRING_LITERAL_{len(literals)-1}__"
        
        # Temporarily replace string literals
        sql_protected = re.sub(r"'[^']*'|\"[^\"]*\"", replace_literal, sql.strip())
        
        # Normalize whitespace (newlines and multiple spaces)
        sql_normalized = re.sub(r'\s+', ' ', sql_protected)
        
        # Restore string literals
        for i, literal in enumerate(literals):
            sql_normalized = sql_normalized.replace(f"__STRING_LITERAL_{i}__", literal)
        
        return sql_normalized.strip()

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> TableRetrieveEvent | TextToSQLEvent:
        """Start the SQLAgent Workflow."""
        start_time = log_step_start("START", query=ev.query, connection_type=ev.connection_payload.get('dbType', 'unknown'))
        
        # Store initial context
        await context.set("table_details", ev.table_details)
        await context.set("connection_payload", ev.connection_payload)
        await context.set("database_description", ev.database_description or "")
        await context.set("user_query", ev.query)
        await context.set("session_information", ev.session_information)
        await context.set("retry_count", 0)

        table_count = len(ev.table_details)
        table_identifiers = [table['tableIdentifier'] for table in ev.table_details]
        
        log_step_start("START", total_tables=table_count, table_names=table_identifiers)

        # Decision: use table retrieval or process all tables
        if table_count > self.num_tables_threshold:
            log_step_start("START", message=f"Table count ({table_count}) exceeds threshold ({self.num_tables_threshold}). Using table retrieval.")
            log_step_end("START", start_time)
            return TableRetrieveEvent(tables=ev.table_details, query=ev.query)
        else:
            log_step_start("START", message=f"Table count ({table_count}) within threshold. Processing all tables.")
            await context.set("relevant_tables", table_identifiers)
            log_step_end("START", start_time)
            return TextToSQLEvent(relevant_tables=table_identifiers, query=ev.query)

    @step
    async def Retrieve_relevant_tables(self, context: Context, ev: TableRetrieveEvent) -> TextToSQLEvent | StopEvent:
        """Retrieve relevant tables using LLM."""
        start_time = log_step_start("RETRIEVE", query=ev.query)
        
        try:
            table_details = await context.get("table_details")
            database_description = await context.get("database_description")

            # Load templates
            from core.templates import TABLE_RETRIEVAL_SKELETON, QUERY_REFINEMENT_SKELETON
            schema = schema_parser(table_details, "Simple", include_sample_data=False)

            # Query translation step
            log_step_start("RETRIEVE", message="Translating query to English")
            translated_query = llm_chat_with_pydantic(
                llm=self.llm,
                prompt=PromptTemplate(QUERY_REFINEMENT_SKELETON.format(
                    user_question=ev.query,
                    database_schema=schema
                )),
                pydantic_model=TranslatedQuery
            )

            query = translated_query.translated_query
            await context.set("user_query", query)
            log_success("RETRIEVE", f"Translated query: {query}")

            # Table retrieval step
            table_retrieval_prompt = TABLE_RETRIEVAL_SKELETON.format(
                database_description=database_description,
                query=query,
                schema=schema
            )
            
            log_prompt(table_retrieval_prompt, "RETRIEVE")
            
            log_step_start("RETRIEVE", message="Querying LLM for relevant tables")
            llm_start_time = datetime.now()
            
            chat_response = llm_chat_with_pydantic(
                llm=self.llm,
                prompt=PromptTemplate(table_retrieval_prompt),
                pydantic_model=ListOfRelevantTables
            )
            
            relevant_tables = chat_response.relevant_tables
            log_llm_operation("RETRIEVE", "LLM response", llm_start_time, chat_response)

            if not relevant_tables:
                log_error("RETRIEVE", "No relevant tables found")
                return StopEvent(result="Cannot find any relevant tables in the database. Please try again with a different question.")

            log_success("RETRIEVE", f"Found {len(relevant_tables)} relevant tables: {relevant_tables}")
            await context.set("relevant_tables", relevant_tables)
            
            log_step_end("RETRIEVE", start_time)
            return TextToSQLEvent(relevant_tables=relevant_tables, query=query)
            
        except Exception as e:
            log_error("RETRIEVE", f"Error during table retrieval: {str(e)}")
            log_step_end("RETRIEVE", start_time)
            return StopEvent(result=f"Error during table retrieval: {str(e)}")

    @step
    async def Generate_sql(self, context: Context, ev: TextToSQLEvent) -> SQLValidatorEvent | StopEvent:
        """Generate SQL based on the user query and table schema."""
        start_time = log_step_start("GENERATE", query=ev.query, tables=ev.relevant_tables)
        
        try:
            # Get context data
            table_details = await context.get("table_details")
            connection_payload = await context.get("connection_payload")
            database_description = await context.get("database_description")
            retry_count = await context.get("retry_count")
            
            dialect = self._get_dialect(connection_payload).upper()
            log_step_start("GENERATE", dialect=dialect)
            
            # Find selected tables efficiently
            selected_tables = self._find_tables_by_names(ev.relevant_tables, table_details)
            
            if not selected_tables:
                log_error("GENERATE", "No valid tables found for SQL generation")
                return StopEvent(result="No valid tables found for the query.")
            
            # Add sample data if not in privacy mode
            if not app_config.PRIVACY_MODE:
                for table in selected_tables:
                    try:
                        table["sample_data"] = get_sample_data_improved(
                            connection_payload=connection_payload, 
                            table_details=table
                        )
                    except Exception as e:
                        log_warning("GENERATE", f"Could not get sample data for table {table['tableIdentifier']}: {e}")
                        table["sample_data"] = []
            
            await context.set("selected_tables", selected_tables)
            
            log_step_start("GENERATE", message=f"Generating SQL for {len(selected_tables)} tables")
            table_schemas = schema_parser(selected_tables, "DDL", include_sample_data=not app_config.PRIVACY_MODE)
            
            # Format prompt
            text_to_sql_prompt = self.text2sql_prompt.format(
                user_question=ev.query,
                table_schemas=table_schemas,
                database_description=database_description,
                dialect=dialect
            )
        
            log_prompt(text_to_sql_prompt, "GENERATE")
            
            # Generate SQL
            log_step_start("GENERATE", message="Querying LLM for SQL generation")
            llm_start_time = datetime.now()
            
            chat_response = llm_chat_with_pydantic(
                llm=self.llm,
                prompt=PromptTemplate(text_to_sql_prompt),
                pydantic_model=SQLQuery
            )
            
            log_llm_operation("GENERATE", "LLM response", llm_start_time, chat_response)
            # Normalize SQL query formatting while preserving string literals
            sql_query = self._normalize_sql_formatting(chat_response.sql_query)

            # Basic validation
            if not sql_query or "SELECT" not in sql_query.upper():
                log_error("GENERATE", "Generated SQL is invalid or not a SELECT statement")
                return StopEvent(result="Could not generate a valid SQL query. Please try again with a different question.")
            
            log_success("GENERATE", f"Generated SQL: {sql_query}")
            log_step_end("GENERATE", start_time)
            
            return SQLValidatorEvent(sql_query=sql_query, retry_count=retry_count)
            
        except Exception as e:
            log_error("GENERATE", f"Error during SQL generation: {str(e)}")
            log_step_end("GENERATE", start_time)
            return StopEvent(result=f"Error during SQL generation: {str(e)}")
    
    @step
    async def Validate_SQL(self, context: Context, ev: SQLValidatorEvent) -> ExecuteSQLEvent | StopEvent | SQLReflectionEvent | TextToSQLEvent:
        """Validate SQL query syntax and table references."""
        start_time = log_step_start("VALIDATE", sql=ev.sql_query)
        
        try:
            connection_payload = await context.get("connection_payload")
            table_details = await context.get("table_details")
            user_query = await context.get("user_query")
            retry_count = ev.retry_count
            
            dialect = self._get_dialect(connection_payload)
            log_step_start("VALIDATE", dialect=dialect)

            # Syntax validation
            is_valid_sql, syntax_error = is_valid_sql_query(ev.sql_query, dialect)
            if not is_valid_sql:
                log_error("VALIDATE", f"SQL syntax error: {syntax_error}")
                retry_count += 1
                await context.set("retry_count", retry_count)
                return SQLReflectionEvent(sql_query=ev.sql_query, error=str(syntax_error), retry_count=retry_count)

            # Table reference validation
            tables_in_sql = extract_tables_from_sql(ev.sql_query, dialect)
            valid_table_names = {self._normalize_table_name(table["tableIdentifier"]) for table in table_details}
            relevant_tables = await context.get("relevant_tables")
            
            log_step_start("VALIDATE", tables_in_sql=tables_in_sql, valid_tables=list(valid_table_names))

            # Check if all SQL tables exist in schema
            invalid_tables = [table for table in tables_in_sql 
                            if self._normalize_table_name(table) not in valid_table_names]
            
            if invalid_tables:
                log_error("VALIDATE", f"SQL references non-existent tables: {invalid_tables}")
                return StopEvent(result="SQL query references tables that don't exist in the database schema.")

            # Check if SQL uses tables not in relevant_tables (expand scope if needed)
            if relevant_tables:
                relevant_normalized = {self._normalize_table_name(table) for table in relevant_tables}
                sql_tables_normalized = {self._normalize_table_name(table) for table in tables_in_sql}
                
                missing_from_relevant = sql_tables_normalized - relevant_normalized
                if missing_from_relevant:
                    log_warning("VALIDATE", f"SQL uses tables not in relevant set: {missing_from_relevant}")
                    # Expand relevant tables to include all tables used in SQL
                    expanded_relevant = list(set(relevant_tables + tables_in_sql))
                    await context.set("relevant_tables", expanded_relevant)
                    log_success("VALIDATE", f"Expanded relevant tables to: {expanded_relevant}")
                    return TextToSQLEvent(relevant_tables=expanded_relevant, query=user_query)

            log_success("VALIDATE", "SQL query validation passed")
            log_step_end("VALIDATE", start_time)
            
            return ExecuteSQLEvent(sql_query=ev.sql_query)
            
        except Exception as e:
            log_error("VALIDATE", f"Error during SQL validation: {str(e)}")
            log_step_end("VALIDATE", start_time)
            return StopEvent(result=f"Error during SQL validation: {str(e)}")

    @step
    async def Execute_SQL(self, context: Context, ev: ExecuteSQLEvent) -> SQLReflectionEvent | StopEvent:
        """Execute SQL query against the database."""
        start_time = log_step_start("EXECUTE", sql=ev.sql_query)
        
        try:
            connection_payload = await context.get("connection_payload")
            retry_count = await context.get("retry_count")
            
            log_step_start("EXECUTE", connection_type=connection_payload.get('dbType', 'unknown'))
            
            # Execute query
            log_step_start("EXECUTE", message="Executing SQL query")
            exec_start_time = datetime.now()
            
            result = execute_sql(connection_payload, ev.sql_query)
            
            log_llm_operation("EXECUTE", "Database execution", exec_start_time)
            
            if result.get("error"):
                error_msg = result["error"]
                log_error("EXECUTE", f"SQL execution failed: {error_msg}")
                
                retry_count += 1
                await context.set("retry_count", retry_count)
                log_step_end("EXECUTE", start_time)
                
                return SQLReflectionEvent(sql_query=ev.sql_query, error=error_msg, retry_count=retry_count)
            
            # Success
            data = result.get("data", [])
            row_count = len(data) if isinstance(data, list) else 0
            
            log_success("EXECUTE", f"SQL executed successfully, returned {row_count} rows")
            log_step_end("EXECUTE", start_time)
            
            return StopEvent(result=ev.sql_query)
            
        except Exception as e:
            log_error("EXECUTE", f"Error during SQL execution: {str(e)}")
            log_step_end("EXECUTE", start_time)
            return StopEvent(result=f"Error during SQL execution: {str(e)}")
    
    @step
    async def Reflect_SQL(self, context: Context, ev: SQLReflectionEvent) -> SQLValidatorEvent | StopEvent:
        """Reflect on SQL errors and generate corrections."""
        start_time = log_step_start("REFLECT", attempt=f"{ev.retry_count}/{self.max_sql_retries}")
        
        # Check retry limit
        if ev.retry_count >= self.max_sql_retries:
            error_msg = f"Maximum retry attempts ({self.max_sql_retries}) reached. Last error: {ev.error}"
            log_error("REFLECT", error_msg)
            log_step_end("REFLECT", start_time)
            return StopEvent(result=error_msg)
        
        try:
            # Get context data
            connection_payload = await context.get("connection_payload")
            user_query = await context.get("user_query")
            table_details = await context.get("table_details")
            database_description = await context.get("database_description")
            relevant_tables = await context.get("relevant_tables")
            
            # Update retry count
            retry_count = ev.retry_count
            await context.set("retry_count", retry_count)
            
            log_step_start("REFLECT", sql_with_error=ev.sql_query, error=ev.error)
            
            # Get selected tables for context
            if not relevant_tables:
                relevant_tables = [table['tableIdentifier'] for table in table_details]
            
            selected_tables = self._find_tables_by_names(relevant_tables, table_details)
            
            if not selected_tables:
                log_error("REFLECT", "No valid tables found for reflection")
                return StopEvent(result="Could not find valid tables for SQL correction.")
            
            # Prepare schema for reflection
            table_schemas = schema_parser(selected_tables, "DDL", include_sample_data=not app_config.PRIVACY_MODE)
            
            # Load error reflection template
            from core.templates import SQL_ERROR_REFLECTION_SKELETON
            
            reflection_prompt = SQL_ERROR_REFLECTION_SKELETON.format(
                database_schema=table_schemas,
                database_description=database_description,
                sql_query=ev.sql_query,
                error_message=ev.error,
                dialect=connection_payload.get("dbType", "").upper(),
                user_query=user_query   
            )
            
            log_prompt(reflection_prompt, "REFLECT")
            
            # Get corrected SQL from LLM
            log_step_start("REFLECT", message="Querying LLM for SQL correction")
            llm_start_time = datetime.now()
            
            chat_response = llm_chat_with_pydantic(
                llm=self.llm,
                prompt=PromptTemplate(reflection_prompt),
                pydantic_model=SQLQuery
            )
            
            log_llm_operation("REFLECT", "LLM response", llm_start_time, chat_response)
            
            corrected_sql = chat_response.sql_query.strip()
            
            # Check if SQL was actually modified
            if corrected_sql == ev.sql_query:
                log_warning("REFLECT", "LLM returned identical SQL - no correction made")
                return StopEvent(result=f"Could not correct SQL error: {ev.error}")
            
            log_success("REFLECT", f"SQL corrected: {corrected_sql}")
            log_step_end("REFLECT", start_time)
            
            return SQLValidatorEvent(sql_query=corrected_sql, retry_count=retry_count)
            
        except Exception as e:
            log_error("REFLECT", f"Error during SQL reflection: {str(e)}")
            log_step_end("REFLECT", start_time)
            return StopEvent(result=f"Error during SQL reflection: {str(e)}")
