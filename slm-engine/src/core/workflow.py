from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from core.utils import (
    extract_table_list,
    extract_tables_from_sql,
    schema_parser,
    schema_clustering,
    show_prompt,
    log_prompt,
    extract_sql_query,
    parse_schema_enrichment
)
from core.templates import (
    TABLE_EXTRACTION_TMPL,
    DATABASE_DESCRIPTION_TMPL,
    SQL_ERROR_REFLECTION_TMPL,
    SCHEMA_ENRICHMENT_TMPL
)
from exceptions.app_exception import AppException
from enums.response_enum import ResponseEnum
from core.services import execute_sql, get_sample_data
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
import re
import logging
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class TableRetrieveEvent(Event):
    """Result of running table retrieval."""

    tables: list[object]
    query: str

class TextToSQLEvent(Event):
    """Text-to-SQL event."""
    relevant_tables: list[str]
    query: str

class SQLValidatorEvent(Event):
    """Execute SQL event."""
    sql_query: str
    retry_count: int = 0

class ExecuteSQLEvent(Event):
    """Execute SQL event."""
    sql_query: str

class SQLReflectionEvent(Event):
    """Execute SQL event."""
    sql_query: str
    error: str
    retry_count: int = 0


class SchemaEnrichmentEvent(Event):
    """Event after analyzing table relationships."""
    database_description: str
    clusters: list[list]

class SQLAgentWorkflow(Workflow):
    """SQLAgent Workflow."""

    def __init__(
        self,
        text2sql_prompt: PromptTemplate,
        table_retrieval_prompt: PromptTemplate,
        llm: Ollama,
        *args, **kwargs
    ) -> None:
        """Initialize the SQLAgent Workflow."""
        super().__init__(*args, **kwargs)
        self.text2sql_prompt = text2sql_prompt
        self.table_retrieval_prompt = table_retrieval_prompt
        self.num_tables_threshold = 3
        self.max_sql_retries = 3
        self.llm = llm
        self._timeout = 300.0

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> TableRetrieveEvent | TextToSQLEvent:
        """Start the SQLAgent Workflow."""
        logger.info("\033[93m[START] Beginning SQL Agent workflow\033[0m")
        start_time = datetime.now()
        await context.set("table_details", ev.table_details)
        await context.set("connection_payload", ev.connection_payload)
        await context.set("database_description", ev.database_description)
        await context.set("user_query", ev.query)

        logger.info(f"\033[93m[START] User query: \"{ev.query}\"\033[0m")
        logger.info(f"\033[93m[START] Connection type: {ev.connection_payload.get('dbType', 'unknown')}\033[0m")
        table_details = ev.table_details
        tables_brief = []
        for table in table_details:
            tables_brief.append({
                "tableIdentifier": table.get("tableIdentifier", ""),
                "tableDescription": table.get("tableDescription", "")
            })
        logger.info(f"\033[93m[START] Total tables available: {len(tables_brief)}\033[0m")

        if len(tables_brief) > self.num_tables_threshold:
            logger.info(
                f"\033[93m[START] Exceeds threshold of {self.num_tables_threshold} tables. Will use table retrieval step.\033[0m")
            logger.info("\033[94m[START] All tables: " + json.dumps(tables_brief) + "\033[0m")
            
            end_time = datetime.now()
            logger.info(f"\033[93m[START] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
            return TableRetrieveEvent(tables=tables_brief, query=ev.query)
        else:
            logger.info(f"\033[93m[START] Under threshold ({len(tables_brief)} <= {self.num_tables_threshold}). Using all tables.\033[0m")
            logger.info("\033[94m[START] Using tables: " + json.dumps(tables_brief) + "\033[0m")
            
            # Set relevant_tables here for the direct path
            await context.set("relevant_tables", [table["tableIdentifier"] for table in tables_brief])
            
            end_time = datetime.now()
            logger.info(f"\033[93m[START] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
            return TextToSQLEvent(relevant_tables=[table["tableIdentifier"] for table in tables_brief], query=ev.query)

    @step
    async def Retrieve_relevant_tables(self, context: Context, ev: TableRetrieveEvent) -> TextToSQLEvent:
        """Retrieve table context."""
        logger.info("\033[93m[RETRIEVE] Starting table relevance filtering\033[0m")
        start_time = datetime.now()
        
        logger.info(f"\033[93m[RETRIEVE] Query: \"{ev.query}\"\033[0m")
        logger.info(f"\033[93m[RETRIEVE] Available tables: {len(ev.tables)}\033[0m")
        
        tables = []
        for table in ev.tables:
            if table["tableDescription"] is not None:
                formatted_table = f"- {table['tableIdentifier']} ({table['tableDescription']})"
            else:
                formatted_table = f"- {table['tableDescription']}"
            tables.append(formatted_table)

        database_description = await context.get("database_description")

        fmt_messages = self.table_retrieval_prompt.format_messages(
            database_description=database_description,
            query_str=ev.query,
            table_names="\n".join(tables)
        )
        
        # Log the prompt
        log_prompt(fmt_messages, "RETRIEVE")
        
        logger.info("\033[93m[RETRIEVE] Querying LLM for relevant tables...\033[0m")
        llm_start_time = datetime.now()
        chat_response = self.llm.chat(fmt_messages)
        llm_end_time = datetime.now()
        
        logger.info(f"\033[93m[RETRIEVE] LLM response time: {(llm_end_time - llm_start_time).total_seconds():.2f} seconds\033[0m")
        logger.info("\033[94m[RETRIEVE] LLM response: " + str(chat_response) + "\033[0m")
        
        relevant_tables = extract_table_list(chat_response)
        logger.info(f"\033[93m[RETRIEVE] Extracted {len(relevant_tables)} relevant tables\033[0m")
        logger.info("\033[94m[RETRIEVE] Relevant tables: " + json.dumps(relevant_tables) + "\033[0m")
        
        await context.set("relevant_tables", relevant_tables)
        
        end_time = datetime.now()
        logger.info(f"\033[93m[RETRIEVE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return TextToSQLEvent(relevant_tables=relevant_tables, query=ev.query)

    @step
    async def Generate_sql(self, context: Context, ev: TextToSQLEvent) -> SQLValidatorEvent:
        """Generate SQL based on the user query and table schema."""
        logger.info("\033[93m[GENERATE] Starting SQL generation\033[0m")
        start_time = datetime.now()
        
        logger.info(f"\033[93m[GENERATE] Query: \"{ev.query}\"\033[0m")
        logger.info(f"\033[93m[GENERATE] Using tables: {json.dumps(ev.relevant_tables)}\033[0m")
        
        table_details = await context.get("table_details")
        connection_payload = await context.get("connection_payload")
        database_description = await context.get("database_description")
        dialect = connection_payload.get("dbType", "").upper()
        logger.info(f"\033[93m[GENERATE] DB dialect: {dialect}\033[0m")
        
        selected_tables = []
        for table_name in ev.relevant_tables:
            for table in table_details:
                if table_name.lower().strip() == table["tableIdentifier"].lower().strip():
                    selected_tables.append(table)
        
        logger.info(f"\033[93m[GENERATE] Preparing schema information for {len(selected_tables)} tables\033[0m")

        logger.info(f"\033[93m[GENERATE] Retrieved {selected_tables}\033[0m")
        print(selected_tables)
        table_schemas = schema_parser(selected_tables, "DDL", include_sample_data=True)
        
        fmt_messages = self.text2sql_prompt.format_messages(
            user_question=ev.query,
            table_schemas=table_schemas,
            database_description=database_description,
            dialect=dialect
        )
    
        # Log the prompt
        log_prompt(fmt_messages, "GENERATE")
        
        logger.info("\033[93m[GENERATE] Querying LLM for SQL generation...\033[0m")
        llm_start_time = datetime.now()
        chat_response = self.llm.chat(fmt_messages)
        llm_end_time = datetime.now()
        
        logger.info(f"\033[93m[GENERATE] LLM response time: {(llm_end_time - llm_start_time).total_seconds():.2f} seconds\033[0m")
        logger.info("\033[94m[GENERATE] LLM response: " + str(chat_response) + "\033[0m")
        
        # Extract the SQL query from the response
        sql_query = extract_sql_query(chat_response.message.content)
        logger.info("\033[92m[GENERATE] Extracted SQL query: " + sql_query + "\033[0m")
        
        end_time = datetime.now()
        logger.info(f"\033[93m[GENERATE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return SQLValidatorEvent(sql_query=sql_query, retry_count=0)
    
    @step
    async def Validate_SQL(self, context: Context, ev: SQLValidatorEvent) -> ExecuteSQLEvent:
        """Validate SQL query."""
        logger.info("\033[93m[VALIDATE] Starting SQL validation\033[0m")
        start_time = datetime.now()
        
        sql_query = ev.sql_query
        logger.info(f"\033[93m[VALIDATE] SQL to validate: {sql_query}\033[0m")
        
        table_details = await context.get("table_details")
        
        # Extract tables from the SQL query
        logger.info("\033[93m[VALIDATE] Extracting tables from SQL query\033[0m")
        table_sql = extract_tables_from_sql(sql_query)
        logger.info("\033[94m[VALIDATE] Tables found in SQL: " + json.dumps(table_sql) + "\033[0m")
        
        # Get all valid table identifiers
        valid_tables = [table["tableIdentifier"].lower().strip() for table in table_details]
        logger.info("\033[94m[VALIDATE] Valid tables: " + json.dumps(valid_tables) + "\033[0m")

        # Check if all the tables in the SQL are in the table_details
        valid = True
        for table in table_sql:
            if table.lower().strip() not in valid_tables:
                logger.error(f"\033[91m[VALIDATE] Invalid table found: '{table}'\033[0m")
                valid = False
                break
        
        if not valid:
            logger.error("\033[91m[VALIDATE] SQL references tables not in provided schema\033[0m")
            raise AppException(ResponseEnum.NOT_RELEVANT_QUERY)
        
        logger.info("\033[92m[VALIDATE] SQL query validated successfully\033[0m")
        
        end_time = datetime.now()
        logger.info(f"\033[93m[VALIDATE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return ExecuteSQLEvent(sql_query=sql_query)

    @step
    async def Excute_SQL(self, context: Context, ev: ExecuteSQLEvent) -> SQLReflectionEvent | StopEvent:
        """ Execute SQL query."""
        logger.info("\033[93m[EXECUTE] Starting SQL execution\033[0m")
        start_time = datetime.now()
        
        sql_query = ev.sql_query
        logger.info(f"\033[93m[EXECUTE] Executing SQL: {sql_query}\033[0m")
        
        connection_payload = await context.get("connection_payload")
        logger.info(f"\033[93m[EXECUTE] Using connection type: {connection_payload.get('dbType', 'unknown')}\033[0m")
        
        logger.info("\033[93m[EXECUTE] Sending query to database...\033[0m")
        exec_start_time = datetime.now()
        payload = execute_sql(connection_payload, sql_query)
        exec_end_time = datetime.now()
        
        logger.info(f"\033[93m[EXECUTE] Database query time: {(exec_end_time - exec_start_time).total_seconds():.2f} seconds\033[0m")
        
        if payload.get("error"):
            error_msg = payload["error"]
            logger.error(f"\033[91m[EXECUTE] SQL execution failed with error: {error_msg}\033[0m")
            
            # Initialize the retry count for a new error
            end_time = datetime.now()
            logger.info(f"\033[93m[EXECUTE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
            return SQLReflectionEvent(sql_query=sql_query, error=error_msg, retry_count=0)
        else:
            logger.info("\033[92m[EXECUTE] SQL executed successfully\033[0m")
            
            # Log result summary (count of rows, etc.)
            if "data" in payload and isinstance(payload["data"], list):
                logger.info(f"\033[93m[EXECUTE] Query returned {len(payload['data'])} rows\033[0m")
            
            end_time = datetime.now()
            logger.info(f"\033[93m[EXECUTE] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
            return StopEvent(result=sql_query)
    
    @step
    async def Reflect_SQL(self, context: Context, ev: SQLReflectionEvent) -> SQLValidatorEvent | StopEvent:
        """Reflect on the SQL query with retry limit."""
        logger.info("\033[93m[REFLECT] Starting SQL reflection\033[0m")
        start_time = datetime.now()
        
        # Check retry count
        if ev.retry_count >= self.max_sql_retries:
            logger.error(f"\033[91m[REFLECT] Reached maximum retry attempts ({self.max_sql_retries}). Stopping workflow.\033[0m")
            error_msg = f"Failed to fix SQL query after {self.max_sql_retries} attempts. Last error: {ev.error}"
            logger.error(f"\033[91m[REFLECT] {error_msg}\033[0m")
            
            end_time = datetime.now()
            logger.info(f"\033[93m[REFLECT] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
            raise AppException(error_msg, 4)
        
        # Increment retry count
        retry_count = ev.retry_count + 1
        logger.info(f"\033[93m[REFLECT] Attempt {retry_count}/{self.max_sql_retries}\033[0m")
        
        sql_query = ev.sql_query
        error = ev.error
        logger.info(f"\033[93m[REFLECT] SQL with error: {sql_query}\033[0m")
        logger.info(f"\033[91m[REFLECT] Error message: {error}\033[0m")
        
        connection_payload = await context.get("connection_payload")
        user_query = await context.get("user_query")
        table_details = await context.get("table_details")
        database_description = await context.get("database_description")

        # Safely get relevant tables
        try:
            relevant_tables = await context.get("relevant_tables")
            logger.info(f"\033[93m[REFLECT] Using cached relevant tables: {json.dumps(relevant_tables)}\033[0m")
        except ValueError:
            # Use all tables if relevant_tables wasn't set
            relevant_tables = [table["tableIdentifier"] for table in table_details]
            logger.info(f"\033[93m[REFLECT] No cached relevant tables. Using all tables: {json.dumps(relevant_tables)}\033[0m")

        # Prepare selected tables
        selected_tables = []
        for table_name in relevant_tables:
            for table in table_details:
                if table_name.lower().strip() == table["tableIdentifier"].lower().strip():
                    selected_tables.append(table)
        
        logger.info(f"\033[93m[REFLECT] Preparing schema information for {len(selected_tables)} tables\033[0m")
        table_schemas = schema_parser(selected_tables, "DDL", include_sample_data=True)
        
        fmt_messages = SQL_ERROR_REFLECTION_TMPL.format_messages(
            database_schema=table_schemas,
            database_description=database_description,
            sql_query=sql_query,
            error_message=error,
            dialect=connection_payload.get("dbType", "").upper(),
            user_query=user_query
        )
        
        # Log the prompt
        log_prompt(fmt_messages, "REFLECT")
        
        logger.info("\033[93m[REFLECT] Querying LLM for SQL correction...\033[0m")
        llm_start_time = datetime.now()
        chat_response = self.llm.chat(fmt_messages)
        llm_end_time = datetime.now()
        
        logger.info(f"\033[93m[REFLECT] LLM response time: {(llm_end_time - llm_start_time).total_seconds():.2f} seconds\033[0m")
        logger.info("\033[94m[REFLECT] LLM response: " + str(chat_response) + "\033[0m")
        
        # Extract the corrected SQL query from the response
        corrected_sql_query = extract_sql_query(chat_response.message.content)
        
        if corrected_sql_query == sql_query:
            logger.warning("\033[93m[REFLECT] LLM returned the same SQL query. No changes made.\033[0m")
        else:
            logger.info("\033[92m[REFLECT] SQL query was modified by LLM\033[0m")
        
        logger.info("\033[92m[REFLECT] Corrected SQL query: " + corrected_sql_query + "\033[0m")
        
        end_time = datetime.now()
        logger.info(f"\033[93m[REFLECT] Step completed in {(end_time - start_time).total_seconds():.2f} seconds\033[0m")
        return SQLValidatorEvent(sql_query=corrected_sql_query, retry_count=retry_count)




class SchemaEnrichmentWorkflow(Workflow):
    """Workflow for enriching database schema with semantic descriptions."""

    def __init__(
        self,
        # schema_enrichment_prompt: PromptTemplate,
        llm: Ollama,
        *args, **kwargs
    ) -> None:
        """Initialize the SQLAgent Workflow."""
        super().__init__(*args, **kwargs)
        # self.schema_enrichment_prompt = schema_enrichment_prompt
        self.llm = llm
        self._timeout = 3000.0
        # Initialize workflow logging metrics
        self.workflow_logs = {
            "start_time": None,
            "end_time": None,
            "total_tables": 0,
            "total_columns": 0,
            "enriched_tables": 0,
            "enriched_columns": 0,
            "failed_tables": 0,
            "failed_columns": 0,
            "clusters": [],
            "processing_times": {
                "total": 0,
                "database_description": 0,
                "schema_enrichment": 0
            },
            "errors": [],
            "warnings": []
        }

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> SchemaEnrichmentEvent:
        import time
        # Start timing the workflow
        self.workflow_logs["start_time"] = time.time()
        
        # Log the start of the workflow
        logger.info("\033[94m[WORKFLOW] Starting Schema Enrichment Workflow\033[0m")
        
        await context.set("connection_payload", ev.connection_payload)
        await context.set("database_schema", ev.database_schema)

        # Count tables and columns for metrics
        total_tables = len(ev.database_schema)
        total_columns = sum(len(table.get('columns', [])) for table in ev.database_schema)
        self.workflow_logs["total_tables"] = total_tables
        self.workflow_logs["total_columns"] = total_columns
        
        logger.info(f"\033[94m[WORKFLOW] Found {total_tables} tables with {total_columns} columns to enrich\033[0m")

        # Generate database description
        desc_start_time = time.time()
        try:
            # Get all table names
            brief_schema_presentation = schema_parser(ev.database_schema, "Simple")
            fmt_messages = DATABASE_DESCRIPTION_TMPL.format_messages(
                schema=brief_schema_presentation
            )
            log_prompt(fmt_messages, "GENERATE")
            chat_response = self.llm.chat(fmt_messages)
            database_description = chat_response.message.content
            logger.info(f"\033[92m[GENERATE] Database description: {database_description}\033[0m")
            self.workflow_logs["processing_times"]["database_description"] = time.time() - desc_start_time
        except Exception as e:
            error_msg = f"Failed to generate database description: {str(e)}"
            logger.error(f"\033[91m[ERROR] {error_msg}\033[0m")
            self.workflow_logs["errors"].append(error_msg)
            database_description = "No description available due to error"

        # Cluster schema
        cluster_start_time = time.time()
        try:
            clusters = schema_clustering(ev.database_schema, resolution_value=2.5)
            
            # Log cluster information
            for i, cluster in enumerate(clusters):
                cluster_info = {
                    "cluster_id": i,
                    "tables": [table.get('tableIdentifier', 'unknown') for table in cluster],
                    "table_count": len(cluster),
                    "column_count": sum(len(table.get('columns', [])) for table in cluster),
                    "status": "pending"
                }
                self.workflow_logs["clusters"].append(cluster_info)
                
            logger.info(f"\033[94m[WORKFLOW] Created {len(clusters)} clusters of related tables\033[0m")
            
            # Add sample data to each table in clusters
            for cluster_idx, cluster in enumerate(clusters):
                for table in cluster:
                    try:
                        table["sample_data"] = get_sample_data(
                            connection_payload=ev.connection_payload, 
                            table_details=table
                        )
                    except Exception as e:
                        warning_msg = f"Failed to get sample data for table {table.get('tableIdentifier', 'unknown')}: {str(e)}"
                        logger.warning(f"\033[93m[WARNING] {warning_msg}\033[0m")
                        self.workflow_logs["warnings"].append(warning_msg)
                        table["sample_data"] = []
            
            await context.set("num_tables", total_tables)
            await context.set("num_clusters", len(clusters))
            await context.set("num_tables_in_clusters", [len(c) for c in clusters])
            
        except Exception as e:
            error_msg = f"Failed to cluster schema: {str(e)}"
            logger.error(f"\033[91m[ERROR] {error_msg}\033[0m")
            self.workflow_logs["errors"].append(error_msg)
            clusters = [[table] for table in ev.database_schema]  # Fallback: each table in its own cluster
            
        return SchemaEnrichmentEvent(database_description=database_description, clusters=clusters)


    @step
    async def Schema_Enrichment(self, context: Context, ev: SchemaEnrichmentEvent) -> StopEvent:
        """Generate database description and cluster schema."""
        import time
        
        enrichment_start_time = time.time()
        cluster_infos = []
        
        # Parse cluster schema
        for cluster in ev.clusters:
            prompt = schema_parser(cluster, "Simple", include_sample_data=True)
            cluster_infos.append(prompt)
            print(prompt)

        # Process each cluster
        cluster_enriched = []
        for cluster_idx, cluster_info in enumerate(cluster_infos):
            cluster_start_time = time.time()
            logger.info(f"\033[94m[WORKFLOW] Processing cluster {cluster_idx+1}/{len(cluster_infos)}\033[0m")
            
            try:
                fmt_messages = SCHEMA_ENRICHMENT_TMPL.format_messages(
                    cluster_info=cluster_info,
                    db_info=ev.database_description
                )
                log_prompt(fmt_messages, "GENERATE")

                retries = 3
                enriched_data = []
                for i in range(retries):
                    chat_response = self.llm.chat(fmt_messages)
                    enriched_data = parse_schema_enrichment(chat_response)

                    if len(enriched_data) > 0:
                        break
                    else:
                        logger.warning(f"\033[93m[WARNING] Failed to enrich cluster {cluster_idx+1} after {i+1}/{retries} retries\033[0m")
                        time.sleep(1)
                        continue
                cluster_enriched.append(enriched_data)
                
                # Update cluster status in logs
                if cluster_idx < len(self.workflow_logs["clusters"]):
                    self.workflow_logs["clusters"][cluster_idx]["status"] = "enriched"
                    self.workflow_logs["clusters"][cluster_idx]["processing_time"] = time.time() - cluster_start_time
                    self.workflow_logs["clusters"][cluster_idx]["enriched_tables"] = len(enriched_data)
                    
                logger.info(f"\033[92m[SUCCESS] Enriched cluster {cluster_idx+1} with {len(enriched_data)} tables\033[0m")
                
            except Exception as e:
                error_msg = f"Failed to enrich cluster {cluster_idx+1}: {str(e)}"
                logger.error(f"\033[91m[ERROR] {error_msg}\033[0m")
                self.workflow_logs["errors"].append(error_msg)
                cluster_enriched.append([])  # Add empty list as fallback
                
                # Update cluster status in logs
                if cluster_idx < len(self.workflow_logs["clusters"]):
                    self.workflow_logs["clusters"][cluster_idx]["status"] = "failed"
                    self.workflow_logs["clusters"][cluster_idx]["processing_time"] = time.time() - cluster_start_time
                    self.workflow_logs["clusters"][cluster_idx]["error"] = str(e)
            
        # Get database_schema from context
        database_schema = await context.get("database_schema")
        
        # Create mapping from table_name to descriptions
        enrichment_map = {}
        for cluster_data in cluster_enriched:
            for table_info in cluster_data:
                table_name = table_info.get('table_name')
                if not table_name:
                    continue
                    
                # Create mapping for table
                if table_name not in enrichment_map:
                    enrichment_map[table_name] = {
                        'table_description': table_info.get('description', ''),
                        'columns': {}
                    }
                
                # Create mapping for columns
                for column_info in table_info.get('columns', []):
                    column_name = column_info.get('column_name')
                    if column_name:
                        enrichment_map[table_name]['columns'][column_name] = column_info.get('description', '')
        
        logger.info(f"\033[92m[GENERATE] Database schema enrichment map created for {len(enrichment_map)} tables\033[0m")
        
        # Track success and failure counts
        enriched_tables_count = 0
        enriched_columns_count = 0
        failed_tables = []
        failed_columns = []
        
        # Update descriptions for tables and columns in database_schema
        for table in database_schema:
            table_name = table.get('tableIdentifier')
            table_enriched = False
            
            if table_name in enrichment_map:
                # Update table description if current field is empty
                table_description = table.get('tableDescription', '')
                if not table_description and enrichment_map[table_name]['table_description']:
                    table['tableDescription'] = enrichment_map[table_name]['table_description']
                    table_enriched = True
                    enriched_tables_count += 1
                
                # Update column descriptions
                columns_enriched = 0
                columns_empty = 0
                
                for column in table.get('columns', []):
                    column_name = column.get('columnIdentifier')
                    column_enriched = False
                    
                    if column_name in enrichment_map[table_name]['columns']:
                        # Update column description if current field is empty
                        if (column.get("columnDescription", "") in ["", "NULL", "''"] or 
                            column.get("columnDescription") is None or 
                            len(str(column.get("columnDescription", ""))) < 10):
                            column["columnDescription"] = enrichment_map[table_name]['columns'][column_name]
                            column_enriched = True
                            columns_enriched += 1
                            enriched_columns_count += 1
                        else:
                            # Column already had a description
                            pass
                    else:
                        # Column not found in enrichment map
                        if (column.get("columnDescription", "") in ["", "NULL", "''"] or 
                            column.get("columnDescription") is None or 
                            len(str(column.get("columnDescription", ""))) <= 1):
                            columns_empty += 1
                            failed_columns.append(f"{table_name}.{column_name}")
                
                logger.info(f"\033[94m[WORKFLOW] Table '{table_name}': Enriched {columns_enriched} columns, {columns_empty} columns without enrichment\033[0m")
                
            else:
                # Table not found in enrichment map
                failed_tables.append(table_name)
                logger.warning(f"\033[93m[WARNING] Table '{table_name}' not found in enrichment map\033[0m")
        
        # Update workflow logs with final counts
        self.workflow_logs["enriched_tables"] = enriched_tables_count
        self.workflow_logs["enriched_columns"] = enriched_columns_count
        self.workflow_logs["failed_tables"] = len(failed_tables)
        self.workflow_logs["failed_columns"] = len(failed_columns)
        self.workflow_logs["processing_times"]["schema_enrichment"] = time.time() - enrichment_start_time
        
        # Calculate overall workflow stats
        self.workflow_logs["end_time"] = time.time()
        self.workflow_logs["processing_times"]["total"] = self.workflow_logs["end_time"] - self.workflow_logs["start_time"]
        
        # Create summary logs
        success_rate_tables = (enriched_tables_count / self.workflow_logs["total_tables"]) * 100 if self.workflow_logs["total_tables"] > 0 else 0
        success_rate_columns = (enriched_columns_count / self.workflow_logs["total_columns"]) * 100 if self.workflow_logs["total_columns"] > 0 else 0
        
        logger.info("\033[94m" + "="*50 + "\033[0m")
        logger.info(f"\033[94m[WORKFLOW SUMMARY] Schema Enrichment Complete\033[0m")
        logger.info(f"\033[94m[WORKFLOW SUMMARY] Total runtime: {self.workflow_logs['processing_times']['total']:.2f} seconds\033[0m")
        logger.info(f"\033[94m[WORKFLOW SUMMARY] Tables: {enriched_tables_count}/{self.workflow_logs['total_tables']} enriched ({success_rate_tables:.1f}%)\033[0m")
        logger.info(f"\033[94m[WORKFLOW SUMMARY] Columns: {enriched_columns_count}/{self.workflow_logs['total_columns']} enriched ({success_rate_columns:.1f}%)\033[0m")
        logger.info(f"\033[94m[WORKFLOW SUMMARY] Errors: {len(self.workflow_logs['errors'])}, Warnings: {len(self.workflow_logs['warnings'])}\033[0m")
        logger.info("\033[94m" + "="*50 + "\033[0m")
        
        # Prepare final result
        result = {
            "_workflow_logs": self.workflow_logs,
            "database_description": ev.database_description,
            "enriched_schema": database_schema
        }
        
        return StopEvent(result=result)

