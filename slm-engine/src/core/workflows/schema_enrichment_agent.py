from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from core.utils import (
    schema_parser,
    schema_clustering,
    parse_schema_enrichment
)
from core.events import SchemaEnrichmentEvent
from core.models import (
    DatabaseDescription,
    SchemaEnrichmentResponse
)
from core.services import get_sample_data_improved, llm_chat, llm_chat_with_pydantic
from response.log_manager import (
    log_step_start,
    log_step_end,
    log_success,
    log_error,
    log_warning,
    log_summary,
    log_prompt
)
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from exceptions.app_exception import AppException
import logging
import time

logger = logging.getLogger(__name__)

class SchemaEnrichmentWorkflow(Workflow):
    """Workflow for enriching database schema with semantic descriptions."""

    def __init__(
        self,
        llm: Ollama,
        *args, **kwargs
    ) -> None:
        """Initialize the Schema Enrichment Workflow."""
        super().__init__(*args, **kwargs)
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
        """Start the Schema Enrichment workflow - analyze database schema and cluster tables."""
        # Start timing the workflow
        self.workflow_logs["start_time"] = time.time()
        
        # Log the start of the workflow
        log_step_start("WORKFLOW", message="Starting Schema Enrichment Workflow")
        
        # Store values in context
        await context.set("connection_payload", ev.connection_payload)
        await context.set("database_schema", ev.database_schema)

        # Count tables and columns for metrics
        total_tables = len(ev.database_schema)
        total_columns = sum(len(table.get('columns', [])) for table in ev.database_schema)
        self.workflow_logs["total_tables"] = total_tables
        self.workflow_logs["total_columns"] = total_columns
        
        log_step_start("WORKFLOW", message=f"Found {total_tables} tables with {total_columns} columns to enrich")

        # Generate database description
        desc_start_time = time.time()
        try:
            # Get all table names
            brief_schema_presentation = schema_parser(ev.database_schema, "Simple")
            
            # Load template from configuration
            from core.templates import DATABASE_DESCRIPTION_SKELETON
            DATABASE_DESCRIPTION_PROMPT = DATABASE_DESCRIPTION_SKELETON.format(
                schema=brief_schema_presentation
            )
            
            # Query LLM for database description
            chat_response = llm_chat_with_pydantic(
                llm=self.llm, 
                prompt=PromptTemplate(DATABASE_DESCRIPTION_PROMPT), 
                pydantic_model=DatabaseDescription
            )
            
            database_description = chat_response.database_description
            log_success("GENERATE", f"Database description: {database_description}")
            self.workflow_logs["processing_times"]["database_description"] = time.time() - desc_start_time
        except Exception as e:
            error_msg = f"Failed to generate database description: {str(e)}"
            log_error("ERROR", error_msg)
            self.workflow_logs["errors"].append(error_msg)
            database_description = "No description available due to error"
            raise AppException(error_msg, 500)

        # Cluster schema
        cluster_start_time = time.time()
        try:
            # Use clustering algorithm to group related tables
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
                
            log_step_start("WORKFLOW", message=f"Created {len(clusters)} clusters of related tables")
            
            # Add sample data to each table in clusters
            for cluster_idx, cluster in enumerate(clusters):
                for table in cluster:
                    try:
                        table["sample_data"] = get_sample_data_improved(
                            connection_payload=ev.connection_payload, 
                            table_details=table
                        )
                    except Exception as e:
                        warning_msg = f"Failed to get sample data for table {table.get('tableIdentifier', 'unknown')}: {str(e)}"
                        log_warning("WARNING", warning_msg)
                        self.workflow_logs["warnings"].append(warning_msg)
                        table["sample_data"] = []
            
            # Store cluster information in context
            await context.set("num_tables", total_tables)
            await context.set("num_clusters", len(clusters))
            await context.set("num_tables_in_clusters", [len(c) for c in clusters])
            
        except Exception as e:
            error_msg = f"Failed to cluster schema: {str(e)}"
            log_error("ERROR", error_msg)
            self.workflow_logs["errors"].append(error_msg)
            # Fallback: each table in its own cluster
            clusters = [[table] for table in ev.database_schema]  
            raise AppException(error_msg, 500)
            
        return SchemaEnrichmentEvent(database_description=database_description, clusters=clusters)

    @step
    async def Schema_Enrichment(self, context: Context, ev: SchemaEnrichmentEvent) -> StopEvent:
        """Process each cluster of tables to generate rich descriptions."""
        enrichment_start_time = time.time()
        cluster_infos = []
        
        # Parse cluster schema
        for cluster in ev.clusters:
            prompt = schema_parser(cluster, "Simple", include_sample_data=True)
            cluster_infos.append(prompt)
            print(prompt)
        database_description = ev.database_description
        # Process each cluster
        cluster_enriched = []
        for cluster_idx, cluster_info in enumerate(cluster_infos):
            cluster_start_time = time.time()
            log_step_start("WORKFLOW", message=f"Processing cluster {cluster_idx+1}/{len(cluster_infos)}")
            
            try:
                # Load template from configuration
                from core.templates import SCHEMA_ENRICHMENT_SKELETON
                print(cluster_info)
                print(database_description)
                SCHEMA_ENRICHMENT_PROMPT = SCHEMA_ENRICHMENT_SKELETON.format(
                    schema=cluster_info,
                    database_description=database_description
                )
                log_prompt("PROMPT", SCHEMA_ENRICHMENT_PROMPT)
                retries = 3
                enriched_data = []
                
                # Try with Pydantic model first
                for i in range(retries):
                    try:
                        chat_response = llm_chat_with_pydantic(
                            llm=self.llm, 
                            prompt=PromptTemplate(SCHEMA_ENRICHMENT_PROMPT), 
                            pydantic_model=SchemaEnrichmentResponse
                        )
                        enriched_data = chat_response.tables
                        print(enriched_data)
                        if len(enriched_data) > 0:
                            log_success("SUCCESS", f"Successfully parsed cluster {cluster_idx+1} with Pydantic model")
                            break
                        
                        log_warning("WARNING", f"LLM returned empty schema for cluster {cluster_idx+1}, retry {i+1}/{retries}")
                        time.sleep(1)
                    except Exception as e:
                        log_warning("WARNING", f"Failed to parse with Pydantic for cluster {cluster_idx+1}, retry {i+1}/{retries}: {str(e)}")
                        time.sleep(1)

                cluster_enriched.append({"tables": enriched_data})
                
                # Update cluster status in logs
                if cluster_idx < len(self.workflow_logs["clusters"]):
                    self.workflow_logs["clusters"][cluster_idx]["status"] = "enriched"
                    self.workflow_logs["clusters"][cluster_idx]["processing_time"] = time.time() - cluster_start_time
                    self.workflow_logs["clusters"][cluster_idx]["enriched_tables"] = len(enriched_data)
                    
                log_success("SUCCESS", f"Enriched cluster {cluster_idx+1} with {len(enriched_data)} tables")
                
            except Exception as e:
                error_msg = f"Failed to enrich cluster {cluster_idx+1}: {str(e)}"
                log_error("ERROR", error_msg)
                self.workflow_logs["errors"].append(error_msg)
                cluster_enriched.append({"tables": []})  # Add empty list as fallback
                
                # Update cluster status in logs
                if cluster_idx < len(self.workflow_logs["clusters"]):
                    self.workflow_logs["clusters"][cluster_idx]["status"] = "failed"
                    self.workflow_logs["clusters"][cluster_idx]["processing_time"] = time.time() - cluster_start_time
                    self.workflow_logs["clusters"][cluster_idx]["error"] = str(e)
            
        # Get database_schema from context
        database_schema = await context.get("database_schema")
        
        # Create mapping from table_name to descriptions
        enrichment_map = self._create_enrichment_map(cluster_enriched)
        log_success("GENERATE", f"Database schema enrichment map created for {len(enrichment_map)} tables")
        
        # Apply enrichment to schema
        enrichment_result = self._apply_enrichment(database_schema, enrichment_map)
        
        # Update workflow logs with final counts
        self.workflow_logs["enriched_tables"] = enrichment_result["enriched_tables"]
        self.workflow_logs["enriched_columns"] = enrichment_result["enriched_columns"]
        self.workflow_logs["failed_tables"] = len(enrichment_result["failed_tables"])
        self.workflow_logs["failed_columns"] = len(enrichment_result["failed_columns"])
        self.workflow_logs["processing_times"]["schema_enrichment"] = time.time() - enrichment_start_time
        
        # Calculate overall workflow stats
        self.workflow_logs["end_time"] = time.time()
        self.workflow_logs["processing_times"]["total"] = self.workflow_logs["end_time"] - self.workflow_logs["start_time"]
        
        # Create summary logs
        success_rate_tables = (enrichment_result["enriched_tables"] / self.workflow_logs["total_tables"]) * 100 if self.workflow_logs["total_tables"] > 0 else 0
        success_rate_columns = (enrichment_result["enriched_columns"] / self.workflow_logs["total_columns"]) * 100 if self.workflow_logs["total_columns"] > 0 else 0
        
        # Log workflow summary
        summary_items = {
            "Total runtime": f"{self.workflow_logs['processing_times']['total']:.2f} seconds",
            "Tables": f"{enrichment_result['enriched_tables']}/{self.workflow_logs['total_tables']} enriched ({success_rate_tables:.1f}%)",
            "Columns": f"{enrichment_result['enriched_columns']}/{self.workflow_logs['total_columns']} enriched ({success_rate_columns:.1f}%)",
            "Errors": f"{len(self.workflow_logs['errors'])}, Warnings: {len(self.workflow_logs['warnings'])}"
        }
        log_summary("WORKFLOW SUMMARY", summary_items)
        
        # Prepare final result
        result = {
            "_workflow_logs": self.workflow_logs,
            "database_description": ev.database_description,
            "enriched_schema": database_schema
        }
        
        return StopEvent(result=result)
        
    def _create_enrichment_map(self, cluster_enriched):
        """Create a mapping from table names to their enriched descriptions."""
        enrichment_map = {}
        
        for cluster in cluster_enriched:
            for table in cluster.get('tables', []):
                table_name = table.table_name
                if not table_name:
                    continue
                    
                # Create mapping for table
                if table_name not in enrichment_map:
                    enrichment_map[table_name] = {
                        'table_description': table.description,
                        'columns': {}
                    }
                
                # Create mapping for columns
                for column in table.columns:
                    column_name = column.column_name
                    if column_name:
                        enrichment_map[table_name]['columns'][column_name] = column.description
                        
        return enrichment_map
        
    def _apply_enrichment(self, database_schema, enrichment_map):
        """Apply enrichment to database schema and track statistics."""
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
                
                log_step_start("WORKFLOW", message=f"Table '{table_name}': Enriched {columns_enriched} columns, {columns_empty} columns without enrichment")
                
            else:
                # Table not found in enrichment map
                failed_tables.append(table_name)
                log_warning("WARNING", f"Table '{table_name}' not found in enrichment map")
        
        return {
            "enriched_tables": enriched_tables_count,
            "enriched_columns": enriched_columns_count,
            "failed_tables": failed_tables,
            "failed_columns": failed_columns
        } 