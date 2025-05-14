from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
import json
import base64
import logging
import time
from typing import List, Dict, Any
from collections import defaultdict

# Import services (adjust import path as needed)
from core.services import get_schema, execute_sql, validate_connection_payload
from core.llm import llm_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define event classes
class SchemaExtractEvent(Event):
    """Event after extracting schema from database."""
    table_details: List[Dict]
    connection_payload: Dict

class TableRelationshipsEvent(Event):
    """Event after analyzing table relationships."""
    table_groups: List[List[str]]
    
class EnrichmentRequestEvent(Event):
    """Event for requesting semantic enrichment from LLM."""
    group_index: int
    table_group: List[str]
    group_info: str
    
class EnrichmentResponseEvent(Event):
    """Event with LLM response for a table group."""
    group_index: int
    response: str
    processed_tables: List[str]
    
class InitiateSynthesisEvent(Event):
    """Event to initiate the synthesis phase."""
    synthesis_prompts: List[Dict]
    enrichment_responses: List[Dict]
    
class SynthesisRequestEvent(Event):
    """Event for requesting synthesis of duplicate table descriptions."""
    table_name: str
    prompt: str
    
class SynthesisResponseEvent(Event):
    """Event with LLM synthesis response."""
    table_name: str
    response: str
    
class AllEnrichmentsEvent(Event):
    """Event after all enrichments are complete."""
    enrichment_responses: List[Dict]
    synthesis_responses: List[Dict]

# Define utility functions
def get_table_examples(connection_payload, table_name, columns, limit=3, max_value_length=50):
    """Get example values from the specified table with the given column order."""
    try:
        # Extract column names from the column definitions
        column_names = []
        for col_def in columns:
            col_name = col_def.split()[0]
            column_names.append(col_name)
        
        # Create the query with specific columns
        columns_query = ", ".join(column_names)
        query = f"SELECT {columns_query} FROM {table_name} LIMIT {limit}"
        
        result = execute_sql(connection_payload, query)
        
        if result.get("error"):
            return []
        
        # Access the data directly
        data = result.get("data", [])
        
        # Format as CSV rows with the same column order
        csv_rows = []
        for record in data:
            row_values = []
            for col_name in column_names:
                value = record.get(col_name, "")
                str_value = str(value) if value is not None else "NULL"
                
                # Truncate if too long
                if len(str_value) > max_value_length:
                    str_value = str_value[:max_value_length] + "..."
                
                # Add quotes if value contains comma
                if "," in str_value:
                    str_value = f'"{str_value}"'
                    
                row_values.append(str_value)
            
            csv_rows.append(", ".join(row_values))
        
        return csv_rows
    except Exception as e:
        logger.error(f"Error getting examples for table {table_name}: {str(e)}")
        return []

def create_table_relationship_arrays(table_details):
    """Create table relationship arrays with the main table at first position."""
    table_relationships = {}
    
    # Initialize empty arrays for each table
    for table in table_details:
        table_name = table["tableIdentifier"]
        table_relationships[table_name] = []
    
    # Find relationships
    for table in table_details:
        table_name = table["tableIdentifier"]
        for column in table["columns"]:
            if "relations" in column and column["relations"]:
                for relation in column["relations"]:
                    related_table = relation["tableIdentifier"]
                    if related_table not in table_relationships[table_name]:
                        table_relationships[table_name].append(related_table)
                    if table_name not in table_relationships[related_table]:
                        table_relationships[related_table].append(table_name)
    
    # Convert to 2D array with main table first
    relationship_arrays = []
    for table_name, related_tables in table_relationships.items():
        array = [table_name] + related_tables
        relationship_arrays.append(array)
    
    return relationship_arrays

def prepare_group_info(table_details, table_group, connection_payload):
    """Prepare detailed information about a table group for the prompt."""
    # Get table structures and examples
    table_structures = {}
    table_examples = {}
    
    for table in table_details:
        if table["tableIdentifier"] in table_group:
            # Get structure
            columns = []
            for column in table["columns"]:
                column_type = column["columnType"]
                pk_indicator = " PRIMARY KEY" if column["isPrimaryKey"] else ""
                columns.append(f"{column['columnIdentifier']} {column_type}{pk_indicator}")
            table_structures[table["tableIdentifier"]] = columns
            
            # Get examples with the same column order
            examples = get_table_examples(connection_payload, table["tableIdentifier"], columns)
            table_examples[table["tableIdentifier"]] = examples
    
    # Format the table group information
    group_info = f"Table group: {table_group}\n"
    for table_name in table_group:
        if table_name in table_structures:
            columns_str = ", ".join(table_structures[table_name])
            group_info += f"- {table_name}: ({columns_str})\n"
            
        # Add example values if available
        if table_examples.get(table_name):
            group_info += f"\n* Example Values for {table_name} (CSV format):\n"
            # First add column headers
            column_headers = ", ".join([col.split()[0] for col in table_structures[table_name]])
            group_info += f"{column_headers}\n"
            
            # Then add data rows
            for example in table_examples[table_name]:
                group_info += f"{example}\n"
            group_info += "\n"
    
    return group_info

def transform_response_format(response):
    """Transform the LLM response from flat format to nested format."""
    try:
        # Parse JSON from response
        if isinstance(response, dict):
            data = response
        else:
            json_start = response.find('```json')
            if json_start != -1:
                json_content = response[json_start + 7:]
                json_end = json_content.find('```')
                if json_end != -1:
                    json_content = json_content[:json_end].strip()
                data = json.loads(json_content)
            else:
                data = json.loads(response)
        
        # Get tables and column descriptions
        tables_dict = data.get("tables", {})
        columns_dict = data.get("columns", {})
        
        # Create new structure
        new_format = []
        
        # Group columns by table
        columns_by_table = {}
        for column_key, description in columns_dict.items():
            table_name, column_name = column_key.split('.')
            
            if table_name not in columns_by_table:
                columns_by_table[table_name] = []
                
            columns_by_table[table_name].append({
                "column_name": column_name,
                "description": description
            })
        
        # Create object for each table
        for table_name, description in tables_dict.items():
            table_obj = {
                "table_name": table_name,
                "description": description,
                "columns": columns_by_table.get(table_name, [])
            }
            new_format.append(table_obj)
        
        # Add tables with columns but no description
        for table_name, columns in columns_by_table.items():
            if table_name not in tables_dict:
                table_obj = {
                    "table_name": table_name,
                    "description": "No description available",
                    "columns": columns
                }
                new_format.append(table_obj)
        
        return new_format
    
    except Exception as e:
        logger.error(f"Error transforming response format: {e}")
        return []

def merge_semantic_descriptions(responses, table_details):
    """Find tables and columns that need synthesis due to multiple descriptions."""
    # Create dictionaries to store all descriptions
    table_descriptions = defaultdict(list)
    column_descriptions = defaultdict(list)
    
    # Extract descriptions from all responses
    for response_obj in responses:
        response = response_obj.get("response", "{}")
        processed_tables = response_obj.get("processed_tables", [])
        
        try:
            # Parse JSON
            if isinstance(response, dict):
                data = response
            else:
                json_start = response.find('```json')
                if json_start != -1:
                    json_content = response[json_start + 7:]
                    json_end = json_content.find('```')
                    if json_end != -1:
                        json_content = json_content[:json_end].strip()
                    data = json.loads(json_content)
                else:
                    data = json.loads(response)
            
            # Add table descriptions
            for table_name, description in data.get("tables", {}).items():
                if table_name in processed_tables:  # Only add if this table was in this group
                    table_descriptions[table_name].append(description)
            
            # Add column descriptions
            for column_key, description in data.get("columns", {}).items():
                table_name, column_name = column_key.split('.')
                if table_name in processed_tables:  # Only add if this table was in this group
                    column_descriptions[column_key].append(description)
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            continue
    
    # Create synthesis prompts
    synthesis_prompts = []
    
    # Create prompts for tables with multiple descriptions
    for table_name, descriptions in table_descriptions.items():
        if len(descriptions) <= 1:
            continue  # Skip if only one description
        
        # Find table structure
        table_structure = None
        for table in table_details:
            if table["tableIdentifier"] == table_name:
                columns = []
                for column in table["columns"]:
                    column_type = column["columnType"]
                    pk_indicator = " PRIMARY KEY" if column["isPrimaryKey"] else ""
                    columns.append(f"{column['columnIdentifier']} {column_type}{pk_indicator}")
                table_structure = f"({', '.join(columns)})"
                break
        
        prompt = (
            f"Synthesize the best semantic description for the table '{table_name}' {table_structure}\n"
            f"based on these multiple descriptions:\n\n"
        )
        
        for i, desc in enumerate(descriptions):
            prompt += f"Description {i+1}: {desc}\n"
        
        prompt += (
            "\nPlease provide a unified and comprehensive description that captures "
            "the main purpose of this table based on these varying descriptions. "
            "Your response should be a JSON object with this structure:\n"
            "```json\n"
            "{\n"
            "  \"table_name\": \"<table name>\",\n"
            "  \"description\": \"<best unified description>\"\n"
            "}\n"
            "```\n"
        )
        
        synthesis_prompts.append({
            "type": "table",
            "table_name": table_name,
            "prompt": prompt
        })
    
    # Similarly for columns with multiple descriptions
    for column_key, descriptions in column_descriptions.items():
        if len(descriptions) <= 1:
            continue  # Skip if only one description
        
        # Split table and column name
        table_name, column_name = column_key.split('.')
        
        # Find column type
        column_type = None
        for table in table_details:
            if table["tableIdentifier"] == table_name:
                for column in table["columns"]:
                    if column["columnIdentifier"] == column_name:
                        column_type = column["columnType"]
                        pk_info = " PRIMARY KEY" if column["isPrimaryKey"] else ""
                        column_type += pk_info
                        break
                break
        
        prompt = (
            f"Synthesize the best semantic description for the column '{column_key}' {column_type}\n"
            f"based on these multiple descriptions:\n\n"
        )
        
        for i, desc in enumerate(descriptions):
            prompt += f"Description {i+1}: {desc}\n"
        
        prompt += (
            "\nPlease provide a unified and comprehensive description that captures "
            "the meaning of this column based on these varying descriptions. "
            "Your response should be a JSON object with this structure:\n"
            "```json\n"
            "{\n"
            "  \"column_key\": \"<table.column>\",\n"
            "  \"description\": \"<best unified description>\"\n"
            "}\n"
            "```\n"
        )
        
        synthesis_prompts.append({
            "type": "column",
            "column_key": column_key,
            "prompt": prompt
        })
    
    return synthesis_prompts

def get_final_semantic_descriptions(enrichment_responses, synthesis_responses):
    """Create final semantic descriptions by combining initial and synthesis responses."""
    # Create temporary dictionaries for tables and columns
    tables_dict = {}
    columns_dict = defaultdict(list)
    
    # Process initial enrichment responses
    for response_obj in enrichment_responses:
        response = response_obj.get("response", "{}")
        
        try:
            # Parse JSON
            if isinstance(response, dict):
                data = response
            else:
                json_start = response.find('```json')
                if json_start != -1:
                    json_content = response[json_start + 7:]
                    json_end = json_content.find('```')
                    if json_end != -1:
                        json_content = json_content[:json_end].strip()
                    data = json.loads(json_content)
                else:
                    data = json.loads(response)
            
            # Add table descriptions (only if not already added)
            for table_name, description in data.get("tables", {}).items():
                if table_name not in tables_dict:
                    tables_dict[table_name] = description
            
            # Add column descriptions
            for column_key, description in data.get("columns", {}).items():
                table_name, column_name = column_key.split('.')
                
                # Check if column already exists
                existing = False
                for col in columns_dict[table_name]:
                    if col["column_name"] == column_name:
                        existing = True
                        break
                
                if not existing:
                    columns_dict[table_name].append({
                        "column_name": column_name,
                        "description": description
                    })
                
        except Exception as e:
            logger.error(f"Error parsing enrichment response: {e}")
            continue
    
    # Process synthesis responses
    for response_obj in synthesis_responses:
        response = response_obj.get("response", "{}")
        
        try:
            # Parse JSON
            if isinstance(response, dict):
                data = response
            else:
                json_start = response.find('```json')
                if json_start != -1:
                    json_content = response[json_start + 7:]
                    json_end = json_content.find('```')
                    if json_end != -1:
                        json_content = json_content[:json_end].strip()
                    data = json.loads(json_content)
                else:
                    data = json.loads(response)
            
            # Update table description
            if "table_name" in data and "description" in data:
                tables_dict[data["table_name"]] = data["description"]
            
            # Update column description
            if "column_key" in data and "description" in data:
                table_name, column_name = data["column_key"].split('.')
                
                # Find and update column
                for col in columns_dict[table_name]:
                    if col["column_name"] == column_name:
                        col["description"] = data["description"]
                        break
                
        except Exception as e:
            logger.error(f"Error parsing synthesis response: {e}")
            continue
    
    # Create final format
    final_descriptions = []
    for table_name, description in tables_dict.items():
        table_obj = {
            "table_name": table_name,
            "description": description,
            "columns": columns_dict.get(table_name, [])
        }
        final_descriptions.append(table_obj)
    
    return final_descriptions

# Define the SchemaEnrichmentWorkflow
class SchemaEnrichmentWorkflow(Workflow):
    """Workflow for enriching database schema with semantic descriptions."""
    
    def __init__(
        self,
        llm,
        *args, **kwargs
    ) -> None:
        """Initialize the Schema Enrichment Workflow."""
        super().__init__(*args, **kwargs)
        self.llm = llm
        
        # Define the prompt templates
        self.enrichment_prompt_tmpl = PromptTemplate(
            "Analyze these database tables and their columns to determine their semantic meaning: "
            "\n\n{group_info}\n\n"
            "IMPORTANT: Respond ONLY with a JSON object using exactly this structure:\n"
            "```json\n"
            "{\n"
            "  \"tables\": {\n"
            "    \"<table_name>\": \"<concise description of table purpose>\",\n"
            "    ...\n"
            "  },\n"
            "  \"columns\": {\n"
            "    \"<table_name>.<column_name>\": \"<concise description of column meaning>\",\n"
            "    ...\n"
            "  }\n"
            "}\n"
            "```\n"
            "Do not include any text outside this JSON structure."
        )

    @step
    async def Start_workflow(self, context: Context, ev: StartEvent) -> SchemaExtractEvent:
        """Start the Schema Enrichment Workflow."""
        logger.info("Starting Schema Enrichment Workflow")
        
        # Extract SQLite path from input
        sqlite_file_path = ev.sqlite_path
        logger.info(f"Using SQLite file: {sqlite_file_path}")
        
        # Prepare connection payload
        connection_payload = {
            "dbType": "sqlite",
            "file": None
        }
        
        # Read SQLite file and encode as base64
        try:
            with open(sqlite_file_path, "rb") as file:
                file_content = file.read()
                connection_payload["file"] = base64.b64encode(file_content).decode('utf-8')
                logger.info(f"Successfully read SQLite file ({len(file_content)} bytes)")
        except Exception as e:
            logger.error(f"Error reading SQLite file: {str(e)}")
            raise ValueError(f"Failed to read SQLite file: {str(e)}")
        
        # Validate connection payload
        is_valid, error_msg = validate_connection_payload(connection_payload)
        if not is_valid:
            logger.error(f"Invalid connection payload: {error_msg}")
            raise ValueError(f"Invalid connection payload: {error_msg}")
        
        # Store connection payload in context
        await context.set("connection_payload", connection_payload)
        
        # Get schema from database
        logger.info("Retrieving database schema...")
        table_details = get_schema(connection_payload)
        logger.info(f"Retrieved schema for {len(table_details)} tables")
        
        # Store table details in context
        await context.set("table_details", table_details)
        
        return SchemaExtractEvent(table_details=table_details, connection_payload=connection_payload)

    @step
    async def Analyze_table_relationships(self, context: Context, ev: SchemaExtractEvent) -> TableRelationshipsEvent:
        """Analyze relationships between tables."""
        logger.info("Analyzing table relationships")
        
        # Create table relationship arrays
        table_groups = create_table_relationship_arrays(ev.table_details)
        logger.info(f"Found {len(table_groups)} table groups")
        
        # Store table groups in context
        await context.set("table_groups", table_groups)
        
        # Initialize enrichment responses array in context
        await context.set("enrichment_responses", [])
        await context.set("current_group_index", 0)
        
        return TableRelationshipsEvent(table_groups=table_groups)

    @step
    async def Request_enrichment(self, context: Context, ev: TableRelationshipsEvent | EnrichmentResponseEvent) -> EnrichmentRequestEvent | InitiateSynthesisEvent:
        """Request semantic enrichment for each table group."""
        table_groups = await context.get("table_groups")
        current_index = await context.get("current_group_index")
        
        # Check if we've processed all groups
        if current_index >= len(table_groups):
            logger.info("All table groups processed, moving to synthesis phase")
            
            # Get all enrichment responses
            enrichment_responses = await context.get("enrichment_responses")
            
            # Get all table details
            table_details = await context.get("table_details")
            
            # Find tables and columns that need synthesis
            synthesis_prompts = merge_semantic_descriptions(enrichment_responses, table_details)
            logger.info(f"Identified {len(synthesis_prompts)} items needing synthesis")
            
            # Return event to initiate synthesis
            return InitiateSynthesisEvent(
                synthesis_prompts=synthesis_prompts,
                enrichment_responses=enrichment_responses
            )
        
        # Get the current table group
        table_group = table_groups[current_index]
        logger.info(f"Processing table group {current_index + 1}/{len(table_groups)}: {table_group}")
        
        # Get connection payload and table details
        connection_payload = await context.get("connection_payload")
        table_details = await context.get("table_details")
        
        # Prepare group info for prompt
        group_info = prepare_group_info(table_details, table_group, connection_payload)
        
        return EnrichmentRequestEvent(
            group_index=current_index,
            table_group=table_group,
            group_info=group_info
        )

    @step
    async def Process_enrichment(self, context: Context, ev: EnrichmentRequestEvent) -> EnrichmentResponseEvent:
        """Process the semantic enrichment request with LLM."""
        logger.info(f"Processing enrichment request for group {ev.group_index + 1}")
        
        # Format the prompt
        prompt = self.enrichment_prompt_tmpl.format_messages(group_info=ev.group_info)
        
        # Query the LLM
        logger.info("Querying LLM for semantic enrichment...")
        start_time = time.time()
        response = self.llm.chat(prompt)
        end_time = time.time()
        
        logger.info(f"LLM response received in {end_time - start_time:.2f} seconds")
        
        # Store response
        enrichment_responses = await context.get("enrichment_responses")
        enrichment_responses.append({
            "group_index": ev.group_index,
            "response": response.message.content,
            "processed_tables": ev.table_group
        })
        await context.set("enrichment_responses", enrichment_responses)
        
        # Increment group index
        current_index = await context.get("current_group_index")
        await context.set("current_group_index", current_index + 1)
        
        return EnrichmentResponseEvent(
            group_index=ev.group_index,
            response=response.message.content,
            processed_tables=ev.table_group
        )

    @step
    async def Initiate_synthesis(self, context: Context, ev: InitiateSynthesisEvent) -> SynthesisRequestEvent | AllEnrichmentsEvent:
        """Initialize the synthesis process."""
        # Store synthesis prompts and initialize index and responses
        synthesis_prompts = ev.synthesis_prompts
        enrichment_responses = ev.enrichment_responses
        
        # If nothing needs synthesis, skip to final processing
        if not synthesis_prompts:
            logger.info("No synthesis needed, proceeding to final processing")
            return AllEnrichmentsEvent(
                enrichment_responses=enrichment_responses, 
                synthesis_responses=[]
            )
        
        await context.set("synthesis_prompts", synthesis_prompts)
        await context.set("current_synthesis_index", 0)
        await context.set("synthesis_responses", [])
        
        # Process first synthesis prompt
        first_prompt = synthesis_prompts[0]
        return SynthesisRequestEvent(
            table_name=first_prompt.get("table_name", "") or first_prompt.get("column_key", ""),
            prompt=first_prompt["prompt"]
        )

    @step
    async def Process_synthesis(self, context: Context, ev: SynthesisRequestEvent) -> SynthesisResponseEvent | AllEnrichmentsEvent:
        """Process synthesis request with LLM."""
        logger.info(f"Processing synthesis request for {ev.table_name}")
        
        # Format and query the LLM
        logger.info("Querying LLM for synthesis...")
        start_time = time.time()
        response = self.llm.complete(ev.prompt)
        end_time = time.time()
        
        logger.info(f"LLM synthesis response received in {end_time - start_time:.2f} seconds")
        
        # Store response
        synthesis_responses = await context.get("synthesis_responses")
        synthesis_responses.append({
            "table_name": ev.table_name,
            "response": response.text
        })
        await context.set("synthesis_responses", synthesis_responses)
        
        # Get current index and all prompts
        current_index = await context.get("current_synthesis_index")
        synthesis_prompts = await context.get("synthesis_prompts")
        
        # Increment synthesis index
        current_index += 1
        await context.set("current_synthesis_index", current_index)
        
        # Check if we've processed all synthesis prompts
        if current_index >= len(synthesis_prompts):
            logger.info("All synthesis prompts processed, moving to final processing")
            
            enrichment_responses = await context.get("enrichment_responses")
            return AllEnrichmentsEvent(
                enrichment_responses=enrichment_responses,
                synthesis_responses=synthesis_responses
            )
        
        # Process next synthesis prompt
        next_prompt = synthesis_prompts[current_index]
        return SynthesisRequestEvent(
            table_name=next_prompt.get("table_name", "") or next_prompt.get("column_key", ""),
            prompt=next_prompt["prompt"]
        )

    @step
    async def Finalize_enrichment(self, context: Context, ev: AllEnrichmentsEvent) -> StopEvent:
        """Finalize the enrichment process and create the final output."""
        logger.info("Finalizing schema enrichment")
        
        # Get all responses
        enrichment_responses = ev.enrichment_responses
        synthesis_responses = ev.synthesis_responses
        
        logger.info(f"Combining {len(enrichment_responses)} enrichment responses and {len(synthesis_responses)} synthesis responses")
        
        # Get final semantic descriptions
        final_descriptions = get_final_semantic_descriptions(enrichment_responses, synthesis_responses)
        
        logger.info(f"Created final descriptions for {len(final_descriptions)} tables")
        
        # Save to file
        output_file = "schema_enrichment_results.json"
        with open(output_file, "w") as f:
            json.dump(final_descriptions, f, indent=2)
        
        logger.info(f"Schema enrichment results saved to {output_file}")
        
        return StopEvent(result=final_descriptions)

# Function to run the workflow
def run_schema_enrichment(sqlite_path, llm):
    """Run the schema enrichment workflow with the given SQLite path and LLM."""
    workflow = SchemaEnrichmentWorkflow(llm=llm)
    
    # Build arguments for the workflow
    args = {"sqlite_path": sqlite_path}
    
    # Run the workflow
    result = workflow.run(**args)
    
    return result

# Example usage
if __name__ == "__main__":
    # Path to SQLite file
    sqlite_path = r"E:\Workspace\Repositories\thesis\test\pipeline\SPIDER\database\bike_1\bike_1.sqlite"
    
    # Run the workflow using the centralized LLM configuration
    result = run_schema_enrichment(sqlite_path, llm_config.get_llm())
    
    print(result)
    print("Schema enrichment complete!")
    print(f"Enriched {len(result)} tables with semantic descriptions")