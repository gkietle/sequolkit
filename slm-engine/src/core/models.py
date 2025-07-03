from pydantic import BaseModel, Field
from typing import List, Optional


class TranslatedQuery(BaseModel):
    translated_query: str = Field(..., description="The translated question in English")

class ListOfRelevantTables(BaseModel):
    relevant_tables: List[str] = Field(..., description="The list of ***ALL POTENTIALLY RELEVANT*** tables to use in the SQL query")

class SQLQuery(BaseModel):
    """Model for SQL query generation and correction."""
    sql_query: str = Field(..., description="The SQL query to execute")
    
    # @classmethod
    # def model_json_schema(cls, *args, **kwargs):
    #     # Customize schema to ensure the model can extract the query from different LLM response formats
    #     schema = super().model_json_schema(*args, **kwargs)
    #     # Add examples to help the LLM understand the expected output format
    #     schema["examples"] = [
    #         {"sql_query": "SELECT * FROM users WHERE age > 18"},
    #         {"sql_query": "WITH user_orders AS (SELECT u.id, u.name, o.order_date FROM users u JOIN orders o ON u.id = o.user_id) SELECT * FROM user_orders"}
    #     ]
    #     return schema

class DatabaseDescription(BaseModel):
    database_description: str = Field(..., description="The brief description of the database")

class ColumnDescription(BaseModel):
    column_name: str = Field(..., description="The name of the column")
    description: str = Field(..., description="The brief description of the column")

class TableDescription(BaseModel):
    table_name: str = Field(..., description="The name of the table")
    description: str = Field(..., description="The brief description of the table")
    columns: List[ColumnDescription] = Field(..., description="The list of columns in the table")

class SchemaEnrichmentResponse(BaseModel):
    tables: List[TableDescription] = Field(..., description="The list of tables with their enriched descriptions") 