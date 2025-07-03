from llama_index.core.workflow import Event

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

class QuestionSuggestionEvent(Event):
    """Event for generating question suggestions based on schema."""
    table_details: list[object]
    database_description: str = ""
    top_k: int = 5 