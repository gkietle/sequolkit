from llama_index.core import PromptTemplate

SCHEMA_ENRICHMENT_PROMPT = (
    "# Task: Analyze database schema semantics\n\n"
    "## Schema Information:\n"
    "Database context: {db_info}\n\n"
    "Tables and columns:\n"
    "{cluster_info}\n\n"
    "## Instructions:\n"
    "1. Analyze each table and column's semantic purpose\n"
    "2. Use brief and clear semantic descriptions for each table and column\n"
    "3. Format as JSON exactly as shown below\n\n"
    "## Response Format:\n"
    "```json\n"
    "[\n"
    "  {{\"table_name\": \"TABLE1\", \"description\": \"brief table purpose\", \"columns\": [\n"
    "    {{\"column_name\": \"COLUMN1\", \"description\": \"brief column purpose\"}},\n"
    "    {{\"column_name\": \"COLUMN2\", \"description\": \"brief column purpose\"}}\n"
    "  ]}},\n"
    "  {{\"table_name\": \"TABLE2\", \"description\": \"brief table purpose\", \"columns\": [\n"
    "    {{\"column_name\": \"COLUMN1\", \"description\": \"brief column purpose\"}}\n"
    "  ]}}\n"
    "]\n"
    "```\n\n"
    "Return only the JSON output with no additional text."
)
SCHEMA_ENRICHMENT_TMPL = PromptTemplate(SCHEMA_ENRICHMENT_PROMPT)

DATABASE_DESCRIPTION_PROMPT = (
    "You are a knowledgeable database schema analyst. Given a database schema consisting of table definitions, "
    "your task is to generate a concise description of the database.\n\n"
    "Database schema:\n"
    "{schema}\n\n"
    "Instructions:\n"
    "1. Analyze the table and column names to infer the purpose of each table.\n"
    "2. Identify any potential relationships between tables (e.g., shared column names, foreign key patterns).\n"
    "3. Summarize the overall purpose of the database in 2 to 4 clear sentences.\n"
    "4. Do not repeat the schema or include headings or bullet points.\n"
    "5. Write only the brief description, as if explaining the database's purpose to a technical stakeholder.\n\n"
    "Database description:\n"
)
DATABASE_DESCRIPTION_TMPL = PromptTemplate(DATABASE_DESCRIPTION_PROMPT)


TEXT_TO_SQL_PROMPT = (
    "You are a {dialect} SQL expert. Generate a SQL query based on the user's question. Only return the SQL query, no explanations.\n"
    "User question: {user_question}\n"
    "Database description: {database_description}\n"
    "Available tables and their schema:\n"
    "{table_schemas}\n"
    "Requirements:\n"
    "1. Write a precise SQL query that answers the user's question exactly\n"
    "2. Use only the tables and columns provided above\n"
    "3. Follow standard SQL syntax compatible with {dialect}\n"
    "4. Include proper JOINs when data needs to be combined from multiple tables\n"
    "5. Use appropriate aggregation functions (COUNT, SUM, AVG, etc.) when needed\n"
    "6. Ensure your query is efficient and follows best practices\n"
    "7. Return ONLY the SQL query without any additional text, comments, or explanations\n"
    "8. If you can't answer the question with the given tables, return \"Cannot generate SQL with available schema\"\n"
    "SQL query:\n"
)

TEXT_TO_SQL_PROMPT_FINETUNED = (
    "-- Database description: {database_description}\n"
    "{table_schemas}\n"
    "-- Using valid {dialect} SQL, answer the following questions for the tables provided above.\n"
    "Question: {user_question}\n"
)

def text2sql_prompt_routing(prompt_type: int) -> PromptTemplate:
    if prompt_type == 0:
        prompt = TEXT_TO_SQL_PROMPT
    elif prompt_type == 1:
        prompt = TEXT_TO_SQL_PROMPT_FINETUNED
    else:
        raise ValueError("Invalid prompt type")
    
    return PromptTemplate(prompt)


TABLE_RETRIEVAL_PROMPT = (
    "Return ONLY the names of SQL tables that MIGHT be relevant to the user question.\n"
    "The question is: {query_str}\n"
    "The database description is: {database_description}\n"
    "The tables are as following format - [table_name (table_description)]:\n\n"
    "{table_names}\n\n"
    "Instructions:\n"
    "1. Include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.\n"
    "2. Return ONLY a Python list of table names in the format: ['table_name1', 'table_name2', 'table_name3']\n"
    "3. Do not include any explanations, additional text, or markdown formatting.\n"
    "4. If no tables are relevant, return an empty list: []\n"
    "5. Make sure your response can be directly parsed as a Python list.\n"
)
TABLE_RETRIEVAL_TMPL = PromptTemplate(TABLE_RETRIEVAL_PROMPT)

TABLE_EXTRACTION_PROMPT = (
    "You are a {dialect} SQL expert. Extract all tables referenced in the SQL query and return them as a Python list.\n"
    "SQL query: {sql_query}\n"
    "Requirements:\n"
    "1. Identify ALL tables referenced in the query\n"
    "2. Include tables from FROM clauses, JOIN statements, subqueries, and CTEs\n"
    "3. Return ONLY a valid Python list of table names as strings\n"
    "4. Format the response exactly as: [\"table1\", \"table2\", \"table3\"]\n"
    "5. Return an empty list [] if no tables are found\n"
    "6. Do not include any explanations, comments, or additional text\n"
    "7. Ensure table names are extracted exactly as they appear in the query\n"
    "8. If table has an alias, only include the original table name, not the alias\n"
    "Python list of tables:\n"
)
TABLE_EXTRACTION_TMPL = PromptTemplate(TABLE_EXTRACTION_PROMPT)

SQL_ERROR_REFLECTION_PROMPT = (
    "You are a {dialect} SQL expert. Reflect on the given SQL query and error message to determine the cause of the error and suggest a possible solution.\n\n"
    "# User query: {user_query}\n"
    "# Database description: {database_description}\n"
    "# Database schema: {database_schema}\n"
    "# Original SQL query: {sql_query}\n"
    "# Error message: {error_message}\n\n"
    "Requirements:\n"
    "1. Analyze the error and determine the cause\n"
    "2. Fix the SQL query to resolve the error\n"
    "3. Ensure compatibility with the database schema provided\n"
    "4. Return ONLY the corrected SQL query with no explanations or comments\n"
    "5. Maintain the original query's intent while fixing the syntax or logical errors\n"
    "6. Follow standard SQL syntax compatible with the database type\n"
    "7. If the query cannot be fixed with the given information, return a simplified valid query\n"
    "8. Do not include any text before or after the SQL query\n"
    "Corrected SQL query without explanations:\n"
)
SQL_ERROR_REFLECTION_TMPL = PromptTemplate(SQL_ERROR_REFLECTION_PROMPT)