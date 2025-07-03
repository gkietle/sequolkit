from llama_index.core import PromptTemplate

SCHEMA_ENRICHMENT_SKELETON = (
    "You are a database schema analyst. Your task is to analyze the database schema and provide a brief and clear semantic description for each table and column.\n\n"
    "### Database description: {database_description}\n"
    "### Database schema:\n"
    "{schema}\n\n"
    "### Instructions:\n"
    "1. Analyze each table and column's semantic purpose\n"
    "2. Use brief and clear semantic descriptions for each table and column\n"
    "3. Format as JSON exactly as shown below\n\n"
    "4. If adding sample data to the description to clarify the meaning of the column, make sure it is relevant to the existing values in the column\n"
    "### Response Format:\n"
    "Return only the JSON output with no additional text."
)

DATABASE_DESCRIPTION_SKELETON = (
    "You are a knowledgeable database schema analyst. Given a database schema consisting of table definitions, "
    "your task is to generate a concise description of the database.\n\n"
    "### Database schema:\n"
    "{schema}\n\n"
    "### Instructions:\n"
    "1. Analyze the table and column names to infer the purpose of each table.\n"
    "2. Identify any potential relationships between tables (e.g., shared column names, foreign key patterns).\n"
    "3. Summarize the overall purpose of the database in 2 to 4 clear sentences.\n"
    "4. Do not repeat the schema or include headings or bullet points.\n"
    "5. Write only the brief description, as if explaining the database's purpose to a technical stakeholder.\n\n"
    "### Response Format:\n"
    "Return only the JSON output with no additional text."
)


TEXT_TO_SQL_SKELETON = (
    "You are a professional Database Engineer expert in {dialect} SQL. Generate a syntactically correct {dialect} SQL query for the given input question. Only return the {dialect} SQL query, no explanations.\n\n"
    "### User question: {user_question}\n"
    "### Database description: {database_description}\n"
    "### Database schema:\n"
    "{table_schemas}\n\n"
    "### Requirements:\n"
    "1. Write a precise SQL query that answers the user's question exactly\n"
    "2. Use only the tables and columns provided above\n"
    "3. Follow standard SQL syntax compatible with {dialect}\n"
    "4. Include proper JOINs when data needs to be combined from multiple tables\n"
    "5. Use appropriate aggregation functions (COUNT, SUM, AVG, etc.) when needed\n"
    "6. Use set operations (UNION, INTERSECT, EXCEPT) when needed\n"
    "7. **DO NOT** use `= NULL` in your query, use `IS NULL` instead.\n"
    "8. **MUST** masking all the values belong to the column that might be sensitive in the query (e.g. phone number, email, password, etc.). DO NOT masking the column names.\n"
    "9. Ensure your query is efficient and follows best practices\n"
    "10. Return ONLY the SQL query without any additional text, comments, or explanations\n"
    "\n\n"
    "### Response Format:\n"
    "Return only the {dialect} SQL query with no additional text."
)

TEXT_TO_SQL_SKELETON_FINETUNED = (
    "-- Database description: {database_description}\n"
    "{table_schemas}\n"
    "-- Using valid {dialect} SQL, answer the following questions for the tables provided above.\n"
    "Question: {user_question}\n"
)

TABLE_RETRIEVAL_SKELETON = (
    "You are a database schema analyst. Your task is to identify the all potentially relevant tables for the given question.\n\n"
    "### Database description: {database_description}\n"
    "### Database schema:\n"
    "{schema}\n\n"
    "### Instructions:\n"
    "0. Analyze the question intent and the database description to determine if the question is **AMBIGUOUS** or **NOT RELATED TO THE DATABASE**. Return an empty list. Do not try to answer the question if it is ambiguous\n"
    "1. Include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.\n"
    "2. Return ONLY a Python list of table names in the format: ['table_name1', 'table_name2', 'table_name3']\n"
    "3. Do not include any explanations, additional text, or markdown formatting.\n"
    "4. If no tables are relevant or the question is not related to the database, return an empty list: []\n"
    "5. Make sure your response can be directly parsed as a Python list.\n"
    "\n"
    "### Question: {query}\n"
    "### Response Format:\n"
    "Return only the Python list with no additional text."
)

QUERY_REFINEMENT_SKELETON = (
    "Translate the user's question from its original language to English without altering its original meaning. If—and only if—the translated question directly relates to the provided database schema for a Text-to-SQL task, refine the translation slightly to match the schema clearly and concisely. Do not introduce any additional details or modifications unrelated to the user's original intent.\n\n"
    "### User question: {user_question}\n"
    "### Database schema: {database_schema}\n"
    "### Rules:\n"
    "1. Translate the user's question from its original language to English without altering its original meaning.\n"
    "2. If—and only if—the translated question directly relates to the provided database schema for a Text-to-SQL task, refine the translation slightly to match the schema clearly and concisely. Do not introduce any additional details or modifications unrelated to the user's original intent.\n"
    "### Response Format:\n"
    "Return only the rewritten English question with no additional text."
)

SQL_ERROR_REFLECTION_SKELETON = (
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

SQL_JUDGER_SKELETON = (
    "You are a {dialect} SQL expert. Given a user query and a SQL query, determine if the SQL generated appropriately answers the given user question taking into account its generated query and response.\n\n"
    "### Database description: {database_description}\n"
    "### Database schema:\n"
    "{schema}\n\n"
    "### Section description:\n"
    "1. User query: This section contains the specific question/task/problem that the sql query is intended to solve.\n"
    "2. SQL query: This is the {dialect} SQL query submitted for evaluation. Analyze it in the context of provided instruction.\n"
    "3. SQL response: This is the first {num_rows} rows of the result set generated after executing the SQL query in the database.\n"
    "### User query: {user_query}\n"
    "### SQL query: {sql_query}\n"
    "### SQL response: {sql_response}\n"
    "### Instructions:\n"
    "1. Analyze the SQL query in the context of the user query.\n"
    "2. Determine if the SQL query is appropriate for the user query.\n"
    "3. If the SQL query is correctly solving the user query, return True. If it is not, return False.\n"
    "4. "
    "### Response Format:\n"
    "Return only the boolean value (True or False) with no additional text."
)

QUESTION_SUGGESTION_SKELETON = (
    "You are a database analyst who specializes in understanding data models and suggesting insightful questions "
    "that can be answered using SQL queries against a database.\n\n"
    "### Database Description:\n{database_description}\n\n"
    "### Database Schema:\n{schema}\n\n"
    "### Instructions:\n"
    "1. Analyze the database schema and understand the relationships between tables\n"
    "2. Generate {top_k} unique, insightful, and specific questions that can be answered using SQL queries against this database\n"
    "3. Focus on questions that:\n"
    "   - Provide valuable business insights that related to the tables provided\n"
    "   - Question MUST BE short (less than 9 words) and concise\n"
    "   - Require joining multiple tables\n"
    "   - Involve aggregations, grouping, or analytical operations. Restrict filter-related questions (WHERE clause).\n"
    "   - Cover different aspects of the database\n"
    "   - Are specific rather than general\n"
    "   - Make sure the questions can be answered using SQL queries against the database and have few rows in the result\n"
    "### Response Format:\n"
    "Return only valid JSON with 'questions' array, with no additional text."
)


def text2sql_prompt_routing(prompt_type: int) -> str:
    if prompt_type == 0:
        prompt = TEXT_TO_SQL_SKELETON
    elif prompt_type == 1:
        prompt = TEXT_TO_SQL_SKELETON_FINETUNED
    else:
        raise ValueError("Invalid prompt type")

    return prompt
