import ast
import re
import logging
from typing import List, Any
from llama_index.core.llms import ChatResponse
import networkx as nx
import community as community_louvain
# Configure logging
logger = logging.getLogger(__name__)

    
def extract_table_list(response: ChatResponse) -> list:
    """Extracts a list of table names from a response string that may contain code blocks."""
    # First, strip any code block formatting
    cleaned_response = response.message.content

    think_pattern = r'<think>[\s\S]*?</think>'
    think_match = re.search(think_pattern, cleaned_response)
    
    if think_match:
        # Only process text after the </think> tag
        cleaned_response = cleaned_response[think_match.end():].strip()
    
    # Remove markdown code block if present
    code_block_pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    code_block_match = re.search(code_block_pattern, cleaned_response)
    
    if code_block_match:
        # Use the content inside the code block
        cleaned_response = code_block_match.group(1).strip()
    
    try:
        # Try to parse as a Python literal
        return ast.literal_eval(cleaned_response)
    except (SyntaxError, ValueError):
        # Fallback: look for strings in single quotes
        pattern = r"'([^']+)'"
        matches = re.findall(pattern, cleaned_response)
        if matches:
            return matches
            
        # Try double quotes if no single quotes found
        pattern = r'"([^"]+)"'
        matches = re.findall(pattern, cleaned_response)
        if matches:
            return matches
            
        # If no quotes found, split by commas and clean up
        items = [item.strip().strip("'\"") for item in cleaned_response.split(',')]
        return [item for item in items if item]


def extract_tables_from_sql(sql_query: str) -> list[str]:
    """
    Extract all table names from a SQL query, excluding tables defined in CTEs.
    
    Args:
        sql_query (str): The SQL query to analyze
        
    Returns:
        List[str]: List of table names referenced in the query (excluding CTE tables)
    """
    # Normalize whitespace and line breaks
    sql_query = " ".join(sql_query.split())
    
    # Step 1: Identify and extract all CTE names to exclude them later
    cte_pattern = r"WITH\s+(?:RECURSIVE\s+)?([^(].*?)\s+AS\s*\("
    cte_tables = []
    
    # Find CTEs at the start of the query
    cte_match = re.search(cte_pattern, sql_query, re.IGNORECASE)
    if cte_match:
        cte_part = cte_match.group(1)
        # Handle multiple CTEs separated by commas
        cte_sections = cte_part.split(',')
        for section in cte_sections:
            # Extract the CTE name
            cte_name_match = re.search(r"^\s*(\w+)(?:\s*\(.*?\))?\s*$", section.strip())
            if cte_name_match:
                cte_tables.append(cte_name_match.group(1).lower())
    
    # Also handle the case of multiple CTEs defined using comma and the AS keyword
    cte_segments_pattern = r"WITH(?:\s+RECURSIVE)?\s+.*?(?:,\s*(\w+)\s+AS\s*\()"
    for cte_segment in re.finditer(cte_segments_pattern, sql_query, re.IGNORECASE):
        if cte_segment.group(1):
            cte_tables.append(cte_segment.group(1).lower())
    
    # Step 2: Extract tables from FROM and JOIN clauses
    # Pattern for FROM clause - look for table names after FROM keyword
    from_pattern = r"FROM\s+([^\s,();]*)(?:\s+AS\s+\w+)?(?:\s*,\s*([^\s,();]*)(?:\s+AS\s+\w+)?)*"
    
    # Pattern for JOIN clauses
    join_pattern = r"JOIN\s+([^\s,();]*)(?:\s+AS\s+\w+)?"
    
    # Find all tables in FROM clauses
    table_names = []
    for match in re.finditer(from_pattern, sql_query, re.IGNORECASE):
        # The first group contains the first table
        if match.group(1) and match.group(1).strip():
            table_name = re.sub(r'^\(|\)$', '', match.group(1).strip())
            if "." in table_name:
                # Extract just the table name if schema.table format is used
                table_name = table_name.split('.')[-1]
            table_names.append(table_name.lower())
        
        # Check if we have multiple tables in the FROM clause
        rest_of_from = match.group(0)[len("FROM ") + len(match.group(1)):].strip()
        if rest_of_from.startswith(','):
            # Split by comma to get additional tables
            additional_tables = re.findall(r',\s*([^\s,();]*)', rest_of_from)
            for table in additional_tables:
                if table and table.strip():
                    table_name = re.sub(r'^\(|\)$', '', table.strip())
                    if "." in table_name:
                        table_name = table_name.split('.')[-1]
                    table_names.append(table_name.lower())
    
    # Find all tables in JOIN clauses
    for match in re.finditer(join_pattern, sql_query, re.IGNORECASE):
        if match.group(1) and match.group(1).strip():
            table_name = re.sub(r'^\(|\)$', '', match.group(1).strip())
            if "." in table_name:
                table_name = table_name.split('.')[-1]
            table_names.append(table_name.lower())
    
    # Step 3: Filter out CTE tables, SQL keywords, and SQL functions
    sql_keywords = {'select', 'where', 'group', 'order', 'limit', 'offset', 'having', 'union', 'intersect', 'except'}
    sql_functions = {'extract', 'sum', 'count', 'avg', 'min', 'max', 'date', 'year', 'month', 'day'}
    filtered_tables = []
    
    # First, identify all column references in EXTRACT functions
    extract_column_refs = set()
    extract_pattern = r'extract\s*\(\s*\w+\s+from\s+(\w+)'
    for match in re.finditer(extract_pattern, sql_query, re.IGNORECASE):
        if match.group(1):
            extract_column_refs.add(match.group(1).lower())
    
    for table in table_names:
        # Skip empty strings
        if not table:
            continue
            
        # Skip CTEs
        if table.lower() in cte_tables:
            continue
            
        # Skip if it's just a SQL keyword
        if table.lower() in sql_keywords:
            continue
            
        # Skip if it's a SQL function
        if table.lower() in sql_functions:
            continue
            
        # Skip if it's a subquery (likely starts with SELECT)
        if table.lower().startswith('select'):
            continue
            
        # Skip if it's a column reference in an EXTRACT function
        if table.lower() in extract_column_refs:
            continue
            
        filtered_tables.append(table)
    
    # Remove duplicates while preserving order
    unique_tables = []
    for table in filtered_tables:
        if table not in unique_tables:
            unique_tables.append(table)
    
    return unique_tables

def schema_clustering(table_details: list, resolution_value = 1.0) -> list:
    """
    Create table relationship arrays with full table details in each cluster.
    
    Args:
        table_details: Danh sách thông tin chi tiết của các bảng
        resolution_value: Tham số resolution cho thuật toán Louvain (mặc định: 1.0)
        
    Returns:
        Danh sách các cụm, mỗi cụm chứa thông tin đầy đủ của các bảng
    """
    try:
        import networkx as nx
        import community as community_louvain
        
        if not table_details:
            return []
        
        # If there are fewer than 5 tables, return one cluster with all tables
        if len(table_details) < 5:
            # Sort tables alphabetically by tableIdentifier
            sorted_tables = sorted(table_details, key=lambda x: x['tableIdentifier'])
            return [sorted_tables]
        
        # Tạo từ điển để lưu thông tin đầy đủ của bảng, với key là tableIdentifier
        table_info_map = {table['tableIdentifier']: table for table in table_details}
        
        # Lấy danh sách các tên bảng
        tables = list(table_info_map.keys())

        # Trích xuất các khóa ngoại
        foreign_keys = []
        for table_info in table_details:
            source_table = table_info['tableIdentifier']
            for column_info in table_info.get('columns', []):
                if column_info.get('relations'): 
                    for relation in column_info['relations']:
                        target_table = relation.get('tableIdentifier')
                        if target_table and source_table in tables and target_table in tables and source_table != target_table:
                            foreign_keys.append((source_table, target_table))
        
        # Loại bỏ các cặp trùng lặp
        foreign_keys = list(set(foreign_keys))

        # Xây dựng đồ thị
        G = nx.Graph()
        G.add_nodes_from(tables)
        G.add_edges_from(foreign_keys)

        if G.number_of_nodes() == 0:
            return []

        # Áp dụng thuật toán Louvain để phân cụm
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(G, resolution=resolution_value)
        else:
            partition = {node: i for i, node in enumerate(G.nodes())}
            
        # Xử lý kết quả phân cụm
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        # Sắp xếp các cụm theo kích thước
        sorted_community_ids = sorted(communities.keys(), 
                                      key=lambda k: len(communities[k]), 
                                      reverse=True)
        
        # Tạo danh sách các cụm với thông tin đầy đủ của bảng
        full_info_clusters = []
        
        for community_id in sorted_community_ids:
            # Lấy danh sách tên bảng trong cụm
            table_ids = communities[community_id]
            # Sắp xếp theo alphabet
            sorted_table_ids = sorted(table_ids)
            
            # Lấy thông tin đầy đủ cho mỗi bảng
            cluster_with_full_info = [table_info_map[table_id] for table_id in sorted_table_ids]
            full_info_clusters.append(cluster_with_full_info)
            
        return full_info_clusters

    except Exception as e:
        print(f"Error in schema_clustering: {e}")
        return []



def schema_parser(tables: list, type: str, include_sample_data: bool = False):
    """
    Phân tích cấu trúc schema và tạo ra các câu lệnh mô tả theo định dạng được chỉ định.
    
    Args:
        tables: Danh sách các bảng cùng thông tin cột và quan hệ
        type: Loại định dạng đầu ra ("DDL", "Synthesis", hoặc "Simple")
        include_sample_data: Có hiển thị dữ liệu mẫu hay không (mặc định: False)
        
    Returns:
        Chuỗi mô tả schema theo định dạng đã chọn
    """
    if type not in ["DDL", "Synthesis", "Simple"]:
        raise Exception("Invalid schema parser type. Must be 'DDL', 'Synthesis', or 'Simple'.")

    if type == "DDL":
        ddl_statements = []
        fk_relationships = []
        
        # First create all tables
        for table in tables:
            table_name = table["tableIdentifier"]
            columns = table["columns"]

            # Create table definition
            column_definitions = []
            
            for column in columns:
                column_def = f"{column['columnIdentifier']} {column['columnType']}"
                if column.get("isPrimaryKey"):
                    column_def += " PRIMARY KEY"
                description = column.get("columnDescription", None)
                if description:
                    if column == columns[-1]:
                        column_def += f" -- {description}"
                    else:
                        column_def += f", -- {description}"
                
                column_definitions.append(column_def)
                
                # Collect foreign key relationships separately
                if "relations" in column and column["relations"]:
                    for relation in column["relations"]:
                        if relation.get("type") == "OTM":  # One-to-Many relationship
                            fk_relation = f"{table_name}.{column['columnIdentifier']} →  {relation['tableIdentifier']}.{relation['toColumn']}"
                            fk_relationships.append(fk_relation)

            # Remove trailing comma from the last column definition
            if column_definitions:
                column_definitions[-1] = column_definitions[-1].rstrip(',')
            
            # Combine into CREATE TABLE statement
            ddl = f"CREATE TABLE {table_name} (\n    " + \
                  "\n    ".join(column_definitions) + "\n);"
            ddl_statements.append(ddl)
            
            # Add sample data if available and requested
            if include_sample_data and "sample_data" in table and table["sample_data"]:
                ddl_statements.append("\n-- Sample Data:")
                # Add column headers
                column_headers = ", ".join([column["columnIdentifier"] for column in columns])
                ddl_statements.append(f"{column_headers}")
                # Add data rows
                for data_row in table["sample_data"]:
                    ddl_statements.append(f"{data_row}")
                ddl_statements.append("")  # Empty line for better readability

        # Add all foreign key relationships at the end
        if fk_relationships:
            ddl_statements.append("-- Foreign Key Relationships:")
            ddl_statements.append("\n".join(fk_relationships))

        return "\n".join(ddl_statements)

    elif type == "Synthesis":
        synthesis_statements = []
        fk_relationships = []
        
        for table in tables:
            table_name = table["tableIdentifier"]
            columns = table["columns"]

            # Generate synthesis description
            column_descriptions = []
            for column in columns:
                description = column.get("columnDescription", "No description available")
                pk_info = " (Primary Key)" if column.get("isPrimaryKey") else ""
                
                # Collect foreign key relationships separately
                if "relations" in column and column["relations"]:
                    for relation in column["relations"]:
                        fk_relation = f"{table_name}.{column['columnIdentifier']} → {relation['tableIdentifier']}.{relation['toColumn']}"
                        fk_relationships.append(fk_relation)
                
                column_descriptions.append(
                    f"{column['columnIdentifier']} ({column['columnType']}){pk_info}: {description}")

            synthesis = f"Table: {table_name}\nColumns:\n    " + \
                "\n    ".join(column_descriptions)
            synthesis_statements.append(synthesis)
            
            # Add sample data if available and requested
            if include_sample_data and "sample_data" in table and table["sample_data"]:
                synthesis_statements.append("\nSample Data:")
                # Add column headers
                column_headers = ", ".join([column["columnIdentifier"] for column in columns])
                synthesis_statements.append(f"    {column_headers}")
                # Add data rows
                for data_row in table["sample_data"]:
                    synthesis_statements.append(f"    {data_row}")
                synthesis_statements.append("")  # Empty line for better readability

        # Add all foreign key relationships at the end
        if fk_relationships:
            synthesis_statements.append("\n\nForeign Key Relationships:")
            synthesis_statements.append("\n".join(fk_relationships))

        return "\n".join(synthesis_statements)
    
    elif type == "Simple":
        simple_statements = []
        fk_relationships = []
        
        for table in tables:
            table_name = table["tableIdentifier"]
            columns = table["columns"]

            columns_chain = []
            for column in columns:
                column_name = column["columnIdentifier"]
                column_type = column["columnType"]
                pk_info = " [PK]" if column.get("isPrimaryKey") else ""
                
                columns_chain.append(f"{column_name} {column_type}{pk_info}")
                
                # Collect foreign key relationships separately
                if "relations" in column and column["relations"]:
                    for relation in column["relations"]:
                        fk_relation = f"{table_name}.{column['columnIdentifier']} →  {relation['tableIdentifier']}.{relation['toColumn']}"
                        fk_relationships.append(fk_relation)
            
            simple_statements.append(f"{table_name} ({', '.join(columns_chain)})")
            
            # Add sample data if available and requested
            if include_sample_data and "sample_data" in table and table["sample_data"]:
                simple_statements.append("Sample Data:")
                # Add column headers
                column_headers = ", ".join([column["columnIdentifier"] for column in columns])
                simple_statements.append(f"  {column_headers}")
                # Add data rows
                for data_row in table["sample_data"]:
                    simple_statements.append(f"  {data_row}")
                simple_statements.append("")  # Empty line for better readability

        # Add all foreign key relationships at the end
        if fk_relationships:
            simple_statements.append("\n\nForeign Key Relationships:")
            simple_statements.append("\n".join(fk_relationships))

        return "\n".join(simple_statements)

def log_prompt(prompt_messages: List[Any], step_name: str) -> None:
    """
    Logs the formatted prompt messages with detailed formatting.
    
    Args:
        prompt_messages (list): A list of ChatMessage objects containing the prompt
        step_name (str): The name of the current workflow step
    """
    logger.info(f"\033[95m===== {step_name} PROMPT =====\033[0m")
    for i, message in enumerate(prompt_messages):
        logger.info(f"\033[96mRole: {message.role}\033[0m")
        for block in message.blocks:
            if block.block_type == "text":
                # Truncate very long content for log readability
                content = block.text
                logger.info(f"\033[97mContent:\n{content}\033[0m")
        if i < len(prompt_messages) - 1:
            logger.info("\033[95m---------------------\033[0m")
    logger.info(f"\033[95m===== END {step_name} PROMPT =====\033[0m")

def show_prompt(prompt_messages: List[Any]) -> None:
    """
    Displays the formatted prompt messages in a readable format.
    This is a more console-friendly version that uses print instead of logging.

    Args:
        prompt_messages (list): A list of ChatMessage objects containing the prompt.
    """
    print("\nFormatted Prompt Messages:")
    print("-" * 80)
    for message in prompt_messages:
        print(f"Role: {message.role}")
        for block in message.blocks:
            if block.block_type == "text":
                print(f"Content:\n{block.text}")
        print("-" * 80)  # Separator for readability

def extract_sql_query(response_text):
        """
        Extract SQL query from LLM response and return as a single line.
        
        Args:
            response_text: The LLM response text that may contain a SQL query
            
        Returns:
            A single line SQL query without any additional elements
        """
        think_pattern = r'<think>[\s\S]*?</think>'
        think_match = re.search(think_pattern, response_text)
        
        if think_match:
            # Only process text after the </think> tag
            response_text = response_text[think_match.end():].strip()
        
        # Try to extract code blocks with sql, SQL, or no language specified
        sql_pattern = r"```(?:sql|SQL)?\s*([\s\S]*?)```"
        sql_matches = re.findall(sql_pattern, response_text)
        
        if sql_matches:
            # Take the first match if multiple code blocks
            sql = sql_matches[0].strip()
        else:
            # Check for complete SQL query ending with semicolon followed by non-SQL text
            semicolon_split_pattern = r"(SELECT[\s\S]+?;)\s*\w+"
            semicolon_split_match = re.search(semicolon_split_pattern, response_text, re.IGNORECASE)
            
            if semicolon_split_match:
                # Extract the SQL part ending with semicolon
                sql = semicolon_split_match.group(1).strip()
            else:
                # If no code blocks with ``` are found, try to extract the full query differently
                
                # First attempt: look for complete queries with nested subqueries and proper formatting
                # This complex pattern captures SQL with nested parentheses, conditions, etc.
                complex_sql_pattern = r"SELECT[\s\S]+?FROM[\s\S]+?(?:WHERE[\s\S]+?)?(?:GROUP BY[\s\S]+?)?(?:HAVING[\s\S]+?)?(?:ORDER BY[\s\S]+?)?(?:LIMIT\s+\d+)?(?:OFFSET\s+\d+)?(?:;|$)"
                complex_matches = re.findall(complex_sql_pattern, response_text, re.IGNORECASE)
                
                if complex_matches:
                    sql = complex_matches[0].strip()
                else:
                    # Second attempt: try to find a complete SQL statement with semicolon
                    semicolon_pattern = r"SELECT[\s\S]+?;|INSERT[\s\S]+?;|UPDATE[\s\S]+?;|DELETE[\s\S]+?;|CREATE[\s\S]+?;|DROP[\s\S]+?;|ALTER[\s\S]+?;"
                    semicolon_matches = re.findall(semicolon_pattern, response_text, re.DOTALL | re.IGNORECASE)
                    
                    if semicolon_matches:
                        # Take the first complete SQL statement with semicolon
                        sql = semicolon_matches[0].strip()
                    else:
                        # Try a simple approach - collect all lines between SELECT and the end of the query
                        lines = response_text.split('\n')
                        sql_lines = []
                        in_sql = False
                        
                        for line in lines:
                            # Start collecting when we see SELECT
                            if re.search(r'\bSELECT\b', line, re.IGNORECASE) and not in_sql:
                                in_sql = True
                                sql_lines.append(line)
                            # Continue collecting if we're in SQL mode
                            elif in_sql:
                                # Stop if we hit an empty line after collecting some SQL or see end markers
                                if (not line.strip() and len(sql_lines) > 3) or re.search(r'\bEXPLAIN\b|\bANALYZE\b', line, re.IGNORECASE):
                                    break
                                sql_lines.append(line)
                        
                        if sql_lines:
                            sql = '\n'.join(sql_lines)
                        else:
                            # If all else fails, fall back to keyword search
                            sql_keywords = r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|DESCRIBE)"
                            potential_sql_lines = re.findall(fr"(?m)^.*{sql_keywords}.*$", response_text)
                            
                            if potential_sql_lines:
                                # Join potential SQL lines
                                sql = " ".join(line.strip() for line in potential_sql_lines)
                            else:
                                # If nothing looks like SQL, return a default query
                                print("No valid SQL query found in the response, returning default query.")
                                return "SELECT 0;"
        
        # Fix incomplete queries by checking for missing parts
        # 1. Is the query missing column parts?
        if "SELECT" in sql.upper() and "FROM" in sql.upper():
            # Extract SELECT part
            select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM)', sql, re.IGNORECASE | re.DOTALL)
            
            if select_match:
                select_columns = select_match.group(1).strip()
                
                # If SELECT part seems truncated (e.g., missing commas between selections)
                if ',' in select_columns and select_columns.count(',') < response_text.count(',') and select_columns.endswith(','):
                    # Try to find all selected columns from the original text
                    select_pattern = r'SELECT\s+(.*?)\s+FROM'
                    select_full_match = re.search(select_pattern, response_text, re.IGNORECASE | re.DOTALL)
                    
                    if select_full_match and len(select_full_match.group(1)) > len(select_columns):
                        # Replace only the SELECT part with the better match
                        sql = sql.replace(select_columns, select_full_match.group(1).strip())
        
        # Handle missing parts of the query by parsing the structure and ensuring completeness
        sql_upper = sql.upper()
        
        # Check if we're missing FROM in a SELECT query
        if "SELECT" in sql_upper and "FROM" not in sql_upper:
            from_pattern = r'FROM\s+\w+(?:\s+AS\s+\w+)?'
            from_match = re.search(from_pattern, response_text, re.IGNORECASE)
            
            if from_match:
                sql += " " + from_match.group(0)
        
        # Check if we're missing WHERE in a query that should have it
        if "WHERE" not in sql_upper and "WHERE" in response_text.upper():
            # Find the WHERE clause including any complex conditions and nested queries
            where_pattern = r'WHERE\s+[\s\S]+?(?:GROUP BY|ORDER BY|LIMIT|HAVING|;|$)'
            where_match = re.search(where_pattern, response_text, re.IGNORECASE)
            
            if where_match:
                where_clause = where_match.group(0)
                # Remove anything after the actual WHERE clause
                for ending in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING", ";"]:
                    if ending in where_clause.upper():
                        where_clause = where_clause[:where_clause.upper().find(ending)]
                        break
                
                sql += " " + where_clause.strip()
        
        # Handle subqueries by ensuring complete parentheses balance
        # Count opening and closing parentheses
        open_parens = sql.count('(')
        close_parens = sql.count(')')
        
        # If unbalanced, check original text for the complete subquery
        if open_parens > close_parens:
            # Find the point where we're missing closing parentheses
            for i in range(close_parens, open_parens):
                subquery_pattern = r'\([^()]*(?:\([^()]*\)[^()]*)*\)'  # Match balanced parentheses
                
                # Search for subqueries in the original text that might be missing
                subquery_candidates = re.findall(subquery_pattern, response_text)
                for candidate in subquery_candidates:
                    if candidate not in sql:
                        # We found a subquery that's missing from our extracted SQL
                        # Try to find where it fits
                        opening_pos = -1
                        for j, char in enumerate(sql):
                            if char == '(':
                                opening_pos = j
                                # Check if this is already balanced
                                if sql[j:].count('(') <= sql[j:].count(')'):
                                    continue
                                # Check if this opening might be part of our missing subquery
                                subquery_starts = candidate.find('(')
                                if subquery_starts == 0 and sql[j:j+10] in candidate[:10]:
                                    # This position looks like where our missing subquery fits
                                    sql = sql[:j] + candidate + sql[j+1:]
                                    break
        
        # Ensure all lines of a multi-line SQL statement are included
        if "SELECT" in sql.upper() and sql.strip().startswith("SELECT"):
            # Check if we're potentially missing parts of the query by comparing with original text
            sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
            present_keywords = [kw for kw in sql_keywords if kw in sql.upper()]
            
            # Find all occurrences of these keywords in the original text
            for i, keyword in enumerate(present_keywords):
                # Check if there should be something between this keyword and the next one
                if i < len(present_keywords) - 1:
                    current_pos = sql.upper().find(keyword)
                    next_pos = sql.upper().find(present_keywords[i+1])
                    
                    if current_pos >= 0 and next_pos >= 0:
                        # Extract what's between these keywords in our current SQL
                        current_content = sql[current_pos + len(keyword):next_pos].strip()
                        
                        # Find the same section in original text
                        orig_current_pos = response_text.upper().find(keyword)
                        orig_next_pos = response_text.upper().find(present_keywords[i+1])
                        
                        if orig_current_pos >= 0 and orig_next_pos >= 0:
                            orig_content = response_text[orig_current_pos + len(keyword):orig_next_pos].strip()
                            
                            # If original has more content, use it instead
                            if len(orig_content) > len(current_content) and not orig_content.startswith(current_content):
                                # Replace the section with the more complete version
                                sql = sql[:current_pos + len(keyword)] + " " + orig_content + " " + sql[next_pos:]
        
        # Clean up the final SQL statement
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # Try to grab any columns that might still be missing (check for S.Song_release_year pattern)
        if "SELECT" in sql.upper() and "FROM" in sql.upper():
            match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
            if match:
                select_part = match.group(1)
                # Check if the SELECT part appears truncated
                if "," in select_part and select_part.count(",") < response_text.count(","):
                    # Look for column patterns that might be missing
                    column_pattern = r'(?:[A-Za-z]\w*\.)?[A-Za-z]\w*(?:\s+AS\s+\w+)?'
                    original_columns = re.findall(column_pattern, response_text)
                    for col in original_columns:
                        # Check if it looks like a column reference but isn't in our SELECT
                        if '.' in col and col not in select_part:
                            # If it's not in a different part of the query (WHERE, etc.)
                            if col not in sql[sql.upper().find("FROM"):]:
                                select_part += f", {col}"
                    
                    # Update the SQL with the more complete SELECT part
                    sql = sql.replace(match.group(1), select_part)
        
        # Check for a complete SQL query ending with semicolon followed by explanation text
        if ";" in sql:
            sql = sql.split(";")[0] + ";"
        
        # If the result is empty or obviously not SQL, return the default
        if not sql or not any(keyword in sql.upper() for keyword in ["SELECT"]):
            print("Extracted content doesn't appear to be a valid SQL query, returning default query.")
            return "SELECT 0;"
            
        return sql


def parse_llm_json_response(response: str) -> dict:
    """
    Phân tích phản hồi có định dạng JSON từ LLM và chuyển đổi thành đối tượng Python.
    
    Args:
        response: Chuỗi phản hồi từ LLM có thể chứa JSON
        
    Returns:
        Dict hoặc List chứa dữ liệu JSON đã được phân tích, hoặc None nếu có lỗi
    """
    import re
    import json
    
    try:
        # Trường hợp 1: Phản hồi đã là JSON hợp lệ
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Trường hợp 2: Tìm kiếm khối code JSON
        json_block_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON code block
            r'```\s*([\s\S]*?)\s*```',       # Markdown code block không chỉ định ngôn ngữ
            r'`([\s\S]*?)`'                  # Markdown inline code
        ]
        
        for pattern in json_block_patterns:
            matches = re.search(pattern, response)
            if matches:
                json_content = matches.group(1).strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    continue
        
        # Trường hợp 3: Tìm chuỗi JSON trong văn bản
        # Tìm dấu ngoặc nhọn/vuông đầu tiên và cuối cùng
        first_brace = min(response.find('{'), response.find('[')) if response.find('{') != -1 and response.find('[') != -1 else max(response.find('{'), response.find('['))
        last_brace = max(response.rfind('}'), response.rfind(']'))
        
        if first_brace != -1 and last_brace != -1:
            json_content = response[first_brace:last_brace+1]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # Nếu không thể phân tích được, trả về None
        print("Không thể phân tích JSON từ phản hồi của LLM")
        return None
        
    except Exception as e:
        print(f"Lỗi khi phân tích JSON: {str(e)}")
        return None


def parse_schema_enrichment(response: ChatResponse) -> list:
    """
    Phân tích phản hồi JSON từ LLM theo định dạng được chỉ định trong SCHEMA_ENRICHMENT_PROMPT.
    
    Args:
        response: Chuỗi phản hồi từ LLM
        
    Returns:
        List chứa thông tin bảng và cột đã được làm phong phú, hoặc list rỗng nếu có lỗi
    """
    try:
        # Phân tích JSON từ phản hồi
        response = response.message.content
        enriched_data = parse_llm_json_response(response)
        
        if not enriched_data or not isinstance(enriched_data, list):
            return []
        
        # Kiểm tra cấu trúc của dữ liệu đã làm phong phú
        validated_data = []
        for table_info in enriched_data:
            if not isinstance(table_info, dict):
                continue
                
            table_name = table_info.get('table_name')
            description = table_info.get('description')
            columns = table_info.get('columns', [])
            
            if not table_name:
                continue
                
            validated_columns = []
            for column in columns:
                if not isinstance(column, dict):
                    continue
                    
                column_name = column.get('column_name')
                column_description = column.get('description', '')
                
                if column_name:
                    validated_columns.append({
                        'column_name': column_name,
                        'description': column_description
                    })
            
            validated_data.append({
                'table_name': table_name,
                'description': description or '',
                'columns': validated_columns
            })
        
        return validated_data
    except Exception as e:
        print(f"Lỗi khi phân tích phản hồi làm phong phú schema: {str(e)}")
        return []