import requests
import os
import base64
import io
from enums.response_enum import ResponseEnum
from exceptions.app_exception import AppException
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
import logging
import time
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


API_HOST = f"http://{os.getenv('EMBED_HOST_API')}"
# API_HOST = "http://localhost:8181"
logger.info("EMBED_HOST_API: " + API_HOST)

_endpoints = {
    "connect": f"{API_HOST}/api/v1/db/get-schema",
    "connect_sqlite": f"{API_HOST}/api/v1/db/get-schema/sqlite",
    "query": f"{API_HOST}/api/v1/db/query",
    "query_sqlite": f"{API_HOST}/api/v1/db/query/sqlite",
}


def validate_connection_payload(connection_payload):
    db_type = connection_payload.get("dbType", "").lower()
    
    if db_type not in ["postgresql", "mysql", "sqlite"]:
        return False, "Unsupported database type"
    
    if db_type == "sqlite":
        if "file" not in connection_payload:
            return False, "Missing SQLite file in connection payload"
    else:
        required_fields = ["url", "username", "password"]
        missing_fields = [field for field in required_fields if field not in connection_payload]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, None

def get_schema(connection_payload) -> list:
    
    db_type = connection_payload.get("dbType", "").lower()
    try:
        if db_type == "sqlite":
            api_url = _endpoints["connect_sqlite"]
            
            # Check if file is a base64 string and decode it
            file_data = connection_payload["file"]
            if isinstance(file_data, str):
                try:
                    # Decode base64 to binary
                    binary_data = base64.b64decode(file_data)
                    files = {'file': ('database.sqlite', binary_data, 'application/octet-stream')}
                except Exception as e:
                    raise ValueError(f"Invalid base64 encoded file: {str(e)}")
            else:
                # Already binary data
                files = {'file': ('database.sqlite', file_data, 'application/octet-stream')}
                
            response = requests.post(api_url, files=files)
        else:
            # For PostgreSQL and MySQL
            api_url = _endpoints["connect"]
            response = requests.post(api_url, json=connection_payload)
    except Exception as e:
        raise AppException(ResponseEnum.CANNOT_CONNECT_TO_EMBEB_SERVER);
    
    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            schema = result.get("data", {})
            if not schema:
                raise ValueError("Schema not found in the API response.")
            table_details = schema.get("tables", [])
            return table_details
        else:
            raise AppException(ResponseEnum.FAILED_TO_GET_SCHEMA);
    else:
        raise AppException(ResponseEnum.FAILED_TO_GET_SCHEMA);

def execute_sql(connection_payload, sql_query):
    db_type = connection_payload.get("dbType", "").lower()
    try:
        if db_type == "sqlite":
            api_url = _endpoints["query_sqlite"]
            
            # Check if file is a base64 string and decode it
            file_data = connection_payload["file"]
            if isinstance(file_data, str):
                try:
                    # Decode base64 to binary
                    binary_data = base64.b64decode(file_data)
                    files = {'file': ('database.sqlite', binary_data, 'application/octet-stream')}
                except Exception as e:
                    raise ValueError(f"Invalid base64 encoded file: {str(e)}")
            else:
                # Already binary data
                files = {'file': ('database.sqlite', file_data, 'application/octet-stream')}
                
            # Update to match curl format - query parameter sent as form field
            response = requests.post(api_url, files=files, data={"query": sql_query}, timeout=200)
        else:
            # For PostgreSQL and MySQL
            api_url = _endpoints["query"]
            # Update to match curl format - include all connection details along with query
            payload = {**connection_payload, "query": sql_query}
            response = requests.post(api_url, json=payload, timeout=200)
    except requests.exceptions.Timeout:
        raise AppException(ResponseEnum.TIMEOUT_ERROR)
    except Exception as e:
        raise AppException(ResponseEnum.CANNOT_CONNECT_TO_EMBEB_SERVER)
    
    result = response.json()
    if response.status_code == 200 and result.get("code") == 0:
        return {
            "data": result.get("data", {}),
            "error": None
        }
    elif response.status_code == 200 and (result.get("code") == 2002 or result.get("code") == 2004):
        return {
            "data": None,
            "error": result.get("message", {})
        }
    else:
        raise AppException(ResponseEnum.FAILED_TO_EXECUTE_QUERY)
    
def get_sample_data(connection_payload, table_details, limit=3):
    """
    Lấy dữ liệu mẫu từ cơ sở dữ liệu với giá trị distinct cho mỗi cột.
    Hỗ trợ PostgreSQL, SQLite và MySQL.
    
    Args:
        connection_payload: Thông tin kết nối đến cơ sở dữ liệu
        table_details: Thông tin chi tiết của bảng
        limit: Số lượng giá trị mẫu cần lấy cho mỗi cột (mặc định: 3)
        
    Returns:
        Danh sách các hàng CSV với nội dung phong phú và đa dạng
    """
    try:
        # Lấy thông tin loại DB từ connection_payload
        db_type = connection_payload.get('db_type', '').lower()
        table_name = table_details['tableIdentifier']
        
        # Lấy danh sách tên cột từ thông tin bảng
        column_names = []
        for column in table_details.get('columns', []):
            column_name = column.get('columnIdentifier')
            if column_name:  # Đảm bảo cột có tên
                column_names.append(column_name)
        print("Column names: ", column_names)
        if not column_names:
            return []
            
        # Định nghĩa cách quote identifier và random ordering cho từng loại DB
        quote_char = '"'
        
        if db_type == 'mysql':
            quote_char = '`'
        elif db_type == 'sqlite':
            quote_char = '"'
        else:  # postgresql hoặc mặc định
            quote_char = '"'
        
        # Hàm trợ giúp để quote đúng tên cột và bảng
        def quote_identifier(identifier):
            return f'{quote_char}{identifier}{quote_char}'
        
        # Lấy dữ liệu mẫu distinct cho mỗi cột
        column_samples = {}
        
        for column_name in column_names:
            # Xây dựng câu truy vấn phù hợp với từng loại DB
            quoted_col = quote_identifier(column_name)
            quoted_table = quote_identifier(table_name)
            
            query = ""
            try:
                # Các truy vấn khác nhau cho từng loại DB
                if db_type == 'postgresql':
                    # Đối với PostgreSQL khi dùng SELECT DISTINCT, ORDER BY phải có trong danh sách select
                    query = f"""
                        SELECT DISTINCT {quoted_col}
                        FROM {quoted_table} 
                        WHERE {quoted_col} IS NOT NULL
                        ORDER BY {quoted_col}
                        LIMIT {limit}
                    """
                elif db_type == 'mysql':
                    query = f"""
                        SELECT DISTINCT {quoted_col}
                        FROM {quoted_table}
                        WHERE {quoted_col} IS NOT NULL
                        ORDER BY RAND()
                        LIMIT {limit}
                    """
                elif db_type == 'sqlite':
                    query = f"""
                        SELECT DISTINCT {quoted_col}
                        FROM {quoted_table}
                        WHERE {quoted_col} IS NOT NULL
                        ORDER BY RANDOM()
                        LIMIT {limit}
                    """
                else:
                    # Mặc định an toàn nhất
                    query = f"""
                        SELECT DISTINCT {quoted_col}
                        FROM {quoted_table}
                        WHERE {quoted_col} IS NOT NULL
                        LIMIT {limit}
                    """
                # Thực thi truy vấn
                result = execute_sql(connection_payload, query)
                
                # Nếu truy vấn lỗi, thử lại với truy vấn đơn giản hơn
                if result.get("error"):
                    simple_query = f"""
                        SELECT DISTINCT {quoted_col}
                        FROM {quoted_table}
                        WHERE {quoted_col} IS NOT NULL
                        LIMIT {limit}
                    """
                    result = execute_sql(connection_payload, simple_query)
                    
                    # Nếu vẫn lỗi, thử không dùng trích dẫn
                    if result.get("error"):
                        no_quote_query = f"""
                            SELECT DISTINCT {column_name}
                            FROM {table_name}
                            WHERE {column_name} IS NOT NULL
                            LIMIT {limit}
                        """
                        result = execute_sql(connection_payload, no_quote_query)
                # Lấy giá trị mẫu từ kết quả
                sample_values = []
                if not result.get("error") and result.get("data"):
                    for record in result.get("data", []):
                        # Thử lấy giá trị với tên cột chính xác
                        value = record.get(column_name)
                        
                        # Nếu không tìm thấy, thử với tên cột viết thường
                        if value is None and column_name.lower() in record:
                            value = record.get(column_name.lower())
                            
                        if value is not None:
                            sample_values.append(value)
                
                column_samples[column_name] = sample_values
                
            except Exception as column_error:
                print(f"Error processing column {column_name}: {str(column_error)}")
                column_samples[column_name] = []
        
        # Lọc ra các cột có dữ liệu mẫu
        valid_columns = [col for col, samples in column_samples.items() if samples]
        
        if not valid_columns:
            return []
        
        # Tạo số hàng tối đa dựa trên số lượng mẫu có được
        max_samples = max([len(column_samples[col]) for col in valid_columns], default=0)
        if max_samples == 0:
            return []
        
        # Tạo các hàng CSV với các giá trị mẫu đa dạng
        csv_rows = []
        for i in range(max_samples):
            row_values = []
            for column_name in valid_columns:
                samples = column_samples[column_name]
                
                # Lấy giá trị mẫu
                value = samples[i % len(samples)]
                str_value = str(value) if value is not None else "NULL"
                
                # Cắt ngắn nếu quá dài
                if len(str_value) > 1000:
                    str_value = str_value[:1000] + "..."
                
                # Thêm dấu ngoặc kép nếu giá trị chứa dấu phẩy
                if "," in str_value:
                    str_value = f'"{str_value}"'
                    
                row_values.append(str_value)
            
            csv_rows.append(", ".join(row_values))
        
        return csv_rows
    except Exception as e:
        # Ghi lại lỗi và trả về danh sách rỗng
        print(f"Error getting sample data: {str(e)}")
        return [str(e)]
    
def llm_chat(llm: Ollama | GoogleGenAI, fmt_messages: PromptTemplate):
   
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            chat_response = llm.chat(fmt_messages)
            return chat_response
        except Exception as e:
            error_str = str(e)
            print(f"\033[91mError in llm_chat: {error_str}\033[0m")
            
            # Check for Google API rate limit with retry delay
            retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", error_str)
            if retry_delay_match:
                retry_seconds = int(retry_delay_match.group(1))
                retry_count += 1
                
                if retry_count < max_retries:
                    print(f"\033[93mRate limit exceeded. Waiting for {retry_seconds} seconds before retry {retry_count}/{max_retries}...\033[0m")
                    time.sleep(retry_seconds)
                    continue
            
            # If we get here, either it's not a rate limit error or we've exhausted retries
            raise e
