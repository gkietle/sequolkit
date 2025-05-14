from enum import Enum

class ResponseEnum(Enum):
    UNAUTHORIZED = (401, "Unauthorized")
    FORBIDDEN = (403, "Forbidden")
    NOT_FOUND = (404, "Not Found")
    INTERNAL_SERVER_ERROR = (500, "Internal Server Error")
    INVALID_USERNAME_OR_PASSWORD = (401, "Invalid username or password")
    USERNAME_ALREADY_EXISTS = (400, "Username already exists")
    TOKEN_EXPIRED = (401, "Token expired")
    TOKEN_INVALID = (401, "Token invalid")
    USER_NOT_FOUND = (404, "User not found")

    SUCCESS = (0, "Success")
    CANNOT_CONNECT_TO_EMBEB_SERVER = (1, "Cannot connect to the Embeb server")
    FAILED_TO_GET_SCHEMA = (2, "Failed to get schema")
    NOT_RELEVANT_QUERY = (3, "Not relevant to the schema")
    FAILED_TO_REFLECT_QUERY = (4, "Failed to reflect on generated SQL Query")
    TIMEOUT_ERROR = (5, "Query execution timed out after 200 seconds")
    FAILED_TO_EXECUTE_QUERY = (6, "Failed to execute query")
    
    def __init__(self, code, message):
        self._code = code
        self._message = message

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message
