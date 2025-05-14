package com.slm.slmembed.enums;

import lombok.Getter;

/**
 * Centralized enum for all response codes and messages
 */
@Getter
public enum ResponseEnum {
    // Success responses
    SUCCESS(0, "Success"),
    SCHEMA_RETRIEVED_SUCCESSFULLY(0, "Schema retrieved successfully"),
    SQLITE_SCHEMA_RETRIEVED_SUCCESSFULLY(0, "SQLite schema retrieved successfully"),
    QUERY_EXECUTED_SUCCESSFULLY(0, "Query executed successfully"),
    SQLITE_QUERY_EXECUTED_SUCCESSFULLY(0, "SQLite query executed successfully"),
    CONNECTION_SUCCESSFUL(0, "Connection successful"),
    SQLITE_CONNECTION_SUCCESSFUL(0, "SQLite connection successful"),

    // HTTP Status Codes
    BAD_REQUEST(400, "Bad Request"),
    UNAUTHORIZED(401, "Unauthorized"),
    FORBIDDEN(403, "Forbidden"),
    NOT_FOUND(404, "Not Found"),
    METHOD_NOT_ALLOWED(405, "Method Not Allowed"),
    CONFLICT(409, "Conflict"),
    PAYLOAD_TOO_LARGE(413, "Payload Too Large"),
    UNSUPPORTED_MEDIA_TYPE(415, "Unsupported Media Type"),
    TOO_MANY_REQUESTS(429, "Too Many Requests"),
    INTERNAL_SERVER_ERROR(500, "Internal Server Error"),
    SERVICE_UNAVAILABLE(503, "Service Unavailable"),

    // Business logic errors (all return HTTP 200 but with error codes)
    VALIDATION_FAILED(1000, "Validation failed"),
    INVALID_REQUEST_FORMAT(1001, "Invalid request format: Please check your JSON syntax"),
    MISSING_REQUIRED_PARAMETER(1002, "Missing required parameter"),
    INVALID_PARAMETER_VALUE(1003, "Invalid parameter value"),

    // Database errors (all return HTTP 200 but with error codes in 2000 range)
    DATABASE_CONNECTION_ERROR(2000, "Database connection error"),
    UNSUPPORTED_DATABASE_TYPE(2001, "Unsupported database type"),
    INVALID_QUERY(2002, "Invalid or unsafe SQL query"),
    DATABASE_ACCESS_ERROR(2003, "Database access error"),
    SQL_ERROR(2004, "SQL error"),

    // File errors (all return HTTP 200 but with error codes in 3000 range)
    FILE_PROCESSING_ERROR(3000, "File processing error"),
    FILE_SIZE_EXCEEDS_LIMIT(3001, "File size exceeds the maximum allowed limit"),
    FILE_UPLOAD_ERROR(3002, "File upload error"),
    FILE_IS_EMPTY(3003, "File is empty"),

    // Security/business errors (all return HTTP 200 but with error codes in 4000 range)
    RATE_LIMIT_EXCEEDED(4000, "Rate limit exceeded. Please try again later"),

    // Resource errors (all return HTTP 200 but with error codes in 5000 range)
    RESOURCE_NOT_FOUND(5000, "Resource not found");

    private final int code;
    private final String message;

    ResponseEnum(int code, String message) {
        this.code = code;
        this.message = message;
    }
}